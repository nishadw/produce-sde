"""Glue used by the scripts: load items for a data config, make or load splits, evaluate and write a run."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
from omegaconf import DictConfig, OmegaConf

from .data import (
    DatasetInfo,
    Item,
    SyntheticConfig,
    dataset_info,
    leave_one_group_out,
    load_splits,
    make_splits,
    save_splits,
    select,
    simulate,
    subsample_visits,
)
from .data.avocado import load_avocado
from .data.features import load_feature_cache
from .data.stevens import load_manifest
from .eval import predict_all, summarize_by_group
from .eval.plots import predicted_vs_true, reliability_diagram
from .models import build_model


def load_items(data_cfg: DictConfig) -> list[Item]:
    """Items for ``data.name in {synthetic, avocado, stevens, a2sl}``, after optional visit subsampling."""
    name = data_cfg.name
    if name == "synthetic":
        keys = {f.name for f in SyntheticConfig.__dataclass_fields__.values()}
        kw = {k: v for k, v in OmegaConf.to_container(data_cfg, resolve=True).items() if k in keys}
        items = simulate(SyntheticConfig(**kw))
    elif name == "avocado":
        feats = load_feature_cache(data_cfg.features_file) if data_cfg.get("features_file") else None
        items = load_avocado(data_cfg.root, data_cfg.get("labels_file"), data_cfg.get("images_dir"), feats)
    elif name == "stevens":
        feats = load_feature_cache(data_cfg.features_file) if data_cfg.get("features_file") else None
        items = load_manifest(data_cfg.manifest, feats)
    elif name == "a2sl":
        from .data.a2sl import load_a2sl_avocado, load_a2sl_pilot, load_scan_cache

        feats = load_feature_cache(data_cfg.features_file) if data_cfg.get("features_file") else None
        scans = load_scan_cache(data_cfg.scans_file) if data_cfg.get("scans_file") else None
        items = load_a2sl_avocado(
            data_cfg.get("manifests", "data/manifests"),
            features=feats,
            scans=scans,
            photo_dim=int(data_cfg.get("photo_dim", 512)),
            labels=data_cfg.get("labels", "jmw2026"),
            scan_view=data_cfg.get("scan_view"),
            scan_standardize=bool(data_cfg.get("scan_standardize", False)),
        )
        if data_cfg.get("include_pilot", False):
            items += load_a2sl_pilot(data_cfg.get("manifests", "data/manifests"), features=feats)
        if data_cfg.get("synthetic"):  # half-synthetic: real window, regression-extrapolated after it
            from .data.a2sl_synth import build_half_synthetic

            items = build_half_synthetic(
                items,
                data_cfg.synthetic,
                photo_dim=int(data_cfg.get("photo_dim", 512)),
                has_scans=bool(data_cfg.get("scans_file")),
                k=int(data_cfg.get("synthetic_k", 4)),
                last_day=int(data_cfg.get("synthetic_last_day", 15)),
                manifests=data_cfg.get("manifests", "data/manifests"),
            )
    else:
        raise KeyError(f"unknown dataset {name!r}")
    every = int(data_cfg.get("subsample_every", 1) or 1)
    if every > 1:
        offset = int(data_cfg.get("subsample_offset", 0) or 0)
        items = [subsample_visits(it, every, offset) for it in items]
    return items


def real_test_items(cfg: DictConfig, test_ids: list[str]) -> list[Item] | None:
    """For a half-synthetic run (``data.synthetic``), the test fruit as the real loader gives them (real visits only,
    the 4 tracked fruit through their first spoiled visit), restricted to the fruit the synthetic set keeps;
    else ``None``."""
    if not cfg.data.get("synthetic"):
        return None
    from omegaconf import OmegaConf

    real_cfg = OmegaConf.merge(cfg.data, {"synthetic": None})
    keep = set(test_ids) & {it.item_id for it in load_items(cfg.data)}
    return [it for it in load_items(real_cfg) if it.item_id in keep]


def splits_for(cfg: DictConfig, items: list[Item]) -> dict[str, list[str]]:
    """Load the split file if it exists, else make one and save it (so every model sees the same items)."""
    mode = cfg.splits.mode
    suffix = cfg.splits.held_out if mode == "logo" else f"seed{cfg.splits.seed}"
    path = Path(cfg.splits.file or f"configs/splits/{cfg.data.name}_{mode}_{suffix}.json")
    if path.exists():
        return load_splits(path)
    if mode == "random":
        sp = make_splits(items, seed=int(cfg.splits.seed))
    elif mode == "logo":
        sp = leave_one_group_out(items, cfg.splits.held_out, seed=int(cfg.splits.seed))
    else:
        raise ValueError(mode)
    save_splits(sp, path, extra={"mode": mode, "seed": int(cfg.splits.seed), "data": cfg.data.name})
    return sp


def model_cfg_dict(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg.model, resolve=True)


def evaluate_and_write(model, items, info: DatasetInfo, cfg: DictConfig, out: Path, key, tag: str = "test") -> dict:
    """Predict on ``items``, compute per-group / overall / fruit-weighted metrics, write JSON and figures."""
    preds = predict_all(
        model, items, info, key, n_obs_list=_n_obs_list(cfg.eval.n_obs_list), batch_size=cfg.eval.batch_size
    )
    out.mkdir(parents=True, exist_ok=True)
    if preds.F.shape[0] == 0:
        (out / f"metrics_{tag}.json").write_text(json.dumps({"overall": {"n": 0}}, indent=2))
        return {
            "overall": {
                "n": 0,
                "mae": float("nan"),
                "crps": float("nan"),
                "coverage80": float("nan"),
                "width80": float("nan"),
                "cindex": float("nan"),
                "ece": float("nan"),
                "n_uncensored": 0,
            }
        }
    metrics = summarize_by_group(preds.F, preds.lo, preds.hi, preds.group, info.groups, preds.item_ids)
    (out / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2))
    reliability_diagram(preds.F, preds.lo, preds.hi, out / f"reliability_{tag}.png", title=f"{cfg.model.name} / {tag}")
    predicted_vs_true(preds.F, preds.hi, out / f"pred_vs_true_{tag}.png", title=f"{cfg.model.name} / {tag}")
    np.savez_compressed(
        out / f"predictions_{tag}.npz",
        F=preds.F,
        lo=preds.lo,
        hi=preds.hi,
        group=preds.group,
        item_ids=preds.item_ids,
        n_obs=preds.n_obs,
        t_now=preds.t_now,
        stage_now=preds.stage_now,
    )
    return metrics


def n_obs_ablation(model, items, info, cfg: DictConfig, out: Path, key) -> dict:
    """Metrics as a function of how many photos are visible (direction 4)."""
    res = {}
    for k in cfg.eval.n_obs_ablation:
        key, sub = jax.random.split(key)
        preds = predict_all(model, items, info, sub, n_obs_list=[int(k)], batch_size=cfg.eval.batch_size)
        if preds.F.shape[0]:
            res[str(k)] = summarize_by_group(preds.F, preds.lo, preds.hi, preds.group, info.groups, preds.item_ids)[
                "overall"
            ]
    (out / "metrics_by_n_obs.json").write_text(json.dumps(res, indent=2))
    return res


def _n_obs_list(x):
    return "all_unspoiled" if isinstance(x, str) else [int(v) for v in x]


def build_from_run(run_dir: str | Path):
    """Rebuild a trained model from a run directory (config.yaml, info.json, model.eqx)."""
    from .train import load_model

    run_dir = Path(run_dir)
    cfg = OmegaConf.load(run_dir / "config.yaml")
    info = DatasetInfo.from_dict(json.loads((run_dir / "info.json").read_text()))
    template = build_model(model_cfg_dict(cfg), info, jax.random.PRNGKey(int(cfg.seed)))
    return load_model(template, run_dir / "model.eqx"), cfg, info


__all__ = [
    "build_from_run",
    "dataset_info",
    "evaluate_and_write",
    "load_items",
    "model_cfg_dict",
    "n_obs_ablation",
    "select",
    "real_test_items",
    "splits_for",
]
