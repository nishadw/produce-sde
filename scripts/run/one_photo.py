"""One-photo study (docs/plans/2026-11-22_one-photo-app.md): the app setting, one photo in -> days until spoiled out.

Train on the public avocados (configs/splits/avocado_random_seed0.json, by fruit), every unspoiled photo its own
example (``ripen.data.schema.one_photo``): no date, no storage group or temperature, no ripeness-stage input. Test on
the held-out public fruit and on all 15 lab avocados (stage labels hidden; the lab fruit never train).

python scripts/run/one_photo.py --model latent_sde --features fruit --seed 1
Writes outputs/one_photo/<model>_<features>/seed<s>/{metrics,predictions}_{test,val,lab}.*
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import jax
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from ripen.data.schema import one_photo
from ripen.models import build_model
from ripen.pipeline import dataset_info, evaluate_and_write, load_items, model_cfg_dict, select, splits_for
from ripen.train import TrainConfig, train

ROOT = Path(__file__).resolve().parents[2]
FEATURES = {  # name -> (public cache, lab cache, dim)
    "full": ("data/features/avocado_resnet18.npz", "data/features/a2sl_resnet18.npz", 512),
    "fruit": ("data/features/avocado_fruit_resnet18.npz", "data/features/a2sl_fruit_resnet18.npz", 512),
    "colour": ("data/features/avocado_colour_pub.npz", "data/features/a2sl_colour_pub.npz", 12),
}
MODEL_ARGS = {  # stage labels are never an input (use_stages=false); the SDE still learns from them in training
    "latent_sde": ["model=latent_sde", "model.use_stages=false", "model.posterior_guidance=false"],
    "latent_ode": ["model=latent_ode", "model.use_stages=false"],
    "transformer": ["model=transformer", "model.use_stages=false", "model.group_dim=0"],
    "matching": ["model=matching", "model.k_visits=1"],
}


def photo_only(it, dim: int):
    """A lab item with the photo part of its features only (the loader appends zeros where the scan would go)."""
    vs = [replace(v, features=None if v.features is None else v.features[..., :dim]) for v in it.visits]
    return replace(it, visits=vs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_ARGS), required=True)
    ap.add_argument("--features", choices=list(FEATURES), required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="outputs/one_photo")
    ap.add_argument("extra", nargs="*", help="extra Hydra overrides (smoke tests)")
    a = ap.parse_args()
    pub_f, lab_f, dim = FEATURES[a.features]
    out = ROOT / a.out / f"{a.model}_{a.features}" / f"seed{a.seed}"
    over = [*MODEL_ARGS[a.model], "data=avocado", f"data.features_file={pub_f}", f"seed={a.seed}",
            "splits.file=configs/splits/avocado_random_seed0.json", f"out_dir={out}", *a.extra]  # fmt: skip
    with initialize_config_dir(str(ROOT / "configs"), version_base="1.3"):
        cfg = compose("config", overrides=over)
        lab_cfg = compose("config", overrides=["data=a2sl", f"data.features_file={lab_f}", "data.scans_file=null",
                                               f"data.photo_dim={dim}"]).data  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "config.yaml")
    items = load_items(cfg.data)
    sp = splits_for(cfg, items)
    tr = one_photo(select(items, sp["train"]))
    va = one_photo(select(items, sp["val"]))
    te = one_photo(select(items, sp["test"]), hide_stages=True)
    lab = one_photo([photo_only(it, dim) for it in load_items(lab_cfg)], hide_stages=True)
    info = dataset_info(tr + va + te)
    (out / "info.json").write_text(json.dumps(info.to_dict(), indent=2))
    print(f"one photo, {a.features}: {len(tr)}/{len(va)}/{len(te)} public photos, {len(lab)} lab photos, "
          f"feat_dim {info.feat_dim}", flush=True)  # fmt: skip
    key = jax.random.PRNGKey(a.seed)
    k_model, k_train, k_eval = jax.random.split(key, 3)
    model = build_model(model_cfg_dict(cfg), info, k_model)
    tcfg = TrainConfig(**OmegaConf.to_container(cfg.train, resolve=True))
    if hasattr(model, "fit"):
        model = model.fit(tr, va, info, tcfg)
    else:
        model, _ = train(model, tr, va, info, tcfg, k_train, out_dir=out)
    for tag, its, k in zip(("test", "val", "lab"), (te, va, lab), jax.random.split(k_eval, 3), strict=True):
        o = evaluate_and_write(model, its, info, cfg, out, k, tag=tag)["overall"]
        print(f"{tag}: MAE {o['mae']:.2f}  CRPS {o['crps']:.2f}  cov80 {o['coverage80']:.2f}  width80 "
              f"{o['width80']:.1f}  (n={o['n']})", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
