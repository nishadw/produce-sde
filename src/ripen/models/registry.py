"""Build any model from a config dict: ``build_model(cfg, info, key)``."""

from __future__ import annotations

import dataclasses

import jax

from ..data.collate import DatasetInfo
from .classical import TemporalMatching, WeibullAFT
from .latent import LatentDynamics
from .sequence import SequenceSurvival


def _seq(encoder):
    def make(cfg, info, key):
        return SequenceSurvival(
            info.feat_dim,
            len(info.groups),
            cfg["horizon"],
            encoder=encoder,
            hidden=cfg.get("hidden", 64),
            visit_dim=cfg.get("visit_dim", 32),
            group_dim=cfg.get("group_dim", 4),
            use_images=cfg.get("use_images", True),
            use_stages=cfg.get("use_stages", False),
            key=key,
            **{k: cfg[k] for k in ("d_model", "heads", "layers") if k in cfg},
        )

    return make


def _latent(kind):
    def make(cfg, info, key):
        kw = {k: v for k, v in cfg.items() if k not in ("name", "horizon")}
        # snapshot data (every item one visit at t = 0) has t_max = 0; the posterior solve needs a non-empty grid
        t_max = max(info.t_max, 1.0)
        return LatentDynamics(info.feat_dim, len(info.groups), cfg["horizon"], t_max, key=key, kind=kind, **kw)

    return make


MODEL_REGISTRY = {
    "transformer": _seq("transformer"),
    "latent_ode": _latent("ode"),
    "latent_sde": _latent("sde"),
    # fit directly (model.fit), not by the gradient loop
    "weibull": lambda cfg, info, key: WeibullAFT(
        info.feat_dim, cfg["horizon"], cfg.get("pca", 4), cfg.get("ridge", 1.0)
    ),
    "matching": lambda cfg, info, key: TemporalMatching(
        info.feat_dim,
        cfg["horizon"],
        cfg.get("pca", 8),
        cfg.get("k_visits", 3),
        cfg.get("tol", 1.01),
        cfg.get("eps", 0.05),
    ),
}


def _resolve_clock(cfg: dict, info: DatasetInfo) -> dict:
    """``clock_kwargs.rate_ref: auto`` -> the rate that crosses ~3.5 stage units in the median event time, so the
    clock starts at the data's time scale (months, cycles, days) instead of the avocado default."""
    ck = dict(cfg.get("clock_kwargs") or {})
    if ck.get("rate_ref") == "auto":
        ck["rate_ref"] = float(3.5 / max(info.median_event_time, 1e-6))
    if ck.get("init_rate") == "auto":
        ck["init_rate"] = float(3.5 / max(info.median_event_time, 1e-6))
    out = dict(cfg)
    if ck:
        out["clock_kwargs"] = ck
    return out


def build_model(cfg: dict, info: DatasetInfo, key=None):
    """``cfg`` must have ``name`` and ``horizon``; other keys are model-specific (see the configs/ directory).
    ``clock_kwargs.rate_ref: auto`` initialises the clock from the data's median event time."""
    cfg = _resolve_clock(cfg, info)
    name = cfg["name"]
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODEL_REGISTRY)}")
    key = key if key is not None else jax.random.PRNGKey(cfg.get("seed", 0))
    # without image features collate() still pads one all-zero feature column, so size the inputs to match
    info = dataclasses.replace(info, feat_dim=max(info.feat_dim, 1))
    return MODEL_REGISTRY[name](dict(cfg), info, key)
