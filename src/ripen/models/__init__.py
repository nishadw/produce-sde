"""Models: the latent ODE / SDE, the transformer, and two fit-directly baselines (Weibull AFT, temporal
matching). Each is an :class:`eqx.Module` with ``loss(batch, key)`` and
``predict_cdf(batch, key)``."""

from .base import ShelfLifeModel
from .classical import StorageDay, TemporalMatching, WeibullAFT
from .clock import ArrheniusClock, FixedClock, GroupClock, make_clock
from .latent import LatentDynamics
from .registry import MODEL_REGISTRY, build_model
from .sequence import SequenceSurvival

__all__ = [
    "ArrheniusClock",
    "FixedClock",
    "GroupClock",
    "LatentDynamics",
    "MODEL_REGISTRY",
    "StorageDay",
    "SequenceSurvival",
    "TemporalMatching",
    "WeibullAFT",
    "ShelfLifeModel",
    "build_model",
    "make_clock",
]
