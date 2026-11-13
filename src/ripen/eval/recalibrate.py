"""Post-hoc recalibration of predicted CDFs on a held-out split (no retraining).

A model is calibrated when the probability integral transform ``U = F(T)`` is uniform. On the validation split we
estimate the distribution ``G`` of ``U`` and replace every forecast by ``G ∘ F`` (Kuleshov, Fenner & Ermon, ICML
2018), which is calibrated in the marginal sense when the validation and test instances are exchangeable.

Censoring is handled the way survival analysis handles it, by Kaplan–Meier on the PIT scale: a right-censored
instance only says ``U > F(lo)``, so it enters as a censoring "time" ``F(lo)``. An interval-censored event
``T ∈ (lo, hi]`` (one day wide here) enters at the midpoint ``(F(lo) + F(hi)) / 2``.

``G`` is non-decreasing with ``G(0) = 0`` and ``G(1) = 1``, so ``G ∘ F`` is still a CDF. It changes widths,
coverage and calibration but cannot change the ranking of instances at a fixed horizon. Under a covariate shift (a
storage temperature unseen in training) the validation PIT is not representative, so it does not fix transfer.
"""

from __future__ import annotations

import numpy as np

from .metrics import cdf_at


def pit_values(F: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(u, event)``: PIT of each instance and whether it is observed (``False`` = right-censored at ``u``)."""
    F, lo, hi = np.asarray(F, float), np.asarray(lo, float), np.asarray(hi, float)
    hi = np.where(hi >= 1e5, np.inf, hi)
    unc = ~np.isinf(hi)
    f_lo = cdf_at(F, lo)
    f_hi = cdf_at(F, np.where(unc, hi, lo))
    u = np.where(unc, 0.5 * (f_lo + f_hi), f_lo)
    return np.clip(u, 0.0, 1.0), unc


def kaplan_meier_cdf(u: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Knots ``(x, G(x))`` of the Kaplan–Meier CDF of ``u`` on [0, 1], with ``(0, 0)`` and ``(1, 1)`` added.

    Mass that KM cannot place (the tail after the last censored value) goes to 1. Ties: events before censorings.
    """
    u, event = np.asarray(u, float), np.asarray(event, bool)
    xs, surv = [0.0], [1.0]
    s = 1.0
    at_risk = len(u)
    for x in np.unique(u):
        here = u == x
        d = int(np.sum(here & event))
        if d and at_risk:
            s *= 1.0 - d / at_risk
            xs.append(float(x))
            surv.append(s)
        at_risk -= int(here.sum())
    G = 1.0 - np.asarray(surv)
    x = np.asarray(xs)
    if x[-1] < 1.0:
        x, G = np.append(x, 1.0), np.append(G, 1.0)
    else:
        G[-1] = 1.0
    return x, np.maximum.accumulate(G)


class PITRecalibrator:
    """Fit on validation ``(F, lo, hi)``; ``__call__`` maps test CDFs ``F -> G(F)``."""

    def __init__(self, x: np.ndarray, G: np.ndarray, weight: float = 1.0):
        self.x, self.G, self.weight = x, G, weight

    @classmethod
    def fit(cls, F: np.ndarray, lo: np.ndarray, hi: np.ndarray, weight: float = 1.0) -> PITRecalibrator:
        """``weight < 1`` returns ``weight G(F) + (1 - weight) F``: with ~100 forecasts the Kaplan-Meier map is a step
        function whose flat parts give some outcomes almost no probability; mixing in the raw forecast bounds that."""
        u, ev = pit_values(F, lo, hi)
        if not ev.any():
            return cls(np.array([0.0, 1.0]), np.array([0.0, 1.0]))  # nothing to learn from: identity
        return cls(*kaplan_meier_cdf(u, ev), weight=weight)

    def __call__(self, F: np.ndarray) -> np.ndarray:
        F = np.clip(np.asarray(F, float), 0.0, 1.0)
        return self.weight * np.interp(F, self.x, self.G) + (1.0 - self.weight) * F
