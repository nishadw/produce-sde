"""Scoring for the mango study, identical for every model and input setting.

Each fruit is seen spoiled-or-not on one day only (its cut day), so a per-fruit "true days until spoiled" does not
exist. Every model is scored at the moment a photo exists — the cut day, with every visit up to it visible and no
stage labels — on what the data can check:

* ``pop_brier`` (main score, lower is better): the batch was stored together and fruit were cut on random days, so
  the fruit cut on a later day ``d_j`` show what fraction of the batch had spoiled by then. For a fruit ``i`` scored
  on day ``d_i`` and every other fruit ``j`` with ``d_j >= d_i``, the forecast ``F_i(d_j - d_i) = P(spoiled within
  d_j - d_i days)`` is compared with ``j``'s status: ``mean (F_i(d_j - d_i) - s_j)^2``. It checks the *days* part of
  the forecast (how soon), which a spoiled-now label alone cannot.
* ``status_brier``: ``(F_i(0) - s_i)^2``, the fruit's own spoiled-now status.
* ``spearman_firmness``: rank correlation of the expected days left with the fruit's measured firmness on that day
  (firmer = longer to go; higher is better). ``spearman_within_day`` removes each cut day's mean from both first,
  so it only credits telling apart fruit of the *same* age — what a sensor adds beyond the calendar.
* ``prefix``: for the microwave settings, the interval-censored scores at every earlier scan (proper log score,
  plus MAE / CRPS / 80 % coverage on the senescent fruit, whose spoilage interval is finite).

Snapshot items (``ripen.data.mango``) are scored one visit each; ``meta['at_cut']`` marks the cut-day instance.
"""

from __future__ import annotations

import equinox as eqx
import jax
import numpy as np
from scipy.stats import spearmanr

from ..data.collate import DatasetInfo, collate
from ..data.mango import strip_stages
from ..data.schema import SPOILED_STAGE, Item
from .metrics import censored_log_score, compute_metrics


@eqx.filter_jit
def _predict_cdf(model, batch, key):
    return model.predict_cdf(batch, key)


def predict(model, items: list[Item], info: DatasetInfo, key, batch_size: int = 64):
    """CDFs for snapshot items with stage labels removed. -> (F, lo, hi) with ``hi = inf`` if right-censored."""
    plain = strip_stages(items)
    pred = _predict_cdf  # module level: a jit built per call recompiles every fold and leaks compiled code on CPU
    Fs, lo, hi = [], [], []
    for s in range(0, len(plain), batch_size):
        b = collate(plain[s : s + batch_size], info, n_obs="all")
        key, sub = jax.random.split(key)
        Fs.append(np.asarray(pred(model, b, sub)))
        lo.append(np.asarray(b.lo_rel))
        hi.append(np.where(np.asarray(b.right_censored), np.inf, np.asarray(b.hi_rel)))
    return np.concatenate(Fs), np.concatenate(lo), np.concatenate(hi)


def expected_days(F: np.ndarray) -> np.ndarray:
    """``E[min(days left, H+1)] = sum_h (1 - F[h])`` on the integer grid."""
    return np.sum(1.0 - F, axis=1)


def _resid_by_day(x: np.ndarray, day: np.ndarray) -> np.ndarray:
    out = x.astype(float).copy()
    for d in np.unique(day):
        m = day == d
        out[m] -= out[m].mean()
    return out


def cutday_scores(F: np.ndarray, test: list[dict], ref: list[dict]) -> dict:
    """``test``/``ref``: dicts with ``item_id, cut_day, spoiled, firmness_n``. ``ref`` is every labelled fruit."""
    H = F.shape[1] - 1
    sq, own = [], []
    for i, ti in enumerate(test):
        own.append((F[i, 0] - float(ti["spoiled"])) ** 2)
        for rj in ref:
            if rj["item_id"] == ti["item_id"] or rj["cut_day"] < ti["cut_day"]:
                continue
            h = int(min(rj["cut_day"] - ti["cut_day"], H))
            sq.append((F[i, h] - float(rj["spoiled"])) ** 2)
    e = expected_days(F)
    firm = np.log(np.asarray([t["firmness_n"] for t in test]))
    day = np.asarray([t["cut_day"] for t in test])
    rho = spearmanr(e, firm).statistic if np.std(e) > 0 else float("nan")
    re_, rf = _resid_by_day(e, day), _resid_by_day(firm, day)
    rho_w = spearmanr(re_, rf).statistic if np.std(re_) > 1e-9 else float("nan")
    med = np.argmax(F >= 0.5, axis=1).astype(float)
    med[~(F >= 0.5).any(axis=1)] = H + 1
    return {
        "n": len(test),
        "pop_brier": float(np.mean(sq)),
        "status_brier": float(np.mean(own)),
        "spearman_firmness": float(rho),
        "spearman_within_day": float(rho_w),
        "mean_expected_days_left": float(e.mean()),
        "mean_predicted_spoil_day": float(np.mean(day + med)),
    }


def prefix_scores(F: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    nll = censored_log_score(F, lo, hi)
    out = {"n": int(len(lo)), "log_score": float(np.mean(nll))}
    unc = np.isfinite(hi)
    if unc.any():
        m = compute_metrics(F[unc], lo[unc], hi[unc])
        out.update({"n_uncensored": int(unc.sum()), "mae": m.mae, "crps": m.crps, "coverage80": m.coverage80})
    return out


def fruit_records(items: list[Item]) -> list[dict]:
    """One record per fruit from its cut-day instance."""
    return [
        {
            "item_id": it.meta["fruit_id"],
            "cut_day": float(it.meta["cut_day"]),
            "spoiled": bool(it.meta["stage"] >= SPOILED_STAGE),
            "firmness_n": float(it.meta["firmness_n"]),
            "stage": int(it.meta["stage"]),
        }
        for it in items
        if it.meta["at_cut"]
    ]
