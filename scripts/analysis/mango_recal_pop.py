"""Mango correction fitted on the population-Brier target itself (cross-fitted by fold).

``mango_recal.py`` fits the avocado PIT map on each fruit's own interval-censored outcome; the mango score instead
compares fruit ``i``'s forecast ``F_i(d_j - d_i)`` with the status of every fruit ``j`` cut on a later day. Here the
map ``g`` is fitted on exactly those pairs: for each test fold, every pair ``(i, j)`` of cut-day fruit from the other
five folds with ``d_j >= d_i``, ``x = F_i(min(d_j - d_i, H))`` and ``y = s_j``. Maps:

* ``iso``: isotonic (non-decreasing) regression of ``y`` on ``x`` (pool-adjacent-violators);
* ``platt``: ``y ~ sigmoid(a + b logit x)``;
* ``platt_h``: ``y ~ sigmoid(a + b logit x + c h / H)``, then a running max over ``h`` to keep each CDF monotone;
* ``auto`` (guarded): inside the five training folds, leave one fold out at a time and score raw / ``iso`` /
  ``platt_h`` on that fold's own pairs (as the test score does); use whichever wins, so a correction is applied only
  when it helps on held-out fruit.

A single non-decreasing ``g`` applied pointwise keeps every forecast a valid CDF. ``g`` is applied to every forecast
of the test fold (cut-day and earlier scans) and rescored with ``mango_cv.write_result``.

    python scripts/analysis/mango_recal_pop.py outputs/mango_views            # iso -> outputs/mango_views_poprecal
    python scripts/analysis/mango_recal_pop.py outputs/mango_views --compare  # all maps on the headline runs
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "run"))
from mango_cv import HORIZON, folds, write_result  # noqa: E402

from ripen.data.mango import load_mango  # noqa: E402
from ripen.eval.mango import fruit_records  # noqa: E402

EPS = 1e-4
MAPS = ("iso", "platt", "platt_h", "auto")
CANDIDATES = ("raw", "iso", "platt_h")


def pav(x: np.ndarray, y: np.ndarray):
    """Isotonic fit -> (knots, values) for ``np.interp`` (ties in ``x`` pooled first)."""
    ux, inv = np.unique(x, return_inverse=True)
    w = np.bincount(inv).astype(float)
    v = np.bincount(inv, weights=y) / w
    vals, wts, starts = [], [], []
    for i in range(len(ux)):
        vals.append(v[i])
        wts.append(w[i])
        starts.append(i)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            tw = wts[-2] + wts[-1]
            vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / tw
            wts[-2] = tw
            vals.pop(), wts.pop(), starts.pop()
    out = np.empty(len(ux))
    for b, s in enumerate(starts):
        e = starts[b + 1] if b + 1 < len(starts) else len(ux)
        out[s:e] = vals[b]
    return ux, out


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _fit_logistic(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    def nll(th):
        z = X @ th
        return float(np.mean(np.logaddexp(0, z) - y * z))

    def grad(th):
        p = 1 / (1 + np.exp(-(X @ th)))
        return X.T @ (p - y) / len(y)

    th0 = np.zeros(X.shape[1])
    th0[1] = 1.0
    return minimize(nll, th0, jac=grad, method="L-BFGS-B").x


def pairs(F: np.ndarray, recs: list[dict]):
    """``F [n, H+1]`` cut-day forecasts with ``recs`` in the same order -> (x, y, h) over every valid pair."""
    d = np.array([r["cut_day"] for r in recs])
    s = np.array([float(r["spoiled"]) for r in recs])
    i, j = np.nonzero((d[None, :] >= d[:, None]) & ~np.eye(len(d), dtype=bool))
    h = np.minimum(d[j] - d[i], HORIZON).astype(int)
    return F[i, h], s[j], h


def fit_map(kind: str, x, y, h):
    """-> function ``F [n, H+1] -> F'``."""
    if kind == "raw":
        return lambda F: F
    if kind == "iso":
        kx, ky = pav(x, y)
        return lambda F: np.interp(F, kx, ky)
    if kind == "platt":
        th = _fit_logistic(np.stack([np.ones_like(x), _logit(x)], 1), y)
        return lambda F: 1 / (1 + np.exp(-(th[0] + th[1] * _logit(F))))
    if kind == "platt_h":
        th = _fit_logistic(np.stack([np.ones_like(x), _logit(x), h / HORIZON], 1), y)

        def g(F):
            hh = np.arange(F.shape[1])[None, :] / HORIZON
            return np.maximum.accumulate(1 / (1 + np.exp(-(th[0] + th[1] * _logit(F) + th[2] * hh))), axis=1)

        return g
    raise ValueError(kind)


def _pop_brier(F, recs) -> tuple[float, int]:
    x, y, _ = pairs(F, recs)
    return float(np.sum((x - y) ** 2)), len(x)


def choose(F: np.ndarray, rec_list: list[dict], fk: np.ndarray) -> str:
    """Inner leave-one-fold-out over the training folds -> the candidate with the lowest pooled pop Brier."""
    best, best_sc = "raw", np.inf
    for kind in CANDIDATES:
        tot, n = 0.0, 0
        for k in np.unique(fk):
            tr, te = fk != k, fk == k
            x, y, h = pairs(F[tr], [r for r, m in zip(rec_list, tr, strict=True) if m])
            sq, c = _pop_brier(fit_map(kind, x, y, h)(F[te]), [r for r, m in zip(rec_list, te, strict=True) if m])
            tot, n = tot + sq, n + c
        if tot / max(n, 1) < best_sc - 1e-12:
            best, best_sc = kind, tot / max(n, 1)
    return best


def correct(z: dict, kind: str, items, fold_list, chosen: list | None = None) -> np.ndarray:
    recs = {r["item_id"]: r for r in fruit_records(items)}
    by_id = {it.item_id: it for it in items}
    fold_of = {f: k for k, sp in enumerate(fold_list) for f in sp["test"]}
    fruit = np.array([by_id[str(i)].meta["fruit_id"] for i in z["item_ids"]])
    fk = np.array([fold_of[f] for f in fruit])
    cut = z["cut"].astype(bool)
    F = z["F"].copy()
    for k in np.unique(fk):
        tr = cut & (fk != k)
        rec_tr = [recs[f] for f in fruit[tr]]
        use = choose(z["F"][tr], rec_tr, fk[tr]) if kind == "auto" else kind
        if chosen is not None:
            chosen.append(use)
        x, y, h = pairs(z["F"][tr], rec_tr)
        F[fk == k] = fit_map(use, x, y, h)(z["F"][fk == k])
    return F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="outputs/mango_views")
    ap.add_argument("--map", default="iso", choices=MAPS)
    ap.add_argument("--compare", action="store_true", help="every map on the headline runs -> <out>/maps.md")
    a = ap.parse_args()
    root = Path(a.root)
    out_root = root.parent / (root.name + "_poprecal")
    items, fold_list = load_mango("mw"), folds()
    if a.compare:
        P = "sep1+sep2+sep3+sep4+sep5"
        runs = ["age_clock", "latent_sde_rgb", f"latent_sde_both_v{P}", "latent_ode_time_hist",
                f"latent_ode_both_v{P}_hist", "latent_ode_rgb_hist", f"latent_ode_samd_v{P}_hist"]  # fmt: skip
        tmp = Path(tempfile.mkdtemp())  # rescored copies are throwaway
        L = ["| run | raw | " + " | ".join(MAPS) + " |", "|---|" + "---|" * (len(MAPS) + 1)]
        res: dict = {}
        for run in runs:
            ps = sorted(root.glob(f"seed*/{run}/predictions.npz"))
            if not ps:
                continue
            raw = [json.loads((p.parent / "metrics.json").read_text())["cutday"]["pop_brier"] for p in ps]
            cells = [f"{np.mean(raw):.3f}"]
            res[run] = {"raw": float(np.mean(raw))}
            for kind in MAPS:
                v = []
                for p in ps:
                    z = dict(np.load(p, allow_pickle=True))
                    o = tmp / run / kind / p.parent.parent.name
                    write_result(o, correct(z, kind, items, fold_list), z["lo"], z["hi"], z["cut"], z["item_ids"],
                                 items, fold_list)  # fmt: skip
                    v.append(json.loads((o / "metrics.json").read_text())["cutday"]["pop_brier"])
                cells.append(f"{np.mean(v):.3f}")
                res[run][kind] = float(np.mean(v))
            L.append(f"| {run} | " + " | ".join(cells) + " |")
        out_root.mkdir(parents=True, exist_ok=True)
        head = "Population Brier (mean over seeds), raw vs each map\n\n"
        (out_root / "maps.md").write_text(head + "\n".join(L) + "\n")
        (out_root / "maps.json").write_text(json.dumps(res, indent=1))
        print("\n".join(L))
        return
    for p in sorted(root.glob("seed*/*/predictions.npz")):
        z = dict(np.load(p, allow_pickle=True))
        out = out_root / p.parent.parent.name / p.parent.name
        write_result(out, correct(z, a.map, items, fold_list), z["lo"], z["hi"], z["cut"], z["item_ids"], items,
                     fold_list)  # fmt: skip


if __name__ == "__main__":
    main()
