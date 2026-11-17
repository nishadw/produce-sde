"""Steps 1-3 of docs/plans/2026-11-17_a2sl-imputation-random-start.md: fill in A2SL photo features, check the fill.

For each tracked fruit T (leave one out), using only the other fruit:
  * PCA (k components) of the photo features of the training fruit's visits;
  * the truncation rule: fit on the window W = days 1..5, on the fruit observed on every day of W;
  * linear fill  z_ic(t) = a_ic + b_c t          (slope shared by all fruit, offset per fruit);
  * S-curve fill z_ic(t) = lo_c + amp_c * sigmoid((t - m_c - s_i) / w_c)   (shape shared, per-fruit time shift s_i);
then extrapolate T from its own days 1..5 to days 7..13 and compare with T's real photos.
Reference fills: carry the day-5 value forward; T's own day 1..5 mean; and (uses data past the window, so not a fair
competitor) the per-day mean of the other tracked fruit. Also the same two fills fit on *all* observed visits.

Writes outputs/a2sl_impute/extrapolation_check.json and prints a table. CPU, seconds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from ripen.data.a2sl import load_a2sl_avocado
from ripen.data.features import load_feature_cache

TRACKED = ["avo01", "avo02", "avo03", "avo11"]
WINDOW = (1, 3, 5)
FUTURE = (7, 9, 11, 13)


def photo_table(feats) -> dict[str, dict[int, np.ndarray]]:
    """{fruit: {study day: 512-d photo feature}} (A2SL t is relative to each fruit's first visit)."""
    import pandas as pd

    from ripen.data.a2sl import photo_key

    v = pd.read_csv("data/manifests/a2sl_avocado_visits.csv")
    out: dict[str, dict[int, np.ndarray]] = {}
    for r in v.itertuples(index=False):
        if isinstance(r.side_photo, str) and photo_key(r.side_photo) in feats:
            out.setdefault(r.item_id, {})[int(r.day)] = feats[photo_key(r.side_photo)]
    return out


def pca(X: np.ndarray, k: int):
    mu = X.mean(0)
    _, s, vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, vt[:k], (s[:k] ** 2) / (s**2).sum()


def fit_linear(Z: dict[str, dict[int, np.ndarray]]) -> np.ndarray:
    """Shared per-component slope from within-fruit (demeaned) least squares."""
    num, den = 0.0, 0.0
    for d in Z.values():
        t = np.array(sorted(d), float)
        if len(t) < 2:
            continue
        z = np.stack([d[int(x)] for x in t])
        tc = t - t.mean()
        num = num + tc @ (z - z.mean(0))
        den += tc @ tc
    return num / den


def linear_predict(b, d_obs: dict[int, np.ndarray], days) -> np.ndarray:
    t = np.array(sorted(d_obs), float)
    z = np.stack([d_obs[int(x)] for x in t])
    a = (z - np.outer(t, b)).mean(0)
    return a + np.outer(np.array(days, float), b)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_scurve(Z: dict[str, dict[int, np.ndarray]], k: int):
    """Shared lo, amp, m (k each) and one width w per component; one time shift per fruit (mean zero)."""
    fruits = sorted(Z)
    rows = [(i, t, z) for i, f in enumerate(fruits) for t, z in sorted(Z[f].items())]
    allz = np.stack([z for _, _, z in rows])

    def unpack(p):
        lo, amp, m, lw = p[:k], p[k : 2 * k], p[2 * k : 3 * k], p[3 * k : 4 * k]
        s = p[4 * k :]
        return lo, amp, m, np.exp(lw), s - s.mean()

    def resid(p):
        lo, amp, m, w, s = unpack(p)
        r = [lo + amp * _sig((t - m - s[i]) / w) - z for i, t, z in rows]
        return np.concatenate([*r, 0.1 * s])  # light ridge on the shifts

    p0 = np.concatenate([allz.min(0), allz.max(0) - allz.min(0), np.full(k, 7.0), np.full(k, np.log(2.0)),
                         np.zeros(len(fruits))])  # fmt: skip
    sol = least_squares(resid, p0, loss="soft_l1", max_nfev=4000)
    lo, amp, m, w, _ = unpack(sol.x)
    return lo, amp, m, w


def scurve_predict(par, d_obs: dict[int, np.ndarray], days) -> np.ndarray:
    lo, amp, m, w = par
    t = np.array(sorted(d_obs), float)
    z = np.stack([d_obs[int(x)] for x in t])
    sol = least_squares(lambda s: (lo + amp * _sig((t[:, None] - m - s[0]) / w) - z).ravel(), [0.0])
    return lo + amp * _sig((np.array(days, float)[:, None] - m - sol.x[0]) / w)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--k", type=int, default=8, help="principal components")
    ap.add_argument("--out", default="outputs/a2sl_impute")
    args = ap.parse_args()
    feats = load_feature_cache("data/features/a2sl_resnet18.npz")
    P = photo_table(feats)
    load_a2sl_avocado(features=feats, labels=None)  # fails loudly on a stale (name-keyed) cache
    res: dict[str, dict] = {}
    for test in TRACKED:
        train = {f: d for f, d in P.items() if f != test}
        X = np.stack([x for d in train.values() for x in d.values()])
        mu, V, evr = pca(X, args.k)
        proj = {f: {t: (x - mu) @ V.T for t, x in d.items()} for f, d in train.items()}
        zt = {t: (x - mu) @ V.T for t, x in P[test].items()}
        obs = {t: zt[t] for t in WINDOW}
        truth = np.stack([zt[t] for t in FUTURE])
        in_window = {f: {t: d[t] for t in WINDOW} for f, d in proj.items() if all(t in d for t in WINDOW)}
        pred = {
            "linear (window)": linear_predict(fit_linear(in_window), obs, FUTURE),
            "S-curve (window)": scurve_predict(fit_scurve(in_window, args.k), obs, FUTURE),
            "linear (all visits)": linear_predict(fit_linear(proj), obs, FUTURE),
            "S-curve (all visits)": scurve_predict(fit_scurve(proj, args.k), obs, FUTURE),
            "carry day 5 forward": np.repeat(zt[5][None], len(FUTURE), 0),
            "own day 1-5 mean": np.repeat(np.mean([zt[t] for t in WINDOW], 0)[None], len(FUTURE), 0),
            "other tracked, per-day mean*": np.stack(
                [np.mean([proj[f][t] for f in TRACKED if f != test], 0) for t in FUTURE]
            ),
        }
        scale = np.sqrt(((truth - truth.mean(0)) ** 2).sum(1).mean()) + 1e-9  # spread of T's own future path
        res[test] = {
            "n_window_fruit": len(in_window),
            "pca_var_explained": float(evr.sum()),
            "rmse": {
                k: [float(np.sqrt(((p[i] - truth[i]) ** 2).sum())) for i in range(len(FUTURE))] for k, p in pred.items()
            },  # fmt: skip
            "future_path_spread": float(scale),
        }
    methods = list(next(iter(res.values()))["rmse"])
    summary = {m: np.mean([res[f]["rmse"][m] for f in TRACKED], 0).tolist() for m in methods}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    out = {"k": args.k, "window": WINDOW, "future": FUTURE, "per_fruit": res, "mean_rmse_by_day": summary}
    (Path(args.out) / "extrapolation_check.json").write_text(json.dumps(out, indent=1))
    print(f"k={args.k}; window fruit {[res[f]['n_window_fruit'] for f in TRACKED]}; "
          f"PCA var {np.mean([res[f]['pca_var_explained'] for f in TRACKED]):.2f}")  # fmt: skip
    print(f"{'fill':32s}" + "".join(f"day {t:>2d}  " for t in FUTURE) + " mean")
    for m in methods:
        r = summary[m]
        print(f"{m:32s}" + "".join(f"{x:7.2f}  " for x in r) + f"{np.mean(r):6.2f}")
    print("* uses the other tracked fruit's real days 7-13 (beyond the window): a reference, not a fair competitor")


if __name__ == "__main__":
    main()
