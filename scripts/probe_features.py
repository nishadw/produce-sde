"""Linear probe of photo-feature caches: how much ripeness signal does each backbone carry, before any training?

python scripts/probe_features.py data/features/avocado_resnet18.npz data/features/avocado_dinov2_vitb14.npz

Per visit, ridge regression from the (view-averaged) feature vector to (a) the ripeness stage and (b) days until the
spoiled stage (uncensored fruit only). Fit on the train split, ridge strength picked on val; the test split is never
read. Prints a table and writes it to results/feature_probe.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ripen.data.avocado import load_avocado
from ripen.data.features import load_feature_cache
from ripen.data.splits import load_splits

LAMBDAS = [1e-2, 1e-1, 1, 10, 100, 1000, 1e4]


def visit_table(items, ids):
    want = set(ids)
    X, stage, left, grp = [], [], [], []
    for it in items:
        if it.item_id not in want:
            continue
        for v in it.visits:
            fv = v.feature_vector
            if fv is None or v.stage is None:
                continue
            X.append(fv)
            stage.append(v.stage)
            left.append(np.nan if it.right_censored else it.spoil_hi - v.t)
            grp.append(it.storage)
    return np.stack(X), np.array(stage, float), np.array(left, float), np.array(grp)


def ridge(Xtr, ytr, Xva, lam):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = (Xtr - mu) / sd, (Xva - mu) / sd
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (ytr - ytr.mean()))
    return B @ w + ytr.mean()


def best(Xtr, ytr, Xva, yva, score):
    return min((score(ridge(Xtr, ytr, Xva, lam), yva), lam) for lam in LAMBDAS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("caches", nargs="+")
    ap.add_argument("--splits", default="results/splits/avocado_random_seed0.json")
    ap.add_argument("--out", default="results/feature_probe.md")
    args = ap.parse_args()
    sp = load_splits(args.splits)
    rows = []
    for cache in args.caches:
        items = load_avocado("data/avocado", features=load_feature_cache(cache))
        Xtr, str_, ltr, _ = visit_table(items, sp["train"])
        Xva, sva, lva, gva = visit_table(items, sp["val"])
        stage_mae, lam_s = best(Xtr, str_, Xva, sva, lambda p, y: float(np.mean(np.abs(p - y))))
        acc = float(np.mean(np.clip(np.round(ridge(Xtr, str_, Xva, lam_s)), 1, None) == sva))
        mtr, mva = ~np.isnan(ltr), ~np.isnan(lva)
        left_mae, _ = best(Xtr[mtr], ltr[mtr], Xva[mva], lva[mva], lambda p, y: float(np.mean(np.abs(p - y))))
        rows.append((Path(cache).stem, Xtr.shape[1], len(Xtr), len(Xva), stage_mae, acc, left_mae))
        print(rows[-1], flush=True)
    lines = [
        "# Linear probe of photo features (avocado, val split)",
        "",
        "Ridge from one visit's features to its stage and to days until spoiled; train split fit, val split scored.",
        "",
        "| cache | dim | train visits | val visits | stage MAE | stage acc. | days-left MAE |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:.3f} | {r[5]:.3f} | {r[6]:.3f} |" for r in rows]
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
