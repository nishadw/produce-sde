"""Paired bootstrap confidence intervals: model A minus model B on the same test predictions.

Per prediction instance, CRPS and |median - true| are averaged over seeds; then fruit (``item_ids``) are resampled
with replacement, because the prefixes of one fruit are correlated. A difference whose 95 % interval excludes 0 is
significant at that level. ``--recal`` applies each run's PIT recalibrator (fitted on its own val predictions).

python scripts/analysis/paired_bootstrap.py --root-a outputs/avocado/round2 --run-a latent_sde_avocado_guided_r18 \
    --root-b outputs/avocado/round1 --run-b gru_gap_avocado --recal
python scripts/analysis/paired_bootstrap.py --pairs outputs/avocado/significance/pairs.txt \
    --out outputs/avocado/significance/summary.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ripen.eval.metrics import quantile_from_cdf
from ripen.eval.recalibrate import PITRecalibrator


def per_instance(run: Path, recal: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """-> (key [N] as 'item|n_obs', item [N], crps [N], abs err [N]) on uncensored test instances."""
    with np.load(run / "predictions_test.npz") as t:
        F, hi, items, n_obs = t["F"], t["hi"], t["item_ids"], t["n_obs"]
    if recal:
        with np.load(run / "predictions_val.npz") as v:
            F = PITRecalibrator.fit(v["F"], v["lo"], v["hi"])(F)
    unc = ~np.isinf(hi)
    h = np.arange(F.shape[1], dtype=float)
    crps = np.sum((F - (h[None, :] >= hi[:, None])) ** 2, axis=1)
    err = np.abs(quantile_from_cdf(F, 0.5) - hi)
    keys = np.array([f"{i}|{k}" for i, k in zip(items, n_obs, strict=True)])
    return keys[unc], items[unc].astype(str), crps[unc], err[unc]


def seed_mean(root: Path, run: str, recal: bool) -> dict[str, tuple[str, float, float]]:
    acc: dict[str, list] = {}
    dirs = sorted(root.glob(f"seed*/{run}"))
    if not dirs:
        raise FileNotFoundError(f"no {root}/seed*/{run}")
    for d in dirs:
        for k, it, c, e in zip(*per_instance(d, recal), strict=True):
            acc.setdefault(k, [it, [], []])
            acc[k][1].append(c)
            acc[k][2].append(e)
    return {k: (v[0], float(np.mean(v[1])), float(np.mean(v[2]))) for k, v in acc.items()}


def compare(root_a, run_a, root_b, run_b, recal: bool, n_boot: int = 5000, seed: int = 0) -> dict:
    a, b = seed_mean(Path(root_a), run_a, recal), seed_mean(Path(root_b), run_b, recal)
    keys = sorted(set(a) & set(b))
    item = np.array([a[k][0] for k in keys])
    d_crps = np.array([a[k][1] - b[k][1] for k in keys])
    d_err = np.array([a[k][2] - b[k][2] for k in keys])
    uniq, inv = np.unique(item, return_inverse=True)
    # per-fruit sums and counts, so one bootstrap draw is a weighted mean over resampled fruit
    s_c, s_e, n = (np.bincount(inv, x, len(uniq)) for x in (d_crps, d_err, np.ones(len(keys))))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    w = np.stack([np.bincount(r, minlength=len(uniq)) for r in draws])  # [n_boot, n_fruit]
    bc, be = (w @ s_c) / (w @ n), (w @ s_e) / (w @ n)
    out = {"n_instances": len(keys), "n_fruit": len(uniq)}
    for name, d, boot in (("crps", d_crps, bc), ("mae", d_err, be)):
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[name] = (
            float(d.mean()),
            float(lo),
            float(hi),
            float(np.mean(boot >= 0) if d.mean() < 0 else np.mean(boot <= 0)),
        )
    return out


def fmt(r: dict, k: str) -> str:
    m, lo, hi, p = r[k]
    star = " *" if lo > 0 or hi < 0 else ""
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-a"), ap.add_argument("--run-a"), ap.add_argument("--root-b"), ap.add_argument("--run-b")
    ap.add_argument("--pairs", help="file of lines: label root_a run_a root_b run_b")
    ap.add_argument("--recal", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.pairs:
        rows = [ln.split() for ln in Path(a.pairs).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    else:
        rows = [[f"{a.run_a} - {a.run_b}", a.root_a, a.run_a, a.root_b, a.run_b]]
    lines = [
        "# Paired bootstrap: A minus B" + (" (both recalibrated)" if a.recal else ""),
        "",
        "Per-prediction CRPS and absolute error averaged over seeds, fruit resampled 5000 times. Negative = A is "
        "better. `*` = the 95 % interval excludes 0.",
        "",
        "| comparison | fruit | predictions | Δ CRPS [95 % CI] | Δ MAE (d) [95 % CI] |",
        "|---|---|---|---|---|",
    ]
    for label, ra, na, rb, nb in rows:
        try:
            r = compare(ra, na, rb, nb, a.recal)
        except FileNotFoundError as e:
            lines.append(f"| {label} | — | — | missing: {e} | |")
            continue
        lines.append(f"| {label} | {r['n_fruit']} | {r['n_instances']} | {fmt(r, 'crps')} | {fmt(r, 'mae')} |")
    text = "\n".join(lines) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
