"""Summary of the one-photo study (docs/plans/2026-11-22_one-photo-app.md, scripts/run/one_photo.py).

For each model x photo variant and each test set (held-out public fruit, lab avocados): censored log score, day error
(median forecast vs the actual spoil window, uncensored photos) and 80 % range coverage, every fruit weighing the same
(its photos share one unit of weight); averaged over seeds. Public results are also split by storage group, which no
model is told.

python scripts/analysis/one_photo_compare.py   -> outputs/one_photo/{summary.md,summary.json}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ripen.eval.metrics import censored_log_score, compute_metrics

ROOT = Path("outputs/one_photo")
MODELS = {"matching": "Image matching", "transformer": "Transformer", "latent_sde": "Latent SDE",
          "latent_ode": "Latent ODE"}  # fmt: skip
FEATS = {"full": "Full photo", "fruit": "Fruit only", "colour": "Skin colour"}
GROUPS = ("T10", "T20", "Tamb")


def scores(npz: Path, group: str | None = None) -> dict:
    z = np.load(npz)
    F, lo, hi = z["F"], z["lo"], z["hi"].astype(float)
    hi = np.where(hi >= 1e5, np.inf, hi)
    fruit = np.array([str(i).partition("@")[0] for i in z["item_ids"]])
    if group is not None:
        keep = np.char.startswith(fruit.astype(str), group + "-")
        F, lo, hi, fruit = F[keep], lo[keep], hi[keep], fruit[keep]
    _, inv, cnt = np.unique(fruit, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    m = compute_metrics(F, lo, hi, weights=w).to_dict()
    ls = censored_log_score(F, lo, hi)
    return {"score": float(np.sum(ls * w) / w.sum()), "mae": m["mae"], "cov80": m["coverage80"],
            "n_photos": int(len(F)), "n_fruit": int(len(cnt))}  # fmt: skip


def mean_over(runs: list[dict]) -> dict:
    keys = ("score", "mae", "cov80")
    out = {k: float(np.mean([r[k] for r in runs])) for k in keys}
    out |= {f"{k}_range": [float(np.min([r[k] for r in runs])), float(np.max([r[k] for r in runs]))] for k in keys}
    return out | {"n_seeds": len(runs), "n_photos": runs[0]["n_photos"], "n_fruit": runs[0]["n_fruit"]}


def main() -> None:
    res, md = {}, ["# One photo in, days until spoiled out", "",
                   "Trained on public avocados; fruit-weighted, mean over seeds. Score: censored log score (lower is "
                   "better). Day error: days between the median forecast and the actual spoil window.", ""]  # fmt: skip
    for m, lab in MODELS.items():
        hdr = ("| Photo | Public score | Public day error | Public 80% cov | Lab score | Lab day error | Lab 80% cov "
               "| Public day error T10 / T20 / Tamb | seeds |")
        md += [f"## {lab}", "", hdr, "|---" * 9 + "|"]
        for f, flab in FEATS.items():
            seeds = sorted((ROOT / f"{m}_{f}").glob("seed*/predictions_lab.npz"))
            if not seeds:
                continue
            r = {"public": mean_over([scores(s.parent / "predictions_test.npz") for s in seeds]),
                 "lab": mean_over([scores(s) for s in seeds]),
                 "by_group": {g: mean_over([scores(s.parent / "predictions_test.npz", g) for s in seeds])
                              for g in GROUPS}}  # fmt: skip
            res.setdefault(lab, {})[flab] = r
            p, q, g = r["public"], r["lab"], r["by_group"]
            md.append(f"| {flab} | {p['score']:.2f} | {p['mae']:.1f} d | {p['cov80']:.0%} | {q['score']:.2f} | "
                      f"{q['mae']:.1f} d | {q['cov80']:.0%} | "
                      + " / ".join(f"{g[k]['mae']:.1f}" for k in GROUPS) + f" | {len(seeds)} |")  # fmt: skip
        md.append("")
    text = "\n".join(md) + "\n"
    (ROOT / "summary.md").write_text(text)
    (ROOT / "summary.json").write_text(json.dumps(res, indent=1))
    print(text)


if __name__ == "__main__":
    main()
