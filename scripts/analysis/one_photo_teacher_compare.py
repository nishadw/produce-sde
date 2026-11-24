"""Summary of phase 2 (scripts/run/one_photo_teacher.py): does a microwave teacher help the one-photo student?

Per model x photo variant x method, the 12 leave-one-fruit-out test folds (avo01-avo12) are pooled into one set of lab
photos and scored like phase 1 (censored log score, day error, 80 % range coverage; every fruit weighs the same),
then averaged over seeds. "Public only" is the phase-1 model with no lab fine-tune, scored on the same 12 fruit.
"Scanned 4" restricts to avo01/02/03/11, the fruit scanned at every visit.

python scripts/analysis/one_photo_teacher_compare.py  -> outputs/one_photo_teacher/{summary.md,summary.json}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ripen.eval.metrics import censored_log_score, compute_metrics

ROOT = Path("outputs/one_photo_teacher")
P1 = Path("outputs/one_photo")
MODELS = {"transformer": "Transformer", "latent_sde": "Latent SDE", "latent_ode": "Latent ODE"}
FEATS = {"fruit": "Fruit only", "colour": "Skin colour"}
METHODS = {"public": "Public only (phase 1)", "finetune": "Fine-tuned on lab photos",
           "kd": "Taught by scan teacher (CAML)", "samd": "Taught by scan teacher (SAMD)",
           "kd_photo": "Control: taught by photo-only teacher",
           "teacher": "Teacher itself (needs scanner)"}  # fmt: skip
TEST = {f"avo{i:02d}" for i in range(1, 13)}
SCANNED = {"avo01", "avo02", "avo03", "avo11"}


def load(npzs: list[Path], keep: set[str]):
    F, lo, hi, ids = [], [], [], []
    for p in npzs:
        z = np.load(p)
        fr = np.array([str(i).partition("@")[0] for i in z["item_ids"]])
        m = np.isin(fr, list(keep))
        F.append(z["F"][m]), lo.append(z["lo"][m]), hi.append(z["hi"][m].astype(float)), ids.append(fr[m])
    return np.concatenate(F), np.concatenate(lo), np.concatenate(hi), np.concatenate(ids)


def scores(F, lo, hi, fruit) -> dict:
    hi = np.where(hi >= 1e5, np.inf, hi)
    _, inv, cnt = np.unique(fruit, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    m = compute_metrics(F, lo, hi, weights=w).to_dict()
    ls = censored_log_score(F, lo, hi)
    per_fruit = {f: float(np.mean(ls[fruit == f])) for f in np.unique(fruit)}
    return {"score": float(np.sum(ls * w) / w.sum()), "mae": m["mae"], "cov80": m["coverage80"],
            "n_photos": int(len(F)), "n_fruit": int(len(cnt)), "per_fruit": per_fruit}  # fmt: skip


def run_npzs(model: str, feat: str, seed: str, method: str) -> list[Path]:
    if method == "public":
        p = P1 / f"{model}_{feat}" / seed / "predictions_lab.npz"
        return [p] if p.exists() else []
    ps = sorted((ROOT / f"{model}_{feat}" / seed).glob(f"fold*/{method}/predictions_test.npz"))
    return ps if len(ps) == len(TEST) else []


def main() -> None:
    res = {}
    md = ["# Phase 2: microwave scans as a training-only teacher", "",
          "Lab avocados, one held out at a time (12 folds, avo01-12), pooled; every model starts from its phase-1 "
          "public model. Score: censored log score (lower is better), every fruit weighs the same; mean over seeds "
          "(range in brackets). Fruit better: of 12 test fruit, on how many the method's seed-mean score beats the "
          "lab fine-tune's.", ""]  # fmt: skip
    for m, mlab in MODELS.items():
        for f, flab in FEATS.items():
            rows, base = [], None
            for meth, lab in METHODS.items():
                seeds = sorted({p.name for p in (ROOT / f"{m}_{f}").glob("seed*")} | {p.name for p in
                               (P1 / f"{m}_{f}").glob("seed*")})  # fmt: skip
                runs, runs4 = [], []
                for s in seeds:
                    ps = run_npzs(m, f, s, meth)
                    if not ps:
                        continue
                    runs.append(scores(*load(ps, TEST)))
                    runs4.append(scores(*load(ps, SCANNED)))
                if not runs:
                    continue
                agg = {k: float(np.mean([r[k] for r in runs])) for k in ("score", "mae", "cov80")}
                agg["score_range"] = [float(min(r["score"] for r in runs)), float(max(r["score"] for r in runs))]
                agg["mae_range"] = [float(min(r["mae"] for r in runs)), float(max(r["mae"] for r in runs))]
                agg["mae_seeds"] = [float(r["mae"]) for r in runs]
                agg["scanned4_score"] = float(np.mean([r["score"] for r in runs4]))
                pf = runs[0]["per_fruit"]
                agg["per_fruit"] = {fr: float(np.mean([r["per_fruit"][fr] for r in runs])) for fr in pf}
                agg["n_seeds"], agg["n_photos"] = len(runs), runs[0]["n_photos"]
                if meth == "finetune":
                    base = agg["per_fruit"]
                res.setdefault(mlab, {}).setdefault(flab, {})[lab] = agg
                rows.append((lab, agg))
            if not rows:
                continue
            md += [f"## {mlab}, {flab.lower()}", "",
                   "| Method | Lab score | Day error | 80% range | Scanned-4 score | Fruit better vs fine-tune "
                   "| seeds |",
                   "|---|---|---|---|---|---|---|"]  # fmt: skip
            for lab, g in rows:
                fb = "—"
                if base is not None and not lab.startswith("Fine-tuned"):
                    fb = f"{sum(g['per_fruit'][k] < base[k] for k in base)}/{len(base)}"
                    g["fruit_better"] = fb
                r = g["score_range"]
                md.append(f"| {lab} | {g['score']:.2f} [{r[0]:.2f}, {r[1]:.2f}] | {g['mae']:.1f} d | {g['cov80']:.0%} "
                          f"| {g['scanned4_score']:.2f} | {fb} | {g['n_seeds']} |")  # fmt: skip
            md.append("")
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "summary.json").write_text(json.dumps(res, indent=2))
    (ROOT / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
