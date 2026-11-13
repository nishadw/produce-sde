"""AML on the A2SL avocados: outputs/a2sl_aml/seed*/<run>/fold*/ -> outputs/a2sl_aml/summary.md (+ summary_recal.md).

Same pooling, metrics, cross-fitted recalibration and paired bootstrap as ``a2sl_views_summary.py``. The bootstrap
compares each photo-only student with the photo-only model trained alone (negative = the teacher helped).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a2sl_views_summary as vs  # noqa: E402

ROOT = Path("outputs/a2sl_aml")
LABEL = {
    "photo": "photo only, no teacher (baseline)",
    "kd": "CAML distillation (alpha 0.9, t 4)",
    "kd_a05_t1": "CAML distillation (alpha 0.5, t 1)",
    "feat": "feature matching (frozen teacher)",
    "kd_feat": "CAML distillation + feature matching",
    "rkd": "relational KD (RKD, frozen teacher)",
    "pkt": "probabilistic KT (PKT, frozen teacher)",
    "sp": "similarity preserving (SP, frozen teacher)",
    "samd": "SAMD (teacher reset each round)",
    "ours": "lab method: reset + shared head + KD",
    "ours_rkd": "reset + RKD",
    "ours_pkt": "reset + PKT",
    "ours_sp": "reset + SP",
    "photo_valall": "control: photo only, stopped on all-visits validation",
    "rounds_only": "control: reset rounds, no teacher phase",
    "reset_only_photoT": "control: reset schedule, photo-only teacher, no teaching term",
    "reset_only": "control: reset schedule, no teaching term",
    "ours_rkd_photoT": "control: reset + RKD, teacher sees photos only",
    "samd_photoT": "control: SAMD, teacher sees photos only",
    "kd_paired": "CAML distillation, teacher terms on scanned fruit only",
    "samd_paired": "SAMD, teacher terms on scanned fruit only",
    "ours_paired": "lab method, teacher terms on scanned fruit only",
    "ours_rkd_paired": "reset + RKD, teacher terms on scanned fruit only",
    "teacher": "teacher: photo + microwave (needs the scanner)",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--recal", action="store_true")
    a = ap.parse_args()
    vs.ROOT, vs.RECAL = Path(a.root), a.recal
    runs = sorted({p.name for p in vs.ROOT.glob("seed*/*")})
    models = sorted({r.split("_", 2)[0] + "_" + r.split("_", 2)[1] for r in runs})
    L = ["# AML on the A2SL avocados: photo-only students taught by a photo + microwave teacher"
         + (", recalibrated" if a.recal else ""), "",
         f"Root: `{vs.ROOT}`. 4 folds (leave one of avocados 1, 2, 3, 11 out) x seeds; labels are training targets "
         "only. Every row except the teacher sees photos only at test time.", ""]  # fmt: skip
    for scope, tracked in (("tracked avocados only (1, 2, 3, 11)", True), ("all test predictions", False)):
        for m, view in sorted(
            {(mm, r.rsplit("_view", 1)[-1]) for mm in models for r in runs if r.startswith(mm + "_")}
        ):
            mine = [r for r in runs if r.startswith(m + "_") and r.endswith(f"_view{view}")]
            base = f"{m}_photo_view{view}"
            L += [f"## {m}, {scope} (teacher microwave input: {view})", "",
                  "| method | " + " | ".join(k[1] for k in vs.KEYS) + " | log score vs photo only |",
                  "|---|" + "---|" * (len(vs.KEYS) + 1)]  # fmt: skip
            known = [f"{m}_{k}_view{view}" for k in LABEL]
            extra = sorted(r for r in mine if r not in known)  # tagged variants (grids, paired-only)
            for run in [r for r in known if r in mine] + extra:
                meth = run[len(m) + 1 : -len(f"_view{view}")]
                lab = LABEL.get(meth, meth)
                vm = vs.metrics(run, tracked)
                cmp = "—" if meth == "photo" else vs.bootstrap(run, base, tracked)
                L.append(f"| {lab} | " + " | ".join(vs.cell(vm[k]) for k, _ in vs.KEYS) + f" | {cmp} |")
            L.append("")
    text = "\n".join(L) + "\n"
    (vs.ROOT / ("summary_recal.md" if a.recal else "summary.md")).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
