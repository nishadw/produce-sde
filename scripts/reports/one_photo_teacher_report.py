"""One-photo study report, phase 2: the microwave teacher (report env). Reads outputs/one_photo_teacher/summary.json
(scripts/analysis/one_photo_teacher_compare.py) and the example figure (one_photo_teacher_figure.py
--run transformer_fruit --seed 1). Writes 2026-11-23_one-photo-phase2-teacher.docx to ../results."""

import json

from _docstyle import GREY, bullet, doc, para, sub, table
from _paths import FIGS, REPO, RESULTS
from docx.shared import Inches

S = json.loads((REPO / "outputs" / "one_photo_teacher" / "summary.json").read_text())
FILLS = {fill: json.loads((REPO / "outputs" / f"one_photo_teacher_fill_{fill}" / "summary.json").read_text())
         for fill in ("linear", "logistic")}  # fmt: skip
OUT = RESULTS / "2026-11-23_one-photo-phase2-teacher.docx"
FIG = FIGS / "one_photo_teacher_examples_transformer_fruit.png"
MODELS = ("Transformer", "Latent SDE", "Latent ODE")
FEATS = ("Fruit only",)
METHODS = {  # summary label -> report label
    "Public only (phase 1)": "Public only (phase 1)",
    "Fine-tuned on lab photos": "Fine-tuned on lab photos",
    "Taught by scan teacher (CAML)": "+ scan teacher, CAML",
    "Taught by scan teacher (SAMD)": "+ scan teacher, SAMD",
}
FT = "Fine-tuned on lab photos"


def cell(g):
    return f"{g['mae']:.1f} days"


def rows(model):
    f = FEATS[0]
    out = []
    for key, label in METHODS.items():
        fills = ["—" if key.startswith("Public") else cell(FILLS[x][model][f][key]) for x in ("linear", "logistic")]
        out.append([label, cell(S[model][f][key]), *fills])
    return out


# =====================================================================================================================
doc.add_paragraph("One Photo In, Days Left Out: Phase 2", style="Title")
para("CMSC 498Z final project · Nishad Wajge · 23 November 2026", color=GREY, size=9.5, after=6)

doc.add_heading("1. Question", level=1)
para("Can the lab's microwave scans, used only during training as a teacher, make the one-photo model better? "
     "The app itself never needs a scanner.")  # fmt: skip

doc.add_heading("2. Methods", level=1)
for lead, text in (
    ("Start: ", "each phase-1 model (trained on the public avocado dataset)."),
    ("Fine-tune: ", "on the lab avocados, one held out at a time (39 photos)."),
    ("Teacher: ", "the same model, also given the scan. Only 20 lab photos have one (mostly avocados 1, 2, 3, 11)."),
    ("Teaching: ", "CAML: copy the teacher's forecast. SAMD: copy the teacher's internal features."),
    ("Models: ", "transformer, latent SDE, latent ODE."),
):  # fmt: skip
    bullet(text, lead=lead)

sub("Figure 1. Lab avocado 2 (held out), transformer, one of its three runs")
doc.add_picture(str(FIG), width=Inches(5.0))
para("Bars: predicted odds of spoiling each day; green: when it actually spoiled.", size=8.5, after=4)

doc.add_heading("3. Results", level=1)
para("Day error: days between predicted and actual spoil day on the 12 held-out lab avocados (mean of 3 runs). "
     "Fill: the lab photographed most avocados only on days 1-5, so the training avocados get extra made-up "
     "photos and scans up to day 15, extended by a straight line (linear) or an S-curve (logistic). "
     "Testing is always on real photos.", size=9.5, after=3)  # fmt: skip
HDR = ["Method", "Real photos only", "+ linear fill", "+ logistic fill"]
WIDTHS = [2.2, 1.4, 1.4, 1.4]
for model in MODELS:
    sub(model)
    table(HDR, rows(model), WIDTHS, size=9, highlight=1, gap=False)
para("Green: plain lab fine-tune, the reference the teacher has to beat.", size=8.5, after=4)

doc.add_heading("4. Takeaways", level=1)
T, E, O = "Transformer", "Latent SDE", "Latent ODE"


def d(m, k):
    return f"{S[m]['Fruit only'][k]['mae']:.1f}"


def fill_d(fill, m, k):
    return f"{FILLS[fill][m]['Fruit only'][k]['mae']:.1f}"


for lead, text in (
    ("Lab fine-tune works well: ", f"every model improves (transformer {d(T, 'Public only (phase 1)')} to "
     f"{d(T, FT)} days, latent SDE {d(E, 'Public only (phase 1)')} to {d(E, FT)}, latent ODE "
     f"{d(O, 'Public only (phase 1)')} to {d(O, FT)})."),
    ("The teacher rarely helps: ", "CAML trims under 0.1 days; SAMD is level or worse."),
    ("Filling in days does not help: ", "the plain fine-tune gets slightly worse with either fill (latent SDE "
     f"{d(E, FT)} to {fill_d('linear', E, FT)} / {fill_d('logistic', E, FT)} days). The teacher looks better "
     "next to it only because the fine-tune got worse; the best result per "
     "model is no better than with real photos only."),
):  # fmt: skip
    bullet(text, lead=lead)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print("wrote", OUT)
