"""One-photo study report, phase 2: the microwave teacher (report env). Reads outputs/one_photo_teacher/summary.json
(scripts/analysis/one_photo_teacher_compare.py) and the example figure (one_photo_teacher_figure.py
--run transformer_fruit --seed 1). Writes 2026-11-23_one-photo-phase2-teacher.docx to ../results."""

import json

from _docstyle import GREY, bullet, doc, para, sub, table
from _paths import FIGS, REPO, RESULTS
from docx.shared import Inches

S = json.loads((REPO / "outputs" / "one_photo_teacher" / "summary.json").read_text())
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
    return [[label] + [cell(S[model][f][key]) for f in FEATS] for key, label in METHODS.items()]


# =====================================================================================================================
doc.add_paragraph("One Photo In, Days Left Out: Phase 2", style="Title")
para("CMSC 498Z final project · Nishad Wajge · 23 November 2026", color=GREY, size=9.5, after=6)

doc.add_heading("1. Question", level=1)
para("Can the lab's microwave scans, used only during training as a teacher, make the one-photo model better? "
     "The app itself never needs a scanner.")  # fmt: skip

doc.add_heading("2. What I did", level=1)
for lead, text in (
    ("Start: ", "each phase-1 model (trained on the public avocados)."),
    ("Fine-tune: ", "on the lab avocados, one held out at a time (12 rounds, 39 test photos)."),
    ("Teacher: ", "the same model, also given the scan. Only 20 lab photos have one (mostly avocados 1, 2, 3, 11)."),
    ("Teaching: ", "same start, photos and settings as the fine-tune, plus one extra term on the scanned photos. "
     "CAML: copy the teacher's forecast. SAMD: copy the teacher's internal features."),
    ("Models: ", "transformer, latent SDE, latent ODE, each reading the photo with the background removed; "
     "3 runs each."),
):  # fmt: skip
    bullet(text, lead=lead)

sub("Figure 1. Lab avocado 2 (held out), transformer, one of its three runs")
doc.add_picture(str(FIG), width=Inches(5.0))
para("Bars: predicted odds of spoiling each day; green: when it actually spoiled.", size=8.5, after=4)

doc.add_heading("3. Results", level=1)
para("Day error: days between predicted and actual spoil day on the 12 held-out lab avocados (mean of 3 runs).",
     size=9.5, after=3)
HDR = ["Method", "Day error"]
WIDTHS = [2.6, 1.6]
for model in MODELS:
    sub(model)
    table(HDR, rows(model), WIDTHS, size=9, highlight=1, gap=False)
para("Green: plain lab fine-tune, the reference the teacher has to beat.", size=8.5, after=4)

doc.add_heading("4. Takeaways", level=1)
T, E, O = "Transformer", "Latent SDE", "Latent ODE"


def d(m, k):
    return f"{S[m]['Fruit only'][k]['mae']:.1f}"


for lead, text in (
    ("Lab fine-tune is the big win: ", f"every model gets closer (transformer {d(T, 'Public only (phase 1)')} to "
     f"{d(T, FT)} days, latent SDE {d(E, 'Public only (phase 1)')} to {d(E, FT)}, latent ODE "
     f"{d(O, 'Public only (phase 1)')} to {d(O, FT)})."),
    ("The teacher barely helps: ", "CAML trims under 0.1 days for the transformer and SDE (about the run-to-run "
     "noise) and slightly hurts the ODE; SAMD is level or worse."),
    ("Why: ", "only 20 scanned photos, and the lab gave every avocado the same spoil window (day 7-9), so "
     "recognising the study day is enough. Lab numbers flatter the app."),
):  # fmt: skip
    bullet(text, lead=lead)

doc.add_heading("5. Next", level=1)
for text in (
    "Keep the lab fine-tune; scan more fruit, each with its own spoil day, before more teacher work.",
    "Test on avocados photographed outside the lab and followed until each one spoils.",
):  # fmt: skip
    bullet(text)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print("wrote", OUT)
