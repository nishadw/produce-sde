"""One-photo study report, phase 2: the microwave teacher (report env). Reads outputs/one_photo_teacher/summary.json
(scripts/analysis/one_photo_teacher_compare.py) and the example figure (one_photo_teacher_figure.py
--run transformer_colour --seed 1). Writes 2026-11-23_one-photo-phase2-teacher.docx to ../results."""

import json

from _docstyle import GREY, bullet, doc, para, sub, table
from _paths import FIGS, REPO, RESULTS
from docx.shared import Inches

S = json.loads((REPO / "outputs" / "one_photo_teacher" / "summary.json").read_text())
OUT = RESULTS / "2026-11-23_one-photo-phase2-teacher.docx"
FIG = FIGS / "one_photo_teacher_examples_transformer_colour.png"
MODELS = ("Transformer", "Latent SDE", "Latent ODE")
FEATS = ("Fruit only", "Skin colour")
METHODS = {  # summary label -> report label
    "Public only (phase 1)": "Public only (phase 1)",
    "Fine-tuned on lab photos": "Fine-tuned on lab photos",
    "Taught by scan teacher (CAML)": "+ scan teacher, CAML",
    "Taught by scan teacher (SAMD)": "+ scan teacher, SAMD",
}
FT, CAML, SAMD = "Fine-tuned on lab photos", "Taught by scan teacher (CAML)", "Taught by scan teacher (SAMD)"


def cell(g):
    lo, hi = g["mae_range"]
    return f"{g['mae']:.1f} d ({lo:.1f}-{hi:.1f})"


def rows(model):
    return [[label] + [cell(S[model][f][key]) for f in FEATS] for key, label in METHODS.items()]


def d(m, f, k):
    return f"{S[m][f][k]['mae']:.2f}"


# =====================================================================================================================
doc.add_paragraph("One Photo In, Days Left Out: Phase 2", style="Title")
para("CMSC 498Z final project · Nishad Wajge · 23 November 2026", color=GREY, size=9.5, after=6)

doc.add_heading("1. Goal", level=1)
para("Same app as phase 1: one photo of an avocado in, days until it spoils out. Phase 2 asks whether the lab's "
     "microwave scans, which see inside the fruit, can make the photo model better when used only during training "
     "(a teacher). The app never needs a scanner.")  # fmt: skip

doc.add_heading("2. What I did", level=1)
for lead, text in (
    ("Started from the phase-1 models. ", "Each model trained on the public avocados is the starting point."),
    ("Fine-tuned on the lab avocados, one held out at a time. ", "12 rounds, each testing one of lab avocados 1-12 "
     "on photos it never saw (39 photos in all). Avocados 13-15 were already spoiled in their first photo and always "
     "train. Training stops by two other unscanned lab fruit, so every scanned fruit is used for training."),
    ("The teacher. ", "The same model with the scan added as an extra input (starting at zero, so it begins as the "
     "photo model), fine-tuned on the lab photos plus scans. Only 20 lab photos have a scan: avocados 1, 2, 3 and 11 "
     "at every visit, avocados 4-7 once."),
    ("Two ways to teach, on top of the same fine-tune. ", "CAML and SAMD start from the same public model and "
     "train on the same lab photos with the same settings as the plain fine-tune; the only difference is an extra "
     "teaching term on the scanned photos. CAML: the student copies the teacher's forecast. SAMD: teacher and "
     "student take turns, the teacher restarted from the student each round, and the student copies its internal "
     "features."),
    ("Three models, two photo variants, 3 seeds each. ", "Transformer, latent SDE, latent ODE; fruit only and skin "
     "colour (image matching cannot learn from a teacher). Every photo has its background removed."),
):  # fmt: skip
    bullet(text, lead=lead)

sub("Figure 1. Lab avocado 2 (held out), transformer on skin colour, one of its three runs")
doc.add_picture(str(FIG), width=Inches(5.0))
para("Bars: the odds it gives for each day; green: when the fruit actually spoiled. The public model calls the dark "
     "fruit nearly spoiled; after the lab fine-tune it is right on days 1-5 and torn on day 7, where CAML leans the "
     "right way and SAMD does not.", size=8.5, after=4)  # fmt: skip

doc.add_heading("3. Results", level=1)
para("On the 12 lab avocados, each scored only when held out. Day error: days between the predicted and actual "
     "spoil day, every fruit counting equally; mean of 3 runs, with the lowest and highest run in brackets.",
     size=9.5, after=3)  # fmt: skip
HDR = ["Method", "Fruit only: day error", "Skin colour: day error"]
WIDTHS = [2.2, 2.0, 2.0]
for model in MODELS:
    sub(model)
    table(HDR, rows(model), WIDTHS, size=9, highlight=1, gap=False)
para("Green: the lab fine-tune, the photo-only reference the teacher has to beat.", size=8.5, after=4)


doc.add_heading("4. What we learned", level=1)
T, E, O = "Transformer", "Latent SDE", "Latent ODE"
C, R = "Skin colour", "Fruit only"
for lead, text in (
    ("A few lab photos fix most of the gap. ", "Fine-tuning on about 30 lab photos cuts the day error from 1.6-7.5 "
     f"days to 0.9-1.5 days for every model and variant (transformer on skin colour: {d(T, C, 'Public only (phase 1)')}"
     f" to {d(T, C, FT)} days). The dark-skin problem from phase 1 goes away once the model has seen the lab "
     "camera's fruit."),
    ("CAML helps a little, SAMD does not. ", f"CAML lowers the day error for the transformer ({d(T, R, FT)} to "
     f"{d(T, R, CAML)}; {d(T, C, FT)} to {d(T, C, CAML)}) and the latent SDE ({d(E, R, FT)} to {d(E, R, CAML)}; "
     f"{d(E, C, FT)} to {d(E, C, CAML)}), but by under 0.12 days, inside the spread between runs. It hurts the "
     f"latent ODE ({d(O, C, FT)} to {d(O, C, CAML)} on skin colour). SAMD is level with the fine-tune or worse."),
    ("Why so little: too few scans, and labels set by date. ", "Only 20 scanned photos from mostly 4 fruit. And the "
     "lab study gave every avocado the same spoil window (day 7 to 9), so a model that recognises the study day from "
     "the photo already does well: Figure 1's forecasts snap to it. Lab results are therefore flattering and do not "
     "show the app working on store-bought fruit."),
):  # fmt: skip
    bullet(text, lead=lead)

doc.add_heading("5. Next", level=1)
for text in (
    "Keep the lab fine-tune in the app model (with CAML if it holds up on more fruit); scan more fruit, each with its "
    "own spoil day recorded (not one date for all), before investing more in the teacher.",
    "Test on avocados photographed outside the lab (a phone, a kitchen) and followed until each one spoils.",
    "Colour correction against the background; optionally a fridge-or-counter tap; then revisit recalibration.",
):  # fmt: skip
    bullet(text)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print("wrote", OUT)
