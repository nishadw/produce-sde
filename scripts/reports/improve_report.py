"""Background removal and recalibration report, 22 November 2026 (report env). Reads
outputs/a2sl_improve/summary.json (scripts/analysis/a2sl_improve_compare.py) and the figure from
fruit_mask_figure.py. Writes 2026-11-22_background-removal-and-calibration.docx to ../results."""

import json

from _docstyle import GREY, bullet, doc, para, sub, table
from _paths import FIGS, REPO, RESULTS
from docx.shared import Inches, Pt

S = json.loads((REPO / "outputs" / "a2sl_improve" / "summary.json").read_text())
REF, REF_MAE = S["ref"], S["ref_mae"]
OUT = RESULTS / "2026-11-22_background-removal-and-calibration.docx"
FEAT = {
    "original photo": "Original photo",
    "fruit only (background removed)": "Fruit only",
    "skin colour (12 numbers)": "Skin colour",
    "ripeness reader trained on public avocados": "Public reader",
    "original photo + CAML-KD": "Original + CAML-KD",
    "original photo + SAMD": "Original + SAMD",
}


def verdict(r):
    lo, hi = r["vs"]["fruit_ci"][:2]
    if lo == hi == 0:
        return "same (uses no-photo)"
    return "better" if hi < 0 else "worse" if lo > 0 else "no clear diff."


def rows(model):
    ref = [["No photo (reference)", f"{REF:.2f}", "—", "—", "—", f"{REF_MAE:.1f} d"]]
    return ref + [[FEAT[r["features"]], f"{r['raw']:.2f}", f"{r['cal']:.2f}", f"{r['w']:.0%}", verdict(r),
                   f"{r['mae']:.1f} d"] for r in S["models"][model]]  # fmt: skip


col = next(r for r in S["models"]["Image matching"] if r["features"].startswith("skin"))

# =====================================================================================================================
doc.add_paragraph("Removing the Background and Recalibrating", style="Title")
para("CMSC 498Z final project · Nishad Wajge · 22 November 2026", color=GREY, size=9.5, after=6)

doc.add_heading("1. Summary", level=1)
para(f"We tried three fixes on the lab avocados. One worked: image matching on the skin colour of the fruit alone "
     f"scores {col['raw']:.2f} against {REF:.2f} for the no-photo forecast (lower is better). It is better on all 8 "
     f"fruit and off by {col['mae']:.1f} days on average, against {REF_MAE:.1f} days. The neural models (transformer, "
     f"latent SDE, latent ODE) gain little or nothing.")  # fmt: skip

doc.add_heading("2. What we changed", level=1)
for lead, text in (
    ("Background removed (all four models). ", "Each photo is cut down to the fruit; everything else is painted "
     "grey. The same image network then reads the fruit alone. All four models were rerun on these photos."),
    ("Skin colour. ", "Instead of the image network, 12 simple numbers: how light, how green-red and how blue-yellow "
     "the skin is (average, spread, darkest and lightest parts), from fruit pixels only."),
    ("Public reader. ", "A ripeness-stage reader trained on the 478 public avocados (1 = unripe to 5 = overripe), "
     "then applied to the lab photos."),
    ("Recalibration. ", "A model's odds can be too sure (or not sure enough). Recalibration turns two knobs on "
     "each forecast after training: (a) sharpen or flatten its odds, and (b) blend it with the no-photo forecast, "
     "e.g. 70 % model + 30 % no-photo. Both knobs are set on the other 11 fruit, never on the fruit being scored. "
     "If the blend ends at 100 % no-photo, the photos added nothing."),
):  # fmt: skip
    bullet(text, lead=lead)

sub("Figure 1. Photos with and without the background")
doc.add_picture(str(FIGS / "fruit_mask_examples.png"), width=Inches(6.9))
para("Lab photos (left) sit on a paper background whose colour and lighting change from day to day; the public "
     "photos (right) use a plain white one. The cut-out sometimes clips glare at the top of dark fruit.",
     size=8.5, after=4)  # fmt: skip

doc.add_heading("3. Results", level=1)
para("Lab avocados, real photos only (no linear or logistic extrapolation, unlike the 18 Nov half-synthetic study); "
     "scored on the 8 fruit seen before and after spoiling. Score: low when the forecast puts high odds on the "
     "actual spoil window; sure-but-wrong odds are penalised hard.", size=9.5, after=3)  # fmt: skip
HDR = ["Photo features", "As trained", "Recalibrated", "No-photo share", "vs no-photo", "Day error"]
WIDTHS = [1.6, 0.8, 0.9, 1.0, 1.4, 0.8]
for model in ("Image matching", "Transformer", "Latent SDE", "Latent ODE"):
    sub(model)
    best = 3 if model == "Image matching" else None
    table(HDR, rows(model), WIDTHS, size=8.5, highlight=best, gap=False)
para("No-photo share: how much of the recalibrated forecast is the no-photo forecast (typical value over the 12 "
     "fruit). \"vs no-photo\" (statistical test across fruit) and \"Day error\" (days between predicted and "
     "actual spoil day) are for the recalibrated forecast; \"same\" means it is 100 % no-photo. CAML-KD and SAMD "
     "were run on the original photos only.", size=8.5, after=4)  # fmt: skip
doc.paragraphs[-1].paragraph_format.space_before = Pt(3)

doc.add_heading("4. What we learned", level=1)
for lead, text in (
    ("The background was misleading image matching. ", "Matching on the full photo mostly found photos with a "
     "similar background and lighting. Fruit only: 1.23 to 0.83; skin colour: 0.28."),
    ("Recalibration helps the weaker forecasts. ", "Image matching on the original photo improves from 1.23 to "
     "0.92, and the transformer from 1.34 to 1.10. It slightly hurts skin colour (0.28 to 0.36), which was already "
     "well calibrated."),
    ("The latent SDE and ODE add nothing yet. ", "For almost every photo input, the best recalibration on the "
     "other fruit is to ignore them and use the no-photo forecast."),
    ("The public reader saturates. ", "It separates day 1 from later days but reads days 3 to 9 as nearly the "
     "same stage, so it cannot tell how close a fruit is to spoiling."),
):  # fmt: skip
    bullet(text, lead=lead)

doc.add_heading("5. Caveats and next steps", level=1)
for text in (
    "Colour is still measured under each day's lighting. All fruit were photographed together each day, so "
    "colour may partly be telling the model the day. Next: correct colour using the background (white balance).",
    "Only 8 fruit are scored; the result needs to hold on new fruit.",
    "Next: feed skin colour into the latent SDE in place of the image network, and test colour on the mangoes.",
):  # fmt: skip
    bullet(text)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print("wrote", OUT)
