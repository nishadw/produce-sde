"""Background removal report, 22 November 2026 (report env): every model variation as trained. Reads
outputs/a2sl_improve/summary.json (scripts/analysis/a2sl_improve_compare.py) and the figure from
fruit_mask_figure.py. Writes 2026-11-22_background-removal.docx to ../results."""

import json

from _docstyle import GREY, bullet, doc, para, sub, table
from _paths import FIGS, REPO, RESULTS
from docx.shared import Inches, Pt

S = json.loads((REPO / "outputs" / "a2sl_improve" / "summary.json").read_text())
OUT = RESULTS / "2026-11-22_background-removal.docx"
FEAT = {
    "original photo": "Original photo",
    "fruit only (background removed)": "Fruit only",
    "skin colour (12 numbers)": "Skin colour",
    "ripeness reader trained on public avocados": "Public reader",
    "original photo + CAML-KD": "Original + CAML-KD",
    "original photo + SAMD": "Original + SAMD",
}


def rows(model):
    return [[FEAT[r["features"]], f"{r['raw']:.2f}", f"{r['raw_mae']:.1f} d"] for r in S["models"][model]]


col = next(r for r in S["models"]["Image matching"] if r["features"].startswith("skin"))

# =====================================================================================================================
doc.add_paragraph("Removing the Background: Model Variations", style="Title")
para("CMSC 498Z final project · Nishad Wajge · 22 November 2026", color=GREY, size=9.5, after=6)

doc.add_heading("1. Summary", level=1)
para(f"We reran every model on lab avocado photos with the background removed, and with two new ways of reading the "
     f"fruit. The best result by far: image matching on the skin colour of the fruit alone, score {col['raw']:.2f} "
     f"(lower is better), off by {col['raw_mae']:.1f} days on average. The neural models (transformer, latent SDE, "
     f"latent ODE) change little with any of the variations.")  # fmt: skip

doc.add_heading("2. Variations", level=1)
for lead, text in (
    ("Original photo. ", "The whole photo, read by an image network (ResNet-18)."),
    ("Fruit only. ", "Each photo is cut down to the fruit; everything else is painted grey. The same image network "
     "then reads the fruit alone. All four models were rerun on these photos."),
    ("Skin colour. ", "Instead of the image network, 12 simple numbers: how light, how green-red and how blue-yellow "
     "the skin is (average, spread, darkest and lightest parts), from fruit pixels only."),
    ("Public reader. ", "A ripeness-stage reader trained on the 478 public avocados (1 = unripe to 5 = overripe), "
     "then applied to the lab photos (fruit only)."),
    ("+ CAML-KD / + SAMD. ", "Original photo, coached during training by a teacher that also sees the microwave "
     "scans; the student needs only the photo."),
):  # fmt: skip
    bullet(text, lead=lead)

sub("Figure 1. Photos with and without the background")
doc.add_picture(str(FIGS / "fruit_mask_examples.png"), width=Inches(6.9))
para("Lab photos (left) sit on a paper background whose colour and lighting change from day to day; the public "
     "photos (right) use a plain white one. The cut-out sometimes clips glare at the top of dark fruit.",
     size=8.5, after=4)  # fmt: skip

doc.add_heading("3. Results", level=1)
para("Lab avocados, real photos only (no linear or logistic extrapolation); each fruit is predicted by a model that "
     "never saw it; scored on the 8 fruit seen before and after spoiling. Score: low when the forecast puts high "
     "odds on the actual spoil window. Day error: days between predicted and actual spoil day.",
     size=9.5, after=3)  # fmt: skip
HDR = ["Photo features", "Score", "Day error"]
WIDTHS = [2.2, 1.0, 1.0]
for model in ("Image matching", "Transformer", "Latent SDE", "Latent ODE"):
    sub(model)
    best = 2 if model == "Image matching" else None
    table(HDR, rows(model), WIDTHS, size=8.5, highlight=best, gap=False)
para("CAML-KD and SAMD were run on the original photos only.", size=8.5, after=4)
doc.paragraphs[-1].paragraph_format.space_before = Pt(3)

doc.add_heading("4. What we learned", level=1)
for lead, text in (
    ("The background was misleading image matching. ", "Matching on the full photo mostly found photos with a "
     "similar background and lighting. Fruit only: 1.23 to 0.83; skin colour: 0.28."),
    ("Simple colour beats the image network. ", "For three of the four models, skin colour gives the best or "
     "near-best score; 12 numbers are easier to learn from 11 training fruit than 512 network features."),
    ("The neural models struggle. ", "The transformer, latent SDE and latent ODE stay at 1.2 to 2.9 with every "
     "variation; with about 11 fruit to learn from they overfit."),
    ("The public reader saturates. ", "It separates day 1 from later days but reads days 3 to 9 as nearly the "
     "same stage, so it cannot tell how close a fruit is to spoiling."),
):  # fmt: skip
    bullet(text, lead=lead)

doc.add_heading("5. Caveats", level=1)
for text in (
    "Colour is still measured under each day's lighting. All fruit were photographed together each day, so "
    "colour may partly be telling the model the day. Next: correct colour using the background (white balance).",
    "Only 8 fruit are scored; the result needs to hold on new fruit.",
):  # fmt: skip
    bullet(text)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print("wrote", OUT)
