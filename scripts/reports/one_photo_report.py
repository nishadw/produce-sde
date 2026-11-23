"""One-photo study report, phase 1 (report env). Reads outputs/one_photo/summary.json
(scripts/analysis/one_photo_compare.py) and the example figure (one_photo_figure.py --run latent_sde_fruit
--seed 2).
Writes 2026-11-23_one-photo-phase1.docx to ../results."""

import json

from _docstyle import GREY, bullet, doc, para, sub, table
from _paths import FIGS, REPO, RESULTS
from docx.shared import Inches, Pt

S = json.loads((REPO / "outputs" / "one_photo" / "summary.json").read_text())
OUT = RESULTS / "2026-11-23_one-photo-phase1.docx"
MODELS = ("Image matching", "Transformer", "Latent SDE", "Latent ODE")
FIG = FIGS / "one_photo_examples_latent_sde_fruit.png"


def rows(model):
    out = []
    for feat, r in S[model].items():
        p, q = r["public"], r["lab"]
        out.append([feat, f"{p['score']:.2f}", f"{p['mae']:.1f} d", f"{p['cov80']:.0%}", f"{q['score']:.2f}",
                    f"{q['mae']:.1f} d", f"{q['cov80']:.0%}"])  # fmt: skip
    return out


def best_row(model):
    feats = list(S[model])
    return min(range(len(feats)), key=lambda i: S[model][feats[i]]["public"]["score"])


# =====================================================================================================================
doc.add_paragraph("One Photo In, Days Left Out: Phase 1", style="Title")
para("CMSC 498Z final project · Nishad Wajge · 23 November 2026", color=GREY, size=9.5, after=6)

doc.add_heading("1. Goal", level=1)
para("The end product is an app: photograph an avocado, get back how many days until it spoils. So every model now "
     "sees one photo and nothing else: no date, no storage day, no storage temperature, no expert ripeness rating. "
     "Recalibration is set aside until this phase is in.")  # fmt: skip

doc.add_heading("2. What I did", level=1)
for lead, text in (
    ("One photo per example. ", "Every photo taken before a fruit spoiled became its own example, labelled with how "
     "many days that fruit had left. Models see that single photo."),
    ("Removed every non-photo input. ", "The models had been given the time since the first photo, the storage "
     "temperature and group, and (in the latent SDE) the expert ripeness stage. All are now hidden. The SDE still "
     "learns from the stage ratings during training, never at prediction time."),
    ("Trained on the public avocados. ", "334 fruit (about 4,100 photos) train, 71 pick settings, 73 (905 photos) "
     "are the test. The 15 lab avocados (42 photos; 3 were already spoiled in their first photo) are a second test "
     "from a different camera and room; they never train."),
    ("Three photo variants. ", "Full photo, fruit only (background removed) and skin colour (12 numbers), "
     "each read the same way for public and lab photos."),
    ("Four models. ", "Image matching, transformer (with one photo in, a direct photo-to-days network), latent SDE "
     "and latent ODE. Matching ran once; the others with 3 random seeds, averaged."),
):  # fmt: skip
    bullet(text, lead=lead)

sub("Figure 1. What the app would say")
doc.add_picture(str(FIG), width=Inches(6.9))
para("Latent SDE on fruit-only photos (one of its three runs), lab avocado 2, which it never trained on. Bars: the "
     "odds it gives for each day; green: when the fruit actually spoiled. Day 1 is about right; once the skin turns "
     "black (day 3) it says too soon.", size=8.5, after=4)  # fmt: skip

doc.add_heading("3. Results", level=1)
para("Score: low when the forecast puts high odds on the days the fruit actually spoiled (lower is better). Day "
     "error: days between the predicted and actual spoil day. 80 % range: how often the actual day fell inside the "
     "model's \"80 % sure\" range (80 % is ideal). Every fruit counts equally.", size=9.5, after=3)  # fmt: skip
HDR = ["Photo", "Public score", "Public day error", "Public 80% range", "Lab score", "Lab day error", "Lab 80% range"]
WIDTHS = [1.1, 0.85, 0.95, 1.0, 0.8, 0.9, 0.9]
for model in MODELS:
    if model not in S:
        continue
    sub(model)
    table(HDR, rows(model), WIDTHS, size=8.5, highlight=best_row(model), gap=False)
para("Green: the model's best photo variant on the public test.", size=8.5, after=4)
doc.paragraphs[-1].paragraph_format.space_before = Pt(3)

doc.add_heading("4. What we learned", level=1)
for lead, text in (
    ("A photo alone gets within 2 to 3 days. ", "On the 73 held-out public avocados every model is off by 2.2 to "
     "3.2 days on average. Skin colour is best for the transformer (score 1.80) and image matching (2.2 days); the "
     "gaps between the top models are small."),
    ("Removing the background matters most in a new setting. ", "On the public photos (plain white background) it "
     "changes little. On the lab photos it cuts the day error for every model: matching 2.3 to 2.0, transformer 4.2 "
     "to 2.1, latent SDE 3.8 to 1.9, latent ODE 2.8 to 2.1 days."),
    ("The latent SDE transfers best to the lab. ", "It has the best lab score (1.84, skin colour) and its 80 % "
     "ranges hold on the lab fruit (73 to 80 %). The transformer on skin colour breaks there (9.1 days off): lab "
     "skin is about 2 standard deviations darker than the average public photo, far from what it learned."),
    ("Dark skin is the hard case. ", "Public avocados turn black just before spoiling; the lab ones were black by "
     "day 3 and lasted 4 to 6 more days. Models trained on public photos call dark lab fruit nearly spoiled."),
    ("Fridge or counter matters. ", "Fruit kept at 10 °C are hardest (2.4 to 4.0 days off, against 1.2 to 2.8 at "
     "20 °C): a photo cannot tell where the fruit will be kept."),
    ("Uncertainty: ", "the latent SDE's 80 % ranges are too wide on the public fruit (right 98 % of the time); "
     "image matching's are too narrow on the lab fruit (48 to 73 %)."),
):  # fmt: skip
    bullet(text, lead=lead)

doc.add_heading("5. Next (phase 2)", level=1)
for text in (
    "Microwave scans as a training-only teacher (CAML-KD, SAMD) on the lab avocados and mangoes: the scan sees "
    "inside the fruit, which matters most once the skin is black. The app still needs only a photo.",
    "Fine-tune the public model on the lab avocados (each fruit held out in turn).",
    "Colour correction against the background, so skin colour means the same under any light.",
    "Optionally, one tap in the app, fridge or counter (not a date), for the 10 °C case.",
    "Then revisit recalibration.",
):  # fmt: skip
    bullet(text)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print("wrote", OUT)
