"""Course progress report, 21 November 2026 (report env). Reads the all-fruit numbers
(outputs/a2sl_all_compare/summary.json, scripts/analysis/a2sl_all_compare.py) and its chart (all_figs.py).
Writes 2026-11-21_progress-report.docx to ../results (``_paths.RESULTS``)."""

import json

from _paths import FIGS, REPO, RESULTS

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

S = json.loads((REPO / "outputs" / "a2sl_all_compare" / "summary.json").read_text())
REF = S["ref_obs"]
OUT = RESULTS / "2026-11-21_progress-report.docx"
GREY = RGBColor(0x55, 0x5B, 0x65)


def get(model, sees="photo"):
    return next(r for r in S["rows"] if r["model"] == model and r["sees"] == sees)


SD, MATCH, TF = get("Storage day"), get("Image matching"), get("Transformer")

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.left_margin = sec.right_margin = Inches(0.8)
sec.top_margin = sec.bottom_margin = Inches(0.7)
st = doc.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(10.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after = Pt(4)
for name, size in (("Title", 20), ("Heading 1", 13)):
    s = doc.styles[name]
    s.font.name, s.font.size, s.font.bold = "Calibri", Pt(size), True
    s.font.color.rgb = RGBColor(0x16, 0x19, 0x1E)
    rf = s.element.rPr.rFonts
    rf.set(qn("w:ascii"), "Calibri")
    rf.set(qn("w:hAnsi"), "Calibri")
    for a in ("w:asciiTheme", "w:hAnsiTheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]
    s.paragraph_format.space_before = Pt(0 if name == "Title" else 10)
    s.paragraph_format.space_after = Pt(4)
    s.paragraph_format.keep_with_next = True


def shade(cell, fill):
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(sh)


def para(text="", lead=None, size=None, color=None, after=None):
    p = doc.add_paragraph()
    if lead:
        p.add_run(lead).bold = True
    r = p.add_run(text)
    if size:
        r.font.size = Pt(size)
    if color:
        r.font.color.rgb = color
    if after is not None:
        p.paragraph_format.space_after = Pt(after)


def bullet(text, lead=None, style="List Bullet"):
    p = doc.add_paragraph(style=style)
    if lead:
        p.add_run(lead).bold = True
    p.add_run(text)
    p.paragraph_format.space_after = Pt(2)


def table(hdr, rows, widths, size=9, highlight=None):
    t = doc.add_table(rows=len(rows) + 1, cols=len(hdr))
    t.style, t.alignment, t.autofit = "Table Grid", WD_TABLE_ALIGNMENT.CENTER, False
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    t._tbl.tblPr.append(layout)
    for j, w in enumerate(widths):
        t.columns[j].width = Inches(w)
    for i, r in enumerate([hdr, *rows]):
        for j, val in enumerate(r):
            c = t.cell(i, j)
            c.width = Inches(widths[j])
            c.text = ""
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(str(val))
            run.font.size = Pt(size)
            run.bold = i == 0 or j == 0
            if i == 0:
                shade(c, "DCE6F2")
            elif highlight is not None and i - 1 == highlight:
                shade(c, "E3F0DC")
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


# =====================================================================================================================
doc.add_paragraph("Progress Report: Predicting When Produce Spoils", style="Title")
para("CMSC 498Z final project · Nishad Wajge · 21 November 2026", color=GREY, size=9.5, after=6)


def sub(text):
    doc.add_paragraph().add_run(text).bold = True


# ---- 1. Introduction ------------------------------------------------------------------------------------------------
doc.add_heading("1. Introduction", level=1)
para("Fruit spoils on its own schedule, and knowing when lets stores and labs act before it is wasted. The goal of "
     "this project is to forecast, from a few photos of a fruit (and, for lab fruit, microwave scans) taken at "
     "irregular times, a calibrated probability distribution over the day it will spoil. The forecast should say not "
     "only the most likely day but how sure it is.")  # fmt: skip
para("Three datasets are used: a public avocado set (478 fruit photographed daily at 3 storage temperatures), 48 lab "
     "mangoes (photo + microwave scans) and 15 lab avocados (photo + microwave scans). The table below gives the "
     "project's status as of this report.")  # fmt: skip
table(
    ["", "Status"],
    [
        ["Code", "Python package ripen (JAX / Equinox / Diffrax): 6 forecasting models, 2 teacher-student training "
                 "methods, exact first-passage PDE solver, CPU test suite, SLURM drivers for the UMIACS cluster"],
        ["Studies", "7 completed, each with a written report; about 110 training runs in the latest one"],
        ["Best result so far", f"New storage-day model: score {SD['obs']:.2f} vs {REF:.2f} for the no-photo "
                               f"forecast, better on {SD['vs']['wins']} of 8 lab avocados; day error "
                               f"{SD['mae']:.1f} d (vs {get('No-photo forecast', 'date only')['mae']:.1f} d)"],
        ["Main open question", "Is the storage-day model reading the fruit, or the photo session it was taken in?"],
    ],
    [1.4, 5.4],
    highlight=2,
)  # fmt: skip

# ---- 2. Methodology -------------------------------------------------------------------------------------------------
doc.add_heading("2. Methodology", level=1)
sub("Core model: latent neural SDE")
for lead, text in (
    ("Ripeness as a random process. ", "A hidden ripeness value drifts and diffuses over time according to a neural "
     "stochastic differential equation (SDE); the photos are noisy readings of it."),
    ("Spoilage as a threshold crossing. ", "The fruit spoils the first time ripeness crosses a boundary. The "
     "probability of that, and the likelihood of interval-censored labels ('fine on day 7, spoiled by day 9'), is "
     "computed exactly from the Kolmogorov PDEs (Crank–Nicolson), not by simulating paths. Validated against the "
     "known inverse-Gaussian law (to 1e-3) and by recovering parameters on synthetic data."),
    ("Temperature as a clock. ", "Storage temperature changes only the speed of ripening (Arrhenius law with a learned "
     "activation energy), so the model can forecast at a temperature it never trained on."),
):  # fmt: skip
    bullet(text, lead=lead)
sub("Comparison models and training methods")
for lead, text in (
    ("Neural: ", "latent ODE (the deterministic version) and a small transformer."),
    ("Classical: ", "a Weibull survival model and image matching (forecast from the most similar training fruit)."),
    ("Storage day (new): ", "a small classifier estimates the fruit's storage day from its photos, combining them "
     "through the days between them; the spoilage day is then read off the training fruit."),
    ("Teacher-student training: ", "CAML-style distillation and SAMD, where a teacher that sees photos + microwave "
     "scans trains a student that needs only photos at test time."),
):  # fmt: skip
    bullet(text, lead=lead)
sub("Evaluation")
bullet(" censored log score (lower is better), plus day error, CRPS and interval coverage.", lead="Scores:")
bullet(" each fruit held out in turn, forecast from every start day; 5 seeds; paired bootstrap over fruit for "
       "significance.", lead="Protocol:")  # fmt: skip
bullet(" every model is compared with a no-photo forecast (how many days fruit usually have left). A model that "
       "cannot beat it is not getting anything useful from the photos.", lead="Reference:")  # fmt: skip

# ---- 3. Results -----------------------------------------------------------------------------------------------------
doc.add_heading("3. Results", level=1)
table(
    ["Date", "Study", "Finding"],
    [
        ["28 Oct", "Public avocados: latent SDE vs baselines",
         "At an unseen temperature the Arrhenius clock cuts error 1.5–2.4×; 80 % intervals cover 94 % (SDE) vs 64 % "
         "(ODE). A small transformer wins in-distribution."],
        ["6 Nov", "Lab mangoes: photo vs microwave", "Microwave scans beat photos; nothing beats knowing the fruit's age."],
        ["10 Nov", "Lab avocados: scanner views", "Rotation views 0°/120°/240° beat 60°/180°/300° (14 of 16 tests)."],
        ["12–13 Nov", "Time-series models × teacher-student",
         "Latent ODE best on both fruits; teacher-student gains come from the training schedule, not the scanner."],
        ["17–18 Nov", "Lab avocados with filled-in days",
         "Found and fixed a photo-caching bug; no neural model beat the no-photo forecast."],
        ["19–21 Nov", "Lab avocados, all 15 fruit, real photos",
         f"Only the storage-day model beats the no-photo forecast ({SD['obs']:.2f} vs {REF:.2f})."],
    ],
    [0.75, 1.9, 4.15],
    size=8.5,
    highlight=5,
)  # fmt: skip
doc.add_picture(str(FIGS / "al_scores.png"), width=Inches(6.6))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
para("Lab avocados, 8 fruit observed to spoil, each model's best setting. Lower is better; dashed line = no-photo "
     "forecast. * = the teacher sees photos only.", size=8.5, color=GREY)  # fmt: skip
sub("What the results show")
for lead, text in (
    ("The simple reference is hard to beat. ", "On the lab avocados most models, including every neural one, score "
     f"worse than the no-photo forecast ({REF:.2f}): transformer {TF['obs']:.2f}, image matching {MATCH['obs']:.2f}."),
    ("The labels decide what a photo can add. ", "The lab avocados are labelled by a day rule (spoiled on day 9), so "
     "every fruit spoils at the same time. A photo cannot explain differences between fruit; it can only tell how far "
     "along the fruit is. The storage-day model is built around exactly that, and it is the one model that wins "
     f"({SD['obs']:.2f}, day error {SD['mae']:.1f} d)."),
    ("Small data favours small models. ", "With about 11 training fruit per fold, the latent ODE/SDE and transformer "
     "overfit; the few-parameter models do best."),
    ("The scanner helps less than expected. ", "On avocados the microwave scanner barely improves even the teacher; on "
     "mangoes it clearly helps."),
):  # fmt: skip
    bullet(text, lead=lead)

# ---- 4. Conclusion and future work ----------------------------------------------------------------------------------
doc.add_heading("4. Conclusion and future work", level=1)
para("The latent SDE delivers calibrated forecasts and transfers to unseen storage temperatures on the large public "
     "set. On the small lab sets, the deciding factor is what the labels can reveal: estimating how far along a fruit "
     "is from its photo beats every heavier model. The main risk to that result is that all fruit were photographed "
     "in the same session each day, so the model might partly recognise the session rather than the fruit.")  # fmt: skip
sub("What remains to be done")
for text in (
    "Check that the storage-day model reads the fruit, not the session: rerun it on photos cropped to the fruit and "
    "on colour-only features.",
    "Run the storage-day model on the lab mangoes and the public avocado set, where spoilage does vary between fruit.",
    "Use the storage-day estimate as the starting state of the latent SDE, so the SDE keeps its calibrated "
    "uncertainty but starts from a good ripeness estimate.",
    "Report a second score (CRPS) next to the log score in every table.",
    "Final report and presentation; clean up the code and push it to GitHub before the cluster account is archived.",
):  # fmt: skip
    bullet(text, style="List Number")
sub("Expected contributions by the end of the semester")
for lead, text in (
    ("A tested latent-SDE spoilage forecaster ", "with exact first-passage likelihoods and a temperature clock that "
     "transfers to unseen storage temperatures (done; written up)."),
    ("An honest benchmark on small lab datasets: ", "six models and two teacher-student methods against a no-photo "
     "reference, with paired significance tests across fruit."),
    ("The storage-day model, ", "with the session check done, as the best photo-based forecaster on the lab avocados "
     "and a clear statement of when it should and should not be trusted."),
    ("Practical findings for the lab: ", "which scanner views to keep, and that a photo-only student gains from the "
     "training schedule rather than from the scanner."),
):  # fmt: skip
    bullet(text, lead=lead)
sub("Longer-term goals, if time allows")
for text in (
    "Per-fruit spoilage labels (e.g. firmness) instead of the day rule, so models can learn why fruit spoil at "
    "different times.",
    "A new batch of fruit photographed on different dates, to rule out session effects for good.",
    "Combine the storage-day encoder with the latent SDE across temperatures and fruit types.",
    "A short paper or workshop submission on exact-likelihood latent SDEs for shelf-life forecasting.",
):  # fmt: skip
    bullet(text)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print("wrote", OUT)
