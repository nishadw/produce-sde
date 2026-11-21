"""Course progress report, 21 November 2026 (report env). Reads the all-fruit numbers
(outputs/a2sl_all_compare/summary.json, scripts/analysis/a2sl_all_compare.py).
Writes 2026-11-21_progress-report.docx to ../results (``_paths.RESULTS``)."""

import json

from _paths import REPO, RESULTS

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


MATCH = get("Image matching")

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
    hdr_row = OxmlElement("w:tblHeader")  # repeat the header row if the table breaks across pages
    hdr_row.set(qn("w:val"), "true")
    t.rows[0]._tr.get_or_add_trPr().append(hdr_row)
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
    p = doc.add_paragraph()
    p.add_run(text).bold = True
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True


# ---- 1. Introduction ------------------------------------------------------------------------------------------------
doc.add_heading("1. Introduction", level=1)
para("A lot of fruit is thrown away because nobody knew in time that it was about to spoil. This project predicts, "
     "from a few photos of a fruit taken days apart, the day it will spoil. It gives odds for each day (\"most likely "
     "day 8, possibly day 9\"), not one guess, so the user knows how sure it is.")  # fmt: skip
para("Data: 478 avocados from a public study, and 48 mangoes and 15 avocados from our lab, which also have microwave "
     "scans (a sensor that sees inside the fruit).")  # fmt: skip

# ---- 2. Methodology -------------------------------------------------------------------------------------------------
doc.add_heading("2. Methodology", level=1)
for lead, text in (
    ("Latent SDE (main model). ", "Each fruit has a hidden \"ripeness meter\" that climbs over time, a little faster "
     "or slower by chance. Photos are noisy readings of it, and the fruit spoils when it crosses a line. Fruit is only "
     "checked every few days, so we know a window (\"fine on day 7, bad by day 9\"), not the exact day; the model "
     "handles these windows exactly."),
    ("Other models. ", "Latent ODE (the same idea without chance), a transformer, and image matching (predict from "
     "the most similar-looking training fruit)."),
    ("Teacher-student training. ", "A teacher sees photos and microwave scans and coaches a student that needs only "
     "photos, so no scanner is needed in use:"),
):  # fmt: skip
    bullet(text, lead=lead)
for lead, text in (
    ("CAML-KD: ", "the teacher is trained first and kept fixed; the student learns to copy its forecasts."),
    ("SAMD: ", "the two take turns over several rounds; the teacher starts as a copy of the student, trains with the "
     "scan, then the student learns to match the teacher's internal summary of the fruit."),
):  # fmt: skip
    bullet(text, lead=lead, style="List Bullet 2")
bullet("each fruit is hidden from training in turn, then predicted. The score rewards high odds on the day the fruit "
       "actually spoiled (lower is better). Every model must beat a no-photo forecast (how many days fruit usually "
       "have left), and differences must pass a statistical test across fruit.",
       lead="How models are judged: ")  # fmt: skip

# ---- 3. Results -----------------------------------------------------------------------------------------------------
doc.add_heading("3. Results", level=1)
para("Lab avocados, real photos; scored on the 8 fruit seen before and after spoiling.", size=9.5, after=3)
WORD = {"better": "better", "worse": "worse", "no clear difference": "no clear diff."}


def row(model, sees, label):
    r = get(model, sees)
    if "vs" not in r:
        return [model, label, f"{r['obs']:.2f}", "—", "—", f"{r['mae']:.1f} d"]
    lo, hi = r["vs"]["fruit_ci"][:2]
    w = "better" if hi < 0 else "worse" if lo > 0 else "no clear difference"
    return [model, label, f"{r['obs']:.2f}", WORD[w], f"{r['vs']['wins']} of 8", f"{r['mae']:.1f} d"]


TEACH = "teacher sees photo + scanner"
rows = [row("No-photo forecast", "date only", "no photo (reference)"), row("Image matching", "photo", "photo")]
for m in ("Transformer", "Latent SDE", "Latent ODE"):
    rows += [row(m, "photo", "photo"), row(m, f"CAML-KD, {TEACH}", "photo + CAML-KD"),
             row(m, f"SAMD, {TEACH}", "photo + SAMD")]
table(["Model", "Trained with", "Score", "vs no-photo", "Fruit better", "Day error"], rows,
      [1.25, 1.55, 0.6, 1.1, 0.9, 0.8], size=8.5)  # fmt: skip
para("Day error = days between predicted and actual spoil day.", size=8.5, color=GREY, after=4)
for lead, text in (
    ("Nothing beats the no-photo forecast yet. ", f"Image matching comes closest ({MATCH['obs']:.2f} vs {REF:.2f}), "
     f"better on {MATCH['vs']['wins']} of 8 fruit but badly wrong on two."),
    ("Why: ", "the lab labels every avocado spoiled on day 9, so all fruit spoil together and a photo has little to "
     "add. With about 11 fruit to learn from, larger models also overfit."),
    ("Coaching: ", "CAML-KD and SAMD help some models and hurt others; none reaches the no-photo forecast."),
    ("Earlier experiments: ", "the latent SDE's \"80 % sure\" ranges were right 94 % of the time on the public "
     "avocados; the microwave scan beats photos on mangoes but barely helps on avocados."),
):  # fmt: skip
    bullet(text, lead=lead)

# ---- 4. Conclusion and future work ----------------------------------------------------------------------------------
doc.add_heading("4. Conclusion and future work", level=1)
para("The latent SDE gives well-calibrated odds on the large public dataset. On the small lab datasets no model yet "
     "beats a simple no-photo forecast, mainly because the labels give every fruit the same spoil day.")  # fmt: skip
sub("What remains")
for text in (
    "Find photo models that beat the no-photo forecast on the lab avocados.",
    "Storage temperature: let warmth speed up the ripeness meter, so the model works at temperatures it has not seen.",
    "Final report and presentation; publish the code on GitHub.",
):  # fmt: skip
    bullet(text, style="List Number")
sub("Expected by the end of the semester")
for text in (
    "A tested spoilage forecaster that gives honest odds, not just one day (done).",
    "A fair comparison of these models and the two coaching methods on the lab data, with statistical tests.",
):
    bullet(text)
sub("Longer term, if time allows")
for text in (
    "Lab labels that measure each fruit (e.g. firmness) instead of a fixed day.",
    "A short workshop paper on the method.",
):
    bullet(text)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print("wrote", OUT)
