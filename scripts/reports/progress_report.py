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


SD, MATCH = get("Storage day"), get("Image matching")

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
    p = doc.add_paragraph()
    p.add_run(text).bold = True
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True


# ---- 1. Introduction ------------------------------------------------------------------------------------------------
doc.add_heading("1. Introduction", level=1)
para("Fruit goes bad on its own schedule, and a lot of it is thrown away because nobody knew in time. This project "
     "builds a computer model that looks at a few photos of a fruit, taken a few days apart, and predicts the day it "
     "will spoil. Instead of a single guess (\"day 8\"), it gives odds for each day (\"most likely day 8, possibly "
     "day 9\"), so the user also knows how sure the model is.")  # fmt: skip
para("I use three datasets: 478 avocados from a public study, photographed daily at three storage temperatures; 48 "
     "mangoes and 15 avocados from our lab, which also have microwave scans (a sensor that sees inside the fruit).")  # fmt: skip
table(
    ["", "Status"],
    [
        ["Done", "A full software package with 6 prediction models, 7 completed experiments, each written up"],
        ["Best result", f"A new, simple \"storage-day\" model is the first to beat the no-photo forecast on the lab "
                        f"avocados: score {SD['obs']:.2f} vs {REF:.2f} (lower is better), and better on "
                        f"{SD['vs']['wins']} of 8 fruit"],
        ["Open question", "Is that model really reading the fruit, or the photo session it was taken in?"],
    ],
    [1.1, 5.7],
    highlight=1,
)  # fmt: skip

# ---- 2. Methodology -------------------------------------------------------------------------------------------------
doc.add_heading("2. Methodology", level=1)
for lead, text in (
    ("Main model (latent SDE). ", "Think of each fruit as having a hidden \"ripeness meter\" that climbs over time, "
     "a little faster or slower by chance. The photos are blurry readings of the meter, and the fruit spoils the "
     "first time the meter crosses a line. The model learns how the meter moves. Because the lab only checks fruit "
     "every couple of days, we never know the exact spoil day, only a window (\"fine on day 7, bad by day 9\"); the "
     "model handles these windows exactly, with a math method (PDEs) rather than by trial simulation."),
    ("Temperature. ", "Warmth just makes the meter run faster, following a standard chemistry law. So the model can "
     "predict at a temperature it has never seen."),
    ("Other models compared. ", "Simpler time-series networks (latent ODE, transformer), a classic survival model "
     "(Weibull), \"image matching\" (find the training fruit that looks most similar) and a new storage-day model, "
     "which guesses from the photo how many days the fruit has been stored, then uses when similar fruit spoiled."),
    ("Teacher-student training. ", "A \"teacher\" model sees photos and microwave scans and coaches a \"student\" "
     "that only needs photos, so the scanner is not needed once the model is in use."),
    ("How models are judged. ", "Each fruit is hidden from training in turn and then predicted. The main score "
     "rewards giving high odds to the day the fruit actually spoiled (lower is better). Every model must beat a "
     "no-photo forecast, which ignores the photo and says how many days fruit usually have left. Differences are "
     "only called real if they hold up in a statistical test across fruit."),
):  # fmt: skip
    bullet(text, lead=lead)

# ---- 3. Results -----------------------------------------------------------------------------------------------------
doc.add_heading("3. Results", level=1)
table(
    ["Date", "Experiment", "What we learned"],
    [
        ["28 Oct", "Public avocados", "The temperature idea works: at an unseen temperature, error drops 1.5–2.4×. "
                                      "Its \"80 % sure\" ranges are right 94 % of the time."],
        ["6–10 Nov", "Lab mangoes and avocados, scanner",
         "On mangoes the microwave scan beats photos; on avocados it barely helps. Three of the six scan angles are "
         "clearly better."],
        ["12–13 Nov", "Teacher-student training",
         "Students improved, but because of how they were trained, not because of what the scanner showed."],
        ["17–21 Nov", "Lab avocados, all 15 fruit",
         "Fixed a data bug. Only the storage-day model beats the no-photo forecast."],
    ],
    [0.75, 1.65, 4.4],
    size=8.5,
    highlight=3,
)  # fmt: skip
doc.add_picture(str(FIGS / "al_scores.png"), width=Inches(5.6))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
para("Lab avocados: each model's score (lower is better). Dashed line = the no-photo forecast; only the blue bar is "
     "below it. * = the teacher sees photos only.", size=8.5, color=GREY, after=4)  # fmt: skip
for lead, text in (
    ("A simple guess is hard to beat. ", f"On the lab avocados every complex model scores worse than the no-photo "
     f"forecast ({REF:.2f}); the best of them, image matching, scores {MATCH['obs']:.2f}."),
    ("Why: the labels. ", "The lab marks every avocado as spoiled on day 9, so all fruit spoil at the same time. The "
     "only useful thing a photo can say is how far along the fruit is (the skin turns from green to black). The "
     f"storage-day model asks exactly that, and its guess of the spoil day is off by only {SD['mae']:.1f} days."),
    ("Too little data for big models. ", "With about 11 fruit to learn from, large models memorise noise; small "
     "models do better."),
):  # fmt: skip
    bullet(text, lead=lead)

# ---- 4. Conclusion and future work ----------------------------------------------------------------------------------
doc.add_heading("4. Conclusion and future work", level=1)
para("The main model works well on the large public dataset, especially at new temperatures. On the small lab "
     "datasets, a simple model that estimates how far along the fruit is wins. One risk remains: every fruit was "
     "photographed in the same session each day, so the model might be recognising the lighting of that day rather "
     "than the fruit itself.")  # fmt: skip
sub("What remains to be done")
for text in (
    "Rule out the session risk: rerun the storage-day model on photos cropped to just the fruit, and on skin colour "
    "alone.",
    "Test the storage-day model on the mangoes and the public avocados, where fruit spoil on different days.",
    "Combine the two: use the storage-day guess as the starting point of the main model.",
    "Write the final report and presentation, and publish the code on GitHub.",
):  # fmt: skip
    bullet(text, style="List Number")
sub("Realistic contributions by the end of the semester")
for text in (
    "A tested spoilage forecaster that gives honest odds and works at new storage temperatures (done).",
    "A fair comparison of six models on small lab datasets, with statistical tests.",
    "The storage-day model with the session check done, and clear advice for the lab on when to trust it and which "
    "scanner angles to keep.",
):  # fmt: skip
    bullet(text)
sub("Longer-term goals, if time allows")
for text in (
    "Lab labels that measure each fruit (e.g. firmness) instead of a fixed day, so models can learn why fruit differ.",
    "A new batch of fruit photographed on different dates, to rule out the session effect for good.",
    "A short workshop paper on the method.",
):  # fmt: skip
    bullet(text)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print("wrote", OUT)
