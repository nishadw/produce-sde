"""Plain-language mango results summary (v2), same layout as the avocado summary.

Run mango_figs.py first (ripen env), then this with python-docx (report env).
Numbers in the text and table are typed here from outputs/mango/summary.md; update them when results change.
"""

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parents[2] / "outputs" / "report_figures"  # charts + numbers live here
HERE.mkdir(parents=True, exist_ok=True)
REPO = HERE.parents[1]
OUT = REPO / "results" / "2026-11-06_mango-photo-vs-microwave.docx"
GREY = RGBColor(0x55, 0x5B, 0x65)

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.left_margin = sec.right_margin = Inches(0.85)
sec.top_margin = sec.bottom_margin = Inches(0.6)
st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(10.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after = Pt(4)
for name, size in (("Title", 19), ("Heading 1", 12.5)):
    s = doc.styles[name]
    s.font.name = "Calibri"
    s.font.size = Pt(size)
    s.font.bold = True
    s.font.color.rgb = RGBColor(0x16, 0x19, 0x1E)
    rf = s.element.rPr.rFonts
    rf.set(qn("w:ascii"), "Calibri")
    rf.set(qn("w:hAnsi"), "Calibri")
    for a in ("w:asciiTheme", "w:hAnsiTheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]
    s.paragraph_format.space_before = Pt(0 if name == "Title" else 9)
    s.paragraph_format.space_after = Pt(3)
    s.paragraph_format.keep_with_next = True


def shade(cell, fill):
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(sh)


def para(text="", lead=None, size=None, color=None, italic=False, after=None, style=None):
    p = doc.add_paragraph(style=style)
    if lead:
        r = p.add_run(lead)
        r.bold = True
    r = p.add_run(text)
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    if color:
        r.font.color.rgb = color
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def cell_text(c, main, bold=False, center=False, size=9.5):
    c.text = ""
    p = c.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(main)
    r.bold = bold
    r.font.size = Pt(size)


def caption(text):
    para(text, size=8.5, color=GREY, italic=True, after=4)


doc.add_paragraph("Predicting When Mangoes Will Spoil", style="Title")
para("Nishad Wajge", color=GREY, size=9.5, after=6)

doc.add_heading("Background", level=1)
para(
    "Goal: tell how many days a mango has left before it spoils, from a photo, a microwave scan, or both. A lab bought 48 "
    "mangoes in one batch and stored them together for two weeks. Every other day they scanned every mango with a "
    "microwave sensor (weak radio waves that pass through the fruit and change as its water and sugar change). On each of "
    "those days they also cut open 6 mangoes to measure firmness and sweetness, and photographed those 6. A mango counts "
    "as spoiled when its flesh has gone fully soft (firmness 1 N or less). That happened only on days 12 and 14."
)
para(
    "a shopper has a phone camera, not a microwave scanner. So we also tried a training method from a 2023 paper "
    "(SAMD): a “teacher” model that sees both the photo and the microwave scan coaches a “student” "
    "model that sees only the photo. Once trained, the student needs only the photo.",
    lead="Teaching a photo model with microwave scans: ",
)

doc.add_heading("Main findings", level=1)
for lead, txt in [
    (
        "Microwave beats photos: ",
        "with the microwave scan, the forecast score improves by about a quarter (0.20 to 0.15) for every AI model we "
        "tried, a gap too large to be luck. The scan also estimates firmness and ripeness better. [Chart 1]",
    ),
    (
        "Teaching did not help: ",
        "the photo-only student coached by the microwave teacher was no better than a photo-only model trained on its "
        "own. It only helped models whose photo-only version was poor to begin with. [Chart 1]",
    ),
    (
        "Age is most of the story: ",
        "a simple rule (estimate how old the mango is, then look up how fast this batch spoiled) beats every AI model. "
        "Knowing the true age is best of all. [Chart 2]",
    ),
    (
        "Weak spot: ",
        "no model can tell which of two mangoes of the same age will spoil first, and the AI models expect spoilage 1 to "
        "3 days too late.",
    ),
]:
    para(txt, lead=lead, style="List Bullet", after=2)

doc.add_heading("Results", level=1)
para("Table 1", after=2)
rows = [
    (
        "Forecast score",
        "Scores the predicted chance of being spoiled by each future day against what happened to the batch. "
        "0 is perfect; 0.25 is guessing 50/50 every time. Lower is better.",
        "0.20", "0.21", "0.15", "0.15",
    ),
    (
        "“Spoiled now?” score",
        "The same score for one question: is this mango spoiled today? Lower is better.",
        "0.16", "0.19", "0.16", "0.14",
    ),
    (
        "Matches firmness",
        "Do mangoes predicted to last longer really feel firmer? 1 is a perfect match; 0 is no relation.",
        "0.25", "0.20", "0.61", "0.74",
    ),
    (
        "Predicted spoilage day",
        "Average day the model expects a mango to spoil. The batch really spoiled between days 10 and 14.",
        "day 15.2", "day 15.8", "day 14.6", "day 14.3",
    ),
    (
        "Firmness estimate",
        "How far off the estimated firmness is, on average. For scale: about 38 N on day 0, 1 N when spoiled.",
        "6.6 N", "—", "5.7 N", "5.6 N",
    ),
    (
        "Ripeness class",
        "How often the estimated class (unripe, underripe, ripe, overripe, spoiled) is right or one step off.",
        "81 %", "—", "94 %", "94 %",
    ),
]  # fmt: skip
hdr = ["What we measure", "What it means", "Photo only", "Photo, taught by microwave", "Microwave only",
       "Photo + microwave"]  # fmt: skip
widths = [1.15, 2.45, 0.75, 0.85, 0.8, 0.8]
t = doc.add_table(rows=len(rows) + 1, cols=len(hdr))
t.style = "Table Grid"
t.alignment = WD_TABLE_ALIGNMENT.CENTER
t.autofit = False
for j, w in enumerate(widths):
    t.columns[j].width = Inches(w)
for j, m in enumerate(hdr):
    c = t.cell(0, j)
    c.width = Inches(widths[j])
    cell_text(c, m, bold=True, center=j > 1, size=9)
    shade(c, "DCE6F2")
for i, row in enumerate(rows, start=1):
    for j, v in enumerate(row):
        c = t.cell(i, j)
        c.width = Inches(widths[j])
        cell_text(c, v, bold=(j == 0), center=j > 1, size=9 if j > 1 else 9.5)
caption(
    "First four rows: the standard AI model (GRU), averaged over 3 training runs; every mango is scored once, by a "
    "model that never saw it in training. Last two rows: a simple read-out from the same inputs. The taught photo "
    "model only forecasts spoilage, so those two cells are empty."
)
para(
    "Adding the microwave scan helps on every measure. The forecast score drops from 0.20 to 0.15, and the "
    "match with firmness rises from 0.25 to 0.74. The firmness estimate is a newton closer, and the ripeness class is "
    "within one step 94 % of the time instead of 81 %. The photo still adds a little on top of the scan. It is what "
    "the cut fruit looked like on its last day, so it helps most with “spoiled now?” (0.16 to 0.14). Teaching "
    "the photo model with the scan left it about where it was (0.21 vs 0.20).",
    lead="Comparing the inputs. ",
    after=4,
)

doc.add_heading("Chart 1: Which input helps", level=1)
doc.add_picture(str(HERE / "mango_inputs.png"), width=Inches(6.3))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
caption(
    "Each group is one choice of input. GRU and the transformer are widely used AI models for data that arrives over "
    "time. The ripeness-score model is the one from the avocado study: a hidden ripeness score that drifts upward "
    "until it crosses a spoilage line. Bars are averages over 3 training runs; black lines show how much they vary. "
    "The ripeness-score model, which led on avocados, "
    "trails here, because it was built to follow a fruit over many days and each mango here gives one snapshot."
)

doc.add_heading("Chart 2: Why age matters most", level=1)
doc.add_picture(str(HERE / "mango_age.png"), width=Inches(6.2))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
caption(
    "The blue “simple rule” bars guess the mango's age from its photo or scan, then read off how much of this "
    "batch had spoiled by that age. The grey bars are the best AI model for that input. The dark bar uses the true "
    "age, which a shopper never knows. Because all 48 mangoes ripened together, age explains most of when they "
    "spoiled."
)
doc.save(OUT)
print(OUT)
