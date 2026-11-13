"""Plain-language A2SL microwave-view + AML summary, same layout as the avocado and mango summaries.

Run a2sl_views_figs.py first (ripen env; writes the charts and a2sl_numbers.json), then this with python-docx
(report env). Table numbers come from a2sl_numbers.json; numbers in the prose are typed from it and from
outputs/a2sl_views/labels_hidden*/summary*.md and scratch/evenodd.py (view-group test).
"""

import json
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
OUT = REPO / "results" / "2026-11-10_avocado-microwave-scanner-views.docx"
N = json.loads((HERE / "a2sl_numbers.json").read_text())
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




def v(key, metric, recal=False, fmt="{:.2f}"):
    return fmt.format(N["views"][key + ("_recal" if recal else "")][metric][0])


doc.add_paragraph("Which Microwave View Best Predicts Avocado Spoilage", style="Title")
para("Nishad Wajge", color=GREY, size=9.5, after=6)

doc.add_heading("Background", level=1)
para(
    "Goal: find which of the microwave scanner's six views best predicts how many days an avocado has left, and "
    "whether the scanner can teach a photo-only model. The lab followed 15 avocados for two weeks. Four of them "
    "(avocados 1, 2, 3 and 11) were photographed and scanned every other day until they spoiled; the other 11 were "
    "photographed daily and scanned once, on the day they were cut open. Each scan turns the fruit and records six "
    "views, 60° apart (0°, 60°, 120°, 180°, 240° and 300°)."
)
para(
    "in this dataset each visit's ripeness rating was set by the study day, not by looking at the fruit. If the "
    "model is given those ratings, it can read the answer off the calendar and never needs the scanner, so in this "
    "study the ratings were used only to grade the model, never as an input. Each model was trained with one of the "
    "four followed avocados held out and then scored on it, four times over, and every run was repeated 3 times.",
    lead="A change from the earlier results: ",
)

doc.add_heading("Main findings", level=1)
for lead, txt in [
    (
        "Three of the six views work better: ",
        "the 0°, 120° and 240° views predicted better than the 60°, 180° and 300° views in 14 of 16 comparisons "
        "(with and without the photo, before and after the correction step, on the four followed avocados and on all "
        "15). No single view stands out within each group. [Chart 1]",
    ),
    (
        "Best combination: ",
        f"the photo plus the average of the 0°, 120° and 240° views was off by {v('photo_mw_0+120+240', 'mae')} days "
        f"on average, against {v('photo_mw_all', 'mae')} using all six views (as presented) and "
        f"{v('time', 'mae')} with no sensor at all. [Table 1]",
    ),
    (
        "The gains are small: ",
        "all four followed avocados spoiled between day 7 and day 9, so knowing the date alone already gets within about "
        "a day. After the correction step, the photo alone does as well as any microwave setup.",
    ),
    (
        "Teaching did not help: ",
        "the photo + microwave teacher was no better than the photo alone, so it had nothing to pass on. None of the "
        "five ways of teaching a photo-only model reliably beat a photo-only model trained on its own. [Chart 2]",
    ),
]:
    para(txt, lead=lead, style="List Bullet", after=2)

doc.add_heading("Results", level=1)
para("Table 1", after=2)
cols = ["time", "photo", "mw_0+120+240", "mw_60+180+300", "photo_mw_0+120+240", "photo_mw_all"]
rows = [
    ("Forecast score", "How much probability the model gave to the day the avocado really spoiled. Lower is "
     "better.", [v(c, "log") for c in cols]),
    ("Days off", "Average gap between the predicted and real spoilage day.", [v(c, "mae") for c in cols]),
    ("Forecast score, corrected", "The same score after a correction step that makes the stated ranges honest, "
     "learned from the other held-out avocados.", [v(c, "log", True) for c in cols]),
    ("80 % range holds the real day", "After correction. The model gives a range it is 80 % sure of; about 80 % "
     "is right.", [v(c, "cov", True, "{:.0%}") for c in cols]),
]  # fmt: skip
hdr = ["What we measure", "What it means", "No sensor (date only)", "Photo only", "Microwave, views 0/120/240",
       "Microwave, views 60/180/300", "Photo + views 0/120/240", "Photo + all six views"]  # fmt: skip
widths = [0.95, 1.6, 0.62, 0.55, 0.78, 0.78, 0.78, 0.74]
t = doc.add_table(rows=len(rows) + 1, cols=len(hdr))
t.style = "Table Grid"
t.alignment = WD_TABLE_ALIGNMENT.CENTER
t.autofit = False
for j, w in enumerate(widths):
    t.columns[j].width = Inches(w)
for j, m in enumerate(hdr):
    c = t.cell(0, j)
    c.width = Inches(widths[j])
    cell_text(c, m, bold=True, center=j > 1, size=8.5)
    shade(c, "DCE6F2")
for i, (name, desc, vals) in enumerate(rows, start=1):
    for j, val in enumerate([name, desc, *vals]):
        c = t.cell(i, j)
        c.width = Inches(widths[j])
        cell_text(c, val, bold=(j == 0), center=j > 1, size=9 if j > 1 else 9)
caption(
    "The ripeness-score model (latent SDE), scored on the four followed avocados, averaged over 3 runs. Each avocado "
    "is scored by a model that never saw it in training. “Averaged” views: the model sees the average and spread "
    "of those views' readings."
)
para(
    f"The 0°, 120° and 240° views beat the date-only model (forecast score {v('mw_0+120+240', 'log')} vs "
    f"{v('time', 'log')}); the other three views add nothing ({v('mw_60+180+300', 'log')}). Adding the photo to "
    f"the better three helps a little more ({v('photo_mw_0+120+240', 'log')}). The best single view, 240°, does "
    f"about as well on its own ({v('mw_240', 'log')}). After the correction step every model's ranges are honest "
    "(about 75 % instead of 100 %), and the differences between inputs shrink to within run-to-run variation.",
    lead="Comparing the inputs. ",
    after=4,
)

doc.add_heading("Chart 1: Every view, alone and combined", level=1)
doc.add_picture(str(HERE / "a2sl_views.png"), width=Inches(6.5))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
caption(
    "Each pair of bars is one choice of scanner view: grey with the microwave alone, blue with the photo added. Bars "
    "are averages over 3 runs; black lines show how much they vary. Dashed line: a model that knows only the date. "
    "Dotted line: the photo alone. Bars below the dashed line beat the calendar. The 0°, 120° and 240° views sit "
    "lower than the 60°, 180° and 300° views; the reason (for example how the scanner's antennas are arranged) is "
    "not known."
)

doc.add_heading("Chart 2: Teaching a photo-only model", level=1)
doc.add_picture(str(HERE / "a2sl_aml.png"), width=Inches(6.3))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
caption(
    "A shopper has a phone camera, not a microwave scanner. Each blue bar is a photo-only model trained with help "
    "from a “teacher” that sees the photo and the 0°, 120° and 240° scans. The teacher shares its forecasts "
    "(CAML, a 2025 method; the paper's setting and a softer one), a summary of what it has seen, or both; SAMD (a 2023 "
    "method) also refreshes the teacher from the student as they train. The dark bar is the teacher itself. SAMD came "
    "out lowest here, but its gap from the photo-only model is smaller than how much the photo-only model varies "
    "between runs, before and after the correction step."
)
doc.save(OUT)
print(OUT)
