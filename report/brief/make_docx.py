"""Plain-language one-page results summary (avocado only). Run from the scratch brief folder with python-docx."""

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CMP = str(HERE / "compare.png")
HELD = str(HERE / "heldout.png")
FIG, OUT = str(HERE / "transfer.png"), str(REPO / "report" / "Produce_Spoilage_SDE_Results.docx")
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


def cell_text(c, main, sub=None, bold=False, center=False, size=9.5):
    c.text = ""
    p = c.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(main)
    r.bold = bold
    r.font.size = Pt(size)
    if sub:
        p2 = c.add_paragraph()
        p2.paragraph_format.space_after = Pt(0)
        if center:
            p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r2 = p2.add_run(sub)
        r2.font.size = Pt(8.5)
        r2.font.color.rgb = GREY


doc.add_paragraph("Predicting When Avocados Will Spoil", style="Title")
para("Nishad Wajge · CMSC 498Z · Results summary, 5 November 2026", color=GREY, size=9.5, after=6)

doc.add_heading("What we built", level=1)
para(
    "A computer model that looks at daily photos of an avocado and predicts how many days are left until it spoils. "
    "Instead of one guess (“5 days”), it gives a range with odds (“most likely 5 days; 80 % chance it is between 4 and 7”). "
    "It also knows that fruit ripens faster when it is warmer."
)
para(
    "How it works: the model keeps a hidden “ripeness score” for each fruit. Each day the score rises a little, with some "
    "randomness, the way a stock price drifts. When the score crosses a fixed line, the fruit counts as spoiled. Each new photo "
    "updates the score. Temperature sets how fast the score rises, using a standard chemistry rule (the Arrhenius equation: "
    "reactions run faster when warm). The technical name for this kind of model is a latent stochastic differential equation "
    "(“latent” = hidden; “stochastic” = includes randomness).",
    after=2,
)

doc.add_heading("Main findings", level=1)
for lead, txt in [
    (
        "Accurate: ",
        "on average the prediction is 1.2 days off, and the true spoilage day lands inside the model's “80 % likely” range 93 % of the time. It ties three standard AI models and clearly beats a neural CDE and three classic statistics models. One model, a transformer, is a little better here (1.0 days off); the gap is small but real.",
    ),
    (
        "Works at new temperatures: ",
        "when trained on two storage temperatures and tested on a third it never saw, the temperature rule cuts the error by 1.5 to 3 times (figure below).",
    ),
    (
        "Needs fewer photos: ",
        "with a photo only every third day it is 1.4 days off, against 1.9 to 2.6 days for every other model, including the transformer. With fewer photos it beats every model we tested, and by more than chance could explain.",
    ),
    (
        "Weak spot: ",
        "for fruit stored colder than anything it trained on, it is about 5 days off and too confident in its range. A simple classic statistics model (the Wiener process, below) does better there, at about 3 days off.",
    ),
]:
    para(txt, lead=lead, style="List Bullet", after=2)

doc.add_heading("Results", level=1)
rows = [
    ("Average error", "How many days off the prediction is, on average. Lower is better.", "1.2 days", "1.5 days"),
    (
        "Range hit rate",
        "How often the real spoilage day falls inside the model's “80 % likely” range. About 80–90 % means the ranges are honest.",
        "93 %",
        "89 %",
    ),
    (
        "Ordering accuracy",
        "Given two fruits, how often the model correctly says which spoils first. 50 % is a coin flip; 100 % is perfect.",
        "93 %",
        "77 %",
    ),
    (
        "Range width",
        "How wide the \u201c80 % likely\u201d range is. Narrower is more useful, as long as the hit rate stays high.",
        "4.3 days",
        "5.0 days",
    ),
    (
        "Overall forecast score (CRPS)",
        "One number that scores the whole range of predictions at once, in days. It rewards being right and being confident in the right place. Lower is better.",
        "0.85",
        "1.29",
    ),
]
t = doc.add_table(rows=len(rows) + 1, cols=4)
t.style = "Table Grid"
t.alignment = WD_TABLE_ALIGNMENT.CENTER
t.autofit = False
widths = [1.45, 3.35, 1.0, 1.0]
for j, w in enumerate(widths):
    t.columns[j].width = Inches(w)
hdr = [
    ("What we measure", None),
    ("What it means", None),
    ("Main avocado photos", "478 fruit"),
    ("Microwave-scan avocados", "4 fruit"),
]
for j, (m, s) in enumerate(hdr):
    c = t.cell(0, j)
    c.width = Inches(widths[j])
    cell_text(c, m, s, bold=True, center=j > 1)
    shade(c, "DCE6F2")
for i, row in enumerate(rows, start=1):
    for j, v in enumerate(row):
        c = t.cell(i, j)
        c.width = Inches(widths[j])
        cell_text(c, v, bold=(j == 0), center=j > 1)
para(
    "Main avocado photos: a public dataset of 478 avocados photographed every day at three storage temperatures "
    "(10 °C, 20 °C and room temperature); results are averaged over 5 separate training runs, scored on fruit the model never saw. "
    "Microwave-scan avocados: a small dataset from a University of Alabama lab that scanned avocados with microwave sensors "
    "as well as cameras. Only 4 fruit were followed until they spoiled, so these numbers are rough. Main-photo ranges were fine-tuned on fruit held back from training; the microwave set is too small for that, so its numbers are untuned.",
    size=8.5,
    color=GREY,
    italic=True,
    after=4,
)

para(
    "With the same model and 5 training runs on each, the main photos give better predictions on every measure: "
    "1.2 vs 1.5 days average error, 93 % vs 77 % ordering accuracy, narrower ranges (4.3 vs 5.0 days) and a better overall "
    "score (0.85 vs 1.29). The microwave numbers come from only 9 predictions on 3 test fruit, so they are rough. "
    "Two things are encouraging: the same model works with a completely different sensor, and its range hit rate (89 %) "
    "stays honest. There is not yet enough microwave data to say whether the sensor itself is better or worse.",
    lead="Comparing the two datasets. ",
    after=4,
)

doc.add_heading("How it compares with other AI models", level=1)
doc.add_picture(CMP, width=Inches(6.3))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
para(
    "Main avocado photos. GRU, GRU-D, ODE-RNN, the transformer and the neural CDE (controlled differential equation) are AI "
    "models built for data that arrives over time. The Wiener and gamma processes are classic statistics models of wear that "
    "build up until a breaking point. Landmark Cox is the standard survival model from medicine. \u201cOur model without randomness\u201d "
    "is the same model with the random part switched off. Bars are averages over 5 training runs; the black lines show how much "
    "they vary between runs. Lower average error is better; the green band is the ideal hit rate. The transformer is slightly "
    "ahead; GRU, GRU-D and ODE-RNN are statistically tied with ours; the rest are clearly behind. The published method gives a "
    "single number with no range, so its \u201crange\u201d almost never contains the true day.",
    size=8.5,
    color=GREY,
    italic=True,
    after=4,
)

doc.add_heading("Why the temperature rule matters", level=1)
doc.add_picture(FIG, width=Inches(6.2))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
para(
    "Each group of bars hides one storage temperature during training, then tests on it. The blue bar is our model; the grey "
    "bars are the same model without the temperature rule. Lower is better. Averages over 5 training runs.",
    size=8.5,
    color=GREY,
    italic=True,
    after=0,
)
doc.add_heading("Against other models at a new temperature", level=1)
doc.add_picture(HELD, width=Inches(6.2))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
para(
    "Same test as above: train on two storage temperatures, test on the third. Left: average error. Right: range hit rate "
    "(about 80 % is ideal). At 20 \u00b0C and room temperature the transformer is the most accurate. In cold storage (10 \u00b0C) "
    "every AI model, ours included, is several days off and gives ranges that almost never contain the true day. The Wiener "
    "process, which only extends the fruit's own trend, holds up best there. Cold storage is our model’s main weak spot.",
    size=8.5,
    color=GREY,
    italic=True,
    after=0,
)
doc.save(OUT)
print(OUT)
