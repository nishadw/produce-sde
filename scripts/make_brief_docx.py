"""Plain-language one-page results summary (avocado only). Run from the scratch brief folder with python-docx."""
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

HERE = "/fs/class-projects/fall2026/cmsc498z/nwajge/scratch/brief/"
FIG, OUT = HERE + "plain_transfer.png", HERE + "Produce_Spoilage_SDE_Results.docx"
GREY = RGBColor(0x55, 0x5B, 0x65)

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.left_margin = sec.right_margin = Inches(0.85)
sec.top_margin = sec.bottom_margin = Inches(0.6)
st = doc.styles["Normal"]
st.font.name = "Calibri"; st.font.size = Pt(10.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after = Pt(4)
for name, size in (("Title", 19), ("Heading 1", 12.5)):
    s = doc.styles[name]; s.font.name = "Calibri"; s.font.size = Pt(size); s.font.bold = True
    s.font.color.rgb = RGBColor(0x16, 0x19, 0x1E)
    rf = s.element.rPr.rFonts; rf.set(qn("w:ascii"), "Calibri"); rf.set(qn("w:hAnsi"), "Calibri")
    for a in ("w:asciiTheme", "w:hAnsiTheme"):
        if rf.get(qn(a)) is not None: del rf.attrib[qn(a)]
    s.paragraph_format.space_before = Pt(0 if name == "Title" else 9); s.paragraph_format.space_after = Pt(3)
    s.paragraph_format.keep_with_next = True

def shade(cell, fill):
    sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(sh)

def para(text="", lead=None, size=None, color=None, italic=False, after=None, style=None):
    p = doc.add_paragraph(style=style)
    if lead:
        r = p.add_run(lead); r.bold = True
    r = p.add_run(text); r.italic = italic
    if size: r.font.size = Pt(size)
    if color: r.font.color.rgb = color
    if after is not None: p.paragraph_format.space_after = Pt(after)
    return p

def cell_text(c, main, sub=None, bold=False, center=False, size=9.5):
    c.text = ""; p = c.paragraphs[0]; p.paragraph_format.space_after = Pt(0)
    if center: p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(main); r.bold = bold; r.font.size = Pt(size)
    if sub:
        p2 = c.add_paragraph(); p2.paragraph_format.space_after = Pt(0)
        if center: p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r2 = p2.add_run(sub); r2.font.size = Pt(8.5); r2.font.color.rgb = GREY

doc.add_paragraph("Predicting When Avocados Will Spoil", style="Title")
para("Nishad Wajge · CMSC 498Z · Results summary, 5 November 2026", color=GREY, size=9.5, after=6)

doc.add_heading("What we built", level=1)
para("A computer model that looks at daily photos of an avocado and predicts how many days are left until it spoils. "
     "Instead of one guess (“5 days”), it gives a range with odds (“most likely 5 days; 80 % chance it is between 4 and 7”). "
     "It also knows that fruit ripens faster when it is warmer.")
para("How it works: the model keeps a hidden “ripeness score” for each fruit. Each day the score rises a little, with some "
     "randomness, the way a stock price drifts. When the score crosses a fixed line, the fruit counts as spoiled. Each new photo "
     "updates the score. Temperature sets how fast the score rises, using a standard chemistry rule (the Arrhenius equation: "
     "reactions run faster when warm). The technical name for this kind of model is a latent stochastic differential equation "
     "(“latent” = hidden; “stochastic” = includes randomness).", after=2)

doc.add_heading("Main findings", level=1)
for lead, txt in [
    ("Accurate: ", "on average the prediction is 1.2 days off, and the true spoilage day lands inside the model's “80 % likely” range 93 % of the time. That matches the best standard AI models we compared against."),
    ("Works at new temperatures: ", "when trained on two storage temperatures and tested on a third it never saw, the temperature rule cuts the error by 1.5 to 2.4 times (figure below)."),
    ("Needs fewer photos: ", "with a photo only every third day its accuracy barely changes, while the best standard model's error roughly doubles."),
    ("Weak spot: ", "for fruit stored colder than anything it trained on, it is about 5 days off and too confident in its range."),
]:
    para(txt, lead=lead, style="List Bullet", after=2)

doc.add_heading("Results", level=1)
rows = [
    ("Average error", "How many days off the prediction is, on average. Lower is better.", "1.2 days", "1.0 days"),
    ("Range hit rate", "How often the real spoilage day falls inside the model's “80 % likely” range. About 80–90 % means the ranges are honest.", "93 %", "89 %"),
    ("Ordering accuracy", "Given two fruits, how often the model correctly says which spoils first. 50 % is a coin flip; 100 % is perfect.", "93 %", "78 %"),
]
t = doc.add_table(rows=len(rows) + 1, cols=4); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
widths = [1.45, 3.35, 1.0, 1.0]
for j, w in enumerate(widths): t.columns[j].width = Inches(w)
hdr = [("What we measure", None), ("What it means", None), ("Main avocado photos", "478 fruit"), ("Microwave-scan avocados", "4 fruit")]
for j, (m, s) in enumerate(hdr):
    c = t.cell(0, j); c.width = Inches(widths[j]); cell_text(c, m, s, bold=True, center=j > 1); shade(c, "DCE6F2")
for i, row in enumerate(rows, start=1):
    for j, v in enumerate(row):
        c = t.cell(i, j); c.width = Inches(widths[j]); cell_text(c, v, bold=(j == 0), center=j > 1)
para("Main avocado photos: a public dataset of 478 avocados photographed every day at three storage temperatures "
     "(10 °C, 20 °C and room temperature); results are averaged over 5 separate training runs, scored on fruit the model never saw. "
     "Microwave-scan avocados: a small dataset from a University of Alabama lab that scanned avocados with microwave sensors "
     "as well as cameras. Only 4 fruit were followed until they spoiled, so these numbers are rough.",
     size=8.5, color=GREY, italic=True, after=4)

doc.add_heading("Why the temperature rule matters", level=1)
doc.add_picture(FIG, width=Inches(6.2))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
para("Each group of bars hides one storage temperature during training, then tests on it. The blue bar is our model; the grey "
     "bars are the same model without the temperature rule. Lower is better. (Results from the first round of experiments, 5 training runs each.)",
     size=8.5, color=GREY, italic=True, after=0)
doc.save(OUT); print(OUT)
