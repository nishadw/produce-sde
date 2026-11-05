from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

FIG = "/fs/class-projects/fall2026/cmsc498z/nwajge/ripen/report/figures/e3_transfer.png"
OUT = "/fs/class-projects/fall2026/cmsc498z/nwajge/scratch/brief/Produce_Spoilage_SDE_Results.docx"
GREY = RGBColor(0x55, 0x5B, 0x65)

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for side in ("left_margin", "right_margin"):
    setattr(sec, side, Inches(0.9))
sec.top_margin = sec.bottom_margin = Inches(0.6)
st = doc.styles["Normal"]
st.font.name = "Calibri"; st.font.size = Pt(10.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after = Pt(4)
for name, size in (("Title", 18), ("Heading 1", 12.5)):
    s = doc.styles[name]; s.font.name = "Calibri"; s.font.size = Pt(size); s.font.bold = True
    s.font.color.rgb = RGBColor(0x16, 0x19, 0x1E)
    s.element.rPr.rFonts.set(qn("w:asciiTheme"), "") if False else None
    rf = s.element.rPr.rFonts; rf.set(qn("w:ascii"), "Calibri"); rf.set(qn("w:hAnsi"), "Calibri")
    for a in ("w:asciiTheme", "w:hAnsiTheme"):
        if rf.get(qn(a)) is not None: del rf.attrib[qn(a)]
    s.paragraph_format.space_before = Pt(10 if name != "Title" else 0); s.paragraph_format.space_after = Pt(3)
    s.paragraph_format.keep_with_next = True

def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), hexcolor); tcPr.append(sh)

def para(text="", bold_lead=None, size=None, color=None, italic=False, after=None, style=None):
    p = doc.add_paragraph(style=style)
    if bold_lead:
        r = p.add_run(bold_lead); r.bold = True
        if size: r.font.size = Pt(size)
    r = p.add_run(text); r.italic = italic
    if size: r.font.size = Pt(size)
    if color: r.font.color.rgb = color
    if after is not None: p.paragraph_format.space_after = Pt(after)
    return p

def table(rows, widths, header_fill="DCE6F2", bold_col0=False, highlight_rows=()):
    t = doc.add_table(rows=len(rows), cols=len(rows[0])); t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
    for j, w in enumerate(widths): t.columns[j].width = Inches(w)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j); c.width = Inches(widths[j]); c.text = ""
            p = c.paragraphs[0]; p.paragraph_format.space_after = Pt(0)
            r = p.add_run(val); r.font.size = Pt(9.5)
            if i == 0: r.bold = True; shade(c, header_fill)
            elif i in highlight_rows: shade(c, "F2F6FB")
            if j > 0: p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if bold_col0 and j == 0 and i > 0: r.bold = True
    return t

doc.add_paragraph("Predicting Produce Spoilage with a Latent Neural SDE", style="Title")
para("Nishad Wajge · CMSC 498Z · Results summary, 5 November 2026 · github.com/nishadw/produce-sde", color=GREY, size=9.5, after=8)

para("From irregular daily photos of a fruit, the model predicts a probability distribution over the day it will spoil. "
     "Ripeness is a hidden random process (a neural stochastic differential equation); the fruit spoils when that process first "
     "crosses a learned threshold. Storage temperature speeds the process up or slows it down through a physical (Arrhenius) law. "
     "Results are for avocados only. The main model below is the latent SDE with the guided posterior, ResNet-18 photo features and post-hoc recalibration.")

doc.add_heading("Key results", level=1)
bul = [
    ("Accurate and calibrated on avocados. ", "Average error is 1.2 days, and its 80 % prediction intervals contain the true spoilage day 93 % of the time. Accuracy ties the best black-box sequence models (GRU, ODE-RNN) and is the most stable across training runs."),
    ("Transfers to an unseen storage temperature. ", "Trained on two temperatures and tested on the third, the physical clock cuts the error 1.5–2.4× compared with the same model using a learned or fixed clock. It wins in all 15 seed × temperature pairs."),
    ("Robust to fewer photos. ", "With a photo only every 3rd day, error barely changes, while the strongest irregular-time baseline (GRU-D) roughly doubles its error."),
]
for lead, txt in bul:
    p = para(txt, bold_lead=lead, style="List Bullet", after=2)

doc.add_heading("Main model results", level=1)
rows = [
    ["Metric", "Main avocado photos\n(Mendeley, 478 fruit)", "Microwave-scan avocados\n(A2SL, 4 fruit tracked)"],
    ["Average error, MAE (days)", "1.2", "1.0"],
    ["Distribution error, CRPS (days, lower is better)", "0.85", "1.20"],
    ["Ranking accuracy, C-index (1 = perfect, 0.5 = chance)", "0.93", "0.78"],
    ["80 % interval coverage (target ≈ 0.80–0.88)", "0.93", "0.89"],
    ["Calibration error, ECE (lower is better)", "0.02", "0.16"],
    ["Error on an unseen storage temperature, MAE (days)", "1.1–5.5", "—"],
    ["Best other model, C-index", "0.93 (ODE-RNN)", "0.84 (FPT filter, ours)"],
]
t = table([[c.replace("\n", " ") if j == 0 else c for j, c in enumerate(r)] for r in rows], [3.3, 1.7, 1.7])
for j in (1, 2):  # split header lines into two paragraphs
    c = t.cell(0, j); first, second = rows[0][j].split("\n"); c.text = ""
    p = c.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(0)
    r = p.add_run(first); r.bold = True; r.font.size = Pt(9.5)
    p2 = c.add_paragraph(); p2.alignment = WD_ALIGN_PARAGRAPH.CENTER; p2.paragraph_format.space_after = Pt(0)
    r2 = p2.add_run(second); r2.font.size = Pt(8.5); r2.font.color.rgb = GREY
para("Main avocado photos: mean of 5 training seeds on test fruit never seen in training. Unseen temperature: range over the "
     "three held-out storage conditions (20 °C 1.1, ambient 1.4, 10 °C 5.5 days). Microwave-scan avocados: one training run "
     "of the earlier (unguided) model, scored on only 9 test cases, so treat it as a demonstration that the pipeline works "
     "with a second sensor, not as a precise estimate.",
     size=8.5, color=GREY, italic=True, after=6)

doc.add_heading("Main result: the physical clock enables transfer", level=1)
doc.add_picture(FIG, width=Inches(4.6))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
para("Error on a storage temperature held out of training (latent SDE, 5 seeds, dots = individual seeds). Only the clock changes "
     "between bars; the Arrhenius clock is lowest in every case.", size=8.5, color=GREY, italic=True, after=6)

doc.save(OUT); print(OUT)
