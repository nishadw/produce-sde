"""Short presentation brief: one section per model (what it is, then its scores) (report env).

Run brief_figs.py first (ripen env; writes br_*.png and brief_numbers.json). Writes results/2026-11-13_results-brief-ode-sde-aml.docx.
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
OUT = REPO / "results" / "2026-11-13_results-brief-ode-sde-aml.docx"
N = json.loads((HERE / "brief_numbers.json").read_text())
M, A = N["models"], N["aml"]
GREY = RGBColor(0x55, 0x5B, 0x65)

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.left_margin = sec.right_margin = Inches(0.7)
sec.top_margin = sec.bottom_margin = Inches(0.55)
st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(10)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after = Pt(3)
for name, size in (("Title", 20), ("Heading 1", 13), ("Heading 2", 11)):
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
    s.paragraph_format.space_before = Pt(0 if name == "Title" else 8)
    s.paragraph_format.space_after = Pt(3)
    s.paragraph_format.keep_with_next = True


def shade(cell, fill):
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(sh)


def para(text="", lead=None, size=None, color=None, italic=False, after=None):
    p = doc.add_paragraph()
    if lead:
        p.add_run(lead).bold = True
    r = p.add_run(text)
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    if color:
        r.font.color.rgb = color
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def bullet(text, lead=None):
    p = doc.add_paragraph(style="List Bullet")
    if lead:
        p.add_run(lead).bold = True
    p.add_run(text)
    p.paragraph_format.space_after = Pt(2)


def caption(text):
    para(text, size=8.5, color=GREY, italic=True, after=8)


def table(hdr, rows, widths, highlight=(), left=(0,), size=8.5):
    t = doc.add_table(rows=len(rows) + 1, cols=len(hdr))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for j, w in enumerate(widths):
        t.columns[j].width = Inches(w)
    for i, row in enumerate([hdr, *rows]):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.width = Inches(widths[j])
            c.text = ""
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            if j not in left:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(val))
            r.font.size = Pt(size)
            r.bold = i == 0 or j == 0
            if i == 0:
                shade(c, "DCE6F2")
            elif i - 1 in highlight:
                shade(c, "FFF4E0")
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return t


def picture(name, width=6.8):
    doc.add_picture(str(HERE / name), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.space_after = Pt(2)


def f2(x):
    return "—" if x is None else f"{x:.2f}"


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def a(m, k, f="log"):
    r = A.get(f"{m}|{k}")
    if r is None:
        return None
    return r[f][0] if f == "log" else r[f]


ode, sde = (lambda i, k="log": M[f"latent_ode|{i}"][k]), (lambda i, k="log": M[f"latent_sde|{i}"][k])
INPUTS = ("Photo", "Scanner only", "Photo + scanner")

# =====================================================================================================================
doc.add_paragraph("Spoilage Forecasts: Models, Methods and Results", style="Title")
para("Avocados and mangoes · November 2026", color=GREY, size=9, after=4)
para(
    "Goal: from a few photos and/or microwave scans of a fruit, forecast the probability it has spoiled by each future "
    "day. Lower is better for every score. Avocado score = log score (how little probability the forecast gave to "
    "the day the fruit actually spoiled), on the 4 lab avocados tracked to spoilage, each held out in turn. Day error "
    "= days between the forecast's median and the true spoilage day. Mango score = Brier score of the forecast "
    "fraction spoiled against what the 48-fruit batch actually showed. Scanner = the microwave scanner's 0°, 120° and "
    "240° rotation views.",
    size=8.5,
    color=GREY,
)

# ----------------------------------------------------------------------------------------------------- models
doc.add_heading("1. The forecasting models", level=1)
table(
    ["Model", "How it works", "What it outputs"],
    [
        ["Latent ODE (ordinary differential equation)",
         "An encoder reads the photos/scans so far and estimates the fruit's hidden ripeness, which then rises along "
         "a smooth learned curve (no randomness). Spoilage = the curve crossing a learned line.",
         "One path; the spread comes only from uncertainty about where the fruit started"],
        ["Latent SDE (stochastic differential equation)",
         "The same hidden ripeness curve plus random day-to-day drift (a learned noise term). Spoilage is the first "
         "time the drifting path crosses the line; its probability is computed exactly by solving a PDE.",
         "A spread of paths; wider, more honest ranges"],
    ],
    [1.55, 4.0, 1.55],
    left=(0, 1, 2),
)  # fmt: skip
rows = [[i, f2(ode(i)), f2(ode(i, "mae")), f3(ode(i, "mango")), f2(sde(i)), f2(sde(i, "mae")), f3(sde(i, "mango"))]
        for i in INPUTS]  # fmt: skip
table(["Input", "ODE avocado score", "ODE day error", "ODE mango score", "SDE avocado score", "SDE day error",
       "SDE mango score"], rows, [1.25, 1.0, 0.9, 1.0, 1.0, 0.9, 1.0])  # fmt: skip
picture("br_models.png", 5.6)
bullet(
    f" photo + scanner is best on avocado ({f2(ode('Photo + scanner'))}), but photo alone ({f2(ode('Photo'))}) is close.",
    lead="Latent ODE is the better model on both fruits:",
)
bullet(
    f" scanner only beats photo for both models (ODE {f3(ode('Scanner only', 'mango'))} vs "
    f"{f3(ode('Photo', 'mango'))}; SDE {f3(sde('Scanner only', 'mango'))} vs {f3(sde('Photo', 'mango'))}).",
    lead="On mango the scanner matters:",
)
bullet(
    " every mango was cut open once and only 9 of 48 spoiled, so no fruit has an observed spoilage day.",
    lead="No mango day error:",
)

# ----------------------------------------------------------------------------------------------------- AML
doc.add_page_break()
doc.add_heading("2. Teaching a photo-only model with the scanner (auxiliary modality learning, AML)", level=1)
para(
    "The scanner is not available everywhere, so the aim is a student model that needs only photos at test time but "
    "learned, during training, from a teacher that also saw the scans. The methods differ in three choices: is the "
    "teacher fixed or rebuilt each round, does the student share the teacher's final layer, and what the student "
    "copies. All are ported from the lab's RepDistiller code (train_student.py, distiller_zoo/).",
    size=9.5,
)
DESC = [
    ("photo", "Photo only", "None", "No", "Nothing (trained alone)", "Baseline"),
    ("reset_only_photoT", "Take-turns control", "Photos only, rebuilt each round", "Yes", "Nothing",
     "Isolates the schedule: no scanner anywhere"),
    ("kd", "CAML-style KD", "Photo + scanner, trained once, then fixed", "No", "Teacher's forecast (KD)",
     "CAML's distillation loss, fixed teacher"),
    ("samd", "SAMD", "Photo + scanner, rebuilt from student each round", "Yes", "Teacher's internal features",
     "Shen et al., ICML 2023"),
    ("ours", "Lab method", "Same as SAMD", "Yes", "Teacher's forecast (KD)", "run_our_steering_distill.sh"),
    ("ours_rkd", "Lab method + RKD", "Same as SAMD", "Yes",
     "How the teacher arranges fruit: pairwise distances and triplet angles", "Park et al., CVPR 2019"),
    ("teacher", "Teacher", "—", "—", "— (it sees the scanner at test time)", "Reference; needs the scanner"),
]  # fmt: skip
table(["Method", "Teacher", "Shared final layer", "What the student copies", "Source / note"],
      [[lab, t, sh, c, n] for _, lab, t, sh, c, n in DESC], [1.25, 1.85, 0.7, 1.85, 1.45], highlight=(1,),
      left=(0, 1, 3, 4), size=8)  # fmt: skip
para(
    "Rebuilt each round (SAMD, lab methods, take-turns control): copy the student's weights into the teacher (scanner "
    "inputs start at zero), train the teacher a few epochs with the scans, hand its final layer back to the student, "
    "then train the student a few epochs. Knowledge distillation (KD) = 0.1 × the student's own loss + 0.9 × match "
    "to the teacher's softened forecast (temperature 4), as in the lab scripts. Relational knowledge distillation "
    "(RKD) matches distances and angles between fruit rather than the fruit themselves.",
    size=8.5,
    color=GREY,
)
rows = [[lab, f2(a("latent_ode", k)), f2(a("latent_ode", k, "mae")), f2(a("latent_sde", k)),
         f3(A.get(f"latent_ode|{k}", {}).get("mango"))] for k, lab, *_ in DESC]  # fmt: skip
table(["Method", "ODE avocado score", "ODE day error", "SDE avocado score", "Mango score (ODE)"], rows,
      [1.6, 1.25, 1.1, 1.25, 1.25], highlight=(1,))  # fmt: skip
para(
    "Avocado: 5 repeats × 4 held-out fruit, separate run from section 1 (compare rows within this table). Mango: 3 "
    "repeats; the take-turns control was not run on mango.",
    size=8,
    color=GREY,
    italic=True,
)
picture("br_aml.png", 5.6)
taught = ("kd", "samd", "ours", "ours_rkd")
best = min(taught, key=lambda k: a("latent_ode", k))
lab = {k: x for k, x, *_ in DESC}
bullet(
    f" the teacher, with the scanner, is barely better than photo only ({f2(a('latent_ode', 'teacher'))} vs "
    f"{f2(a('latent_ode', 'photo'))}), so there is little to transfer. CAML-style KD, with a fixed teacher, matches "
    f"photo only ({f2(a('latent_ode', 'kd'))}).",
    lead="Little for the student to learn:",
)
bullet(
    f" {lab[best]} ({f2(a('latent_ode', best))}) ties the take-turns control ({f2(a('latent_ode', 'reset_only_photoT'))}), "
    "which never sees the scanner. The gain is the rebuild-and-share-final-layer schedule, not the scans.",
    lead="Gains come from the training rounds:",
)
bullet(
    f" the lower score comes with a larger day error ({f2(a('latent_ode', 'photo', 'mae'))} → "
    f"{f2(a('latent_ode', best, 'mae'))} d): forecasts become less over-confident, not more accurate.",
    lead="Trade-off:",
)
bullet(
    f" no student beats photo only ({f3(A['latent_ode|photo']['mango'])}); the teacher does "
    f"({f3(A['latent_ode|teacher']['mango'])}).",
    lead="Mango:",
)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print("wrote", OUT)
