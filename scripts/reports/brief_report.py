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
sec.left_margin = sec.right_margin = Inches(0.8)
sec.top_margin = sec.bottom_margin = Inches(0.6)
st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(10.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after = Pt(4)
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
    s.paragraph_format.space_before = Pt(0 if name == "Title" else 10)
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


def table(hdr, rows, widths, highlight=()):
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
            if j > 0 and not hdr[j].startswith("What"):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(val))
            r.font.size = Pt(9)
            r.bold = i == 0 or (j == 0)
            if i == 0:
                shade(c, "DCE6F2")
            elif i - 1 in highlight:
                shade(c, "FFF4E0")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def picture(name, width=6.8):
    doc.add_picture(str(HERE / name), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


def f2(x):
    return "—" if x is None else f"{x:.2f}"


def f3(x):
    return "—" if x is None else f"{x:.3f}"


INPUTS = ("Photo", "Scanner only", "Photo + scanner")


def model_section(m, title, what, takeaway):
    doc.add_heading(title, level=1)
    for w in what:
        bullet(w)
    rows = [[i, f2(M[f"{m}|{i}"]["log"]), f2(M[f"{m}|{i}"]["mae"]), f3(M[f"{m}|{i}"]["mango"])] for i in INPUTS]
    table(["Input", "Avocado score", "Avocado day error", "Mango score"], rows, [1.9, 1.3, 1.5, 1.3])
    para(takeaway)


# =====================================================================================================================
doc.add_paragraph("Spoilage Forecasts: Results", style="Title")
para("Avocados and mangoes · November 2026", color=GREY, size=9.5, after=6)
para(
    "All scores: lower is better. Avocado score = forecast score on the 4 tracked avocados; day error in days. "
    "Mango score = spoilage score on 48 fruit. Scanner = the microwave scanner's 0°, 120° and 240° rotation views.",
    size=9,
    color=GREY,
)

ode, sde = (lambda i, k="log": M[f"latent_ode|{i}"][k]), (lambda i, k="log": M[f"latent_sde|{i}"][k])
model_section(
    "latent_ode",
    "Latent ODE (ordinary differential equation)",
    [
        "Learns a smooth, hidden ripeness curve for each fruit. Each new photo or scan moves the curve; the forecast "
        "is when the curve crosses the spoiled line."
    ],  # fmt: skip
    f"Best model. Photo + scanner is best on avocado ({f2(ode('Photo + scanner'))}); photo alone "
    f"({f2(ode('Photo'))}) and scanner alone ({f2(ode('Scanner only'))}) are close. On mango, scanner only "
    f"({f3(ode('Scanner only', 'mango'))}) clearly beats photo ({f3(ode('Photo', 'mango'))}).",
)
model_section(
    "latent_sde",
    "Latent SDE (stochastic differential equation)",
    [
        "The same hidden ripeness curve, plus random day-to-day drift. It forecasts a range of spoilage days rather "
        "than following a single path."
    ],  # fmt: skip
    f"Behind the ODE on avocado ({f2(sde('Photo + scanner'))} vs {f2(ode('Photo + scanner'))}). Scanner only "
    f"({f2(sde('Scanner only'))}) edges photo only ({f2(sde('Photo'))}); on mango it is much better "
    f"({f3(sde('Scanner only', 'mango'))} vs {f3(sde('Photo', 'mango'))}).",
)
picture("br_models.png", 6.6)

# ----------------------------------------------------------------------------------------------------- AML
doc.add_page_break()
doc.add_heading("Teaching a photo-only model (AML)", level=1)
para(
    "Auxiliary modality learning (AML): a teacher model sees photo + scanner during training; a student sees photos "
    "only, so it can run without a scanner. Methods ported from the lab's RepDistiller code.",
)
DESC = [
    ("photo", "Photo only", "the student trained on its own, with no teacher. The baseline."),
    ("reset_only_photoT", "Take-turns, no teacher",
     "the student trains in rounds against a copy of itself, with no scanner anywhere. Tests whether the training "
     "schedule alone explains the gains."),
    ("kd", "CAML", "Collaborative AML. A fixed, fully trained teacher; the student copies the teacher's forecast "
     "(knowledge distillation, KD)."),
    ("samd", "SAMD", "Specialized auxiliary modality distillation. The teacher is rebuilt from the student each round "
     "and the two share a final layer; the student matches the teacher's internal features."),
    ("ours", "Lab method", "the lab's own script: SAMD's rounds, but the student copies the teacher's forecast (KD) "
     "instead of its features."),
    ("ours_rkd", "Lab method + RKD", "relational knowledge distillation. SAMD's rounds; the student copies how far "
     "apart the teacher places each pair of fruit."),
    ("teacher", "Teacher", "the photo + scanner model itself. The ceiling; needs the scanner at test time."),
]  # fmt: skip
for _, lab_, d in DESC:
    bullet(" " + d, lead=lab_ + ":")
para(
    "Avocado: 5 repeats in a separate run from the tables above, so compare rows within this table only.",
    size=8.5,
    color=GREY,
    italic=True,
)


def a(m, k, f="log"):
    r = A.get(f"{m}|{k}")
    if r is None:
        return None
    return r[f][0] if f == "log" else r[f]


rows = [[lab_, f2(a("latent_ode", k)), f2(a("latent_ode", k, "mae")), f2(a("latent_sde", k)),
         f3(A.get(f"latent_ode|{k}", {}).get("mango"))] for k, lab_, _ in DESC]  # fmt: skip
table(["Method", "ODE avocado score", "ODE day error", "SDE avocado score", "Mango score (ODE)"], rows,
      [1.9, 1.2, 1.1, 1.2, 1.2], highlight=(1,))  # fmt: skip
picture("br_aml.png", 6.4)
taught = [k for k, _, _ in DESC if k not in ("photo", "reset_only_photoT", "teacher") and a("latent_ode", k)]
best = min(taught, key=lambda k: a("latent_ode", k))
lab = {k: x for k, x, _ in DESC}
bullet(
    f" the scanner barely helps even the teacher ({f2(a('latent_ode', 'teacher'))} vs photo only "
    f"{f2(a('latent_ode', 'photo'))}), so there is little for a student to learn. CAML, with no training rounds, "
    f"matches photo only ({f2(a('latent_ode', 'kd'))}).",
    lead="Avocado:",
)
bullet(
    f" every gain comes from the training rounds, not the scanner: {lab[best]} ({f2(a('latent_ode', best))}) is "
    f"level with take-turns and no teacher ({f2(a('latent_ode', 'reset_only_photoT'))}).",
    lead="Rounds, not teaching:",
)
bullet(
    f" lower score, but day error rises ({f2(a('latent_ode', 'photo', 'mae'))} → "
    f"{f2(a('latent_ode', best, 'mae'))} d): forecasts get less over-confident, not more accurate.",
    lead="Trade-off:",
)
bullet(
    f" no student beats photo only ({f3(A['latent_ode|photo']['mango'])}); the teacher "
    f"({f3(A['latent_ode|teacher']['mango'])}) does.",
    lead="Mango:",
)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print("wrote", OUT)
