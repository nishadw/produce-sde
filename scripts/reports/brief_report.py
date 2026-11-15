"""Professor brief: which scanner view predicts best, and AML for a photo-only model (report env).

Run brief_figs.py first (ripen env; writes br_*.png and brief_numbers.json to outputs/report_figures). Writes
2026-11-13_results-brief-ode-sde-aml.docx to ../results (``_paths.RESULTS``, outside the repository).
"""

import json
from pathlib import Path

from _paths import RESULTS

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parents[2] / "outputs" / "report_figures"  # charts + numbers live here
HERE.mkdir(parents=True, exist_ok=True)
REPO = HERE.parents[1]
OUT = RESULTS / "2026-11-13_results-brief-ode-sde-aml.docx"
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


V = json.loads((HERE / "views_aml_numbers.json").read_text())["avo"]  # views study: 3 seeds x 4 held-out fruit


def view(m, key):
    """Mean log score; '*' when the paired bootstrap against date only excludes 0."""
    r, b = V[f"{m}_{key}"], V.get(f"{m}_{key}_boot") or ""
    return f2(r["log"][0]) + (" *" if b.endswith("*") else "")


def ode_aml(k, f="log"):
    return a("latent_ode", k, f)


# =====================================================================================================================
doc.add_paragraph("Avocado Spoilage: Scanner Views and a Photo-Only Model", style="Title")
para("Lab avocados (A2SL) · November 2026", color=GREY, size=9, after=4)
para(
    "Questions asked: (1) with one time-series model per microwave view, which of the 6 views predicts spoilage best; "
    "(2) can AML / CAML teach a photo-only model from a teacher that also sees the scanner. Data: 4 avocados "
    "photographed and scanned every other day until they spoiled, each held out in turn. Score = log score (lower is "
    "better): how little probability the forecast gave to the two-day window in which the fruit spoiled. "
    "* = significantly better than the comparison (95 % paired bootstrap).",
    size=8.5,
    color=GREY,
)
doc.add_heading("In short", level=1)
bullet(" 240° for the latent SDE (the only single view that beats a date-only model significantly). For the latent ODE "
       "every view helps and all six are within 0.05 of each other, so no view stands out.", lead="Best view:")
bullet(" photo + the 0°/120°/240° views is the best input for the latent ODE, but photo alone is close behind. "
       "The latent ODE beats the latent SDE on almost every input.", lead="Best input:")
bullet(" SAMD and the lab method improve the photo-only model, but a control that never sees the scanner improves it "
       "just as much. The gain comes from the take-turns training, not from the scanner.", lead="AML:")

# ----------------------------------------------------------------------------------------------------- views
doc.add_heading("1. One model per scanner view", level=1)
para(
    "Each view is fed to its own model as a time series (every scan up to the forecast day). Two models: the latent "
    "ODE (a hidden ripeness score that rises smoothly; spoiled when it crosses a line) and the latent SDE (the same "
    "plus random day-to-day drift).",
    size=9.5,
)
ROWS = [("Date only (no images)", "time"), ("Photo only", "photo")] + [(f"Scanner {v}°", f"mw_{v}") for v in
        ("0", "60", "120", "180", "240", "300")] + [("Scanner 0° + 120° + 240°", "mw_0+120+240"),
        ("Scanner, all 6 views", "mw_all"), ("Photo + scanner 0°/120°/240°", "photo_mw_0+120+240")]  # fmt: skip
table(["Input (* = beats date only)", "Latent ODE score", "Latent SDE score"],
      [[lab, view("latent_ode", k), view("latent_sde", k)] for lab, k in ROWS], [2.6, 1.4, 1.4],
      highlight=(10,))  # fmt: skip
picture("br_views.png", 6.4)
bullet(
    f" every view beats date only ({view('latent_ode', 'time')}), but they span only "
    f"{min(V[f'latent_ode_mw_{v}']['log'][0] for v in ('0', '60', '120', '180', '240', '300')):.2f}–"
    f"{max(V[f'latent_ode_mw_{v}']['log'][0] for v in ('0', '60', '120', '180', '240', '300')):.2f}, "
    "less than the run-to-run spread.",
    lead="Latent ODE:",
)
bullet(
    f" 240° ({f2(V['latent_sde_mw_240']['log'][0])}) and 0° ({f2(V['latent_sde_mw_0']['log'][0])}) are best; "
    "60°, 180° and 300° are no better than date only. The 0°/120°/240° views beat the 60°/180°/300° views in 14 of 16 "
    "significance checks; we do not know why, so it is worth asking the lab how the views are produced.",
    lead="Latent SDE:",
)
bullet(
    f" combining all six views is no better than the best single view, and photo + scanner "
    f"({f2(V['latent_ode_photo_mw_0+120+240']['log'][0])} ODE) is only a little better than photo alone "
    f"({f2(V['latent_ode_photo']['log'][0])}). All 4 fruit spoiled on day 7–9 of a fixed schedule, so the date "
    "already pins spoilage to about a day; images can only refine that.",
    lead="Why the differences are small:",
)

# ----------------------------------------------------------------------------------------------------- AML
doc.add_heading("2. Teaching a photo-only model (AML)", level=1)
para(
    "The student sees only photos at test time; during training a teacher also sees the 0°/120°/240° scans. Latent "
    "ODE, 5 repeats × 4 held-out fruit (a separate run from section 1, so compare rows within this table).",
    size=9.5,
)
AMLROWS = [
    ("photo", "Photo only", "Student trained alone (baseline)"),
    ("kd", "CAML-style KD", "Fixed teacher; student matches its forecast (0.1 own loss + 0.9 teacher, temperature 4)"),
    ("samd", "SAMD (Shen et al. 2023)", "Teacher rebuilt from the student each round and shares its predictor; "
     "student matches the teacher's features"),
    ("ours", "Lab method", "SAMD's rounds, with CAML-style KD as the matching loss"),
    ("reset_only_photoT", "Take-turns control", "SAMD's rounds, but the teacher never sees the scanner"),
    ("teacher", "Teacher", "Photo + scanner; needs the scanner at test time"),
]  # fmt: skip
table(["Method", "What it does", "Score", "vs photo only", "Day error"],
      [[lab, d, f2(ode_aml(k)), A[f"latent_ode|{k}"]["vs_photo"] or "—", f"{ode_aml(k, 'mae'):.2f} d"]
       for k, lab, d in AMLROWS], [1.35, 2.85, 0.6, 1.3, 0.75], highlight=(4,), left=(0, 1), size=8)  # fmt: skip
picture("br_aml.png", 5.8)
bullet(
    f" with the scanner the teacher scores {f2(ode_aml('teacher'))} against {f2(ode_aml('photo'))} for photo only, "
    f"not a significant gain, so there is little to pass on. CAML-style KD matches photo only ({f2(ode_aml('kd'))}).",
    lead="The scanner adds little, even to the teacher:",
)
bullet(
    f" SAMD ({f2(ode_aml('samd'))}) and the lab method ({f2(ode_aml('ours'))}) beat photo only, but the take-turns "
    f"control ({f2(ode_aml('reset_only_photoT'))}) does as well without any scanner.",
    lead="The gain comes from the training rounds:",
)
bullet(
    f" the better score comes with a larger day error ({ode_aml('photo', 'mae'):.2f} → "
    f"{ode_aml('samd', 'mae'):.2f} d for SAMD): forecasts concentrate on the spoilage window, but their middle guess "
    "moves further from the observed day.",
    lead="Trade-off:",
)
para(
    "To tell views and methods apart reliably we need more avocados tracked to spoilage, with varied spoilage days; "
    "four fruit that all spoil on day 7–9 cannot separate differences under about 0.1.",
    lead="Next: ",
    size=9.5,
)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print("wrote", OUT)
