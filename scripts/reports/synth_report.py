"""Brief: half-synthetic A2SL models against the no-photo forecast (report env).

Run scripts/analysis/a2sl_pop_compare.py and synth_figs.py first. Writes 2026-11-18_half-synthetic-vs-no-photo.docx
to ../results (``_paths.RESULTS``).
"""

import json

from _paths import FIGS, REPO, RESULTS

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

S = json.loads((REPO / "outputs" / "a2sl_pop_compare" / "summary.json").read_text())
SC, PAIRS = S["scores"], {(p["a"], p["b"]): p for p in S["pairs"]}
OUT = RESULTS / "2026-11-18_half-synthetic-vs-no-photo.docx"
GREY = RGBColor(0x55, 0x5B, 0x65)
REF = SC["No-photo forecast"]["log"]

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
    s.paragraph_format.space_before = Pt(0 if name == "Title" else 8)
    s.paragraph_format.space_after = Pt(3)
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


def bullet(text, lead=None):
    p = doc.add_paragraph(style="List Bullet")
    if lead:
        p.add_run(lead).bold = True
    p.add_run(text)
    p.paragraph_format.space_after = Pt(2)


def table(hdr, rows, widths, highlight=(), size=8.5):
    t = doc.add_table(rows=len(rows) + 1, cols=len(hdr))
    t.style, t.alignment, t.autofit = "Table Grid", WD_TABLE_ALIGNMENT.CENTER, False
    for j, w in enumerate(widths):
        t.columns[j].width = Inches(w)
    for i, row in enumerate([hdr, *rows]):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.width = Inches(widths[j])
            c.text = ""
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            if j:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(val))
            r.font.size = Pt(size)
            r.bold = i == 0 or j == 0
            if i == 0:
                shade(c, "DCE6F2")
            elif i - 1 in highlight:
                shade(c, "FFF4E0")
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def picture(name, width=6.8):
    doc.add_picture(str(FIGS / name), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


def vs(a, b="No-photo forecast"):
    p = PAIRS[(a, b)]
    lo, hi = p["fruit_ci"][:2]
    sig = "yes" if lo > 0 or hi < 0 else "no"
    return f"{p['diff']:+.2f}", f"[{lo:+.2f}, {hi:+.2f}]", f"{p['wins']}/4", sig


def g(k, f="log"):
    return f"{SC[k][f]:.2f}"


# =====================================================================================================================
doc.add_paragraph("Avocado Spoilage: Does Any Model Beat the No-Photo Forecast?", style="Title")
para("Lab avocados (A2SL), half-synthetic set · 18 November 2026", color=GREY, size=9, after=4)
para(
    "Data: 9 avocados photographed on days 1–5, extended to day 15 by a linear or logistic fit to their photos. Every "
    "model is trained with random start days and scored on the real visits of the 4 fruit tracked to spoilage, each "
    "held out in turn, from every start day. Score = log score (lower is better). The no-photo forecast ignores the "
    "photo and says how many days fruit usually have left at this point; a model that cannot beat it is not using "
    "the photos usefully.",
    size=8.5,
    color=GREY,
)
doc.add_heading("In short", level=1)
bullet(f" image matching with the logistic fill ({g('Matching, photo (logistic fill)')} vs {REF:.2f}). It is "
       "better on 3 of the 4 fruit and level on the fourth.", lead="Only one model beats the no-photo forecast:")  # fmt: skip
bullet(f" the best, latent SDE + SAMD, scores {g('Latent SDE + SAMD (linear)')}, worse on all 4 fruit. SAMD does "
       f"improve the plain latent SDE ({g('Latent SDE, photo (linear)')}) on 3 of 4.",
       lead="Every latent ODE / SDE run is worse than the no-photo forecast:")  # fmt: skip
bullet(" mixing a latent model's forecast with the no-photo forecast gets steadily better the more no-photo "
       "forecast is in the mix, and is best at 100 %. The latent models carry no useful information beyond the date "
       "here, so over-confidence is not the whole problem.", lead="Blending does not rescue them:")  # fmt: skip

doc.add_heading("1. Every model against the no-photo forecast", level=1)
ROWS = [("No-photo forecast", "No-photo forecast (reference)"),
        ("Matching, photo (logistic fill)", "Image matching, logistic fill"),
        ("Matching, photo (linear fill)", "Image matching, linear fill"),
        ("Latent SDE + SAMD (linear)", "Latent SDE + SAMD"),
        ("Latent SDE, photo (linear)", "Latent SDE"),
        ("Latent SDE + CAML-KD (linear)", "Latent SDE + CAML-KD"),
        ("Weibull, photo (logistic fill)", "Weibull, logistic fill"),
        ("Latent ODE + CAML-KD (linear)", "Latent ODE + CAML-KD"),
        ("Latent ODE, photo (linear)", "Latent ODE")]  # fmt: skip
rows = []
for k, lab in ROWS:
    d, ci, w, sig = ("—",) * 4 if k == "No-photo forecast" or (k, "No-photo forecast") not in PAIRS else vs(k)
    if d == "—" and k != "No-photo forecast":
        d = f"{SC[k]['log'] - REF:+.2f}"
    rows.append([lab, g(k), d, ci, w, sig, f"{SC[k]['mae']:.2f} d"])
table(["Model (photos only at test time)", "Score", "vs no-photo", "95 % interval", "Fruit better", "Significant",
       "Day error"], rows, [2.2, 0.6, 0.85, 1.1, 0.8, 0.8, 0.75], highlight=(0, 1))  # fmt: skip
picture("sy_scores.png", 6.4)
para("Latent models use the linear fill; the logistic fill gives the same picture (within 0.07). The interval "
     "resamples the 4 fruit, so it can only exclude 0 when the fruit agree.", size=8.5, color=GREY)  # fmt: skip
bullet(" matching gives the best day error of all models (about 1.2 days off vs 1.6 for the no-photo forecast "
       "and 1.7–2.2 for the latent models).", lead="Matching:")  # fmt: skip
bullet(f" matching beats the no-photo forecast only with the logistic fill; with the linear fill it is worse "
       f"({g('Matching, photo (linear fill)')}). Part of what it matches is the shape of our extrapolation curve, "
       "not only real ripening.", lead="But it depends on the fill:")  # fmt: skip

doc.add_heading("2. Can a latent model be rescued by blending?", level=1)
para("Each forecast is mixed with the no-photo forecast, (1 − w) × model + w × no-photo. Chosen on the validation "
     "fruit, w comes out at 1 (no-photo only) for every latent run. The chart scores every w on the test fruit to "
     "show this is not a validation fluke.", size=9.5)  # fmt: skip
picture("sy_blend.png", 6.4)
bullet(" every curve falls all the way to w = 1. Had the models only been too sure of themselves, a partial blend "
       "would have beaten the no-photo forecast.", lead="No sweet spot:")  # fmt: skip

doc.add_heading("Caveats and next steps", level=1)
bullet(" 4 fruit, all spoiling on day 7–9 under the day rule, and days 7–15 of every photo series are synthetic.",
       lead="Small, partly synthetic data:")  # fmt: skip
bullet(" matching and Weibull have no randomness, so their 5 seeds are identical; their numbers are one run per fold.",
       lead="Deterministic baselines:")  # fmt: skip
bullet(" real photos past day 5 and per-fruit labels from the lab, and more fruit with varied spoilage days. Until "
       "then the no-photo forecast is the bar every new model (the transformer next) must clear.", lead="Next:")  # fmt: skip

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print("wrote", OUT)
