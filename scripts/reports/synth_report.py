"""Brief: half-synthetic A2SL models against the no-photo forecast (report env).

Run scripts/analysis/a2sl_pop_compare.py and synth_figs.py first. Writes 2026-11-18_half-synthetic-vs-no-photo.docx
to ../results (``_paths.RESULTS``). One table per fill (linear, logistic), rows grouped by what the model sees.
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
REF, T = S["ref"], S["tables"]
OUT = RESULTS / "2026-11-18_half-synthetic-vs-no-photo.docx"
GREY = RGBColor(0x55, 0x5B, 0x65)
SAMD = "SAMD, teacher sees photo + scanner"


def row(fill, model, sees):
    return next((r for r in T[fill] if r["model"] == model and r["sees"] == sees), None)


def sc(fill, model, sees="photo"):
    r = row(fill, model, sees)
    return "—" if r is None else f"{r['random']:.2f}"


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


def table(hdr, rows, widths, size=8):
    """``rows``: lists of cells, or ``("section", title)`` for a grey full-width group heading."""
    t = doc.add_table(rows=len(rows) + 1, cols=len(hdr))
    t.style, t.alignment, t.autofit = "Table Grid", WD_TABLE_ALIGNMENT.CENTER, False
    for j, w in enumerate(widths):
        t.columns[j].width = Inches(w)
    for i, r in enumerate([hdr, *rows]):
        if isinstance(r, tuple):
            c = t.cell(i, 0).merge(t.cell(i, len(hdr) - 1))
            c.text = ""
            run = c.paragraphs[0].add_run(r[1])
            run.font.size, run.bold, run.italic = Pt(size), True, True
            c.paragraphs[0].paragraph_format.space_after = Pt(0)
            shade(c, "EEEEEE")
            continue
        for j, val in enumerate(r):
            c = t.cell(i, j)
            c.width = Inches(widths[j])
            c.text = ""
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            if j > 1:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(str(val))
            run.font.size = Pt(size)
            run.bold = i == 0 or j == 0
            if i == 0:
                shade(c, "DCE6F2")
            elif r[0] == "No-photo forecast":
                shade(c, "FFF4E0")
            elif "better" in str(r[4]) and "no clear" not in str(r[4]):
                shade(c, "E3F0DC")
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def picture(name, width=6.8):
    doc.add_picture(str(FIGS / name), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


def fill_table(fill):
    hdr = ["Model", "What it sees", "Fixed start", "Random start", "vs no-photo forecast", "Fruit better",
           "Day error"]  # fmt: skip
    rows, section = [], None
    for r in T[fill]:
        if r["section"] != section:
            section = r["section"]
            if not section.startswith("Reference"):
                rows.append(("section", section))
        vs = r.get("vs")
        if vs is None:
            verdict, wins = "—", "—"
        else:
            lo, hi = vs["fruit_ci"][:2]
            word = "better" if hi < 0 else "worse" if lo > 0 else "no clear difference"
            verdict, wins = f"{vs['diff']:+.2f} {word}", f"{vs['wins']} of 4"
        fixed = "—" if r["fixed"] is None else f"{r['fixed']:.2f}"
        rows.append([r["model"], r["sees"], fixed, f"{r['random']:.2f}", verdict, wins, f"{r['mae']:.1f} d"])
    table(hdr, rows, [1.05, 2.05, 0.65, 0.7, 1.2, 0.6, 0.65])


def best(fill):
    r = min((r for r in T[fill] if r["sees"] != "date only" and "scanner at test" not in r["sees"]),
            key=lambda r: r["random"])
    return f"{r['model']} ({r['sees']}, {r['random']:.2f})"


# =====================================================================================================================
doc.add_paragraph("Avocado Spoilage: Does Any Model Beat the No-Photo Forecast?", style="Title")
para("Lab avocados (A2SL), half-synthetic set · 18 November 2026", color=GREY, size=9, after=4)
para(
    "Data: 9 avocados photographed on days 1, 3 and 5. Days 7–15 are filled in by fitting a curve to each fruit's "
    "photos, either a straight line (linear fill) or an S-curve (logistic fill); results are shown separately for "
    "each. Models are scored on the real photos of the 4 avocados tracked until they spoiled, each held out in turn, "
    "forecasting from every start day. Score = log score: lower is better. The no-photo forecast ignores the photo "
    "and says how many days fruit usually have left at this point; a model that cannot beat it is not getting "
    "anything useful from the photos.",
    size=8.5,
    color=GREY,
)
doc.add_heading("In short", level=1)
bullet(f" image matching with the logistic fill ({sc('logistic', 'Image matching')} vs {REF:.2f}, better on 3 of the "
       f"4 fruit). With the linear fill it is no better ({sc('linear', 'Image matching')}).",
       lead="Only one model beats the no-photo forecast:")  # fmt: skip
KDP = "CAML-KD, teacher sees photo only"
bullet(f" photo only {sc('linear', 'Transformer')} (linear fill) / {sc('logistic', 'Transformer')} (logistic), "
       f"date only {sc('linear', 'Transformer', 'date only')}, best with CAML-KD {sc('linear', 'Transformer', KDP)} / "
       f"{sc('logistic', 'Transformer', KDP)}. The closest neural model to the no-photo forecast, but still slightly "
       "worse on every fruit. With SAMD it did not train (see caveats).", lead="The transformer (new):")  # fmt: skip
bullet(f" the best is the latent SDE with SAMD ({sc('linear', 'Latent SDE', SAMD)}). The latent ODE is far worse "
       f"({sc('linear', 'Latent ODE')}).", lead="Every latent ODE / SDE run is worse than the no-photo forecast:")  # fmt: skip
bullet(" mixing a latent model with the no-photo forecast keeps improving it all the way to the no-photo forecast "
       "alone: the latent models add nothing beyond the date here, so it is not only over-confidence. The transformer "
       "with CAML-KD is the one exception, and a small one (best mix 1.14 vs 1.17, picked on the test fruit).",
       lead="Blending does not rescue them:")  # fmt: skip

doc.add_heading("How to read the tables", level=1)
for lead, text in (
    ("Fixed / random start:", " trained only from day 1, or from a random day each time (the setting we use)."),
    ("vs no-photo forecast:", f" random-start score minus {REF:.2f}. 'Better' or 'worse' only when the 4 fruit agree "
     "(95 % bootstrap over fruit); otherwise 'no clear difference'."),
    ("Fruit better:", " on how many of the 4 fruit the model beats the no-photo forecast."),
    ("Day error:", " days between the forecast's middle and the observed spoilage day (random start)."),
):  # fmt: skip
    bullet(text, lead=lead)

for n, fill in enumerate(("linear", "logistic"), 1):
    doc.add_heading(f"{n}. {fill.capitalize()} fill", level=1)
    para(f"Days 7–15 of every photo series filled in with a {'straight line' if fill == 'linear' else 'S-curve'}. "
         f"Best photo-only model: {best(fill)}.", size=9.5)  # fmt: skip
    fill_table(fill)

doc.add_heading("3. The two fills side by side", level=1)
picture("sy_scores.png", 6.6)
bullet(" image matching is the only model where the fill changes the verdict. Part of what it matches is the shape "
       "of our filled-in curve, not only real ripening.", lead="The fill matters for matching:")  # fmt: skip
bullet(" the teacher's extra scanner input never helps: a teacher that sees only photos does about as well.",
       lead="The scanner:")  # fmt: skip

doc.add_heading("4. Can a model be rescued by blending?", level=1)
para("Each forecast is mixed with the no-photo forecast. Chosen on validation fruit, the mix is almost always 100 % "
     "no-photo forecast. The chart scores every mix on the test fruit, to check this is not a validation fluke.",
     size=9.5)  # fmt: skip
picture("sy_blend.png", 6.4)
bullet(" the latent ODE and SDE curves fall all the way to the no-photo forecast alone.", lead="Latent models:")
bullet(" a 60–80 % mix of the transformer with CAML-KD reaches 1.14, just below 1.17. That mix was found by looking "
       "at the test fruit, and validation picked 100 % no-photo, so it is a hint of signal at most.",
       lead="Transformer + CAML-KD:")  # fmt: skip

doc.add_heading("Caveats and next steps", level=1)
bullet(" 4 fruit, all spoiling on day 7–9, and days 7–15 of every photo series are filled in, not real.",
       lead="Small, partly synthetic data:")  # fmt: skip
bullet(" matching and Weibull have no randomness, so their 5 repeats are identical.", lead="Deterministic models:")
bullet(" its validation score got worse from the first round on, so early stopping kept an almost untrained model "
       "(a flat forecast, about 6 days off). Those two runs are left out; SAMD would need a gentler setting (fewer "
       "epochs per round or a lower learning rate) for the transformer.", lead="Transformer + SAMD did not train:")  # fmt: skip
bullet(" real photos past day 5, per-fruit labels from the lab, and more fruit with varied spoilage days.",
       lead="What would settle it:")  # fmt: skip

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print("wrote", OUT)
