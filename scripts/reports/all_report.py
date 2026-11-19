"""Brief: all 15 A2SL avocados, real photos only, every model against the no-photo forecast (report env).

Run scripts/analysis/a2sl_all_compare.py and all_figs.py first. Writes 2026-11-19_all-avocados-real-photos.docx to
../results (``_paths.RESULTS``). The prose is built from the numbers, so it stays true when they change.
"""

import json

from _paths import FIGS, REPO, RESULTS

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

S = json.loads((REPO / "outputs" / "a2sl_all_compare" / "summary.json").read_text())
REF, REF_ALL, R = S["ref_obs"], S["ref_all"], S["rows"]
OUT = RESULTS / "2026-11-19_all-avocados-real-photos.docx"
GREY = RGBColor(0x55, 0x5B, 0x65)


def get(model, sees="photo"):
    return next((r for r in R if r["model"] == model and r["sees"] == sees), None)


def sc(model, sees="photo"):
    r = get(model, sees)
    return "—" if r is None else f"{r['obs']:.2f}"


def word(r):
    lo, hi = r["vs"]["fruit_ci"][:2]
    return "better" if hi < 0 else "worse" if lo > 0 else "no clear difference"


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


def main_table():
    hdr = ["Model", "What it sees", "All 12 fruit", "8 observed", "vs no-photo forecast", "Fruit better",
           "Day error"]  # fmt: skip
    rows, section = [], None
    for r in R:
        if r["section"] != section:
            section = r["section"]
            if not section.startswith("Reference"):
                rows.append(("section", section))
        verdict = wins = "—"
        if "vs" in r:
            verdict, wins = f"{r['vs']['diff']:+.2f} {word(r)}", f"{r['vs']['wins']} of 8"
        rows.append([r["model"], r["sees"], f"{r['all']:.2f}", f"{r['obs']:.2f}", verdict, wins, f"{r['mae']:.1f} d"])
    table(hdr, rows, [1.05, 2.05, 0.65, 0.7, 1.2, 0.6, 0.65])


usable = [r for r in R if "vs" in r and "scanner at test" not in r["sees"]]
better = [r for r in usable if word(r) == "better"]
best = min(usable, key=lambda r: r["obs"])
neural = [r for r in usable if r["model"] in ("Latent ODE", "Latent SDE", "Transformer")]
best_nn = min(neural, key=lambda r: r["obs"]) if neural else None
blend_gain = [r for r in usable if r.get("blend") is not None and r["blend"] < REF - 0.02]
MATCH = get("Image matching")
LOSSES = sorted(MATCH["vs"]["per_fruit"].items(), key=lambda kv: -kv[1])
LOSSES = [(f, d) for f, d in LOSSES if d > 0]
TCURVE = S["curve"].get("Transformer, photo", {})
TBEST = min(TCURVE.items(), key=lambda kv: kv[1]) if TCURVE else None

# =====================================================================================================================
doc.add_paragraph("Avocado Spoilage: All 15 Avocados, Real Photos Only", style="Title")
para("Lab avocados (A2SL) · 19 November 2026", color=GREY, size=9, after=4)
para(
    "Every avocado in the A2SL set is used, with its real photos only (no filled-in days). Each fruit with an "
    "unspoiled photo (avo01–12) is held out in turn and forecast from every start day; avo13–15 were first "
    "photographed already spoiled, so they only help train. Score = log score: lower is better. Main comparison: the "
    "8 fruit seen unspoiled and then spoiled. The other 4 were cut before day 9 and only tell us 'not spoiled yet', "
    "which every forecast gets right. The no-photo forecast ignores the photo and says how many days fruit usually "
    "have left at this point; a model that cannot beat it is not getting anything useful from the photos.",
    size=8.5,
    color=GREY,
)
doc.add_heading("In short", level=1)
if better:
    bullet(" " + "; ".join(f"{r['model']} ({r['sees']}, {r['obs']:.2f}, better on {r['vs']['wins']} of 8 fruit)"
                           for r in better) + f", against {REF:.2f}.",
           lead="Beats the no-photo forecast:")  # fmt: skip
else:
    bullet(f" no model is clearly better than the no-photo forecast ({REF:.2f}) across the 8 fruit. The best is "
           f"{best['model']} ({best['sees']}, {best['obs']:.2f}: {word(best)}, better on {best['vs']['wins']} of 8).",
           lead="Nothing beats the no-photo forecast:")  # fmt: skip
if best_nn is not None:
    bullet(f" best is {best_nn['model']} ({best_nn['sees']}, {best_nn['obs']:.2f}: {word(best_nn)}). Photo only: "
           f"latent ODE {sc('Latent ODE')}, latent SDE {sc('Latent SDE')}, transformer {sc('Transformer')}.",
           lead="Neural models:")  # fmt: skip
bullet(f" better than the no-photo forecast on {MATCH['vs']['wins']} of 8 fruit, but much worse on "
       + " and ".join(f"{f} (+{d:.2f})" for f, d in LOSSES[:2])
       + f", so overall {sc('Image matching')}: {word(MATCH)}. Its day error is the lowest of any model "
       f"({MATCH['mae']:.1f} d vs {get('No-photo forecast', 'date only')['mae']:.1f} d).",
       lead="Image matching is closest:")  # fmt: skip
bullet(f" Weibull {sc('Weibull')} (photo), {sc('Weibull', 'date only')} (date only).", lead="Weibull:")  # fmt: skip
bullet(" " + ("mixed with the no-photo forecast (mix chosen on validation fruit), "
              + ", ".join(f"{r['model']} ({r['sees']}) reaches {r['blend']:.2f}" for r in blend_gain)
              + f" against {REF:.2f}: a small gain, not tested for significance. Every other model's best mix is the "
              "no-photo forecast alone."
              if blend_gain else "mixing any model with the no-photo forecast, the mix chosen on validation fruit, "
              "never beats the no-photo forecast."), lead="Blending:")  # fmt: skip

doc.add_heading("How to read the table", level=1)
for lead, text in (
    ("All 12 / 8 observed:", " score over every held-out fruit, and over the 8 whose spoilage was seen."),
    ("vs no-photo forecast:", f" 8-observed score minus {REF:.2f}. 'Better' or 'worse' only when the 95 % bootstrap "
     "over the 8 fruit excludes 0; otherwise 'no clear difference'."),
    ("Fruit better:", " of the 8 observed fruit, on how many the model beats the no-photo forecast."),
    ("Day error:", " days between the forecast's middle and the observed spoilage day (8 observed fruit)."),
    ("Training:", " random start everywhere; 5 repeats x 12 held-out fruit."),
):  # fmt: skip
    bullet(text, lead=lead)

doc.add_heading("1. Every model", level=1)
main_table()
doc.add_heading("2. Best setting of each model", level=1)
picture("al_scores.png", 6.6)
para("* = teacher sees photos only.", size=8.5, color=GREY)
doc.add_heading("3. Can a model be rescued by blending?", level=1)
para("Each forecast is mixed with the no-photo forecast. The chart scores every mix on the test fruit, to show "
     "whether any mix could help; the table's blend uses the mix chosen on validation fruit.", size=9.5)  # fmt: skip
picture("al_blend.png", 6.4)
if TBEST is not None and TBEST[1] < REF - 0.02:
    bullet(f" a {float(TBEST[0]) * 100:.0f} % no-photo mix reaches {TBEST[1]:.2f} on the test fruit, but validation "
           "fruit picked almost 100 % no-photo, so this mix could not have been found without looking at the test "
           "fruit. A hint that the photo carries a little signal, not a usable model.",
           lead="Transformer, photo:")  # fmt: skip
bullet(" the latent ODE and SDE curves fall all the way to the no-photo forecast alone.", lead="Latent models:")

doc.add_heading("Caveats", level=1)
bullet(" labels follow the day rule (day 9 = spoiled), so every fruit spoils between day 7 and 9. More fruit give "
       "more test cases, not fruit-to-fruit differences for the photos to explain.", lead="Same answer for every fruit:")  # fmt: skip
bullet(" only avo01–03 and 11 have scans at every visit; the teacher sees scans where they exist.",
       lead="Scanner coverage:")  # fmt: skip
bullet(" matching and Weibull have no randomness, so their 5 repeats are identical.", lead="Deterministic models:")
bullet(" per-fruit spoilage labels from the lab, and more fruit with varied spoilage days.", lead="What would settle it:")

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print("wrote", OUT)
