"""Plain-language report: scanner views, time-series models and teaching a photo model, avocados and mangoes.

Run views_aml_figs.py first (ripen env; writes the va_*.png charts and views_aml_numbers.json), then this with
python-docx (report env). Tables are filled from views_aml_numbers.json; prose numbers are read from it too, except
the recalibration test on mangoes (scratch/mango_recal_diag.py), typed below.
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
OUT = REPO / "results" / "2026-11-12_views-time-series-models-and-aml.docx"
N = json.loads((HERE / "views_aml_numbers.json").read_text())
A, AML, G, GRID = N["avo"], N["aml"], N["mango"], N["grid"]
P = "sep1+sep2+sep3+sep4+sep5"
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
for name, size in (("Title", 19), ("Heading 1", 12.5), ("Heading 2", 11)):
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


def cell_text(c, main, bold=False, center=False, size=9):
    c.text = ""
    p = c.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(main)
    r.bold = bold
    r.font.size = Pt(size)


def caption(text):
    para(text, size=8.5, color=GREY, italic=True, after=6)


def table(hdr, rows, widths, bold_first=True):
    t = doc.add_table(rows=len(rows) + 1, cols=len(hdr))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for j, w in enumerate(widths):
        t.columns[j].width = Inches(w)
    for j, m in enumerate(hdr):
        c = t.cell(0, j)
        c.width = Inches(widths[j])
        cell_text(c, m, bold=True, center=j > 0, size=8.5)
        shade(c, "DCE6F2")
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.width = Inches(widths[j])
            cell_text(c, val, bold=(j == 0 and bold_first), center=j > 0)
    return t


def picture(name, width):
    doc.add_picture(str(HERE / name), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


def a(key, metric="log", recal=False, fmt="{:.2f}", src=None):
    d = (src or A).get(key + ("_recal" if recal else ""))
    return "—" if not d else fmt.format(d[metric][0])


def g(run, metric="pop_brier", fmt="{:.3f}"):
    d = G.get(run)
    return "—" if not d else fmt.format(d[metric][0])


def sig(boot):
    """Bootstrap string -> True if the 95 % interval excludes 0 and the difference is negative (better)."""
    return bool(boot) and "*" in boot and boot.strip().startswith("-")


# =====================================================================================================================
doc.add_paragraph("Scanner Views, Time-Series Models and Teaching a Photo Model: Avocados and Mangoes", style="Title")
para("Nishad Wajge", color=GREY, size=9.5, after=6)

doc.add_heading("Background", level=1)
para(
    "This study asks three questions about the microwave scanner, on two fruits. First, which of the scanner's views "
    "carries the most information, and is it better to use one view, merge all of them, or train one model per view "
    "and combine their forecasts? Second, which kind of time-series model makes the best use of repeated scans? Third, "
    "can a model that sees both the photo and the scan teach a model that sees only the photo, which is what a shopper "
    "would have? For the third question we used the teaching methods in the lab's code (RepDistiller and TMC) and the "
    "lab's own method, plus two methods from our earlier work."
)
para(
    "15 avocados followed for two weeks. Four (avocados 1, 2, 3 and 11) were photographed and scanned every "
    "other day until they spoiled; the other 11 were scanned once. Each scan records six views, 60° apart. Each model "
    "was trained with one followed avocado held out and scored on it, four times over, and every run was repeated 3 "
    "times. The score is how much probability the forecast gave to the day the avocado really spoiled (lower is better).",
    lead="Avocados: ",
)
para(
    "48 mangoes from one batch, scanned every other day for two weeks; on each of those days 6 were cut open, "
    "photographed on two sides and checked for spoilage. The scanner is a ring of 10 antennas; we grouped the 45 antenna "
    "pairs by how far apart the two antennas are (1 to 5 places around the ring; 5 apart is straight across), plus "
    "the scanner's own image reconstruction. The score compares each mango's forecast with what happened to the other "
    "mangoes over the next days (0 is perfect, 0.25 is a coin flip; lower is better).",
    lead="Mangoes: ",
)

# ---------------------------------------------------------------------------------------------------------------------
doc.add_heading("Main findings", level=1)
best_sde = min(["ours_rkd", "ours_sp", "samd_paired"], key=lambda m: AML[f"latent_sde_{m}"]["log"][0])
findings = [
    ("Avocado views: ", "with the latent SDE the 0°, 120° and 240° views are again the most useful. With the latent ODE "
     "every view does about equally well, and the differences are small next to run-to-run variation. Training six "
     "separate models and combining their forecasts was no better than merging the views into one model. [Chart 1]"),
    ("Best avocado model: ", f"the latent ODE beat the latent SDE with every input (photo + 0/120/240°: "
     f"{a('latent_ode_photo_mw_0+120+240')} vs {a('latent_sde_photo_mw_0+120+240')}). [Table 1]"),
    ("Teaching a photo model, avocados: ", "the methods that rebuild the teacher from the student each round "
     f"(SAMD, the lab's method and its variants) beat the photo-only model, by up to a third (latent SDE: "
     f"{a('latent_sde_photo', src=AML)} to {a('latent_sde_' + best_sde, src=AML)}). Methods with a fixed teacher did not "
     "help. [Chart 2, Table 2]"),
    ("That gain does not come from the scanner: ", "the same take-turns training with no teaching term at all does as "
     f"well or better (latent SDE {a('latent_sde_reset_only', src=AML)}, latent ODE {a('latent_ode_reset_only', src=AML)}"
     f", against {a('latent_sde_photo', src=AML)} and {a('latent_ode_photo', src=AML)} for photo-only), and so does a "
     f"teacher that sees only photos (SAMD: {a('latent_sde_samd_photoT', src=AML)} and "
     f"{a('latent_ode_samd_photoT', src=AML)}). What helps is the way the photo model is trained, not what the scanner "
     "teaches it. The teacher itself, with the scanner, was never better than the photo-only model. [Table 2]"),
    ("Mango views: ", f"every antenna spacing beats the photo alone ({g('latent_sde_rgb')}). Spacings 1, 2, 4 and 5 are "
     f"within 0.01 of each other (best: 5 apart, straight across, with the photo, {g('latent_sde_both_vsep5')}). "
     "Antennas 3 apart are the weakest "
     f"({g('latent_sde_both_vsep3')}). Combining six per-view models ({g('latent_sde_both_vfuse-mean')}) is as good as "
     "the best single spacing. The two photo sides are interchangeable. [Chart 3]"),
    ("Mango time series: ", f"once a model sees every earlier scan and knows how long the mango has been stored, elapsed "
     f"time alone is enough. The latent ODE with time only ({g('latent_ode_time_hist')}) matches the known-age rule "
     f"({g('age_clock')}); adding the scans ({g(f'latent_ode_both_v{P}_hist')}) or photos ({g('latent_ode_rgb_hist')}) "
     "does not help. The latent ODE beats the latent SDE with every input. [Chart 4, Table 3]"),
    ("Teaching a photo model, mangoes: ", f"no method beat the photo-only model ({g('latent_ode_rgb_hist')}); all "
     f"landed within 0.012 of it, even though the teacher itself does better ({g(f'latent_ode_both_v{P}_hist')}). "
     "[Chart 5]"),
    ("Correction step: ", "the step that makes stated ranges honest improved most avocado scores. After it, the "
     f"take-turns gains hold for the latent SDE (photo-only {a('latent_sde_photo', recal=True, src=AML)}, take-turns "
     f"{a('latent_sde_reset_only', recal=True, src=AML)}) but not for the latent ODE (photo-only "
     f"{a('latent_ode_photo', recal=True, src=AML)}, best method {a('latent_ode_ours_rkd', recal=True, src=AML)}). On "
     "mangoes it made every model worse in all six versions we tried, so mango numbers here are uncorrected."),
]  # fmt: skip
for lead, txt in findings:
    para(txt, lead=lead, style="List Bullet", after=2)

# ---------------------------------------------------------------------------------------------------------------------
doc.add_heading("Avocados", level=1)
doc.add_heading("Chart 1: Every scanner view, two models", level=2)
picture("va_avo_views.png", 6.7)
caption(
    "Each pair of bars is one choice of scanner view: grey with the microwave alone, blue with the photo added. Bars "
    "are averages over 3 runs; black lines show how much they vary. Dashed line: a model that knows only the date. "
    "Dotted line: the photo alone. “6 models combined”: one model per view, forecasts averaged afterwards."
)
para("Table 1: Two time-series models", after=2)
MODELS = [("latent_sde", "Latent SDE"), ("latent_ode", "Latent ODE")]
COLS = [("time", "Date only"), ("photo", "Photo only"), ("mw_0+120+240", "Microwave 0/120/240°"),
        ("photo_mw_0+120+240", "Photo + 0/120/240°"), ("photo_mw_all", "Photo + all six"),
        ("photo_mw_fuse-mean", "Photo + six models combined")]  # fmt: skip
rows = []
for m, lab in MODELS:
    rows.append([lab] + [f"{a(f'{m}_{c}')} ({a(f'{m}_{c}', 'mae')} d)" for c, _ in COLS])
table(["Model"] + [c[1] for c in COLS], rows, [0.95, 0.9, 0.9, 0.95, 0.95, 0.9, 1.05])
caption(
    "Forecast score, lower is better, with the average days off in brackets. Followed avocados, 3 runs each. The "
    "latent SDE and latent ODE imagine a hidden ripeness that drifts over time (the SDE adds random jolts)."
)

doc.add_heading("Chart 2: Teaching a photo-only model", level=2)
picture("va_avo_aml.png", 6.7)
caption(
    "Each blue or orange bar is a photo-only model trained with help from a teacher that also sees the 0°, 120° and "
    "240° scans. Orange: clearly better than the photo-only model (the 95 % range of the difference excludes zero). The "
    "dark bar is the teacher itself. The three “control” rows use the same training schedule, but either drop the "
    "teaching term or give the teacher photos only. Blank rows were not run for that model."
)
para("Table 2: Teaching methods, before and after the correction step", after=2)
TM = [("photo", "Photo only, no teacher"), ("kd", "Teacher's forecasts (KD)"), ("samd", "SAMD"),
      ("ours", "Lab method: reset + KD"), ("ours_rkd", "Reset + RKD"), ("ours_sp", "Reset + SP"),
      ("samd_paired", "SAMD, scanned fruit only"), ("reset_only", "Control: no teaching term"),
      ("samd_photoT", "Control: SAMD, photo-only teacher"), ("ours_rkd_photoT", "Control: reset + RKD, photo-only teacher"),
      ("teacher", "The teacher (needs the scanner)")]  # fmt: skip
rows = []
for meth, lab in TM:
    r = [lab]
    for m in ("latent_sde", "latent_ode"):
        k = f"{m}_{meth}"
        r += [a(k, src=AML), a(k, recal=True, src=AML)]
    rows.append(r)
table(["Method", "Latent SDE", "Latent SDE, corrected", "Latent ODE", "Latent ODE, corrected"], rows,
      [2.4, 1.0, 1.1, 1.0, 1.1])  # fmt: skip
caption(
    "Forecast score, lower is better; followed avocados, 3 runs each. “Corrected”: after the step that makes stated "
    "ranges honest, learned from the other held-out avocados."
)
gr = GRID.get("ours_rkd", {}).get("best")
gs = GRID.get("samd_paired", {}).get("best")
if gr and gs:
    para(
        "We tried 3 teaching strengths × 3 learning speeds for the two best methods and picked the setting with the "
        f"best score on fruit held aside for checking, not on the test avocados. For reset + RKD the pick was strength "
        f"{gr['beta']:g}, speed {gr['lr']:g} (score {gr['test_log']:.2f}). For SAMD on scanned fruit it was strength "
        f"{gs['beta']:g}, speed {gs['lr']:g} (score {gs['test_log']:.2f}; the same setting scored "
        f"{a('latent_sde_samd_paired', src=AML)} in an earlier run, so repeat runs vary by about 0.2). The teaching "
        "strength barely mattered: tenfold changes moved the score less than repeating a run. The learning speed "
        "mattered much more, which fits the finding that the training schedule, not the teacher, is what helps.",
        lead="Tuning. ",
    )

# ---------------------------------------------------------------------------------------------------------------------
doc.add_heading("Mangoes", level=1)
doc.add_heading("Chart 3: One scan, by antenna spacing", level=2)
picture("va_mango_views.png", 6.4)
caption(
    "Latent SDE, one scan per mango, 3 runs each. Grey: microwave alone; blue: photo added. Dotted line: photo alone. "
    "Dashed line: a rule that knows each mango's true age and looks up how fast this batch spoiled. Every scanner "
    "input beats the photo; none comes near the known-age rule."
)
doc.add_heading("Chart 4: Every earlier scan, two time-series models", level=2)
picture("va_mango_hist.png", 6.4)
caption(
    "Here each forecast sees all of that mango's earlier scans and knows how long it has been stored. Microwave: all "
    "five antenna spacings averaged. The photo is taken only on the day the mango is cut, so “Photos” adds one photo "
    "to elapsed time."
)
para("Table 3: Mango, every earlier scan", after=2)
MI = [("time", "Elapsed time only"), ("rgb", "Photo"), (f"mw_v{P}", "Microwave"), (f"both_v{P}", "Photo + microwave")]
rows = [[lab] + [g(f"{m}_{s}_hist") for s, _ in MI] for m, lab in MODELS]
rows.append(["Known age + batch curve", g("age_clock"), "", "", ""])
table(["Model"] + [x[1] for x in MI], rows, [1.9, 1.2, 1.2, 1.2, 1.3])
caption("Spoilage score, lower is better; 3 runs each.")

doc.add_heading("Chart 5: Teaching a photo-only model", level=2)
picture("va_mango_aml.png", 5.6)
caption(
    "Latent ODE with every earlier scan. The teacher sees the photo and the microwave scans. No method moved the "
    "photo-only model by more than its run-to-run variation."
)

# ---------------------------------------------------------------------------------------------------------------------
doc.add_heading("The teaching methods", level=1)
for lead, txt in [
    ("Teacher's forecasts (KD): ", "the student is pushed to give the same forecast as the teacher."),
    ("Teacher's internal summary: ", "the student is pushed to build the same internal summary of the fruit."),
    ("RKD, PKT, SP: ", "instead of copying the summary itself, the student copies how the teacher sees fruit relative to "
     "each other: the distances between fruit (RKD), which fruit are near neighbours (PKT), or how similar each pair is "
     "(SP)."),
    ("SAMD and the lab's method (“reset”): ", "the teacher and student share their last layer and take turns training. "
     "Every round the teacher is rebuilt from the student, keeps learning with the scan, and hands its last layer back. "
     "SAMD then matches the summaries; the lab's method matches the forecasts; “reset + RKD/SP” combine the reset "
     "with RKD or SP."),
    ("Take-turns training (control): ", "the same rounds as SAMD, with the shared last layer, but the student gets no "
     "teaching term; any gain then comes from the schedule itself. A second control gives the teacher photos only."),
    ("Scanned fruit only: ", "the teacher's terms count only for fruit that actually have scans."),
    ("Six models combined: ", "from the lab's TMC code: train one model per view and combine their forecasts afterwards, "
     "by averaging, by a geometric mean (views must agree), or weighting the more confident views more. Averaging was "
     "best; the three differed by less than run-to-run variation."),
]:  # fmt: skip
    para(txt, lead=lead, style="List Bullet", after=2)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print(OUT)
