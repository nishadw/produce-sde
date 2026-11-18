"""Repository audit, 14 Nov 2026: what was wrong, what was fixed, what is still open (report env, python-docx).

Writes ../results/2026-11-14_repo-audit.docx (see _paths.py). Content is hand-written from the audit rounds
(git log 516e764..HEAD).
"""

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from _paths import RESULTS

OUT = RESULTS / "2026-11-14_repo-audit.docx"
GREY = RGBColor(0x55, 0x5B, 0x65)

doc = Document()
sec = doc.sections[0]
sec.left_margin = sec.right_margin = Inches(0.8)
sec.top_margin = sec.bottom_margin = Inches(0.7)
st = doc.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(10.5)


def shade(cell, hex_fill):
    tc = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), hex_fill)
    tc.append(sh)


def table(hdr, rows, widths):
    t = doc.add_table(rows=1, cols=len(hdr))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, h in enumerate(hdr):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        r.bold, r.font.size = True, Pt(9.5)
        shade(c, "E8EAED")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            cells[i].paragraphs[0].add_run(str(v)).font.size = Pt(9.5)
    for row in t.rows:
        for i, w in enumerate(widths):
            row.cells[i].width = Inches(w)
    doc.add_paragraph()


def bullets(items):
    for head, body in items:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(head).bold = True
        p.add_run(body)


def note(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size, r.font.color.rgb = Pt(9), GREY


doc.add_heading("Repository audit: errors found and fixed", level=1)
note(
    "14 Nov 2026 · produce-sde (package ripen) · every tracked file read line by line by independent builder agents, "
    "then re-checked by separate verifier agents until a round came back clean. Tests: 96 pass; lint clean."
)

doc.add_heading("At a glance", level=2)
table(
    ["", "Result"],
    [
        ["Files changed", "80 of 297 tracked files"],
        ["Rounds", "5: one builder pass (5 agents), three verify-and-fix rounds, one convergence check"],
        [
            "Bugs that change results",
            "1: photo-every-2nd/3rd-day runs (fixed and rerun; the sparse-photo win goes away)",
        ],
        [
            "Bugs with no effect on past results",
            "3: a clock flag that was ignored, a rerun-script edge case, 2 weak tests",
        ],
        ["Wrong statements fixed", "dozens in reports, docs and the brief (numbers, method descriptions, paths)"],
        ["Checked and correct", "all core math (PDE solver, bridge factors, Girsanov KL, metrics), all copied tables"],
    ],
    [2.2, 4.6],
)

doc.add_heading("1. The one bug that changes results", level=2)
doc.add_paragraph(
    "Photo every 2nd or 3rd day (subsample_visits). To simulate fewer photos we keep every k-th daily photo. If a fruit "
    "first looked spoiled on a day the sparser schedule skipped, the old code treated it as never spoiling. That hit "
    "about 55 % of the test fruit that do spoil (test split: 65 spoiled fruit at daily photos, only 28 at every 2nd "
    "day and 30 at every 3rd). Now the fruit counts as first seen spoiled at the next scheduled photo, so its spoilage "
    "window just widens. A new test file (tests/test_subsample.py) covers it."
)
bullets(
    [
        (
            "Affected: ",
            "the every-2nd/3rd-day rows of the 28 Oct avocado study (Round 1, Round 2b, new baselines, "
            "paired tests) and the README headline 'latent SDE CRPS 0.97 at a photo every 3rd day'.",
        ),
        (
            "Done: ",
            "old runs moved to outputs/_superseded_subsample_bug_2026-11-14/ (kept, not deleted); rerun on "
            "15 Nov (bash run_umiacs.sh sparse, 30 jobs) plus the paired tests; tables, figure and text updated.",
        ),
        (
            "What changed: ",
            "the latent SDE is no longer the clear winner with sparse photos. Every 3rd day its CRPS is "
            "1.30 (was 0.97), tied with GRU-D (1.27) and the transformer (1.20); it still beats the Wiener process. "
            "Every 2nd day the transformer is significantly better. All 65 spoiling test fruit now count at every rate.",
        ),
    ]
)

doc.add_heading("2. Corrections to the professor brief", level=2)
table(
    ["Was", "Now", "Why"],
    [
        [
            "SAMD and the lab method share the teacher's final layer",
            "share the whole predictor (every layer after the encoder)",
            "samd.share_predictor copies drift, noise, readout, head and clock",
        ],
        [
            "Rebuilt-teacher students are less over-confident",
            "they are sharper: 80 % range 3.7 → 2.4 days, more "
            "probability on the spoilage window, but the middle guess moves further off",
            "measured in outputs/a2sl_schedule",
        ],
        ["Latent SDE gives 'more honest' ranges", "wider ranges", "both models' 80 % ranges already cover every fruit"],
        ["Scanner = 0/120/240° views", "views for avocados; all five antenna spacings for mangoes", "mango inputs"],
        [
            "All methods ported from RepDistiller",
            "SAMD follows Shen et al.; KD, RKD and the lab method are ported",
            "SAMD was built from the paper",
        ],
    ],
    [2.1, 2.6, 2.1],
)
note(
    "Still 2 pages. The other four Word reports got similar fixes (e.g. a 'lower is better' score described backwards)."
)

doc.add_heading("3. Other fixes, by area", level=2)
table(
    ["Area", "Examples"],
    [
        [
            "Study write-ups",
            "C-index/Cox/gamma values mismatched to tables; 'recalibration lowers CRPS 2–5 %' (really "
            "0–8 %); '30 mangoes cut' (36); a broken table; teacher-phase 'wider' (sharper)",
        ],
        ["README", "Word reports now built into ../results (outside git); --local flag on a stage that rejects it"],
        [
            "Code",
            "ArrheniusClock ignored learn_rate_ref=False (never used, so no result changes); ~10 docstrings "
            "that disagreed with the code",
        ],
        ["Tests", "two 'hidden visits must not leak' tests skipped the first hidden visit"],
        ["Run scripts", "--local summaries tried to sbatch; dead doc paths; CPU cap comment (32, not 16)"],
        [
            "Theory / foundations",
            "prior missing √κ on the noise; test tolerances misquoted; foundations scripts wrote "
            "to a folder that does not exist; figure licence CC BY 4.0 (not 3.0), also in the lecture deck",
        ],
    ],
    [1.5, 5.3],
)

doc.add_heading("4. Checked: not a bug", level=2)
doc.add_paragraph(
    "SAMD with the scanner teacher and with a photo-only teacher give near-identical latent ODE scores. The scanner "
    "does reach the teacher; the reset schedule gives its new scanner weights only ~10 small training steps per round "
    "before they are zeroed again, so the teacher stays ~99 % photo-driven. Worth a sentence in the write-up: a longer "
    "teacher phase would test whether the scanner can help at all."
)

doc.add_heading("5. Open questions for you", level=2)
bullets(
    [
        (
            "Paper wording: ",
            "the properness result we cite (Yanagisawa & Akiyama) may be stated under a "
            "'constant-sum condition', not 'independent monitoring'; check the theorem.",
        ),
        ("Data naming: ", "some files call the lab avocado set 'A2SL / Stevens', others 'University of Alabama'."),
    ]
)

doc.save(OUT)
print(OUT)
