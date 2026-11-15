"""Shared paths for the report builders.

Charts and numbers live in ``outputs/report_figures`` (git-ignored). The Word reports are written to
``RIPEN_RESULTS_DIR`` if set, else to ``results/`` next to the repository folder (outside git).
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIGS = REPO / "outputs" / "report_figures"
RESULTS = Path(os.environ.get("RIPEN_RESULTS_DIR", REPO.parent / "results"))
FIGS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
