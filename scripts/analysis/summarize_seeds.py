"""Mean ± std over seeds for the independent avocado rerun (outputs/avocado/round1/seed*/<run>/metrics_test.json).

python scripts/analysis/summarize_seeds.py            # writes outputs/avocado/round1/summary.md
python scripts/analysis/summarize_seeds.py --root outputs/avocado/round2 --recal   # -> summary_recal.md
Held-out runs named ``<model>_avocado_logo_<g>[_<clock>]<suffix>``; Round 2 uses ``--suffix _guided_r18``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("outputs/avocado/round1")
METRICS = "metrics_test.json"
SUFFIX = ""
COLS = [("mae", "MAE (d)"), ("crps", "CRPS"), ("coverage80", "80% cov."), ("width80", "80% width"),
        ("cindex", "C-index"), ("ece", "ECE")]  # fmt: skip
LOGO_MODELS = ["direct_reg", "gru_d", "fpt_filter", "latent_sde", "latent_sde+clock_group", "latent_sde+clock_fixed"]


def load() -> dict[str, list[dict]]:
    """run name -> list of per-seed metrics_test.json dicts."""
    runs: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(ROOT.glob(f"seed*/*/{METRICS}")):
        runs[p.parent.name].append(json.loads(p.read_text()))
    return runs


def ms(vals: list[float]) -> str:
    v = np.array([x for x in vals if x == x], float)  # drop NaN
    if len(v) == 0:
        return "—"
    return f"{v.mean():.3f}" if len(v) == 1 else f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"


def logo_run(model: str, g: str) -> str:
    base, _, clock = model.partition("+")
    return f"{base}_avocado_logo_{g}" + (f"_{clock}" if clock else "") + SUFFIX


def main() -> None:
    global ROOT, METRICS, SUFFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--recal", action="store_true", help="read metrics_test_recal.json (from recalibrate.py)")
    ap.add_argument("--suffix", default="", help="run-name suffix of the held-out runs, e.g. _guided_r18")
    a = ap.parse_args()
    SUFFIX = a.suffix
    ROOT, METRICS = Path(a.root), ("metrics_test_recal.json" if a.recal else "metrics_test.json")
    runs = load()
    seeds = sorted({p.parent.parent.name for p in ROOT.glob(f"seed*/*/{METRICS}")})
    lines = [
        f"# Avocado multi-seed results: `{ROOT}`" + (" (post-hoc recalibrated on val)" if a.recal else ""),
        "",
        f"Seeds: {', '.join(seeds) or 'none yet'}. Cells are mean ± sample std over seeds; the split is fixed "
        "(same test items for every seed), so the spread is training variance only. Test split, every unspoiled "
        "prefix of every test item.",
        "",
        "## Overall, every run",
        "",
        "| run | seeds | " + " | ".join(c for _, c in COLS) + " |",
        "|---|---|" + "---|" * len(COLS),
    ]
    for name, ms_list in sorted(runs.items()):
        o = [m["overall"] for m in ms_list]
        lines.append(f"| {name} | {len(o)} | " + " | ".join(ms([x[k] for x in o]) for k, _ in COLS) + " |")

    lines += ["", "## Held-out storage temperature: MAE (d) and 80% coverage side by side", "",
              "Coverage should be near 0.80; far below means the intervals are overconfident on the new condition.",
              "", "| model | " + " | ".join(f"{g} MAE | {g} cov." for g in ("T10", "T20", "Tamb")) + " |",
              "|---|" + "---|" * 6]  # fmt: skip
    for model in LOGO_MODELS:
        cells = []
        for g in ("T10", "T20", "Tamb"):
            o = [m["overall"] for m in runs.get(logo_run(model, g), [])]
            cells += [ms([x["mae"] for x in o]), ms([x["coverage80"] for x in o])]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")

    lines += ["", "## Clock ablation: MAE ratio (ablated / Arrhenius), mean over seeds", "",
              "| held out | group / Arrhenius | fixed / Arrhenius |", "|---|---|---|"]  # fmt: skip
    for g in ("T10", "T20", "Tamb"):
        base = [m["overall"]["mae"] for m in runs.get(logo_run("latent_sde", g), [])]
        cells = []
        for c in ("clock_group", "clock_fixed"):
            abl = [m["overall"]["mae"] for m in runs.get(logo_run(f"latent_sde+{c}", g), [])]
            cells.append(f"{np.mean(abl) / np.mean(base):.2f}×" if base and abl else "—")
        lines.append(f"| {g} | " + " | ".join(cells) + " |")

    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / ("summary_recal.md" if a.recal else "summary.md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
