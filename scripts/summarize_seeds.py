"""Mean ± std over seeds for the independent avocado rerun (results/avocado_seeds/seed*/<run>/metrics_test.json).

python scripts/summarize_seeds.py            # writes results/avocado_seeds/summary.md
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("results/avocado_seeds")
COLS = [("mae", "MAE (d)"), ("crps", "CRPS"), ("coverage80", "80% cov."), ("width80", "80% width"),
        ("cindex", "C-index"), ("ece", "ECE")]  # fmt: skip
LOGO_MODELS = ["direct_reg", "gru_d", "fpt_filter", "latent_sde", "latent_sde+clock_group", "latent_sde+clock_fixed"]


def load() -> dict[str, list[dict]]:
    """run name -> list of per-seed metrics_test.json dicts."""
    runs: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(ROOT.glob("seed*/*/metrics_test.json")):
        runs[p.parent.name].append(json.loads(p.read_text()))
    return runs


def ms(vals: list[float]) -> str:
    v = np.array([x for x in vals if x == x], float)  # drop NaN
    if len(v) == 0:
        return "—"
    return f"{v.mean():.3f}" if len(v) == 1 else f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"


def logo_run(model: str, g: str) -> str:
    base, _, clock = model.partition("+")
    return f"{base}_avocado_logo_{g}" + (f"_{clock}" if clock else "")


def main() -> None:
    runs = load()
    seeds = sorted({p.parent.parent.name for p in ROOT.glob("seed*/*/metrics_test.json")})
    lines = [
        "# Avocado, independent multi-seed rerun",
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
    (ROOT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
