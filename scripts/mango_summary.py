"""Comparison tables for the mango study: results/mango/seed*/<run>/metrics.json -> results/mango/summary.md.

Every run is scored the same way (``ripen.eval.mango``); cells are mean ± std over seeds. The paired bootstrap
compares two runs fruit by fruit (each fruit's population-Brier part, averaged over seeds; fruit resampled 5000
times): negative = the first run is better, ``*`` = the 95 % interval excludes 0.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("results/mango")
SETTINGS = ["rgb", "samd", "mw", "both"]
LABEL = {"rgb": "photo only", "samd": "photo only, SAMD-distilled", "mw": "microwave only", "both": "photo + microwave"}
MODELS = ["gru_gap", "gru_d", "ode_rnn", "neural_cde", "transformer", "latent_sde", "latent_ode", "fpt_filter"]
REFS = {
    "age_clock": "known age + batch spoilage curve (oracle reference)",
    "rgb_clock": "photo -> estimated age + batch curve",
    "mw_clock": "microwave -> estimated age + batch curve",
    "both_clock": "photo + microwave -> estimated age + batch curve",
}


def load() -> dict[str, list[dict]]:
    runs = defaultdict(list)
    for p in sorted(ROOT.glob("seed*/*/metrics.json")):
        runs[p.parent.name].append(json.loads(p.read_text()))
    return runs


def ms(v, fmt="{:.3f}") -> str:
    v = np.array([x for x in v if x == x], float)
    if not len(v):
        return "—"
    return fmt.format(v.mean()) if len(v) == 1 else f"{fmt.format(v.mean())} ± {fmt.format(v.std(ddof=1))}"


def per_fruit(run: str) -> dict[str, tuple[float, float]]:
    """fruit -> (sum of squared errors, pair count), averaged over seeds."""
    acc = defaultdict(list)
    for p in sorted(ROOT.glob(f"seed*/{run}/predictions.npz")):
        with np.load(p) as z:
            for f, (s, n, _) in zip(z["pf_fruit"], z["pf"], strict=True):
                acc[str(f)].append((s, n))
    return {f: (float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v]))) for f, v in acc.items()}


def bootstrap(a: str, b: str, n_boot: int = 5000, seed: int = 0) -> str:
    pa, pb = per_fruit(a), per_fruit(b)
    keys = sorted(set(pa) & set(pb))
    if not keys:
        return "—"
    sa = np.array([pa[k][0] for k in keys])
    sb = np.array([pb[k][0] for k in keys])
    n = np.array([pa[k][1] for k in keys])
    diff = (sa.sum() - sb.sum()) / n.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    boot = (sa[idx].sum(1) - sb[idx].sum(1)) / np.maximum(n[idx].sum(1), 1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    star = " *" if lo > 0 or hi < 0 else ""
    return f"{diff:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}"


def main() -> None:
    runs = load()
    seeds = sorted({p.parent.parent.name for p in ROOT.glob("seed*/*/metrics.json")})

    def cell(run, block, key, fmt="{:.3f}"):
        return ms([m.get(block, {}).get(key, float("nan")) for m in runs.get(run, [])], fmt)

    L = [
        "# Mango study: every model x input setting",
        "",
        f"Seeds: {', '.join(seeds) or 'none'}; 6-fold cross-validation by fruit (each fruit is tested once per seed). "
        "Cells: mean ± std over seeds. Scored on each fruit's cut day, without stage labels (`ripen.eval.mango`).",
        "",
        "## Main score: population Brier of the days-until-spoiled forecast (lower is better)",
        "",
        "| model | " + " | ".join(LABEL[s] for s in SETTINGS) + " |",
        "|---|" + "---|" * len(SETTINGS),
    ]
    for m in MODELS:
        L.append(f"| {m} | " + " | ".join(cell(f"{m}_{s}", "cutday", "pop_brier") for s in SETTINGS) + " |")
    L += ["", "References (closed form, no neural network):", "", "| reference | population Brier | status Brier | "
          "Spearman vs firmness | within-day Spearman |", "|---|---|---|---|---|"]  # fmt: skip
    for r, lab in REFS.items():
        L.append(f"| {lab} | " + " | ".join(cell(r, "cutday", k) for k in
                 ("pop_brier", "status_brier", "spearman_firmness", "spearman_within_day")) + " |")  # fmt: skip
    for key, title in [
        ("status_brier", "Spoiled-now Brier on the cut day (lower is better)"),
        ("spearman_firmness", "Spearman: expected days left vs measured firmness (higher is better)"),
        ("spearman_within_day", "Within-day Spearman (fruit of the same age only; higher is better)"),
        ("mean_predicted_spoil_day", "Mean predicted spoilage day (the batch spoiled between days 10 and 14)"),
    ]:
        L += ["", f"## {title}", "", "| model | " + " | ".join(LABEL[s] for s in SETTINGS) + " |",
              "|---|" + "---|" * len(SETTINGS)]  # fmt: skip
        for m in MODELS:
            L.append(f"| {m} | " + " | ".join(cell(f"{m}_{s}", "cutday", key) for s in SETTINGS) + " |")
    L += ["", "## Earlier scans (microwave settings): interval-censored log score, lower is better", "",
          "| model | microwave only | photo + microwave |", "|---|---|---|"]  # fmt: skip
    for m in MODELS:
        L.append(f"| {m} | {cell(f'{m}_mw', 'prefix', 'log_score')} | {cell(f'{m}_both', 'prefix', 'log_score')} |")
    L += ["", "## Paired bootstrap on population Brier (first minus second; negative = first is better)", "",
          "| model | SAMD − photo only | photo + microwave − photo only | SAMD − photo→age reference |",
          "|---|---|---|---|"]  # fmt: skip
    for m in MODELS:
        L.append(f"| {m} | {bootstrap(f'{m}_samd', f'{m}_rgb')} | {bootstrap(f'{m}_both', f'{m}_rgb')} | "
                 f"{bootstrap(f'{m}_samd', 'rgb_clock')} |")  # fmt: skip
    L += [
        "",
        "## Other outputs on the cut day: firmness and ripeness class read out by ridge regression",
        "",
        "| features | firmness MAE (N) | log-firmness MAE | class accuracy | class within one |",
        "|---|---|---|---|---|",
    ]
    for r in ("rgb_clock", "mw_clock", "both_clock"):
        L.append(f"| {REFS[r].split(' ->')[0]} | " + " | ".join(cell(r, "readout", k) for k in
                 ("firmness_mae_n", "log_firmness_mae", "stage_accuracy", "stage_within_one")) + " |")  # fmt: skip
    text = "\n".join(L) + "\n"
    (ROOT / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
