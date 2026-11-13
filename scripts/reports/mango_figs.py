"""Charts for the plain-language mango summary: mango_inputs.png, mango_age.png (from outputs/mango)."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[2] / "outputs" / "report_figures"  # charts + numbers live here
HERE.mkdir(parents=True, exist_ok=True)
S = HERE.parents[1] / "outputs" / "mango"
BLUE, LIGHT, DARK, FAINT = "#2a78d6", "#b4b2ab", "#6f6d68", "#dedcd5"
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#cccccc",
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "axes.axisbelow": True,
        "legend.frameon": False,
    }
)


def vals(run, key="pop_brier"):
    return np.array([json.loads(p.read_text())["cutday"][key] for p in sorted(S.glob(f"seed*/{run}/metrics.json"))])


# ---- Chart 1: which input helps -------------------------------------------------------------------------------
groups = [("rgb", "Photo only"), ("samd", "Photo only,\ntaught by microwave"), ("mw", "Microwave only"),
          ("both", "Photo +\nmicrowave")]  # fmt: skip
series = [
    ("Standard AI model (GRU)", "gru_gap", LIGHT),
    ("Transformer AI model", "transformer", DARK),
    ("Ripeness-score model (latent SDE)", "latent_sde", BLUE),
]
fig, ax = plt.subplots(figsize=(7.2, 3.3))
w = 0.26
for i, (lab, m, c) in enumerate(series):
    xs = np.arange(len(groups)) + (i - 1) * (w + 0.02)
    v = [vals(f"{m}_{g}") for g, _ in groups]
    mu = [x.mean() for x in v]
    ax.bar(xs, mu, w, color=c, label=lab, yerr=[x.std(ddof=1) for x in v], error_kw={"elinewidth": 1, "capsize": 2})
    for x, y in zip(xs, mu, strict=True):
        ax.text(x, y + 0.012, f"{y:.2f}", ha="center", va="bottom", fontsize=8.5, color="#333333")
ax.axhline(0.25, color="#999999", lw=1, ls="--")
ax.text(len(groups) - 0.55, 0.253, "guessing 50/50", ha="right", va="bottom", fontsize=8.5, color="#777777")
ax.set_xticks(range(len(groups)), [t for _, t in groups])
ax.set_ylabel("Forecast score (lower is better)")
ax.set_ylim(0, 0.36)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper right", fontsize=8.5, ncol=1)
ax.set_title("How well each input predicts spoilage", loc="left", fontsize=10.5, fontweight="bold")
fig.tight_layout()
fig.savefig(HERE / "mango_inputs.png", dpi=200)

# ---- Chart 2: age is most of the story ---------------------------------------------------------------------
best = lambda s: min((vals(f"{m}_{s}").mean(), m) for m in  # noqa: E731
                     ("gru_gap", "neural_cde", "transformer", "latent_sde", "latent_ode", "fpt_filter"))  # fmt: skip
rows = [
    ("Guessing 50/50", 0.25, FAINT),
    ("Best AI model, photo only", best("rgb")[0], LIGHT),
    ("Simple rule, photo only", vals("rgb_clock").mean(), BLUE),
    ("Best AI model, photo + microwave", best("both")[0], LIGHT),
    ("Simple rule, microwave only", vals("mw_clock").mean(), BLUE),
    ("Simple rule, photo + microwave", vals("both_clock").mean(), BLUE),
    ("Knowing the fruit's true age\n(not possible for a shopper)", vals("age_clock").mean(), DARK),
]
fig, ax = plt.subplots(figsize=(7.2, 3.0))
ys = np.arange(len(rows))[::-1]
for y, (lab, v, c) in zip(ys, rows, strict=True):
    ax.barh(y, v, 0.62, color=c)
    ax.text(v + 0.004, y, f"{v:.2f}", va="center", fontsize=9, color="#333333")
ax.set_yticks(ys, [r[0] for r in rows], fontsize=9)
ax.set_xlim(0, 0.29)
ax.set_xlabel("Forecast score (lower is better)")
ax.grid(axis="y", visible=False)
ax.set_title("A simple age-based rule beats every AI model", loc="left", fontsize=10.5, fontweight="bold")
fig.tight_layout()
fig.savefig(HERE / "mango_age.png", dpi=200)
print("best:", best("rgb"), best("both"))
