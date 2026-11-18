"""Charts for the half-synthetic brief (ripen env). Reads outputs/a2sl_pop_compare/summary.json
(scripts/analysis/a2sl_pop_compare.py); writes sy_*.png to outputs/report_figures. synth_report.py builds the docx."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "outputs" / "report_figures"
HERE.mkdir(parents=True, exist_ok=True)
S = json.loads((REPO / "outputs" / "a2sl_pop_compare" / "summary.json").read_text())
DARK, LIGHT, BLUE = "#6f6d68", "#b4b2ab", "#2a78d6"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#cccccc", "axes.grid": True, "grid.color": "#e6e5e0", "axes.axisbelow": True,
                     "legend.frameon": False})  # fmt: skip

# chart 1: every model's score against the no-photo line
ROWS = [("Matching, photo (logistic fill)", "Matching\n(logistic)"), ("Matching, photo (linear fill)", "Matching\n(linear)"),
        ("Latent SDE + SAMD (linear)", "Latent SDE\n+ SAMD"), ("Latent SDE, photo (linear)", "Latent SDE"),
        ("Latent SDE + CAML-KD (linear)", "Latent SDE\n+ CAML-KD"), ("Weibull, photo (logistic fill)", "Weibull\n(logistic)"),
        ("Latent ODE + CAML-KD (linear)", "Latent ODE\n+ CAML-KD"), ("Latent ODE, photo (linear)", "Latent ODE")]  # fmt: skip
ref = S["scores"]["No-photo forecast"]["log"]
y = [S["scores"][k]["log"] for k, _ in ROWS]
fig, ax = plt.subplots(figsize=(7.2, 2.7))
ax.bar(range(len(y)), y, 0.6, color=[BLUE if v < ref else LIGHT for v in y])
for i, v in enumerate(y):
    ax.text(i, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
ax.axhline(ref, color=DARK, lw=1, ls="--")
ax.set_xticks(range(len(y)), [lab for _, lab in ROWS], fontsize=7.5)
ax.set_ylim(0, max(y) * 1.15)
ax.set_title(f"Score on the real visits, all start days (lower is better); dashed = no-photo forecast {ref:.2f}", loc="left", fontsize=9.5,
             fontweight="bold")  # fmt: skip
fig.tight_layout()
fig.savefig(HERE / "sy_scores.png", dpi=200)
plt.close(fig)

# chart 2: blending with the no-photo forecast never beats it
CUR = [("Latent SDE + SAMD (linear)", "Latent SDE + SAMD", BLUE), ("Latent SDE, photo (linear)", "Latent SDE", DARK),
       ("Latent ODE + CAML-KD (linear)", "Latent ODE + CAML-KD", LIGHT)]  # fmt: skip
fig, ax = plt.subplots(figsize=(7.2, 2.5))
for k, lab, c in CUR:
    cv = S["curve"][k]
    ax.plot([float(w) for w in cv], list(cv.values()), color=c, lw=2, marker="o", ms=4, label=lab)
ax.axhline(ref, color=DARK, lw=1, ls="--")
ax.text(0.02, ref - 0.13, "no-photo forecast", fontsize=7.5, color=DARK)
ax.set_xlabel("Weight on the no-photo forecast (0 = model alone, 1 = no-photo forecast alone)", fontsize=8)
ax.set_ylim(1.0, 3.0)
ax.legend(fontsize=7.5, loc="upper right")
ax.set_title("Blending a latent model with the no-photo forecast", loc="left", fontsize=9.5, fontweight="bold")
fig.tight_layout()
fig.savefig(HERE / "sy_blend.png", dpi=200)
plt.close(fig)
print("wrote sy_scores.png, sy_blend.png")
