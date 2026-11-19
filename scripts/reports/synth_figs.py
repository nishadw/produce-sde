"""Charts for the half-synthetic brief (ripen env). Reads outputs/a2sl_pop_compare/summary.json
(scripts/analysis/a2sl_pop_compare.py); writes sy_*.png to outputs/report_figures. synth_report.py builds the docx."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "outputs" / "report_figures"
HERE.mkdir(parents=True, exist_ok=True)
S = json.loads((REPO / "outputs" / "a2sl_pop_compare" / "summary.json").read_text())
REF = S["ref"]
DARK, LIGHT, BLUE, ORANGE = "#6f6d68", "#b4b2ab", "#2a78d6", "#d9822b"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#cccccc", "axes.grid": True, "grid.color": "#e6e5e0", "axes.axisbelow": True,
                     "legend.frameon": False})  # fmt: skip


def score(fill, model, sees):
    for r in S["tables"][fill]:
        if r["model"] == model and r["sees"] == sees:
            return r["random"]
    return None


# chart 1: photo-only models, random start, linear vs logistic fill, against the no-photo line
SAMD = "SAMD, teacher sees photo + scanner"
BARS = [("Image matching", "photo", "Image\nmatching"), ("Weibull", "photo", "Weibull"),
        ("Latent ODE", "photo", "Latent ODE"), ("Latent SDE", "photo", "Latent SDE"),
        ("Latent SDE", SAMD, "Latent SDE\n+ SAMD"), ("Transformer", "photo", "Transformer"),
        ("Transformer", "CAML-KD, teacher sees photo only", "Transformer\n+ CAML-KD")]  # fmt: skip
fig, ax = plt.subplots(figsize=(7.2, 2.8))
x = np.arange(len(BARS))
for off, fill, col in ((-0.19, "linear", LIGHT), (0.19, "logistic", BLUE)):
    ys = [score(fill, m, s) for m, s, _ in BARS]
    ax.bar(x + off, [np.nan if y is None else y for y in ys], 0.36, color=col, label=f"{fill} fill")
    for xi, y in zip(x + off, ys, strict=True):
        if y is not None:
            ax.text(xi, y + 0.04, f"{y:.2f}", ha="center", va="bottom", fontsize=6.5)
ax.axhline(REF, color=DARK, lw=1, ls="--")
ax.set_xticks(x, [lab for *_, lab in BARS], fontsize=7.5)
ax.set_ylabel("Score (lower is better)", fontsize=8)
ax.legend(fontsize=7.5, loc="upper left", ncol=2)
ax.set_title(f"Photo-only models, random start; dashed = no-photo forecast {REF:.2f}", loc="left", fontsize=9.5,
             fontweight="bold")  # fmt: skip
ax.set_ylim(0, 3.3)
fig.tight_layout()
fig.savefig(HERE / "sy_scores.png", dpi=200)
plt.close(fig)

# chart 2: blending with the no-photo forecast never beats it
COL = [BLUE, DARK, LIGHT, ORANGE, "#8a5cc2"]
fig, ax = plt.subplots(figsize=(7.2, 2.6))
for (lab, cv), c in zip(S["curve"].items(), COL, strict=False):
    ax.plot([float(w) for w in cv], list(cv.values()), color=c, lw=2, marker="o", ms=3.5, label=lab)
ax.axhline(REF, color=DARK, lw=1, ls="--")
ax.set_xlabel("Share of the no-photo forecast in the mix (0 = model alone, 1 = no-photo forecast alone)", fontsize=8)
ax.set_ylabel("Score", fontsize=8)
ax.set_ylim(1.0, 2.0)
ax.legend(fontsize=7, loc="upper right")
ax.set_title("Mixing a model with the no-photo forecast (linear fill)", loc="left", fontsize=9.5, fontweight="bold")
fig.tight_layout()
fig.savefig(HERE / "sy_blend.png", dpi=200)
plt.close(fig)
print("wrote sy_scores.png, sy_blend.png")
