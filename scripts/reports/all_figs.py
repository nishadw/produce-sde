"""Charts for the all-fruit brief (ripen env). Reads outputs/a2sl_all_compare/summary.json
(scripts/analysis/a2sl_all_compare.py); writes al_*.png to outputs/report_figures. all_report.py builds the docx."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "outputs" / "report_figures"
HERE.mkdir(parents=True, exist_ok=True)
S = json.loads((REPO / "outputs" / "a2sl_all_compare" / "summary.json").read_text())
REF = S["ref_obs"]
DARK, LIGHT, BLUE, ORANGE = "#6f6d68", "#b4b2ab", "#2a78d6", "#d9822b"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#cccccc", "axes.grid": True, "grid.color": "#e6e5e0", "axes.axisbelow": True,
                     "legend.frameon": False})  # fmt: skip


def get(model, sees):
    return next((r for r in S["rows"] if r["model"] == model and r["sees"] == sees), None)


def label(model, sees):
    short = {"photo": "", "date only": "\n(date only)", "SAMD, teacher sees photo + scanner": "\n+ SAMD",
             "CAML-KD, teacher sees photo + scanner": "\n+ CAML-KD", "CAML-KD, teacher sees photo only": "\n+ CAML-KD*",
             "SAMD, teacher sees photo only": "\n+ SAMD*"}  # fmt: skip
    return model.replace("Image matching", "Image\nmatching") + short.get(sees, "")


# chart 1: the best setting of each model (8 observed fruit) against the no-photo line
cands = [r for r in S["rows"] if "scanner at test" not in r["sees"] and r["model"] != "No-photo forecast"]
best = {}
for r in cands:
    if r["model"] not in best or r["obs"] < best[r["model"]]["obs"]:
        best[r["model"]] = r
rows = sorted(best.values(), key=lambda r: r["obs"])
fig, ax = plt.subplots(figsize=(7.2, 2.8))
y = [r["obs"] for r in rows]
ax.bar(range(len(rows)), y, 0.6, color=[BLUE if v < REF else LIGHT for v in y])
for i, v in enumerate(y):
    ax.text(i, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
ax.axhline(REF, color=DARK, lw=1, ls="--")
ax.set_xticks(range(len(rows)), [label(r["model"], r["sees"]) for r in rows], fontsize=7.5)
ax.set_ylabel("Score (lower is better)", fontsize=8)
ax.set_ylim(0, max(y) * 1.2)
ax.set_title(f"Each model's best setting, 8 observed fruit; dashed = no-photo forecast {REF:.2f}", loc="left",
             fontsize=9.5, fontweight="bold")  # fmt: skip
fig.tight_layout()
fig.savefig(HERE / "al_scores.png", dpi=200)
plt.close(fig)

# chart 2: blend curves
COL = [BLUE, DARK, LIGHT, ORANGE, "#8a5cc2"]
fig, ax = plt.subplots(figsize=(7.2, 2.6))
top = REF
for (lab, cv), c in zip(S["curve"].items(), COL, strict=False):
    v = list(cv.values())
    top = max(top, min(max(v), 3.0))
    ax.plot([float(w) for w in cv], v, color=c, lw=2, marker="o", ms=3.5, label=lab)
ax.axhline(REF, color=DARK, lw=1, ls="--")
ax.set_xlabel("Share of the no-photo forecast in the mix (0 = model alone, 1 = no-photo forecast alone)", fontsize=8)
ax.set_ylabel("Score", fontsize=8)
ax.set_ylim(max(0.0, np.floor((min(REF, *(min(c.values()) for c in S["curve"].values())) - 0.1) * 10) / 10), top + 0.1)
ax.legend(fontsize=7, loc="upper right")
ax.set_title("Mixing a model with the no-photo forecast (8 observed fruit)", loc="left", fontsize=9.5,
             fontweight="bold")  # fmt: skip
fig.tight_layout()
fig.savefig(HERE / "al_blend.png", dpi=200)
plt.close(fig)
print("wrote al_scores.png, al_blend.png")
