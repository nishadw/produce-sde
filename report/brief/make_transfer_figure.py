import json
from pathlib import Path
import numpy as np, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = Path(__file__).resolve().parents[2] / "results" / "avocado_seeds"


def vals(run):
    return np.array(
        [json.loads(p.read_text())["overall"]["mae"] for p in sorted(S.glob(f"seed*/{run}/metrics_test.json"))]
    )


groups = [
    ("T10", "Tested at 10 °C\n(cool storage)"),
    ("T20", "Tested at 20 °C"),
    ("Tamb", "Tested at\nroom temperature"),
]
series = [
    ("Our model: speed follows temperature", "", "#2a78d6"),
    ("Same model, separate speed per group", "_clock_group", "#b4b2ab"),
    ("Same model, one fixed speed", "_clock_fixed", "#6f6d68"),
]
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
fig, ax = plt.subplots(figsize=(7.2, 3.6))
w = 0.26
for i, (lab, suf, c) in enumerate(series):
    xs = np.arange(3) + (i - 1) * (w + 0.02)
    m = [vals(f"latent_sde_avocado_logo_{g}{suf}").mean() for g, _ in groups]
    bars = ax.bar(xs, m, w, color=c, label=lab)
    for x, v in zip(xs, m):
        ax.text(x, v + 0.12, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color="#333333")
ax.set_xticks(range(3), [t for _, t in groups])
ax.set_ylabel("Average error (days)")
ax.set_ylim(0, 11)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper right", fontsize=9)
ax.set_title("Error at a storage temperature the model never trained on", loc="left", fontsize=10.5, fontweight="bold")
fig.tight_layout()
fig.savefig(Path(__file__).resolve().parent / "transfer.png", dpi=200)
print("ok")
