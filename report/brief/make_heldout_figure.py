import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[2] / "results"


def vals(root, run, key):
    ps = sorted((R / root).glob(f"seed*/{run}/metrics_test.json"))
    return np.array([json.loads(p.read_text())["overall"][key] for p in ps])


groups = [("T10", "Tested at 10 °C\n(colder than training)"), ("T20", "Tested at 20 °C"),
          ("Tamb", "Tested at\nroom temperature")]  # fmt: skip
series = [  # label, root, run pattern, color
    ("Our model", "avocado_round2", "latent_sde_avocado_logo_{g}_guided_r18", "#2a78d6"),
    ("Transformer", "baselines2", "transformer_avocado_logo_{g}", "#b4b2ab"),
    ("GRU-D", "avocado_seeds", "gru_d_avocado_logo_{g}", "#8a8883"),
    ("Wiener process (classic)", "baselines2", "wiener_avocado_logo_{g}", "#e08a2c"),
]
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#cccccc", "axes.grid": True, "grid.color": "#e6e5e0", "axes.axisbelow": True})  # fmt: skip
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
x = np.arange(len(groups))
w = 0.8 / len(series)
for ax, key, title, scale in [(axes[0], "mae", "Average error (days)", 1),
                              (axes[1], "coverage80", "Range hit rate (%)", 100)]:  # fmt: skip
    for i, (lab, root, pat, c) in enumerate(series):
        v = [vals(root, pat.format(g=g), key) * scale for g, _ in groups]
        m, sd = [a.mean() for a in v], [a.std(ddof=1) for a in v]
        xs = x - 0.4 + w * (i + 0.5)
        ax.bar(xs, m, w * 0.92, yerr=sd, color=c, label=lab, error_kw={"ecolor": "#555", "elinewidth": 1, "capsize": 2})
        for xx, a, s in zip(xs, m, sd, strict=True):
            ax.text(xx, a + s + (0.15 if scale == 1 else 2), f"{a:.1f}" if scale == 1 else f"{a:.0f}",
                    ha="center", fontsize=7.5, color="#333")  # fmt: skip
        print(lab, key, [round(a, 2) for a in m])
    ax.set_xticks(x, [lab for _, lab in groups])
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.grid(axis="x", visible=False)
    if scale == 100:
        ax.axhspan(80, 90, color="#1baf7a", alpha=0.12, lw=0)
        ax.set_ylim(0, 112)
axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
fig.tight_layout()
fig.savefig(Path(__file__).resolve().parent / "heldout.png", dpi=200)
