import json
from pathlib import Path
import numpy as np, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[2] / "results"


def vals(root, run, key, recal=True):
    f = "metrics_test_recal.json" if recal else "metrics_test.json"
    return np.array([json.loads(p.read_text())["overall"][key] for p in sorted((R / root).glob(f"seed*/{run}/{f}"))])


models = [  # label, root, run, recal?, ours?
    ("Our model", "avocado_round2", "latent_sde_avocado_guided_r18", True, True),
    ("ODE-RNN (standard AI model)", "avocado_seeds", "ode_rnn_avocado", True, False),
    ("GRU + time gap (standard AI model)", "avocado_seeds", "gru_gap_avocado", True, False),
    ("GRU-D (standard AI model)", "avocado_seeds", "gru_d_avocado", True, False),
    ("Our model without randomness", "avocado_seeds", "latent_ode_avocado", True, False),
    (
        "Published method\n(classify ripeness, then estimate days)",
        "avocado_seeds",
        "stage_linear_avocado",
        False,
        False,
    ),
][::-1]
plt.rcParams.update(
    {
        "font.size": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#cccccc",
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "axes.axisbelow": True,
    }
)
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), sharey=True, gridspec_kw={"width_ratios": [1, 1]})
ys = np.arange(len(models))
for ax, key, title, scale, fmt in [
    (axes[0], "mae", "Average error (days)", 1, "{:.2f}"),
    (axes[1], "coverage80", "Range hit rate (%)", 100, "{:.0f}%"),
]:
    v = [vals(r, run, key, rc) * scale for _, r, run, rc, _ in models]
    m = [x.mean() for x in v]
    sd = [x.std(ddof=1) for x in v]
    cols = ["#2a78d6" if o else "#b4b2ab" for *_, o in models]
    ax.barh(ys, m, xerr=sd, color=cols, height=0.62, error_kw={"ecolor": "#555", "elinewidth": 1, "capsize": 2})
    for y, a, s in zip(ys, m, sd):
        ax.text(a + s + (0.03 if scale == 1 else 2), y, fmt.format(a), va="center", fontsize=9, color="#333")
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.grid(axis="y", visible=False)
    if scale == 100:
        ax.axvspan(80, 90, color="#1baf7a", alpha=0.12, lw=0)
        ax.set_xlim(0, 112)
    else:
        ax.set_xlim(0, 2.25)
axes[0].set_yticks(ys, [l for l, *_ in models])
fig.tight_layout()
fig.savefig(Path(__file__).resolve().parent / "compare.png", dpi=200)
for l, r, run, rc, _ in models[::-1]:
    print(
        l.replace("\n", " "), [round(vals(r, run, k, rc).mean(), 3) for k in ("mae", "coverage80", "crps", "width80")]
    )
