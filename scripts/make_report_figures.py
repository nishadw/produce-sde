"""Figures for report/REPORT.md (run after scripts/summarize.py and scripts/summarize_seeds.py).

Avocado figures (E1, E3, E4) show mean ± sample std over the independent seeds in
results/avocado_seeds/seed*/<run>/, with one dot per seed; the rest read results/runs/*/metrics_test.json.

python scripts/make_report_figures.py            # writes report/figures/*.png
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
SEEDS = ROOT / "results" / "avocado_seeds"
ROUND2 = ROOT / "results" / "avocado_round2"
OUT = ROOT / "report" / "figures"
BEFORE = ROOT / "report" / "before_fix"

# reference categorical slots 1-3 (validated all-pairs), muted gray for context bars, text inks
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, INK2, GRID = "#b4b2ab", "#0b0b0b", "#52514e", "#e6e5e0"
NAMES = {
    "stage_linear": "stage→linear (published)",
    "direct_reg": "direct regressor",
    "discrete_surv": "discrete survival",
    "gru_gap": "GRU + gap",
    "gru_d": "GRU-D",
    "ode_rnn": "ODE-RNN",
    "latent_ode": "latent ODE",
    "latent_sde": "latent SDE (ours)",
    "fpt_filter": "FPT filter (ours)",
}
MODELS = list(NAMES)


def style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "legend.frameon": False,
        }
    )


def metrics(run: str, split: str = "overall", path: Path | None = None) -> dict | None:
    p = path or RUNS / run / "metrics_test.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get(split)


def seed_vals(run: str, key: str, split: str = "overall") -> np.ndarray:
    """One value per seed for an avocado run (NaN-free); falls back to the single results/runs value."""
    paths = sorted(SEEDS.glob(f"seed*/{run}/metrics_test.json")) or [RUNS / run / "metrics_test.json"]
    v = [json.loads(p.read_text()).get(split, {}).get(key, float("nan")) for p in paths if p.exists()]
    return np.array([x for x in v if x == x], float)


def mean_sd(v: np.ndarray) -> tuple[float, float]:
    if len(v) == 0:
        return float("nan"), 0.0
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def seed_dots(ax, xs, vals_list, horizontal=False):
    """Overlay one small dot per seed, jittered across the bar."""
    rng = np.random.default_rng(0)
    for x, v in zip(xs, vals_list, strict=True):
        j = x + rng.uniform(-0.08, 0.08, len(v))
        if horizontal:
            ax.scatter(v, j, s=9, color=INK, alpha=0.55, zorder=3, linewidths=0)
        else:
            ax.scatter(j, v, s=9, color=INK, alpha=0.55, zorder=3, linewidths=0)


def label_bars(ax, bars, fmt="{:.2f}", horizontal=True):
    for b in bars:
        v = b.get_width() if horizontal else b.get_height()
        if horizontal:
            ax.text(v, b.get_y() + b.get_height() / 2, " " + fmt.format(v), va="center", fontsize=8, color=INK2)
        else:
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=8, color=INK2)


def fig_e1():
    rows = [(m, f"{m}_avocado") for m in MODELS if len(seed_vals(f"{m}_avocado", "mae"))]
    names = [NAMES[m] for m, _ in rows][::-1]
    colors = [BLUE if m in ("latent_sde", "fpt_filter") else GRAY for m, _ in rows][::-1]
    n_seeds = max(len(seed_vals(r, "mae")) for _, r in rows)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), sharey=True)
    for ax, key, title, fmt in [
        (axes[0], "crps", "CRPS (days) ↓", "{:.2f}"),
        (axes[1], "coverage80", "80% coverage (dashed: target)", "{:.2f}"),
        (axes[2], "cindex", "C-index ↑", "{:.3f}"),
    ]:
        vals = [seed_vals(r, key) for _, r in rows][::-1]
        ms = [mean_sd(v) for v in vals]
        ys = np.arange(len(names))
        ax.barh(ys, [m for m, _ in ms], xerr=[s for _, s in ms], color=colors, height=0.6,
                error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})  # fmt: skip
        seed_dots(ax, ys, vals, horizontal=True)
        for y, (m, sd) in zip(ys, ms, strict=True):
            ax.text(m + sd, y, " " + fmt.format(m), va="center", fontsize=8, color=INK2)
        ax.set_yticks(ys, names)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        if key == "coverage80":
            ax.axvline(0.8, color=INK2, lw=1, ls="--")
            ax.set_xlim(0, 1.15)
        if key == "cindex":
            ax.set_xlim(0.5, 1.0)
    fig.suptitle(f"E1 · Mendeley avocado test set · mean ± std over {n_seeds} seeds (dots: each seed)",
                 x=0.01, ha="left", fontweight="bold")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "e1_avocado.png")
    plt.close(fig)


def _grouped_bars(ax, series, groups, w=0.26):
    k = len(series)
    for i, (lab, pat, c) in enumerate(series):
        vals = [seed_vals(pat.format(g=g), "mae") for g in groups]
        ms = [mean_sd(v) for v in vals]
        xs = [j + (i - (k - 1) / 2) * (w + 0.02) for j in range(len(groups))]
        ax.bar(xs, [m for m, _ in ms], w, yerr=[s for _, s in ms], color=c, label=lab,
               error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})  # fmt: skip
        seed_dots(ax, xs, vals)
        for x, (m, sd) in zip(xs, ms, strict=True):
            ax.text(x, m + sd, f"{m:.1f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(len(groups)), [f"held out: {g}" for g in groups])
    ax.set_ylabel("MAE on held-out group (days)")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.legend(loc="upper center", ncol=len(series), bbox_to_anchor=(0.5, 1.0), fontsize=8.5)


def fig_transfer():
    groups = ["T10", "T20", "Tamb"]
    fig, ax = plt.subplots(figsize=(7.5, 3.9))
    _grouped_bars(ax, [
        ("Arrhenius clock (temperature)", "latent_sde_avocado_logo_{g}", BLUE),
        ("per-group clock", "latent_sde_avocado_logo_{g}_clock_group", ORANGE),
        ("fixed clock", "latent_sde_avocado_logo_{g}_clock_fixed", AQUA),
    ], groups)  # fmt: skip
    ax.set_title("E3 · Clock ablation, latent SDE (mean ± std, 5 seeds)", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "e3_transfer.png")
    plt.close(fig)

    # same held-out groups, our two models vs the black-box baselines (gray = baselines)
    fig, ax = plt.subplots(figsize=(7.5, 3.9))
    _grouped_bars(ax, [
        ("latent SDE (ours)", "latent_sde_avocado_logo_{g}", BLUE),
        ("FPT filter (ours)", "fpt_filter_avocado_logo_{g}", "#174f91"),
        ("GRU-D", "gru_d_avocado_logo_{g}", GRAY),
        ("direct regressor", "direct_reg_avocado_logo_{g}", "#dcdad3"),
    ], groups, w=0.2)  # fmt: skip
    ax.set_title("E3 · Ours vs black-box baselines (mean ± std, 5 seeds)", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "e3_vs_baselines.png")
    plt.close(fig)


def fig_sampling():
    series = [("GRU-D", "gru_d", GRAY), ("latent ODE", "latent_ode", ORANGE), ("latent SDE (ours)", "latent_sde", BLUE)]
    xs = ["daily", "every 2nd day", "every 3rd day"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, title in [(axes[0], "crps", "CRPS (days) ↓"), (axes[1], "coverage80", "80% coverage")]:
        for lab, m, c in series:
            runs = [f"{m}_avocado", f"{m}_avocado_every2", f"{m}_avocado_every3"]
            ms = [mean_sd(seed_vals(r, key)) for r in runs]
            mu, sd = np.array([a for a, _ in ms]), np.array([b for _, b in ms])
            ax.fill_between(range(3), mu - sd, mu + sd, color=c, alpha=0.15, linewidth=0)
            ax.plot(range(3), mu, color=c, lw=2, marker="o", ms=7, mec="white", mew=1.5, label=lab)
            ax.annotate(lab, (2, mu[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8,
                        color=INK2)  # fmt: skip
        ax.set_xticks(range(3), xs)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
        ax.set_xlim(-0.2, 2.9)
        if key == "coverage80":
            ax.axhline(0.8, color=INK2, lw=1, ls="--")
    axes[0].legend(loc="upper left")
    fig.suptitle("E4 · Training on sparser photo schedules (mean, band = ± std over 5 seeds)", x=0.01, ha="left",
                 fontweight="bold")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "e4_sampling.png")
    plt.close(fig)


def fig_photo_count():
    series = [("GRU-D", "gru_d", GRAY), ("latent ODE", "latent_ode", ORANGE), ("latent SDE (ours)", "latent_sde", BLUE)]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for lab, m, c in series:
        paths = sorted(SEEDS.glob(f"seed*/{m}_avocado/metrics_by_n_obs.json")) or [
            RUNS / f"{m}_avocado" / "metrics_by_n_obs.json"
        ]
        ds = [json.loads(p.read_text()) for p in paths if p.exists()]
        if not ds:
            continue
        ks = sorted(int(k) for k in ds[0] if k.isdigit() and ds[0][k]["n"] >= 20)[:12]
        mu = np.array([np.mean([d[str(k)]["crps"] for d in ds if str(k) in d]) for k in ks])
        per_k = [[d[str(k)]["crps"] for d in ds if str(k) in d] for k in ks]
        sd = np.array([np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in per_k])
        ax.fill_between(ks, mu - sd, mu + sd, color=c, alpha=0.15, linewidth=0)
        ax.plot(ks, mu, color=c, lw=2, marker="o", ms=6, mec="white", mew=1.2, label=lab)
    ax.set_xlabel("photos seen so far")
    ax.set_ylabel("CRPS (days) ↓")
    ax.legend()
    ax.set_title("E4 · Forecast error vs. number of photos observed (mean ± std, 5 seeds)", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "e4_photo_count.png")
    plt.close(fig)


def fig_pbc2_fix():
    before = {
        "latent ODE": metrics("", path=BEFORE / "latent_ode_pbc2.json"),
        "GRU-D": {"cindex": 0.5, "mae": float("nan")},  # NaN predictions before the fix (C reported as 0.5)
    }
    after = {"latent ODE": metrics("latent_ode_pbc2"), "GRU-D": metrics("gru_d_pbc2")}
    ref = {n: metrics(f"{m}_pbc2") for n, m in [("ODE-RNN", "ode_rnn"), ("discrete survival", "discrete_surv")]}
    names = ["latent ODE", "GRU-D"]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    w = 0.36
    b1 = ax.bar(
        [i - w / 2 - 0.01 for i in range(2)], [before[n]["cindex"] for n in names], w, color=GRAY, label="before fix"
    )
    b2 = ax.bar(
        [i + w / 2 + 0.01 for i in range(2)], [after[n]["cindex"] for n in names], w, color=BLUE, label="after fix"
    )
    label_bars(ax, list(b1) + list(b2), "{:.2f}", horizontal=False)
    # reference lines stop short of their labels; the two reference labels sit above / below their lines
    for (n, r), dy in zip(ref.items(), (-10, 6), strict=True):
        if r:
            ax.hlines(r["cindex"], -0.6, 1.5, color=INK2, lw=1, ls=":")
            ax.annotate(
                f"{n} {r['cindex']:.2f}", (1.52, r["cindex"]), xytext=(0, dy), textcoords="offset points",
                va="center", fontsize=8, color=INK2,
            )  # fmt: skip
    ax.hlines(0.5, -0.6, 1.5, color=INK2, lw=1, ls="--")
    ax.text(1.52, 0.5, "chance", va="center", fontsize=8, color=INK2)
    ax.set_xticks(range(2), names)
    ax.set_xlim(-0.6, 2.2)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("C-index on PBC2 test ↑")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    ax.set_title("E5 · PBC2 (liver disease) before and after the two bug fixes", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "e5_pbc2_fix.png")
    plt.close(fig)


def fig_synthetic_fix():
    """E0: fitted rates and sigma, unguided vs guided posterior, against the simulator's truth."""
    paths = {
        k: ROOT / "results" / d / "recovery.json"
        for k, d in (("unguided", "synthetic"), ("guided", "synthetic_guided"))
    }
    if not all(p.exists() for p in paths.values()):
        return
    rec = {k: json.loads(p.read_text()) for k, p in paths.items()}
    groups = list(rec["unguided"]["rates_true"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [3, 1.2]})
    ys = np.arange(len(groups))[::-1]
    for lab, key, c, dy in (
        ("unguided posterior (before)", "unguided", GRAY, 0.12),
        ("guided posterior (after)", "guided", BLUE, -0.12),
    ):
        fit = [rec[key]["rates_fit"][g] / rec[key]["rates_true"][g] - 1 for g in groups]
        ax.scatter(np.array(fit) * 100, ys + dy, s=60, color=c, zorder=3, label=lab, edgecolors="white", linewidths=1.2)
        for y, v in zip(ys + dy, fit, strict=True):
            ax.text(v * 100, y, f"  {v * 100:+.0f}%", va="center", fontsize=8, color=INK2)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(ys, [f"{g} (true {rec['guided']['rates_true'][g]:.3f}/day)" for g in groups])
    ax.set_xlabel("fitted rate error vs truth (%)")
    ax.set_xlim(-5, 32)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Clock rates", loc="left")
    sig = [rec["unguided"]["sigma_fit"], rec["guided"]["sigma_fit"]]
    ax2.bar([0, 1], sig, color=[GRAY, BLUE], width=0.55)
    ax2.axhline(rec["guided"]["sigma_true"], color=INK, lw=1, ls="--")
    for x, v in enumerate(sig):
        ax2.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax2.set_xticks([0, 1], ["before", "after"])
    ax2.set_ylim(0, 0.35)
    ax2.grid(axis="x", visible=False)
    ax2.set_title("σ (dashed: true 0.25)", loc="left")
    fig.suptitle("E0 · Synthetic recovery, 600 fake fruit: the rate bias before and after the fix", x=0.01, ha="left",
                 fontweight="bold")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "synthetic_fix.png")
    plt.close(fig)


def copy_existing():
    for src, dst in [
        (SEEDS / "seed1" / "latent_sde_avocado" / "reliability_test.png", "latent_sde_reliability.png"),
        (SEEDS / "seed1" / "latent_sde_avocado" / "pred_vs_true_test.png", "latent_sde_pred_vs_true.png"),
        (ROOT / "results" / "synthetic" / "recovery.png", "synthetic_recovery.png"),
    ]:
        if src.exists():
            shutil.copy(src, OUT / dst)


def _vals(root: Path, run: str, key: str = "crps", split: str = "overall") -> np.ndarray:
    v = [
        json.loads(p.read_text()).get(split, {}).get(key, float("nan"))
        for p in sorted(root.glob(f"seed*/{run}/metrics_test.json"))
    ]
    return np.array([x for x in v if x == x], float)


def fig_round2():
    """Round 2: which change moved the latent SDE, and what DINOv2 did to every other model (in-distribution CRPS)."""
    if not ROUND2.exists():
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1, 1.25]})
    abl = [
        ("Round 1 (ResNet-18, unguided)", SEEDS, "latent_sde_avocado", GRAY),
        ("+ DINOv2 only", ROUND2, "latent_sde_avocado_dino_unguided", GRAY),
        ("+ guidance only", ROUND2, "latent_sde_avocado_guided_r18", BLUE),
        ("+ both (Round 2)", ROUND2, "latent_sde_avocado", BLUE),
    ][::-1]
    ys = np.arange(len(abl))
    vals = [_vals(r, run) for _, r, run, _ in abl]
    ms = [mean_sd(v) for v in vals]
    a.barh(ys, [m for m, _ in ms], xerr=[s for _, s in ms], color=[c for *_, c in abl], height=0.6,
           error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})  # fmt: skip
    seed_dots(a, ys, vals, horizontal=True)
    for y, (m, sd) in zip(ys, ms, strict=True):
        a.text(m + sd + 0.01, y, f" {m:.2f}", va="center", fontsize=8, color=INK2)
    best = mean_sd(_vals(SEEDS, "gru_gap_avocado"))[0]
    a.axvline(best, color=INK2, lw=1, ls="--")
    a.set_ylim(-0.95, len(abl) - 0.5)
    a.text(best, -0.8, "GRU + gap, Round 1 ", fontsize=8, color=INK2, va="center", ha="right")
    a.set_yticks(ys, [n for n, *_ in abl])
    a.set_xlim(0.6, 1.1)
    a.grid(axis="y", visible=False)
    a.set_title("Latent SDE: CRPS (days) ↓ by change", loc="left")

    models = ["gru_gap", "ode_rnn", "gru_d", "fpt_filter", "discrete_surv", "latent_ode"]
    w = 0.38
    xs = np.arange(len(models))
    for i, (lab, root, c) in enumerate([("ResNet-18 (Round 1)", SEEDS, GRAY), ("DINOv2 (Round 2)", ROUND2, ORANGE)]):
        v = [_vals(root, f"{m}_avocado") for m in models]
        mm = [mean_sd(x) for x in v]
        pos = xs + (i - 0.5) * (w + 0.02)
        b.bar(pos, [m for m, _ in mm], w, yerr=[s for _, s in mm], color=c, label=lab,
              error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})  # fmt: skip
        seed_dots(b, pos, v)
    b.set_xticks(xs, [NAMES[m].replace(" (ours)", "").replace(" ", "\n", 1) for m in models], fontsize=8.5)
    b.set_ylim(0.6, 1.2)
    b.grid(axis="x", visible=False)
    b.legend(loc="upper left", fontsize=8.5, ncol=2)
    b.set_title("Other models: CRPS (days) ↓ by photo features", loc="left")
    fig.suptitle("Round 2 · avocado test set · mean ± std over 5 seeds (dots: each seed)", x=0.01, ha="left",
                 fontweight="bold")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "round2_ablation.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    fig_e1()
    fig_transfer()
    fig_sampling()
    fig_photo_count()
    fig_pbc2_fix()
    fig_synthetic_fix()
    fig_round2()
    copy_existing()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
