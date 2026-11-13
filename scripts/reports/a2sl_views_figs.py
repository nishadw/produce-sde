"""Charts + numbers for the plain-language A2SL views / AML summary (ripen env).

Reads outputs/a2sl_views/labels_hidden_std (round 3-4: labels hidden, microwave numbers z-scored) and
outputs/a2sl_aml_std (teacher 0+120+240). Writes a2sl_views.png, a2sl_aml.png and a2sl_numbers.json next to this file.
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[2] / "outputs" / "report_figures"  # charts + numbers live here
HERE.mkdir(parents=True, exist_ok=True)
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
import a2sl_views_summary as vs  # noqa: E402

from ripen.eval.metrics import censored_log_score  # noqa: E402

VIEWS_ROOT = REPO / "outputs" / "a2sl_views" / "labels_hidden_std"
AML_ROOT = REPO / "outputs" / "a2sl_aml_std"
BLUE, LIGHT, DARK = "#2a78d6", "#b4b2ab", "#6f6d68"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#cccccc", "axes.grid": True, "grid.color": "#e6e5e0", "axes.axisbelow": True,
                     "legend.frameon": False})  # fmt: skip


def per_seed(root: Path, run: str, recal: bool = False, tracked: bool = True) -> dict[str, np.ndarray]:
    """Per seed: mean log score, MAE, CRPS, coverage (tracked avocados)."""
    vs.ROOT, vs.RECAL = root, recal
    out = {"log": [], "mae": [], "crps": [], "cov": []}
    for z in vs.pooled(run).values():
        m = np.isin(z["item_ids"], list(vs.TRACKED)) if tracked else np.ones(len(z["lo"]), bool)
        out["log"].append(float(np.mean(censored_log_score(z["F"][m], z["lo"][m], z["hi"][m]))))
        d = vs.compute_metrics(z["F"][m], z["lo"][m], z["hi"][m]).to_dict()
        out["mae"].append(d["mae"])
        out["crps"].append(d["crps"])
        out["cov"].append(d["coverage80"])
    return {k: np.array(v) for k, v in out.items()}


def summ(d):
    return {k: [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0] for k, v in d.items()}


nums = {"views": {}, "aml": {}}
# ---- Chart 1: every view, microwave only and photo + microwave -------------------------------------------------
views = ["0", "120", "240", "60", "180", "300", "0+120+240", "60+180+300", "all", "concat"]
vlab = {"all": "all 6\navg.", "concat": "all 6\nside by\nside", "0+120+240": "0+120\n+240°\navg.",
        "60+180+300": "60+180\n+300°\navg."}  # fmt: skip
ctrl = per_seed(VIEWS_ROOT, "latent_sde_time")
photo = per_seed(VIEWS_ROOT, "latent_sde_photo")
nums["views"]["time"] = summ(ctrl)
nums["views"]["photo"] = summ(photo)
nums["views"]["time_recal"] = summ(per_seed(VIEWS_ROOT, "latent_sde_time", recal=True))
nums["views"]["photo_recal"] = summ(per_seed(VIEWS_ROOT, "latent_sde_photo", recal=True))
fig, ax = plt.subplots(figsize=(7.6, 3.6))
w = 0.38
for i, (inp, lab, c) in enumerate((("mw", "Microwave only", LIGHT), ("photo_mw", "Photo + microwave", BLUE))):
    mu, sd = [], []
    for v in views:
        d = per_seed(VIEWS_ROOT, f"latent_sde_{inp}_view{v}")
        nums["views"][f"{inp}_{v}"] = summ(d)
        nums["views"][f"{inp}_{v}_recal"] = summ(per_seed(VIEWS_ROOT, f"latent_sde_{inp}_view{v}", recal=True))
        mu.append(d["log"].mean())
        sd.append(d["log"].std(ddof=1))
    xs = np.arange(len(views)) + (i - 0.5) * (w + 0.02) + np.where(np.arange(len(views)) >= 6, 0.5, 0.0)
    ax.bar(xs, mu, w, color=c, label=lab, yerr=sd, error_kw={"elinewidth": 1, "capsize": 2})
ax.axhline(ctrl["log"].mean(), color="#555555", lw=1, ls="--")
ax.text(10.05, ctrl["log"].mean(), "no sensor\n(date only)", fontsize=8, color="#555555", va="center")
ax.axhline(photo["log"].mean(), color=DARK, lw=1, ls=":")
ax.text(10.05, photo["log"].mean() - 0.02, "photo only", fontsize=8, color=DARK, va="top")
ticks = np.arange(len(views)) + np.where(np.arange(len(views)) >= 6, 0.5, 0.0)
ax.set_xticks(ticks, [vlab.get(v, f"{v}°") for v in views], fontsize=8)
ax.set_xlim(-0.6, 10.9)
ax.set_ylabel("Forecast score (lower is better)")
lo = min(min(r[0] for k, r in [(k, v["log"]) for k, v in nums["views"].items() if not k.endswith("recal")]) - 0.15, 0.6)
ax.set_ylim(lo, None)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper right", fontsize=8.5, ncol=2, bbox_to_anchor=(1.0, 1.12))
ax.set_title("Single views, then combinations", loc="left", fontsize=10.5, fontweight="bold")
fig.tight_layout()
fig.savefig(HERE / "a2sl_views.png", dpi=200)

# ---- Chart 2: teaching a photo-only model ------------------------------------------------------------------------
V = "0+120+240"
methods = [("photo", "Photo only, no teacher"), ("kd", "Teacher's forecasts (CAML)"),
           ("kd_a05_t1", "Teacher's forecasts, softer (CAML)"), ("feat", "Teacher's internal summary"),
           ("kd_feat", "Both"), ("samd", "SAMD (teacher refreshed)"),
           ("teacher", "The teacher itself:\nphoto + microwave (needs scanner)")]  # fmt: skip
fig, ax = plt.subplots(figsize=(7.2, 3.1))
ys = np.arange(len(methods))[::-1]
for y, (m, lab) in zip(ys, methods, strict=True):
    d = per_seed(AML_ROOT, f"latent_sde_{m}_view{V}")
    nums["aml"][m] = summ(d)
    nums["aml"][m + "_recal"] = summ(per_seed(AML_ROOT, f"latent_sde_{m}_view{V}", recal=True))
    c = DARK if m == "teacher" else LIGHT if m == "photo" else BLUE
    ax.barh(y, d["log"].mean(), 0.62, color=c, xerr=d["log"].std(ddof=1), error_kw={"elinewidth": 1, "capsize": 2})
    ax.text(d["log"].mean() + d["log"].std(ddof=1) + 0.01, y, f"{d['log'].mean():.2f}", va="center", fontsize=9)
ax.set_yticks(ys, [m[1] for m in methods], fontsize=9)
base = nums["aml"]["photo"]["log"][0]
ax.axvline(base, color="#555555", lw=1, ls="--")
ax.set_xlim(base - 0.35, base + 0.35)
ax.set_xlabel("Forecast score (lower is better); every blue bar sees photos only")
ax.grid(axis="y", visible=False)
ax.set_title("Teaching a photo model with the microwave", loc="left", fontsize=10.5, fontweight="bold")
fig.tight_layout()
fig.savefig(HERE / "a2sl_aml.png", dpi=200)
(HERE / "a2sl_numbers.json").write_text(json.dumps(nums, indent=1))
print(json.dumps({k: {kk: vv["log"] for kk, vv in v.items() if not kk.endswith("recal")} for k, v in nums.items()}))
