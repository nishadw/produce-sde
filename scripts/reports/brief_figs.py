"""Charts + numbers for the short presentation brief (ripen env).

Reads views_aml_numbers.json (views_aml_figs.py: latent ODE / SDE by input, mango) and outputs/a2sl_schedule
(5 seeds x 4 folds: photo-only students taught by a photo + scanner teacher). Writes br_*.png and brief_numbers.json
next to this file; brief_report.py (report env) builds results/2026-11-13_results-brief-ode-sde-aml.docx.
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

SCHED = REPO / "outputs" / "a2sl_schedule"
OLD = json.loads((HERE / "views_aml_numbers.json").read_text())
DARK, LIGHT, BLUE, ORANGE = "#6f6d68", "#b4b2ab", "#2a78d6", "#d9822b"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#cccccc", "axes.grid": True, "grid.color": "#e6e5e0", "axes.axisbelow": True,
                     "legend.frameon": False})  # fmt: skip
VIEW = "0+120+240"
P = "sep1+sep2+sep3+sep4+sep5"
MODELS = {"latent_ode": "Latent ODE", "latent_sde": "Latent SDE"}
# avocado input -> views-study key; mango input -> mango run suffix (every earlier scan)
INPUTS = [("Photo", "photo", "rgb_hist"),
          ("Scanner only", f"mw_{VIEW}", f"mw_v{P}_hist"),
          ("Photo + scanner", f"photo_mw_{VIEW}", f"both_v{P}_hist")]  # fmt: skip
AML = ["photo", "reset_only_photoT", "kd", "samd", "ours", "ours_rkd", "teacher"]
MANGO_AML = {"photo": "rgb_hist", "kd": f"kd_v{P}_hist", "samd": f"samd_v{P}_hist", "ours": f"ours_v{P}_hist",
             "ours_rkd": f"ours_rkd_v{P}_hist", "teacher": f"both_v{P}_hist"}  # fmt: skip


def avo(run: str) -> dict | None:
    """Mean / sd over seeds of the tracked-avocado log score and MAE."""
    vs.ROOT, vs.RECAL = SCHED, False
    lg, mae = [], []
    for z in vs.pooled(run).values():
        m = np.isin(z["item_ids"], list(vs.TRACKED))
        lg.append(float(np.mean(censored_log_score(z["F"][m], z["lo"][m], z["hi"][m]))))
        mae.append(vs.compute_metrics(z["F"][m], z["lo"][m], z["hi"][m]).to_dict()["mae"])
    if not lg:
        return None
    return {"log": [float(np.mean(lg)), float(np.std(lg, ddof=1)), len(lg)], "mae": float(np.mean(mae))}


def mango(run: str) -> float | None:
    r = OLD["mango"].get(run)
    return None if r is None else float(r["pop_brier"][0])


N: dict = {"models": {}, "aml": {}, "age_clock": mango("age_clock")}
for m in MODELS:
    for lab, ak, mk in INPUTS:
        a = OLD["avo"][f"{m}_{ak}"]
        N["models"][f"{m}|{lab}"] = {"log": a["log"][0], "mae": a["mae"][0], "mango": mango(f"{m}_{mk}")}
    for meth in AML:
        r = avo(f"{m}_{meth}_view{VIEW}")
        if r is None:
            continue
        vs.ROOT = SCHED
        ref = f"{m}_photo_view{VIEW}"
        r["vs_photo"] = None if meth == "photo" else vs.bootstrap(f"{m}_{meth}_view{VIEW}", ref, True)
        r["mango"] = mango(f"latent_ode_{MANGO_AML[meth]}") if m == "latent_ode" and meth in MANGO_AML else None
        N["aml"][f"{m}|{meth}"] = r

# ---------------------------------------------------------------- chart 1: latent ODE vs SDE by input
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ax, key, title in zip(axes, ("log", "mango"), ("Avocado score", "Mango score"), strict=True):
    for j, (m, lab) in enumerate(MODELS.items()):
        y = [N["models"][f"{m}|{i[0]}"][key] for i in INPUTS]
        x = np.arange(len(INPUTS)) + (j - 0.5) * 0.38
        ax.bar(x, y, 0.36, color=(BLUE, ORANGE)[j], label=lab)
        for xi, yi in zip(x, y, strict=True):
            ax.text(xi, yi, f"{yi:.2f}" if key == "log" else f"{yi:.3f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(len(INPUTS)), [i[0].replace(" ", "\n", 1) for i in INPUTS], fontsize=8)
    ax.set_title(title + " (lower is better)", loc="left", fontsize=9.5, fontweight="bold")
fig.legend(*axes[1].get_legend_handles_labels(), fontsize=8, loc="lower center", ncol=2)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(HERE / "br_models.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- chart 2: teaching a photo-only model
LAB = {"photo": "Photo only", "reset_only_photoT": "Take-turns\ncontrol", "kd": "CAML-style\nKD", "samd": "SAMD",
       "ours": "Lab method", "ours_rkd": "Lab method\n+ RKD", "teacher": "Teacher\n(needs scanner)"}  # fmt: skip
fig, ax = plt.subplots(figsize=(7.2, 2.8))
meths = [k for k in AML if all(f"{m}|{k}" in N["aml"] for m in MODELS)]
for j, (m, lab) in enumerate(MODELS.items()):
    y = [N["aml"][f"{m}|{k}"]["log"][0] for k in meths]
    e = [N["aml"][f"{m}|{k}"]["log"][1] for k in meths]
    x = np.arange(len(meths)) + (j - 0.5) * 0.38
    ax.bar(x, y, 0.36, color=(BLUE, ORANGE)[j], label=lab, yerr=e, capsize=2, error_kw={"lw": 0.7, "color": "#333"})
    for xi, yi, ei in zip(x, y, e, strict=True):
        ax.text(xi, yi + ei + 0.02, f"{yi:.2f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(range(len(meths)), [LAB[k] for k in meths], fontsize=8)
ax.set_title("Avocado score, photo-only student (lower is better)", loc="left", fontsize=9.5, fontweight="bold")
ax.legend(fontsize=8, loc="upper right", ncol=2)
ax.set_ylim(0, max(N["aml"][f"{m}|{k}"]["log"][0] for m in MODELS for k in meths) * 1.35)
fig.tight_layout()
fig.savefig(HERE / "br_aml.png", dpi=200)
plt.close(fig)

(HERE / "brief_numbers.json").write_text(json.dumps(N, indent=1, default=str))
print(json.dumps(N, indent=1, default=str))
