"""Charts + numbers for the short presentation brief (ripen env).

Reads views_aml_numbers.json (views_aml_figs.py: latent ODE / SDE by input, mango) and outputs/a2sl_schedule
(5 seeds x 4 folds: photo-only students taught by a photo + scanner teacher). Writes br_*.png and brief_numbers.json
to outputs/report_figures; brief_report.py (report env) builds ../results/2026-11-13_results-brief-ode-sde-aml.docx.
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

# ---------------------------------------------------------------- chart 1: one model per scanner view
VIEWS = ["0", "60", "120", "180", "240", "300"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
for ax, (m, lab) in zip(axes, MODELS.items(), strict=True):
    rows = [OLD["avo"][f"{m}_mw_{v}"]["log"] for v in VIEWS]
    best = int(np.argmin([r[0] for r in rows]))
    ax.bar(range(6), [r[0] for r in rows], 0.6, yerr=[r[1] for r in rows], capsize=2,
           color=[BLUE if i == best else LIGHT for i in range(6)], error_kw={"lw": 0.7, "color": "#333"})
    for i, r in enumerate(rows):
        ax.text(i, 0.52, f"{r[0]:.2f}", ha="center", va="bottom", fontsize=7, color="white" if i == best else "#333")
    for key, ls, txt in (("time", "--", "date only"), ("photo", ":", "photo only")):
        y = OLD["avo"][f"{m}_{key}"]["log"][0]
        ax.axhline(y, color=DARK, lw=1, ls=ls)
        ax.text(5.5, y, txt, fontsize=7, color=DARK, va="center")
    ax.set_xticks(range(6), [v + "°" for v in VIEWS], fontsize=8)
    ax.set_xlim(-0.6, 6.6)
    ax.set_ylim(0.5, 1.2)
    ax.set_title(f"{lab}: scanner view alone", loc="left", fontsize=9.5, fontweight="bold")
axes[0].set_ylabel("Avocado score (lower is better)", fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "br_views.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- chart 2: teaching a photo-only latent ODE
LAB = {"photo": "Photo only", "kd": "CAML-style\nKD", "samd": "SAMD", "ours": "Lab method",
       "reset_only_photoT": "Take-turns\n(no scanner)", "teacher": "Teacher\n(needs scanner)"}  # fmt: skip
fig, ax = plt.subplots(figsize=(7.2, 2.5))
meths = [k for k in LAB if f"latent_ode|{k}" in N["aml"]]
y = [N["aml"][f"latent_ode|{k}"]["log"][0] for k in meths]
e = [N["aml"][f"latent_ode|{k}"]["log"][1] for k in meths]
cols = [DARK if k == "photo" else LIGHT if k in ("teacher", "reset_only_photoT") else BLUE for k in meths]
ax.bar(range(len(meths)), y, 0.6, color=cols, yerr=e, capsize=2, error_kw={"lw": 0.7, "color": "#333"})
for xi, (yi, ei) in enumerate(zip(y, e, strict=True)):
    ax.text(xi, yi + ei + 0.02, f"{yi:.2f}", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(range(len(meths)), [LAB[k] for k in meths], fontsize=8)
ax.set_title("Latent ODE, photo-only student (lower is better)", loc="left", fontsize=9.5, fontweight="bold")
ax.set_ylim(0, max(y) * 1.3)
fig.tight_layout()
fig.savefig(HERE / "br_aml.png", dpi=200)
plt.close(fig)

(HERE / "brief_numbers.json").write_text(json.dumps(N, indent=1, default=str))
print(json.dumps(N, indent=1, default=str))
