"""Charts + numbers for the plain-language views / time-series / AML report, avocados and mangoes (ripen env).

Reads outputs/a2sl_views/labels_hidden_std, outputs/a2sl_aml_std, outputs/mango_views(_recal). Writes
va_*.png and views_aml_numbers.json next to this file; views_aml_report.py (report env) builds the docx from them.
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
import mango_summary as msum  # noqa: E402

from ripen.eval.metrics import censored_log_score  # noqa: E402

AVO = REPO / "outputs" / "a2sl_views" / "labels_hidden_std"
AML = REPO / "outputs" / "a2sl_aml_std"
MANGO = REPO / "outputs" / "mango_views"
BLUE, LIGHT, DARK, ORANGE = "#2a78d6", "#b4b2ab", "#6f6d68", "#d9822b"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#cccccc", "axes.grid": True, "grid.color": "#e6e5e0", "axes.axisbelow": True,
                     "legend.frameon": False})  # fmt: skip
N: dict = {"avo": {}, "aml": {}, "grid": {}, "mango": {}, "mango_recal": {}}


# ---------------------------------------------------------------- avocado helpers
def avo(root: Path, run: str, recal: bool = False) -> dict | None:
    """Mean/sd over seeds on the tracked avocados: log score, MAE, CRPS, 80 % coverage."""
    vs.ROOT, vs.RECAL = root, recal
    out = {"log": [], "mae": [], "crps": [], "cov": []}
    for z in vs.pooled(run).values():
        m = np.isin(z["item_ids"], list(vs.TRACKED))
        out["log"].append(float(np.mean(censored_log_score(z["F"][m], z["lo"][m], z["hi"][m]))))
        d = vs.compute_metrics(z["F"][m], z["lo"][m], z["hi"][m]).to_dict()
        out["mae"].append(d["mae"])
        out["crps"].append(d["crps"])
        out["cov"].append(d["coverage80"])
    if not out["log"]:
        return None
    return {k: [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v)] for k, v in out.items()}


def avo_store(root, run, key, bucket="avo"):
    r = avo(root, run)
    if r is None:
        return None
    N[bucket][key] = r
    N[bucket][key + "_recal"] = avo(root, run, recal=True)
    N[bucket][key + "_vs_ref"] = None
    return r


def avo_boot(root, a, b):
    vs.ROOT, vs.RECAL = root, False
    return vs.bootstrap(a, b, True)


# ---------------------------------------------------------------- Avocado: views x models
AVO_MODELS = {"latent_sde": "latent SDE", "latent_ode": "latent ODE"}
AVO_VIEWS = ["0", "120", "240", "60", "180", "300", "0+120+240", "60+180+300", "all", "concat", "fuse-mean",
             "fuse-geo", "fuse-trust"]  # fmt: skip
for m in AVO_MODELS:
    for inp in ("time", "photo"):
        avo_store(AVO, f"{m}_{inp}", f"{m}_{inp}")
    for inp in ("mw", "photo_mw"):
        for v in AVO_VIEWS:
            r = avo_store(AVO, f"{m}_{inp}_view{v}", f"{m}_{inp}_{v}")
            if r is not None:
                N["avo"][f"{m}_{inp}_{v}_boot"] = avo_boot(AVO, f"{m}_{inp}_view{v}", f"{m}_time")

vlab = {"all": "all 6\navg.", "concat": "all 6\nside by\nside", "0+120+240": "0+120\n+240°", "60+180+300": "60+180\n+300°",
        "fuse-mean": "6 models\ncombined"}  # fmt: skip
views = ["0", "120", "240", "60", "180", "300", "0+120+240", "60+180+300", "all", "fuse-mean"]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), sharey=True)
for ax, m in zip(axes, ("latent_sde", "latent_ode"), strict=True):
    w = 0.38
    for i, (inp, lab, c) in enumerate((("mw", "Microwave only", LIGHT), ("photo_mw", "Photo + microwave", BLUE))):
        mu = [N["avo"].get(f"{m}_{inp}_{v}", {"log": [np.nan, 0]})["log"][0] for v in views]
        sd = [N["avo"].get(f"{m}_{inp}_{v}", {"log": [np.nan, 0]})["log"][1] for v in views]
        xs = np.arange(len(views)) + (i - 0.5) * (w + 0.02) + np.where(np.arange(len(views)) >= 6, 0.5, 0.0)
        ax.bar(xs, mu, w, color=c, label=lab, yerr=sd, error_kw={"elinewidth": 1, "capsize": 2})
    t, p = N["avo"][f"{m}_time"]["log"][0], N["avo"][f"{m}_photo"]["log"][0]
    ax.axhline(t, color="#555555", lw=1, ls="--")
    ax.axhline(p, color=DARK, lw=1, ls=":")
    ax.text(len(views) + 0.45, t, "date\nonly", fontsize=7.5, color="#555555", va="center")
    ax.text(len(views) + 0.45, p, "photo\nonly", fontsize=7.5, color=DARK, va="center")
    ticks = np.arange(len(views)) + np.where(np.arange(len(views)) >= 6, 0.5, 0.0)
    ax.set_xticks(ticks, [vlab.get(v, f"{v}°") for v in views], fontsize=7)
    ax.set_xlim(-0.6, len(views) + 1.2)
    ax.grid(axis="x", visible=False)
    ax.set_title(AVO_MODELS[m], loc="left", fontsize=10.5, fontweight="bold")
axes[0].set_ylabel("Forecast score (lower is better)")
lo = np.nanmin([v["log"][0] for k, v in N["avo"].items() if isinstance(v, dict) and "log" in v
                and k.startswith(("latent_sde", "latent_ode")) and not k.endswith("_recal")])  # fmt: skip
axes[0].set_ylim(max(lo - 0.2, 0), None)
axes[1].legend(loc="upper right", fontsize=8, ncol=1)
fig.tight_layout()
fig.savefig(HERE / "va_avo_views.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Avocado: AML
AML_METHODS = [("photo", "Photo only, no teacher"), ("kd", "Teacher's forecasts (KD)"),
               ("feat", "Teacher's internal summary"), ("rkd", "Distances between fruit (RKD)"),
               ("pkt", "Neighbour pattern (PKT)"), ("sp", "Similarity pattern (SP)"),
               ("samd", "SAMD: teacher reset each round"), ("ours", "Lab method: reset + KD"),
               ("ours_rkd", "Reset + RKD"), ("ours_pkt", "Reset + PKT"), ("ours_sp", "Reset + SP"),
               ("kd_paired", "KD, scanned fruit only"), ("samd_paired", "SAMD, scanned fruit only"),
               ("ours_rkd_paired", "Reset + RKD, scanned fruit only"),
               ("reset_only", "Control: reset schedule, no teaching term"),
               ("samd_photoT", "Control: SAMD, teacher sees photos only"),
               ("ours_rkd_photoT", "Control: reset + RKD, teacher photos only"),
               ("teacher", "The teacher: photo + microwave\n(needs the scanner)")]  # fmt: skip
V = "0+120+240"
for m in ("latent_sde", "latent_ode"):
    for meth, _ in AML_METHODS:
        run = f"{m}_{meth}_view{V}"
        r = avo_store(AML, run, f"{m}_{meth}", bucket="aml")
        if r is not None and meth != "photo":
            N["aml"][f"{m}_{meth}_boot"] = avo_boot(AML, run, f"{m}_photo_view{V}")

fig, axes = plt.subplots(1, 2, figsize=(9.8, 5.8), sharey=True)
ys = np.arange(len(AML_METHODS))[::-1]
for ax, m in zip(axes, ("latent_sde", "latent_ode"), strict=True):
    base = N["aml"][f"{m}_photo"]["log"][0]
    for y, (meth, _) in zip(ys, AML_METHODS, strict=True):
        d = N["aml"].get(f"{m}_{meth}")
        if d is None:
            continue
        sig = "*" in (N["aml"].get(f"{m}_{meth}_boot") or "") and d["log"][0] < base
        c = DARK if meth == "teacher" else LIGHT if meth == "photo" else ORANGE if sig else BLUE
        ax.barh(y, d["log"][0], 0.62, color=c, xerr=d["log"][1], error_kw={"elinewidth": 1, "capsize": 2})
        ax.text(d["log"][0] + d["log"][1] + 0.01, y, f"{d['log'][0]:.2f}", va="center", fontsize=8)
    ax.axvline(base, color="#555555", lw=1, ls="--")
    ax.set_xlim(base - 0.5, base + 0.45)
    ax.grid(axis="y", visible=False)
    ax.set_title(AVO_MODELS[m], loc="left", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Forecast score (lower is better)")
axes[0].set_yticks(ys, [lab for _, lab in AML_METHODS], fontsize=8.5)
fig.tight_layout()
fig.savefig(HERE / "va_avo_aml.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Avocado: small tuning grid, chosen on validation
vs.ROOT, vs.RECAL = AML, False
for meth in ("ours_rkd", "samd_paired"):
    rows = []
    for b in ("01", "1", "10"):
        for lr in ("3e4", "1e3", "3e3"):
            run = f"latent_sde_{meth}_b{b}_lr{lr}_view{V}"
            vals = [json.loads(p.read_text())["overall"]["nll"] for p in AML.glob(f"seed*/{run}/fold*/metrics_val.json")]
            t = avo(AML, run)
            if vals and t:
                rows.append({"run": run, "beta": {"01": 0.1, "1": 1, "10": 10}[b], "lr": float(lr.replace("e", "e-")),
                             "val_nll": float(np.mean(vals)), "test_log": t["log"][0], "test_log_sd": t["log"][1]})  # fmt: skip
    if rows:
        best = min(rows, key=lambda r: r["val_nll"])
        best["boot"] = avo_boot(AML, best["run"], f"latent_sde_photo_view{V}")
        N["grid"][meth] = {"rows": rows, "best": best}


# ---------------------------------------------------------------- Mango
def mango_runs(root: Path) -> dict[str, list[dict]]:
    msum.ROOT = root
    return msum.load()


def mango_store(root: Path, bucket: str, ref_map: dict[str, str]):
    runs = mango_runs(root)
    for r, ms_ in runs.items():
        cut = [m.get("cutday", {}) for m in ms_]
        pre = [m.get("prefix", {}) for m in ms_]
        rec = {k: [float(np.nanmean([c.get(k, np.nan) for c in cut])),
                   float(np.nanstd([c.get(k, np.nan) for c in cut], ddof=1)) if len(cut) > 1 else 0.0, len(cut)]
               for k in ("pop_brier", "status_brier", "spearman_within_day")}  # fmt: skip
        rec["prefix_log"] = [float(np.nanmean([p.get("log_score", np.nan) for p in pre])), 0.0, len(pre)]
        N[bucket][r] = rec
    msum.ROOT = root
    for r, ref in ref_map.items():
        if r in runs and ref in runs:
            N[bucket][r]["boot"] = msum.bootstrap(r, ref)


P = "sep1+sep2+sep3+sep4+sep5"
refs = {}
for s in ("mw", "both"):
    for v in ("sep1", "sep2", "sep3", "sep4", "sep5", "img", "concat", P, "fuse-mean", "fuse-geo", "fuse-trust"):
        refs[f"latent_sde_{s}_v{v}"] = "latent_sde_rgb"
    refs[f"latent_sde_{s}"] = "latent_sde_rgb"
for m in ("latent_sde", "latent_ode"):
    for s in ("time", "rgb", f"mw_v{P}", f"both_v{P}"):
        refs[f"{m}_{s}_hist"] = "age_clock"
    refs[f"{m}_both_v{P}"] = "latent_sde_rgb"
AML_M = ("kd", "feat", "rkd", "pkt", "sp", "samd", "ours", "ours_rkd", "ours_pkt", "ours_sp")
for me in AML_M:
    refs[f"latent_ode_{me}_v{P}_hist"] = "latent_ode_rgb_hist"
mango_store(MANGO, "mango", refs)
mango_store(MANGO.parent / (MANGO.name + "_recal"), "mango_recal", refs)
G = N["mango"]


def mb(r, k="pop_brier"):
    return G.get(r, {}).get(k, [np.nan, 0.0])


# Chart: mango single views
mviews = ["sep1", "sep2", "sep3", "sep4", "sep5", "img", P, "concat", "fuse-mean"]
mlab = {f"sep{s}": f"{s} apart" for s in range(1, 6)} | {"img": "image", P: "all 5\npooled", "concat": "all 6\nside by\nside",
                                                        "fuse-mean": "6 models\ncombined"}  # fmt: skip
fig, ax = plt.subplots(figsize=(8.2, 3.5))
w = 0.38
for i, (s, lab, c) in enumerate((("mw", "Microwave only", LIGHT), ("both", "Photo + microwave", BLUE))):
    mu = [mb(f"latent_sde_{s}_v{v}")[0] for v in mviews]
    sd = [mb(f"latent_sde_{s}_v{v}")[1] for v in mviews]
    xs = np.arange(len(mviews)) + (i - 0.5) * (w + 0.02) + np.where(np.arange(len(mviews)) >= 6, 0.5, 0.0)
    ax.bar(xs, mu, w, color=c, label=lab, yerr=sd, error_kw={"elinewidth": 1, "capsize": 2})
for ref, lab, ls in (("latent_sde_rgb", "photo only", ":"), ("age_clock", "known age", "--")):
    ax.axhline(mb(ref)[0], color="#555555", lw=1, ls=ls)
    ax.text(len(mviews) + 0.45, mb(ref)[0] + 0.004, lab, fontsize=7.5, color="#555555", va="bottom")
ticks = np.arange(len(mviews)) + np.where(np.arange(len(mviews)) >= 6, 0.5, 0.0)
ax.set_xticks(ticks, [mlab[v] for v in mviews], fontsize=7.5)
ax.set_xlim(-0.6, len(mviews) + 1.5)
ax.set_ylim(0, 0.3)
ax.set_ylabel("Spoilage score (lower is better)")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper right", fontsize=8, ncol=2, bbox_to_anchor=(1.0, 1.13))
ax.set_title("Mango, one scan: antenna spacing", loc="left", fontsize=10.5, fontweight="bold")
fig.tight_layout()
fig.savefig(HERE / "va_mango_views.png", dpi=200)
plt.close(fig)

# Chart: mango time-series models with the scan history
MM = {"latent_sde": "latent SDE", "latent_ode": "latent ODE"}
inputs = [("time", "Elapsed time only", LIGHT), ("rgb", "Photos", DARK), (f"mw_v{P}", "Microwave", ORANGE),
          (f"both_v{P}", "Photos + microwave", BLUE)]  # fmt: skip
fig, ax = plt.subplots(figsize=(8.2, 3.4))
w = 0.2
for i, (s, lab, c) in enumerate(inputs):
    mu = [mb(f"{m}_{s}_hist")[0] for m in MM]
    sd = [mb(f"{m}_{s}_hist")[1] for m in MM]
    ax.bar(np.arange(len(MM)) + (i - 1.5) * w, mu, w * 0.92, color=c, label=lab, yerr=sd,
           error_kw={"elinewidth": 1, "capsize": 2})  # fmt: skip
ax.axhline(mb("age_clock")[0], color="#555555", lw=1, ls="--", label="Known age + batch curve")
ax.set_xticks(np.arange(len(MM)), list(MM.values()))
ax.set_ylabel("Spoilage score (lower is better)")
ax.set_ylim(0, 0.25)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", fontsize=8, ncol=3)
ax.set_title("Mango, every earlier scan: two time-series models", loc="left", fontsize=10.5, fontweight="bold")
fig.tight_layout()
fig.savefig(HERE / "va_mango_hist.png", dpi=200)
plt.close(fig)

# Chart: mango AML on the latent ODE with history
MLAB = dict(AML_METHODS)
rows = [("latent_ode_rgb_hist", "Photo only, no teacher", LIGHT)] + [
    (f"latent_ode_{me}_v{P}_hist", MLAB[me], BLUE) for me in AML_M] + [
    (f"latent_ode_both_v{P}_hist", "The teacher: photo + microwave\n(needs the scanner)", DARK),
    ("latent_ode_time_hist", "Elapsed time only", LIGHT)]  # fmt: skip
rows = [r for r in rows if r[0] in G]
fig, ax = plt.subplots(figsize=(7.0, 0.32 * len(rows) + 1.0))
ys = np.arange(len(rows))[::-1]
base = mb("latent_ode_rgb_hist")[0]
for y, (r, lab, c) in zip(ys, rows, strict=True):
    mu, sd = mb(r)[0], mb(r)[1]
    if c == BLUE and "*" in G[r].get("boot", "") and mu < base:
        c = ORANGE
    ax.barh(y, mu, 0.62, color=c, xerr=sd, error_kw={"elinewidth": 1, "capsize": 2})
    ax.text(mu + sd + 0.002, y, f"{mu:.3f}", va="center", fontsize=8)
ax.axvline(base, color="#555555", lw=1, ls="--")
ax.set_yticks(ys, [r[1] for r in rows], fontsize=8.5)
ax.set_xlabel("Spoilage score (lower is better)")
ax.grid(axis="y", visible=False)
ax.set_title("Mango: teaching a photo model", loc="left", fontsize=10.5, fontweight="bold")
fig.tight_layout()
fig.savefig(HERE / "va_mango_aml.png", dpi=200)
plt.close(fig)

(HERE / "views_aml_numbers.json").write_text(json.dumps(N, indent=1, default=str))
print("wrote", HERE / "views_aml_numbers.json")
print({k: len(v) for k, v in N.items()})
