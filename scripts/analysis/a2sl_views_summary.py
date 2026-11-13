"""Which microwave view predicts best? <root>/seed*/<run>/fold*/ -> <root>/summary.md (and summary_recal.md).

python scripts/analysis/a2sl_views_summary.py --root outputs/a2sl_views/labels_hidden [--recal]

``--recal``: cross-fitted PIT recalibration. Each fold's forecasts are recalibrated with a map fitted on the other three
folds' test forecasts of the same run and seed (a fold's own validation set is 2 fruit, too few to fit on), so no
forecast is ever recalibrated with its own fruit. The map is 0.9 x Kaplan-Meier + 0.1 x the raw forecast.

1. Time series (scripts/run/a2sl_views.py): every run's test predictions are pooled over the 4 folds, then scored
   (MAE, CRPS, 80 % coverage, C-index). Cells are mean ± std over seeds. Two scopes: all test predictions, and only
   the 4 avocados scanned at every visit (the only fruit whose history contains more than one scan). Paired
   bootstrap: the interval-censored log score of each run minus the time-only control, prediction by prediction
   (averaged over seeds), fruit resampled 5000 times; negative = the run beats the control, * = 95 % CI excludes 0.
2. Snapshot (closed form, no training): from one visit's scan of one view, estimate the fruit's study day and its
   ripeness class, leave-one-fruit-out ridge regression over all 35 scanned visits.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ripen.data.a2sl import JMW2026_DAY_STAGE, SCAN_VIEWS, load_scan_cache, scan_summary, view_keys
from ripen.eval.metrics import censored_log_score, compute_metrics
from ripen.eval.recalibrate import PITRecalibrator

ROOT = Path("outputs/a2sl_views/labels_hidden")
RECAL = False
RECAL_WEIGHT = 0.9  # 0.9 KM map + 0.1 raw forecast (scratch/recal_diag.py: 0.8-0.95 all better than 1.0 and 0.5)
TRACKED = {"avo01", "avo02", "avo03", "avo11"}
VIEWS = ["all", "concat", "0+120+240", "60+180+300", *[str(v) for v in SCAN_VIEWS]]
FUSED = ["fuse-mean", "fuse-geo", "fuse-trust"]  # late_fusion.py fold: six per-view models combined


def view_label(v: str) -> str:
    if v.startswith("fuse-"):
        return f"six separate models, combined ({v[5:]})"
    if "+" in v:
        return ", ".join(f"{a}°" for a in v.split("+")) + " pooled"
    return {"all": "six pooled", "concat": "six side by side"}.get(v, f"{v}°")


MODELS = {
    "latent_sde": "latent SDE",
    "latent_ode": "latent ODE",
    "neural_cde": "neural CDE",
    "gru_d": "GRU-D",
    "fpt_filter": "exact filter (FPT)",
}
INPUTS = {"mw": "microwave only", "photo_mw": "photo + microwave"}
KEYS = [("mae", "MAE (d)"), ("crps", "CRPS"), ("coverage80", "80% cov."), ("cindex", "C-index")]


def pooled(run: str) -> dict[str, dict]:
    """seed -> concatenated test predictions over folds (None if a fold is missing)."""
    out = {}
    for sd in sorted(ROOT.glob(f"seed*/{run}")):
        parts = [dict(np.load(f, allow_pickle=True)) for f in sorted(sd.glob("fold*/predictions_test.npz"))]
        if len(parts) < 4:
            continue
        if RECAL:
            Fs = []
            for k, p in enumerate(parts):
                o = [q for j, q in enumerate(parts) if j != k]
                cal = PITRecalibrator.fit(
                    *(np.concatenate([q[x] for q in o]) for x in ("F", "lo", "hi")), weight=RECAL_WEIGHT
                )
                Fs.append(cal(p["F"]))
            for p, F in zip(parts, Fs, strict=True):
                p["F"] = F
        out[sd.parent.name] = {k: np.concatenate([p[k] for p in parts]) for k in ("F", "lo", "hi", "item_ids", "n_obs")}
    return out


def metrics(run: str, tracked_only: bool) -> dict[str, list[float]]:
    res = defaultdict(list)
    for z in pooled(run).values():
        m = np.isin(z["item_ids"], list(TRACKED)) if tracked_only else np.ones(len(z["lo"]), bool)
        d = compute_metrics(z["F"][m], z["lo"][m], z["hi"][m]).to_dict()
        for k, _ in KEYS:
            res[k].append(d[k])
    return res


def cell(v) -> str:
    v = np.array([x for x in v if x == x], float)
    if not len(v):
        return "—"
    return f"{v.mean():.2f}" if len(v) == 1 else f"{v.mean():.2f} ± {v.std(ddof=1):.2f}"


def per_pred_score(run: str) -> dict[tuple, float]:
    acc = defaultdict(list)
    for z in pooled(run).values():
        ls = censored_log_score(z["F"], z["lo"], z["hi"])
        for i, n, s in zip(z["item_ids"], z["n_obs"], ls, strict=True):
            acc[(str(i), int(n))].append(float(s))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def bootstrap(a: str, b: str, tracked_only: bool, n_boot: int = 5000) -> str:
    pa, pb = per_pred_score(a), per_pred_score(b)
    keys = [k for k in pa if k in pb and (not tracked_only or k[0] in TRACKED)]
    if not keys:
        return "—"
    fruit = sorted({k[0] for k in keys})
    d = {f: [pa[k] - pb[k] for k in keys if k[0] == f] for f in fruit}
    sums, cnts = np.array([sum(d[f]) for f in fruit]), np.array([len(d[f]) for f in fruit])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(fruit), size=(n_boot, len(fruit)))
    boot = sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return f"{sums.sum() / cnts.sum():+.2f} [{lo:+.2f}, {hi:+.2f}]" + (" *" if hi < 0 or lo > 0 else "")


def ridge_loo(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, lam: float = 1.0, n_pc: int | None = None
) -> np.ndarray:
    """Leave-one-fruit-out ridge predictions; with ``n_pc``, PCA (fitted on the training fruit) first."""
    pred = np.zeros(len(y))
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        Xtr, Xte = X[tr], X[te]
        if n_pc:
            mu0 = Xtr.mean(0)
            _, _, vt = np.linalg.svd(Xtr - mu0, full_matrices=False)
            Xtr, Xte = (Xtr - mu0) @ vt[:n_pc].T, (Xte - mu0) @ vt[:n_pc].T
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Z = (Xtr - mu) / sd
        w = np.linalg.solve(Z.T @ Z + lam * tr.sum() * np.eye(Z.shape[1]), Z.T @ (y[tr] - y[tr].mean()))
        pred[te] = ((Xte - mu) / sd) @ w + y[tr].mean()
    return pred


def _score(day_hat, cls_raw, day, stage, fruit, rng):
    cls_hat = np.clip(np.rint(cls_raw), 1, 5)
    err = np.abs(day_hat - day)
    uf = np.unique(fruit)
    boot = [err[np.isin(fruit, rng.choice(uf, len(uf)))].mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return err.mean(), lo, hi, (cls_hat == stage).mean(), (np.abs(cls_hat - stage) <= 1).mean()


def snapshot_table() -> list[str]:
    visits = pd.read_csv("data/manifests/a2sl_avocado_visits.csv")
    visits = visits[visits.scan_keys.notna()]
    scans = load_scan_cache("data/a2sl/scans_avocado_jun2023.npz")
    day = visits.day.to_numpy(float)
    stage = np.array([JMW2026_DAY_STAGE[int(d)] for d in day], float)
    fruit = visits.item_id.to_numpy()
    mask = np.isfinite(scans[visits.scan_keys.iloc[0].split(";")[0]])
    rng = np.random.default_rng(0)
    # no-scan baseline: guess the training fruit's average day / class
    z = np.zeros((len(day), 1))
    rows = {"none": {"summary": _score(ridge_loo(z, day, fruit), ridge_loo(z, stage, fruit), day, stage, fruit, rng)}}
    for v in VIEWS:
        summ, pix = [], []
        for keys in visits.scan_keys:
            ks = view_keys(keys.split(";"), {"all": None, "concat": "concat"}.get(v, v))
            ks = sorted(ks, key=lambda k: int(k.rsplit("_r", 1)[1]))
            a = np.stack([scan_summary(scans[k]) for k in ks])
            pool = v == "all" or "+" in v
            summ.append(np.concatenate([a.mean(0), a.std(0)]) if pool else a.ravel() if v == "concat" else a[0])
            px = np.stack([np.nan_to_num(scans[k][mask]) for k in ks])
            pix.append(px.ravel() if v == "concat" else px.mean(0))
        rows[v] = {}
        for name, X, npc in (("summary", np.stack(summ), None), ("pixels", np.stack(pix), 8)):
            rows[v][name] = _score(ridge_loo(X, day, fruit, n_pc=npc), ridge_loo(X, stage, fruit, n_pc=npc), day, stage,
                                   fruit, rng)  # fmt: skip
    L = ["| view | age error, 12 summary numbers (days) [95% CI] | class exact / within one | "
         "age error, image pixels (days) [95% CI] | class exact / within one |", "|---|---|---|---|---|"]  # fmt: skip
    for v, r in rows.items():
        name = {
            "none": "no scan (guess the average)",
            "all": "six pooled (mean + spread)",
            "concat": "six side by side",
        }.get(v, view_label(v))
        cells = []
        for k in ("summary", "pixels"):
            if k not in r:
                cells += ["—", "—"]
                continue
            e, lo, hi, ex, w1 = r[k]
            cells += [f"{e:.2f} [{lo:.2f}, {hi:.2f}]", f"{ex:.0%} / {w1:.0%}"]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append(f"\n35 scanned visits on 11 fruit, leave-one-fruit-out ridge regression. Pixels: the reconstruction's "
             f"{int(mask.sum())} in-disk pixels (six pooled: their average; side by side: all six), "
             "8 principal components fitted on the "
             "training fruit.")  # fmt: skip
    return L


def main() -> None:
    global ROOT, RECAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--recal", action="store_true")
    a = ap.parse_args()
    ROOT, RECAL = Path(a.root), a.recal
    seeds = sorted({p.name for p in ROOT.glob("seed*")})
    title = "# Which microwave view predicts best (A2SL avocados)" + (", recalibrated" if RECAL else "")
    L = [title, "", f"Root: `{ROOT}`.",
         f"Seeds: {', '.join(seeds) or 'none'}; 4 folds (leave one of avocados 1, 2, 3, 11 out). Lower is better "
         "except coverage (about 80 % is ideal) and C-index (higher is better).", ""]  # fmt: skip
    for scope, tracked in (("Tracked avocados only (1, 2, 3, 11)", True), ("All test predictions", False)):
        for m, mlab in MODELS.items():
            if not any(ROOT.glob(f"seed*/{m}_*")):
                continue
            L += [f"## Time series, {mlab}: {scope.lower()}", "",
                  "| input | view | " + " | ".join(k[1] for k in KEYS) + " | log score vs time-only control |",
                  "|---|---|" + "---|" * (len(KEYS) + 1)]  # fmt: skip
            ctrl = f"{m}_time"
            for inp, ilab in INPUTS.items():
                for v in VIEWS + FUSED:
                    run = f"{m}_{inp}_view{v}"
                    vm = metrics(run, tracked)
                    vlab = view_label(v)
                    L.append(f"| {ilab} | {vlab} | " + " | ".join(cell(vm[k]) for k, _ in KEYS)
                             + f" | {bootstrap(run, ctrl, tracked)} |")  # fmt: skip
            for run, lab in ((f"{m}_photo", "photo only"), (ctrl, "**time only (control)**")):
                vm = metrics(run, tracked)
                L.append(f"| {lab} | — | " + " | ".join(cell(vm[k]) for k, _ in KEYS) + " | — |")
            L.append("")
    L += ["## Snapshot: one scan of one view -> how old is the fruit, which ripeness class", "", *snapshot_table(), ""]
    text = "\n".join(L) + "\n"
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / ("summary_recal.md" if RECAL else "summary.md")).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
