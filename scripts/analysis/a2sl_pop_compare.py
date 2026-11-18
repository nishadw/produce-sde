"""Half-synthetic A2SL: every model against the no-photo forecast, paired tests, and a blend with that forecast.

The no-photo forecast is temporal matching on the date alone (``matching_time``): every training fruit weighted
equally, i.e. "how many days fruit usually have left" at this start day. It is deterministic, so one seed suffices.

1. Paired tests. Per forecast (tracked fruit, start, visible prefix) the censored log score is averaged over seeds,
   paired by forecast between two runs, and the mean difference A - B is bootstrapped over the 4 tracked fruit
   (the forecasts of one fruit are correlated) and, as a looser check, over the 16 fruit x start clusters.
   ``wins`` = fruit on which A scores lower. With 4 fruit the fruit bootstrap has only 35 distinct resamples, so a
   "significant" difference there means all 4 fruit agree.
2. Blend. ``(1 - w) F + w F_pop``, with ``w`` chosen per seed and fold on that fold's validation forecasts (never the
   test fruit), from 0, 0.05, ..., 1. If over-confidence is the whole problem, the blended model should reach or
   beat the no-photo forecast; if it stays above, the model's ranking of fruit is also off.

python scripts/analysis/a2sl_pop_compare.py   -> outputs/a2sl_pop_compare/{summary.md,summary.json}
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ripen.eval.metrics import censored_log_score, compute_metrics

TRACKED = ("avo01", "avo02", "avo03", "avo11")
RS, AML = Path("outputs/a2sl_synth_rs"), Path("outputs/a2sl_synth_aml")
POP = (RS, "matching_time_synth-linear_start-random")
OUT = Path("outputs/a2sl_pop_compare")
W_GRID = np.round(np.arange(0, 1.0001, 0.05), 2)

# label -> (root holding seed*/, run)
RUNS = {
    "No-photo forecast": POP,
    "Matching, photo (logistic fill)": (RS, "matching_photo_synth-logistic_start-random"),
    "Matching, photo (linear fill)": (RS, "matching_photo_synth-linear_start-random"),
    "Weibull, photo (logistic fill)": (RS, "weibull_photo_synth-logistic_start-random"),
    "Weibull, photo (linear fill)": (RS, "weibull_photo_synth-linear_start-random"),
    "Weibull, date only": (RS, "weibull_time_synth-linear_start-random"),
    "Latent SDE, photo (linear)": (RS, "latent_sde_photo_synth-linear_start-random"),
    "Latent SDE, photo (logistic)": (RS, "latent_sde_photo_synth-logistic_start-random"),
    "Latent SDE, date only": (RS, "latent_sde_time_synth-linear_start-random"),
    "Latent ODE, photo (linear)": (RS, "latent_ode_photo_synth-linear_start-random"),
    "Latent ODE, photo (logistic)": (RS, "latent_ode_photo_synth-logistic_start-random"),
    "Latent SDE + SAMD (linear)": (AML / "linear", "latent_sde_samd_viewall"),
    "Latent SDE + SAMD (logistic)": (AML / "logistic", "latent_sde_samd_viewall"),
    "Latent SDE + CAML-KD (linear)": (AML / "linear", "latent_sde_kd_viewall"),
    "Latent ODE + CAML-KD (linear)": (AML / "linear", "latent_ode_kd_viewall"),
    "Latent ODE + SAMD (linear)": (AML / "linear", "latent_ode_samd_viewall"),
}
BLEND = [k for k in RUNS if k.startswith(("Latent", "Weibull, photo"))]
PAIRS = [
    ("Matching, photo (logistic fill)", "No-photo forecast"),
    ("Matching, photo (linear fill)", "No-photo forecast"),
    ("Latent SDE + SAMD (linear)", "No-photo forecast"),
    ("Latent SDE, photo (linear)", "No-photo forecast"),
    ("Matching, photo (logistic fill)", "Latent SDE + SAMD (linear)"),
    ("Latent SDE + SAMD (linear)", "Latent SDE, photo (linear)"),
    ("blend: Latent SDE + SAMD (linear)", "No-photo forecast"),
    ("blend: Latent SDE, photo (linear)", "No-photo forecast"),
    ("Matching, photo (logistic fill)", "blend: Latent SDE + SAMD (linear)"),
]


def key(z: dict, i: int) -> str:
    return f"{z['item_ids'][i]}|{int(z['n_obs'][i])}|{float(z['t_now'][i]):.3f}"


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as t:
        return {k: t[k] for k in ("F", "lo", "hi", "item_ids", "n_obs", "t_now")}


def mix(z: dict, ref: dict, w: float) -> tuple[np.ndarray, np.ndarray]:
    """-> (blended F, mask of rows with a reference forecast) for ``z``'s rows, paired by key."""
    idx = {key(ref, i): i for i in range(len(ref["F"]))}
    j = np.array([idx.get(key(z, i), -1) for i in range(len(z["F"]))])
    F = z["F"].copy()
    ok = j >= 0
    F[ok] = (1 - w) * z["F"][ok] + w * ref["F"][j[ok]]
    return F, ok


def per_forecast(label: str, blend: bool = False) -> tuple[dict[str, tuple[str, int, float, float]], list[float]]:
    """key -> (fruit, start, seed-mean log score, seed-mean |median error|) on tracked real visits; chosen w's."""
    root, run = RUNS[label]
    acc: dict[str, list] = defaultdict(list)
    ws = []
    for fd in sorted(root.glob(f"seed*/{run}/fold*")):
        z = load(fd / "predictions_test_real.npz")
        F = z["F"]
        if blend:
            pfd = POP[0] / fd.parts[-3] / POP[1] / fd.name
            if not pfd.exists():
                pfd = sorted(POP[0].glob(f"seed*/{POP[1]}/{fd.name}"))[0]  # deterministic: any seed
            v, pv = load(fd / "predictions_val.npz"), load(pfd / "predictions_val.npz")
            vs = []
            for w in W_GRID:
                Fv, ok = mix(v, pv, w)
                if not ok.any():
                    raise RuntimeError(f"no val overlap {fd} vs {pfd}")
                vs.append(np.mean(censored_log_score(Fv[ok], v["lo"][ok], v["hi"][ok])))
            w = float(W_GRID[int(np.argmin(vs))])
            ws.append(w)
            F, ok = mix(z, load(pfd / "predictions_test_real.npz"), w)
            if not ok.all():
                raise RuntimeError(f"test rows without a reference forecast in {fd}")
        lg = censored_log_score(F, z["lo"], z["hi"])
        med = np.array([compute_metrics(F[i : i + 1], z["lo"][i : i + 1], z["hi"][i : i + 1]).to_dict()["mae"]
                        for i in range(len(F))])  # fmt: skip
        for i in range(len(F)):
            base, _, k = str(z["item_ids"][i]).partition("@")
            if base in TRACKED:
                acc[key(z, i)].append((base, int(k or 0), float(lg[i]), float(med[i])))
    return {
        k: (v[0][0], v[0][1], float(np.mean([t[2] for t in v])), float(np.nanmean([t[3] for t in v])))
        for k, v in acc.items()
    }, ws


def boot(d: np.ndarray, cl: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Cluster bootstrap of the mean of ``d``: (lo, hi, share of resamples on the other side of 0)."""
    u, inv = np.unique(cl, return_inverse=True)
    s, c = np.bincount(inv, d, len(u)), np.bincount(inv, minlength=len(u)).astype(float)
    r = np.random.default_rng(seed).integers(0, len(u), (n, len(u)))
    W = np.stack([np.bincount(x, minlength=len(u)) for x in r])
    b = (W @ s) / (W @ c)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(lo), float(hi), float(np.mean(b >= 0) if d.mean() < 0 else np.mean(b <= 0))


def compare(a: dict, b: dict) -> dict:
    ks = sorted(set(a) & set(b))
    d = np.array([a[k][2] - b[k][2] for k in ks])
    fruit = np.array([a[k][0] for k in ks])
    fs = np.array([f"{a[k][0]}@{a[k][1]}" for k in ks])
    per_fruit = {f: float(d[fruit == f].mean()) for f in np.unique(fruit)}
    lo_f, hi_f, p_f = boot(d, fruit)
    lo_c, hi_c, p_c = boot(d, fs)
    return {"n": len(ks), "diff": float(d.mean()), "fruit_ci": [lo_f, hi_f, p_f], "cluster_ci": [lo_c, hi_c, p_c],
            "wins": int(sum(v < 0 for v in per_fruit.values())), "per_fruit": per_fruit}  # fmt: skip


def summary(pf: dict) -> tuple[float, float]:
    v = list(pf.values())
    return float(np.mean([t[2] for t in v])), float(np.nanmean([t[3] for t in v]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pf, ws = {}, {}
    for lab in RUNS:
        pf[lab], _ = per_forecast(lab)
    for lab in BLEND:
        pf[f"blend: {lab}"], ws[lab] = per_forecast(lab, blend=True)
    ref = summary(pf["No-photo forecast"])[0]
    md = ["Half-synthetic A2SL, real visits of the 4 tracked fruit, random-start training, every start day scored. "
          "Log score lower is better (mean over forecasts, seed-mean first). Blend weight w on the no-photo "
          "forecast is chosen on each fold's validation fruit.", "",
          "## Scores", "",
          "| model | log score | vs no-photo | day error | blended log score | blended day error | w (median, range) |",
          "|---|---|---|---|---|---|---|"]  # fmt: skip
    out: dict = {"scores": {}, "pairs": []}
    for lab in RUNS:
        lg, mae = summary(pf[lab])
        row = [lab, f"{lg:.2f}", f"{lg - ref:+.2f}", f"{mae:.2f}"]
        rec = {"log": lg, "mae": mae}
        if lab in BLEND:
            blg, bmae = summary(pf[f"blend: {lab}"])
            w = np.array(ws[lab])
            row += [f"{blg:.2f}", f"{bmae:.2f}", f"{np.median(w):.2f} ({w.min():.2f}–{w.max():.2f})"]
            rec |= {"blend_log": blg, "blend_mae": bmae, "w": ws[lab]}
        else:
            row += ["—", "—", "—"]
        md.append("| " + " | ".join(row) + " |")
        out["scores"][lab] = rec
    md += ["", "## Paired differences (A - B; negative = A better)", "",
           "| A | B | forecasts | A - B | 95 % CI by fruit | 95 % CI by fruit x start | fruit A wins | per fruit |",
           "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for a, b in PAIRS:
        r = compare(pf[a], pf[b])
        star = lambda ci: " *" if ci[0] > 0 or ci[1] < 0 else ""  # noqa: E731
        md.append(
            f"| {a} | {b} | {r['n']} | {r['diff']:+.2f} | [{r['fruit_ci'][0]:+.2f}, {r['fruit_ci'][1]:+.2f}]"
            f"{star(r['fruit_ci'])} | [{r['cluster_ci'][0]:+.2f}, {r['cluster_ci'][1]:+.2f}]{star(r['cluster_ci'])}"
            f" | {r['wins']}/4 | " + ", ".join(f"{f} {v:+.2f}" for f, v in r["per_fruit"].items()) + " |"
        )
        out["pairs"].append({"a": a, "b": b, **r})
    md += ["", "* = the 95 % interval excludes 0. With 4 fruit, the fruit bootstrap can exclude 0 only when all 4 "
           "fruit move the same way."]  # fmt: skip
    text = "\n".join(md) + "\n"
    (OUT / "summary.md").write_text(text)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    print(text)


if __name__ == "__main__":
    main()
