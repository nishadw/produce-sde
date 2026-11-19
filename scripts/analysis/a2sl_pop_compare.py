"""Half-synthetic A2SL: every model against the no-photo forecast, paired tests, and a blend with that forecast.

The no-photo forecast is temporal matching on the date alone (``matching_time``): every training fruit weighted
equally, i.e. "how many days fruit usually have left" at this start day. It is deterministic, so one seed suffices.

1. Paired tests against the no-photo forecast. Per forecast (tracked fruit, start, visible prefix) the censored log
   score is averaged over seeds, paired by forecast, and the mean difference A - B is bootstrapped over the 4 fruit
   (the forecasts of one fruit are correlated) and, as a looser check, over the 16 fruit x start clusters.
   ``wins`` = fruit on which A scores lower. With 4 fruit the fruit bootstrap has only 35 distinct resamples, so a
   "significant" difference there means all 4 fruit agree.
2. Blend. ``(1 - w) F + w F_pop``, with ``w`` chosen per seed and fold on that fold's validation forecasts (never the
   test fruit), from 0, 0.05, ..., 1. If over-confidence is the whole problem, the blended model should reach or
   beat the no-photo forecast; if it stays above, the model's ranking of fruit is also off.

Output: one table per fill (linear, logistic), rows grouped as reference / date only / photo / taught by a teacher,
with fixed- and random-start scores, the paired test against the no-photo forecast and the blend; then the blend
curve diagnostic. Runs that do not exist yet are left out.

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
CURVE = np.round(np.arange(0, 1.0001, 0.1), 1)

FILLS = ("linear", "logistic")
MODELS = {"matching": "Image matching", "weibull": "Weibull", "latent_ode": "Latent ODE", "latent_sde": "Latent SDE",
          "transformer": "Transformer"}  # fmt: skip
TAUGHT = ("latent_ode", "latent_sde", "transformer")  # CAML-KD / SAMD only on the gradient-trained models
METHODS = {"kd": "CAML-KD, teacher sees photo + scanner", "kd_photoT": "CAML-KD, teacher sees photo only",
           "samd": "SAMD, teacher sees photo + scanner", "samd_photoT": "SAMD, teacher sees photo only",
           "teacher": "the teacher itself (needs the scanner at test time)"}  # fmt: skip
SECTIONS = ("Reference", "Date only (no images; the same in both tables)", "Photo",
            "Photo-only model taught by a teacher during training (random start only)")  # fmt: skip
CURVE_ROWS = [("linear", "Latent SDE", "SAMD, teacher sees photo + scanner"), ("linear", "Latent SDE", "photo"),
              ("linear", "Latent ODE", "CAML-KD, teacher sees photo + scanner"), ("linear", "Transformer", "photo"),
              ("linear", "Transformer", "SAMD, teacher sees photo + scanner")]  # fmt: skip


def rows(fill: str) -> list[dict]:
    """One table row = a model + what it sees, with its fixed-start and random-start runs (``None`` = not run)."""

    def g(m, inp, st):
        return (RS, f"{m}_{inp}_synth-{'linear' if inp == 'time' else fill}_start-{st}")

    out = [{"section": SECTIONS[0], "model": "No-photo forecast", "sees": "date only", "fixed": POP, "random": POP}]
    out += [{"section": SECTIONS[1], "model": lab, "sees": "date only", "fixed": g(m, "time", "fixed"),
             "random": g(m, "time", "random")} for m, lab in MODELS.items() if m != "matching"]  # fmt: skip
    out += [{"section": SECTIONS[2], "model": lab, "sees": "photo", "fixed": g(m, "photo", "fixed"),
             "random": g(m, "photo", "random")} for m, lab in MODELS.items()]  # fmt: skip
    out += [{"section": SECTIONS[3], "model": MODELS[m], "sees": d, "fixed": None,
             "random": (AML / fill, f"{m}_{me}_viewall")} for m in TAUGHT for me, d in METHODS.items()]  # fmt: skip
    return [r for r in out if complete(r["random"])]


def complete(spec: tuple | None) -> bool:
    """Every seed x fold of the run has its real-visit predictions (5 x 4; deterministic models count the same)."""
    return spec is not None and len(list(spec[0].glob(f"seed*/{spec[1]}/fold*/predictions_test_real.npz"))) >= 20


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


def per_forecast(spec: tuple, blend: bool = False, fixed_w: float | None = None) -> tuple[dict, list[float]]:
    """key -> (fruit, start, seed-mean log score, seed-mean |median error|) on tracked real visits; chosen w's.

    ``blend``: w chosen on validation; ``fixed_w``: one w everywhere (the test-score curve, a diagnostic only)."""
    blend = blend or fixed_w is not None
    root, run = spec
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
            w = float(W_GRID[int(np.argmin(vs))]) if fixed_w is None else fixed_w
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


def start_weights(starts: np.ndarray) -> np.ndarray:
    """Each start day counts equally (as in a2sl_synth_rs_summary's "all starts"), however many forecasts it has."""
    u, inv, cnt = np.unique(starts, return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


def boot(d: np.ndarray, wt: np.ndarray, cl: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Cluster bootstrap of the weighted mean of ``d``: (lo, hi, share of resamples on the other side of 0)."""
    u, inv = np.unique(cl, return_inverse=True)
    s, c = np.bincount(inv, wt * d, len(u)), np.bincount(inv, wt, len(u))
    r = np.random.default_rng(seed).integers(0, len(u), (n, len(u)))
    W = np.stack([np.bincount(x, minlength=len(u)) for x in r])
    b = (W @ s) / (W @ c)
    lo, hi = np.percentile(b, [2.5, 97.5])
    m = np.sum(wt * d) / np.sum(wt)
    return float(lo), float(hi), float(np.mean(b >= 0) if m < 0 else np.mean(b <= 0))


def compare(a: dict, b: dict) -> dict:
    ks = sorted(set(a) & set(b))
    d = np.array([a[k][2] - b[k][2] for k in ks])
    fruit = np.array([a[k][0] for k in ks])
    fs = np.array([f"{a[k][0]}@{a[k][1]}" for k in ks])
    wt = start_weights(np.array([a[k][1] for k in ks]))
    per_fruit = {f: float(np.average(d[fruit == f], weights=wt[fruit == f])) for f in np.unique(fruit)}
    lo_f, hi_f, p_f = boot(d, wt, fruit)
    lo_c, hi_c, p_c = boot(d, wt, fs)
    return {"n": len(ks), "diff": float(np.average(d, weights=wt)), "fruit_ci": [lo_f, hi_f, p_f],
            "cluster_ci": [lo_c, hi_c, p_c], "wins": int(sum(v < 0 for v in per_fruit.values())),
            "per_fruit": per_fruit}  # fmt: skip


def summary(pf: dict) -> tuple[float, float]:
    v = list(pf.values())
    wt = start_weights(np.array([t[1] for t in v]))
    lg, e = np.array([t[2] for t in v]), np.array([t[3] for t in v])
    ok = ~np.isnan(e)
    return float(np.average(lg, weights=wt)), float(np.average(e[ok], weights=wt[ok]))


def verdict(r: dict) -> str:
    lo, hi = r["fruit_ci"][:2]
    word = "better" if hi < 0 else "worse" if lo > 0 else "no clear difference"
    return f"{r['diff']:+.2f} ({word})"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pop = per_forecast(POP)[0]
    ref = summary(pop)[0]
    out: dict = {"ref": ref, "tables": {}, "curve": {}}
    md = ["# Half-synthetic avocados: every model against the no-photo forecast", "",
          "How to read: each number is the log score on the real photos of the 4 avocados tracked to spoilage, "
          "averaged over every start day (lower is better). *Fixed start* = trained from day 1 only; *random start* "
          "= trained from a random day (the setting we use). *vs no-photo* = random-start score minus the no-photo "
          f"forecast ({ref:.2f}), with 'better' / 'worse' only when all 4 fruit agree (95 % bootstrap over fruit). "
          "*Fruit better* = on how many of the 4 fruit the model beats the no-photo forecast. *Day error* = days "
          "between the forecast's middle and the observed spoilage day (random start). *Blend* = the model mixed "
          "with the no-photo forecast, the mix chosen on validation fruit (w = share of no-photo forecast).",
          ""]  # fmt: skip
    for fill in FILLS:
        md += [f"## {fill.capitalize()} fill (days 7-15 extrapolated with a {fill} fit)", "",
               "| Model | Sees | Fixed start | Random start | vs no-photo | Fruit better | Day error | Blend (w) |",
               "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        tab, section = [], None
        for r in rows(fill):
            if r["section"] != section:
                section = r["section"]
                md.append(f"| **{section}** | | | | | | | |")
            fx = summary(per_forecast(r["fixed"])[0])[0] if complete(r["fixed"]) else None
            pf = per_forecast(r["random"])[0]
            lg, mae = summary(pf)
            rec = {k: r[k] for k in ("section", "model", "sees")} | {"fixed": fx, "random": lg, "mae": mae}
            vs = wins = blend = "—"
            if r["random"] != POP:
                c = compare(pf, pop)
                rec["vs"] = c
                vs, wins = verdict(c), f"{c['wins']}/4"
                if r["model"] not in ("Image matching",) and r["sees"] != "date only":
                    bpf, ws = per_forecast(r["random"], blend=True)
                    rec["blend"], rec["w"] = summary(bpf)[0], ws
                    blend = f"{rec['blend']:.2f} ({np.median(ws):.2f})"
            md.append(f"| {r['model']} | {r['sees']} | {'—' if fx is None else f'{fx:.2f}'} | {lg:.2f} | {vs} | "
                      f"{wins} | {mae:.1f} d | {blend} |")  # fmt: skip
            tab.append(rec)
            if (fill, r["model"], r["sees"]) in CURVE_ROWS:
                out["curve"][f"{r['model']}, {r['sees']}"] = {
                    f"{w:.1f}": summary(per_forecast(r["random"], fixed_w=float(w))[0])[0] for w in CURVE
                }
        out["tables"][fill] = tab
        md.append("")
    md += ["## Diagnostic: blend scored on the test fruit at every w (not a valid score; shows whether any mix helps)",
           "", "| Model (linear fill) | " + " | ".join(f"w={w:.1f}" for w in CURVE) + " |",
           "|---|" + "---|" * len(CURVE)]  # fmt: skip
    for lab, c in out["curve"].items():
        md.append(f"| {lab} | " + " | ".join(f"{v:.2f}" for v in c.values()) + " |")
    text = "\n".join(md) + "\n"
    (OUT / "summary.md").write_text(text)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    print(text)


if __name__ == "__main__":
    main()
