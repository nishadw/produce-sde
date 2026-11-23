"""All-fruit A2SL: does each model get better with (1)+(2) settings chosen on 11 fruit instead of 2, and with new photo
features (fruit only, skin colour, public-avocado ripeness reader)? Storage day is left out (it learns from each
training photo's storage day, which the day-rule labels turn into the answer).

(1)+(2) "recalibrated": for each held-out fruit, a temperature T (flattens or sharpens the forecast's odds) and a
blend weight w with the no-photo forecast are chosen on the other 11 held-out fruit's test forecasts (leave one fruit
out over the outer folds, so the scored fruit never picks its own settings), then applied to that fruit. Raw = the
forecast as trained (no blend). Scoring, the paired bootstrap and the 8 observed fruit are a2sl_all_compare's.

python scripts/analysis/a2sl_improve_compare.py   -> outputs/a2sl_improve/{summary.md,summary.json}
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a2sl_all_compare as ac  # noqa: E402
import a2sl_pop_compare as pc  # noqa: E402

from ripen.eval.metrics import censored_log_score, compute_metrics  # noqa: E402

OUT = Path("outputs/a2sl_improve")
TS = (0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)  # T < 1 sharpens, T > 1 flattens
WS = np.round(np.arange(0, 1.0001, 0.1), 1)
FEATS = {"": "original photo", "_fruit": "fruit only (background removed)", "_colour": "skin colour (12 numbers)",
         "_public": "ripeness reader trained on public avocados"}  # fmt: skip
MODELS = {"matching": "Image matching", "transformer": "Transformer", "latent_sde": "Latent SDE",
          "latent_ode": "Latent ODE"}  # fmt: skip
TEACH = {"kd": "+ CAML-KD", "samd": "+ SAMD"}  # original photo only (teacher sees the scanner)


def rows_of(spec):
    """All test rows of a run: list of (fold dir, dict)."""
    root, run = spec
    return [(fd, pc.load(fd / "predictions_test.npz")) for fd in sorted(root.glob(f"seed*/{run}/fold*"))]


def ref_for(fd: Path) -> dict:
    pfd = ac.POP[0] / fd.parts[-3] / ac.POP[1] / fd.name
    if not pfd.exists():
        pfd = sorted(ac.POP[0].glob(f"seed*/{ac.POP[1]}/{fd.name}"))[0]
    return pc.load(pfd / "predictions_test.npz")


def temper(F: np.ndarray, T: float) -> np.ndarray:
    """Raise the day probabilities (plus the 'beyond the horizon' tail) to 1/T and renormalise."""
    p = np.diff(np.c_[np.zeros(len(F)), F], axis=1)
    p = np.c_[p, 1 - F[:, -1]].clip(1e-12, None) ** (1.0 / T)
    p /= p.sum(1, keepdims=True)
    return np.cumsum(p[:, :-1], 1)


def recalibrated(spec) -> dict:
    """key -> (fruit, start, seed-mean log score, seed-mean day error) after leave-one-fruit-out (T, w)."""
    data = []  # (fruit, start, key, F, Fref, lo, hi)
    for fd, z in rows_of(spec):
        ref = ref_for(fd)
        idx = {pc.key(ref, i): i for i in range(len(ref["F"]))}
        for i in range(len(z["F"])):
            base, _, k = str(z["item_ids"][i]).partition("@")
            if base not in ac.ALL:
                continue
            data.append(
                (base, int(k or 0), pc.key(z, i), z["F"][i], ref["F"][idx[pc.key(z, i)]], z["lo"][i], z["hi"][i])
            )
    fruit = np.array([d[0] for d in data])
    start = np.array([d[1] for d in data])
    F, R = np.stack([d[3] for d in data]), np.stack([d[4] for d in data])
    lo, hi = np.array([d[5] for d in data]), np.array([d[6] for d in data])
    # every (T, w) scored once on every row
    S = np.zeros((len(TS), len(WS), len(F)))
    for a, T in enumerate(TS):
        Ft = temper(F, T)
        for b, w in enumerate(WS):
            S[a, b] = censored_log_score((1 - w) * Ft + w * R, lo, hi)
    out, chosen = defaultdict(list), {}
    for f in ac.ALL:
        other = fruit != f
        wt = np.zeros(len(F))
        for g in set(fruit[other]):  # each other fruit, and each start within it, counts equally
            sel = fruit == g
            wt[sel] = pc.start_weights(start[sel]) / pc.start_weights(start[sel]).sum()
        a, b = np.unravel_index(np.argmin((S[:, :, other] * wt[other]).sum(-1)), S.shape[:2])
        chosen[f] = (TS[a], float(WS[b]))
        Fc = (1 - WS[b]) * temper(F[fruit == f], TS[a]) + WS[b] * R[fruit == f]
        for j, i in enumerate(np.where(fruit == f)[0]):
            e = compute_metrics(Fc[j : j + 1], lo[i : i + 1], hi[i : i + 1]).to_dict()["mae"]
            out[data[i][2]].append((f, data[i][1], float(S[a, b, i]), float(e)))
    pf = {k: (v[0][0], v[0][1], float(np.mean([t[2] for t in v])), float(np.nanmean([t[3] for t in v])))
          for k, v in out.items()}  # fmt: skip
    return pf, chosen


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pop = ac.only(pc.per_forecast(ac.POP)[0], ac.OBSERVED)
    ref, ref_mae = pc.summary(pop)
    res = {"ref": ref, "ref_mae": ref_mae, "models": {}}
    md = ["# All-fruit A2SL: settings chosen on 11 fruit, and new photo features", "",
          f"Score on the 8 observed fruit, lower is better; no-photo forecast {ref:.2f} (day error {ref_mae:.1f} d). "
          "Raw = as trained. Recalibrated = temperature T and blend w with the no-photo forecast chosen on the other "
          "11 held-out fruit. 'vs no-photo' = recalibrated minus the no-photo forecast, 95 % bootstrap over the 8 "
          "fruit.",
          ""]  # fmt: skip
    for m, lab in MODELS.items():
        variants = [(FEATS[t], (ac.GRID, f"{m}_photo_synth-real_start-random{t}")) for t in FEATS]
        if m != "matching":
            variants += [(f"original photo {d}", (ac.AML, f"{m}_{me}_viewall")) for me, d in TEACH.items()]
        md += [f"## {lab}", "", "| Photo features | Raw | Recalibrated | vs no-photo | Fruit better | Day error | "
               "T / w (median) |", "|---|---|---|---|---|---|---|"]  # fmt: skip
        res["models"][lab] = []
        for name, spec in variants:
            if not ac.complete(spec):
                continue
            raw, raw_mae = pc.summary(ac.only(pc.per_forecast(spec)[0], ac.OBSERVED))
            pf, ch = recalibrated(spec)
            po = ac.only(pf, ac.OBSERVED)
            cal, mae = pc.summary(po)
            c = pc.compare(po, pop)
            w = float(np.median([v[1] for v in ch.values()]))
            T = float(np.median([v[0] for v in ch.values() if v[1] < 1])) if w < 1 else float("nan")
            md.append(f"| {name} | {raw:.2f} | {cal:.2f} | {pc.verdict(c)} | {c['wins']} of 8 | {mae:.1f} d | "
                      f"{'—' if np.isnan(T) else f'{T:g}'} / {w:.1f} |")  # fmt: skip
            res["models"][lab].append(
                {"features": name, "raw": raw, "raw_mae": raw_mae, "cal": cal, "mae": mae, "vs": c, "T": T, "w": w}
            )
        md.append("")
    text = "\n".join(md) + "\n"
    (OUT / "summary.md").write_text(text)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(text)


if __name__ == "__main__":
    main()
