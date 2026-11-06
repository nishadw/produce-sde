"""Mango study: one model x one input setting, 6-fold cross-validation by fruit, several seeds, in one process.

python scripts/mango_cv.py --model latent_sde --setting samd --seeds 1 2 3
python scripts/mango_cv.py --model refs        # closed-form references + firmness/stage read-outs

Settings (``ripen.data.mango``): ``rgb`` photo only; ``mw`` microwave only; ``both`` photo + microwave;
``samd`` photo-only student distilled from a photo + microwave teacher (``ripen.train.samd``). Every model is scored
on the same fruit, at the same moment (its cut day), with ``ripen.eval.mango`` — see that module for the scores.

Outputs: ``results/mango/seed<S>/<model>_<setting>/{metrics.json, predictions.npz, history_fold<k>.csv}``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import jax
import numpy as np
from omegaconf import OmegaConf

from ripen.data.collate import dataset_info
from ripen.data.mango import SETTINGS, fold_item_ids, load_mango, mango_tables, write_mango_folds
from ripen.data.splits import load_splits
from ripen.eval.mango import cutday_scores, fruit_records, predict, prefix_scores
from ripen.models import build_model
from ripen.train import TrainConfig, train
from ripen.train.samd import train_samd

HORIZON = 20
N_FOLDS = 6
REF_MODELS = ("age_clock", "rgb_clock", "mw_clock", "both_clock")
# mango-wide overrides: stages are targets, never inputs; one storage group with an unknown temperature, so the clock
# is a single learned rate started at ~3.5 stage units per 12 days; no guided posterior (it reads labels at test time)
OVERRIDES = {
    "*": {"use_stages": False},
    "latent_sde": {"posterior_guidance": False, "event_mean": "log_mean",
                   "clock_kwargs": {"ea_j_mol": 50000.0, "rate_ref": 0.3, "t_ref_c": 20.0}},
    "latent_ode": {"clock_kwargs": {"ea_j_mol": 50000.0, "rate_ref": 0.3, "t_ref_c": 20.0}},
    "fpt_filter": {"image_emission": "always", "clock_kwargs": {"ea_j_mol": 50000.0, "rate_ref": 0.3, "t_ref_c": 20.0}},
}  # fmt: skip


def model_cfg(name: str, extra: list[str]) -> dict:
    base = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs" / "model" / f"{name}.yaml")
    cfg = OmegaConf.create({"horizon": HORIZON, "model": base})
    m = OmegaConf.to_container(cfg, resolve=True)["model"]
    m.update(OVERRIDES["*"])
    m.update(OVERRIDES.get(name, {}))
    for kv in extra:
        k, v = kv.split("=", 1)
        try:
            m[k] = json.loads(v)
        except json.JSONDecodeError:
            m[k] = v
    return m


def folds() -> list[dict[str, list[str]]]:
    paths = [Path(f"results/splits/mango_fold{k}.json") for k in range(N_FOLDS)]
    if not all(p.exists() for p in paths):
        write_mango_folds(n_folds=N_FOLDS)
    return [load_splits(p) for p in paths]


def select(items, ids):
    want = set(ids)
    return [it for it in items if it.item_id in want]


# ----- closed-form references -----------------------------------------------------------------------------------
def current_status_cdf(day: np.ndarray, status: np.ndarray, t_max: int) -> np.ndarray:
    """NPMLE of P(T <= t) from current-status data (isotonic regression of status on cut day), on days 0..t_max,
    linear between observed days, 0 before the first and flat after the last."""
    ud = np.unique(day)
    y = np.array([status[day == d].mean() for d in ud])
    w = np.array([(day == d).sum() for d in ud], float)
    blocks = [[y[i], w[i], 1] for i in range(len(ud))]  # pool adjacent violators
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            a, b = blocks[i], blocks[i + 1]
            blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1]), a[1] + b[1], a[2] + b[2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    fit = np.concatenate([[v] * n for v, _, n in blocks])
    return np.interp(np.arange(t_max + 1), ud, fit, left=0.0, right=fit[-1])


def ridge(x: np.ndarray, y: np.ndarray, lam: float = 1.0):
    mu, sd = x.mean(0), x.std(0) + 1e-6
    z = (x - mu) / sd
    w = np.linalg.solve(z.T @ z + lam * len(y) * np.eye(z.shape[1]), z.T @ (y - y.mean()))
    resid = y - (z @ w + y.mean())
    return (lambda xn: ((xn - mu) / sd) @ w + y.mean()), float(np.sqrt(np.mean(resid**2)) + 0.5)


def age_cdf(G: np.ndarray, a_hat: np.ndarray, a_sd: np.ndarray | float) -> np.ndarray:
    """``F(h) = E_a G(a + h)``, ``a ~ N(a_hat, a_sd)`` clipped to the study (Gauss-Hermite, 21 nodes)."""
    x, w = np.polynomial.hermite_e.hermegauss(21)
    w = w / w.sum()
    a = np.clip(a_hat[:, None] + np.atleast_1d(a_sd)[:, None] * x[None], 0, len(G) - 1 - HORIZON)
    h = np.arange(HORIZON + 1)
    idx = a[..., None] + h[None, None]
    Gv = np.interp(idx, np.arange(len(G)), G)
    return np.einsum("k,nkh->nh", w, Gv)


def run_refs(fold_list, seeds, out_root):
    by = {s: load_mango(s) for s in SETTINGS}
    fruit, _, _ = mango_tables()
    for name in REF_MODELS:
        setting = {"age_clock": "mw", "rgb_clock": "rgb", "mw_clock": "mw", "both_clock": "both"}[name]
        items = by[setting]
        F_all, ids_all, lo_all, hi_all, cut_all = [], [], [], [], []
        probe = []
        for sp in fold_list:
            isp = fold_item_ids(items, sp)
            tr = select(items, isp["train"] + isp["val"])
            te = select(items, isp["test"])
            trc = [it for it in tr if it.meta["at_cut"]]
            cut_days = np.array([it.meta["cut_day"] for it in trc])
            status = np.array([it.meta["spoiled_at_cut"] for it in trc], float)
            G = current_status_cdf(cut_days, status, 14 + HORIZON + 2)
            day_te = np.array([it.meta["day"] for it in te])
            if name == "age_clock":
                F = age_cdf(G, day_te, 0.0)
            else:
                xtr = np.stack([it.visits[0].feature_vector for it in tr])
                f, sd = ridge(xtr, np.array([it.meta["day"] for it in tr]))
                F = age_cdf(G, f(np.stack([it.visits[0].feature_vector for it in te])), sd)
                # other outputs on the cut day: firmness (log N) and stage, read out by ridge
                xc = np.stack([it.visits[0].feature_vector for it in trc])
                fl, _ = ridge(xc, np.log([it.meta["firmness_n"] for it in trc]))
                tec = [it for it in te if it.meta["at_cut"]]
                pred_f = np.exp(fl(np.stack([it.visits[0].feature_vector for it in tec])))
                for it, pf in zip(tec, pred_f, strict=True):
                    probe.append((it.meta["firmness_n"], float(pf), it.meta["stage"], stage_of(float(pf))))
            F_all.append(F)
            ids_all += [it.item_id for it in te]
            lo_all.append(np.array([it.spoil_lo for it in te]))
            hi_all.append(np.array([it.spoil_hi for it in te]))
            cut_all.append(np.array([it.meta["at_cut"] for it in te]))
        F, lo, hi, cut = (np.concatenate(v) for v in (F_all, lo_all, hi_all, cut_all))
        ids = np.asarray(ids_all)
        for seed in seeds:  # deterministic: same numbers under every seed, written per seed for the summary
            write_result(
                out_root / f"seed{seed}" / name, F, lo, hi, cut, ids, items, fold_list, extra=probe_scores(probe)
            )


def stage_of(firmness_n: float) -> int:
    """Ripeness class from firmness, as in the study: >30 N, 15-30, 3-15, 1-3, <=1."""
    return 1 if firmness_n > 30 else 2 if firmness_n > 15 else 3 if firmness_n > 3 else 4 if firmness_n > 1 else 5


def probe_scores(probe) -> dict:
    if not probe:
        return {}
    t, p, st, sp = (np.array(v) for v in zip(*probe, strict=True))
    return {
        "firmness_mae_n": float(np.mean(np.abs(t - p))),
        "log_firmness_mae": float(np.mean(np.abs(np.log(t) - np.log(p)))),
        "stage_accuracy": float(np.mean(st == sp)),
        "stage_within_one": float(np.mean(np.abs(st - sp) <= 1)),
    }


# ----- results ----------------------------------------------------------------------------------------------------
def write_result(out: Path, F, lo, hi, cut, ids, items, fold_list, extra=None):
    out.mkdir(parents=True, exist_ok=True)
    recs = {r["item_id"]: r for r in fruit_records(items)}
    by_id = {it.item_id: it for it in items}
    cut_ids = [by_id[i].meta["fruit_id"] for i in ids[cut]]
    Fc = F[cut]
    fold_of = {f: k for k, sp in enumerate(fold_list) for f in sp["test"]}
    # population Brier against fold-mates only (fruit never seen in training of that fold's model)
    scores, per_fruit = [], {}
    for k in range(len(fold_list)):
        m = np.array([fold_of[c] == k for c in cut_ids])
        if not m.any():
            continue
        test = [recs[c] for c in np.asarray(cut_ids)[m]]
        sc = cutday_scores(Fc[m], test, test)
        scores.append((sc, int(m.sum())))
    keys = scores[0][0].keys()
    tot = sum(n for _, n in scores)
    agg = {k: float(sum(s[k] * n for s, n in scores) / tot) for k in keys if k not in ("n",)}
    # rank correlations are computed over all fruit at once (more stable than per-fold)
    allsc = cutday_scores(Fc, [recs[c] for c in cut_ids], [recs[c] for c in cut_ids])
    for k in (
        "spearman_firmness",
        "spearman_within_day",
        "status_brier",
        "mean_expected_days_left",
        "mean_predicted_spoil_day",
    ):
        agg[k] = allsc[k]
    for i, c in enumerate(cut_ids):  # per-fruit population-Brier parts for paired bootstrap
        mates = [
            recs[o] for o in cut_ids if fold_of[o] == fold_of[c] and o != c and recs[o]["cut_day"] >= recs[c]["cut_day"]
        ]
        sq = [(Fc[i, int(min(r["cut_day"] - recs[c]["cut_day"], HORIZON))] - float(r["spoiled"])) ** 2 for r in mates]
        per_fruit[c] = (float(np.sum(sq)), len(sq), float((Fc[i, 0] - float(recs[c]["spoiled"])) ** 2))
    metrics = {"cutday": {"n": len(cut_ids), **agg}}
    if (~cut).any():
        metrics["prefix"] = prefix_scores(F[~cut], lo[~cut], hi[~cut])
    if extra:
        metrics["readout"] = extra
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(out / "predictions.npz", F=F, lo=lo, hi=hi, cut=cut, item_ids=ids,
                        pf_fruit=np.array(list(per_fruit)), pf=np.array(list(per_fruit.values())))  # fmt: skip
    print(f"{out}: " + "  ".join(f"{k} {v:.3f}" for k, v in metrics["cutday"].items() if isinstance(v, float)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--setting", default="rgb", choices=[*SETTINGS, "samd"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", default="results/mango")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=None, help="smoke test")
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("overrides", nargs="*", help="model.key=value")
    a = ap.parse_args()
    fold_list = folds()
    if a.folds:
        fold_list = [fold_list[k] for k in a.folds]
    out_root = Path(a.out)
    if a.model == "refs":
        run_refs(fold_list, a.seeds, out_root)
        return
    mcfg = model_cfg(a.model, [o.removeprefix("model.") for o in a.overrides])
    stu_setting = "rgb" if a.setting == "samd" else a.setting
    items = load_mango(stu_setting)
    info = dataset_info(items)
    if a.setting == "samd":
        t_items = load_mango("both")
        t_info = dataset_info(t_items)
        t_by_id = {it.item_id: it for it in t_items}
        pair = {it.item_id: it.item_id for it in items}  # cut-day instance ids coincide across settings
        n_main = info.feat_dim
    for seed in a.seeds:
        out = out_root / f"seed{seed}" / f"{a.model}_{a.setting}"
        out.mkdir(parents=True, exist_ok=True)
        run_cfg = {"model": mcfg, "setting": a.setting, "seed": seed, "rounds": a.rounds, "k": a.k, "beta": a.beta}
        (out / "config.json").write_text(json.dumps(run_cfg, indent=2))
        tcfg = TrainConfig(epochs=a.epochs, batch_size=32, lr=1e-3, patience=10, n_obs_mode="all", seed=seed,
                           max_steps=a.max_steps, log_every=10)  # fmt: skip
        F_all, ids_all, lo_all, hi_all, cut_all = [], [], [], [], []
        for k, sp in enumerate(fold_list):
            isp = fold_item_ids(items, sp)
            tr, va, te = (select(items, isp[s]) for s in ("train", "val", "test"))
            key = jax.random.PRNGKey(1000 * seed + k)
            k_m, k_t, k_e, k_tea = jax.random.split(key, 4)
            model = build_model(mcfg, info, k_m)
            if a.setting == "samd":
                teacher = build_model(mcfg, t_info, k_tea)
                t_tr = select(t_items, fold_item_ids(t_items, sp)["train"])
                smoke = a.max_steps is not None
                model, hist = train_samd(
                    model, teacher, tr, va, t_tr, pair, t_by_id, info, t_info, tcfg, k_t, n_main,
                    rounds=1 if smoke else a.rounds, k=1 if smoke else a.k, beta=a.beta,
                )  # fmt: skip
            else:
                model, hist = train(model, tr, va, info, tcfg, k_t, out_dir=None)
            with open(out / f"history_fold{k}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(hist[0]))
                w.writeheader()
                w.writerows(hist)
            F, lo, hi = predict(model, te, info, k_e)
            F_all.append(F)
            lo_all.append(lo)
            hi_all.append(hi)
            ids_all += [it.item_id for it in te]
            cut_all.append(np.array([it.meta["at_cut"] for it in te]))
        F, lo, hi, cut = (np.concatenate(v) for v in (F_all, lo_all, hi_all, cut_all))
        write_result(out, F, lo, hi, cut, np.asarray(ids_all), items, fold_list)


if __name__ == "__main__":
    main()
