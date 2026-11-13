"""Late fusion of per-view forecasts (TMC-style; Han et al., ICLR 2021): no new training.

Each view's model gives its own forecast of days-until-spoiled; fuse them per fruit and moment:

* ``mean``  linear pool, the average of the views' distributions;
* ``geo``   log pool, the normalised geometric mean (views must agree for a bin to keep mass);
* ``trust`` linear pool weighted per forecast by ``exp(-entropy)``: confident views count more, the role TMC's
  Dempster-Shafer combination gives the views' uncertainty masses.

Avocado (fold layout of ``a2sl_views.py``):
    python scripts/analysis/late_fusion.py fold --root outputs/a2sl_views/labels_hidden_std --model latent_sde \
        --input mw --views 0 60 120 180 240 300
Mango (one run per seed, ``mango_cv.py`` layout):
    python scripts/analysis/late_fusion.py mango --root outputs/mango_views --runs <run>... --name <fused run>
Writes a run named ``<model>_<input>_viewfuse-<kind>`` (avocado) or ``<name>-<kind>`` (mango) next to the inputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

KINDS = ("mean", "geo", "trust")


def pmf(F: np.ndarray) -> np.ndarray:
    p = np.diff(F, axis=-1, prepend=0.0)
    return np.clip(np.concatenate([p, 1.0 - F[..., -1:]], axis=-1), 1e-9, None)


def fuse(Fs: list[np.ndarray], kind: str) -> np.ndarray:
    """``V x [N, H+1]`` CDFs -> ``[N, H+1]``."""
    P = np.stack([pmf(F) for F in Fs])  # [V, N, H+2]
    P = P / P.sum(-1, keepdims=True)
    if kind == "mean":
        q = P.mean(0)
    elif kind == "geo":
        q = np.exp(np.log(P).mean(0))
    elif kind == "trust":
        ent = -(P * np.log(P)).sum(-1)  # [V, N]
        w = np.exp(-ent)
        w = w / w.sum(0, keepdims=True)
        q = (w[..., None] * P).sum(0)
    else:
        raise ValueError(kind)
    q = q / q.sum(-1, keepdims=True)
    return np.clip(np.cumsum(q, -1)[..., :-1], 0.0, 1.0)


def _key(z):
    return [f"{i}|{n}" for i, n in zip(z["item_ids"], z["n_obs"], strict=True)]


def fold_mode(a) -> None:
    root = Path(a.root)
    for sd in sorted(root.glob("seed*")):
        runs = [sd / f"{a.model}_{a.input}_view{v}" for v in a.views]
        if not all(r.exists() for r in runs):
            print(f"{sd.name}: missing a view, skipped", file=sys.stderr)
            continue
        for fold in sorted(p.name for p in runs[0].glob("fold*")):
            for split in ("test", "val"):
                zs = [dict(np.load(r / fold / f"predictions_{split}.npz", allow_pickle=True)) for r in runs]
                ref = _key(zs[0])
                Fs = []
                for z in zs:
                    idx = {k: i for i, k in enumerate(_key(z))}
                    Fs.append(z["F"][[idx[k] for k in ref]])
                for kind in KINDS:
                    out = sd / f"{a.model}_{a.input}_viewfuse-{kind}" / fold
                    out.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(out / f"predictions_{split}.npz", **{**zs[0], "F": fuse(Fs, kind)})
                    (out / "metrics_test.json").touch()
        print(f"{sd.name}: fused {len(a.views)} views of {a.model}_{a.input}")


def mango_mode(a) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "run"))
    from mango_cv import folds, write_result

    from ripen.data.mango import load_mango

    items = load_mango(a.setting, history=a.history)  # ids and labels only; the views' features do not matter here
    fold_list = folds()
    root = Path(a.root)
    for sd in sorted(root.glob("seed*")):
        runs = [sd / r for r in a.runs]
        if not all((r / "predictions.npz").exists() for r in runs):
            print(f"{sd.name}: missing a view, skipped", file=sys.stderr)
            continue
        zs = [dict(np.load(r / "predictions.npz", allow_pickle=True)) for r in runs]
        ref = list(zs[0]["item_ids"])
        Fs = []
        for z in zs:
            idx = {k: i for i, k in enumerate(z["item_ids"])}
            Fs.append(z["F"][[idx[k] for k in ref]])
        for kind in KINDS:
            z = zs[0]
            write_result(sd / f"{a.name}-{kind}", fuse(Fs, kind), z["lo"], z["hi"], z["cut"], z["item_ids"], items,
                         fold_list)  # fmt: skip
        print(f"{sd.name}: fused {len(runs)} runs -> {a.name}-*")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("fold")
    f.add_argument("--root", required=True)
    f.add_argument("--model", required=True)
    f.add_argument("--input", required=True)
    f.add_argument("--views", nargs="+", default=["0", "60", "120", "180", "240", "300"])
    m = sub.add_parser("mango")
    m.add_argument("--root", required=True)
    m.add_argument("--runs", nargs="+", required=True)
    m.add_argument("--name", required=True)
    m.add_argument("--setting", default="mw")
    m.add_argument("--history", action="store_true")
    a = ap.parse_args()
    fold_mode(a) if a.mode == "fold" else mango_mode(a)


if __name__ == "__main__":
    main()
