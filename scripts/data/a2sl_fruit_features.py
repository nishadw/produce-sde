"""Fruit-only photo features for the A2SL avocados (background removed), plus a ripeness reader trained on the public
avocado set. Writes, keyed like the A2SL cache (``ripen.data.a2sl.photo_key``):

* data/features/a2sl_fruit_resnet18.npz   ResNet-18 on the cut-out fruit (grey background, cropped to the fruit)
* data/features/a2sl_colour.npz           12 skin-colour numbers (``ripen.data.fruit_mask.colour_features``), z-scored
                                          over every A2SL photo (label-free)
* data/features/a2sl_public_stage.npz     a ripeness-stage reader (ridge multinomial logistic on 64 PCA components of
                                          fruit-only ResNet-18) trained on the 478 public avocados' photo stages
                                          (Ripening Index 1-5), applied to the A2SL photos: 5 log-probabilities + the
                                          expected stage. Uses no A2SL label.

python scripts/data/a2sl_fruit_features.py [--device cuda]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import minimize

from ripen.data.a2sl import load_a2sl_avocado, photo_key
from ripen.data.avocado import load_avocado
from ripen.data.features import extract_features, save_feature_cache
from ripen.data.fruit_mask import colour_features, fruit_mask, fruit_only

OUT = Path("data/features")
CUT = Path("data/fruit_only")


def cut_all(paths: list[str], root: Path, keep: int) -> tuple[list[str], dict[str, np.ndarray]]:
    """Write each photo's fruit-only crop under ``root`` (same last two path parts); return new paths and colours."""
    out, col = [], {}
    for i, p in enumerate(paths):
        q = root / Path(p).parent.name / Path(p).name
        im = Image.open(p)
        im.thumbnail((keep, keep))
        m = fruit_mask(im)
        col[p] = colour_features(im, m)
        if not q.exists():
            q.parent.mkdir(parents=True, exist_ok=True)
            fruit_only(im, m).save(q, quality=95)
        out.append(str(q))
        if i % 1000 == 0:
            print(f"  cut {i}/{len(paths)}", flush=True)
    return out, col


def fit_logistic(Z: np.ndarray, y: np.ndarray, K: int, ridge: float) -> np.ndarray:
    Zn = np.c_[Z, np.ones(len(Z))]
    Y = np.eye(K)[y]
    q = Zn.shape[1]

    def f(wf):
        W = wf.reshape(q, K)
        L = Zn @ W
        L -= L.max(1, keepdims=True)
        lse = np.log(np.exp(L).sum(1))
        P = np.exp(L - lse[:, None])
        reg = np.r_[W[:-1], 0 * W[-1:]]
        return -np.sum(Y * (L - lse[:, None])) + ridge * np.sum(W[:-1] ** 2), (Zn.T @ (P - Y) + 2 * ridge * reg).ravel()

    return minimize(f, np.zeros(q * K), jac=True, method="L-BFGS-B", options={"maxiter": 500}).x.reshape(q, K)


def logp(Z: np.ndarray, W: np.ndarray) -> np.ndarray:
    L = np.c_[Z, np.ones(len(Z))] @ W
    return L - np.log(np.exp(L - L.max(1, keepdims=True)).sum(1, keepdims=True)) - L.max(1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    # ---- A2SL ----
    lab = sorted({p for it in load_a2sl_avocado(labels=None) for v in it.visits for p in v.image_paths})
    print(f"A2SL: {len(lab)} photos", flush=True)
    lab_cut, lab_col = cut_all(lab, CUT / "a2sl", 1024)
    f_lab = extract_features(lab_cut, "resnet18", 32, a.device, key_fn=photo_key)
    save_feature_cache(OUT / "a2sl_fruit_resnet18.npz", f_lab)
    C = np.stack([lab_col[p] for p in lab])
    mu, sd = C.mean(0), C.std(0) + 1e-6
    colour = {photo_key(p): ((lab_col[p] - mu) / sd).astype(np.float32) for p in lab}
    save_feature_cache(OUT / "a2sl_colour.npz", colour)
    # ---- public avocados: fruit-only features + Ripening Index stage per photo ----
    items = load_avocado("data/avocado")
    rows = [(it.item_id, p, v.stage) for it in items for v in it.visits if v.stage for p in v.image_paths
            if Path(p).exists()]  # fmt: skip
    print(f"public: {len(rows)} labelled photos of {len({r[0] for r in rows})} fruit", flush=True)
    pub_cut, _ = cut_all([r[1] for r in rows], CUT / "public", 512)
    f_pub = extract_features(pub_cut, "resnet18", 64, a.device, key_fn=lambda p: Path(p).name)
    X = np.stack([f_pub[Path(p).name] for p in pub_cut]).astype(np.float64)
    y = np.array([r[2] for r in rows]) - 1
    fruit = np.array([r[0] for r in rows])
    m = X.mean(0)
    _, _, vt = np.linalg.svd(X - m, full_matrices=False)
    V = vt[:64]
    Z = (X - m) @ V.T
    zs = Z.std(0) + 1e-6
    Z /= zs
    # ridge by a held-out 20 % of public fruit
    u = np.unique(fruit)
    rng = np.random.default_rng(0)
    va = np.isin(fruit, rng.choice(u, len(u) // 5, replace=False))
    best = None
    for r in (0.1, 1.0, 10.0, 100.0, 1000.0):
        W = fit_logistic(Z[~va], y[~va], 5, r)
        nll = -np.mean(logp(Z[va], W)[np.arange(va.sum()), y[va]])
        acc = np.mean(logp(Z[va], W).argmax(1) == y[va])
        print(f"  ridge {r}: held-out public NLL {nll:.3f}, accuracy {acc:.3f}", flush=True)
        if best is None or nll < best[0]:
            best = (nll, r)
    W = fit_logistic(Z, y, 5, best[1])
    XL = np.stack([f_lab[photo_key(p)] for p in lab]).astype(np.float64)
    LP = logp(((XL - m) @ V.T) / zs, W)
    es = (np.exp(LP) * np.arange(1, 6)).sum(1, keepdims=True)
    feats = np.c_[LP, es].astype(np.float32)
    save_feature_cache(OUT / "a2sl_public_stage.npz", {photo_key(p): feats[i] for i, p in enumerate(lab)})
    print("A2SL expected stage by day folder:")
    for d in sorted({Path(p).parent.name for p in lab}):
        sel = [i for i, p in enumerate(lab) if Path(p).parent.name == d]
        print(f"  {d}: {es[sel, 0].mean():.2f}")
    print("wrote", [str(OUT / f) for f in ("a2sl_fruit_resnet18.npz", "a2sl_colour.npz", "a2sl_public_stage.npz")])


if __name__ == "__main__":
    main()
