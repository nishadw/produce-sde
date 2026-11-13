"""Mango ripening study (Spring 2026): destructive labels, a daily-ish microwave series, and one photo per fruit.

48 mangoes from one batch were stored together. Every other day (study days 0, 2, ..., 14) every remaining fruit was
microwave-scanned and six were cut open, measured (hue, firmness, sweetness) and photographed; the firmness fixes a
ripeness class (stage 1 Unripe ... 5 Senescent, <= 1 N). ``scripts/data/ingest_mango.py`` builds the manifests.

**Spoilage** is stage 5 (senescent). It is only seen on the cut day, so each fruit is *current-status* data:

* senescent when cut on day ``d``: spoiled in ``(d0, d]``, where ``d0`` is the last study day by which **no** cut
  fruit was senescent (day 10 here: all 30 fruit cut on days 0-10 were stage <= 4). This assumes the batch did not
  start spoiling before the first day any cut fruit was senescent; it is what makes early scans informative.
* not senescent when cut on day ``d``: right-censored at ``d``.

**Instances.** Every instance is a *snapshot*: one visit "now" at ``t = 0`` (no age information), and the target is
the spoilage interval in days from that scan. A fruit gives one instance per scan day (``item_id
mango<NN>_d<DD>``); folds are by fruit. On a scan before the cut day the interval of a senescent fruit is
``(max(d0 - s, 0), d - s]`` (assumed unspoiled at the scan); on the cut day it is "spoiled now"; a fruit not senescent
when cut is right-censored at ``d - s``.

**Inputs** (``setting``), in one feature layout so that a model on a richer setting is a *supermodel* of the
photo-only model (the condition SAMD needs, Shen et al., ICML 2023):

* ``rgb``  — the cut-day instance only (the photo exists then), features ``[rgb]``; its target is spoiled-now or not.
* ``mw``   — every scan day, features ``[mw]`` (current scan and its change since day 0).
* ``both`` — every scan day, ``[rgb | photo-present flag | mw]`` with the photo only on the cut day.

Photo features are the mean of the two side photos' ResNet-18 vectors; microwave features are the per-scan vectors of
the ingest script. Both are reduced by PCA fitted on all fruit **without labels** (unsupervised, so it uses no test
labels), then standardised. Microwave visits carry ``[pc_now, pc_now - pc_day0]`` so the change since the first scan
is explicit. Stage labels sit on the cut-day visit only; they are training targets, never inputs
(``use_stages=false``), and evaluation strips them.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import SPOILED_STAGE, Item, Visit
from .splits import save_splits

MAN = Path("data/manifests")
SETTINGS = ("rgb", "mw", "both", "time")
MW_VIEWS = ("sep1", "sep2", "sep3", "sep4", "sep5", "img")  # scripts/data/ingest_mango_views.py


def _pca(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (mean [D], components [k, D], scale [k]) of standardised-by-feature ``x``."""
    mu = x.mean(0)
    sd = x.std(0) + 1e-6
    z = (x - mu) / sd
    _, s, vt = np.linalg.svd(z, full_matrices=False)
    k = min(k, vt.shape[0])
    comp = vt[:k] / sd[None, :]  # project raw x - mu
    proj = (x - mu) @ comp.T
    return mu, comp, proj.std(0) + 1e-6


def spoil_free_until(fruit: pd.DataFrame) -> float:
    """Last study day by which no cut fruit was senescent (every fruit cut on or before it was stage < 5)."""
    days = sorted(fruit.cut_day.unique())
    last = -math.inf
    for d in days:
        if (fruit[fruit.cut_day <= d].stage >= SPOILED_STAGE).any():
            break
        last = float(d)
    return last


def mango_tables(manifests: str | Path = MAN) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    man = Path(manifests)
    return (
        pd.read_csv(man / "mango_fruit.csv"),
        pd.read_csv(man / "mango_photos.csv"),
        pd.read_csv(man / "mango_scans.csv"),
    )


def load_mango(
    setting: str = "both",
    manifests: str | Path = MAN,
    features_file: str | Path = "data/features/mango_resnet18.npz",
    scans_file: str | Path = "data/mango/mango_scans.npz",
    n_pc_rgb: int = 16,
    n_pc_mw: int = 16,
    features: dict[str, np.ndarray] | None = None,
    scans: dict[str, np.ndarray] | None = None,
    mw_view: str | None = None,
    photo_side: int | None = None,
    history: bool = False,
    views_file: str | Path = "data/mango/mango_scan_views.npz",
) -> list[Item]:
    """Snapshot items (see the module docstring); ``meta['fruit_id']`` groups the instances of one fruit.

    ``mw_view``: ``None`` (all 45 antenna pairs averaged + image, as first presented), one of ``MW_VIEWS``, a pooled
    set ``"sep1+sep2"`` (per-feature mean and sd over the views), or ``"concat"`` (the six side by side). PCA is
    fitted per setting on all scans, without labels.

    ``photo_side``: ``None`` (mean of both side photos), 1 or 2 (falls back to the other side for the 3 fruit with
    one photo).

    ``history``: every instance carries the fruit's scans up to now as visits at their study days (a time series);
    the target interval is in study days, so the forecast is from the last visit. Off: one visit at ``t = 0`` (no
    age information). The ``rgb`` and ``time`` settings put the cut-day photo (or nothing) on the last visit and
    empty features on earlier ones, so a photo-only model is a sub-model of photo + microwave on the same visits.
    ``time``: no features at all (with ``history``: elapsed time only, the age control)."""
    if setting not in SETTINGS:
        raise ValueError(f"setting must be one of {SETTINGS}")
    fruit, photos, scan_rows = mango_tables(manifests)
    if features is None:
        with np.load(features_file) as z:
            features = {k: z[k] for k in z.files}
    if scans is None:
        with np.load(scans_file) as z:
            scans = {k: z[k] for k in z.files}

    # photo: mean over the fruit's side photos, then PCA over fruit
    ids = fruit.fruit.tolist()

    def photo_vec(f):
        ph = photos[photos.fruit == f]
        if photo_side is not None and (ph.side == photo_side).any():
            ph = ph[ph.side == photo_side]
        return np.mean([features[Path(p).name] for p in ph.path], axis=0)

    rgb_raw = np.stack([photo_vec(f) for f in ids]).astype(np.float64)
    mu, comp, sc = _pca(rgb_raw, n_pc_rgb)
    rgb = {f: ((rgb_raw[i] - mu) @ comp.T / sc).astype(np.float32) for i, f in enumerate(ids)}

    # microwave: PCA over every scan; per visit [pc_now, pc_now - pc_day0]
    keys = scan_rows.key.tolist()
    if mw_view is not None:
        with np.load(views_file) as z:
            vz = {k: z[k] for k in z.files}
        if mw_view == "concat":
            mw_raw = np.stack([np.concatenate([vz[f"{k}/{v}"] for v in MW_VIEWS]) for k in keys])
        elif "+" in mw_view:
            vs = mw_view.split("+")
            if any(v not in MW_VIEWS[:5] for v in vs):
                raise ValueError(f"pooled views must be antenna separations, got {mw_view}")
            stack = np.stack([[vz[f"{k}/{v}"] for v in vs] for k in keys])  # [N, V, D]
            mw_raw = np.concatenate([stack.mean(1), stack.std(1)], axis=-1)
        elif mw_view in MW_VIEWS:
            mw_raw = np.stack([vz[f"{k}/{mw_view}"] for k in keys])
        else:
            raise ValueError(f"mw_view must be None, concat, a pooled set or one of {MW_VIEWS}")
        mw_raw = mw_raw.astype(np.float64)
    else:
        mw_raw = np.stack([scans[k] for k in keys]).astype(np.float64)
    keep = mw_raw.std(0) > 1e-9  # constant columns (e.g. no image) carry nothing
    mw_raw = mw_raw[:, keep]
    mu, comp, sc = _pca(mw_raw, n_pc_mw)
    pc = {k: ((mw_raw[i] - mu) @ comp.T / sc).astype(np.float32) for i, k in enumerate(keys)}

    d0 = spoil_free_until(fruit)
    items: list[Item] = []
    for r in fruit.itertuples(index=False):
        f, d, stage = int(r.fruit), float(r.cut_day), int(r.stage)
        spoiled = stage >= SPOILED_STAGE
        rows = scan_rows[scan_rows.fruit == f].sort_values("day")
        first = pc[rows.key.iloc[0]]

        def visit_x(key, at_cut, f=f, first=first):
            mw = np.concatenate([pc[key], pc[key] - first])
            if setting == "time":
                return None
            if setting == "rgb":
                return rgb[f] if at_cut else np.zeros_like(rgb[f])
            if setting == "mw":
                return mw
            photo = rgb[f] if at_cut else np.zeros_like(rgb[f])
            return np.concatenate([photo, [1.0 if at_cut else 0.0], mw])

        for s in rows.itertuples(index=False):
            day = float(s.day)
            at_cut = day == d
            if setting in ("rgb", "time") and not at_cut and not (setting == "time" and history):
                continue  # the photo exists on the cut day only; a time-only snapshot carries nothing
            if setting == "rgb" and not at_cut:
                continue
            # spoilage interval in days from this scan
            if not spoiled:
                lo, hi = d - day, math.inf
            elif at_cut:
                lo = hi = 0.0  # spoiled now: zero width at 0 (the models widen it to the first grid step)
            else:
                lo, hi = max(min(d0, d) - day, 0.0), d - day
            if history:
                past = rows[rows.day <= day]
                visits = []
                for q in past.itertuples(index=False):
                    now = float(q.day) == day
                    x = visit_x(q.key, at_cut and now)
                    visits.append(Visit(t=float(q.day), stage=stage if (at_cut and now) else None,
                                        features=None if x is None else np.asarray(x, np.float32)[None]))  # fmt: skip
                lo, hi = lo + day, hi + day  # study days: the forecast is from the last visit (t_now = day)
            else:
                x = visit_x(s.key, at_cut)
                visits = [Visit(t=0.0, stage=stage if at_cut else None,
                                features=None if x is None else np.asarray(x, np.float32)[None])]  # fmt: skip
            items.append(
                Item(
                    item_id=f"mango{f:02d}_d{int(day):02d}",
                    produce="mango",
                    storage="ambient",
                    temp_c=math.nan,
                    visits=visits,
                    spoil_lo=lo,
                    spoil_hi=hi,
                    meta={
                        "source": "mango_spring2026",
                        "fruit": f,
                        "fruit_id": f"mango{f:02d}",
                        "day": day,
                        "at_cut": at_cut,
                        "cut_day": d,
                        "stage": stage,
                        "firmness_n": float(r.firmness_n),
                        "hue": float(r.hue),
                        "brix": float(r.brix),
                        "spoiled_at_cut": spoiled,
                        "spoil_free_until": d0,
                        "setting": setting,
                        "mw_view": mw_view,
                        "photo_side": photo_side,
                        "history": history,
                    },
                )
            )
    return items


def strip_stages(items: list[Item]) -> list[Item]:
    """The same items with every stage label removed (what a model may see at prediction time)."""
    return [replace(it, visits=[replace(v, stage=None) for v in it.visits]) for it in items]


def mango_folds(fruit: pd.DataFrame, n_folds: int = 6, seed: int = 0) -> list[dict[str, list[str]]]:
    """Cross-validation stratified by cut day: within each cut day the fruit are shuffled and dealt to the folds,
    so every test fold holds one fruit per study day (with 6 fruit per day and 6 folds). Validation is the next fold;
    the rest train."""
    rng = np.random.default_rng(seed)
    fold_of: dict[str, int] = {}
    for _, g in fruit.groupby("cut_day"):
        ids = [f"mango{int(f):02d}" for f in g.fruit]  # fruit ids; expand to instances with fold_item_ids
        rng.shuffle(ids)
        for i, iid in enumerate(ids):
            fold_of[iid] = i % n_folds
    out = []
    for k in range(n_folds):
        test = sorted(i for i, f in fold_of.items() if f == k)
        val = sorted(i for i, f in fold_of.items() if f == (k + 1) % n_folds)
        train = sorted(i for i, f in fold_of.items() if f not in (k, (k + 1) % n_folds))
        out.append({"train": train, "val": val, "test": test})
    return out


def fold_item_ids(items: list[Item], fold: dict[str, list[str]]) -> dict[str, list[str]]:
    """Fruit-level fold -> instance ids of ``items`` (every scan of a fruit stays on the same side)."""
    return {k: [it.item_id for it in items if it.meta["fruit_id"] in set(v)] for k, v in fold.items()}


def write_mango_folds(
    manifests: str | Path = MAN, out_dir: str | Path = "configs/splits", n_folds: int = 6
) -> list[Path]:
    fruit, _, _ = mango_tables(manifests)
    paths = []
    for k, sp in enumerate(mango_folds(fruit, n_folds)):
        p = Path(out_dir) / f"mango_fold{k}.json"
        save_splits(sp, p, extra={"mode": "cv", "fold": k, "n_folds": n_folds, "data": "mango"})
        paths.append(p)
    return paths
