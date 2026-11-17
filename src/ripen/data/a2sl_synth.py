"""Half-synthetic A2SL avocados: real visits on a window every kept fruit shares, regression-extrapolated after it.

docs/plans/2026-11-17_a2sl-imputation-random-start.md (revised 2026-11-17). The rule:

1. Keep the fruit photographed on every day of ``window`` (days 1, 3, 5: avo01-03, 06-11, 9 fruit). The others
   (avo04, 05 cut early; avo12-15 started late) are dropped.
2. Fit a regression on the window only, pooled over the kept fruit, and extrapolate every kept fruit, including the 4
   tracked to the end, to ``last_day`` (default 15). Real visits after the window are ignored.
   * ``linear``: per component, a slope in study day shared by all fruit + each fruit's own offset;
   * ``logistic``: per component an S-curve ``lo + amp * sigmoid((day - m - s_i) / w)`` with the shape shared and a
     time shift ``s_i`` per fruit (its "biological age"), fit jointly by robust least squares.
   Photos are fit in a ``k``-component PCA space of the window photos and mapped back with each fruit's own mean
   off-PCA residual. Scan summaries are fit the same way (own PCA). A fruit with no window scan gets the population
   curve (offset / shift 0); a window day without a scan is filled from the fit too.
3. Labels follow the day rule (``JMW2026_DAY_STAGE``; day 15 is stage 5), so every fruit spoils in (day 7, day 9].
   Visits after the first spoiled one are kept (``meta["post_spoil_visits"]``); ``collate.choose_n_obs`` never shows
   a model more than the first spoiled visit, because a spoilage forecast made after the fruit spoiled has no target.

The data rule fits on every kept fruit, test fruit included: the fit sees photos (days 1-5) only, never labels, and
a test forecast sees those same photos anyway. Score on the real tracked visits too (``train.py`` writes
``*_test_real``); a score on synthetic test visits measures agreement with the regression curve.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .a2sl import JMW2026_DAY_STAGE, MAN
from .schema import Item, Visit, spoil_interval_from_visits

DAY_STAGE = {**JMW2026_DAY_STAGE, 15: 5}


def _sig(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # = 1 / (1 + exp(-x)), no overflow


class _Fit:
    """Window regression of one feature block (photo or scan) in a PCA space of the window observations."""

    def __init__(self, obs: dict[str, dict[int, np.ndarray]], method: str, k: int):
        X = np.stack([x for d in obs.values() for x in d.values()])
        self.mu = X.mean(0)
        _, _, vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.V = vt[: min(k, vt.shape[0])]
        self.method = method
        self.resid = {f: np.mean([x - self.mu - self._z(x) @ self.V for x in d.values()], 0) for f, d in obs.items()}
        Z = {f: {t: self._z(x) for t, x in d.items()} for f, d in obs.items()}
        (self._fit_linear if method == "linear" else self._fit_logistic)(Z)

    def _z(self, x):
        return (x - self.mu) @ self.V.T

    def _fit_linear(self, Z):
        num, den = 0.0, 0.0
        for d in Z.values():
            if len(d) < 2:
                continue
            t = np.array(sorted(d), float)
            z = np.stack([d[int(s)] for s in t])
            tc = t - t.mean()
            num, den = num + tc @ (z - z.mean(0)), den + tc @ tc
        self.b = num / den if den > 0 else np.zeros(self.V.shape[0])
        self.a = {f: np.mean([z - self.b * t for t, z in d.items()], 0) for f, d in Z.items()}
        self.a0 = np.mean(list(self.a.values()), 0)

    def _fit_logistic(self, Z):
        fruits = sorted(Z)
        rows = [(i, t, z) for i, f in enumerate(fruits) for t, z in sorted(Z[f].items())]
        allz = np.stack([z for _, _, z in rows])
        q = allz.shape[1]

        def unpack(p):
            s = p[4 * q :]
            return p[:q], p[q : 2 * q], p[2 * q : 3 * q], np.exp(p[3 * q : 4 * q]), s - s.mean()

        def resid(p):
            lo, amp, m, w, s = unpack(p)
            r = [lo + amp * _sig((t - m - s[i]) / w) - z for i, t, z in rows]
            return np.concatenate([*r, 0.1 * s])  # light ridge on the per-fruit shifts

        span = allz.max(0) - allz.min(0) + 1e-3
        p0 = np.concatenate([allz.min(0), span, np.full(q, 5.0), np.full(q, np.log(2.0)), np.zeros(len(fruits))])
        sol = least_squares(resid, p0, loss="soft_l1", max_nfev=5000)
        self.lo, self.amp, self.m, self.w, s = unpack(sol.x)
        self.shift = dict(zip(fruits, s, strict=True))

    def predict(self, fruit: str | None, days) -> dict[int, np.ndarray]:
        t = np.array(days, float)
        if self.method == "linear":
            a = self.a.get(fruit, self.a0)
            Z = a + np.outer(t, self.b)
        else:
            s = self.shift.get(fruit, 0.0)
            Z = self.lo + self.amp * _sig((t[:, None] - self.m - s) / self.w)
        r = self.resid.get(fruit, 0.0)
        return {int(d): self.mu + z @ self.V + r for d, z in zip(days, Z, strict=True)}


def _observed(man: Path) -> dict[str, dict[int, tuple[bool, bool]]]:
    v = pd.read_csv(man / "a2sl_avocado_visits.csv")
    out: dict[str, dict[int, tuple[bool, bool]]] = {}
    for r in v.itertuples(index=False):
        out.setdefault(r.item_id, {})[int(r.day)] = (isinstance(r.side_photo, str), isinstance(r.scan_keys, str))
    return out


def build_half_synthetic(
    items: list[Item],
    method: str = "linear",
    photo_dim: int = 512,
    has_scans: bool = True,
    k: int = 4,
    window: tuple[int, ...] = (1, 3, 5),
    last_day: int = 15,
    manifests: str | Path = MAN,
) -> list[Item]:
    """The half-synthetic set (module docstring) from the loader's real ``items`` (``load_a2sl_avocado``)."""
    if method not in ("linear", "logistic"):
        raise ValueError(f"unknown synthetic method {method!r}")
    seen = _observed(Path(manifests))
    keep = [it for it in items if it.item_id in seen and all(seen[it.item_id].get(d, (False,))[0] for d in window)]
    day0 = {it.item_id: min(seen[it.item_id]) for it in keep}
    with_feat = [v for it in keep for v in it.visits if v.features is not None]
    feat_dim = with_feat[0].features.shape[-1] if with_feat else 0
    scan_dim = feat_dim - photo_dim if has_scans else 0
    photos: dict[str, dict[int, np.ndarray]] = {}
    scans: dict[str, dict[int, np.ndarray]] = {}
    for it in keep:
        f = it.item_id
        for v in it.visits:
            day = int(round(v.t + day0[f]))
            if v.features is None or day not in window:
                continue
            x = v.feature_vector
            photos.setdefault(f, {})[day] = x[:photo_dim]
            if scan_dim > 0 and seen[f][day][1]:
                scans.setdefault(f, {})[day] = x[photo_dim : photo_dim + scan_dim]
    fp = _Fit(photos, method, k) if photos and photo_dim > 0 else None
    fs = _Fit(scans, method, k) if scans else None
    days = [d for d in range(window[0], last_day + 1, 2)]
    out: list[Item] = []
    for it in keep:
        f = it.item_id
        labelled = it.meta.get("label_source") == "jmw2026"
        pp = fp.predict(f, days) if fp is not None else {}
        ps = fs.predict(f if f in scans else None, days) if fs is not None else {}
        vs, synth = [], []
        for d in days:
            if feat_dim:
                p = photos.get(f, {}).get(d, pp.get(d))
                parts = [p if p is not None else np.zeros(photo_dim, np.float32)]
                if feat_dim > photo_dim:
                    s = scans.get(f, {}).get(d) if scan_dim > 0 else None
                    s = ps.get(d) if s is None and ps else s
                    parts.append(s if s is not None else np.zeros(feat_dim - photo_dim, np.float32))
                x = np.concatenate(parts)[None].astype(np.float32)
            else:
                x = None
            if d not in window or (scan_dim > 0 and d not in scans.get(f, {})):
                synth.append(d)
            vs.append(Visit(t=float(d - days[0]), stage=DAY_STAGE.get(d) if labelled else None, features=x))
        lo, hi = spoil_interval_from_visits(vs) if labelled else (vs[-1].t, math.inf)
        meta = {**it.meta, "synthetic": method, "synthetic_days": synth, "post_spoil_visits": True}
        out.append(dataclasses.replace(it, visits=vs, spoil_lo=lo, spoil_hi=hi, meta=meta))
    return out
