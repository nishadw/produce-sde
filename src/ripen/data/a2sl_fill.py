"""Fill in the missing visits of A2SL training fruit (random-start helpers: ``ripen.data.schema.shift_start``).

docs/plans/2026-11-17_a2sl-imputation-random-start.md, steps 4-5. Only *training* fruit are filled, from a fill fit on
the training fruit alone, so validation and test scores still come from real visits only.

Fill (``method``):

* ``linear``: per feature, a slope in study day shared by every training fruit plus each fruit's own offset (a
  random-intercept linear model), fit on every observed visit. Photos are filled in a ``k``-component PCA space and
  mapped back with the fruit's own mean off-PCA residual; scan summaries are filled directly. The extrapolation check
  (``scripts/analysis/a2sl_impute.py``) found this the best fill, ahead of an S-curve and of fitting on days 1-5 only.
* ``carry``: the nearest observed visit's value (forward, and backward for late starters). The control arm.

Each filled fruit gets visits on every study day of the schedule from day 1 (``backward=True``; otherwise from its
first visit) to its last labelled day, labels from the same day rule as the loader, and ``meta["filled_days"]``.
``copies > 1`` adds noisy copies (``<id>#<c>``) whose filled values get Gaussian noise at the fit's residual scale
(multiple imputation).
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .a2sl import JMW2026_DAY_STAGE, MAN
from .schema import Item, Visit, spoil_interval_from_visits

SCHEDULE = (1, 3, 5, 7, 9, 11, 13)


def _observed(man: Path) -> dict[str, dict[int, tuple[bool, bool]]]:
    """{fruit: {study day: (has photo, has scan)}} from the visits manifest."""
    v = pd.read_csv(man / "a2sl_avocado_visits.csv")
    out: dict[str, dict[int, tuple[bool, bool]]] = {}
    for r in v.itertuples(index=False):
        out.setdefault(r.item_id, {})[int(r.day)] = (isinstance(r.side_photo, str), isinstance(r.scan_keys, str))
    return out


class _LinearBlock:
    """Shared slope + per-fruit offset, optionally in a PCA space of the block."""

    def __init__(self, obs: dict[str, dict[int, np.ndarray]], k: int | None):
        X = np.stack([x for d in obs.values() for x in d.values()])
        self.mu = X.mean(0)
        if k is not None and k < X.shape[1]:
            _, _, vt = np.linalg.svd(X - self.mu, full_matrices=False)
            self.V = vt[:k]
        else:
            self.V = np.eye(X.shape[1])
        Z = {f: {t: self._z(x) for t, x in d.items()} for f, d in obs.items()}
        num, den = 0.0, 0.0
        for d in Z.values():
            if len(d) < 2:
                continue
            t = np.array(sorted(d), float)
            z = np.stack([d[int(s)] for s in t])
            tc = t - t.mean()
            num = num + tc @ (z - z.mean(0))
            den += tc @ tc
        self.b = num / den if den > 0 else np.zeros(self.V.shape[0])
        res = [z - self._offset(d) - self.b * t for d in Z.values() if len(d) >= 2 for t, z in d.items()]
        self.sd = np.std(np.stack(res), 0) if res else np.zeros(self.V.shape[0])

    def _z(self, x):
        return (x - self.mu) @ self.V.T

    def _offset(self, zd: dict[int, np.ndarray]) -> np.ndarray:
        return np.mean([z - self.b * t for t, z in zd.items()], 0)

    def fill(self, d_obs: dict[int, np.ndarray], days) -> dict[int, np.ndarray]:
        zd = {t: self._z(x) for t, x in d_obs.items()}
        a = self._offset(zd)
        resid = np.mean([x - self.mu - zd[t] @ self.V for t, x in d_obs.items()], 0)  # the fruit's off-PCA part
        return {t: self.mu + (a + self.b * t) @ self.V + resid for t in days}

    def noise(self, rng: np.random.Generator) -> np.ndarray:
        return (rng.normal(size=self.sd.shape) * self.sd) @ self.V


def _carry(d_obs: dict[int, np.ndarray], days) -> dict[int, np.ndarray]:
    obs = sorted(d_obs)
    return {t: d_obs[min(obs, key=lambda s: (abs(s - t), -s))] for t in days}


def fill_items(
    items: list[Item],
    method: str = "linear",
    photo_dim: int = 512,
    k: int = 4,
    backward: bool = True,
    copies: int = 1,
    seed: int = 0,
    manifests: str | Path = MAN,
) -> list[Item]:
    """Return ``items`` with missing visits filled (see the module docstring). Fit on ``items`` only: pass the
    training fruit. Non-A2SL items and fruit with no photo are returned unchanged."""
    if method not in ("linear", "carry"):
        raise ValueError(f"unknown fill method {method!r}")
    seen = _observed(Path(manifests))
    by_id = {it.item_id: it for it in items if it.item_id in seen}
    day0 = {f: min(seen[f]) for f in by_id}
    feat_dim = next(v.features.shape[-1] for it in by_id.values() for v in it.visits if v.features is not None)
    scan_dim = feat_dim - photo_dim
    photos: dict[str, dict[int, np.ndarray]] = {}
    scans: dict[str, dict[int, np.ndarray]] = {}
    for f, it in by_id.items():
        for v in it.visits:
            if v.features is None:
                continue
            day = int(round(v.t + day0[f]))
            has_p, has_s = seen[f].get(day, (False, False))
            x = v.feature_vector
            if has_p and photo_dim > 0:
                photos.setdefault(f, {})[day] = x[:photo_dim]
            if has_s and scan_dim > 0:
                scans.setdefault(f, {})[day] = x[photo_dim:]
    blocks = {}
    if method == "linear":
        blocks["p"] = _LinearBlock(photos, k) if photos else None
        blocks["s"] = _LinearBlock(scans, None) if scans and any(len(d) >= 2 for d in scans.values()) else None
    rng = np.random.default_rng(seed)
    out: list[Item] = []
    for it in items:
        f = it.item_id
        if f not in by_id or f not in photos:
            out.append(it)
            continue
        labelled = it.meta.get("label_source") == "jmw2026"
        last = max(int(round(v.t + day0[f])) for v in it.visits)
        first = SCHEDULE[0] if backward else day0[f]
        end = 9 if labelled else last  # labelled fruit stop at the first stage-5 visit: day 9 under the day rule
        days = [d for d in SCHEDULE if first <= d <= max(end, last)]

        def fill_block(name, obs_d, days=days):
            if obs_d is None:
                return {}
            need = [d for d in days if d not in obs_d]
            if not need:
                return {}
            if method == "carry" or blocks.get(name) is None:
                return _carry(obs_d, need)
            return blocks[name].fill(obs_d, need)

        fp = fill_block("p", photos.get(f))
        fs = fill_block("s", scans.get(f)) if scan_dim > 0 and f in scans else {}
        for c in range(copies):
            vs, filled = [], []
            for d in days:
                p = photos[f].get(d)
                if p is None:  # filled; copies after the first get noise at the fit's residual scale
                    noisy = c > 0 and blocks.get("p") is not None
                    p = fp[d] + (blocks["p"].noise(rng) if noisy else 0.0)
                parts = [p]
                if scan_dim > 0:
                    s = scans.get(f, {}).get(d)
                    if s is None and d in fs:
                        s = fs[d] + (blocks["s"].noise(rng) if c > 0 and blocks.get("s") is not None else 0.0)
                    if s is None:
                        s = _observed_scan_block(it, d, day0[f], photo_dim)
                    parts.append(s)
                was_real = d in photos[f]
                if not was_real or (scan_dim > 0 and d in fs):
                    filled.append(d)
                stage = JMW2026_DAY_STAGE.get(d) if labelled else None
                x = np.concatenate(parts)[None].astype(np.float32)
                vs.append(Visit(t=float(d - days[0]), stage=stage, features=x, image_paths=_paths(it, d, day0[f])))
                if stage is not None and stage >= 5:
                    break
            lo, hi = spoil_interval_from_visits(vs) if labelled else (vs[-1].t, math.inf)
            meta = {**it.meta, "filled_days": filled, "fill": method, "first_day": days[0]}
            out.append(dataclasses.replace(it, item_id=f if c == 0 else f"{f}#{c}", visits=vs, spoil_lo=lo,
                                           spoil_hi=hi, meta=meta))  # fmt: skip
    return out


def _observed_scan_block(it: Item, day: int, d0: int, photo_dim: int) -> np.ndarray:
    for v in it.visits:
        if int(round(v.t + d0)) == day and v.features is not None:
            return v.feature_vector[photo_dim:]
    return np.zeros(next(v.features.shape[-1] for v in it.visits if v.features is not None) - photo_dim, np.float32)


def _paths(it: Item, day: int, d0: int) -> tuple[str, ...]:
    for v in it.visits:
        if int(round(v.t + d0)) == day:
            return v.image_paths
    return ()
