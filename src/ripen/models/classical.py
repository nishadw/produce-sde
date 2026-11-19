"""Two non-neural models, fit directly instead of by the gradient loop (``model.fit(train, val, info, cfg)``).

* :class:`WeibullAFT`: a Weibull accelerated-failure-time model of the days left from the last visible visit,
  ``log scale = b + w . z + c t_now`` with ``z`` the last visible photo in a ``pca``-dimensional PCA space of the
  training photos and ``t_now`` the days since the fruit was first seen. Maximum likelihood under the same interval /
  right censoring as the evaluation score, with a ridge penalty on ``w`` and ``c``. With random start it is fit on
  every start of every training fruit (``all_starts``), so ``t_now`` is not the calendar.
* :class:`TemporalMatching`: analog forecasting. The library is every pre-spoil visit (an "anchor") of every training
  fruit with a known spoilage interval. A query's last ``k_visits`` visible photos are aligned with a library fruit
  by their time gaps before now (query visit ``i`` taken ``d_i`` days ago is compared with the library fruit's visit
  nearest ``d_i`` days before the anchor), so only gaps are used, never the calendar. Anchors are weighted by
  ``exp(-dist / 2h^2)`` and each contributes a uniform distribution on its remaining days ``(lo - t, hi - t]``.
  A gap with no library visit within ``tol`` days costs the median distance. ``h^2`` is a multiple of that median,
  chosen on the validation fruit by the censored log score. A small ``eps`` of the unweighted anchor mixture keeps
  every forecast proper.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.optimize import minimize

from ..data.collate import Batch, DatasetInfo, collate
from ..data.schema import Item, all_starts
from ..eval.evaluate import prediction_instances
from ..eval.metrics import censored_log_score


def _pca(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(0)
    k = max(1, min(k, X.shape[0], X.shape[1]))
    _, _, vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu.astype(np.float32), vt[:k].astype(np.float32)


def _train_features(items: list[Item], dim: int) -> np.ndarray:
    xs = [v.feature_vector for it in items for v in it.visits if v.features is not None]
    return np.stack(xs) if xs else np.zeros((2, dim), np.float32)


def _instances(items: list[Item], info: DatasetInfo) -> Batch | None:
    inst = prediction_instances(items, "all_unspoiled")
    if not inst:
        return None
    return collate([items[i] for i, _ in inst], info, n_obs=[k for _, k in inst])


def _last_visible(batch: Batch) -> Array:
    idx = jnp.maximum(batch.n_obs.astype(jnp.int32) - 1, 0)
    return jnp.take_along_axis(batch.feat, idx[:, None, None], axis=1)[:, 0]  # [B, D]


def _weibull_log_lik(lam, k, lo, hi, cens):
    """log P(lo < T <= hi) (log P(T > lo) if censored) for T ~ Weibull(scale lam, shape k)."""
    lo = jnp.maximum(lo, 0.0)
    hi = jnp.where(cens, lo + 1.0, hi)
    lo = jnp.where(~cens & (hi - lo < 1e-6), jnp.maximum(hi - 1.0, 0.0), lo)  # zero width: the day bin ending at hi
    log_s_lo = -((lo / lam) ** k)
    log_s_hi = -((hi / lam) ** k)
    interval = log_s_lo + jnp.log(-jnp.expm1(jnp.minimum(log_s_hi - log_s_lo, -1e-12)))
    return jnp.where(cens, log_s_lo, interval)


class WeibullAFT(eqx.Module):
    horizon: int = eqx.field(static=True)
    pca: int = eqx.field(static=True)
    ridge: float = eqx.field(static=True)
    mu: Array
    V: Array
    z_mean: Array
    z_std: Array
    b: Array
    w: Array
    c: Array
    log_k: Array

    def __init__(self, feat_dim: int, horizon: int, pca: int = 4, ridge: float = 1.0):
        self.horizon, self.pca, self.ridge = horizon, pca, ridge
        q = max(1, min(pca, feat_dim))
        self.mu, self.V = jnp.zeros(feat_dim), jnp.zeros((q, feat_dim))
        self.z_mean, self.z_std = jnp.zeros(q), jnp.ones(q)
        self.b, self.w, self.c, self.log_k = jnp.asarray(1.5), jnp.zeros(q), jnp.asarray(0.0), jnp.asarray(0.5)

    def _z(self, batch: Batch) -> Array:
        return ((_last_visible(batch) - self.mu) @ self.V.T - self.z_mean) / self.z_std

    def _scale(self, batch: Batch) -> Array:
        return jnp.exp(self.b + self._z(batch) @ self.w + self.c * batch.t_now / 10.0)

    def loss(self, batch: Batch, key: Array) -> Array:
        ll = _weibull_log_lik(self._scale(batch), jnp.exp(self.log_k), batch.lo_rel, batch.hi_rel, batch.right_censored)
        return -jnp.mean(ll) + 0.5 * self.ridge * (jnp.sum(self.w**2) + self.c**2) / batch.batch_size

    def predict_cdf(self, batch: Batch, key: Array) -> Array:
        h = jnp.arange(self.horizon + 1, dtype=jnp.float32)
        lam = self._scale(batch)[:, None]
        return -jnp.expm1(-((h[None, :] / lam) ** jnp.exp(self.log_k)))

    def fit(self, train: list[Item], val: list[Item], info: DatasetInfo, cfg) -> WeibullAFT:
        X = _train_features(train, info.feat_dim)
        mu, V = _pca(X, self.pca)
        Z = (X - mu) @ V.T
        m = eqx.tree_at(lambda s: (s.mu, s.V, s.w), self, (jnp.asarray(mu), jnp.asarray(V), jnp.zeros(V.shape[0])))
        m = eqx.tree_at(lambda s: (s.z_mean, s.z_std), m, (jnp.asarray(Z.mean(0)), jnp.asarray(Z.std(0) + 1e-6)))
        items = all_starts(train, cfg.random_start_max) if getattr(cfg, "random_start", False) else train
        batch = _instances(items, info)
        if batch is None:
            return m
        keep = ("b", "w", "c", "log_k")
        flat0 = np.concatenate([np.ravel(np.asarray(getattr(m, n))) for n in keep]).astype(np.float64)
        sizes = [int(np.size(getattr(m, n))) for n in keep]

        def unflat(x):
            out, i = m, 0
            for n, s in zip(keep, sizes, strict=True):
                val_ = jnp.reshape(x[i : i + s], np.shape(getattr(m, n)))
                out = eqx.tree_at(lambda t, n=n: getattr(t, n), out, val_)
                i += s
            return out

        f = jax.jit(jax.value_and_grad(lambda x: unflat(x).loss(batch, jax.random.PRNGKey(0)) * batch.batch_size))
        res = minimize(lambda x: tuple(np.asarray(v, np.float64) for v in f(jnp.asarray(x, jnp.float32))), flat0,
                       jac=True, method="L-BFGS-B")  # fmt: skip
        return unflat(jnp.asarray(res.x, jnp.float32))


class TemporalMatching(eqx.Module):
    horizon: int = eqx.field(static=True)
    pca: int = eqx.field(static=True)
    k_visits: int = eqx.field(static=True)
    tol: float = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    mu: Array
    V: Array
    lib_t: Array  # [J, L] visit times of library fruit
    lib_z: Array  # [J, L, q] their photos in PCA space
    lib_m: Array  # [J, L] real visit
    anchor: Array  # [J, L] pre-spoil visit usable as an anchor
    lib_lo: Array  # [J]
    lib_hi: Array  # [J]
    pen: Array  # median query-library distance (cost of an unmatched gap)
    h2: Array  # kernel bandwidth^2

    def __init__(self, feat_dim: int, horizon: int, pca: int = 8, k_visits: int = 3, tol: float = 1.01,
                 eps: float = 0.05):  # fmt: skip
        self.horizon, self.pca, self.k_visits, self.tol, self.eps = horizon, pca, k_visits, tol, eps
        q = max(1, min(pca, feat_dim))
        self.mu, self.V = jnp.zeros(feat_dim), jnp.zeros((q, feat_dim))
        self.lib_t, self.lib_z = jnp.zeros((1, 1)), jnp.zeros((1, 1, q))
        self.lib_m, self.anchor = jnp.zeros((1, 1), bool), jnp.ones((1, 1), bool)
        self.lib_lo, self.lib_hi = jnp.zeros(1), jnp.ones(1)
        self.pen, self.h2 = jnp.asarray(1.0), jnp.asarray(1.0)

    def _anchor_weights(self, batch: Batch) -> Array:
        """``[B, J, L]`` normalised anchor weights."""
        qz = (batch.feat - self.mu) @ self.V.T  # [B, V, q]
        n = batch.n_obs.astype(jnp.int32)[:, None]
        i = jnp.arange(batch.t.shape[1])[None, :]
        use = batch.visible & (i >= n - self.k_visits)  # the last k visible visits
        d = batch.t_now[:, None] - batch.t  # days before now, [B, V]
        target = self.lib_t[None, None, :, :] - d[:, :, None, None]  # [B, V, J, L(anchor)]
        gap = jnp.abs(self.lib_t[None, None, :, None, :] - target[..., None])  # [B, V, J, A, L]
        gap = jnp.where(self.lib_m[None, None, :, None, :], gap, jnp.inf)
        near, best = jnp.argmin(gap, -1), jnp.min(gap, -1)  # [B, V, J, A]
        pd = jnp.sum((qz[:, :, None, None, :] - self.lib_z[None, None]) ** 2, -1)  # [B, V, J, L]
        d2 = jnp.take_along_axis(pd, near, axis=3)  # [B, V, J, A]: the matched library visit's distance
        matched = use[:, :, None, None] & (best <= self.tol)
        cost = jnp.where(matched, d2, self.pen)
        dist = jnp.sum(jnp.where(use[:, :, None, None], cost, 0.0), 1) / jnp.maximum(jnp.sum(use, 1), 1)[:, None, None]
        logw = jnp.where(self.anchor[None], -dist / (2.0 * self.h2), -jnp.inf)
        flat = jax.nn.softmax(logw.reshape(logw.shape[0], -1), axis=-1)
        return flat.reshape(logw.shape)

    def _anchor_cdf(self) -> Array:
        """``[J, L, H+1]`` uniform CDF on each anchor's remaining days."""
        h = jnp.arange(self.horizon + 1, dtype=jnp.float32)
        rlo = self.lib_lo[:, None] - self.lib_t
        rhi = self.lib_hi[:, None] - self.lib_t
        return jnp.clip((h - rlo[..., None]) / jnp.maximum(rhi - rlo, 1e-3)[..., None], 0.0, 1.0)

    def predict_cdf(self, batch: Batch, key: Array) -> Array:
        Fa = self._anchor_cdf()
        w = self._anchor_weights(batch)
        F = jnp.einsum("bjl,jlh->bh", w, Fa)
        a = self.anchor.astype(jnp.float32)
        F_pop = jnp.einsum("jl,jlh->h", a, Fa) / jnp.maximum(a.sum(), 1.0)
        return (1.0 - self.eps) * F + self.eps * F_pop[None]

    def loss(self, batch: Batch, key: Array) -> Array:  # only for interface completeness: fit() does the work
        return jnp.asarray(0.0)

    def fit(self, train: list[Item], val: list[Item], info: DatasetInfo, cfg) -> TemporalMatching:
        X = _train_features(train, info.feat_dim)
        mu, V = _pca(X, self.pca)
        # fruit first photographed already spoiled (zero-width interval) have no moment to forecast from: not anchors
        lib = [it for it in train if np.isfinite(it.spoil_hi) and it.spoil_hi > it.spoil_lo]
        if not lib:
            raise ValueError("temporal matching needs training fruit with a known spoilage interval")
        L = max(it.n_visits for it in lib)
        q = V.shape[0]
        T, Zl, M = np.zeros((len(lib), L)), np.zeros((len(lib), L, q)), np.zeros((len(lib), L), bool)
        for j, it in enumerate(lib):
            for v_i, v in enumerate(it.visits):
                T[j, v_i] = v.t
                M[j, v_i] = True
                if v.features is not None:
                    Zl[j, v_i] = (v.feature_vector - mu) @ V.T
        lo = np.array([it.spoil_lo for it in lib])
        hi = np.array([it.spoil_hi for it in lib])
        A = M & (T <= lo[:, None])
        flat = Zl[A]
        dd = np.sum((flat[:, None] - flat[None]) ** 2, -1)
        pen = float(np.median(dd[np.triu_indices(len(flat), 1)])) if len(flat) > 1 else 1.0
        pen = max(pen, 1e-6)
        m = eqx.tree_at(
            lambda s: (s.mu, s.V, s.lib_t, s.lib_z, s.lib_m, s.anchor, s.lib_lo, s.lib_hi, s.pen),
            self,
            tuple(jnp.asarray(x) for x in (mu, V, T.astype(np.float32), Zl.astype(np.float32), M, A,
                                           lo.astype(np.float32), hi.astype(np.float32), np.float32(pen))),
        )  # fmt: skip
        vitems = all_starts(val, cfg.random_start_max) if (val and getattr(cfg, "random_start", False)) else val
        vb = _instances(vitems, info) if vitems else None
        best, best_s = 0.3, np.inf
        if vb is not None:
            pred = eqx.filter_jit(lambda mm, b: mm.predict_cdf(b, jax.random.PRNGKey(0)))
            vhi = np.where(np.asarray(vb.right_censored), np.inf, np.asarray(vb.hi_rel))
            for s in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
                mm = eqx.tree_at(lambda t: t.h2, m, jnp.asarray(s * pen, jnp.float32))
                sc = float(np.mean(censored_log_score(np.asarray(pred(mm, vb)), np.asarray(vb.lo_rel), vhi)))
                if sc < best_s:
                    best, best_s = s, sc
        print(f"matching: {len(lib)} library fruit, {int(A.sum())} anchors, bandwidth {best} x median "
              f"(val log score {best_s:.3f})", flush=True)  # fmt: skip
        return eqx.tree_at(lambda t: t.h2, m, jnp.asarray(best * pen, jnp.float32))
