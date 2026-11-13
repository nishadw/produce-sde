"""Classical baselines: two degradation processes and a landmark proportional-hazards model.

All three read the **stage history** (the labelled ripeness stage at each visible visit) as their health indicator
and the storage temperature. None of them looks at the photos; they are the statistics answer to "when will a
degrading thing cross a threshold", which is what the latent SDE is compared against.

* :class:`WienerDegradation`: ``X(t) = X0 + mu * L(t) + sigma * B(L(t))`` with a per-fruit random drift
  ``mu ~ N(m, s^2)``, observed with Gaussian error, ``L(t) = kappa(T) t`` an Arrhenius clock (accelerated
  degradation testing; Whitmore & Schenkelberg 1997, Si et al. 2011). A Kalman filter gives the posterior of
  ``(X_now, mu)``; the remaining life is the inverse-Gaussian first passage of ``X`` over the threshold ``w``,
  averaged over that posterior with Gauss-Hermite quadrature.
* :class:`GammaDegradation`: a monotone gamma process, increments ``Gamma(alpha dL, rate=beta)`` with a per-fruit
  random rate ``beta ~ Gamma(delta, gamma)`` (Lawless & Crowder 2004), on the running maximum of the stage
  history (a gamma process cannot go down). The posterior of ``beta`` is conjugate:
  ``Gamma(delta + alpha L_obs, gamma + X_now - X_first)``. The remaining life is
  ``P(G(u) >= w - X_now) = E_beta[Q(alpha u, beta d)]``.
* :class:`LandmarkCox`: a discrete-time proportional-hazards model (complementary log-log link; Prentice &
  Gloeckler 1978) fitted at every landmark ("now") with covariates from the history: current stage, stage slope,
  elapsed days, number of visits and the Arrhenius covariate ``1000/T``. Its interval-censored likelihood is exact
  on the day grid.

The degradation models are fitted by maximising the likelihood of each item's observed spoilage interval given
its visible history (plus, for the Wiener model, the Kalman likelihood of the visible stages). Predictions are the
CDF of days-left on the shared grid.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import gammaincc, log_ndtr, ndtr

from ..data.collate import Batch
from ..data.schema import N_STAGES
from .base import horizon_grid, interval_log_prob_from_logits, pmf_to_cdf
from .clock import make_clock

_GH_N = 8
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(_GH_N)  # probabilists' Hermite: E[f(Z)], Z ~ N(0, 1)
_GH_W = _GH_W / _GH_W.sum()


def _observed(batch: Batch) -> Array:
    """``[B, V]`` mask of visible visits that carry a stage label."""
    return batch.visible & (batch.stage > 0)


def _interval_prob(F_lo: Array, F_hi: Array, right_censored: Array) -> Array:
    p = jnp.where(right_censored, 1.0 - F_lo, F_hi - F_lo)
    return jnp.log(jnp.clip(p, 1e-8, 1.0))


def _rel_bounds(batch: Batch) -> tuple[Array, Array]:
    lo = jnp.clip(batch.lo_rel, 0.0, None)
    hi = jnp.where(batch.right_censored, lo + 1.0, batch.hi_rel)
    # a zero-width interval (spoiled at the very first visit) means "within the day before"
    lo = jnp.where(hi - lo < 1e-6, jnp.clip(hi - 1.0, 0.0, None), lo)
    return lo, hi


def inverse_gaussian_cdf(u: Array, d: Array, mu: Array, sigma: Array) -> Array:
    """P(first passage of ``mu t + sigma B_t`` over level ``d`` happens by clock time ``u``). ``d <= 0`` -> 1."""
    u = jnp.clip(u, 1e-6, None)
    dpos = jnp.clip(d, 1e-6, None)
    s = sigma * jnp.sqrt(u)
    a = ndtr((mu * u - dpos) / s)
    log_b = 2.0 * mu * dpos / sigma**2 + log_ndtr(-(mu * u + dpos) / s)
    F = a + jnp.exp(jnp.clip(log_b, None, 0.0))
    return jnp.where(d <= 0.0, 1.0, jnp.clip(F, 0.0, 1.0))


class WienerDegradation(eqx.Module):
    x0: Array
    log_p0: Array
    m: Array
    log_s: Array
    log_sigma: Array
    log_tau: Array
    w: Array
    clock: eqx.Module
    horizon: int = eqx.field(static=True)
    obs_weight: float = eqx.field(static=True)

    def __init__(
        self,
        n_groups: int,
        horizon: int,
        clock: str = "arrhenius",
        clock_kwargs: dict | None = None,
        obs_weight: float = 1.0,
        *,
        key=None,
    ):
        self.x0, self.log_p0 = jnp.asarray(1.5), jnp.log(jnp.asarray(0.5))
        self.m, self.log_s = jnp.asarray(1.0), jnp.log(jnp.asarray(0.3))
        self.log_sigma, self.log_tau = jnp.log(jnp.asarray(0.4)), jnp.log(jnp.asarray(0.3))
        self.w = jnp.asarray(N_STAGES - 0.5)
        self.clock = make_clock(clock, n_groups, **(clock_kwargs or {}))
        self.horizon, self.obs_weight = horizon, float(obs_weight)

    def _filter(self, t: Array, y: Array, obs: Array, kappa: Array) -> tuple[Array, Array, Array]:
        """Kalman filter on ``(X, mu)`` over one item's visits -> (mean at last obs, cov, log-likelihood)."""
        sig2, tau2 = jnp.exp(2 * self.log_sigma), jnp.exp(2 * self.log_tau)
        mean0 = jnp.stack([self.x0, self.m])
        cov0 = jnp.diag(jnp.stack([jnp.exp(2 * self.log_p0), jnp.exp(2 * self.log_s)]))

        def step(carry, inp):
            mean, cov, t_prev, ll = carry
            tk, yk, ok = inp
            dl = kappa * jnp.clip(tk - t_prev, 0.0, None)
            F = jnp.array([[1.0, 0.0], [0.0, 1.0]]).at[0, 1].set(dl)
            mp = F @ mean
            Pp = F @ cov @ F.T + jnp.diag(jnp.stack([sig2 * dl, 0.0]))
            S = Pp[0, 0] + tau2
            K = Pp[:, 0] / S
            r = yk - mp[0]
            mu_new = mp + K * r
            P_new = Pp - jnp.outer(K, Pp[0, :])
            ll_k = -0.5 * (jnp.log(2 * jnp.pi * S) + r**2 / S)
            mean = jnp.where(ok, mu_new, mean)
            cov = jnp.where(ok, P_new, cov)
            t_prev = jnp.where(ok, tk, t_prev)
            return (mean, cov, t_prev, ll + jnp.where(ok, ll_k, 0.0)), None

        (mean, cov, t_last, ll), _ = jax.lax.scan(step, (mean0, cov0, jnp.asarray(0.0), jnp.asarray(0.0)), (t, y, obs))
        return mean, cov, t_last, ll

    def _cdf_at(self, mean: Array, cov: Array, kappa: Array, gap: Array, h: Array) -> Array:
        """CDF of days-left at ``h`` (any shape) for one item; ``gap`` = time from the last labelled visit to now."""
        L = jnp.linalg.cholesky(cov + 1e-6 * jnp.eye(2))
        z = jnp.stack(jnp.meshgrid(jnp.asarray(_GH_X), jnp.asarray(_GH_X), indexing="ij"), -1).reshape(-1, 2)
        wts = jnp.outer(jnp.asarray(_GH_W), jnp.asarray(_GH_W)).reshape(-1)
        nodes = mean[None, :] + z @ L.T  # [N, 2]: (X at last obs, mu)
        sigma = jnp.exp(self.log_sigma)
        u = kappa * (gap + h[..., None])  # clock time since the last labelled visit
        F = inverse_gaussian_cdf(u, self.w - nodes[:, 0], nodes[:, 1], sigma)
        # passage before "now" is impossible (the item was unspoiled at its last visit): condition on it
        F_now = inverse_gaussian_cdf(kappa * gap + 0.0 * h[..., None], self.w - nodes[:, 0], nodes[:, 1], sigma)
        Fc = (F - F_now) / jnp.clip(1.0 - F_now, 1e-6, None)
        return jnp.sum(wts * jnp.clip(Fc, 0.0, 1.0), axis=-1)

    def _per_item(self, batch: Batch):
        kappa = self.clock(batch.temp_k, batch.group)
        y = batch.stage.astype(jnp.float32)
        obs = _observed(batch)
        mean, cov, t_last, ll = jax.vmap(self._filter)(batch.t, y, obs, kappa)
        gap = jnp.clip(batch.t_now - t_last, 0.0, None)
        return kappa, mean, cov, gap, ll

    def loss(self, batch: Batch, key: Array) -> Array:
        kappa, mean, cov, gap, ll = self._per_item(batch)
        lo, hi = _rel_bounds(batch)
        F = jax.vmap(self._cdf_at)(mean, cov, kappa, gap, jnp.stack([lo, hi], -1))
        ev = _interval_prob(F[:, 0], F[:, 1], batch.right_censored)
        return -jnp.mean(ev + self.obs_weight * ll)

    def predict_cdf(self, batch: Batch, key: Array) -> Array:
        kappa, mean, cov, gap, _ = self._per_item(batch)
        h = jnp.broadcast_to(horizon_grid(self.horizon), (batch.batch_size, self.horizon + 1))
        return jax.lax.cummax(jax.vmap(self._cdf_at)(mean, cov, kappa, gap, h), axis=1)


class GammaDegradation(eqx.Module):
    log_alpha: Array
    log_delta: Array
    log_gamma: Array
    w: Array
    clock: eqx.Module
    horizon: int = eqx.field(static=True)

    def __init__(
        self, n_groups: int, horizon: int, clock: str = "arrhenius", clock_kwargs: dict | None = None, *, key=None
    ):
        self.log_alpha = jnp.log(jnp.asarray(1.0))
        self.log_delta, self.log_gamma = jnp.log(jnp.asarray(5.0)), jnp.log(jnp.asarray(5.0))
        self.w = jnp.asarray(N_STAGES - 0.5)
        self.clock = make_clock(clock, n_groups, **(clock_kwargs or {}))
        self.horizon = horizon

    def _stats(self, batch: Batch):
        """Per item: first stage, running max now, clock time observed, gap from the last labelled visit to now."""
        obs = _observed(batch)
        y = jnp.where(obs, batch.stage.astype(jnp.float32), -jnp.inf)
        x_now = jnp.max(y, axis=1)
        first = jnp.argmax(obs, axis=1)
        x_first = jnp.take_along_axis(batch.stage, first[:, None], 1)[:, 0].astype(jnp.float32)
        t_first = jnp.take_along_axis(batch.t, first[:, None], 1)[:, 0]
        t_last = jnp.max(jnp.where(obs, batch.t, -jnp.inf), axis=1)
        any_obs = obs.any(axis=1)
        x_now = jnp.where(any_obs, x_now, 1.0)
        x_first = jnp.where(any_obs, x_first, 1.0)
        t_first = jnp.where(any_obs, t_first, batch.t_now)
        t_last = jnp.where(any_obs, t_last, batch.t_now)
        kappa = self.clock(batch.temp_k, batch.group)
        alpha = jnp.exp(self.log_alpha)
        shape_post = jnp.exp(self.log_delta) + alpha * kappa * (t_last - t_first)
        rate_post = jnp.exp(self.log_gamma) + (x_now - x_first)
        gap = jnp.clip(batch.t_now - t_last, 0.0, None)
        return x_now, kappa, shape_post, rate_post, gap

    def _cdf_at(self, x_now, kappa, a_post, b_post, gap, h):
        """P(remaining life <= h) for one item, ``h`` any shape. beta posterior ~ Gamma(a_post, b_post), integrated
        with a moment-matched log-normal and Gauss-Hermite nodes."""
        alpha = jnp.exp(self.log_alpha)
        d = self.w - x_now
        mean, var = a_post / b_post, a_post / b_post**2
        s2 = jnp.log1p(var / mean**2)
        beta = jnp.exp(jnp.log(mean) - 0.5 * s2 + jnp.sqrt(s2) * jnp.asarray(_GH_X))  # [N]
        wts = jnp.asarray(_GH_W)

        def F_of(u):
            shape = jnp.clip(alpha * u, 1e-6, None)[..., None]
            return jnp.sum(wts * gammaincc(shape, beta * jnp.clip(d, 1e-6, None)), axis=-1)

        F_now = F_of(kappa * gap * jnp.ones_like(h))
        F = F_of(kappa * (gap + h))
        Fc = (F - F_now) / jnp.clip(1.0 - F_now, 1e-6, None)
        return jnp.where(d <= 0.0, 1.0, jnp.clip(Fc, 0.0, 1.0))

    def loss(self, batch: Batch, key: Array) -> Array:
        x_now, kappa, a, b, gap = self._stats(batch)
        lo, hi = _rel_bounds(batch)
        F = jax.vmap(self._cdf_at)(x_now, kappa, a, b, gap, jnp.stack([lo, hi], -1))
        return -jnp.mean(_interval_prob(F[:, 0], F[:, 1], batch.right_censored))

    def predict_cdf(self, batch: Batch, key: Array) -> Array:
        x_now, kappa, a, b, gap = self._stats(batch)
        h = jnp.broadcast_to(horizon_grid(self.horizon), (batch.batch_size, self.horizon + 1))
        return jax.lax.cummax(jax.vmap(self._cdf_at)(x_now, kappa, a, b, gap, h), axis=1)


class LandmarkCox(eqx.Module):
    """Discrete-time proportional hazards (cloglog) on landmark covariates; logits over days 0..H and "after H"."""

    base: Array  # [H + 1] baseline log cumulative-hazard increments per day
    beta: Array
    horizon: int = eqx.field(static=True)

    N_COV = N_STAGES + 1 + 4  # one-hot current stage (0 = unlabelled), slope, elapsed, n visits, 1000/T

    def __init__(self, horizon: int, *, key=None):
        self.base = jnp.full((horizon + 1,), -3.0, jnp.float32)
        self.beta = jnp.zeros((self.N_COV,), jnp.float32)
        self.horizon = horizon

    @staticmethod
    def covariates(batch: Batch) -> Array:
        obs = _observed(batch)
        y = batch.stage.astype(jnp.float32)
        last = jnp.max(jnp.where(obs, y, 0.0), axis=1)
        first_i = jnp.argmax(obs, axis=1)
        first = jnp.take_along_axis(y, first_i[:, None], 1)[:, 0]
        t_first = jnp.take_along_axis(batch.t, first_i[:, None], 1)[:, 0]
        t_last = jnp.max(jnp.where(obs, batch.t, -jnp.inf), axis=1)
        any_obs = obs.any(axis=1)
        slope = jnp.where(any_obs, (last - first) / (jnp.clip(t_last - t_first, 0.0, None) + 1.0), 0.0)
        stage_now = jnp.where(any_obs, last, 0.0).astype(jnp.int32)
        onehot = jax.nn.one_hot(stage_now, N_STAGES + 1)
        cont = jnp.stack(
            [
                slope,
                batch.t_now / 10.0,
                jnp.sum(batch.visible, axis=1) / 10.0,
                1000.0 / batch.temp_k - 1000.0 / 293.15,  # Arrhenius covariate, 0 at 20 C
            ],
            -1,
        )
        return jnp.concatenate([onehot, cont], -1)

    def logits(self, batch: Batch) -> Array:
        eta = self.covariates(batch) @ self.beta  # [B]
        log_cumhaz_inc = self.base[None, :] + eta[:, None]  # [B, H+1]: log of -log(1 - h_d)
        cum = jnp.exp(log_cumhaz_inc)
        log_surv_after = -jnp.cumsum(cum, axis=1)  # log S(d)
        log_surv_before = jnp.concatenate([jnp.zeros_like(cum[:, :1]), log_surv_after[:, :-1]], 1)
        log_h = jnp.log(-jnp.expm1(-jnp.clip(cum, 1e-10, None)))
        log_pmf = log_surv_before + log_h
        return jnp.concatenate([log_pmf, log_surv_after[:, -1:]], 1)

    def loss(self, batch: Batch, key: Array) -> Array:
        lp = interval_log_prob_from_logits(self.logits(batch), batch.lo_rel, batch.hi_rel, batch.right_censored)
        return -jnp.mean(lp) + 1e-3 * jnp.sum(self.beta**2)

    def predict_cdf(self, batch: Batch, key: Array) -> Array:
        return pmf_to_cdf(self.logits(batch))
