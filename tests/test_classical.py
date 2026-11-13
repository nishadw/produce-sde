"""Classical baselines: the closed-form first-passage CDFs match simulation, and no future visit leaks in."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaincc

from ripen.data import collate
from ripen.models import build_model
from ripen.models.classical import inverse_gaussian_cdf


def test_inverse_gaussian_matches_simulation():
    rng = np.random.default_rng(0)
    mu, sigma, d, dt, n = 0.8, 0.5, 2.0, 1e-3, 20_000
    x, hit = np.zeros(n), np.full(n, np.inf)
    for k in range(1, 6001):  # up to t = 6
        x += mu * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
        hit[(x >= d) & np.isinf(hit)] = k * dt
    for u in (1.5, 2.5, 4.0):
        F = float(inverse_gaussian_cdf(jnp.asarray(u), jnp.asarray(d), jnp.asarray(mu), jnp.asarray(sigma)))
        assert abs(F - np.mean(hit <= u)) < 0.02, (u, F, np.mean(hit <= u))


def test_gamma_passage_matches_simulation():
    rng = np.random.default_rng(1)
    alpha, beta, d, u = 1.3, 2.0, 1.5, 3.0
    g = rng.gamma(alpha * u, 1.0 / beta, 200_000)
    assert abs(float(gammaincc(alpha * u, beta * d)) - np.mean(g >= d)) < 0.005


def test_classical_ignore_hidden_visits(synthetic_items, synthetic_info):
    items = synthetic_items[:4]
    b1 = collate(items, synthetic_info, n_obs=2)
    b2 = eqx.tree_at(lambda b: b.stage, b1, b1.stage.at[:, 3:].set(5))
    for name in ("wiener", "gamma_process", "landmark_cox"):
        model = build_model({"name": name, "horizon": 12}, synthetic_info, jax.random.PRNGKey(0))
        F1, F2 = model.predict_cdf(b1, jax.random.PRNGKey(0)), model.predict_cdf(b2, jax.random.PRNGKey(0))
        assert np.allclose(np.asarray(F1), np.asarray(F2)), name
