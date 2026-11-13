"""Every model in the registry: builds, has a finite loss and gradient, and predicts a valid CDF."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ripen.data import collate
from ripen.models import MODEL_REGISTRY, build_model

HORIZON = 12
EXTRA = {
    "latent_sde": {"n_samples": 8, "n_train_samples": 2, "n_sub": 1},
    "latent_ode": {"n_samples": 8, "n_train_samples": 2, "n_sub": 1},
}


@pytest.fixture(scope="module")
def batch(synthetic_items, synthetic_info):
    return collate(synthetic_items[:6], synthetic_info, n_obs="random", rng=np.random.default_rng(0))


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_model_one_batch(name, synthetic_info, batch):
    cfg = {"name": name, "horizon": HORIZON, **EXTRA.get(name, {})}
    model = build_model(cfg, synthetic_info, jax.random.PRNGKey(0))
    loss, grads = eqx.filter_value_and_grad(lambda m: m.loss(batch, jax.random.PRNGKey(1)))(model)
    assert np.isfinite(float(loss))
    leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))
    assert leaves and all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
    assert any(bool(jnp.any(g != 0)) for g in leaves), "no parameter received gradient"
    F = np.asarray(model.predict_cdf(batch, jax.random.PRNGKey(2)))
    assert F.shape == (batch.batch_size, HORIZON + 1)
    assert (F >= -1e-6).all() and (F <= 1 + 1e-6).all()
    assert (np.diff(F, axis=1) >= -1e-6).all()


def test_sequence_encoders_ignore_hidden_visits(synthetic_items, synthetic_info):
    """Changing a visit beyond n_obs must not change the prediction."""
    items = synthetic_items[:4]
    b1 = collate(items, synthetic_info, n_obs=2)
    feat2 = b1.feat.at[:, 3:].set(99.0)
    b2 = eqx.tree_at(lambda b: b.feat, b1, feat2)
    for name in ("gru_gap", "gru_d", "ode_rnn", "neural_cde", "transformer"):
        model = build_model({"name": name, "horizon": HORIZON}, synthetic_info, jax.random.PRNGKey(0))
        F1, F2 = model.predict_cdf(b1, jax.random.PRNGKey(0)), model.predict_cdf(b2, jax.random.PRNGKey(0))
        assert np.allclose(np.asarray(F1), np.asarray(F2)), name


@pytest.mark.parametrize("method", ["backprop", "checkpoint", "adjoint"])
def test_latent_sde_gradient_methods_agree(method, synthetic_info, batch):
    """The three ways of differentiating the SDE solve give the same loss and close gradients."""
    cfg = {
        "name": "latent_sde",
        "horizon": HORIZON,
        "n_samples": 4,
        "n_train_samples": 1,
        "n_sub": 1,
        "solver": "reversible_heun",
        "brownian": "virtual_tree",
        "method": method,
    }
    model = build_model(cfg, synthetic_info, jax.random.PRNGKey(0))
    loss, grads = eqx.filter_value_and_grad(lambda m: m.loss(batch, jax.random.PRNGKey(1)))(model)
    ref_cfg = {**cfg, "method": "backprop"}
    ref = build_model(ref_cfg, synthetic_info, jax.random.PRNGKey(0))
    loss_ref, grads_ref = eqx.filter_value_and_grad(lambda m: m.loss(batch, jax.random.PRNGKey(1)))(ref)
    assert float(loss) == pytest.approx(float(loss_ref), rel=1e-4)
    g1 = jnp.concatenate([g.ravel() for g in jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))])
    g2 = jnp.concatenate([g.ravel() for g in jax.tree.leaves(eqx.filter(grads_ref, eqx.is_inexact_array))])
    rel = float(jnp.max(jnp.abs(g1 - g2)) / (jnp.max(jnp.abs(g2)) + 1e-12))
    # The continuous adjoint is an approximation, error ~1/steps (foundations/experiments/adjoint_tradeoffs.py);
    # with one step per day and a running-max spoilage term it lands near 1e-1 here. Checkpointing is exact.
    assert rel < (0.2 if method == "adjoint" else 1e-5)


def test_posterior_guidance_points_at_the_next_label(synthetic_info):
    """The guided posterior's h-transform pushes the progression coordinate up when the next label is a higher stage,
    down when it is a lower one, vanishes with no labelled visit ahead, and leaves the loss and gradients finite."""
    from ripen.models import LatentDynamics

    m = LatentDynamics(
        synthetic_info.feat_dim, len(synthetic_info.groups), horizon=24, t_max=synthetic_info.t_max,
        key=jax.random.PRNGKey(0), latent_dim=1, use_images=False, drift_kind="linear", sigma_kind="const",
        clock="group", fixed_thresholds=True, link_scale=0.1, learn_link_scale=False, posterior_guidance=True,
    )  # fmt: skip
    tv, vis = jnp.array([0.0, 1.0, 2.0]), jnp.array([True, True, False])
    kappa, r = jnp.asarray(0.5), jnp.array([2.5])  # in stage 2
    up = m.guidance(jnp.asarray(0.5), r, kappa, tv, vis, jnp.array([2, 4, 0]))[0]
    down = m.guidance(jnp.asarray(0.5), r, kappa, tv, vis, jnp.array([2, 1, 0]))[0]
    stay = m.guidance(jnp.asarray(0.5), r, kappa, tv, vis, jnp.array([2, 2, 0]))[0]
    none = m.guidance(jnp.asarray(1.5), r, kappa, tv, vis, jnp.array([2, 2, 0]))[0]  # next visit hidden
    assert float(up) > 0 > float(down) and abs(float(stay)) < abs(float(up)) and float(none) == 0.0


def test_guided_loss_and_gradients_are_finite(synthetic_items, synthetic_info):
    from ripen.models import LatentDynamics

    m = LatentDynamics(
        synthetic_info.feat_dim, len(synthetic_info.groups), horizon=24, t_max=synthetic_info.t_max,
        key=jax.random.PRNGKey(0), latent_dim=1, use_images=False, clock="group", fixed_thresholds=True,
        posterior_guidance=True, event_mean="log_mean",
    )  # fmt: skip
    b = collate(synthetic_items[:8], synthetic_info, n_obs="all")
    loss, g = eqx.filter_value_and_grad(lambda mm: mm.loss(b, jax.random.PRNGKey(1)))(m)
    leaves = jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_inexact_array))
    assert np.isfinite(float(loss)) and all(bool(jnp.all(jnp.isfinite(x))) for x in leaves)
