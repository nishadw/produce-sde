"""Mango study pieces that do not need the raw data: SAMD's supermodel reset, the current-status reference, scoring."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import jax
import numpy as np
import pytest

from ripen.data.collate import collate, dataset_info
from ripen.data.schema import Item, Visit
from ripen.eval.mango import cutday_scores
from ripen.models import build_model
from ripen.train.samd import features, reset_teacher, share_predictor


def _items(x: np.ndarray, stage=None) -> list[Item]:
    out = []
    for i, row in enumerate(x):
        spoiled = i % 3 == 0
        out.append(
            Item(
                f"f{i}",
                "mango",
                "ambient",
                math.nan,
                [Visit(0.0, stage, features=row[None].astype(np.float32))],
                0.0,
                0.0 if spoiled else math.inf,
            )
        )
    return out


@pytest.mark.parametrize("name", ["gru_gap", "transformer", "latent_sde", "fpt_filter"])
def test_reset_teacher_is_exact(name):
    """With the auxiliary columns zeroed, the teacher computes exactly the student's function, whatever I_A is."""
    rng = np.random.default_rng(0)
    xm = rng.normal(size=(6, 4))
    xa = rng.normal(size=(6, 5))
    s_items, t_items = _items(xm), _items(np.concatenate([xm, xa], 1))
    s_info, t_info = dataset_info(s_items), dataset_info(t_items)
    cfg = {"name": name, "horizon": 6, "use_stages": False}
    if name == "latent_sde":
        cfg.update({"pde_steps": 400, "n_samples": 8})
    if name == "fpt_filter":
        cfg.update({"pde_steps": 400})
    student = build_model(cfg, s_info, jax.random.PRNGKey(1))
    teacher = reset_teacher(build_model(cfg, t_info, jax.random.PRNGKey(2)), student, n_main=4)
    sb, tb = collate(s_items, s_info, n_obs="all"), collate(t_items, t_info, n_obs="all")
    key = jax.random.PRNGKey(3)
    np.testing.assert_allclose(features(student, sb), features(teacher, tb), atol=1e-5)
    np.testing.assert_allclose(student.predict_cdf(sb, key), teacher.predict_cdf(tb, key), atol=1e-5)
    # sharing the predictor back leaves the student's feature network untouched
    shared = share_predictor(student, teacher)
    np.testing.assert_allclose(features(shared, sb), features(student, sb), atol=1e-6)


def _mango_cv():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run" / "mango_cv.py"
    spec = importlib.util.spec_from_file_location("mango_cv", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_current_status_cdf_is_monotone_and_fits_pooled_rates():
    cv = _mango_cv()
    day = np.repeat([0, 2, 4, 6], 4)
    status = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0], float)  # day 2 > day 4: pooled
    G = cv.current_status_cdf(day, status, 10)
    assert np.all(np.diff(G) >= -1e-12)
    assert G[0] == 0.0 and np.isclose(G[2], 0.125) and np.isclose(G[4], 0.125) and np.isclose(G[6], 0.75)
    assert np.isclose(G[10], 0.75)


def test_cutday_scores_reward_the_right_clock():
    """A forecast that spoils at day 12 beats one that spoils at day 4 on population Brier, and ranks firmness."""
    recs = [
        {"item_id": f"f{d}", "cut_day": float(d), "spoiled": d >= 12, "firmness_n": 40.0 / (1 + d)}
        for d in range(0, 16, 2)
    ]
    H = 20
    good = np.stack([(np.arange(H + 1) >= max(12 - r["cut_day"], 0)).astype(float) for r in recs])
    bad = np.stack([(np.arange(H + 1) >= max(4 - r["cut_day"], 0)).astype(float) for r in recs])
    sg, sb = cutday_scores(good, recs, recs), cutday_scores(bad, recs, recs)
    assert sg["pop_brier"] == 0.0 and sb["pop_brier"] > 0.3
    assert sg["spearman_firmness"] > 0.9


def test_kd_loss_is_zero_only_for_identical_forecasts():
    """CAML's distillation term: 0 when student and teacher agree, positive otherwise, finite at the edges."""
    import jax.numpy as jnp

    from ripen.train.aml import kd_loss

    F = jnp.array([[0.1, 0.3, 0.6, 0.9], [0.0, 0.0, 0.5, 1.0]])
    G = jnp.array([[0.0, 0.1, 0.2, 0.4], [0.2, 0.4, 0.8, 1.0]])
    assert float(kd_loss(F, F, 4.0)) < 1e-7
    assert float(kd_loss(G, F, 4.0)) > 1e-4 and np.isfinite(float(kd_loss(G, F, 1.0)))


def test_recalibrator_weight_mixes_in_the_raw_forecast():
    from ripen.eval.recalibrate import PITRecalibrator

    rng = np.random.default_rng(0)
    F = np.sort(rng.uniform(size=(40, 6)), axis=1)
    lo = rng.integers(0, 5, 40).astype(float)
    hi = lo + 1
    full, mixed = PITRecalibrator.fit(F, lo, hi), PITRecalibrator.fit(F, lo, hi, weight=0.9)
    assert np.allclose(mixed(F), 0.9 * full(F) + 0.1 * F)
    assert np.all(np.diff(mixed(F), axis=1) >= 0)


def test_rep_losses_are_zero_for_identical_features_and_respect_weights():
    import jax
    import jax.numpy as jnp

    from ripen.train.aml import REPS, rep_loss

    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    f = jax.random.normal(k1, (6, 5))
    g = jax.random.normal(k2, (6, 5))
    w = jnp.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    for kind in REPS:
        assert float(rep_loss(kind, f, f)) < 1e-5, kind
        assert float(rep_loss(kind, g, f)) > 1e-4, kind
        # items with weight 0 do not matter
        g2 = g.at[4:].set(100.0)
        assert abs(float(rep_loss(kind, g, f, w)) - float(rep_loss(kind, g2, f, w))) < 1e-4, kind


def test_rep_loss_gradients_are_finite():
    import jax

    from ripen.train.aml import REPS, rep_loss

    f = jax.random.normal(jax.random.PRNGKey(1), (5, 4))
    t = jax.random.normal(jax.random.PRNGKey(2), (5, 4))
    for kind in REPS:
        g = jax.grad(lambda x, kind=kind: rep_loss(kind, x, t))(f)
        assert bool(jax.numpy.all(jax.numpy.isfinite(g))), kind
