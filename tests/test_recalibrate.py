import numpy as np
import pytest

from ripen.eval.metrics import compute_metrics
from ripen.eval.recalibrate import PITRecalibrator, kaplan_meier_cdf, pit_values


def _gauss_cdf(mu: np.ndarray, sd: float, H: int) -> np.ndarray:
    from scipy.stats import norm

    h = np.arange(H + 1)
    return norm.cdf((h[None, :] + 0.5 - mu[:, None]) / sd)


def test_km_without_censoring_is_the_empirical_cdf():
    u = np.array([0.1, 0.4, 0.4, 0.9])
    x, G = kaplan_meier_cdf(u, np.ones(4, bool))
    assert np.interp(0.4, x, G) == pytest.approx(0.75)
    assert G[0] == 0.0 and G[-1] == 1.0 and np.all(np.diff(G) >= 0)


def test_km_with_censoring_matches_lifelines():
    lifelines = pytest.importorskip("lifelines")
    rng = np.random.default_rng(0)
    u = rng.uniform(size=200)
    ev = rng.uniform(size=200) < 0.7
    x, G = kaplan_meier_cdf(u, ev)
    km = lifelines.KaplanMeierFitter().fit(u, ev)
    for q in (0.2, 0.5, 0.8):
        ref = 1.0 - float(km.survival_function_at_times(q).iloc[0])
        assert np.interp(q, x, G) == pytest.approx(ref, abs=0.02)  # ours interpolates between steps


def test_recalibration_shrinks_overdispersed_forecasts():
    rng = np.random.default_rng(1)
    H, n = 40, 3000
    mu = rng.uniform(8, 25, size=2 * n)
    t = np.ceil(mu + rng.normal(0, 1.5, size=2 * n))
    F = _gauss_cdf(mu, 5.0, H)  # truth sd 1.5, forecast sd 5: over-wide, over-covers
    lo, hi = t - 1, t
    val, test = slice(0, n), slice(n, 2 * n)
    before = compute_metrics(F[test], lo[test], hi[test])
    recal = PITRecalibrator.fit(F[val], lo[val], hi[val])
    after = compute_metrics(recal(F[test]), lo[test], hi[test])
    oracle = compute_metrics(_gauss_cdf(mu[test], 1.5, H), lo[test], hi[test])
    assert before.coverage80 > 0.95
    # integer-day quantiles with inclusive ends over-cover even for the true law, so compare with the oracle
    assert abs(after.coverage80 - oracle.coverage80) < 0.04
    assert after.ece < 0.02
    assert after.width80 < before.width80
    assert after.crps < before.crps
    Fr = recal(F)
    assert np.all(np.diff(Fr, axis=1) >= -1e-12) and Fr.min() >= 0 and Fr.max() <= 1


def test_right_censored_instances_enter_as_censored_pit():
    F = np.linspace(0, 1, 11)[None].repeat(2, 0)
    u, ev = pit_values(F, np.array([2.0, 5.0]), np.array([3.0, np.inf]))
    assert ev.tolist() == [True, False]
    assert u[0] == pytest.approx(0.25) and u[1] == pytest.approx(0.5)
