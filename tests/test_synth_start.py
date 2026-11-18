"""Half-synthetic A2SL data (ripen.data.a2sl_synth) and random-start shifts (ripen.data.schema)."""

import math

import jax
import numpy as np
import pytest
from test_a2sl import _write_fake_manifests

from ripen.data.a2sl import load_a2sl_avocado
from ripen.data.a2sl_synth import build_half_synthetic
from ripen.data.collate import choose_n_obs
from ripen.data.schema import (
    Item,
    Visit,
    all_starts,
    paired_random_start,
    random_start,
    shift_start,
    start_choices,
)


def _item(n=5, lo=6.0, hi=8.0):
    vs = [Visit(t=2.0 * i, stage=min(i + 1, 5), features=np.full((1, 2), float(i), np.float32)) for i in range(n)]
    return Item("x", "avocado", "ambient", math.nan, vs, lo, hi)


def test_shift_start_rezeroes_time_and_interval():
    sh = shift_start(_item(), 2)
    assert [v.t for v in sh.visits] == [0.0, 2.0, 4.0]
    assert (sh.spoil_lo, sh.spoil_hi) == (2.0, 4.0) and sh.meta["start_shift"] == 4.0
    assert float(sh.visits[0].features[0, 0]) == 2.0  # the visits themselves are untouched
    with pytest.raises(ValueError):
        shift_start(_item(), 4)  # day-8 visit is already spoiled


def test_start_choices_and_all_starts():
    it = _item()
    assert start_choices(it, 6.0) == [0, 1, 2, 3]
    assert start_choices(it, 3.0) == [0, 1]
    ids = [x.item_id for x in all_starts([it], 6.0)]
    assert ids == ["x", "x@1", "x@2", "x@3"]
    rs = random_start([it] * 50, 6.0, np.random.default_rng(0))
    assert {r.visits[0].features[0, 0] for r in rs} == {0.0, 1.0, 2.0, 3.0}


@pytest.mark.parametrize("method", ["linear", "logistic"])
def test_half_synthetic_keeps_the_window_and_extrapolates_to_day_15(tmp_path, method):
    man, scans, feats = _write_fake_manifests(tmp_path)
    items = load_a2sl_avocado(man, features=feats, scans=scans, photo_dim=8)
    out = build_half_synthetic(items, method, photo_dim=8, k=2, manifests=man)
    assert [it.item_id for it in out] == ["avo01"]  # avo04 has no photo on days 3 and 5: dropped
    a1, real = out[0], {it.item_id: it for it in items}["avo01"]
    assert [v.t for v in a1.visits] == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
    assert [v.stage for v in a1.visits] == [1, 2, 3, 4, 5, 5, 5, 5] and (a1.spoil_lo, a1.spoil_hi) == (6.0, 8.0)
    assert a1.meta["synthetic_days"] == [7, 9, 11, 13, 15] and a1.meta["post_spoil_visits"]
    for v_new, v_old in zip(a1.visits[:3], real.visits, strict=True):  # the window stays real
        np.testing.assert_allclose(v_new.features, v_old.features)
    if method == "linear":  # avo01's photo rises by 1 per study day
        np.testing.assert_allclose(a1.visits[3].features[0, :8], 7.0, atol=1e-4)
    assert all(np.isfinite(v.features).all() for v in a1.visits)
    # models never see past the first spoiled visit, even with every visit requested
    assert choose_n_obs(a1, "all", None) == 5 and choose_n_obs(real, "all", None) == real.n_visits


def test_paired_random_start_shifts_student_and_teacher_alike():
    stu = [_item() for _ in range(40)]
    tea = [_item() for _ in range(40)]
    s2, t2 = paired_random_start(stu, tea, 6.0, np.random.default_rng(1))
    assert [x.visits[0].features[0, 0] for x in s2] == [x.visits[0].features[0, 0] for x in t2]
    assert len({x.n_visits for x in s2}) > 1  # the starts do vary


def _spoiling_items(n=6, seed=0):
    """Fruit that each age at their own pace: the photo is (age, noise), spoilage at age 8, visits every 2 days."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        age0 = float(rng.uniform(0, 4))
        vs = [Visit(t, 5 if age0 + t >= 8 else 1, features=np.array([[age0 + t, rng.normal()]], np.float32))
              for t in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)]  # fmt: skip
        from ripen.data.schema import spoil_interval_from_visits

        lo, hi = spoil_interval_from_visits(vs)
        out.append(Item(f"f{i}", "avocado", "ambient", math.nan, vs, lo, hi, meta={"label_source": "jmw2026"}))
    return out


@pytest.mark.parametrize("name", ["weibull", "matching"])
def test_fit_directly_models_forecast_properly(name):
    from types import SimpleNamespace

    from ripen.data.collate import collate, dataset_info
    from ripen.models import build_model

    tr, va = _spoiling_items(6, 0), _spoiling_items(2, 1)
    info = dataset_info(tr + va)
    m = build_model({"name": name, "horizon": 12}, info, jax.random.PRNGKey(0))
    m = m.fit(tr, va, info, SimpleNamespace(random_start=True, random_start_max=6.0))
    b = collate(va, info, n_obs=[1, 3])
    F = np.asarray(m.predict_cdf(b, jax.random.PRNGKey(1)))
    assert F.shape == (2, 13) and np.all(np.isfinite(F))
    assert np.all(np.diff(F, axis=1) >= -1e-6) and F.min() >= 0 and F.max() <= 1 + 1e-6
    # an older-looking fruit should be forecast to spoil sooner (median days left)
    old = collate([va[0]], info, n_obs=[3])
    young = collate([va[0]], info, n_obs=[1])
    med = lambda bb: int(np.argmax(np.asarray(m.predict_cdf(bb, jax.random.PRNGKey(1)))[0] >= 0.5))  # noqa: E731
    assert med(old) <= med(young)
