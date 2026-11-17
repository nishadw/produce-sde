"""A2SL fill-in (ripen.data.a2sl_fill) and random-start shifts (ripen.data.schema)."""

import math

import numpy as np
import pytest
from test_a2sl import _write_fake_manifests

from ripen.data.a2sl import load_a2sl_avocado
from ripen.data.a2sl_fill import fill_items
from ripen.data.schema import Item, Visit, all_starts, random_start, shift_start, start_choices


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


@pytest.mark.parametrize("method", ["linear", "carry"])
def test_fill_completes_cut_fruit_and_keeps_real_visits(tmp_path, method):
    man, scans, feats = _write_fake_manifests(tmp_path)
    items = load_a2sl_avocado(man, features=feats, scans=scans, photo_dim=8)
    filled = {it.item_id: it for it in fill_items(items, method, photo_dim=8, k=2, manifests=man)}
    a4 = filled["avo04"]  # cut on day 1: filled to day 9 under the day rule, which ends at its first stage-5 visit
    assert [v.t for v in a4.visits] == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert a4.meta["filled_days"] == [3, 5, 7, 9] and (a4.spoil_lo, a4.spoil_hi) == (6.0, 8.0)
    assert [v.stage for v in a4.visits] == [1, 2, 3, 4, 5]
    orig = {it.item_id: it for it in items}
    np.testing.assert_array_equal(a4.visits[0].features, orig["avo04"].visits[0].features)
    for v_new, v_old in zip(filled["avo01"].visits, orig["avo01"].visits, strict=False):  # real visits unchanged
        np.testing.assert_allclose(v_new.features, v_old.features)
    if method == "carry":
        np.testing.assert_array_equal(a4.visits[3].features[0, :8], orig["avo04"].visits[0].features[0, :8])
    else:  # avo01's photos rise by 1 per study day, so the shared slope carries avo04 up the same way
        np.testing.assert_allclose(a4.visits[3].features[0, :8] - a4.visits[0].features[0, :8], 6.0, atol=1e-4)


def test_fill_copies_add_noise_only_to_filled_visits(tmp_path):
    man, scans, feats = _write_fake_manifests(tmp_path)
    items = load_a2sl_avocado(man, features=feats, scans=scans, photo_dim=8)
    out = {it.item_id: it for it in fill_items(items, "linear", photo_dim=8, k=2, copies=3, manifests=man)}
    assert {"avo04", "avo04#1", "avo04#2"} <= set(out)
    np.testing.assert_array_equal(out["avo04#1"].visits[0].features, out["avo04"].visits[0].features)
    assert out["avo04#1"].visits[0].features.shape == out["avo04"].visits[0].features.shape
