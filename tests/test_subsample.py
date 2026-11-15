"""`subsample_visits`: a sparser photo schedule widens the spoilage interval; it never loses the event."""

import pytest

from ripen.data import SPOILED_STAGE, Item, SyntheticConfig, Visit, simulate, subsample_visits
from ripen.data.schema import spoil_interval_from_visits


def daily_item(stages: list[int]) -> Item:
    """One visit per day with the given stages; stops at the first spoiled visit, as the loaders do."""
    visits = []
    for d, s in enumerate(stages):
        visits.append(Visit(t=float(d), stage=s))
        if s >= SPOILED_STAGE:
            break
    lo, hi = spoil_interval_from_visits(visits)
    return Item("a", "avocado", "T20", 20.0, visits, lo, hi, meta={"src": 1})


def test_on_schedule_spoilage():
    it = daily_item([1, 2, 3, 4, 4, 4, 5])  # first spoiled at index 6
    sub = subsample_visits(it, 3)
    assert [v.t for v in sub.visits] == [0.0, 3.0, 6.0]
    assert sub.visits[-1].stage == SPOILED_STAGE
    assert (sub.spoil_lo, sub.spoil_hi) == (3.0, 6.0)
    assert not sub.meta["subsample"]["hi_extrapolated"]


@pytest.mark.parametrize("every, k, lo, hi", [(3, 4, 3.0, 6.0), (3, 5, 3.0, 6.0), (2, 5, 4.0, 6.0), (2, 3, 2.0, 4.0)])
def test_off_schedule_spoilage_widens_interval(every, k, lo, hi):
    it = daily_item([1] * k + [SPOILED_STAGE])
    assert (it.spoil_lo, it.spoil_hi) == (k - 1.0, float(k))
    sub = subsample_visits(it, every)
    assert not sub.right_censored
    assert (sub.spoil_lo, sub.spoil_hi) == (lo, hi)
    assert sub.spoil_lo <= it.spoil_lo and sub.spoil_hi >= it.spoil_hi
    # only kept, unspoiled visits are visible: the off-schedule spoiled photo is not an input
    assert all(i % every == 0 for i in (int(v.t) for v in sub.visits))
    assert all(v.stage < SPOILED_STAGE for v in sub.visits)
    assert sub.t_last == lo
    assert sub.meta["subsample"]["hi_extrapolated"] and sub.meta["src"] == 1
    assert "subsample" not in it.meta  # input item untouched


def test_off_schedule_uses_visit_spacing():
    visits = [Visit(t=2.0 * i, stage=1) for i in range(4)] + [Visit(t=8.0, stage=SPOILED_STAGE)]
    it = Item("b", "avocado", "T20", 20.0, visits, 6.0, 8.0)
    sub = subsample_visits(it, 3)  # kept indices 0, 3; spoiled at 4 -> first seen at index 6 = t 12
    assert (sub.spoil_lo, sub.spoil_hi) == (6.0, 12.0)


def test_offset_schedule():
    it = daily_item([1, 1, 1, 1, SPOILED_STAGE])  # spoiled at 4; offset 1 keeps 0, 1, 3, 5
    sub = subsample_visits(it, 2, offset=1)
    assert [v.t for v in sub.visits] == [0.0, 1.0, 3.0]
    assert (sub.spoil_lo, sub.spoil_hi) == (3.0, 5.0)


def test_right_censored_stays_censored():
    it = daily_item([1, 2, 3, 4, 4, 4, 4, 4])
    assert it.right_censored
    sub = subsample_visits(it, 3)
    assert sub.right_censored
    assert [v.t for v in sub.visits] == [0.0, 3.0, 6.0]
    assert sub.spoil_lo == 6.0


def test_every_one_is_identity():
    for it in [daily_item([1, 2, 5]), daily_item([1, 2, 3])]:
        assert subsample_visits(it, 1) == it


def test_no_event_lost_on_synthetic():
    items = simulate(SyntheticConfig(n_items=60, t_max=30.0, seed=4))
    for every in (2, 3):
        subs = [subsample_visits(it, every) for it in items]
        assert [s.right_censored for s in subs] == [it.right_censored for it in items]
        for it, s in zip(items, subs, strict=True):
            if not s.right_censored:
                assert s.spoil_lo <= it.spoil_lo < it.spoil_hi <= s.spoil_hi
                assert s.spoil_hi - s.spoil_lo <= 2 * every - 1 + 1e-9


@pytest.mark.parametrize("every", [2, 3])
def test_first_visit_spoiled_and_single_visit(every):
    sub = subsample_visits(daily_item([SPOILED_STAGE]), every)  # spoiled at the first (always kept) visit
    assert [v.t for v in sub.visits] == [0.0] and (sub.spoil_lo, sub.spoil_hi) == (0.0, 0.0)
    assert not sub.meta["subsample"]["hi_extrapolated"]
    single = subsample_visits(daily_item([2]), every)  # one unspoiled visit: right-censored at it
    assert single.right_censored and single.spoil_lo == 0.0 and len(single.visits) == 1


def test_rejects_bad_every():
    with pytest.raises(ValueError):
        subsample_visits(daily_item([1, 5]), 0)
