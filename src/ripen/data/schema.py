"""The one record every loader emits.

An :class:`Item` is one piece of produce followed over time. Its spoilage time
is never observed exactly: it lies in ``(spoil_lo, spoil_hi]`` where
``spoil_lo`` is the last visit at which the item was seen unspoiled and
``spoil_hi`` the first visit at which it was seen spoiled. If it was still
unspoiled at the last visit, ``spoil_hi`` is ``inf`` (right-censored).
After :func:`subsample_visits`, ``spoil_hi`` can be the time of a scheduled visit
past the last kept one (``meta["subsample"]["hi_extrapolated"]``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

N_STAGES = 5
SPOILED_STAGE = 5  # end of shelf life in the 5-stage Ripening Index of Xavier et al. (2024)


@dataclass(frozen=True)
class Visit:
    """One observation of an item.

    Args:
        t: days since the item's first visit.
        stage: ripeness stage in ``1..N_STAGES``, or ``None`` if unlabelled.
        image_paths: paths of the photos taken at this visit (any number of views).
        features: cached encoder features, shape ``(n_images, D)``, or ``None``.
    """

    t: float
    stage: int | None = None
    image_paths: tuple[str, ...] = ()
    features: np.ndarray | None = None

    @property
    def feature_vector(self) -> np.ndarray | None:
        """Mean over views, shape ``(D,)``; ``None`` if no features are cached."""
        if self.features is None:
            return None
        return np.asarray(self.features, dtype=np.float32).reshape(-1, self.features.shape[-1]).mean(0)


@dataclass
class Item:
    """One piece of produce and its visit history.

    Args:
        item_id: unique identifier.
        produce: produce type, e.g. ``"avocado"``.
        storage: storage group label, e.g. ``"T10"``.
        temp_c: storage temperature in Celsius (``nan`` if unknown).
        visits: visits sorted by time; the first has ``t == 0``.
        spoil_lo: last time the item was known unspoiled.
        spoil_hi: first time the item was known spoiled; ``inf`` if right-censored.
        rh: relative humidity in percent (``nan`` if unknown).
        meta: free-form extras (ground truth for synthetic items, source rows, ...).
    """

    item_id: str
    produce: str
    storage: str
    temp_c: float
    visits: list[Visit]
    spoil_lo: float
    spoil_hi: float
    rh: float = math.nan
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.visits:
            raise ValueError(f"{self.item_id}: an item needs at least one visit")
        ts = [v.t for v in self.visits]
        if ts != sorted(ts):
            raise ValueError(f"{self.item_id}: visits must be sorted by time")
        if not (self.spoil_lo <= self.spoil_hi):
            raise ValueError(f"{self.item_id}: need spoil_lo <= spoil_hi, got {self.spoil_lo} > {self.spoil_hi}")

    @property
    def right_censored(self) -> bool:
        return math.isinf(self.spoil_hi)

    @property
    def censor_type(self) -> str:
        """``"right"``, ``"exact"`` (interval of zero width) or ``"interval"``."""
        if self.right_censored:
            return "right"
        return "exact" if self.spoil_hi - self.spoil_lo < 1e-9 else "interval"

    @property
    def n_visits(self) -> int:
        return len(self.visits)

    @property
    def t_last(self) -> float:
        return self.visits[-1].t

    def prefix(self, n_visits: int) -> Item:
        """The same item with only the first ``n_visits`` visits visible. The spoilage interval is unchanged."""
        n = max(1, min(n_visits, len(self.visits)))
        return replace(self, visits=self.visits[:n])


def spoil_interval_from_visits(visits: list[Visit]) -> tuple[float, float]:
    """Derive ``(spoil_lo, spoil_hi)`` from labelled visits: first visit at ``SPOILED_STAGE`` ends the interval."""
    lo = visits[0].t
    for v in visits:
        if v.stage is not None and v.stage >= SPOILED_STAGE:
            return lo, v.t
        lo = v.t
    return lo, math.inf


def subsample_visits(item: Item, every: int, offset: int = 0) -> Item:
    """Keep every ``every``-th visit (by index, starting at ``offset``), then recompute the spoilage interval.

    This is how "photos every other day" is simulated from daily data. Visit index ``i`` is on the schedule if
    ``i == 0`` (the first visit is always kept) or ``(i - offset) % every == 0``. Loaders stop each item at its
    first spoiled visit ``k``. Spoilage persists, so under the sparser schedule the fruit is first *seen* spoiled
    at the first scheduled index ``k' >= k``:

    - ``k`` on the schedule: the kept visits end at visit ``k``; the interval is
      ``(t of the last kept visit before k, t_k]``.
    - ``k`` off the schedule: the loaders drop every photo after ``k``, so ``t_{k'}`` is extrapolated as
      ``t_k + (k' - k) * dt`` with ``dt`` the item's median visit spacing (1 day for the daily avocado data).
      The interval widens to ``(t of the last kept visit before k, t_{k'}]`` and the item stays uncensored.
      The visible visits end at that last kept unspoiled visit, so the off-schedule spoiled photo is never an
      input. ``meta["subsample"]["hi_extrapolated"]`` is ``True`` for these items.
    - no spoiled visit (right-censored): still right-censored, at the last kept visit.

    ``every == 1`` returns the item unchanged.
    """
    if every < 1:
        raise ValueError("every must be >= 1")
    if every == 1:
        return item

    def on_schedule(i: int) -> bool:
        return i == 0 or (i - offset) % every == 0

    k = next((i for i, v in enumerate(item.visits) if v.stage is not None and v.stage >= SPOILED_STAGE), None)
    end = len(item.visits) if k is None else k + 1
    out = [v for i, v in enumerate(item.visits[:end]) if on_schedule(i)]
    meta = {**item.meta, "subsample": {"every": every, "offset": offset, "hi_extrapolated": False}}
    if k is None or on_schedule(k):
        lo, hi = spoil_interval_from_visits(out)
        return replace(item, visits=out, spoil_lo=lo, spoil_hi=hi, meta=meta)
    # k is off the schedule (so k > 0 and `out` holds only unspoiled visits): first seen spoiled at k' > k
    k_next = k + 1
    while not on_schedule(k_next):
        k_next += 1
    gaps = np.diff([v.t for v in item.visits[: k + 1]])
    gaps = gaps[gaps > 0]
    dt = float(np.median(gaps)) if gaps.size else 1.0
    hi = float(item.visits[k].t + (k_next - k) * dt)
    meta["subsample"]["hi_extrapolated"] = True
    return replace(item, visits=out, spoil_lo=out[-1].t, spoil_hi=hi, meta=meta)


# ---- random start: the model sees a fruit from some visit on, with time re-zeroed there ----------------------


def shift_start(item: Item, k: int) -> Item:
    """``item`` seen from its ``k``-th visit on, with time re-zeroed there, so the model never sees how long the fruit
    was stored before. ``k`` must leave at least one visit before the spoilage interval (``t_k <= spoil_lo``)."""
    if k == 0:
        return item
    t0 = item.visits[k].t
    if t0 > item.spoil_lo + 1e-9:
        raise ValueError(f"start visit {k} of {item.item_id} is past its last unspoiled visit")
    vs = [replace(v, t=v.t - t0) for v in item.visits[k:]]
    return replace(item, visits=vs, spoil_lo=item.spoil_lo - t0, spoil_hi=item.spoil_hi - t0,
                               meta={**item.meta, "start_shift": t0})  # fmt: skip


def start_choices(item: Item, max_shift: float) -> list[int]:
    """Visit indices a random start may begin at: on or before the last unspoiled visit, at most ``max_shift`` days
    after the first visit."""
    return [i for i, v in enumerate(item.visits) if v.t <= min(item.spoil_lo, max_shift) + 1e-9]


def random_start(items: list[Item], max_shift: float, rng: np.random.Generator) -> list[Item]:
    """Every item from a random allowed start (``start_choices``); used once per training epoch."""
    return [shift_start(it, int(rng.choice(start_choices(it, max_shift)))) for it in items]


def all_starts(items: list[Item], max_shift: float) -> list[Item]:
    """Every item from every allowed start; a shifted copy is named ``<id>@<k>`` (``k`` = first visit kept), so the
    unshifted original keeps its id and scores exactly as before."""
    out = []
    for it in items:
        for k in start_choices(it, max_shift):
            sh = shift_start(it, k)
            out.append(sh if k == 0 else replace(sh, item_id=f"{it.item_id}@{k}"))
    return out


def paired_random_start(
    items: list[Item], partners: list[Item], max_shift: float, rng: np.random.Generator
) -> tuple[list[Item], list[Item]]:
    """One random start per fruit, applied to both views of it (student and teacher items of the same fruit and
    visits), so distillation compares forecasts made from the same visits."""
    ks = [int(rng.choice(start_choices(it, max_shift))) for it in items]
    return [shift_start(it, k) for it, k in zip(items, ks, strict=True)], [
        shift_start(p, k) for p, k in zip(partners, ks, strict=True)
    ]
