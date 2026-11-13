"""Per-view microwave features for the mango study (``ripen.data.mango``, ``mw_view``).

The scanner is a ring of 10 antennas; ``*_Dividedarray_<k>_<i>_<j>.s2p`` is the sweep between antennas ``i`` and
``j = i + k``. Pairs at the same chord distance ``s = min(k, 10 - k)`` see the fruit the same way, so the 45 pairs
fall into five views: ``sep1`` .. ``sep5`` (10, 10, 10, 10 and 5 pairs; ``sep5`` passes through the centre). A sixth
view is the image reconstruction (``img``). Per sweep view: |S21| and |S11| relative to that day's empty-scanner scan,
averaged over the view's pairs in 20 frequency bands (2 x 20), and their spread over those pairs (2 x 20); ``img`` is
the A2SL reconstruction summary (12). Same numbers as ``ingest_mango.scan_vector``, split by view.

Writes ``data/mango/mango_scan_views.npz`` with keys ``<scan key>/<view>`` (not committed). Run on a compute node:
``python scripts/data/ingest_mango_views.py``.
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest_mango import DAY0, N_BANDS, RAW, read_s2p  # noqa: E402

N_ANT = 10
SWEEP_VIEWS = tuple(f"sep{s}" for s in range(1, N_ANT // 2 + 1))
VIEWS = (*SWEEP_VIEWS, "img")


def sweeps_by_pair(folder: Path) -> dict[tuple[int, int, int], np.ndarray]:
    """``(k, i, j) -> [F, 2]`` |S21|, |S11| in dB."""
    out = {}
    for f in folder.glob("*.s2p"):
        k, i, j = (int(x) for x in re.findall(r"_(\d+)", f.stem)[-3:])
        _, s = read_s2p(f)
        out[(k, i, j)] = 20 * np.log10(np.abs(s[:, [1, 0]]) + 1e-12)
    return out


def view_vectors(sw: dict, air: dict, grid: np.ndarray | None) -> dict[str, np.ndarray]:
    from ripen.data.a2sl import scan_summary

    out = {}
    for s in range(1, N_ANT // 2 + 1):
        pairs = sorted(p for p in sw if min(p[0], N_ANT - p[0]) == s and p in air)
        rel = np.stack([sw[p] - air[p] for p in pairs])  # [P, F, 2]
        bands = np.array_split(np.arange(rel.shape[1]), N_BANDS)
        mean = np.stack([rel[:, b].mean(axis=(0, 1)) for b in bands])
        std = np.stack([rel[:, b].mean(axis=1).std(axis=0) for b in bands])
        out[f"sep{s}"] = np.concatenate([mean.T.ravel(), std.T.ravel()]).astype(np.float32)
    out["img"] = (scan_summary(grid) if grid is not None else np.zeros(12)).astype(np.float32)
    return out


def main() -> None:
    rows = pd.read_csv(ROOT / "data" / "manifests" / "mango_scans.csv")
    want = set(rows.key)
    out, n_pairs = {}, {}
    for ddir in sorted((RAW / "Spring 2026").glob("[0-9][0-9]_[0-9][0-9]_2026")):
        mm, dd, yy = (int(x) for x in ddir.name.split("_"))
        day = (date(yy, mm, dd) - DAY0).days
        air = sweeps_by_pair(next(ddir.glob("air_*")))
        for fdir in sorted(ddir.glob("*_mango_*")):
            key = f"m{int(fdir.name.split('_')[0]):02d}_d{day:02d}"
            if key not in want:
                continue
            xl = next(iter(fdir.glob("*.xlsx")), None)
            grid = pd.read_excel(xl, header=None).to_numpy(float) if xl is not None else None
            sw = sweeps_by_pair(fdir)
            for v, x in view_vectors(sw, air, grid).items():
                out[f"{key}/{v}"] = x
            n_pairs[key] = len(sw)
        print(f"{ddir.name}: day {day}", flush=True)
    missing = want - set(n_pairs)
    np.savez_compressed(RAW / "mango_scan_views.npz", **out)
    print(f"{len(n_pairs)} scans x {len(VIEWS)} views; missing {sorted(missing) or 'none'}; "
          f"pairs per scan {sorted(set(n_pairs.values()))}")  # fmt: skip


if __name__ == "__main__":
    main()
