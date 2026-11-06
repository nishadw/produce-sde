"""Turn the raw mango drop (``data/mango/``, never committed) into manifests and a microwave-feature cache.

Layout of the raw data (Spring 2026, one batch of mangoes bought together, stored together):

* ``Mango data classification.xlsx`` — one row per fruit, measured on the day it was **cut open** (destructive):
  study day (0, 2, ..., 14), sample number, hue angle, firmness (N), sweetness (°Brix), and a ripeness class from
  firmness: Unripe > 30 N, Underripe 15–30, Ripe 3–15, Overripe 1–3, Senescent ≤ 1 N. Six fruit per day.
* ``Mango pictures/D{day}_{id}_{side}.JPG`` — two photos of each fruit, taken on its cut day only.
* ``Spring 2026/{MM_DD_YYYY}/{id}_mango_{date}/`` — a microwave scan of every not-yet-cut fruit every other day:
  45 two-port S-parameter sweeps (``*_Dividedarray_*.s2p``, 3.3–8 GHz, 201 points; one per antenna pair) and a
  72x72 image reconstruction (``.xlsx``, NaN outside the imaged disk). ``air_{date}`` is the empty-scanner scan of
  that day.

Outputs (``data/manifests/mango_*.csv`` are committed; the ``.npz`` is not):

    data/manifests/mango_fruit.csv     one row per labelled fruit: cut day, hue, firmness, sweetness, class, stage 1..5
    data/manifests/mango_scans.csv     one row per (fruit, scan day): scan key
    data/manifests/mango_photos.csv    one row per photo
    data/mango/mango_scans.npz         scan key -> raw scan vector (see ``scan_vector``)

Run once on a compute node: ``python scripts/ingest_mango.py``.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "mango"
MAN = ROOT / "data" / "manifests"
DAY0 = date(2026, 1, 14)
STAGE = {"Unripe": 1, "Underripe": 2, "Ripe": 3, "Overripe": 4, "Senecent": 5, "Senescent": 5}
N_BANDS = 20


def read_s2p(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """-> (freq_hz [F], S [F, 4] complex: S11, S21, S12, S22)."""
    rows = [ln.split() for ln in path.read_text().splitlines() if ln.strip() and ln.lstrip()[0] not in "!#"]
    a = np.asarray(rows, float)
    s = a[:, 1::2] + 1j * a[:, 2::2]
    return a[:, 0], s


def sweep_db(folder: Path) -> np.ndarray:
    """``[45, F, 2]``: |S21| and |S11| in dB for each antenna pair, pairs in sorted file order."""
    files = sorted(folder.glob("*.s2p"), key=lambda p: [int(x) for x in re.findall(r"_(\d+)", p.stem)[-3:]])
    out = []
    for f in files:
        _, s = read_s2p(f)
        out.append(20 * np.log10(np.abs(s[:, [1, 0]]) + 1e-12))
    return np.stack(out)


def scan_vector(sweep: np.ndarray, air: np.ndarray, grid: np.ndarray | None) -> np.ndarray:
    """Fixed (unlearned) features of one scan.

    * |S21| and |S11| relative to that day's empty-scanner scan, averaged over the 45 antenna pairs, in ``N_BANDS``
      frequency bands (2 x 20), and their spread over pairs (2 x 20);
    * the image-reconstruction summary of the A2SL loader (12).
    """
    from ripen.data.a2sl import scan_summary

    rel = sweep - air  # [45, F, 2] dB relative to air
    bands = np.array_split(np.arange(rel.shape[1]), N_BANDS)
    mean = np.stack([rel[:, b].mean(axis=(0, 1)) for b in bands])  # [N_BANDS, 2]
    std = np.stack([rel[:, b].mean(axis=1).std(axis=0) for b in bands])  # [N_BANDS, 2]
    img = scan_summary(grid) if grid is not None else np.zeros(12, np.float32)
    return np.concatenate([mean.T.ravel(), std.T.ravel(), img]).astype(np.float32)


def main() -> None:
    MAN.mkdir(parents=True, exist_ok=True)
    cls = pd.read_excel(RAW / "Mango data classification.xlsx", sheet_name="Sheet1").iloc[:, :9]
    cls.columns = ["cut_day", "fruit", "hue", "hue_sd", "firmness_n", "firmness_sd", "brix", "brix_sd", "cls"]
    cls["cls"] = cls["cls"].str.strip()
    cls["stage"] = cls["cls"].map(STAGE).astype(int)
    cls["fruit"] = cls["fruit"].astype(int)
    cls = cls.sort_values(["cut_day", "fruit"]).reset_index(drop=True)
    cls.to_csv(MAN / "mango_fruit.csv", index=False)
    labelled = set(cls.fruit)

    photos = []
    for p in sorted((RAW / "Mango pictures").glob("*.JPG")):
        m = re.match(r"D(\d+)_(\d+)_(\d+)", p.stem)
        photos.append(
            {
                "fruit": int(m.group(2)),
                "day": int(m.group(1)),
                "side": int(m.group(3)),
                "path": str(p.relative_to(ROOT)),
            }
        )
    photos = pd.DataFrame(photos)
    photos.to_csv(MAN / "mango_photos.csv", index=False)

    scans, rows, unlabelled = {}, [], set()
    for ddir in sorted((RAW / "Spring 2026").glob("[0-9][0-9]_[0-9][0-9]_2026")):
        mm, dd, yy = (int(x) for x in ddir.name.split("_"))
        day = (date(yy, mm, dd) - DAY0).days
        air_dir = next(ddir.glob("air_*"))
        air = sweep_db(air_dir)
        for fdir in sorted(ddir.glob("*_mango_*")):
            fruit = int(fdir.name.split("_")[0])
            if fruit not in labelled:
                unlabelled.add(fruit)
                continue
            xl = next(iter(fdir.glob("*.xlsx")), None)
            grid = pd.read_excel(xl, header=None).to_numpy(float) if xl is not None else None
            key = f"m{fruit:02d}_d{day:02d}"
            scans[key] = scan_vector(sweep_db(fdir), air, grid)
            rows.append({"fruit": fruit, "day": day, "key": key, "n_s2p": len(list(fdir.glob("*.s2p")))})
        print(f"{ddir.name}: day {day}, {sum(r['day'] == day for r in rows)} labelled fruit scanned")
    sc = pd.DataFrame(rows).sort_values(["fruit", "day"])
    sc.to_csv(MAN / "mango_scans.csv", index=False)
    np.savez_compressed(RAW / "mango_scans.npz", **scans)

    cut = dict(zip(cls.fruit, cls.cut_day, strict=True))
    last_scan = sc.groupby("fruit").day.max()
    bad = {f: (int(last_scan.get(f, -1)), int(d)) for f, d in cut.items() if last_scan.get(f, -1) != d}
    print(
        f"{len(cls)} labelled fruit, {len(photos)} photos, {len(sc)} scans; "
        f"scanned but unlabelled: {sorted(unlabelled)}"
    )
    print(f"fruit whose last scan is not on the cut day (last scan, cut day): {bad or 'none'}")


if __name__ == "__main__":
    main()
