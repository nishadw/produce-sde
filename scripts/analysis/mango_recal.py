"""Cross-fitted recalibration of every mango run: <root>/seed*/<run>/ -> <root>_recal/seed*/<run>/.

Same map as the avocado study (``PITRecalibrator``, 0.9 Kaplan-Meier PIT map + 0.1 raw forecast): for each test
fold, fitted on the other five folds' test forecasts of the same run and seed (cut-day and earlier-scan instances),
so no fruit's own label enters its forecast. Then rescored with ``mango_cv.write_result``.

    python scripts/analysis/mango_recal.py outputs/mango_views
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "run"))
from mango_cv import folds, write_result  # noqa: E402

from ripen.data.mango import load_mango  # noqa: E402
from ripen.eval.recalibrate import PITRecalibrator  # noqa: E402

WEIGHT = 0.9


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/mango_views")
    out_root = root.parent / (root.name + "_recal")
    items = load_mango("mw")  # every instance id (cut day and earlier scans) with its labels
    fold_list = folds()
    fold_of = {f: k for k, sp in enumerate(fold_list) for f in sp["test"]}
    for p in sorted(root.glob("seed*/*/predictions.npz")):
        out = out_root / p.parent.parent.name / p.parent.name
        if (out / "metrics.json").exists() and (out / "metrics.json").stat().st_mtime > p.stat().st_mtime:
            continue
        z = dict(np.load(p, allow_pickle=True))
        fk = np.array([fold_of[str(i).split("_d")[0]] for i in z["item_ids"]])
        F = z["F"].copy()
        for k in np.unique(fk):
            o = fk != k
            cal = PITRecalibrator.fit(z["F"][o], z["lo"][o], z["hi"][o], weight=WEIGHT)
            F[fk == k] = cal(z["F"][fk == k])
        write_result(out, F, z["lo"], z["hi"], z["cut"], z["item_ids"], items, fold_list)


if __name__ == "__main__":
    main()
