"""Recalibrate finished runs from their saved predictions: fit on predictions_val.npz, apply to predictions_test.npz.

python scripts/analysis/recalibrate.py outputs/avocado/round1              # every run dir below this root
python scripts/analysis/recalibrate.py outputs/runs/latent_sde_avocado     # one run

Writes ``metrics_test_recal.json`` (same layout as ``metrics_test.json``) next to the predictions. CPU only, seconds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ripen.eval.metrics import summarize_by_group
from ripen.eval.recalibrate import PITRecalibrator


def recalibrate_run(run: Path) -> dict | None:
    pv, pt = run / "predictions_val.npz", run / "predictions_test.npz"
    if not (pv.exists() and pt.exists()):
        return None
    groups = json.loads((run / "info.json").read_text())["groups"]
    with np.load(pv) as v, np.load(pt) as t:
        recal = PITRecalibrator.fit(v["F"], v["lo"], v["hi"])
        F = recal(t["F"])
        out = summarize_by_group(F, t["lo"], t["hi"], t["group"], groups, t["item_ids"])
    out["recalibrator"] = {"x": recal.x.tolist(), "G": recal.G.tolist()}
    (run / "metrics_test_recal.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    runs = []
    for p in map(Path, args.paths):
        found = [p] if (p / "predictions_val.npz").exists() else [q.parent for q in p.rglob("predictions_val.npz")]
        runs += sorted(found)
    for run in runs:
        out = recalibrate_run(run)
        if out is None:
            print(f"skip {run}: missing predictions")
            continue
        before = json.loads((run / "metrics_test.json").read_text())["overall"]
        a = out["overall"]
        print(
            f"{run}: cov80 {before['coverage80']:.2f}->{a['coverage80']:.2f} "
            f"width {before['width80']:.2f}->{a['width80']:.2f} CRPS {before['crps']:.3f}->{a['crps']:.3f} "
            f"ECE {before['ece']:.3f}->{a['ece']:.3f}"
        )


if __name__ == "__main__":
    main()
