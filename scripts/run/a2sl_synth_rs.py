"""A2SL half-synthetic data x random start (docs/plans/2026-11-17_a2sl-imputation-random-start.md).

One run = one model x input x synthetic method x start, over seeds x the 4 leave-one-tracked-fruit-out folds of
scripts/run/a2sl_views.py (labels hidden). Data: ``ripen.data.a2sl_synth`` (real days 1-5, regression to day 15).
Every run scores each test fruit from every start (``data.eval_all_starts``), on the synthetic test fruit
(``predictions_test``) and on their real visits (``predictions_test_real``).

python scripts/run/a2sl_synth_rs.py --model latent_ode --input photo --synth logistic --start random
Writes outputs/a2sl_synth_rs/seed<s>/<run>/fold<k>/; summary: scripts/analysis/a2sl_synth_rs_summary.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a2sl_views import HIDDEN_ARGS, MODEL_ARGS, folds_for  # noqa: E402

# the two fit-directly baselines (ripen.models.classical) take no extra arguments
MODEL_ARGS = {
    **MODEL_ARGS,
    "weibull": ["model=weibull"],
    "matching": ["model=matching"],
    "storage_day": ["model=storage_day"],
}
HIDDEN_ARGS = {**HIDDEN_ARGS, "weibull": [], "matching": [], "storage_day": []}

ROOT = Path(__file__).resolve().parents[2]
INPUTS = {"photo": ["data.scans_file=null"], "time": ["data.features_file=null", "data.scans_file=null"]}


def run_name(model: str, inp: str, synth: str, start: str) -> str:
    return f"{model}_{inp}_synth-{synth}_start-{start}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_ARGS), required=True)
    ap.add_argument("--input", choices=list(INPUTS), required=True)
    ap.add_argument("--synth", choices=["linear", "logistic", "real"], required=True,
                    help="real = the loader's real visits only, no fill")  # fmt: skip
    ap.add_argument("--split", choices=["tracked4", "all"], default="tracked4",
                    help="tracked4: avo01/02/03/11 held out (4 folds); all: each of avo01-12 (12 folds)")  # fmt: skip
    ap.add_argument("--start", choices=["fixed", "random"], default="fixed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--out", default="outputs/a2sl_synth_rs")
    ap.add_argument("--features", default=None, help="photo feature cache (default: the data config's ResNet-18)")
    ap.add_argument("--tag", default="", help="appended to the run name, e.g. _fruit for another feature cache")
    ap.add_argument("extra", nargs="*", help="extra Hydra overrides (smoke tests)")
    a = ap.parse_args()
    n_folds, fold_path = folds_for(a.split)
    a.folds = list(range(n_folds)) if a.folds is None else a.folds
    name = run_name(a.model, a.input, a.synth, a.start) + a.tag
    margs = [x for x in MODEL_ARGS[a.model] if not x.startswith("model.posterior_guidance")] + HIDDEN_ARGS[a.model]
    run_args = [*INPUTS[a.input], "data.scan_standardize=true", "data.eval_all_starts=true",
                f"train.random_start={'true' if a.start == 'random' else 'false'}"]  # fmt: skip
    run_args += [] if a.synth == "real" else [f"data.synthetic={a.synth}"]
    run_args += [f"data.features_file={a.features}"] if a.features else []
    fail = 0
    for s in a.seeds:
        for k in a.folds:
            out = Path(a.out) / f"seed{s}" / name / f"fold{k}"
            if (out / "metrics_test.json").exists():
                continue
            cmd = [sys.executable, str(ROOT / "scripts" / "run" / "train.py"), "data=a2sl", "horizon=20",
                   "train.batch_size=8", *margs, *run_args, f"splits.file={fold_path(k)}", f"seed={s}",
                   f"+train.seed={s}", f"run_name={name}", f"out_dir={out}", *a.extra]  # fmt: skip
            print(">>", name, "seed", s, "fold", k, flush=True)
            fail |= subprocess.run(cmd, cwd=ROOT).returncode != 0
    sys.exit(fail)


if __name__ == "__main__":
    main()
