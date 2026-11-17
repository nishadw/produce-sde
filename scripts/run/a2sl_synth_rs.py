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
from a2sl_views import HIDDEN_ARGS, MODEL_ARGS, N_FOLDS, fold_path, write_folds  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
INPUTS = {"photo": ["data.scans_file=null"], "time": ["data.features_file=null", "data.scans_file=null"]}


def run_name(model: str, inp: str, synth: str, start: str) -> str:
    return f"{model}_{inp}_synth-{synth}_start-{start}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_ARGS), required=True)
    ap.add_argument("--input", choices=list(INPUTS), required=True)
    ap.add_argument("--synth", choices=["linear", "logistic"], required=True)
    ap.add_argument("--start", choices=["fixed", "random"], default="fixed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--folds", type=int, nargs="*", default=list(range(N_FOLDS)))
    ap.add_argument("--out", default="outputs/a2sl_synth_rs")
    ap.add_argument("extra", nargs="*", help="extra Hydra overrides (smoke tests)")
    a = ap.parse_args()
    if not all(fold_path(k).exists() for k in range(N_FOLDS)):
        write_folds()
    name = run_name(a.model, a.input, a.synth, a.start)
    margs = [x for x in MODEL_ARGS[a.model] if not x.startswith("model.posterior_guidance")] + HIDDEN_ARGS[a.model]
    run_args = [*INPUTS[a.input], "data.scan_standardize=true", "data.eval_all_starts=true",
                f"train.random_start={'true' if a.start == 'random' else 'false'}"]  # fmt: skip
    run_args += [f"data.synthetic={a.synth}"]
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
