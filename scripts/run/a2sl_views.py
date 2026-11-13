"""Which microwave view predicts best? One model x one input x one view, over 4 folds x seeds (A2SL avocados).

python scripts/run/a2sl_views.py --model latent_sde --input mw --view 120 --seeds 1 2 3

Inputs: ``mw`` microwave only; ``photo_mw`` photo + microwave (the presented setup); ``photo`` photo only;
``time`` neither (ripeness stages and visit times only: the control). ``--view`` is a rotation angle (0, 60, ..., 300),
``all`` (mean and spread over the six, as presented) or ``concat`` (all six side by side in one model). Only avocados
1, 2, 3 and 11 were scanned at every visit, so each fold tests one of them (leave-one-tracked-fruit-out) plus about
three of the fruit cut open on a study day.

``--labels``: ``input`` (as presented: every visit's ripeness label is a model input) or ``hidden`` (labels are
training targets only, never inputs, and the latent SDE's guided posterior, which reads labels at prediction time, is
off). The A2SL labels are set by study day, so with ``input`` the label history alone gives the answer and no sensor
can matter; ``hidden`` makes the model read ripeness from the scan or photo.

Outputs: outputs/a2sl_views/labels_<mode>/seed<S>/<run>/fold<k>/ (the usual train.py run directory). Summary:
``python scripts/analysis/a2sl_views_summary.py --root outputs/a2sl_views/labels_<mode>``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACKED = ["avo01", "avo02", "avo03", "avo11"]
OTHERS = ["avo04", "avo05", "avo06", "avo07", "avo08", "avo09", "avo10", "avo12", "avo13", "avo14", "avo15"]
N_FOLDS = len(TRACKED)
MODEL_ARGS = {
    "latent_sde": ["model=latent_sde", "model.posterior_guidance=true", "model.event_mean=log_mean"],
    "fpt_filter": ["model=fpt_filter"],
    "gru_d": ["model=gru_d"],
    "latent_ode": ["model=latent_ode"],
    "neural_cde": ["model=neural_cde"],
}
INPUT_ARGS = {
    "mw": ["data.features_file=null", "data.photo_dim=0"],
    "photo_mw": [],
    "photo": ["data.scans_file=null"],
    "time": ["model.use_images=false"],  # labels hidden: no features at all (see main)
}
HIDDEN_ARGS = {"latent_sde": ["model.use_stages=false", "model.posterior_guidance=false"],
               "fpt_filter": ["model.use_stages=false"], "gru_d": ["model.use_stages=false"],
               "latent_ode": ["model.use_stages=false"], "neural_cde": ["model.use_stages=false"]}  # fmt: skip


def fold_path(k: int) -> Path:
    return ROOT / "configs" / "splits" / f"a2sl_view_fold{k}.json"


def write_folds() -> None:
    groups = [OTHERS[k::N_FOLDS] for k in range(N_FOLDS)]
    for k in range(N_FOLDS):
        test = [TRACKED[k], *groups[k]]
        val = groups[(k + 1) % N_FOLDS][:2]
        train = [f for f in TRACKED + OTHERS if f not in test and f not in val]
        sp = {"splits": {"train": train, "val": val, "test": test}, "mode": "a2sl_view_cv", "fold": k, "data": "a2sl"}
        fold_path(k).write_text(json.dumps(sp, indent=2))


def scan_view_arg(view: str) -> str:
    """Hydra value for ``data.scan_view``: an int angle, ``concat``, or a quoted pooled subset like ``'0+120+240'``."""
    if view == "concat":
        return view
    return f"'{view}'" if "+" in view else str(int(view))


def run_name(model: str, inp: str, view: str) -> str:
    return f"{model}_{inp}" + ("" if inp in ("photo", "time") else f"_view{view}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_ARGS), required=True)
    ap.add_argument("--input", choices=list(INPUT_ARGS), required=True)
    ap.add_argument(
        "--view",
        default="all",
        help="0, 60, ..., 300, all (six pooled), concat (six side by side) or a pooled subset like 0+120+240",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--folds", type=int, nargs="*", default=list(range(N_FOLDS)))
    ap.add_argument("--labels", choices=["input", "hidden"], default="hidden")
    ap.add_argument("--out", default=None, help="default outputs/a2sl_views/labels_<labels>")
    ap.add_argument("extra", nargs="*", help="extra Hydra overrides (smoke tests)")
    a = ap.parse_args()
    if not all(fold_path(k).exists() for k in range(N_FOLDS)):
        write_folds()
    name = run_name(a.model, a.input, a.view)
    a.out = a.out or f"outputs/a2sl_views/labels_{a.labels}"
    margs = list(MODEL_ARGS[a.model])
    iargs = list(INPUT_ARGS[a.input])
    if a.labels == "hidden":
        margs = [x for x in margs if not x.startswith("model.posterior_guidance")] + HIDDEN_ARGS[a.model]
        if a.input == "time":
            iargs = ["data.features_file=null", "data.scans_file=null"]
    if a.view == "all" or a.input in ("photo", "time"):
        view = []
    else:
        view = [f"data.scan_view={scan_view_arg(a.view)}"]
    fail = 0
    for s in a.seeds:
        for k in a.folds:
            out = Path(a.out) / f"seed{s}" / name / f"fold{k}"
            if (out / "metrics_test.json").exists():
                continue
            cmd = [sys.executable, str(ROOT / "scripts" / "run" / "train.py"), "data=a2sl", "horizon=20",
                   "train.batch_size=8", *margs, *iargs, *view,
                   f"splits.file={fold_path(k)}", f"seed={s}", f"+train.seed={s}", f"run_name={name}",
                   f"out_dir={out}", *a.extra]  # fmt: skip
            print(">>", name, "seed", s, "fold", k, flush=True)
            fail |= subprocess.run(cmd, cwd=ROOT).returncode != 0
    sys.exit(fail)


if __name__ == "__main__":
    main()
