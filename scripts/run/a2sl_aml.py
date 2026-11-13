"""AML on the A2SL avocados: a photo + microwave teacher, photo-only students, over the same 4 folds as the view study.

python scripts/run/a2sl_aml.py --model latent_sde --method kd --view concat --seeds 1 2 3

Methods (every student sees photos only at test time; objectives in ``ripen.train.aml``):
  photo     photo-only model, no teacher (the baseline)
  teacher   the photo + microwave model itself (the ceiling; needs the scanner at test time)
  frozen teacher (RepDistiller train_partial):
    kd        CAML: 0.1 task + 0.9 t^2 KL(teacher || student) on the days-until-spoiled forecast, t = 4
    feat      task + beta ||F_student - F_teacher||^2 (FitNet-style feature matching)
    kd_feat   both
    rkd / pkt / sp   task + beta x relational KD / probabilistic KT / similarity preserving on the representation
  schedule controls (no microwave anywhere):
    photo_valall  photo, early-stopped on the all-visits validation loss the reset schedule uses
    rounds_only   the reset schedule's rounds and early stopping, without the teacher phase
  teacher reset from the student every round, shared predictor (SAMD; the lab's train_ours_partial):
    samd      + feature MSE
    ours      + forecast KD, gamma 0.1 alpha 0.9 t 4 (the lab's run_our_steering_distill*.sh)
    ours_rkd / ours_pkt / ours_sp
``--alpha/--temp/--beta/--gamma`` override the method's defaults (use ``--tag`` to name the variant);
``--paired-only`` applies the teacher terms only to fruit with at least one scan and trains the teacher on those.
Labels are hidden as inputs (training targets only), as in the view study's ``labels_hidden`` round.

Outputs: outputs/a2sl_aml/seed<S>/<run>/fold<k>/ in the train.py layout (predictions_{val,test}.npz, metrics).
Summary: ``python scripts/analysis/a2sl_aml_summary.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "run"))
from a2sl_views import HIDDEN_ARGS, MODEL_ARGS, N_FOLDS, fold_path, scan_view_arg, write_folds  # noqa: E402

from ripen.models import build_model  # noqa: E402
from ripen.pipeline import dataset_info, evaluate_and_write, load_items, model_cfg_dict, select  # noqa: E402
from ripen.train import TrainConfig, train  # noqa: E402
from ripen.train.aml import METHODS as AML_METHODS  # noqa: E402
from ripen.train.aml import train_distill  # noqa: E402
from ripen.train.samd import train_samd  # noqa: E402

# schedule controls, photo only everywhere (no microwave): plain training with the reset schedule's early-stopping
# loss (every visit visible), and the reset schedule's rounds with no teacher phase
CONTROLS = ("photo_valall", "rounds_only")
METHODS = ("photo", "teacher", *CONTROLS, *AML_METHODS)


def compose_cfg(model: str, view: str, fold: int, seed: int, extra: list[str]):
    margs = [x for x in MODEL_ARGS[model] if not x.startswith("model.posterior_guidance")] + HIDDEN_ARGS[model]
    view_arg = [] if view == "all" else [f"data.scan_view={scan_view_arg(view)}"]
    ov = ["data=a2sl", "horizon=20", "train.batch_size=8", *margs, *view_arg, f"splits.file={fold_path(fold)}",
          f"seed={seed}", f"+train.seed={seed}", *extra]  # fmt: skip
    with initialize_config_dir(str(ROOT / "configs"), version_base="1.3"):
        return compose("config", overrides=ov)


def photo_only(items, n_main: int):
    """Student items: the same fruit and visits with only the photo part of the features."""
    out = []
    for it in items:
        vs = [dataclasses.replace(v, features=None if v.features is None else v.features[..., :n_main])
              for v in it.visits]  # fmt: skip
        out.append(dataclasses.replace(it, visits=vs))
    return out


def has_scan(item, n_main: int) -> bool:
    """Any visit with a non-zero auxiliary (microwave) part."""
    return any(v.features is not None and bool(np.any(v.features[..., n_main:] != 0)) for v in item.visits)


def _strip(argv: list[str]) -> list[str]:
    """argv without its --seeds / --folds values."""
    out, skip = [], False
    for x in argv:
        if x in ("--seeds", "--folds"):
            skip = True
            continue
        if skip and x.lstrip("-").isdigit() and not x.startswith("--"):
            continue
        skip = False
        out.append(x)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_ARGS), required=True)
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--view", default="concat", help="teacher's microwave input: an angle, all (pooled) or concat")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--folds", type=int, nargs="*", default=list(range(N_FOLDS)))
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None, help="override train.lr")
    ap.add_argument("--paired-only", action="store_true")
    ap.add_argument(
        "--teacher-photo-only",
        action="store_true",
        help="control: the teacher sees photos only, so any gain comes from the training schedule",
    )
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tag", default="", help="suffix for the run name (hyperparameter variants)")
    ap.add_argument("--out", default="outputs/a2sl_aml")
    ap.add_argument("extra", nargs="*", help="extra Hydra overrides (smoke tests)")
    a = ap.parse_args()
    if not all(fold_path(k).exists() for k in range(N_FOLDS)):
        write_folds()
    name = f"{a.model}_{a.method}{a.tag}_view{a.view}"
    if len(a.seeds) * len(a.folds) > 1:
        # one process per seed x fold: on CPU, XLA's compiled code is never freed within a process and the jitted
        # distillation step is rebuilt per fold, which exhausted mapped memory after a few folds
        argv = [x for x in sys.argv[1:]]
        fail = 0
        for s in a.seeds:
            for k in a.folds:
                if (ROOT / a.out / f"seed{s}" / name / f"fold{k}" / "metrics_test.json").exists():
                    continue
                cmd = [sys.executable, __file__, *_strip(argv), "--seeds", str(s), "--folds", str(k)]
                fail |= subprocess.run(cmd, cwd=ROOT).returncode != 0
        sys.exit(fail)
    for s in a.seeds:
        for k in a.folds:
            out = ROOT / a.out / f"seed{s}" / name / f"fold{k}"
            if (out / "metrics_test.json").exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            extra = [*a.extra, *([f"train.lr={a.lr}"] if a.lr is not None else [])]
            cfg = compose_cfg(a.model, a.view, k, s, extra)
            OmegaConf.save(cfg, out / "config.yaml")
            (out / "aml.json").write_text(json.dumps(vars(a), indent=2))
            t_items = load_items(cfg.data)
            n_main = int(cfg.data.photo_dim)
            s_items = photo_only(t_items, n_main)
            if a.teacher_photo_only:
                t_items = s_items
            sp = json.loads(fold_path(k).read_text())["splits"]
            info_t, info_s = dataset_info(t_items), dataset_info(s_items)
            tcfg = TrainConfig(**OmegaConf.to_container(cfg.train, resolve=True))
            mcfg = model_cfg_dict(cfg)
            key = jax.random.PRNGKey(1000 * s + k)
            k_tea, k_stu, k_tt, k_st, k_ev = jax.random.split(key, 5)
            tr_t, va_t = select(t_items, sp["train"]), select(t_items, sp["val"])
            tr_s, va_s, te_s = (select(s_items, sp[x]) for x in ("train", "val", "test"))
            print(f">> {name} seed {s} fold {k}: feat teacher {info_t.feat_dim} student {info_s.feat_dim}", flush=True)
            if a.method == "teacher":
                model, _ = train(build_model(mcfg, info_t, k_tea), tr_t, va_t, info_t, tcfg, k_tt, out_dir=out)
                info, te = info_t, select(t_items, sp["test"])
                va = va_t
            elif a.method in ("photo", "photo_valall"):
                if a.method == "photo_valall":
                    tcfg = dataclasses.replace(tcfg, val_n_obs_mode="all")
                model, _ = train(build_model(mcfg, info_s, k_stu), tr_s, va_s, info_s, tcfg, k_st, out_dir=out)
                info, te, va = info_s, te_s, va_s
            elif a.method == "rounds_only":
                ident = {it.item_id: it.item_id for it in s_items}
                by_id = {it.item_id: it for it in s_items}
                spec0 = AML_METHODS["reset_only"][0]
                model, _ = train_samd(build_model(mcfg, info_s, k_stu), build_model(mcfg, info_s, k_tea), tr_s, va_s,
                                      tr_s, ident, by_id, info_s, info_s, tcfg, k_st, n_main, rounds=a.rounds, k=a.k,
                                      spec=spec0, share=False)  # fmt: skip
                info, te, va = info_s, te_s, va_s
            else:
                spec, reset = AML_METHODS[a.method]
                over = {k_: v for k_, v in (("alpha", a.alpha), ("temp", a.temp), ("beta", a.beta),
                                            ("gamma", a.gamma)) if v is not None}  # fmt: skip
                spec = dataclasses.replace(spec, **over)
                t_by_id = {it.item_id: it for it in t_items}
                has_aux = {it.item_id: has_scan(it, n_main) for it in t_items} if a.paired_only else None
                if has_aux is not None:
                    tr_t = [it for it in tr_t if has_aux[it.item_id]]
                    va_t = [it for it in va_t if has_aux[it.item_id]] or va_t
                student = build_model(mcfg, info_s, k_stu)
                if reset:
                    model, _ = train_samd(student, build_model(mcfg, info_t, k_tea), tr_s, va_s, tr_t,
                                          {it.item_id: it.item_id for it in s_items}, t_by_id, info_s, info_t, tcfg,
                                          k_st, n_main, rounds=a.rounds, k=a.k, spec=spec, has_aux=has_aux)  # fmt: skip
                else:
                    teacher, _ = train(build_model(mcfg, info_t, k_tea), tr_t, va_t, info_t, tcfg, k_tt, out_dir=None)
                    model, _ = train_distill(student, teacher, tr_s, va_s, t_by_id, info_s, info_t, tcfg, k_st,
                                             spec=spec, has_aux=has_aux)  # fmt: skip
                info, te, va = info_s, te_s, va_s
            (out / "info.json").write_text(json.dumps(info.to_dict(), indent=2))
            k1, k2 = jax.random.split(k_ev)
            evaluate_and_write(model, va, info, cfg, out, k2, tag="val")
            o = evaluate_and_write(model, te, info, cfg, out, k1, tag="test")["overall"]
            print(f"   test MAE {o['mae']:.2f}  CRPS {o['crps']:.2f}  cov80 {o['coverage80']:.2f}", flush=True)


if __name__ == "__main__":
    main()
