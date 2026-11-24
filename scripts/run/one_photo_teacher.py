"""One-photo study, phase 2: the microwave scans as a training-only teacher (docs/plans/2026-11-22_one-photo-app.md).

Every run starts from the phase-1 model trained on the public avocados (outputs/one_photo/<model>_<features>/seed<s>)
and fine-tunes on the lab avocados, one lab fruit held out at a time (avo01-avo12; avo13-15 were photographed already
spoiled and always train). Every example is still one photo; the student never sees a scan.

Methods (fold<k>/<method>/):
  finetune   photo-only fine-tune on the lab training photos (no teacher; the control for everything below)
  teacher    the public model widened to photo + scan (scan inputs start at zero, SAMD's supermodel reset), fine-tuned
             on the lab photos with their scans; scored WITH scans (the ceiling: needs the scanner)
  kd         CAML: student fine-tuned with 0.1 task + 0.9 t^2 KL(teacher || student), t = 4, on scanned photos only
  kd_photo   control: the same, but the teacher is the photo-only fine-tune (any gain here is not from the scans)
  samd       SAMD: teacher reset from the student every round, shared predictor, feature MSE on scanned photos
``--fill linear|logistic``: the training fruit are extended with regression-extrapolated photos and scans
(``ripen.data.a2sl_synth``, fit on the training fruit only); validation and test stay real. kd_photo is skipped.

python scripts/run/one_photo_teacher.py --model latent_sde --features fruit --seed 1 [--folds 0 1]
Writes outputs/one_photo_teacher/<model>_<features>/seed<s>/fold<k>/<method>/{metrics,predictions}_test.*
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

from ripen.data.a2sl_synth import build_half_synthetic
from ripen.data.schema import one_photo
from ripen.models import build_model
from ripen.pipeline import build_from_run, dataset_info, evaluate_and_write, load_items
from ripen.train import TrainConfig, train
from ripen.train.aml import METHODS as AML
from ripen.train.aml import train_distill
from ripen.train.samd import reset_teacher, train_samd

ROOT = Path(__file__).resolve().parents[2]
LAB_FEATURES = {  # lab photo features, the same variants as phase 1
    "fruit": ("data/features/a2sl_fruit_resnet18.npz", 512),
    "colour": ("data/features/a2sl_colour_pub.npz", 12),
}
TEST_FRUIT = [f"avo{i:02d}" for i in range(1, 13)]
VAL_POOL = ["avo08", "avo09", "avo10", "avo12"]  # tracked, never scanned: every scanned fruit stays in training
METHODS = ("finetune", "teacher", "kd", "kd_photo", "samd")


def fold_split(k: int) -> dict[str, list[str]]:
    test = TEST_FRUIT[k]
    val = [f for f in VAL_POOL if f != test][:2]
    train_ = [f"avo{i:02d}" for i in range(1, 16) if f"avo{i:02d}" not in (test, *val)]
    return {"train": train_, "val": val, "test": [test]}


def photo_only(it, dim: int):
    vs = [dataclasses.replace(v, features=v.features[..., :dim]) for v in it.visits]
    return dataclasses.replace(it, visits=vs)


def fruit_of(it) -> str:
    return it.item_id.split("@")[0]


def run_fold(a, k: int) -> None:
    lab_f, dim = LAB_FEATURES[a.features]
    src = ROOT / "outputs" / "one_photo" / f"{a.model}_{a.features}" / f"seed{a.seed}"
    public, cfg, info_s = build_from_run(src)
    root = ROOT / a.out / f"{a.model}_{a.features}" / f"seed{a.seed}" / f"fold{k}"
    with initialize_config_dir(str(ROOT / "configs"), version_base="1.3"):
        lab_cfg = compose("config", overrides=["data=a2sl", f"data.features_file={lab_f}", f"data.photo_dim={dim}",
                                               "data.scan_standardize=true"]).data  # fmt: skip
    lab = load_items(lab_cfg)
    sp = fold_split(k)
    pick = {s: [it for it in lab if it.item_id in sp[s]] for s in sp}
    if a.fill != "none":  # training fruit only: the regression never sees the test or validation fruit
        syn = build_half_synthetic(pick["train"], a.fill, photo_dim=dim, has_scans=True,
                                   manifests=lab_cfg.get("manifests", "data/manifests"))  # fmt: skip
        ids = {it.item_id for it in syn}
        pick["train"] = syn + [it for it in pick["train"] if it.item_id not in ids]
        print(f"   {a.fill} fill: {len(ids)} training fruit extrapolated to day 15", flush=True)
    tr_t, va_t = one_photo(pick["train"]), one_photo(pick["val"])
    te_t = one_photo(pick["test"], hide_stages=True)
    tr_s, va_s, te_s = ([photo_only(it, dim) for it in x] for x in (tr_t, va_t, te_t))
    scan_dim = tr_t[0].visits[0].features.shape[-1] - dim
    info_t = dataclasses.replace(info_s, feat_dim=dim + scan_dim)
    assert dataset_info(tr_s).feat_dim == info_s.feat_dim
    has_aux = {it.item_id: bool(np.any(it.visits[0].features[..., dim:] != 0)) for it in tr_t + va_t + te_t}
    t_by_id = {it.item_id: it for it in tr_t}
    s_by_id = {it.item_id: it for it in tr_s}
    ft = {"lr": a.lr, "batch_size": 8, "epochs": a.epochs, "patience": a.patience, "n_obs_mode": "all"}
    tcfg = TrainConfig(**{**OmegaConf.to_container(cfg.train, resolve=True), **ft})
    mcfg = OmegaConf.to_container(cfg.model, resolve=True)
    key = jax.random.PRNGKey(1000 * a.seed + k)
    kt, ks, kf, kk, kp, ksm, ke = jax.random.split(key, 7)
    n_scan = sum(has_aux[it.item_id] for it in tr_t)
    print(f">> fold {k} test {sp['test']} val {sp['val']}: {len(tr_s)} train photos ({n_scan} scanned), "
          f"{len(te_s)} test", flush=True)  # fmt: skip
    root.mkdir(parents=True, exist_ok=True)
    (root / "split.json").write_text(json.dumps({**sp, "args": vars(a)}, indent=2))

    def done(m):
        return (root / m / "metrics_test.json").exists()

    def write(m, model, info, te):
        o = evaluate_and_write(model, te, info, cfg, root / m, ke, tag="test")["overall"]
        print(f"   {m:9s} MAE {o['mae']:.2f}  CRPS {o['crps']:.2f}  cov80 {o['coverage80']:.2f}", flush=True)

    finetuned = None
    if not done("finetune") or (a.fill == "none" and not done("kd_photo")):
        finetuned, _ = train(public, tr_s, va_s, info_s, tcfg, kf)
        if not done("finetune"):
            write("finetune", finetuned, info_s, te_s)
    teacher = None
    if not done("teacher") or not done("kd"):
        wide = reset_teacher(build_model(mcfg, info_t, kt), public, dim)
        teacher, _ = train(wide, tr_t, va_t, info_t, tcfg, kt)
        if not done("teacher"):
            write("teacher", teacher, info_t, te_t)
    spec = dataclasses.replace(AML["kd"][0])
    if not done("kd"):
        m, _ = train_distill(public, teacher, tr_s, va_s, t_by_id, info_s, info_t, tcfg, kk, spec=spec, has_aux=has_aux)
        write("kd", m, info_s, te_s)
    if a.fill == "none" and not done("kd_photo"):
        m, _ = train_distill(public, finetuned, tr_s, va_s, s_by_id, info_s, info_s, tcfg, kp, spec=spec,
                             has_aux=has_aux)  # fmt: skip
        write("kd_photo", m, info_s, te_s)
    if not done("samd"):
        tea_train = [it for it in tr_t if has_aux[it.item_id]]
        m, _ = train_samd(public, build_model(mcfg, info_t, kt), tr_s, va_s, tea_train,
                          {it.item_id: it.item_id for it in tr_s}, t_by_id, info_s, info_t, tcfg, ksm, dim,
                          rounds=a.rounds, k=5, spec=AML["samd"][0], has_aux=has_aux)  # fmt: skip
        write("samd", m, info_s, te_s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["latent_sde", "latent_ode", "transformer"], required=True)
    ap.add_argument("--features", choices=list(LAB_FEATURES), required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument(
        "--fill",
        choices=["none", "linear", "logistic"],
        default="none",
        help="extend the training fruit with regression-extrapolated photos/scans (half-synthetic set)",
    )
    ap.add_argument("--out", default=None, help="default outputs/one_photo_teacher[_fill_<fill>]")
    a = ap.parse_args()
    if a.out is None:
        a.out = "outputs/one_photo_teacher" + ("" if a.fill == "none" else f"_fill_{a.fill}")
    folds = list(range(len(TEST_FRUIT))) if a.folds is None else a.folds
    if len(folds) == 1:
        run_fold(a, folds[0])
        return
    # one process per fold: on CPU, XLA never frees compiled code and the distillation steps are rebuilt per fold
    base = [sys.executable, __file__, "--model", a.model, "--features", a.features, "--seed", str(a.seed), "--lr",
            str(a.lr), "--epochs", str(a.epochs), "--patience", str(a.patience), "--rounds", str(a.rounds), "--out",
            a.out, "--fill", a.fill]  # fmt: skip
    fail = 0
    for k in folds:
        fail |= subprocess.run([*base, "--folds", str(k)], cwd=ROOT).returncode != 0
    sys.exit(fail)


if __name__ == "__main__":
    main()
