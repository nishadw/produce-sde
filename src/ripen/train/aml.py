"""Auxiliary-modality learning: photo-only students taught by a photo + microwave teacher.

The teacher and the student see the same fruit and the same visible prefix of visits. The student objective is

    gamma L_task + alpha t^2 KL(q_teacher || q_student) + beta L_rep(F_student, F_teacher)

with ``q`` the forecast distribution over days-until-spoiled softened by temperature ``t`` and ``F`` the
representation the predictor reads (``ripen.train.samd.features``). ``L_rep`` is one of

* ``mse``: feature matching (FitNet / SAMD's consistency term);
* ``rkd``: relational KD (Park et al., CVPR 2019): pairwise distances (weight 25) and triplet angles (weight 50);
* ``pkt``: probabilistic knowledge transfer (Passalis & Tefas, ECCV 2018): KL between cosine-kernel neighbour
  distributions;
* ``sp``: similarity preserving (Tung & Mori, ICCV 2019): Frobenius distance of the row-normalised batch Gram matrices.

These are the RepDistiller losses (Tian et al., ICLR 2020) the lab's steering code uses, rewritten for a ``[B, d]``
representation. A per-item weight ``w`` restricts the teacher terms to fruit that actually have the auxiliary
modality (``paired_only``; the lab's "partial" setting, without its mismatched unpaired-batch KD term).

``train_distill`` keeps the teacher frozen (RepDistiller ``train_partial``); the reset-every-round variant (SAMD, the
lab's ``train_ours_partial``) is ``ripen.train.samd.train_samd``, which takes the same ``DistillSpec``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..data.collate import DatasetInfo, choose_n_obs, collate
from ..data.schema import Item
from .loop import TrainConfig, evaluate_loss
from .samd import features

REPS = ("mse", "rkd", "pkt", "sp")


@dataclass(frozen=True)
class DistillSpec:
    gamma: float = 1.0  # task loss
    alpha: float = 0.0  # forecast KD
    temp: float = 4.0
    beta: float = 0.0  # representation loss
    rep: str = "mse"

    def __post_init__(self):
        if self.rep not in REPS:
            raise ValueError(f"rep must be one of {REPS}")


# method name -> (spec, reset teacher every round)
METHODS: dict[str, tuple[DistillSpec, bool]] = {
    "kd": (DistillSpec(gamma=0.1, alpha=0.9, temp=4.0), False),  # CAML / RepDistiller kd
    "feat": (DistillSpec(beta=1.0), False),  # FitNet-style
    "kd_feat": (DistillSpec(gamma=0.1, alpha=0.9, temp=4.0, beta=1.0), False),
    "rkd": (DistillSpec(beta=1.0, rep="rkd"), False),
    "pkt": (DistillSpec(beta=1.0, rep="pkt"), False),
    "sp": (DistillSpec(beta=1.0, rep="sp"), False),
    "samd": (DistillSpec(beta=1.0), True),  # SAMD: reset + shared head + feature MSE
    "reset_only": (DistillSpec(), True),  # control: reset schedule + shared head, no distillation term
    "ours": (DistillSpec(gamma=0.1, alpha=0.9, temp=4.0), True),  # lab run_our_steering_distill*.sh: reset + kd
    "ours_rkd": (DistillSpec(beta=1.0, rep="rkd"), True),
    "ours_pkt": (DistillSpec(beta=1.0, rep="pkt"), True),
    "ours_sp": (DistillSpec(beta=1.0, rep="sp"), True),
}


def forecast_log_pmf(F: jax.Array) -> jax.Array:
    """``[B, H+1]`` CDF over days 0..H from now -> ``[B, H+2]`` log-probabilities (last bin: later than H)."""
    p = jnp.diff(F, axis=-1, prepend=jnp.zeros_like(F[..., :1]))
    p = jnp.concatenate([p, 1.0 - F[..., -1:]], axis=-1)
    return jnp.log(jnp.clip(p, 1e-6, 1.0))


def _wmean(x: jax.Array, w: jax.Array | None) -> jax.Array:
    if w is None:
        return jnp.mean(x)
    return jnp.sum(x * w) / jnp.maximum(jnp.sum(w), 1.0)


def kd_loss(F_student: jax.Array, F_teacher: jax.Array, temp: float, w: jax.Array | None = None) -> jax.Array:
    """``t^2 KL(q_T || q_S)`` with ``q = softmax(log p / t)``, (weighted) mean over the batch."""
    ls = jax.nn.log_softmax(forecast_log_pmf(F_student) / temp, axis=-1)
    lt = jax.nn.log_softmax(forecast_log_pmf(F_teacher) / temp, axis=-1)
    return temp**2 * _wmean(jnp.sum(jnp.exp(lt) * (lt - ls), axis=-1), w)


def _huber(x):
    a = jnp.abs(x)
    return jnp.where(a < 1.0, 0.5 * x**2, a - 0.5)


def _norm(x):
    """Euclidean norm over the last axis with a finite gradient at 0 (jnp.linalg.norm's is NaN there)."""
    return jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1e-12)


def _pdist(f):
    d2 = jnp.sum(f**2, -1)[:, None] + jnp.sum(f**2, -1)[None, :] - 2 * f @ f.T
    return jnp.sqrt(jnp.maximum(d2, 1e-12))


def rep_loss(kind: str, fs: jax.Array, ft: jax.Array, w: jax.Array | None = None) -> jax.Array:
    """Representation loss between ``[B, d]`` student and teacher features; ``w [B]`` keeps only weighted items."""
    B = fs.shape[0]
    w = jnp.ones(B) if w is None else w
    if kind == "mse":
        return _wmean(jnp.mean((fs - ft) ** 2, -1), w)
    W = w[:, None] * w[None, :] * (1.0 - jnp.eye(B))  # pairs of distinct weighted items
    if kind == "rkd":
        ds, dt = _pdist(fs), _pdist(ft)
        ds = ds / (jnp.sum(ds * W) / jnp.maximum(jnp.sum(W), 1.0) + 1e-6)
        dt = dt / (jnp.sum(dt * W) / jnp.maximum(jnp.sum(W), 1.0) + 1e-6)
        dist = jnp.sum(_huber(ds - dt) * W) / jnp.maximum(jnp.sum(W), 1.0)

        def angles(f):
            e = f[None, :, :] - f[:, None, :]  # e[i, j] = f_j - f_i
            e = e / (_norm(e) + 1e-6)
            return jnp.einsum("ijd,ikd->ijk", e, e)

        W3 = W[:, :, None] * W[:, None, :]
        ang = jnp.sum(_huber(angles(fs) - angles(ft)) * W3) / jnp.maximum(jnp.sum(W3), 1.0)
        return 25.0 * dist + 50.0 * ang
    if kind == "pkt":

        def cond(f):
            f = f / (_norm(f) + 1e-6)
            k = (f @ f.T + 1.0) / 2.0 * W
            return k / (jnp.sum(k, -1, keepdims=True) + 1e-6)

        ps, pt = cond(fs), cond(ft)
        kl = jnp.sum(pt * (jnp.log(pt + 1e-6) - jnp.log(ps + 1e-6)), -1)
        return _wmean(kl, w)
    if kind == "sp":

        def gram(f):
            g = (f @ f.T) * W
            return g / (_norm(g) + 1e-6)

        n = jnp.maximum(jnp.sum(w), 1.0)
        return jnp.sum((gram(fs) - gram(ft)) ** 2) / n**2
    raise ValueError(kind)


def teacher_targets(spec: DistillSpec, teacher, tb, key):
    """Frozen teacher outputs the objective needs: (forecast CDF or None, features or None)."""
    F_t = jax.lax.stop_gradient(teacher.predict_cdf(tb, key)) if spec.alpha > 0 else None
    f_t = jax.lax.stop_gradient(features(teacher, tb)) if spec.beta > 0 else None
    return F_t, f_t


def student_objective(spec: DistillSpec, mm, sb, F_t, f_t, key, w=None):
    """-> (objective, task loss)."""
    k1, k2 = jax.random.split(key)
    task = mm.loss(sb, k1)
    out = spec.gamma * task
    if spec.alpha > 0:
        out = out + spec.alpha * kd_loss(mm.predict_cdf(sb, k2), F_t, spec.temp, w)
    if spec.beta > 0:
        out = out + spec.beta * rep_loss(spec.rep, features(mm, sb), f_t, w)
    return out, task


def paired_batches(items: list[Item], bs: int, rng: np.random.Generator, n_obs_mode: str, shuffle: bool = True):
    """Chunks of items with one visible-prefix length per item, shared by student and teacher."""
    order = np.arange(len(items))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(items), bs):
        chunk = [items[j] for j in order[i : i + bs]]
        yield chunk, [choose_n_obs(it, n_obs_mode, rng) for it in chunk]


def train_distill(
    student,
    teacher,
    stu_train: list[Item],
    stu_val: list[Item],
    tea_by_id: dict[str, Item],
    info_s: DatasetInfo,
    info_t: DatasetInfo,
    cfg: TrainConfig,
    key,
    spec: DistillSpec,
    has_aux: dict[str, bool] | None = None,
    log=print,
):
    """Frozen teacher. Returns ``(best_student, history)``; early stopping on the student's own validation loss.
    ``has_aux[item_id]`` (paired-only mode) restricts the teacher terms to fruit with the auxiliary modality."""
    rng = np.random.default_rng(cfg.seed)
    opt = optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adamw(cfg.lr, weight_decay=cfg.weight_decay))
    state = opt.init(eqx.filter(student, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(m, st, tea, sb, tb, w, kk):
        k1, k2 = jax.random.split(kk)
        F_t, f_t = teacher_targets(spec, tea, tb, k1)
        (loss, task), g = eqx.filter_value_and_grad(
            lambda mm: student_objective(spec, mm, sb, F_t, f_t, k2, w), has_aux=True
        )(m)
        upd, st = opt.update(g, st, eqx.filter(m, eqx.is_inexact_array))
        return eqx.apply_updates(m, upd), st, loss, task

    best, best_val, bad, hist, t0, n_steps = student, float("inf"), 0, [], time.time(), 0
    for epoch in range(cfg.epochs):
        tot, tot_task, n = 0.0, 0.0, 0
        for chunk, n_obs in paired_batches(stu_train, cfg.batch_size, rng, cfg.n_obs_mode):
            sb = collate(chunk, info_s, n_obs=n_obs)
            tb = collate([tea_by_id[it.item_id] for it in chunk], info_t, n_obs=n_obs)
            w = None if has_aux is None else jnp.asarray([float(has_aux[it.item_id]) for it in chunk])
            key, sub = jax.random.split(key)
            student, state, loss, task = step(student, state, teacher, sb, tb, w, sub)
            if not bool(jnp.isfinite(loss)):
                raise FloatingPointError(f"non-finite distillation loss at epoch {epoch}")
            tot += float(loss) * len(chunk)
            tot_task += float(task) * len(chunk)
            n += len(chunk)
            n_steps += 1
            if cfg.max_steps is not None and n_steps >= cfg.max_steps:
                break
        key, sub = jax.random.split(key)
        val = evaluate_loss(student, stu_val, info_s, cfg.batch_size, sub, cfg.n_obs_mode) if stu_val else float("nan")
        rec = {"epoch": epoch, "train_loss": tot / max(n, 1), "train_task": tot_task / max(n, 1), "val_loss": val,
               "seconds": time.time() - t0}  # fmt: skip
        hist.append(rec)
        if epoch % cfg.log_every == 0:
            log(f"epoch {epoch:3d}  train {rec['train_loss']:.4f} (task {rec['train_task']:.4f})  val {val:.4f}")
        if val < best_val or not stu_val:
            best, best_val, bad = student, val, 0
        else:
            bad += 1
            if bad >= cfg.patience:
                log(f"early stop at epoch {epoch} (best val {best_val:.4f})")
                break
        if cfg.max_steps is not None and n_steps >= cfg.max_steps:
            break
    return best, hist
