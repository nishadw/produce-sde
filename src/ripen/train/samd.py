"""SAMD: auxiliary-modality distillation with supermodel resets (Shen, Wang, Gao & Lin, ICML 2023, App. A.7).

A student sees the main modality ``I_M`` (photo); a teacher of the same class sees ``{I_M, I_A}`` (photo + microwave).
The teacher is a *supermodel* of the student: its input layer has extra columns for ``I_A``, so zeroing them makes it
compute exactly the student's function. A learner is a feature network ``F`` (here the visit/sequence encoders) and a
predictor ``D`` (everything after them), and ``D`` is shared. Training runs ``t`` rounds:

    reset teacher with student        (copy every weight; the I_A input columns set to 0)
    k epochs, teacher:   theta_tea <- theta_tea - eta grad L_A(D(F_tea(I_M, I_A)), GT)              (Eq. 2)
    D_student <- D_teacher   (the predictor is shared)
    k epochs, student:   L_M = L_sup(D(F_stu(I_M)), GT) + beta L_con(F_stu(I_M), F_tea(I_M, I_A))      (Eq. 1, 3)

``L_con`` is the mean squared distance between the student's and the (frozen) teacher's features on the same fruit
at the same moment. The paper uses SGD at lr 1e-3, ``k = 5``; ``t`` until the student converges. Here the student is
early-stopped on its validation loss across rounds.

Every model in ``ripen.models`` with a ``visit_enc`` qualifies; its features are what the predictor reads:
``context`` (sequence models), the posterior context (latent SDE/ODE) or the visit embedding (FPT filter).
"""

from __future__ import annotations

import dataclasses
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..data.collate import DatasetInfo, collate
from ..data.schema import Item
from .loop import TrainConfig, evaluate_loss

FEATURE_FIELDS = ("visit_enc", "seq_enc")


def features(model, batch) -> jax.Array:
    """``[B, d]`` the representation the predictor ``D`` reads."""
    if hasattr(model, "context"):
        return model.context(batch)
    if hasattr(model, "encode"):
        return model.encode(batch)[0]
    if hasattr(model, "visit_enc"):  # FPT filter: embedding of the last visible visit
        idx = batch.n_obs - 1
        f = batch.feat[jnp.arange(batch.batch_size), idx]
        return jax.vmap(model.visit_enc)(f, jnp.zeros(batch.batch_size, jnp.int32))
    raise TypeError(f"{type(model).__name__} has no feature network for SAMD")


def _arrays(m):
    return jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_array))


def reset_teacher(teacher, student, n_main: int):
    """Teacher weights := student weights, with the auxiliary input columns zeroed.

    Input layers differ only in their last axis: the teacher's features are ``[main (n_main) | aux (n_extra)]`` where
    the student's are ``[main]``; anything after the features in the same layer (e.g. a context vector) keeps its
    relative position. Every other array must have the same shape in both.
    """
    t_par, t_static = eqx.partition(teacher, eqx.is_array)
    t_leaves, treedef = jax.tree_util.tree_flatten(t_par)
    s_leaves = _arrays(student)
    if len(t_leaves) != len(s_leaves):
        raise ValueError("teacher and student must be the same model class and configuration")
    out = []
    for t, s in zip(t_leaves, s_leaves, strict=True):
        if t.shape == s.shape:
            out.append(s)
            continue
        extra = t.shape[-1] - s.shape[-1]
        if t.shape[:-1] != s.shape[:-1] or extra <= 0:
            raise ValueError(f"cannot reset a {t.shape} teacher weight from a {s.shape} student weight")
        z = jnp.zeros(s.shape[:-1] + (extra,), s.dtype)
        out.append(jnp.concatenate([s[..., :n_main], z, s[..., n_main:]], axis=-1))
    return eqx.combine(jax.tree_util.tree_unflatten(treedef, out), t_static)


def share_predictor(student, teacher):
    """Copy every non-feature, non-static field (the predictor ``D``) from teacher to student."""
    names = [
        f.name
        for f in dataclasses.fields(student)
        if f.name not in FEATURE_FIELDS and not f.metadata.get("static", False) and getattr(student, f.name) is not None
    ]
    return eqx.tree_at(lambda m: [getattr(m, n) for n in names], student, [getattr(teacher, n) for n in names])


def train_samd(
    student,
    teacher,
    stu_train: list[Item],
    stu_val: list[Item],
    tea_train: list[Item],
    pair: dict[str, str],
    tea_by_id: dict[str, Item],
    info_s: DatasetInfo,
    info_t: DatasetInfo,
    cfg: TrainConfig,
    key,
    n_main: int,
    rounds: int = 12,
    k: int = 5,
    beta: float = 1.0,
    log=print,
):
    """Returns ``(best_student, history)``. ``pair[student_item_id] -> teacher_item_id`` (same fruit, same moment)."""
    rng = np.random.default_rng(cfg.seed)
    opt = optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adamw(cfg.lr, weight_decay=cfg.weight_decay))
    s_state = opt.init(eqx.filter(student, eqx.is_inexact_array))

    @eqx.filter_jit
    def t_step(m, st, batch, kk):
        loss, g = eqx.filter_value_and_grad(lambda mm: mm.loss(batch, kk))(m)
        upd, st = opt.update(g, st, eqx.filter(m, eqx.is_inexact_array))
        return eqx.apply_updates(m, upd), st, loss

    @eqx.filter_jit
    def s_step(m, st, tea, sb, tb, kk):
        f_t = jax.lax.stop_gradient(features(tea, tb))

        def L(mm):
            return mm.loss(sb, kk) + beta * jnp.mean((features(mm, sb) - f_t) ** 2)

        loss, g = eqx.filter_value_and_grad(L)(m)
        upd, st = opt.update(g, st, eqx.filter(m, eqx.is_inexact_array))
        return eqx.apply_updates(m, upd), st, loss

    def batches(items, info, shuffle=True):
        order = np.arange(len(items))
        if shuffle:
            rng.shuffle(order)
        for i in range(0, len(items), cfg.batch_size):
            yield [items[j] for j in order[i : i + cfg.batch_size]]

    best, best_val, bad, hist, t0 = student, float("inf"), 0, [], time.time()
    for r in range(rounds):
        teacher = reset_teacher(teacher, student, n_main)
        t_state = opt.init(eqx.filter(teacher, eqx.is_inexact_array))
        for _ in range(k):
            for chunk in batches(tea_train, info_t):
                key, sub = jax.random.split(key)
                teacher, t_state, _ = t_step(teacher, t_state, collate(chunk, info_t, n_obs="random", rng=rng), sub)
        student = share_predictor(student, teacher)
        for e in range(k):
            tot, n = 0.0, 0
            for chunk in batches(stu_train, info_s):
                sb = collate(chunk, info_s, n_obs="all")
                tb = collate([tea_by_id[pair[it.item_id]] for it in chunk], info_t, n_obs="all")
                key, sub = jax.random.split(key)
                student, s_state, loss = s_step(student, s_state, teacher, sb, tb, sub)
                if not bool(jnp.isfinite(loss)):
                    raise FloatingPointError(f"non-finite SAMD loss in round {r}")
                tot += float(loss) * len(chunk)
                n += len(chunk)
            key, sub = jax.random.split(key)
            val = evaluate_loss(student, stu_val, info_s, cfg.batch_size, sub, "all") if stu_val else float("nan")
            hist.append({"round": r, "epoch": r * k + e, "train_loss": tot / max(n, 1), "val_loss": val,
                         "seconds": time.time() - t0})  # fmt: skip
            if val < best_val:
                best, best_val, bad = student, val, 0
            else:
                bad += 1
        log(f"round {r:2d}  student train {hist[-1]['train_loss']:.4f}  val {hist[-1]['val_loss']:.4f}")
        if bad >= cfg.patience:
            log(f"early stop after round {r} (best val {best_val:.4f})")
            break
    return best, hist
