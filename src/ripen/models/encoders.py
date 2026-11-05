"""Encoders from a visible visit history to a context vector.

All three run over the padded visit axis with ``lax.scan`` and respect the
``visible`` mask, so hidden (future) visits never leak in.

* ``gru``: GRU on ``[x_k, dt_k]`` — the "GRU with time-gap features" baseline.
* ``gru_d``: the hidden state decays towards zero with the gap before each
  update, ``h <- h * exp(-softplus(w) * dt)`` (Che et al. 2016, hidden-state
  decay; input decay is unneeded because every visit has all its inputs).
* ``ode_rnn``: the hidden state follows ``dh/dt = f(h)`` between visits
  (fixed-step RK4) and a GRU update at each visit (Rubanova et al. 2019).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..data.schema import N_STAGES


class VisitEncoder(eqx.Module):
    """Per-visit input: ``[features (if used), one-hot stage (if used)] -> MLP -> x``."""

    net: eqx.nn.MLP
    use_images: bool = eqx.field(static=True)
    use_stages: bool = eqx.field(static=True)
    feat_dim: int = eqx.field(static=True)

    def __init__(self, feat_dim: int, out_dim: int, use_images: bool, use_stages: bool, width: int = 64, *, key):
        if not (use_images or use_stages):
            raise ValueError("a visit encoder needs images or stages")
        self.use_images, self.use_stages, self.feat_dim = use_images, use_stages, feat_dim
        in_dim = (feat_dim if use_images else 0) + (N_STAGES + 1 if use_stages else 0)
        self.net = eqx.nn.MLP(in_dim, out_dim, width, 1, activation=jax.nn.gelu, key=key)

    def __call__(self, feat: Array, stage: Array) -> Array:
        """``feat``: ``[D]``, ``stage``: scalar int (0 = missing). Returns ``[out_dim]``."""
        parts = []
        if self.use_images:
            parts.append(feat)
        if self.use_stages:
            parts.append(jax.nn.one_hot(stage, N_STAGES + 1))
        return self.net(jnp.concatenate(parts))


class SequenceEncoder(eqx.Module):
    kind: str = eqx.field(static=True)
    cell: eqx.nn.GRUCell
    decay: Array | None
    ode_net: eqx.nn.MLP | None
    hidden: int = eqx.field(static=True)
    ode_substeps: int = eqx.field(static=True)

    def __init__(self, kind: str, in_dim: int, hidden: int, *, key, ode_substeps: int = 4):
        if kind not in ("gru", "gru_d", "ode_rnn"):
            raise ValueError(kind)
        k1, k2 = jax.random.split(key)
        self.kind, self.hidden, self.ode_substeps = kind, hidden, ode_substeps
        self.cell = eqx.nn.GRUCell(in_dim + 1, hidden, key=k1)  # +1: the time gap as an input
        self.decay = jnp.zeros(hidden, jnp.float32) if kind == "gru_d" else None
        self.ode_net = (
            eqx.nn.MLP(hidden, hidden, hidden, 1, activation=jax.nn.tanh, key=k2) if kind == "ode_rnn" else None
        )

    def _evolve(self, h: Array, dt: Array) -> Array:
        if self.kind == "gru_d":
            return h * jnp.exp(-jax.nn.softplus(self.decay) * dt)
        if self.kind == "ode_rnn":
            step = dt / self.ode_substeps

            def rk4(h, _):
                k1 = self.ode_net(h)
                k2 = self.ode_net(h + 0.5 * step * k1)
                k3 = self.ode_net(h + 0.5 * step * k2)
                k4 = self.ode_net(h + step * k3)
                return h + step / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), None

            h, _ = jax.lax.scan(rk4, h, None, length=self.ode_substeps)
            return h
        return h

    def __call__(self, x: Array, t: Array, visible: Array) -> Array:
        """``x``: ``[V, in_dim]`` per-visit inputs, ``t``: ``[V]``, ``visible``: ``[V]`` bool -> final ``[hidden]``."""
        # padded visits have t = 0, so their raw gap is -t_last; on long time scales (months) the GRU-D decay
        # exp(+...) overflows there and the masked-out inf still turns the gradient into NaN
        dt = jnp.where(visible, jnp.diff(t, prepend=t[:1]), 0.0)

        def step(h, inp):
            xk, dtk, vis = inp
            h_evolved = self._evolve(h, dtk)
            h_new = self.cell(jnp.concatenate([xk, dtk[None]]), h_evolved)
            return jnp.where(vis, h_new, h), None

        h0 = jnp.zeros(self.hidden)
        h, _ = jax.lax.scan(step, h0, (x, dt, visible))
        return h


class CDEEncoder(eqx.Module):
    """Neural CDE (Kidger et al. 2020) driven by the piecewise-linear interpolation of ``[x_k, t_k]``.

    On each interval the control moves by ``dX = [x_{k+1} - x_k, dt]``, so ``dz = f(z) dX`` integrates in the
    interval's own parameter ``s in [0, 1]`` as ``dz/ds = f(z) @ dX`` (fixed-step RK4). That is the exact CDE
    solution for a linear control, independent of the gap length, and it skips masked visits.
    """

    init: eqx.nn.Linear
    field: eqx.nn.MLP
    hidden: int = eqx.field(static=True)
    in_dim: int = eqx.field(static=True)
    substeps: int = eqx.field(static=True)

    def __init__(self, in_dim: int, hidden: int, width: int = 64, substeps: int = 4, *, key):
        k1, k2 = jax.random.split(key)
        self.in_dim, self.hidden, self.substeps = in_dim + 1, hidden, substeps
        self.init = eqx.nn.Linear(in_dim + 1, hidden, key=k1)
        self.field = eqx.nn.MLP(hidden, hidden * (in_dim + 1), width, 1, activation=jax.nn.softplus,
                                final_activation=jax.nn.tanh, key=k2)  # fmt: skip

    def _f(self, z: Array, dX: Array) -> Array:
        return self.field(z).reshape(self.hidden, self.in_dim) @ dX

    def __call__(self, x: Array, t: Array, visible: Array) -> Array:
        X = jnp.concatenate([x, (t / 10.0)[:, None]], -1)  # [V, in_dim + 1]
        z0 = self.init(X[0])
        h = 1.0 / self.substeps

        def step(carry, inp):
            z, X_prev = carry
            Xk, vis = inp
            dX = Xk - X_prev

            def rk4(z, _):
                k1 = self._f(z, dX)
                k2 = self._f(z + 0.5 * h * k1, dX)
                k3 = self._f(z + 0.5 * h * k2, dX)
                k4 = self._f(z + h * k3, dX)
                return z + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), None

            z_new, _ = jax.lax.scan(rk4, z, None, length=self.substeps)
            return (jnp.where(vis, z_new, z), jnp.where(vis, Xk, X_prev)), None

        (z, _), _ = jax.lax.scan(step, (z0, X[0]), (X[1:], visible[1:]))
        return z


def time_encoding(t: Array, dim: int, max_period: float = 100.0) -> Array:
    """Sinusoidal encoding of a scalar time, ``[..., dim]``."""
    freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(dim // 2) / max(dim // 2, 1))
    ang = t[..., None] * freqs
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], -1)


class _Block(eqx.Module):
    ln1: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    ln2: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP

    def __init__(self, d: int, heads: int, *, key):
        k1, k2 = jax.random.split(key)
        self.ln1, self.ln2 = eqx.nn.LayerNorm(d), eqx.nn.LayerNorm(d)
        self.attn = eqx.nn.MultiheadAttention(heads, d, key=k1)
        self.mlp = eqx.nn.MLP(d, d, 2 * d, 1, activation=jax.nn.gelu, key=k2)

    def __call__(self, h: Array, mask: Array) -> Array:
        a = jax.vmap(self.ln1)(h)
        h = h + self.attn(a, a, a, mask=mask)
        return h + jax.vmap(self.mlp)(jax.vmap(self.ln2)(h))


class TransformerEncoder(eqx.Module):
    """Pre-LN transformer over the visible visits. Each token is ``x_k`` plus encodings of the visit time ``t_k``
    and the gap since the previous visit; padded and hidden visits are masked out of the attention. The output is
    the last visible token, so the context is "now"."""

    proj: eqx.nn.Linear
    blocks: list
    ln: eqx.nn.LayerNorm
    out: eqx.nn.Linear
    d: int = eqx.field(static=True)

    def __init__(self, in_dim: int, hidden: int, d: int = 64, heads: int = 4, layers: int = 2, *, key):
        ks = jax.random.split(key, layers + 2)
        self.d = d
        self.proj = eqx.nn.Linear(in_dim, d, key=ks[0])
        self.blocks = [_Block(d, heads, key=k) for k in ks[1:-1]]
        self.ln = eqx.nn.LayerNorm(d)
        self.out = eqx.nn.Linear(d, hidden, key=ks[-1])

    def __call__(self, x: Array, t: Array, visible: Array) -> Array:
        gap = jnp.where(visible, jnp.diff(t, prepend=t[:1]), 0.0)
        h = jax.vmap(self.proj)(x) + 0.5 * (time_encoding(t, self.d) + time_encoding(gap, self.d, 20.0))
        mask = visible[None, :] & visible[:, None] | jnp.eye(visible.shape[0], dtype=bool)  # no all-masked rows
        for blk in self.blocks:
            h = blk(h, mask)
        last = jnp.clip(jnp.sum(visible) - 1, 0, None)
        return self.out(self.ln(h[last]))


def make_sequence_encoder(kind: str, in_dim: int, hidden: int, *, key, **kw):
    if kind == "cde":
        return CDEEncoder(in_dim, hidden, substeps=kw.get("substeps", 4), key=key)
    if kind == "transformer":
        return TransformerEncoder(in_dim, hidden, d=kw.get("d_model", 64), heads=kw.get("heads", 4),
                                  layers=kw.get("layers", 2), key=key)  # fmt: skip
    return SequenceEncoder(kind, in_dim, hidden, key=key)
