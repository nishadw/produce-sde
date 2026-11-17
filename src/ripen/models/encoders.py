"""Encoders from a visible visit history to a context vector.

All of them run over the padded visit axis and respect the ``visible`` mask,
so hidden (future) visits never leak in.

* ``gru``: GRU on ``[x_k, dt_k]``; the latent ODE/SDE use it as their posterior context encoder.
* ``transformer``: pre-LN transformer over the visible visits; the last visible token is the context.
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
    hidden: int = eqx.field(static=True)

    def __init__(self, kind: str, in_dim: int, hidden: int, *, key):
        if kind != "gru":
            raise ValueError(f"unknown sequence encoder {kind!r} (only 'gru'; the transformer is TransformerEncoder)")
        self.kind, self.hidden = kind, hidden
        self.cell = eqx.nn.GRUCell(in_dim + 1, hidden, key=key)  # +1: the time gap as an input

    def __call__(self, x: Array, t: Array, visible: Array) -> Array:
        """``x``: ``[V, in_dim]`` per-visit inputs, ``t``: ``[V]``, ``visible``: ``[V]`` bool -> final ``[hidden]``."""
        dt = jnp.where(visible, jnp.diff(t, prepend=t[:1]), 0.0)  # padded visits (t = 0) get a zero gap

        def step(h, inp):
            xk, dtk, vis = inp
            h_new = self.cell(jnp.concatenate([xk, dtk[None]]), h)
            return jnp.where(vis, h_new, h), None

        h0 = jnp.zeros(self.hidden)
        h, _ = jax.lax.scan(step, h0, (x, dt, visible))
        return h


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
    if kind == "transformer":
        return TransformerEncoder(in_dim, hidden, d=kw.get("d_model", 64), heads=kw.get("heads", 4),
                                  layers=kw.get("layers", 2), key=key)  # fmt: skip
    return SequenceEncoder(kind, in_dim, hidden, key=key)
