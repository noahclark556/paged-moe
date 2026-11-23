# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Faster backward passes for frozen quantized weights and causal attention.

The forward of everything here is the same MLX op as before, so outputs do
not change. Only the backward is rewritten:

  quantized matmul   MLX's gradient wrt x runs the quantized kernel with
                     transpose=False, about 4x slower than the forward. The
                     weights are frozen, so the backward dequantizes them once
                     and does a dense matmul instead. Same values to bf16
                     rounding (the kernel dequantizes to the same numbers).
  gather_qmm         same thing for expert matmuls (dense gather_mm with
                     the token sort kept).
  token sort         the MoE sorts tokens by expert. The gradient of those
                     gathers was a scatter-add; here it is a gather and a sum.
  causal attention   rows of LONG_ATTN tokens or more get a chunked backward
                     that only builds the lower triangle of the score matrix,
                     about half the peak memory of the stock backward at the
                     same speed. Shorter rows keep the stock backward.

    PAGED_MOE_TRAIN_FASTGRAD=0   keep MLX's own backward everywhere
"""

from __future__ import annotations

import functools
import os

import mlx.core as mx
import mlx.nn as nn

LONG_ATTN = 4096
ATTN_CHUNK = 256

_saved: dict = {}


def enabled() -> bool:
    return os.environ.get("PAGED_MOE_TRAIN_FASTGRAD", "1") != "0"


def _deq(w, scales, biases, group_size, bits, mode, cot):
    # Tied to the cotangent so the dense copy is built when its gradient
    # arrives, not all at once at the start of the backward.
    w = mx.depends(w, cot)
    return mx.dequantize(w, scales, biases, group_size=group_size, bits=bits, mode=mode).astype(cot.dtype)


@functools.lru_cache(maxsize=None)
def _qmm_fn(group_size: int, bits: int, mode: str, has_biases: bool):
    def fwd(x, w, scales, biases):
        return mx.quantized_matmul(x, w, scales, biases if has_biases else None, transpose=True,
                                   group_size=group_size, bits=bits, mode=mode)

    f = mx.custom_function(fwd)

    @f.vjp
    def _vjp(primals, cot, _out):
        x, w, scales, biases = primals
        W = _deq(w, scales, biases if has_biases else None, group_size, bits, mode, cot)
        dx = (cot @ W).astype(x.dtype)
        return dx, mx.zeros_like(w), mx.zeros_like(scales), mx.zeros_like(biases)

    return f


def qmm(x, w, scales, biases, *, group_size, bits, mode="affine"):
    """quantized_matmul(x, w.T) with a dequantize-and-matmul backward for x."""
    b = biases if biases is not None else scales
    return _qmm_fn(int(group_size), int(bits), str(mode), biases is not None)(x, w, scales, b)


@functools.lru_cache(maxsize=None)
def _gqmm_fn(group_size: int, bits: int, mode: str, has_biases: bool, sort: bool):
    def fwd(x, w, scales, biases, idx):
        return mx.gather_qmm(x, w, scales, biases if has_biases else None, rhs_indices=idx,
                             transpose=True, group_size=group_size, bits=bits, mode=mode,
                             sorted_indices=sort)

    f = mx.custom_function(fwd)

    @f.vjp
    def _vjp(primals, cot, _out):
        x, w, scales, biases, idx = primals
        W = _deq(w, scales, biases if has_biases else None, group_size, bits, mode, cot)
        dx = mx.gather_mm(cot, W, rhs_indices=idx, sorted_indices=sort)
        # x may broadcast over the expert axis (unsorted path: one x row, k experts)
        if dx.shape != x.shape:
            axes = tuple(i for i, (a, b) in enumerate(zip(x.shape, dx.shape)) if a == 1 and b != 1)
            dx = dx.sum(axis=axes, keepdims=True)
        return (dx.astype(x.dtype), mx.zeros_like(w), mx.zeros_like(scales),
                mx.zeros_like(biases), mx.zeros_like(idx))

    return f


def gather_qmm(x, w, scales, biases, idx, *, group_size, bits, mode="affine", sorted_indices=False):
    """gather_qmm(x, w.T)[idx] with a dense gather_mm backward for x."""
    b = biases if biases is not None else scales
    fn = _gqmm_fn(int(group_size), int(bits), str(mode), biases is not None, bool(sorted_indices))
    return fn(x, w, scales, b, idx)


@functools.lru_cache(maxsize=None)
def _gqmm2_fn(specs: tuple, sort: bool):
    """Two expert matmuls on the same x (gate and up) in one function.

    Their gradients arrive together, so as separate ops both dense copies
    would be alive at once. Here the second is built after the first is used.
    """
    (g1, b1, m1, h1), (g2, b2, m2, h2) = specs

    def one(x, w, s, b, idx, gs, bits, mode, has_b):
        return mx.gather_qmm(x, w, s, b if has_b else None, rhs_indices=idx, transpose=True,
                             group_size=gs, bits=bits, mode=mode, sorted_indices=sort)

    def fwd(x, w1, s1, bb1, w2, s2, bb2, idx):
        return one(x, w1, s1, bb1, idx, g1, b1, m1, h1), one(x, w2, s2, bb2, idx, g2, b2, m2, h2)

    f = mx.custom_function(fwd)

    @f.vjp
    def _vjp(primals, cots, _outs):
        x, w1, s1, bb1, w2, s2, bb2, idx = primals
        c1, c2 = cots
        W1 = _deq(w1, s1, bb1 if h1 else None, g1, b1, m1, c1)
        dx = mx.gather_mm(c1, W1, rhs_indices=idx, sorted_indices=sort)
        W2 = _deq(w2, s2, bb2 if h2 else None, g2, b2, m2, mx.depends(c2, dx))
        dx = dx + mx.gather_mm(c2, W2, rhs_indices=idx, sorted_indices=sort)
        if dx.shape != x.shape:
            axes = tuple(i for i, (a, b) in enumerate(zip(x.shape, dx.shape)) if a == 1 and b != 1)
            dx = dx.sum(axis=axes, keepdims=True)
        z = mx.zeros_like
        return (dx.astype(x.dtype), z(w1), z(s1), z(bb1), z(w2), z(s2), z(bb2), z(idx))

    return f


def gather_qmm2(x, a: tuple, b: tuple, idx, *, sorted_indices=False):
    """(gather_qmm(x, a), gather_qmm(x, b)); a and b are
    (weight, scales, biases, group_size, bits, mode)."""
    spec = tuple((int(t[3]), int(t[4]), str(t[5]), t[2] is not None) for t in (a, b))
    fn = _gqmm2_fn(spec, bool(sorted_indices))
    return fn(x, a[0], a[1], a[2] if a[2] is not None else a[1],
              b[0], b[1], b[2] if b[2] is not None else b[1], idx)


@mx.custom_function
def spread(x, order, inv):
    """x[order // M] where every row of x is used M times (M = len(order) / len(x))."""
    return x[order // (order.size // x.shape[0])]


@spread.vjp
def _spread_vjp(primals, cot, _out):
    x, order, inv = primals
    M = order.size // x.shape[0]
    g = cot[inv].reshape((x.shape[0], M) + cot.shape[1:])
    dx = g.astype(mx.float32).sum(axis=1).astype(x.dtype)
    return dx, mx.zeros_like(order), mx.zeros_like(inv)


@mx.custom_function
def unsort(y, inv, order):
    """y[inv], where inv and order are inverse permutations."""
    return y[inv]


@unsort.vjp
def _unsort_vjp(primals, cot, _out):
    y, inv, order = primals
    return cot[order], mx.zeros_like(inv), mx.zeros_like(order)


def _causal_bwd(q, k, v, o, go, scale: float, C: int):
    B, Hq, T, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    q5 = q.reshape(B, Hkv, G, T, D)
    g5 = go.reshape(B, Hkv, G, T, D)
    k5 = k[:, :, None]
    v5 = v[:, :, None]
    delta = (g5.astype(mx.float32) * o.reshape(B, Hkv, G, T, D).astype(mx.float32)).sum(-1, keepdims=True)
    dqs, dk, dv = [], None, None
    for s in range(0, T, C):
        e = min(T, s + C)
        qc, gc = q5[..., s:e, :], g5[..., s:e, :]
        kc, vc = k5[..., :e, :], v5[..., :e, :]
        logits = (qc @ kc.swapaxes(-1, -2)).astype(mx.float32) * scale
        keep = mx.arange(s, e)[:, None] >= mx.arange(e)[None]
        p = mx.softmax(mx.where(keep, logits, -mx.inf), axis=-1)
        dp = (gc @ vc.swapaxes(-1, -2)).astype(mx.float32)
        ds = (p * (dp - delta[..., s:e, :]) * scale).astype(q.dtype)
        dqs.append(ds @ kc)
        pad = [(0, 0)] * 3 + [(0, T - e), (0, 0)]
        dkc = mx.pad((ds.swapaxes(-1, -2) @ qc).astype(mx.float32).sum(2, keepdims=True), pad)
        dvc = mx.pad((p.astype(q.dtype).swapaxes(-1, -2) @ gc).astype(mx.float32).sum(2, keepdims=True), pad)
        dk = dkc if dk is None else dk + dkc
        dv = dvc if dv is None else dv + dvc
    dq = mx.concatenate(dqs, axis=-2).reshape(B, Hq, T, D)
    return dq, dk.reshape(B, Hkv, T, D).astype(k.dtype), dv.reshape(B, Hkv, T, D).astype(v.dtype)


@functools.lru_cache(maxsize=None)
def _attn_fn(scale: float, chunk: int):
    def fwd(q, k, v):
        return _saved["sdpa"](q, k, v, scale=scale, mask="causal")

    f = mx.custom_function(fwd)

    @f.vjp
    def _vjp(primals, cot, out):
        q, k, v = primals
        return _causal_bwd(q, k, v, out, cot, scale, chunk)

    return f


def _sdpa(q, k, v, *args, scale, mask=None, sinks=None, **kw):
    T = q.shape[-2]
    if (
        args or kw or sinks is not None or not isinstance(mask, str) or mask != "causal"
        or T < LONG_ATTN or k.shape[-2] != T or q.ndim != 4 or q.shape[1] % k.shape[1]
    ):
        return _saved["sdpa"](q, k, v, *args, scale=scale, mask=mask, sinks=sinks, **kw)
    return _attn_fn(float(scale), ATTN_CHUNK)(q, k, v)


def _linear_call(self, x):
    y = qmm(x, self["weight"], self["scales"], self.get("biases"),
            group_size=self.group_size, bits=self.bits, mode=getattr(self, "mode", "affine"))
    if "bias" in self:
        y = y + self["bias"]
    return y


def _embed_as_linear(self, x):
    return qmm(x, self["weight"], self["scales"], self.get("biases"),
               group_size=self.group_size, bits=self.bits, mode=getattr(self, "mode", "affine"))


def _switch_call(self, x, indices, sorted_indices=False):
    y = gather_qmm(x, self["weight"], self["scales"], self.get("biases"), indices,
                   group_size=self.group_size, bits=self.bits, mode=getattr(self, "mode", "affine"),
                   sorted_indices=sorted_indices)
    if "bias" in self:
        y = y + mx.expand_dims(self["bias"][indices], -2)
    return y


def _targets():
    out = [(nn.QuantizedLinear, "__call__", _linear_call),
           (nn.QuantizedEmbedding, "as_linear", _embed_as_linear),
           (mx.fast, "scaled_dot_product_attention", _sdpa)]
    try:
        from mlx_lm.models.switch_layers import QuantizedSwitchLinear
        out.append((QuantizedSwitchLinear, "__call__", _switch_call))
    except ImportError:
        pass
    return out


def install() -> bool:
    """Swap in the fast backwards. Returns True if they are active."""
    if not enabled():
        return False
    if _saved:
        return True
    _saved["sdpa"] = mx.fast.scaled_dot_product_attention
    for owner, name, fn in _targets():
        _saved[(owner, name)] = getattr(owner, name)
        setattr(owner, name, fn)
    return True


def uninstall() -> bool:
    """Restore MLX's originals. True if they were installed."""
    if not _saved:
        return False
    for key, orig in list(_saved.items()):
        if isinstance(key, tuple):
            setattr(key[0], key[1], orig)
    _saved.clear()
    return True


def active() -> bool:
    return bool(_saved)
