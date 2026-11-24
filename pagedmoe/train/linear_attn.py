# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chunked gated delta rule for training the qwen3_next / qwen3_5 hybrids,
and the per-channel (Kimi delta) form for glm5_next.

mlx-lm's Metal kernel for gated delta has no gradient, so in train mode it
falls back to gated_delta_ops: a Python loop with one small step per token.
That is fine for a forward pass, but autodiff then walks a chain of T steps
per layer, and on a 3:1 linear/full hybrid it is about 95% of training time.

This is the chunked form of the same recurrence (the one flash-linear-
attention trains with). Inside a chunk of C tokens it is a handful of
matmuls; only the state carried between chunks is sequential, so a 4k
sequence is 32 steps instead of 4096. The math is the same recurrence,
rearranged, so outputs match the loop up to float32 rounding.

While installed, every whole-sequence call (no mask, no incoming state) goes
through it, in train and eval mode alike, so reference and old-policy
logprobs are scored with the same numerics as the training forward. Cached
decode keeps the original path, and rollouts uninstall it for generation.

    PAGED_MOE_TRAIN_CHUNKED_DELTA=0   keep mlx-lm's per-token loop
"""

from __future__ import annotations

import os
import sys

import mlx.core as mx
import mlx.nn as nn

CHUNK = 128  # 64 is ~15% slower to differentiate, same accuracy

_original = None
_patched: list[dict] = []  # namespaces rewritten to _update
_scopes: list[dict] = []  # model code namespaces seen by install(model)


@mx.custom_function
def _inv_grad(L: mx.array) -> mx.array:
    return _inv_unit_lower(L, L.shape[-1])


@_inv_grad.vjp
def _inv_grad_vjp(primals, cot, W):
    # W = (I - L)^-1, so dW = W dL W and dL = W^T dW W^T. Autodiff through
    # the block gathers would scatter at every level instead.
    Wt = W.swapaxes(-1, -2)
    return Wt @ cot @ Wt


def _inv_unit_lower(L: mx.array, n: int) -> mx.array:
    """(I - L)^-1 for strictly lower triangular L [..., n, n], n a power of 2.

    Block recursion: with both diagonal halves inverted (Ai, Di), the
    lower-left block of the inverse is Di @ L21 @ Ai. Doubling the block
    size log2(n) times gives the whole inverse with no large intermediate
    (a power series in L would overflow before it cancels).
    """
    lead = L.shape[:-2]
    inv = mx.ones(lead + (n, 1, 1), dtype=L.dtype)  # 1x1 diagonal blocks
    b = 1
    while b < n:
        P = n // (2 * b)
        pairs = inv.reshape(lead + (P, 2, b, b))
        Ai = pairs[..., 0, :, :]
        Di = pairs[..., 1, :, :]
        # L21 of each pair: rows [2pb + b, 2pb + 2b), cols [2pb, 2pb + b)
        base = mx.arange(P)[:, None] * (2 * b) + mx.arange(b)[None, :]
        r = (base + b)[:, :, None]
        c = base[:, None, :]
        L21 = L[..., r, c]
        LL = Di @ L21 @ Ai
        top = mx.concatenate([Ai, mx.zeros_like(Ai)], axis=-1)
        bot = mx.concatenate([LL, Di], axis=-1)
        inv = mx.concatenate([top, bot], axis=-2)
        b *= 2
    return inv.reshape(lead + (n, n))


def chunked(q, k, v, log_g, beta, chunk: int = CHUNK):
    """Gated delta rule over a whole sequence, zero initial state.

    q, k     [B, T, Hk, Dk]   (already normalized and scaled, as mlx-lm does)
    v        [B, T, Hv, Dv]
    log_g    [B, T, Hv]       log of the per-step decay
    beta     [B, T, Hv]
    returns  y [B, T, Hv, Dv] in q's dtype, final state [B, Hv, Dv, Dk] f32
    """
    out_dtype = q.dtype
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if (r := Hv // Hk) > 1:
        q = mx.repeat(q, r, -2)
        k = mx.repeat(k, r, -2)

    f32 = mx.float32
    # [B, H, T, D]
    q, k, v = (t.astype(f32).transpose(0, 2, 1, 3) for t in (q, k, v))
    beta = beta.astype(f32).transpose(0, 2, 1)
    log_g = log_g.astype(f32).transpose(0, 2, 1)

    C = int(chunk)
    if C & (C - 1):
        raise ValueError("chunk must be a power of 2")
    pad = (C - T % C) % C
    if pad:
        q, k, v = (mx.pad(t, [(0, 0), (0, 0), (0, pad), (0, 0)]) for t in (q, k, v))
        beta = mx.pad(beta, [(0, 0), (0, 0), (0, pad)])
        log_g = mx.pad(log_g, [(0, 0), (0, 0), (0, pad)])
    N = (T + pad) // C
    H = Hv

    q, k, v = (t.reshape(B, H, N, C, -1) for t in (q, k, v))
    beta = beta.reshape(B, H, N, C)
    gc = mx.cumsum(log_g.reshape(B, H, N, C), axis=-1)  # cumulative log decay

    k_beta = k * beta[..., None]
    v_beta = v * beta[..., None]

    lower = mx.tril(mx.ones((C, C), dtype=mx.bool_), k=-1)
    causal = mx.tril(mx.ones((C, C), dtype=mx.bool_), k=0)
    # decay from position j to i inside a chunk, for j <= i
    diff = gc[..., :, None] - gc[..., None, :]
    decay = mx.exp(mx.where(causal, diff, -mx.inf))

    A = mx.where(lower, -(k_beta @ k.swapaxes(-1, -2)) * decay, 0.0)
    W = _inv_grad(A)
    u = W @ v_beta  # [B, H, N, C, Dv]
    w = W @ (k_beta * mx.exp(gc)[..., None])  # [B, H, N, C, Dk]

    qk = mx.where(causal, (q @ k.swapaxes(-1, -2)) * decay, 0.0)
    g_last = gc[..., -1]  # [B, H, N]
    k_tail = k * mx.exp(g_last[..., None] - gc)[..., None]  # decay to chunk end

    S = mx.zeros((B, H, Dk, Dv), dtype=f32)
    ys = []
    for n in range(N):
        v_new = u[:, :, n] - w[:, :, n] @ S
        y = (q[:, :, n] * mx.exp(gc[:, :, n])[..., None]) @ S + qk[:, :, n] @ v_new
        ys.append(y)
        S = S * mx.exp(g_last[:, :, n])[..., None, None] + k_tail[:, :, n].swapaxes(-1, -2) @ v_new

    y = mx.stack(ys, axis=2).reshape(B, H, N * C, Dv)[:, :, :T]
    y = y.transpose(0, 2, 1, 3).astype(out_dtype)
    return y, S.swapaxes(-1, -2)


# Kimi delta attention decays each key channel separately, so the in-chunk
# decay is exp(G_i - G_j) per channel and cannot be pulled out as a scalar.
# It is factored as (k * e^G) @ (k * e^-G)^T instead, which needs e^-G to stay
# finite in fp32: with the gate's lower bound b, a chunk of C steps reaches
# e^(C * |b|). 16 steps at b = -5 is e^80, well inside fp32.
KDA_CHUNK = 16
_KDA_MAX_LOG = 80.0


def chunked_kda(q, k, v, log_g, beta, chunk: int = KDA_CHUNK):
    """Gated delta rule with a per-channel decay, whole sequence, zero state.

    q, k     [B, T, H, Dk]   (already normalized and scaled)
    v        [B, T, H, Dv]
    log_g    [B, T, H, Dk]   log of the per-step, per-channel decay
    beta     [B, T, H]
    returns  y [B, T, H, Dv] in q's dtype, final state [B, H, Dv, Dk] f32
    """
    out_dtype = q.dtype
    B, T, H, Dk = q.shape
    Dv = v.shape[-1]
    f32 = mx.float32
    q, k, v, log_g = (t.astype(f32).transpose(0, 2, 1, 3) for t in (q, k, v, log_g))
    beta = beta.astype(f32).transpose(0, 2, 1)

    C = int(chunk)
    pad = (C - T % C) % C
    if pad:
        q, k, v, log_g = (
            mx.pad(t, [(0, 0), (0, 0), (0, pad), (0, 0)]) for t in (q, k, v, log_g)
        )
        beta = mx.pad(beta, [(0, 0), (0, 0), (0, pad)])
    N = (T + pad) // C

    q, k, v = (t.reshape(B, H, N, C, -1) for t in (q, k, v))
    beta = beta.reshape(B, H, N, C)
    gc = mx.cumsum(log_g.reshape(B, H, N, C, Dk), axis=-2)  # [B, H, N, C, Dk]
    eg = mx.exp(gc)
    # Pair the in-chunk factors around the chunk midpoint: e^(G_i - m) and
    # e^(m - G_j) span half the range of e^G and e^-G, which keeps the GPU's
    # reduced-precision fp32 matmuls accurate.
    mid = mx.stop_gradient(gc[..., C // 2 - 1 : C // 2, :]) if C > 1 else 0.0
    eg_c = mx.exp(gc - mid)
    k_in = k * mx.exp(mid - gc)

    k_beta = k * beta[..., None]
    v_beta = v * beta[..., None]

    lower = mx.tril(mx.ones((C, C), dtype=mx.bool_), k=-1)
    causal = mx.tril(mx.ones((C, C), dtype=mx.bool_), k=0)

    A = mx.where(lower, -((k_beta * eg_c) @ k_in.swapaxes(-1, -2)), 0.0)
    W = _inv_grad(A)
    u = W @ v_beta  # [B, H, N, C, Dv]
    w = W @ (k_beta * eg)  # [B, H, N, C, Dk]

    qe = q * eg
    qk = mx.where(causal, (q * eg_c) @ k_in.swapaxes(-1, -2), 0.0)
    g_last = gc[..., -1:, :]  # [B, H, N, 1, Dk]
    k_tail = k * mx.exp(g_last - gc)  # decay to chunk end

    S = mx.zeros((B, H, Dk, Dv), dtype=f32)
    ys = []
    for n in range(N):
        v_new = u[:, :, n] - w[:, :, n] @ S
        ys.append(qe[:, :, n] @ S + qk[:, :, n] @ v_new)
        S = S * mx.exp(g_last[:, :, n, 0])[..., None] + k_tail[:, :, n].swapaxes(-1, -2) @ v_new

    y = mx.stack(ys, axis=2).reshape(B, H, N * C, Dv)[:, :, :T]
    return y.transpose(0, 2, 1, 3).astype(out_dtype), S.swapaxes(-1, -2)


_kda_orig: dict[int, object] = {}


def _kda_update_for(orig):
    def update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True,
               lower_bound=None):
        # Unbounded gates (no safe lower bound) could overflow e^-G; keep those
        # on the model's own path, as are cached and masked calls.
        if mask is not None or state is not None or lower_bound is None:
            return orig(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel, lower_bound)
        if abs(float(lower_bound)) * KDA_CHUNK > _KDA_MAX_LOG:
            return orig(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel, lower_bound)
        beta = mx.sigmoid(b)
        log_g = lower_bound * mx.sigmoid(
            mx.exp(A_log.astype(mx.float32)) * (a.astype(mx.float32) + dt_bias.astype(mx.float32))
        )
        return chunked_kda(q, k, v, log_g, beta)

    update.kda_chunked = True
    return update


def _update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
    if mask is not None or state is not None:
        return _original(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel)
    beta = mx.sigmoid(b)
    log_g = -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)
    return chunked(q, k, v, log_g, beta)


def model_scopes(model) -> list[dict]:
    """Globals of every module class in `model` that calls gated_delta_update.

    A model file loaded from the checkpoint is not always the module in
    sys.modules, so the layers' own namespaces are the ones that count.
    """
    out: list[dict] = []
    for _name, m in model.named_modules():
        g = getattr(type(m).__call__, "__globals__", None)
        if isinstance(g, dict) and "gated_delta_update" in g and all(g is not s for s in out):
            out.append(g)
    return out


def _patch(ns: dict) -> None:
    if any(ns is p for p in _patched):
        return
    fn = ns.get("gated_delta_update")
    if fn is _original:
        ns["gated_delta_update"] = _update
        _patched.append(ns)
    elif getattr(fn, "kda", False):
        _kda_orig[id(ns)] = fn
        ns["gated_delta_update"] = _kda_update_for(fn)
        _patched.append(ns)


def _is_chunked(fn) -> bool:
    return fn is _update or getattr(fn, "kda_chunked", False)


def install(model=None) -> bool:
    """Route training-mode gated delta through the chunked form.

    With `model`, its layers' namespaces are patched too and remembered, so
    a later install() (after a rollout) covers them again.
    """
    global _original
    if os.environ.get("PAGED_MOE_TRAIN_CHUNKED_DELTA", "1") == "0":
        return False
    try:
        from mlx_lm.models import gated_delta
    except ImportError:
        return False
    if _original is None:
        _original = gated_delta.gated_delta_update
    for name, mod in list(sys.modules.items()):
        if not name.startswith("mlx_lm.models.") or mod is None or mod is gated_delta:
            continue
        _patch(mod.__dict__)
    if model is not None:
        for ns in model_scopes(model):
            if all(ns is not s for s in _scopes):
                _scopes.append(ns)
    for ns in _scopes:
        _patch(ns)
    return True


def unpatched(model) -> int:
    """How many of the model's gated delta namespaces still use the loop."""
    return sum(1 for ns in model_scopes(model) if not _is_chunked(ns.get("gated_delta_update")))


def uninstall() -> bool:
    """Restore the original functions. True if it was installed."""
    was = bool(_patched)
    for ns in _patched:
        fn = ns.get("gated_delta_update")
        if fn is _update:
            ns["gated_delta_update"] = _original
        elif getattr(fn, "kda_chunked", False) and id(ns) in _kda_orig:
            ns["gated_delta_update"] = _kda_orig.pop(id(ns))
    _patched.clear()
    return was
