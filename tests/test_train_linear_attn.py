# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chunked gated delta (training) against mlx-lm's per-token loop.

    python tests/test_train_linear_attn.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gated_delta import compute_g, gated_delta_ops

from pagedmoe.train import linear_attn

mx.disable_compile()


def inputs(B, T, Hk, Hv, Dk, Dv, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((B, T, Hk, Dk)) * Dk**-0.5
    k = mx.random.normal((B, T, Hk, Dk))
    k = k / mx.linalg.norm(k, axis=-1, keepdims=True)
    q = q * Dk**-0.5
    v = mx.random.normal((B, T, Hv, Dv))
    a = mx.random.normal((B, T, Hv))
    b = mx.random.normal((B, T, Hv))
    A_log = mx.random.uniform(-1, 1, (Hv,))
    dt_bias = mx.random.normal((Hv,)) * 0.5
    return q, k, v, a, b, A_log, dt_bias


def ref(q, k, v, a, b, A_log, dt_bias):
    g = compute_g(A_log, a, dt_bias)
    return gated_delta_ops(q, k, v, g, mx.sigmoid(b), None, None)


def new(q, k, v, a, b, A_log, dt_bias):
    log_g = -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)
    return linear_attn.chunked(q, k, v, log_g, mx.sigmoid(b))


def rel(x, y):
    return (mx.abs(x - y).max() / mx.maximum(mx.abs(x).max(), 1e-8)).item()


def main():
    # CPU float32 shows the math is exact. On the GPU float32 matmuls run at
    # reduced precision (about 1e-3), still under bf16 rounding.
    for dev, tol in ((mx.cpu, 1e-5), (mx.gpu, 5e-3)):
        mx.set_default_device(dev)
        print(dev)
        check(tol)
    mx.set_default_device(mx.gpu)
    tail()


def check(tol):
    for B, T, Hk, Hv, Dk, Dv in [(2, 100, 2, 4, 32, 16), (1, 64, 2, 2, 16, 16), (1, 7, 1, 2, 8, 8), (1, 300, 2, 4, 32, 32)]:
        args = inputs(B, T, Hk, Hv, Dk, Dv)
        y0, s0 = ref(*args)
        y1, s1 = new(*args)
        dy, ds = rel(y0, y1), rel(s0, s1)

        cot = mx.random.normal(y0.shape)

        def lr(*a):
            return (ref(*a)[0] * cot).sum()

        def ln(*a):
            return (new(*a)[0] * cot).sum()

        idx = list(range(7))
        g0 = mx.grad(lr, argnums=idx)(*args)
        g1 = mx.grad(ln, argnums=idx)(*args)
        dg = max(rel(x, y) for x, y in zip(g0, g1))
        print(f"  B{B} T{T} Hk{Hk} Hv{Hv}: y {dy:.1e}  state {ds:.1e}  grads {dg:.1e}")
        assert max(dy, ds, dg) < tol, "chunked gated delta drifted"


def tail():
    # worst case for the in-chunk inverse: every key the same, beta near 1
    q, k, v, a, b, A_log, dt_bias = inputs(1, 256, 2, 2, 32, 16, seed=2)
    k = mx.broadcast_to(k[:, :1], k.shape)
    b = mx.full(b.shape, 6.0)
    a = mx.full(a.shape, -8.0)
    y0, _ = ref(q, k, v, a, b, A_log, dt_bias)
    y1, _ = new(q, k, v, a, b, A_log, dt_bias)
    d = rel(y0, y1)
    print(f"identical keys: y {d:.1e}")
    assert d < 5e-3 and not mx.isnan(y1).any().item()

    kk = mx.random.normal((3, 64, 32))
    kk = kk / mx.linalg.norm(kk, axis=-1, keepdims=True)
    beta = mx.random.uniform(0, 1, (3, 64, 1))
    L = mx.tril(-(beta * kk) @ kk.swapaxes(-1, -2), k=-1)
    W = linear_attn._inv_unit_lower(L, 64)
    err = mx.abs((mx.eye(64) - L) @ W - mx.eye(64)).max().item()
    print(f"inverse residual {err:.1e}")
    assert err < 1e-2

    q, k, v, a, b, A_log, dt_bias = inputs(1, 130, 2, 4, 32, 16, seed=1)
    yb0, _ = ref(q.astype(mx.bfloat16), k.astype(mx.bfloat16), v.astype(mx.bfloat16), a, b, A_log, dt_bias)
    yb1, _ = new(q.astype(mx.bfloat16), k.astype(mx.bfloat16), v.astype(mx.bfloat16), a, b, A_log, dt_bias)
    print(f"bf16 inputs: y {rel(yb0.astype(mx.float32), yb1.astype(mx.float32)):.1e}  dtype {yb1.dtype}")
    assert yb1.dtype == mx.bfloat16

    from mlx_lm.models import qwen3_5, gated_delta
    assert linear_attn.install()
    assert qwen3_5.gated_delta_update is not gated_delta.gated_delta_update
    linear_attn.uninstall()
    assert qwen3_5.gated_delta_update is gated_delta.gated_delta_update

    # model code loaded from a checkpoint file runs in its own namespace,
    # outside sys.modules, so install() has to find it through the model
    ns = {"gated_delta_update": gated_delta.gated_delta_update, "nn": nn}
    exec("class Layer(nn.Module):\n    def __call__(self, x):\n        return x\n", ns)
    model = nn.Module()
    model.layers = [ns["Layer"]()]
    assert linear_attn.unpatched(model) == 1
    assert linear_attn.install(model) and linear_attn.unpatched(model) == 0
    linear_attn.uninstall()
    assert ns["gated_delta_update"] is gated_delta.gated_delta_update
    linear_attn.install()
    assert linear_attn.unpatched(model) == 0
    linear_attn.uninstall()
    print("OK")


if __name__ == "__main__":
    main()
