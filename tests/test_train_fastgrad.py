# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fast backwards give MLX's forward exactly and its gradient to rounding.

    python tests/test_train_fastgrad.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import mlx.nn as nn

from pagedmoe.train import fastgrad

BF = mx.bfloat16


def rel(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return (mx.abs(a - b).max() / (mx.abs(b).max() + 1e-12)).item()


def qweights(shape, bits=4, gs=64, mode="affine"):
    w = mx.random.normal(shape) * 0.05
    return mx.quantize(w.astype(BF), group_size=gs, bits=bits, mode=mode)


def check_qmm():
    for bits, gs in ((4, 64), (8, 32)):
        wq, s, b = qweights((96, 128), bits, gs)
        x = mx.random.normal((3, 7, 128)).astype(BF)
        ref = lambda a: mx.quantized_matmul(a, wq, s, b, transpose=True, group_size=gs, bits=bits)
        new = lambda a: fastgrad.qmm(a, wq, s, b, group_size=gs, bits=bits)
        assert mx.array_equal(ref(x), new(x)).item(), "forward must be the same op"
        g = mx.random.normal((3, 7, 96)).astype(BF)
        dr = mx.vjp(ref, [x], [g])[1][0]
        dn = mx.vjp(new, [x], [g])[1][0]
        W = mx.dequantize(wq, s, b, group_size=gs, bits=bits).astype(mx.float32)
        exact = g.astype(mx.float32) @ W
        assert rel(dn, exact) <= rel(dr, exact) * 1.5 + 1e-3, (rel(dn, exact), rel(dr, exact))
    # a mode with no biases
    wq, s = mx.quantize((mx.random.normal((64, 64)) * 0.05).astype(BF), mode="mxfp4")[:2]
    x = mx.random.normal((4, 64)).astype(BF)
    ref = lambda a: mx.quantized_matmul(a, wq, s, None, transpose=True, group_size=32, bits=4, mode="mxfp4")
    new = lambda a: fastgrad.qmm(a, wq, s, None, group_size=32, bits=4, mode="mxfp4")
    assert mx.array_equal(ref(x), new(x)).item()
    g = mx.random.normal((4, 64)).astype(BF)
    assert rel(mx.vjp(new, [x], [g])[1][0], mx.vjp(ref, [x], [g])[1][0]) < 2e-2


def check_gather():
    E, O, I, N, M = 6, 64, 128, 40, 3
    wq, s, b = qweights((E, O, I))
    idx = mx.random.randint(0, E, (N, M)).astype(mx.uint32)
    x = mx.random.normal((N, 1, 1, I)).astype(BF)
    g = mx.random.normal((N, M, 1, O)).astype(BF)

    def ref(a):
        return mx.gather_qmm(a, wq, s, b, rhs_indices=idx, transpose=True, group_size=64, bits=4)

    def new(a):
        return fastgrad.gather_qmm(a, wq, s, b, idx, group_size=64, bits=4)

    assert mx.array_equal(ref(x), new(x)).item()
    assert rel(mx.vjp(new, [x], [g])[1][0], mx.vjp(ref, [x], [g])[1][0]) < 2e-2

    # sorted path with the spread / unsort permutations, as TrainSwitch runs it
    flat = idx.flatten()
    order = mx.argsort(flat)
    inv = mx.argsort(order)
    xs = x.flatten(0, -3)

    def ref_s(a):
        y = mx.gather_qmm(a[order // M], wq, s, b, rhs_indices=flat[order], transpose=True,
                          group_size=64, bits=4, sorted_indices=True)
        return y[inv]

    def new_s(a):
        y = fastgrad.gather_qmm(fastgrad.spread(a, order, inv), wq, s, b, flat[order],
                                group_size=64, bits=4, sorted_indices=True)
        return fastgrad.unsort(y, inv, order)

    gs_ = mx.random.normal((N * M, 1, O)).astype(BF)
    assert mx.array_equal(ref_s(xs), new_s(xs)).item()
    assert rel(mx.vjp(new_s, [xs], [gs_])[1][0], mx.vjp(ref_s, [xs], [gs_])[1][0]) < 2e-2

    # gate and up as one op, sorted and unsorted
    w2, s2, b2 = qweights((E, O, I), bits=8, gs=32)
    for xx, ii, srt in ((x, idx, False), (xs[order // M], flat[order], True)):
        def ref2(a):
            p = mx.gather_qmm(a, wq, s, b, rhs_indices=ii, transpose=True, group_size=64, bits=4, sorted_indices=srt)
            q = mx.gather_qmm(a, w2, s2, b2, rhs_indices=ii, transpose=True, group_size=32, bits=8, sorted_indices=srt)
            return p * q

        def new2(a):
            p, q = fastgrad.gather_qmm2(a, (wq, s, b, 64, 4, "affine"), (w2, s2, b2, 32, 8, "affine"), ii,
                                        sorted_indices=srt)
            return p * q

        gg = mx.random.normal(ref2(xx).shape).astype(BF)
        assert mx.array_equal(ref2(xx), new2(xx)).item()
        assert rel(mx.vjp(new2, [xx], [gg])[1][0], mx.vjp(ref2, [xx], [gg])[1][0]) < 2e-2


def check_attention():
    old = fastgrad.LONG_ATTN, fastgrad.ATTN_CHUNK
    fastgrad.LONG_ATTN, fastgrad.ATTN_CHUNK = 16, 8
    fastgrad.install()
    try:
        B, Hq, Hkv, T, D = 2, 8, 2, 40, 32
        q, k, v, g = (mx.random.normal(s).astype(BF) for s in
                      ((B, Hq, T, D), (B, Hkv, T, D), (B, Hkv, T, D), (B, Hq, T, D)))
        sc = D ** -0.5
        orig = fastgrad._saved["sdpa"]
        ref = lambda a, b_, c: orig(a, b_, c, scale=sc, mask="causal")
        new = lambda a, b_, c: mx.fast.scaled_dot_product_attention(a, b_, c, scale=sc, mask="causal")
        assert mx.array_equal(ref(q, k, v), new(q, k, v)).item()
        exact = mx.vjp(lambda a, b_, c: orig(a, b_, c, scale=sc, mask="causal"),
                       [t.astype(mx.float32) for t in (q, k, v)], [g.astype(mx.float32)])[1]
        dr = mx.vjp(ref, [q, k, v], [g])[1]
        dn = mx.vjp(new, [q, k, v], [g])[1]
        for a, b_, e in zip(dn, dr, exact):
            assert rel(a, e) <= rel(b_, e) * 1.5 + 2e-3, (rel(a, e), rel(b_, e))
        # anything but a plain causal mask keeps the stock path
        arr = mx.zeros((T, T), BF)
        assert mx.array_equal(new(q, k, v), ref(q, k, v)).item()
        mx.fast.scaled_dot_product_attention(q, k, v, scale=sc, mask=arr)
    finally:
        fastgrad.uninstall()
        fastgrad.LONG_ATTN, fastgrad.ATTN_CHUNK = old


def check_install():
    from mlx_lm.tuner.lora import LoRALinear

    orig = nn.QuantizedLinear.__call__
    lin = nn.QuantizedLinear(128, 64, bias=True)
    lin.set_dtype(BF)
    lora = LoRALinear.from_base(lin, r=4)
    lora.lora_b = mx.random.normal(lora.lora_b.shape).astype(lora.lora_b.dtype) * 0.1
    x = mx.random.normal((5, 128)).astype(BF)

    def loss(m, a):
        return (m(a).astype(mx.float32) ** 2).sum()

    y0 = lora(x)
    g0 = nn.value_and_grad(lora, loss)(lora, x)[1]
    assert fastgrad.install()
    try:
        assert nn.QuantizedLinear.__call__ is not orig
        assert mx.array_equal(lora(x), y0).item()
        g1 = nn.value_and_grad(lora, loss)(lora, x)[1]
        for name in ("lora_a", "lora_b"):
            assert rel(g1[name], g0[name]) < 2e-2, name
    finally:
        fastgrad.uninstall()
    assert nn.QuantizedLinear.__call__ is orig
    assert not fastgrad.active()


def main():
    mx.random.seed(0)
    check_qmm()
    check_gather()
    check_attention()
    check_install()
    print("OK")


def test_train_fastgrad():
    main()


if __name__ == "__main__":
    main()
