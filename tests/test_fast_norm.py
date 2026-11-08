# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The fused Gemma norm gives the same bits as the eager fp32 formula.

    python tests/test_fast_norm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import numpy as np


def test_gemma_norm_matches_eager():
    mx.random.seed(0)
    eps = 1e-6
    for shape in ((2, 64, 128), (1, 7, 64, 128), (1, 1, 6144)):
        x = mx.random.normal(shape).astype(mx.bfloat16)
        w = mx.random.normal((shape[-1],)).astype(mx.float32) * 0.05
        xf = x.astype(mx.float32)
        eager = (xf * mx.rsqrt(xf.square().mean(-1, keepdims=True) + eps) * (1.0 + w)).astype(
            x.dtype
        )
        fast = mx.fast.rms_norm(xf, (1.0 + w).astype(mx.float32), eps).astype(x.dtype)
        mx.eval(eager, fast)
        diff = np.max(np.abs(np.array(eager.astype(mx.float32)) - np.array(fast.astype(mx.float32))))
        assert diff == 0.0, (shape, diff)


if __name__ == "__main__":
    test_gemma_norm_matches_eager()
    print("OK: fast norm")
