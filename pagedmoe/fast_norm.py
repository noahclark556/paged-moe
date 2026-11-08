# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gemma-style RMSNorm through mx.fast.rms_norm.

Same fp32 formula (normalize, scale by 1 + weight), one kernel instead of the
eager chain. MiniMax-M3 pays this on every layer twice, plus per-head QK norm.
The input has to go in as fp32: bf16 into the fast kernel is not identical.
"""

from __future__ import annotations

import mlx.core as mx

_PATCHED: set[type] = set()


def _patch(cls) -> None:
    if cls in _PATCHED or cls.__name__ != "GemmaRMSNorm":
        return

    def __call__(self, x):
        ot = x.dtype
        w = (1.0 + self.weight).astype(mx.float32)
        y = mx.fast.rms_norm(x.astype(mx.float32), w, float(self.eps))
        return y.astype(ot)

    cls.__call__ = __call__
    _PATCHED.add(cls)


def install(model) -> None:
    """Patch every GemmaRMSNorm class in `model`. Safe to call more than once."""
    for module in model.modules():
        _patch(type(module))
