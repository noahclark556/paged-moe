# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
PagedMoE: run MoE models bigger than your RAM by streaming experts from disk
on demand (Apple Silicon / MLX).

Public API:

    from pagedmoe import load, get_stats

    model, tokenizer = load("mlx-community/Qwen3-30B-A3B-4bit")

    from mlx_lm import stream_generate
    for chunk in stream_generate(model, tokenizer, prompt="hello", max_tokens=100):
        print(chunk.text, end="")

    print(get_stats(model))

See README.md for install and usage. The import name remains ``pagedmoe``.
"""

from __future__ import annotations

import os
from typing import Any

# MLX fast synch stays off. With it on, a wait between two streams is a
# fence_wait kernel that spins on the GPU until the other stream's counter
# moves. If that update never lands, the spinner never ends: macOS cannot
# preempt it, WindowServer starves until its watchdog kills it, and the
# process's GPU memory stays held until a reboot. That was the hang seen
# after long agent runs. Driver events cost about 2%.
# Must be set before the Metal device initializes, hence here. setdefault
# so a user override wins.
os.environ.setdefault("MLX_METAL_FAST_SYNCH", "0")

# Larger command buffers. Decode commits less often: about 5% on Coder-Next
# exact, neutral on Flash. The 40 MB cap is unchanged, so big prefill kernels
# still flush on size. PAGED_MOE_FAST_BUFFERS=0 leaves MLX's per-chip default,
# which is the right call when a long prompt is already out of memory.
# setdefault so an explicit MLX_MAX_OPS_PER_BUFFER still wins.
if os.environ.get("PAGED_MOE_FAST_BUFFERS", "1") != "0":
    os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "500")

__version__ = "0.3.1"
__all__ = [
    "load",
    "get_stats",
    "relieve_pressure",
    "resolve_model_path",
]


def __getattr__(name: str) -> Any:
    # Lazy: keep ``python -m pagedmoe.mlx_hook status`` Metal-free.
    if name in __all__:
        from . import loader

        return getattr(loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
