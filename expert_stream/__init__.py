# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
PagedMoE: run MoE models bigger than your RAM by streaming experts from disk
on demand (Apple Silicon / MLX).

Public API:

    from expert_stream import load, get_stats

    model, tokenizer = load("mlx-community/Qwen3-30B-A3B-4bit")

    from mlx_lm import stream_generate
    for chunk in stream_generate(model, tokenizer, prompt="hello", max_tokens=100):
        print(chunk.text, end="")

    print(get_stats(model))

See README.md for install and usage. The import name remains ``expert_stream``.
"""

from __future__ import annotations

import os
from typing import Any

# Cheaper GPU-completion waits (shared Metal events instead of command-buffer
# notify listeners). A streamed decode pays one blocking sync per MoE layer -
# 94/token on Qwen3-235B - so wait latency is on the critical path 94 times
# per token; this measured +2% end-to-end. Must be set before the Metal
# device initializes, hence here. setdefault so a user override wins.
os.environ.setdefault("MLX_METAL_FAST_SYNCH", "1")

__version__ = "0.2.4"
__all__ = [
    "load",
    "get_stats",
    "relieve_pressure",
    "resolve_model_path",
]


def __getattr__(name: str) -> Any:
    # Lazy: keep ``python -m expert_stream.mlx_hook status`` Metal-free.
    if name in __all__:
        from . import loader

        return getattr(loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
