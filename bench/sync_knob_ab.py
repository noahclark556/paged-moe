#!/usr/bin/env python
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One greedy decode, for A/B of Metal sync knobs.

The knobs are read when the Metal device is created, so set
MLX_MAX_OPS_PER_BUFFER and MLX_METAL_FAST_SYNCH in the environment
before starting this process. Prints one line: tok/s over the timed
tokens, after a short warmup on the same cache.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from pagedmoe import load  # noqa: E402
from pagedmoe.streaming import exact_step  # noqa: E402


def main() -> int:
    model_path = sys.argv[1]
    prompt_n = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    timed = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    label = os.environ.get("SYNC_AB_LABEL", "run")
    model, tok = load(model_path, mode="streamed", verbose=False)
    text = (ROOT / "pagedmoe" / "cache.py").read_text(errors="replace")
    ids = list(tok.encode(text))
    while len(ids) < prompt_n + 8:
        ids = ids + ids
    ids = ids[:prompt_n]
    cache = make_prompt_cache(model)
    with exact_step():
        mx.eval(model(mx.array([ids]), cache=cache))
        x = mx.array([[ids[-1]]])
        for _ in range(warmup):
            logits = model(x, cache=cache)
            mx.eval(logits)
            x = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
        t0 = time.perf_counter()
        got = []
        for _ in range(timed):
            logits = model(x, cache=cache)
            mx.eval(logits)
            x = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
            got.append(int(x.item()))
        mx.eval(x)
    dt = time.perf_counter() - t0
    ec = model._pagedmoe_cache
    print(
        f"{label}  tok/s {timed / dt:.3f}  {timed} tok in {dt:.2f}s  "
        f"read_gb {ec.bytes_read / 1e9:.2f}  tokens {got[:6]}..{got[-3:]}  "
        f"ops={os.environ.get('MLX_MAX_OPS_PER_BUFFER', '<unset>')}  "
        f"fast={os.environ.get('MLX_METAL_FAST_SYNCH', '<unset>')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
