#!/usr/bin/env python
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Time a prefill that starts this layer's expert reads before attention.

The gate on the pre-attention residual names about 94% of the real union
(bench/prefill_early_route.py). This issues that read through the existing
prefetch path, then runs the normal exact prefill, which still computes the
router's own experts. Compared with the same chunk shape and no early issue.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PAGED_MOE_SIDECAR", "0")
os.environ.setdefault("PAGED_MOE_PRUNE", "0")
os.environ.setdefault("PAGED_MOE_FLOW", "0")

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from pagedmoe import load  # noqa: E402
from pagedmoe.streaming import cache_only, exact_step  # noqa: E402

EARLY = {"on": False, "issued": 0}


def _install(model) -> None:
    layer = model.model.layers[0]
    dec_cls = type(layer)
    orig = dec_cls.__call__

    def dec_call(self, h, rope, mask, conv_mask, cache, idx_cache, ids, prev_ctx):
        if EARLY["on"] and h.shape[-2] > 20:
            mixed = self.mlp_hyper_connection(h)
            x = mixed[0] if isinstance(mixed, tuple) else mixed
            logits = self.mlp.gate(x.astype(mx.float32))
            k = self.mlp.top_k
            idx = mx.argpartition(-logits, k - 1, axis=-1)[..., :k]
            mx.eval(idx)
            sw = self.mlp.switch_mlp
            if hasattr(sw, "cache") and hasattr(sw, "layer_key"):
                ids_u = np.unique(np.array(idx).reshape(-1))
                sw.cache.prefetch(sw.layer_key, ids_u.tolist())
                EARLY["issued"] += int(ids_u.size)
        return orig(self, h, rope, mask, conv_mask, cache, idx_cache, ids, prev_ctx)

    dec_cls.__call__ = dec_call


def _chunk(model, cache, ids) -> tuple[float, float, int]:
    ec = model._pagedmoe_cache
    b0, w0, s0 = ec.bytes_read, ec.disk_wait_s, ec.spec_skipped
    t0 = time.perf_counter()
    with exact_step(), cache_only():
        mx.eval(model(mx.array([ids]), cache=cache))
    return (
        time.perf_counter() - t0,
        (ec.bytes_read - b0) / 1e9,
        ec.spec_skipped - s0,
    )


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / "mlx-models/qwen38-coder-flash")
    prefix_n, chunk_n = 1024, 512
    model, tok = load(model_path, mode="streamed", verbose=False)
    _install(model)
    text = "\n".join(
        p.read_text(errors="replace")[:40_000]
        for p in sorted((ROOT / "pagedmoe").glob("*.py"))[:8]
    )
    ids = list(tok.encode(text))
    while len(ids) < prefix_n + chunk_n * 4 + 8:
        ids = ids + ids
    cache = make_prompt_cache(model)
    with exact_step(), cache_only():
        mx.eval(model(mx.array([ids[:prefix_n]]), cache=cache))
    pos = prefix_n
    for name, early in (("base-1", False), ("base-2", False), ("early", True), ("base-3", False)):
        EARLY["on"] = early
        EARLY["issued"] = 0
        wall, gb, skipped = _chunk(model, cache, ids[pos : pos + chunk_n])
        pos += chunk_n
        print(
            f"{name}: {wall:.2f}s  {gb:.2f} GB  spec_skipped {skipped}  "
            f"early_ids {EARLY['issued']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
