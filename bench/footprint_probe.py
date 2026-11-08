#!/usr/bin/env python
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where does the process's memory go, phase by phase?

The RAM budget only covers what the engine plans for (backbone, slab,
activations, KV). The kernel sees phys_footprint, which also has the MLX
buffer pool, numpy read buffers and the Python heap. When those sit next to
a full slab, macOS compresses and the pressure governor shrinks the slab.

Walks one agent-shaped session (big prefill, decode, small prefill, decode,
idle) and prints, at each phase:

  foot     phys_footprint (what Activity Monitor and the kernel count)
  active   mx.get_active_memory()  (live Metal buffers, includes the slab)
  pool     mx.get_cache_memory()   (freed Metal buffers MLX keeps for reuse)
  slab     slab bytes
  host     numpy read buffers held by BufferPool
  other    foot - active - pool - host
  avail    what the kernel can hand out without compressing anyone

    python bench/footprint_probe.py --model ~/mlx-models/qwen3-235-4bit
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from pagedmoe import load, mempressure  # noqa: E402
from pagedmoe.streaming import StreamedSwitchGLU  # noqa: E402


def cache_of(model):
    for m in model.modules():
        if isinstance(m, StreamedSwitchGLU):
            return m.cache
    raise SystemExit("not a streamed model")


def sample(label: str, cache) -> None:
    gb = 1e9
    foot = mempressure.footprint_gb()
    active = mx.get_active_memory() / gb
    pool = mx.get_cache_memory() / gb
    slab = (cache.slab.nbytes if cache.slab is not None else 0) / gb
    host = cache._buffers._free_bytes / gb
    avail = (mempressure.available_bytes() or 0) / gb
    print(
        f"{label:>22}: foot {foot:5.1f}  active {active:5.1f}  pool {pool:4.1f}  "
        f"slab {slab:5.1f}  host {host:4.1f}  other {foot - active - pool - host:5.1f}  "
        f"| avail {avail:5.1f}  level {mempressure.name(mempressure.level())}  "
        f"compressor {mempressure.compressor_gb():4.1f}",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--big", type=int, default=9000, help="first prefill (slab-less)")
    ap.add_argument("--small", type=int, default=512, help="tool-round prefill")
    ap.add_argument("--decode", type=int, default=24)
    ap.add_argument("--idle", type=float, default=5.0)
    args = ap.parse_args()

    model, tok = load(args.model, verbose=False)
    cache = cache_of(model)
    sample("loaded", cache)

    text = "\n".join(p.read_text(errors="replace") for p in sorted(Path(__file__).parents[1].rglob("*.py"))[:80])
    ids = tok.encode(text)
    while len(ids) < args.big + 4 * args.small:
        ids = ids + ids
    kv = make_prompt_cache(model)
    at = 0

    def prefill(n, label):
        nonlocal at
        logits = model(mx.array(ids[at : at + n])[None], cache=kv)
        mx.eval(logits)
        at += n
        sample(label, cache)
        return logits

    def decode(logits, n, label):
        for _ in range(n):
            logits = model(mx.argmax(logits[:, -1, :], axis=-1)[None], cache=kv)
            mx.eval(logits)
        sample(label, cache)
        return logits

    logits = prefill(args.big, f"prefill {args.big}")
    logits = decode(logits, args.decode, "decode")
    for turn in (1, 2):
        logits = prefill(args.small, f"prefill {args.small} #{turn}")
        logits = decode(logits, args.decode, f"decode #{turn}")
    time.sleep(args.idle)
    sample(f"idle {args.idle:.0f}s", cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
