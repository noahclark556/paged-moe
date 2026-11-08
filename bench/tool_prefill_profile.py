# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where a tool-result prefill spends its time, on an agent-shaped cache.

Seeds a long context, decodes with the app's speed settings so the slab holds
the decode working set, then prefills chunks the size of real tool results
and profiles the main thread (cProfile, reader threads uninstrumented).

  PAGED_MOE_PRUNE=0.9 PAGED_MOE_WAIT_ABOVE=0.45 \
  python bench/tool_prefill_profile.py --model ~/mlx-models/qwen38-coder-flash --chunks 300 1300
"""

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from pagedmoe import load, streaming  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prefill", type=int, default=16000)
    ap.add_argument("--decode", type=int, default=96)
    ap.add_argument("--chunks", type=int, nargs="+", default=[300, 1300])
    ap.add_argument("--top", type=int, default=22)
    args = ap.parse_args()

    model, tok = load(args.model, mode="streamed", verbose=False)
    ec = model._pagedmoe_cache
    root = Path(__file__).resolve().parents[1] / "pagedmoe"
    text = "".join((root / f).read_text(errors="replace") for f in ("cache.py", "streaming.py", "server.py", "loader.py"))
    ids = tok.encode(text)
    while len(ids) < args.prefill + sum(args.chunks) + args.decode * (len(args.chunks) + 1) + 64:
        ids = ids + ids
    cache = make_prompt_cache(model)
    pos = 0
    with streaming.exact_step():
        while pos < args.prefill:
            n = min(4096, args.prefill - pos)
            mx.eval(model(mx.array([ids[pos : pos + n]]), cache=cache))
            pos += n

    def decode(n):
        nonlocal pos
        t = time.perf_counter()
        for _ in range(n):
            mx.eval(model(mx.array([[ids[pos]]]), cache=cache))
            pos += 1
        return time.perf_counter() - t

    decode(args.decode)
    for n in args.chunks:
        b0, w0, m0 = ec.bytes_read, ec.disk_wait_s, ec.materialize_s
        prof = cProfile.Profile()
        t0 = time.perf_counter()
        prof.enable()
        with streaming.exact_step():
            mx.eval(model(mx.array([ids[pos : pos + n]]), cache=cache))
        prof.disable()
        wall = time.perf_counter() - t0
        pos += n
        gb = (ec.bytes_read - b0) / 1e9
        print(
            f"\n=== chunk {n} tok: {wall:.2f}s  read {gb:.1f} GB ({gb / wall:.1f} GB/s)  "
            f"disk_wait {ec.disk_wait_s - w0:.2f}s  materialize {ec.materialize_s - m0:.2f}s",
            flush=True,
        )
        s = io.StringIO()
        pstats.Stats(prof, stream=s).sort_stats("tottime").print_stats(args.top)
        print("\n".join(s.getvalue().splitlines()[:args.top + 12]))
        dt = decode(args.decode)
        print(f"decode after: {args.decode / dt:.2f} tok/s", flush=True)


if __name__ == "__main__":
    main()
