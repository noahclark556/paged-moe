# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Agent-shaped tool rounds: tool-result prefill, then decode, repeated.

One process per arm (env is read at import), so run it twice with different
env and compare. Each round prefills a tool-result sized chunk on top of the
seeded context, then decodes, and reports prefill time / GB / peak memory and
decode speed for the first 32 tokens and the rest.

  PAGED_MOE_PREFILL_SHRINK_TOKENS=512 python bench/tool_round_ab.py --model ~/mlx-models/qwen38-coder-flash
"""

import argparse
import os
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
    ap.add_argument("--rounds", type=int, nargs="+", default=[1300, 300, 2600, 900])
    ap.add_argument("--decode", type=int, default=96)
    ap.add_argument("--label", default="")
    ap.add_argument("--cache-only", action="store_true", help="prefill like the server does (streaming.cache_only)")
    args = ap.parse_args()
    import contextlib

    pre_ctx = streaming.cache_only if args.cache_only else contextlib.nullcontext

    model, tok = load(args.model, mode="streamed", verbose=False)
    ec = model._pagedmoe_cache
    root = Path(__file__).resolve().parents[1] / "pagedmoe"
    ids = tok.encode("".join((root / f).read_text(errors="replace") for f in ("cache.py", "streaming.py", "server.py", "loader.py")))
    while len(ids) < args.prefill + sum(args.rounds) + args.decode * (len(args.rounds) + 1) + 64:
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
        marks = []
        t = time.perf_counter()
        for i in range(n):
            mx.eval(model(mx.array([[ids[pos]]]), cache=cache))
            pos += 1
            if i == 31:
                marks.append(time.perf_counter() - t)
        total = time.perf_counter() - t
        return 32 / marks[0], (n - 32) / (total - marks[0])

    decode(args.decode)
    env = {k: v for k, v in os.environ.items() if k.startswith("PAGED_MOE_PREFILL_SHRINK") or k.startswith("PAGED_MOE_PROBE")}
    print(f"arm {args.label} {env}", flush=True)
    tot_p = tot_d = 0.0
    for n in args.rounds:
        mx.reset_peak_memory()
        b0 = ec.bytes_read
        t = time.perf_counter()
        with streaming.exact_step(), pre_ctx():
            model(mx.array([ids[pos : pos + n]]), cache=cache)
            mx.eval([c.state for c in cache])
        dt = time.perf_counter() - t
        pos += n
        gb = (ec.bytes_read - b0) / 1e9
        peak = mx.get_peak_memory() / 1e9
        t1 = time.perf_counter()
        first, rest = decode(args.decode)
        dd = time.perf_counter() - t1
        tot_p += dt
        tot_d += dd
        slots = getattr(getattr(ec, "slab", None), "slots", None)
        need = getattr(ec, "chunk_need", 0) / 1e9
        print(
            f"round {n:5} tok: prefill {dt:6.2f}s {gb:6.1f} GB ({gb / dt:4.1f} GB/s) peak {peak:5.1f} GB need {need:4.2f} GB | "
            f"decode first32 {first:5.2f} tok/s rest {rest:5.2f} tok/s | slab {slots}",
            flush=True,
        )
    print(f"total prefill {tot_p:.1f}s decode {tot_d:.1f}s  round wall {tot_p + tot_d:.1f}s", flush=True)


if __name__ == "__main__":
    main()
