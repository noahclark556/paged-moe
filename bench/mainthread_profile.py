#!/usr/bin/env python3
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which main-thread python calls are starving the reader threads?

bench/gil_probe.py established the mechanism: mx.eval releases the GIL, but
pure-python work on the main thread costs up to 74% of read bandwidth, and
decode achieves 7.54 GB/s against a 14.5 GB/s drive. So the question is no
longer *whether* main-thread python is the bottleneck but *which* calls it
is made of.

cProfile only instruments the thread that enables it, which is exactly what
is wanted here: the reader threads stay uninstrumented and keep running at
full speed while every main-thread call is counted.

Read the output as a ranking, not as absolute times - cProfile's per-call
overhead inflates functions with many short calls, which is the same
population being hunted, so the bias points the right way but the magnitudes
are not wall clock. `tottime` (time in the function itself, excluding
callees) is the column that matters, because that is the GIL-held work.

The `per token` column is the honest one for deciding what to fix: a function
called 94 times per token for 30 us is a bigger problem than one called twice
for 1 ms.

Usage:
  PAGED_MOE_CACHE_GB=28 PAGED_MOE_HOOK=0 \
  ~/.mlx-env/bin/python bench/mainthread_profile.py \
      --model ~/mlx-models/qwen3-235-4bit --max-tokens 24
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import io
import time

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from pagedmoe import load

_PROMPT = (
    "Explain, in careful detail, the engineering trade-offs between "
    "capacity, bandwidth and latency in a tiered storage system that pages "
    "model weights from SSD into unified memory. Cover cache sizing, "
    "prefetch, and how to measure whether a change actually helped. "
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--top", type=int, default=28)
    args = ap.parse_args()

    model, tokenizer = load(args.model)
    prompt = _PROMPT
    while len(tokenizer.encode(prompt)) < args.prompt_tokens:
        prompt += _PROMPT
    ids = tokenizer.encode(prompt)[: args.prompt_tokens]
    prompt = tokenizer.decode(ids)
    sampler = make_sampler(temp=0.0)

    # One generator, profiler switched on only after the first token. Prefill
    # has a completely different cost structure - grouped streaming, soft
    # installs, the slab handed back - and a separate stream_generate call
    # would re-prefill the whole prompt. An earlier version of this script did
    # exactly that and produced a profile of prefill wearing decode's label
    # (5137 MB/token read, `_run` and `_read_span_sliced` at the top, both of
    # which decode never touches).
    cache = model._pagedmoe_cache
    it = iter(
        stream_generate(
            model, tokenizer, prompt,
            max_tokens=args.max_tokens + 1, sampler=sampler,
        )
    )
    next(it)  # prefill + first token

    before = (cache.bytes_read, cache.demand_miss_bytes, cache.spec_wasted_bytes)
    pr = cProfile.Profile()
    n = 0
    t0 = time.perf_counter()
    pr.enable()
    for _ in it:
        n += 1
    pr.disable()
    dt = time.perf_counter() - t0

    read = (cache.bytes_read - before[0]) / 1e9
    demand = (cache.demand_miss_bytes - before[1]) / 1e9
    wasted = (cache.spec_wasted_bytes - before[2]) / 1e9
    print(f"\n{n} tokens in {dt:.2f}s = {n/dt:.2f} tok/s (profiled; slower "
          f"than unprofiled)")
    print(f"read {read:.1f} GB ({read*1e3/n:.0f} MB/token), "
          f"demand {demand*1e3/n:.0f} MB/token, "
          f"wasted {wasted*1e3/n:.0f} MB/token "
          f"({wasted/max(read,1e-9):.0%} of traffic)")

    buf = io.StringIO()
    st = pstats.Stats(pr, stream=buf)
    st.sort_stats("tottime")

    # cProfile in 3.12 installs the profile hook on every thread, so this
    # covers the reader pools too. That is useful (the thread-pool and lock
    # traffic the read path generates is exactly what holds the GIL) but it
    # means "self time" in preadv is mostly GIL-released syscall time and
    # should not be read as overhead.
    print(f"\ndecode python across all threads, ranked by self time "
          f"({args.top} worst)")
    print(f"{'tottime':>9}{'calls':>9}{'per tok':>9}{'us/call':>9}  function")
    rows = []
    for func, (cc, nc, tt, ct, _callers) in st.stats.items():
        rows.append((tt, nc, ct, func))
    rows.sort(reverse=True)
    for tt, nc, ct, func in rows[: args.top]:
        fn = f"{func[2]} ({func[0].rsplit('/', 1)[-1]}:{func[1]})"
        print(f"{tt:9.3f}{nc:9d}{nc/n:9.1f}{tt/max(nc,1)*1e6:9.1f}  {fn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
