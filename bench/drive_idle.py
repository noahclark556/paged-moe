# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""How much of a decode token is the drive actually moving bytes?

decode_profile says the miss path reads 2.6 GB/token and bench/disk_ceiling.py
says the drive does ~14.4 GB/s, which would be ~183 ms - but a token takes
~337 ms. Aggregate throughput cannot tell "the drive is saturated and we need
fewer bytes" apart from "the drive is idle half the time waiting for python",
and those have completely different fixes.

So sample ExpertCache._reads.busy (one byte per read worker, set around the
pread) from an observer thread and report the distribution of how many reads
are in flight. Occupancy is the fraction of samples with at least one active
read: 1.0 means bytes are the only lever left, anything well below means there
is idle drive time to reclaim by keeping the queue fed.

Usage:
  python bench/drive_idle.py --model ~/mlx-models/qwen3-235-4bit
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--hz", type=float, default=2000.0)
    args = ap.parse_args()

    from pagedmoe.loader import load

    model, tokenizer = load(args.model)
    cache = model._pagedmoe_cache
    pool = getattr(cache, "_reads", None)
    if pool is None:
        raise SystemExit(
            "no read pool (PAGED_MOE_READ_POOL_THREADS=-1, or slabs off) - "
            "this probe measures the pooled read path"
        )

    # Same prompt as bench/decode_profile.py so the two are comparable: a
    # repetitive prompt routes to a handful of experts and inflates the hit
    # rate to the point that the miss path barely runs.
    para = (
        "The design of storage systems involves fundamental tradeoffs between "
        "capacity, bandwidth, and latency. Solid state drives offer high "
        "sequential throughput but random access patterns reveal their "
        "internal parallelism limits. Memory hierarchies exploit locality. "
    )
    ids = tokenizer.encode(para * 400)[: args.prompt_tokens]
    msgs = [
        {"role": "user", "content": tokenizer.decode(ids) + "\n\nSummarize the above."}
    ]
    prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)

    samples: list[int] = []
    stop = threading.Event()

    def sample() -> None:
        busy = pool.busy
        period = 1.0 / args.hz
        nxt = time.perf_counter()
        while not stop.is_set():
            samples.append(sum(busy))
            nxt += period
            d = nxt - time.perf_counter()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.perf_counter()

    from mlx_lm.generate import stream_generate

    gen = stream_generate(model, tokenizer, prompt, max_tokens=args.max_tokens)
    next(gen)  # prefill + first token, which is not what we are measuring

    snap_read = cache.bytes_read
    t = threading.Thread(target=sample, daemon=True)
    t.start()
    t0 = time.perf_counter()
    n = 0
    for _ in gen:
        n += 1
    dt = time.perf_counter() - t0
    stop.set()
    t.join(timeout=1.0)

    read_gb = (cache.bytes_read - snap_read) / 1e9
    a = np.asarray(samples, dtype=np.int32)
    occ = float((a > 0).mean())
    depth_busy = float(a[a > 0].mean()) if (a > 0).any() else 0.0

    print(f"\ndecode: {n} tokens in {dt:.2f}s = {n / dt:.2f} tok/s")
    print(f"read: {read_gb:.2f} GB ({read_gb / max(1, n) * 1000:.0f} MB/token)")
    print(f"samples: {len(a)} at ~{len(a) / dt:.0f} Hz over {pool.threads} workers")
    print(f"drive occupancy   : {occ * 100:5.1f}% of wall has >=1 read in flight")
    print(f"idle drive time   : {(1 - occ) * dt / max(1, n) * 1000:5.1f} ms/token")
    print(f"mean queue depth  : {a.mean():5.2f} workers busy (over all samples)")
    print(f"  when busy       : {depth_busy:5.2f}")
    if occ > 0:
        print(f"throughput while busy: {read_gb / (occ * dt):5.2f} GB/s")
    print("  depth histogram (workers busy -> % of wall):")
    for k in range(0, pool.threads + 1):
        pct = float((a == k).mean()) * 100
        if pct >= 0.5:
            print(f"    {k:3d}: {pct:5.1f}%  {'#' * int(pct / 2)}")
    print(f"  p50/p90/p99 depth: {np.percentile(a, [50, 90, 99]).round(1).tolist()}")
    print(f"  (statistics.mean sanity: {statistics.mean(a.tolist()):.2f})")


if __name__ == "__main__":
    main()
