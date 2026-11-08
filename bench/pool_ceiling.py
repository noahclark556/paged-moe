# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native read pool ceiling on this checkpoint's real expert layout.

Reads random (layer, expert) pairs into slab slots through the same pool and
plans decode uses, `--inflight` experts at a time, no compute. Clobbers the
slab, so run it in its own process.

  python bench/pool_ceiling.py --model ~/mlx-models/qwen3-235-4bit
"""

from __future__ import annotations

import argparse
import random
import threading
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--experts", type=int, default=600)
    ap.add_argument("--inflight", default="2,4,8,16,32")
    args = ap.parse_args()

    from pagedmoe import load
    from pagedmoe.cache import _SlotLatch
    from concurrent.futures import Future

    model, _ = load(args.model)
    c = model._pagedmoe_cache
    pool, slab = c._reads, c.slab
    layers = list(c._read_plan)
    n_exp = max(1, c.slab.slots)
    rng = random.Random(7)
    seen = set()

    def pick():
        while True:
            lk = rng.choice(layers)
            e = rng.randrange(128)
            if (lk, e) not in seen:
                seen.add((lk, e))
                return lk, e

    for depth in [int(x) for x in args.inflight.split(",")]:
        sem = threading.Semaphore(depth)
        done = threading.Event()
        left = [args.experts]
        lock = threading.Lock()
        t0 = time.perf_counter()
        for i in range(args.experts):
            sem.acquire()
            lk, e = pick()
            plan = c._read_plan[lk]
            slot = i % n_exp
            comps = [(fd, base + e * stride, mv) for (fd, base, stride), mv in zip(plan, slab.slot_dests[slot])]
            fut: Future = Future()
            fut.set_running_or_notify_cancel()

            def cb(f):
                sem.release()
                with lock:
                    left[0] -= 1
                    if left[0] == 0:
                        done.set()

            fut.add_done_callback(cb)
            pool.submit_expert(comps, _SlotLatch(1, fut, {}, None), urgent=True)
        done.wait()
        dt = time.perf_counter() - t0
        gb = args.experts * slab.expert_bytes / 1e9
        print(f"inflight {depth:3d} experts: {gb / dt:6.2f} GB/s  ({dt * 1000 / args.experts:.2f} ms/expert)")


if __name__ == "__main__":
    main()
