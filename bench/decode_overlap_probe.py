# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Is the drive busy while the main thread is NOT waiting on it?

Splits decode wall time into "main thread blocked on reads" (inside
ExpertCache._finish) and everything else (GPU sync, python, sampler), and
samples the read pool's busy flags in both. Reads in flight outside the
blocked windows are overlap. Idle drive outside them is time the drive
could have spent on the next layer's reads.

Usage:
  PAGED_MOE_PRUNE=0 PAGED_MOE_WAIT_ABOVE=0 \
  python bench/decode_overlap_probe.py --model ~/mlx-models/qwen3-235-4bit \
      --prompt-file /tmp/esb/prompt.txt
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--hz", type=float, default=4000.0)
    ap.add_argument("--switch-ms", type=float, default=0.0,
                    help="sys.setswitchinterval for this run (0 = leave it)")
    args = ap.parse_args()
    if args.switch_ms > 0:
        import sys
        sys.setswitchinterval(args.switch_ms / 1000.0)

    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    from pagedmoe import config, load
    from pagedmoe.cache import ExpertCache

    state = {"wait": False}
    orig_finish = ExpertCache._finish

    def finish(self, *a, **k):
        state["wait"] = True
        try:
            return orig_finish(self, *a, **k)
        finally:
            state["wait"] = False

    ExpertCache._finish = finish

    from pagedmoe import cache as cache_mod

    done_log: list[tuple[float, int]] = []
    orig_done = cache_mod._SlotLatch.done

    def latch_done(self, exc=None):
        done_log.append((time.perf_counter(), int(state["wait"])))
        return orig_done(self, exc)

    cache_mod._SlotLatch.done = latch_done

    model, tok = load(args.model)
    cache = model._pagedmoe_cache
    pool = cache._reads
    busy = pool.busy

    with open(args.prompt_file, encoding="utf-8") as fh:
        ids = tok.encode(fh.read())[: args.prompt_tokens]
    msgs = [{"role": "user", "content": tok.decode(ids) + "\n\nSummarize the above."}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)

    samples: list[tuple[int, int]] = []
    stop = threading.Event()

    def sample() -> None:
        period = 1.0 / args.hz
        nxt = time.perf_counter()
        while not stop.is_set():
            samples.append((int(state["wait"]), sum(busy)))
            nxt += period
            d = nxt - time.perf_counter()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.perf_counter()

    gen = stream_generate(
        model, tok, prompt, max_tokens=args.max_tokens,
        sampler=make_sampler(temp=0.0), kv_bits=8, kv_group_size=64,
        prefill_step_size=config.PREFILL_CHUNK,
    )
    next(gen)
    done_log.clear()
    r0, h0, m0 = cache.bytes_read, cache.hits, cache.misses
    th = threading.Thread(target=sample, daemon=True)
    th.start()
    t0 = time.perf_counter()
    n = 0
    for _ in gen:
        n += 1
    dt = time.perf_counter() - t0
    stop.set()
    th.join(1.0)

    a = np.asarray(samples, dtype=np.int32)
    w, b = a[:, 0] == 1, a[:, 1]
    per = dt / max(1, n) * 1000
    gb = (cache.bytes_read - r0) / 1e9
    print(f"\ndecode {n} tok {n / dt:.2f} tok/s, {per:.0f} ms/tok, "
          f"{gb / n * 1000:.0f} MB/tok, hit "
          f"{(cache.hits - h0) / max(1, cache.hits - h0 + cache.misses - m0):.3f}")
    print(f"blocked on reads : {w.mean() * per:6.1f} ms/tok, drive busy "
          f"{(b[w] > 0).mean() * 100 if w.any() else 0:5.1f}% depth {b[w].mean() if w.any() else 0:.1f}")
    o = ~w
    print(f"not blocked      : {o.mean() * per:6.1f} ms/tok, drive busy "
          f"{(b[o] > 0).mean() * 100 if o.any() else 0:5.1f}% depth {b[o].mean() if o.any() else 0:.1f}")
    print(f"drive idle total : {(b == 0).mean() * per:6.1f} ms/tok")
    print(f"avg rate         : {gb / dt:5.2f} GB/s")
    eb = cache.slab.expert_bytes if cache.slab is not None else 0
    in_w = sum(1 for _, wv in done_log if wv)
    out_w = len(done_log) - in_w
    tw, to = w.mean() * dt, (~w).mean() * dt
    print(f"experts landed   : {len(done_log) / n:.1f}/tok, while blocked "
          f"{in_w * eb / 1e9 / max(tw, 1e-9):5.2f} GB/s, otherwise "
          f"{out_w * eb / 1e9 / max(to, 1e-9):5.2f} GB/s")


if __name__ == "__main__":
    main()
