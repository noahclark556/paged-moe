# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Verify pass cost under the app's speed settings, not a warm exact cache.

verify_cost_probe.py measures a pass on a cache that exact decode warmed. In
the app, plain steps run PRUNE / WAIT_ABOVE, so the slab holds the pruned
working set and an exact pass has to read everything those steps skipped.
This seeds residency with pruned decode first, then interleaves:

  plain   1 position, app settings
  exact   N positions under streaming.exact_step()  (what verify does now)
  approx  N positions under the app settings         (per-row selector)

  PAGED_MOE_PRUNE=0.9 PAGED_MOE_WAIT_ABOVE=0.45 \
  python bench/verify_live_cost.py --model ~/mlx-models/qwen38-coder-flash --prefill 16000
"""

import argparse
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
    ap.add_argument("--warm", type=int, default=128)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--between", type=int, default=6, help="plain steps between measured arms")
    args = ap.parse_args()

    model, tok = load(args.model, mode="streamed", verbose=False)
    ec = model._pagedmoe_cache
    root = Path(__file__).resolve().parents[1] / "pagedmoe"
    text = "".join((root / f).read_text(errors="replace") for f in ("cache.py", "streaming.py", "server.py"))
    ids = tok.encode(text)
    need = args.prefill + args.warm + args.reps * (3 * args.between + 2 * args.n + 4) + 64
    while len(ids) < need:
        ids = ids + ids
    cache = make_prompt_cache(model)
    t = time.perf_counter()
    pos = 0
    with streaming.exact_step():
        while pos < args.prefill:
            n = min(4096, args.prefill - pos)
            mx.eval(model(mx.array([ids[pos : pos + n]]), cache=cache))
            pos += n
    print(f"prefill {args.prefill} tok {time.perf_counter() - t:.1f}s", flush=True)

    def step(n, exact):
        nonlocal pos
        inp = mx.array([ids[pos : pos + n]])
        b0, w0 = ec.bytes_read, ec.disk_wait_s
        t0 = time.perf_counter()
        if exact:
            with streaming.exact_step():
                mx.eval(model(inp, cache=cache))
        else:
            mx.eval(model(inp, cache=cache))
        dt = time.perf_counter() - t0
        pos += n
        return dt, (ec.bytes_read - b0) / 1e9, ec.disk_wait_s - w0

    for _ in range(args.warm):
        step(1, False)

    res = {"plain": [], "exact": [], "approx": []}
    for r in range(args.reps):
        order = ["plain", "exact", "approx"] if r % 2 == 0 else ["approx", "exact", "plain"]
        for arm in order:
            for _ in range(args.between):
                step(1, False)
            if arm == "plain":
                res[arm].append(step(1, False))
            else:
                res[arm].append(step(args.n, arm == "exact"))

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2]

    base = med([x[0] for x in res["plain"]])
    for arm, xs in res.items():
        n = 1 if arm == "plain" else args.n
        m = med([x[0] for x in xs])
        print(
            f"{arm:6} N={n:2}  median {m * 1e3:7.1f} ms  x{m / base:5.2f} of plain  "
            f"read {med([x[1] for x in xs]):5.2f} GB  disk_wait {med([x[2] for x in xs]) * 1e3:6.1f} ms  "
            f"break-even accept {max(0.0, m / base - 1):4.1f} tok",
            flush=True,
        )


if __name__ == "__main__":
    main()
