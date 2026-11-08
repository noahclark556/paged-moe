# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prefill time with the streamed MoE stubbed out, next to the real thing.

Splits a tool-result sized chunk into "backbone" (attention, linear attention,
norms, residual mixing, routers) and "experts" (reads + expert matmuls), at a
realistic context length. The stubbed pass returns zeros from every streamed
MoE layer, so its output is meaningless; only the timing is used, and each arm
gets its own copy of the seeded cache.

  python bench/backbone_prefill_cost.py --model ~/mlx-models/qwen38-coder-flash --chunks 300 1300 2600
"""

import argparse
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from pagedmoe import load, streaming  # noqa: E402
from pagedmoe.streaming import StreamedSwitchGLU  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prefill", type=int, default=16000)
    ap.add_argument("--chunks", type=int, nargs="+", default=[300, 1300, 2600])
    args = ap.parse_args()

    model, tok = load(args.model, mode="streamed", verbose=False)
    root = Path(__file__).resolve().parents[1] / "pagedmoe"
    ids = tok.encode("".join((root / f).read_text(errors="replace") for f in ("cache.py", "streaming.py", "server.py")))
    while len(ids) < args.prefill + max(args.chunks) + 8:
        ids = ids + ids
    cache = make_prompt_cache(model)
    pos = 0
    with streaming.exact_step():
        while pos < args.prefill:
            n = min(4096, args.prefill - pos)
            mx.eval(model(mx.array([ids[pos : pos + n]]), cache=cache))
            pos += n
    mx.eval([c.state for c in cache])

    real_call = StreamedSwitchGLU.__call__

    def stub(self, x, indices):
        return mx.zeros((*indices.shape, x.shape[-1]), dtype=x.dtype)

    for n in args.chunks:
        chunk = mx.array([ids[pos : pos + n]])
        out = {}
        for arm in ("stub", "real", "stub2"):
            c = copy.deepcopy(cache)
            mx.eval([e.state for e in c])
            StreamedSwitchGLU.__call__ = stub if arm.startswith("stub") else real_call
            t = time.perf_counter()
            with streaming.exact_step():
                mx.eval(model(chunk, cache=c))
            out[arm] = time.perf_counter() - t
            del c
            mx.clear_cache()
        StreamedSwitchGLU.__call__ = real_call
        bb = min(out["stub"], out["stub2"])
        print(
            f"chunk {n:5}: real {out['real']:6.2f}s  backbone-only {bb:6.2f}s "
            f"({bb / out['real'] * 100:4.0f}%)  backbone ms/tok {bb / n * 1e3:5.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
