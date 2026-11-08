#!/usr/bin/env python
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where one tool-result prefill spends its wall time.

Prints expert-read bytes, time blocked on those reads, and time inside the
PLE row reader. Exact prefill. No server change.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from pagedmoe import load  # noqa: E402
from pagedmoe import ple_offload  # noqa: E402
from pagedmoe.streaming import cache_only, exact_step  # noqa: E402

PLE = {"s": 0.0, "jobs": 0, "bytes": 0}


def _patch_ple() -> None:
    orig = ple_offload._RowReader.read

    def read(self, jobs):
        n = 0
        for job in jobs:
            n += len(job[2])
        t0 = time.perf_counter()
        orig(self, jobs)
        PLE["s"] += time.perf_counter() - t0
        PLE["jobs"] += len(jobs)
        PLE["bytes"] += n

    ple_offload._RowReader.read = read


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / "mlx-models/qwen38-coder-flash")
    prefix_n, chunk_n = 4096, 1300
    _patch_ple()
    model, tok = load(model_path, mode="streamed", verbose=False)
    text = "\n".join(
        p.read_text(errors="replace")[:60_000]
        for p in sorted((ROOT / "pagedmoe").glob("*.py"))[:12]
    )
    ids = list(tok.encode(text))
    while len(ids) < prefix_n + chunk_n + 8:
        ids = ids + ids
    cache = make_prompt_cache(model)
    with exact_step(), cache_only():
        pos = 0
        while pos < prefix_n:
            n = min(4096, prefix_n - pos)
            mx.eval(model(mx.array([ids[pos : pos + n]]), cache=cache))
            pos += n
    ec = model._pagedmoe_cache
    print(
        f"after seed slab_slots={ec.slab.slots if ec.slab else 0} "
        f"resident_gb={ec.resident_bytes / 1e9:.2f}",
        flush=True,
    )
    PLE["s"] = PLE["jobs"] = PLE["bytes"] = 0
    b0, w0 = ec.bytes_read, ec.disk_wait_s
    t0 = time.perf_counter()
    with exact_step(), cache_only():
        mx.eval(model(mx.array([ids[pos : pos + chunk_n]]), cache=cache))
    wall = time.perf_counter() - t0
    gb = (ec.bytes_read - b0) / 1e9
    blocked = ec.disk_wait_s - w0
    print(
        f"chunk {chunk_n}: wall {wall:.2f}s  expert_read {gb:.2f} GB "
        f"({gb / max(wall, 1e-9):.2f} GB/s)  disk_wait {blocked:.2f}s  "
        f"ple_read {PLE['s']:.2f}s  ple_jobs {PLE['jobs']}  "
        f"ple_mb {PLE['bytes'] / 1e6:.1f}  "
        f"slab_slots={ec.slab.slots if ec.slab else 0}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
