# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
PLE n-gram gather: the batched path (row cache, parallel reads, one
dequantize) must return exactly what the per-shard path returns.

Run:

    python tests/test_ple_offload.py
"""

import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pagedmoe.ple_offload import (
    DiskShardedQuantEmbedding,
    _DiskQuantShard,
    _infer_quant,
)
from pagedmoe.safetensors_index import FilePool, read_headers

N_SHARDS, ROWS, DIM = 4, 96, 160


def build(tmp: Path) -> dict:
    mx.random.seed(0)
    tensors = {}
    for s in range(N_SHARDS):
        w = mx.random.normal((ROWS, DIM)).astype(mx.bfloat16)
        q, scales, biases = mx.quantize(w, group_size=32, bits=6)
        base = f"emb.shard_{s}"
        tensors[f"{base}.weight"] = q
        tensors[f"{base}.scales"] = scales
        tensors[f"{base}.biases"] = biases
    mx.save_safetensors(str(tmp / "model-00001.safetensors"), tensors)
    return read_headers(tmp)


def shards_of(index) -> list:
    out = []
    for s in range(N_SHARDS):
        base = f"emb.shard_{s}"
        w, sc, b = index[f"{base}.weight"], index[f"{base}.scales"], index[f"{base}.biases"]
        gs, bits = _infer_quant(w, sc, DIM)
        out.append(_DiskQuantShard(w, sc, b, gs, bits, DIM))
    return out


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        index = build(Path(d))
        pool = FilePool(nocache=False)
        shards = shards_of(index)
        slow = DiskShardedQuantEmbedding(shards, ROWS, DIM, pool, fast=False)
        fast = DiskShardedQuantEmbedding(shards, ROWS, DIM, pool, row_cache=64)
        tiny = DiskShardedQuantEmbedding(shards, ROWS, DIM, pool, row_cache=0, read_threads=1)
        assert slow._fast is None and fast._fast is not None

        rng = np.random.default_rng(1)
        for step in range(6):
            # Decode-sized and prefill-sized lookups, with repeats, so the
            # cache sees hits, misses, eviction, and batches larger than it.
            n = 16 if step % 2 else 400
            gid = rng.integers(0, N_SHARDS * ROWS, size=(1, n, 1)).astype(np.int64)
            if step == 3:
                gid[:, : n // 2] = gid[:, n // 2 : n // 2 * 2]
            g = mx.array(gid)
            ref = np.array(slow(g))
            for mod in (fast, tiny):
                got = np.array(mod(g))
                assert got.shape == ref.shape, (got.shape, ref.shape)
                assert got.dtype == ref.dtype
                assert np.array_equal(got, ref), f"step {step}: batched gather differs"
        f = fast._fast
        assert f.hits > 0 and f.misses > 0, (f.hits, f.misses)
        assert len(f.lru) <= 64
        print(f"ok: bit-identical over 6 lookups (cache hits {f.hits}, misses {f.misses})")

        try:
            fast(mx.array([[N_SHARDS * ROWS]]))
        except IndexError:
            pass
        else:
            raise AssertionError("out-of-range row should raise")
        fast._fast.close()
        tiny._fast.close()
        pool.close()


if __name__ == "__main__":
    main()
