# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offload Qwen4 PLE n-gram embedding tables from Metal.

Qwen3.8-Flash-Next's PLE table is ~51B hashed n-gram parameters, designed to
live off-accelerator: only ~16 rows are touched per token. Leaving those
shards resident makes the checkpoint unbootable on a 48 GB Mac even after MoE
experts are streamed (the table alone is ~45 GB at 6-bit).

This module swaps the in-tree sharded QuantizedEmbedding for a disk-backed
gather that preads only the hashed rows, then dequantizes them. Logits match a
resident table for the same rows.

Rows are tiny (a few hundred bytes over three tensors) and n-grams repeat, so
raw rows sit in a small LRU, misses are read in parallel, and every shard's
rows go through one dequantize. A serial pread per component plus a sync per
shard cost ~9 ms of every decode token and most of a long prefill.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from . import config
from .checkpoint_keys import lookup_tensor, tensor_key_candidates
from .safetensors_index import _DTYPES, FilePool, TensorLoc, read_tensor_numpy

# Substrings that mark the huge n-gram shards (not the small PLE projections).
_NGRAM_SHARD_MARK = ".ple.ple_embedding.ngram_embedding.shard_"

# Checkpoint mass that never becomes resident Metal weights for text serving.
_SKIP_PREFIXES = (
    "vision_tower.",
    "model.visual.",
    "mtp.",
)


def ngram_shard_tensor_names(tensor_index: dict[str, TensorLoc]) -> set[str]:
    return {n for n in tensor_index if _NGRAM_SHARD_MARK in n}


def skipped_resident_tensor_names(tensor_index: dict[str, TensorLoc]) -> set[str]:
    """Tensors counted out of the resident backbone budget.

    Vision / MTP are dropped by sanitize for text runtime; n-gram shards are
    paged by patch_ple_ngram(). Both must leave the backbone figure.
    """
    out = set(ngram_shard_tensor_names(tensor_index))
    for name in tensor_index:
        bare = (
            name[len("language_model.") :]
            if name.startswith("language_model.")
            else name
        )
        if bare.startswith(_SKIP_PREFIXES) or name.startswith(_SKIP_PREFIXES):
            out.add(name)
    return out


def _infer_quant(
    weight_loc: TensorLoc, scales_loc: TensorLoc, dims: int
) -> tuple[int, int]:
    n_groups = int(scales_loc.shape[-1])
    if n_groups <= 0 or dims % n_groups != 0:
        raise ValueError(
            f"cannot infer PLE quant groups from scales {scales_loc.shape} dims={dims}"
        )
    group_size = dims // n_groups
    packed = int(weight_loc.shape[-1])
    bits = (packed * 32) // dims
    if bits not in (2, 3, 4, 5, 6, 8) or (dims * bits) % 32 != 0:
        raise ValueError(
            f"cannot infer PLE quant bits from weight {weight_loc.shape} dims={dims}"
        )
    return group_size, bits


def _bf16_array(np_u16: np.ndarray) -> mx.array:
    # Safetensors BF16 is stored as raw uint16; MLX reinterprets.
    return mx.array(np_u16).view(mx.bfloat16)


def _row_locs(loc: TensorLoc, rows: np.ndarray) -> list[tuple[int, TensorLoc]]:
    """(original_index_into_rows, TensorLoc of that row) pairs, sorted by offset."""
    n_rows = int(loc.shape[0])
    stride = loc.nbytes // n_rows
    row_shape = tuple(loc.shape[1:])
    out: list[tuple[int, TensorLoc]] = []
    for i, r in enumerate(rows.tolist()):
        r = int(r)
        if r < 0 or r >= n_rows:
            raise IndexError(f"PLE row {r} out of range for {loc.shape}")
        out.append(
            (
                i,
                TensorLoc(
                    file=loc.file,
                    dtype=loc.dtype,
                    shape=row_shape,
                    offset=loc.offset + r * stride,
                    nbytes=stride,
                ),
            )
        )
    out.sort(key=lambda t: t[1].offset)
    return out


class _DiskQuantShard:
    """One n-gram shard's weight/scales/biases on disk. Not an nn.Module."""

    __slots__ = ("weight", "scales", "biases", "group_size", "bits", "mode", "dims")

    def __init__(
        self,
        weight: TensorLoc,
        scales: TensorLoc,
        biases: TensorLoc | None,
        group_size: int,
        bits: int,
        dims: int,
        mode: str = "affine",
    ):
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.dims = dims

    def gather(self, pool: FilePool, row_ids: np.ndarray) -> mx.array:
        """Dequantized rows for `row_ids` (int64 host array), shape [N, dims]."""
        if row_ids.size == 0:
            return mx.zeros((0, self.dims), dtype=mx.float32)

        # Unique rows, then scatter back - avoids re-reading duplicates.
        uniq, inv = np.unique(row_ids, return_inverse=True)
        w_parts = []
        s_parts = []
        b_parts = [] if self.biases is not None else None
        for _, wloc in _row_locs(self.weight, uniq):
            w_parts.append(read_tensor_numpy(pool, wloc))
        for _, sloc in _row_locs(self.scales, uniq):
            s_parts.append(read_tensor_numpy(pool, sloc))
        if self.biases is not None:
            for _, bloc in _row_locs(self.biases, uniq):
                b_parts.append(read_tensor_numpy(pool, bloc))

        w = mx.array(np.stack(w_parts, axis=0))
        if self.scales.dtype == "BF16":
            s = _bf16_array(np.stack(s_parts, axis=0))
        else:
            s = mx.array(np.stack(s_parts, axis=0))
        b = None
        if b_parts is not None:
            if self.biases.dtype == "BF16":
                b = _bf16_array(np.stack(b_parts, axis=0))
            else:
                b = mx.array(np.stack(b_parts, axis=0))

        deq = mx.dequantize(
            w,
            scales=s,
            biases=b,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        # Map unique rows back to the caller's order. Force float32 so the
        # host scatter below can use numpy (bfloat16 has no PEP 3118 buffer).
        return deq[mx.array(inv)].astype(mx.float32)


class _Latch:
    """Counts reader jobs down; first error wins."""

    def __init__(self, n: int):
        self._left = n
        self._exc: BaseException | None = None
        self._lock = threading.Lock()
        self._ev = threading.Event()
        if n <= 0:
            self._ev.set()

    def done(self, exc: BaseException | None = None) -> None:
        with self._lock:
            if exc is not None and self._exc is None:
                self._exc = exc
            self._left -= 1
            if self._left > 0:
                return
        self._ev.set()

    def wait(self) -> None:
        self._ev.wait()
        if self._exc is not None:
            raise self._exc


def _pread_into(fd: int, offset: int, mv: memoryview) -> None:
    while len(mv) > 0:
        n = os.preadv(fd, [mv], offset)
        if n <= 0:
            raise IOError(f"short PLE read at {offset}")
        offset += n
        mv = mv[n:]


class _RowReader:
    """Parallel preads of many small row components."""

    def __init__(self, threads: int):
        self.threads = max(1, int(threads))
        self._native = None
        self._py: ThreadPoolExecutor | None = None
        if config.NATIVE_READ != "off":
            try:
                from . import native_read as _nr

                mod = _nr.get_module()
                if mod is not None:
                    self._native = _nr.NativeReadPool(self.threads, mod=mod)
            except Exception:  # noqa: BLE001 - python threads still work
                self._native = None

    def read(self, jobs: list[tuple[int, int, memoryview]]) -> None:
        if not jobs:
            return
        if self._native is not None:
            # The native pool caps one job at 32 reads; its threads spread them.
            chunks = [jobs[i : i + 32] for i in range(0, len(jobs), 32)]
            latch = _Latch(len(chunks))
            for chunk in chunks:
                self._native.submit_expert(chunk, latch)
            latch.wait()
            return
        n = min(self.threads, max(1, len(jobs) // 16))
        chunks = [jobs[i::n] for i in range(n)]
        if n == 1:
            for fd, off, mv in jobs:
                _pread_into(fd, off, mv)
            return
        if self._py is None:
            self._py = ThreadPoolExecutor(self.threads, thread_name_prefix="ple-read")

        def run(chunk):
            for fd, off, mv in chunk:
                _pread_into(fd, off, mv)

        for f in [self._py.submit(run, c) for c in chunks]:
            f.result()

    def close(self) -> None:
        if self._native is not None:
            self._native.shutdown()
            self._native = None
        if self._py is not None:
            self._py.shutdown(wait=False)
            self._py = None


def _np_dtype(st_dtype: str):
    return _DTYPES[st_dtype][0]


class _RowTable:
    """Flat per-shard offsets for one tensor kind (weight / scales / biases)."""

    def __init__(self, locs: list[TensorLoc], rows: int, pool: FilePool):
        self.dtype = locs[0].dtype
        self.row_shape = tuple(locs[0].shape[1:])
        self.stride = locs[0].nbytes // int(locs[0].shape[0])
        self.fds = [pool._fd(loc.file) for loc in locs]
        self.base = np.array([loc.offset for loc in locs], dtype=np.int64)
        self.rows = rows

    def uniform_with(self, locs: list[TensorLoc]) -> bool:
        return all(
            loc.dtype == self.dtype
            and tuple(loc.shape[1:]) == self.row_shape
            and int(loc.shape[0]) == self.rows
            and loc.nbytes // int(loc.shape[0]) == self.stride
            for loc in locs
        )

    def jobs(self, gids: np.ndarray, buf: np.ndarray, out: list) -> None:
        shard = gids // self.rows
        offs = self.base[shard] + (gids % self.rows) * self.stride
        fds = self.fds
        for i, (s, off) in enumerate(zip(shard.tolist(), offs.tolist())):
            out.append((fds[s], off, memoryview(buf[i])))

    def to_mx(self, buf: np.ndarray) -> mx.array:
        arr = buf.view(_np_dtype(self.dtype)).reshape(buf.shape[0], *self.row_shape)
        if self.dtype == "BF16":
            return _bf16_array(arr)
        return mx.array(arr)


class _FastGather:
    """Batched gather for shards that share one quant layout (always, in
    shipped checkpoints). Raises ValueError otherwise so the caller keeps the
    per-shard path. Not an nn.Module: its dicts must stay out of parameters().
    """

    def __init__(
        self,
        shards: list[_DiskQuantShard],
        rows: int,
        dim: int,
        pool: FilePool,
        cap: int,
        threads: int,
    ):
        s0 = shards[0] if shards else None
        if s0 is None:
            raise ValueError("no shards")
        for s in shards:
            if (
                s.group_size != s0.group_size
                or s.bits != s0.bits
                or s.mode != s0.mode
                or s.dims != s0.dims
                or (s.biases is None) != (s0.biases is None)
            ):
                raise ValueError("mixed PLE shard quant")
        kinds = {"weight": [s.weight for s in shards], "scales": [s.scales for s in shards]}
        if s0.biases is not None:
            kinds["biases"] = [s.biases for s in shards]
        self.tables: dict[str, _RowTable] = {}
        for name, locs in kinds.items():
            if any(int(loc.shape[0]) != rows for loc in locs):
                raise ValueError("uneven PLE shards")
            t = _RowTable(locs, rows, pool)
            if not t.uniform_with(locs):
                raise ValueError("mixed PLE row layout")
            self.tables[name] = t
        self.s0 = s0
        self.limit = len(shards) * rows
        self.dim = dim
        self.reader = _RowReader(threads)
        self.cap = max(0, int(cap))
        self.lru: OrderedDict[int, int] = OrderedDict()
        self.free = list(range(self.cap - 1, -1, -1))
        self.store = {
            name: np.empty((self.cap, t.stride), dtype=np.uint8)
            for name, t in self.tables.items()
        }
        self.hits = 0
        self.misses = 0

    def _rows(self, uniq: np.ndarray) -> dict[str, np.ndarray]:
        """Raw row bytes for each unique global id, cache first."""
        n = int(uniq.size)
        bufs = {
            name: np.empty((n, t.stride), dtype=np.uint8)
            for name, t in self.tables.items()
        }
        lru = self.lru
        hit_i, hit_s, miss_i = [], [], []
        for i, g in enumerate(uniq.tolist()):
            slot = lru.get(g)
            if slot is None:
                miss_i.append(i)
            else:
                lru.move_to_end(g)
                hit_i.append(i)
                hit_s.append(slot)
        self.hits += len(hit_i)
        self.misses += len(miss_i)
        if hit_i:
            hi, hs = np.array(hit_i), np.array(hit_s)
            for name, buf in bufs.items():
                buf[hi] = self.store[name][hs]
        if miss_i:
            mi = np.array(miss_i)
            gids = uniq[mi]
            parts = {
                name: np.empty((mi.size, t.stride), dtype=np.uint8)
                for name, t in self.tables.items()
            }
            jobs: list = []
            for name, t in self.tables.items():
                t.jobs(gids, parts[name], jobs)
            self.reader.read(jobs)
            for name, buf in bufs.items():
                buf[mi] = parts[name]
            self._admit(gids, parts)
        return bufs

    def _admit(self, gids: np.ndarray, parts: dict[str, np.ndarray]) -> None:
        if self.cap <= 0:
            return
        start = max(0, int(gids.size) - self.cap)
        lru, free = self.lru, self.free
        slots = []
        for g in gids[start:].tolist():
            if free:
                slot = free.pop()
            else:
                _, slot = lru.popitem(last=False)
            lru[g] = slot
            slots.append(slot)
        idx = np.array(slots)
        for name, part in parts.items():
            self.store[name][idx] = part[start:]

    def __call__(self, gid: mx.array) -> mx.array:
        flat = np.array(gid.reshape(-1), copy=False).astype(np.int64, copy=False)
        if flat.size == 0:
            return mx.zeros((*gid.shape, self.dim), dtype=mx.float32)
        if flat.min() < 0 or flat.max() >= self.limit:
            raise IndexError(f"PLE row out of range ({self.limit} rows)")
        uniq, inv = np.unique(flat, return_inverse=True)
        bufs = self._rows(uniq)
        t, s0 = self.tables, self.s0
        deq = mx.dequantize(
            t["weight"].to_mx(bufs["weight"]),
            scales=t["scales"].to_mx(bufs["scales"]),
            biases=t["biases"].to_mx(bufs["biases"]) if "biases" in t else None,
            group_size=s0.group_size,
            bits=s0.bits,
            mode=s0.mode,
        )
        out = deq[mx.array(inv.reshape(-1))].astype(mx.float32)
        return out.reshape(*gid.shape, self.dim)

    def close(self) -> None:
        self.reader.close()


class DiskShardedQuantEmbedding(nn.Module):
    """Drop-in for qwen4_exp `_ShardedEmbedding` that never materializes shards."""

    def __init__(
        self,
        shards: list[_DiskQuantShard],
        rows_per_shard: int,
        dim: int,
        pool: FilePool,
        row_cache: int | None = None,
        read_threads: int | None = None,
        fast: bool | None = None,
    ):
        super().__init__()
        self._shards = shards
        self.n_shards = len(shards)
        self.rows = rows_per_shard
        self.dim = dim
        self._pool = pool
        self._fast = None
        if config.PLE_FAST if fast is None else fast:
            try:
                self._fast = _FastGather(
                    shards,
                    rows_per_shard,
                    dim,
                    pool,
                    config.PLE_ROW_CACHE if row_cache is None else row_cache,
                    config.PLE_READ_THREADS if read_threads is None else read_threads,
                )
            except ValueError:
                self._fast = None
        # Freeze so nothing tries to train / quantize us; no parameters live here.
        self.freeze()

    def __call__(self, gid: mx.array) -> mx.array:
        if self._fast is not None:
            return self._fast(gid)
        # Indices must land on the host so we can decide which shards to touch.
        flat = np.array(gid.reshape(-1), copy=False).astype(np.int64, copy=False)
        shard_of = flat // self.rows
        row_of = flat % self.rows
        touched = np.unique(shard_of)
        out = np.zeros((flat.size, self.dim), dtype=np.float32)
        for s in touched.tolist():
            s = int(s)
            if s < 0 or s >= self.n_shards:
                raise IndexError(f"PLE shard {s} out of range ({self.n_shards})")
            sel = np.nonzero(shard_of == s)[0]
            rows = row_of[sel]
            emb = self._shards[s].gather(self._pool, rows)
            mx.eval(emb)
            out[sel] = np.array(emb, copy=False)
        return mx.array(out).reshape(*gid.shape, self.dim)


def _shard_paths(module_path: str, shard_i: int) -> dict[str, str]:
    base = f"{module_path}.shard_{shard_i}"
    return {
        "weight": f"{base}.weight",
        "scales": f"{base}.scales",
        "biases": f"{base}.biases",
    }


def find_ngram_embedding_modules(model) -> list[tuple[str, nn.Module]]:
    found = []
    for path, module in model.named_modules():
        if not path.endswith("ple_embedding.ngram_embedding"):
            continue
        if not hasattr(module, "n_shards"):
            continue
        found.append((path, module))
    return found


def patch_ple_ngram(
    model,
    tensor_index: dict[str, TensorLoc],
    pool: FilePool | None = None,
) -> FilePool | None:
    """Replace resident n-gram shards with disk gathers. Returns the file pool
    kept alive on the model, or None when there is nothing to patch."""
    modules = find_ngram_embedding_modules(model)
    if not modules:
        return None

    pool = pool or FilePool(nocache=True)
    patched = 0
    for path, module in modules:
        n_shards = int(getattr(module, "n_shards", 0) or 0)
        rows = int(getattr(module, "rows", 0) or 0)
        dim = int(getattr(module, "dim", 0) or 0)
        if n_shards <= 0 or rows <= 0 or dim <= 0:
            continue

        shards: list[_DiskQuantShard] = []
        for i in range(n_shards):
            keys = _shard_paths(path, i)
            weight = lookup_tensor(tensor_index, keys["weight"])
            scales = lookup_tensor(tensor_index, keys["scales"])
            if weight is None or scales is None:
                # Fall back: maybe sanitize already rewrote keys in the index
                # under a different layout; leave this module alone.
                shards = []
                break
            biases = lookup_tensor(tensor_index, keys["biases"])
            group_size, bits = _infer_quant(weight, scales, dim)
            shards.append(
                _DiskQuantShard(
                    weight, scales, biases, group_size, bits, dim, mode="affine"
                )
            )
        if len(shards) != n_shards:
            continue

        replacement = DiskShardedQuantEmbedding(shards, rows, dim, pool)
        parent = model
        parts = path.split(".")
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        setattr(parent, parts[-1], replacement)
        patched += 1

    if patched:
        # Keep the pool open for the life of the model (shard gathers need it).
        model._pagedmoe_ple_pool = pool
        return pool
    pool.close()
    return None
