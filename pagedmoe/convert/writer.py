# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Safetensors shards with the header written up front.

Every tensor's shape, dtype and offset is known from the plan, so each shard
gets its final header and size before any data exists, and tensors land at
fixed offsets in whatever order they finish. A stopped run picks up where it
left off by rewriting only the layers it didn't finish.

Writes go through F_NOCACHE in MAX_IO_BYTES pieces, so a 400 GB output
doesn't sit in the page cache on its way to disk.
"""

from __future__ import annotations

import fcntl
import json
import os
from dataclasses import dataclass
from pathlib import Path

from ..safetensors_index import MAX_IO_BYTES

ST_NAMES = {
    "bool": "BOOL", "uint8": "U8", "int8": "I8", "uint16": "U16", "int16": "I16",
    "uint32": "U32", "int32": "I32", "uint64": "U64", "int64": "I64",
    "float16": "F16", "bfloat16": "BF16", "float32": "F32", "float64": "F64",
}


class WriteError(RuntimeError):
    pass


@dataclass
class Slot:
    shard: int
    start: int  # absolute file offset of the first data byte
    nbytes: int
    shape: tuple
    dtype: str


def shard_names(n: int) -> list:
    if n == 1:
        return ["model.safetensors"]
    return [f"model-{i + 1:05d}-of-{n:05d}.safetensors" for i in range(n)]


def assign(layout, shard_bytes: int) -> list:
    """Split the layout (in write order) into shards of about shard_bytes."""
    shards: list[list] = [[]]
    size = 0
    for t in layout:
        if shards[-1] and size + t.nbytes > shard_bytes:
            shards.append([])
            size = 0
        shards[-1].append(t)
        size += t.nbytes
    return shards


def header_bytes(tensors) -> bytes:
    head: dict = {"__metadata__": {"format": "mlx"}}
    off = 0
    for t in tensors:
        head[t.name] = {"dtype": ST_NAMES[t.dtype], "shape": list(t.shape),
                        "data_offsets": [off, off + t.nbytes]}
        off += t.nbytes
    raw = json.dumps(head, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    return len(raw).to_bytes(8, "little") + raw


class ShardWriter:
    def __init__(self, out_dir: str | Path, layout, shard_bytes: int):
        self.dir = Path(out_dir)
        groups = assign(layout, max(1, int(shard_bytes)))
        self.names = shard_names(len(groups))
        self.slots: dict[str, Slot] = {}
        self.sizes: list[int] = []
        self.headers: list[bytes] = []
        for i, group in enumerate(groups):
            head = header_bytes(group)
            off = len(head)
            for t in group:
                self.slots[t.name] = Slot(i, off, t.nbytes, tuple(t.shape), t.dtype)
                off += t.nbytes
            self.headers.append(head)
            self.sizes.append(off)
        self._fds: dict[int, int] = {}
        self._dirty: set[int] = set()

    @property
    def paths(self) -> list:
        return [self.dir / n for n in self.names]

    def weight_map(self) -> dict:
        return {name: self.names[s.shard] for name, s in sorted(self.slots.items())}

    def matches_disk(self) -> bool:
        """Shards on disk have this layout's sizes and headers (for resume)."""
        for p, size, head in zip(self.paths, self.sizes, self.headers):
            try:
                if p.stat().st_size != size:
                    return False
                with open(p, "rb") as f:
                    if f.read(len(head)) != head:
                        return False
            except OSError:
                return False
        return True

    def create(self) -> None:
        """Fresh shards: header in place, data area sized (sparse until written)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(self.paths):
            fd = os.open(p, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                os.pwrite(fd, self.headers[i], 0)
                os.ftruncate(fd, self.sizes[i])
            finally:
                os.close(fd)

    def _fd(self, shard: int) -> int:
        fd = self._fds.get(shard)
        if fd is None:
            fd = os.open(self.paths[shard], os.O_RDWR)
            if hasattr(fcntl, "F_NOCACHE"):
                fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
            self._fds[shard] = fd
        return fd

    def put(self, name: str, arr) -> int:
        """Write one evaluated mx.array (or numpy array) to its slot."""
        slot = self.slots.get(name)
        if slot is None:
            raise WriteError(f"{name} isn't in the planned layout")
        buf = _bytes_of(arr)
        shape = tuple(int(s) for s in arr.shape)
        dtype = str(arr.dtype).replace("mlx.core.", "")
        if shape != slot.shape or dtype != slot.dtype or buf.nbytes != slot.nbytes:
            raise WriteError(f"{name}: got {dtype} {shape} ({buf.nbytes} bytes), the plan says "
                             f"{slot.dtype} {slot.shape} ({slot.nbytes} bytes)")
        fd = self._fd(slot.shard)
        done = 0
        while done < slot.nbytes:
            n = os.pwrite(fd, buf[done: done + MAX_IO_BYTES], slot.start + done)
            if n <= 0:
                raise WriteError(f"short write on {self.names[slot.shard]}")
            done += n
        self._dirty.add(slot.shard)
        return slot.nbytes

    def sync(self) -> None:
        for s in sorted(self._dirty):
            os.fsync(self._fds[s])
        self._dirty.clear()

    def close(self) -> None:
        self.sync()
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def write_index(self, extra_meta: dict | None = None) -> None:
        if len(self.names) == 1:
            return
        meta = {"total_size": sum(s.nbytes for s in self.slots.values())}
        meta.update(extra_meta or {})
        idx = {"metadata": meta, "weight_map": self.weight_map()}
        (self.dir / "model.safetensors.index.json").write_text(json.dumps(idx, indent=4) + "\n")


def _bytes_of(arr) -> memoryview:
    import numpy as np

    if not isinstance(arr, np.ndarray):
        import mlx.core as mx

        if arr.dtype == mx.bfloat16:
            arr = arr.view(mx.uint16)
        arr = np.array(arr, copy=False)
    arr = np.ascontiguousarray(arr)
    return memoryview(arr).cast("B")


def verify(out_dir: str | Path, writer: ShardWriter) -> list:
    """Headers on disk against the plan. Returns problems, empty when clean."""
    from ..safetensors_index import read_headers

    problems = []
    try:
        locs = read_headers(out_dir)
    except Exception as e:  # noqa: BLE001
        return [f"couldn't read the output headers: {e}"]
    for name, slot in writer.slots.items():
        loc = locs.get(name)
        if loc is None:
            problems.append(f"{name} is missing")
            continue
        if tuple(loc.shape) != slot.shape or loc.nbytes != slot.nbytes:
            problems.append(f"{name}: {tuple(loc.shape)} on disk, {slot.shape} planned")
    extra = set(locs) - set(writer.slots)
    if extra:
        problems.append(f"{len(extra)} unplanned tensors, e.g. {sorted(extra)[:2]}")
    return problems
