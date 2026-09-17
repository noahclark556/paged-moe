# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Read individual experts straight out of standard safetensors checkpoints.

Why this exists
---------------
A quantized MLX MoE checkpoint stores each layer's experts *stacked* in a few
big tensors, e.g.

    model.layers.7.mlp.switch_mlp.gate_proj.weight   shape [n_experts, out, in/8]
    model.layers.7.mlp.switch_mlp.gate_proj.scales   shape [n_experts, out, in/gs]
    model.layers.7.mlp.switch_mlp.gate_proj.biases   shape [n_experts, out, in/gs]
    ... same for up_proj and down_proj ...

Safetensors tensors are contiguous and row-major, so expert `e` of a stacked
tensor is a single contiguous byte range whose offset we can compute from the
file header alone.  That means we can stream one expert with one pread-style
read, *without converting the checkpoint into any custom format*.  The model
directory on the SSD stays a plain, unmodified MLX checkpoint.

The safetensors format is simply:
    [8 bytes little-endian u64: header_size] [header_size bytes JSON] [raw data]
Each JSON entry: {"dtype": "F16", "shape": [...], "data_offsets": [begin, end]}
with offsets relative to the end of the header.
"""

from __future__ import annotations

import fcntl
import glob
import json
import os
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# safetensors dtype -> (numpy dtype used for raw reading, itemsize)
# bfloat16 has no numpy dtype: we read it as uint16 and reinterpret in MLX.
_DTYPES = {
    "F64": (np.float64, 8),
    "F32": (np.float32, 4),
    "F16": (np.float16, 2),
    "BF16": (np.uint16, 2),  # reinterpreted as bfloat16 by the consumer
    "I64": (np.int64, 8),
    "I32": (np.int32, 4),
    "I16": (np.int16, 2),
    "I8": (np.int8, 1),
    "U8": (np.uint8, 1),
    "U32": (np.uint32, 4),
    "BOOL": (np.bool_, 1),
}


@dataclass(frozen=True)
class TensorLoc:
    """Where one tensor lives on disk."""

    file: str
    dtype: str  # safetensors dtype string
    shape: tuple
    offset: int  # absolute byte offset of tensor start within file
    nbytes: int

    @property
    def is_bf16(self) -> bool:
        return self.dtype == "BF16"

    def expert_slice(self, expert_id: int) -> "TensorLoc":
        """Location of experts[expert_id] within a stacked [E, ...] tensor."""
        n_experts = self.shape[0]
        stride = self.nbytes // n_experts
        return TensorLoc(
            file=self.file,
            dtype=self.dtype,
            shape=tuple(self.shape[1:]),
            offset=self.offset + expert_id * stride,
            nbytes=stride,
        )


class ExpertLocator:
    """Resolves (expert_id) -> TensorLoc for one component of one MoE layer.

    Checkpoints come in two layouts and both are streamable without any
    conversion:
      - stacked:    one big tensor  "<glu>.gate_proj.weight" [E, out, in]
                    -> expert e is a contiguous slice (StackedLocator)
      - per-expert: separate tensors "<mlp>.experts.<e>.gate_proj.weight"
                    -> expert e is its own tensor (DirectLocator)
    """

    def loc(self, expert_id: int) -> TensorLoc:  # pragma: no cover
        raise NotImplementedError


class StackedLocator(ExpertLocator):
    def __init__(self, stacked: TensorLoc):
        self.stacked = stacked

    def loc(self, expert_id: int) -> TensorLoc:
        return self.stacked.expert_slice(expert_id)

    def span(self, first: int, count: int) -> TensorLoc:
        """Location of experts[first : first+count] - one contiguous range.

        This is what makes prefill fast: consecutive experts of a stacked
        tensor are physically adjacent on disk, so a whole run can be read
        with a single large pread at sequential-read bandwidth instead of
        hundreds of small scattered reads.
        """
        n_experts = self.stacked.shape[0]
        stride = self.stacked.nbytes // n_experts
        return TensorLoc(
            file=self.stacked.file,
            dtype=self.stacked.dtype,
            shape=(count,) + tuple(self.stacked.shape[1:]),
            offset=self.stacked.offset + first * stride,
            nbytes=stride * count,
        )


class DirectLocator(ExpertLocator):
    def __init__(self, per_expert: dict[int, TensorLoc]):
        self.per_expert = per_expert

    def loc(self, expert_id: int) -> TensorLoc:
        return self.per_expert[expert_id]


def read_headers(model_dir: str | Path) -> dict[str, TensorLoc]:
    """Map every tensor name in the checkpoint to its on-disk location."""
    index: dict[str, TensorLoc] = {}
    files = sorted(glob.glob(str(Path(model_dir) / "model*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no model*.safetensors in {model_dir}")
    for path in files:
        with open(path, "rb") as f:
            (header_size,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_size))
        data_start = 8 + header_size
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            begin, end = meta["data_offsets"]
            index[name] = TensorLoc(
                file=path,
                dtype=meta["dtype"],
                shape=tuple(meta["shape"]),
                offset=data_start + begin,
                nbytes=end - begin,
            )
    return index


class FilePool:
    """One shared read-only fd per checkpoint file; reads via pread().

    With nocache=True (the default, macOS F_NOCACHE) reads bypass the kernel
    page cache entirely.  This matters enormously on a memory-constrained
    machine: a prefill pass over a long prompt reads nearly the whole expert
    mass (tens of GB), and with mmap/page-cached reads every one of those
    bytes also landed in the kernel's file cache *in addition to* our own
    expert LRU.  That double-caching is what drove a 48 GB Mac into swap.
    With F_NOCACHE, expert bytes live in exactly one place: the ExpertCache.

    pread() is positional and thread-safe on a shared fd, so the reader
    thread pool needs no per-thread state.
    """

    def __init__(self, nocache: bool = True):
        self._fds: dict[str, int] = {}
        self._nocache = nocache
        self._lock = threading.Lock()

    def _fd(self, path: str) -> int:
        fd = self._fds.get(path)
        if fd is None:
            with self._lock:
                fd = self._fds.get(path)
                if fd is None:
                    fd = os.open(path, os.O_RDONLY)
                    if self._nocache and hasattr(fcntl, "F_NOCACHE"):
                        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                    self._fds[path] = fd
        return fd

    def read_bytes(self, path: str, offset: int, nbytes: int) -> bytes:
        fd = self._fd(path)
        chunks = []
        remaining = nbytes
        pos = offset
        while remaining > 0:
            b = os.pread(fd, remaining, pos)
            if not b:
                raise IOError(f"short read at {pos} in {path}")
            chunks.append(b)
            pos += len(b)
            remaining -= len(b)
        return chunks[0] if len(chunks) == 1 else b"".join(chunks)

    def read_into(self, path: str, offset: int, buf) -> None:
        """pread directly into an existing writable buffer (no allocation).

        This is what makes read-buffer reuse possible: os.pread returns a
        fresh bytes object per call, which on big-expert models allocates
        (and zero-fill page-faults) gigabytes per decoded token. preadv into
        a recycled buffer touches only warm pages.
        """
        fd = self._fd(path)
        view = memoryview(buf)
        pos = offset
        while len(view) > 0:
            n = os.preadv(fd, [view], pos)
            if n <= 0:
                raise IOError(f"short read at {pos} in {path}")
            pos += n
            view = view[n:]

    def close(self):
        with self._lock:
            for fd in self._fds.values():
                os.close(fd)
            self._fds.clear()


def read_tensor_numpy(pool: FilePool, loc: TensorLoc) -> np.ndarray:
    """Read one tensor (or expert slice) out of the checkpoint into RAM.

    ``EXPERT_STREAM_READ_DELAY_US_PER_MB`` (device sandbox) adds sleep so a
    fast host SSD can stand in for a slower target disk. 0 / unset = off.
    """
    delay_us = _read_delay_us_per_mb()
    if delay_us > 0 and loc.nbytes > 0:
        time.sleep(loc.nbytes * (delay_us / 1e6) / (1 << 20))
    data = pool.read_bytes(loc.file, loc.offset, loc.nbytes)
    np_dtype, _itemsize = _DTYPES[loc.dtype]
    arr = np.frombuffer(data, dtype=np_dtype)
    return arr.reshape(loc.shape)


_READ_DELAY_US_PER_MB: float | None = None


def _read_delay_us_per_mb() -> float:
    global _READ_DELAY_US_PER_MB
    if _READ_DELAY_US_PER_MB is None:
        raw = os.environ.get("EXPERT_STREAM_READ_DELAY_US_PER_MB", "").strip()
        try:
            _READ_DELAY_US_PER_MB = float(raw) if raw else 0.0
        except ValueError:
            _READ_DELAY_US_PER_MB = 0.0
    return _READ_DELAY_US_PER_MB
