# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Slot-addressed, Metal-backed storage for uniformly shaped experts.

Why this exists
---------------
The per-expert cache stores each expert's components as their own mx arrays,
which makes the decode path pay twice for the same bytes:

  * **A copy.** Workers pread into numpy buffers, then the GPU thread runs
    `mx.array(np_buf)` - a host->device copy of the whole per-token miss mass
    (~65 ms/token on Qwen3-235B).
  * **Per-expert dispatch.** Separate arrays can only be fed to
    `quantized_matmul` one expert at a time: 3 calls x experts x layers is
    ~1300 tiny kernels per token (~112 ms/token at 235B shapes, measured by
    `bench/gather_qmm_probe.py`). `mx.gather_qmm` does all of a layer's
    experts in one call and is 5.3x faster - but it needs the experts stacked,
    and stacking them per call costs exactly what it saves.

Both disappear if experts *live* in one contiguous tensor. A slab is
`[slots, *expert_shape]` per component; slot i holds some (layer, expert) and
is addressed by index, so:

  * reads land in their final location - `pread` writes straight into the
    slab's bytes, so there is no copy and no materialization step at all
  * compute is one `gather_qmm` per projection with `rhs_indices` = the slot
    of each row's expert (see `streaming._run_slab`)
  * memory is allocated once at load. No per-token allocation, no zero-fill
    page faults, no `mx.clear_cache()` cadence to tune - the churn that made
    big-expert models thrash the allocator simply has no source anymore.

The hack, and why it is safe
----------------------------
Writing into MLX-owned memory through `np.asarray(slab)` is outside MLX's
contract: it hands out a writeable view of unified memory and has no idea the
contents changed. Three things make it work, all verified by
`bench/slab_probe.py` (which asserts bit-identical `gather_qmm` output):

  1. Apple Silicon is unified memory, so a CPU write to the buffer is visible
     to the GPU with no transfer, and `pread` can target it directly.
  2. Slabs are allocated and evaluated once, up front. Every later mx op on
     them (`.view()`, `gather_qmm`) reads the buffer at eval time, so writes
     that land before the read are seen.
  3. A slot is only written while it holds no live expert. Freed slots go
     through quarantine and are not reused until a token boundary, by which
     point the previous token's lazy graph has been forced (every MoE layer
     syncs on its router, and the sampler forces the last one). See
     `ExpertCache.recycle_slots`.

numpy has no bfloat16, so BF16 components are stored as uint16 and exposed to
compute through a persistent `.view(mx.bfloat16)` - metadata only (0.025 ms on
a 9 GB slab), and it reflects writes made after the view was created.
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx

# safetensors dtype -> (numpy storage, mx storage, mx compute)
#
# Storage and compute differ only for bfloat16: numpy cannot name it, so the
# slab is uint16 and compute sees a persistent bit-cast view.
_STORAGE = {
    "U32": (np.uint32, mx.uint32, mx.uint32),
    "I32": (np.int32, mx.int32, mx.int32),
    "U16": (np.uint16, mx.uint16, mx.uint16),
    "I16": (np.int16, mx.int16, mx.int16),
    "U8": (np.uint8, mx.uint8, mx.uint8),
    "I8": (np.int8, mx.int8, mx.int8),
    "F16": (np.float16, mx.float16, mx.float16),
    "F32": (np.float32, mx.float32, mx.float32),
    "BF16": (np.uint16, mx.uint16, mx.bfloat16),
}


def supported_dtype(dtype: str) -> bool:
    return dtype in _STORAGE


class SlabStore:
    """One slab per component, each `[slots, *expert_shape]`.

    Dumb storage: slot lifetime, LRU and quarantine are the cache's business
    (see `ExpertCache`). Must be constructed on the MLX/GPU thread; after that
    `dest()` is a plain numpy view and is safe to write from read workers.
    """

    def __init__(self, specs: dict[str, tuple[tuple, str, int]], slots: int):
        """`specs` maps component name -> (per-expert shape, dtype, nbytes)."""
        if slots < 1:
            raise ValueError(f"slab needs at least one slot, got {slots}")
        self.slots = int(slots)
        self.nbytes = 0
        self._storage: dict[str, mx.array] = {}
        self._compute: dict[str, mx.array] = {}
        self._bytes: dict[str, np.ndarray] = {}
        self._stride: dict[str, int] = {}

        for name, (shape, dtype, nbytes) in specs.items():
            np_dt, mx_dt, comp_dt = _STORAGE[dtype]
            expect = int(np.prod(shape)) * np.dtype(np_dt).itemsize
            if expect != nbytes:
                # A stride mismatch would silently read a shifted tensor.
                raise ValueError(
                    f"{name}: checkpoint says {nbytes} B/expert, "
                    f"{shape}x{dtype} implies {expect} B"
                )
            slab = mx.zeros((self.slots, *shape), dtype=mx_dt)
            mx.eval(slab)

            flat = np.asarray(slab)
            if not flat.flags.writeable or not flat.flags.c_contiguous:
                raise RuntimeError(
                    f"{name}: slab view is not a writeable contiguous view "
                    "(MLX no longer exposes its buffer; slab path unusable)"
                )
            self._storage[name] = slab
            self._bytes[name] = flat.reshape(-1).view(np.uint8)
            self._stride[name] = int(nbytes)
            view = slab.view(comp_dt) if comp_dt is not mx_dt else slab
            if view is not slab:
                mx.eval(view)
            self._compute[name] = view
            self.nbytes += slab.nbytes

        self.names = tuple(specs)
        self.specs = dict(specs)  # kept so the store can be rebuilt verbatim
        self.expert_bytes = sum(self._stride.values())
        # Per-slot pread destinations in names order, built once.
        # Hot path used to call dest() nine times/expert (~2000 numpy slices
        # per token on 235B, GIL-held; gil_probe.py). Indexing here instead.
        # memoryview so os.preadv takes it directly; a short-read slice
        # allocates nothing.
        self.slot_dests: tuple[tuple[memoryview, ...], ...] = tuple(
            tuple(
                memoryview(
                    self._bytes[name][slot * self._stride[name]
                                      : (slot + 1) * self._stride[name]]
                )
                for name in self.names
            )
            for slot in range(self.slots)
        )
        self.strides = tuple(self._stride[name] for name in self.names)

    def release(self) -> None:
        """Drop every slab so Metal can hand the memory back.

        Callers must have discarded all slot references first (no graph may
        still read a slab). Used when a big prefill needs the memory for
        activations - see `ExpertCache.enter_prefill`.
        """
        # Destination views alias slab memory - clear first so a surviving
        # memoryview cannot keep the buffer alive or receive a post-free pread.
        self.slot_dests = ()
        self._storage.clear()
        self._compute.clear()
        self._bytes.clear()
        self.nbytes = 0

    def has(self, name: str) -> bool:
        return name in self._compute

    def dest(self, name: str, slot: int) -> np.ndarray:
        """Writeable byte range for one component of one slot (pread target)."""
        stride = self._stride[name]
        return self._bytes[name][slot * stride : (slot + 1) * stride]

    def compute(self, name: str) -> mx.array:
        """The full `[slots, ...]` tensor to hand to gather_qmm."""
        return self._compute[name]

    def expert(self, name: str, slot: int) -> mx.array:
        """One slot as a standalone array, for the per-expert fallback path."""
        return self._compute[name][slot]
