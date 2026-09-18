# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
ExpertCache: keeps recently-used experts in unified (CPU/GPU) memory.

Design
------
- One cache is shared by all MoE layers.  A cache key is (layer_key, expert_id)
  where layer_key is the module path, e.g. "model.layers.7.mlp.switch_mlp".
- Values are dicts of ready-to-use MLX arrays for one expert.
- Eviction is LRU by *byte budget*.  mx.clear_cache() is called (throttled)
  after eviction bursts so Metal actually returns pages to the OS.
- Worker threads only produce numpy arrays; mx.array happens on the GPU thread
  (MLX GPU streams are thread-local - touching mx on a worker aborts).
- Reads go through FilePool with F_NOCACHE: expert bytes exist in exactly one
  place in RAM (this cache), never also in the kernel page cache.

Install modes
-------------
  "lru"   normal: insert, evicting LRU entries over budget (decode path).
  "soft"  insert only while there is free room - never evict (prefill path:
          a long prompt touches nearly every expert; hard-installing them
          would churn the whole cache every layer for zero benefit).
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed

import numpy as np

import mlx.core as mx

from . import config
from .config import _env
from .slab import SlabStore, supported_dtype
from .safetensors_index import (
    _DTYPES,
    DirectLocator,
    ExpertLocator,
    FilePool,
    StackedLocator,
)

# safetensors dtype string -> numpy dtype for viewing pooled read buffers.
_LOC_DTYPES = {k: v[0] for k, v in _DTYPES.items()}

# Cap speculative numpy staging so prefetch can't balloon RAM. Bytes is the
# honest unit (one expert is ~1.6 MB on qwen3-next and ~13 MB on GLM-Air), with
# a count cap as a backstop against pathological tiny-expert models.
_MAX_RAW_READY = 512

# Frequency counter ceiling: high enough to protect the always-on experts, low
# enough that a stale favourite ages out within a few eviction sweeps.
_FREQ_MAX = 8

# Fallback clear_cache() cadence when the owner doesn't size one (see
# ExpertCache(clear_bytes=...)). Clearing on every eviction thrashes the
# allocator during heavy streaming.
_CLEAR_THRESHOLD = 2 << 30


# Components at least this big are byte-sliced across multiple parallel preads
# so one cold read never serializes behind a single thread. Env-overridable
# (EXPERT_STREAM_SLICE_MB): small slices minimize one cold expert's latency,
# but on a saturated pipe (big-expert decode) every slice is a Future plus a
# GIL window, and the scheduling overhead competes with the GPU thread.
_SLICE_BYTES = int(_env("EXPERT_STREAM_SLICE_MB", 2, float) * (1 << 20))

# Prefill reads whole layers, so it has no latency to minimize and plenty of
# parallelism already (many spans, many components, two groups in flight). Its
# scarce resource is bytes per syscall / per GIL window, so it slices coarsely.
_PREFILL_SLICE_BYTES = config.PREFILL_SLICE_BYTES

# Slot reads only fan a component to the slice pool at or above this size;
# smaller ones read inline on the expert worker. See _read_slab_inner.
# 0 = fan everything; a huge value = fully inline. Both kept for A/B.
_FANOUT_BYTES = int(_env("EXPERT_STREAM_FANOUT_MB", 1.0, float) * (1 << 20))


class _SlotLatch:
    """Completion counter for the component reads of one expert.

    A Future per component shipped first and was too expensive: nine
    components x ~250 misses/token is ~2300 Futures, each with a work item,
    queue put, and lock. bench/mainthread_profile.py counted ~10700 lock
    acquisitions/token; bench/gil_probe.py shows main-thread python costing
    up to 74% of read bandwidth.

    Nine reads share one counter; the single Future is the one `_inflight` /
    `add_done_callback` already need. First exception wins; later components
    still count down so a failed read cannot wedge a fetch.
    """

    __slots__ = ("_entry", "_exc", "_fut", "_left", "_lock", "_on_done")

    def __init__(self, n: int, fut: Future, entry: dict, on_done=None):
        self._left = n
        self._lock = threading.Lock()
        self._fut = fut
        self._entry = entry
        self._exc: BaseException | None = None
        self._on_done = on_done

    def done(self, exc: BaseException | None = None) -> None:
        with self._lock:
            if exc is not None and self._exc is None:
                self._exc = exc
            self._left -= 1
            if self._left > 0:
                return
            failure = self._exc
        if self._on_done is not None:
            self._on_done()
        # Outside the lock: resolves the expert Future; its callback takes
        # the cache lock (_attach_staging).
        if failure is not None:
            self._fut.set_exception(failure)
        else:
            self._fut.set_result(self._entry)


class _ReadPool:
    """Persistent pread workers fed by a plain queue.

    Not a ThreadPoolExecutor: submitting one allocates a Future and work
    item per job, and decode submits thousands per token (see _SlotLatch).
    A job here is a 4-tuple on a SimpleQueue (C-level, no python lock).

    Sized independently of the expert-level reader count because this *is*
    the queue depth: miss threads push reads straight in. disk_ceiling.py
    needs ~qd16 to saturate on 3 MB requests and still gains at qd64 on
    0.2 MB ones, so the default is deeper than the old 16.
    """

    def __init__(self, threads: int):
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self.threads = max(1, int(threads))
        # Per-worker busy flag around the pread. An observer samples these
        # to tell "drive saturated" from "drive idle waiting on python"
        # (bench/drive_idle.py). Only worker i writes busy[i], so no lock.
        self.busy = bytearray(self.threads)
        for i in range(self.threads):
            threading.Thread(
                target=self._run, args=(i,), name=f"expert-read-{i}", daemon=True
            ).start()

    def _run(self, idx: int) -> None:
        get = self._q.get
        busy = self.busy
        while True:
            job = get()
            if job is None:
                return
            fd, offset, mv, latch = job
            busy[idx] = 1
            try:
                _pread_full(fd, offset, mv)
            except BaseException as e:  # noqa: BLE001 - forwarded to the Future
                busy[idx] = 0
                latch.done(e)
            else:
                busy[idx] = 0
                latch.done()

    def submit(self, fd: int, offset: int, mv: memoryview, latch: _SlotLatch) -> None:
        self._q.put((fd, offset, mv, latch))

    def shutdown(self) -> None:
        """Stop the workers. Unused in production (`ga` owns one process per
        model), but tests that build caches in a loop need it to avoid leaks."""
        for _ in range(self.threads):
            self._q.put(None)


def _pread_full(fd: int, offset: int, mv: memoryview) -> None:
    """pread `mv`'s full length at `offset`. One syscall in the normal case.

    Separate from FilePool.read_into, which resolves path->fd and rebuilds a
    memoryview each call. The miss path already has both from the read plan,
    and this runs thousands of times per token.
    """
    while True:
        n = os.preadv(fd, [mv], offset)
        if n <= 0:
            raise OSError(f"short read at {offset} (fd {fd})")
        if n >= len(mv):
            return
        mv = mv[n:]
        offset += n


# Left free when sizing a slab rebuild: the next token's activations, the KV
# the rest of this turn will append, and Metal's own recycled-buffer pool.
_SLAB_MARGIN = 3 << 30


class BufferPool:
    """Recycled numpy read buffers, keyed by exact size.

    Expert tensors come in a handful of fixed sizes per model, so a freed
    buffer is a perfect fit for the next read of the same component. Without
    this, every miss allocates fresh memory that the kernel must zero-fill
    page by page - on a model that misses 2+ GB per decoded token that churn
    rivals the disk time itself.
    """

    def __init__(self, cap_bytes: int = 3 << 30):
        self._free: dict[int, list[np.ndarray]] = defaultdict(list)
        self._free_bytes = 0
        self._cap = cap_bytes
        self._lock = threading.Lock()

    def acquire(self, nbytes: int) -> np.ndarray:
        with self._lock:
            lst = self._free.get(nbytes)
            if lst:
                self._free_bytes -= nbytes
                return lst.pop()
        return np.empty(nbytes, dtype=np.uint8)

    def release(self, bufs) -> None:
        with self._lock:
            for buf in bufs:
                if self._free_bytes + buf.nbytes > self._cap:
                    continue  # over cap: let the GC have it
                self._free[buf.nbytes].append(buf)
                self._free_bytes += buf.nbytes

    def clear(self) -> None:
        with self._lock:
            self._free.clear()
            self._free_bytes = 0


def _release_raw(pool: BufferPool, raw: dict) -> None:
    """Return a raw entry's pooled buffers. Idempotent (pop-once)."""
    bufs = raw.pop("__bufs__", None)
    if bufs:
        pool.release(bufs)


def _materialize(raw: dict) -> dict:
    """numpy -> mx. MUST run on the MLX/GPU thread (usually main).

    This copy looks like the obvious thing to remove - it is ~40% of an
    incremental prefill's wall time (`bench/prefill_cost.py`), and reading
    straight into MLX-backed buffers the way `slab.py` does makes it disappear
    from the accounting entirely (16.7s -> 0.2s, measured, byte-identical).

    It buys nothing. Total prefill time did not move (41.4s -> 42.2s): the bytes
    still have to reach a contiguous GPU-usable array, so MLX just does the same
    copy lazily at eval and it reappears under compute. Don't re-attempt this
    without a plan for the underlying data movement, not its accounting.
    """
    if "__slot__" in raw:
        # Slab read: the bytes were pread straight into their final slot in
        # Metal-backed memory, so there is no copy to make and nothing to
        # evaluate. The entry is just the slot number; compute addresses it by
        # index (see streaming._run_slab).
        return raw
    out = {"__nbytes__": raw["__nbytes__"]}
    for name, value in raw.items():
        if name in ("__nbytes__", "__bufs__"):
            continue
        np_arr, is_bf16 = value
        arr = mx.array(np_arr)
        if is_bf16:
            arr = arr.view(mx.bfloat16)
        out[name] = arr
    return out


def _eval_experts(entries: list[dict]):
    arrays = [
        v
        for e in entries
        for k, v in e.items()
        if k != "__nbytes__" and isinstance(v, mx.array)
    ]
    if arrays:
        mx.eval(*arrays)


# How many LRU-tail keys the residency advisor may reprieve in one eviction
# pass. Bounded so a cache of uniformly hot experts still makes progress.
_PROTECT_SCAN = 32


class _Begun:
    """In-flight state for one fetch: resolved hits + pending disk reads."""

    __slots__ = ("hits", "raw", "futs")

    def __init__(self):
        self.hits: dict[int, dict] = {}
        self.raw: list[tuple[int, dict]] = []
        self.futs: list[tuple[int, Future]] = []


class ExpertCache:
    def __init__(
        self,
        budget_bytes: int,
        read_threads: int = 16,
        nocache: bool = True,
        clear_bytes: int | None = None,
        staging_bytes: int | None = None,
    ):
        self.budget_bytes = budget_bytes
        self.pool = FilePool(nocache=nocache)

        # How many bytes may be evicted before Metal's recycled-buffer pool is
        # flushed back to the OS. This has to scale with the model: a 10 MB-
        # expert model evicts multiple GB per decoded *token*, and clearing at
        # a fixed 2 GB threw away the warm buffer pool once per token - every
        # subsequent mx.array() then paid fresh zero-fill page faults for the
        # whole miss mass. Sizing the threshold to the cache budget keeps
        # clears rare while set_cache_limit (see loader) bounds the pool.
        self.clear_bytes = (
            clear_bytes if clear_bytes is not None else _CLEAR_THRESHOLD
        )
        # Cap on prefetched-but-unclaimed numpy staging. Scales with expert
        # size (a fixed 1 GB holds 400+ qwen3-next experts but fewer than 100
        # of a 235B's, which dropped correct predictions before their layer
        # ran and re-read them from disk as blocking misses).
        self.staging_bytes = (
            staging_bytes
            if staging_bytes is not None
            else (config.STAGING_BYTES or (1 << 30))
        )

        self._layouts: dict[str, dict[str, ExpertLocator]] = {}
        # layer_key -> ((fd, base_offset, stride), ...) in slab.names order.
        # Empty until slabs are on, or for layouts that are not
        # base + expert*stride (see _build_read_plans).
        self._read_plan: dict[str, tuple[tuple[int, int, int], ...]] = {}
        self._expert_nbytes: dict[str, int] = {}
        self._lru: "OrderedDict[tuple, dict]" = OrderedDict()
        self._entry_bytes: dict[tuple, int] = {}
        # Use count per resident expert, for frequency-protected eviction.
        self._freq: dict[tuple, int] = {}
        self._raw_ready: "OrderedDict[tuple, dict]" = OrderedDict()
        self._raw_bytes = 0
        self._inflight: dict[tuple, Future] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=read_threads, thread_name_prefix="expert-read"
        )
        # Second pool for intra-expert parallelism: one expert's components
        # (and 2 MB slices of its big weight tensors) are read concurrently
        # here while the expert-level worker waits. Separate pool on purpose -
        # workers of one pool blocking on tasks of the same pool deadlocks.
        # This cuts a cold expert's read latency from "sum of 9 serial preads"
        # to roughly "latency of the largest slice", which matters because
        # decode blocks on exactly these latencies (route prediction only buys
        # a few layers of lead time).
        self._slice_executor = ThreadPoolExecutor(
            max_workers=read_threads, thread_name_prefix="expert-slice"
        )
        # Decode slot reads go here once slabs + a read plan exist. None
        # disables the path so the executor lanes above stay A/B-able.
        self._reads: _ReadPool | None = None
        want_pool = config.READ_POOL_THREADS
        self._read_pool_threads = (
            0 if want_pool < 0
            else want_pool if want_pool > 0
            else max(16, read_threads * 2)
        )
        # A dedicated speculative executor lane was tried and measured 6-9%
        # slower: shared drive, large unsliced preads monopolize it, and the
        # shared FIFO's demand-before-spec ordering was doing useful work.
        # See docs/improvements.md.
        self._buffers = BufferPool()
        # Slot-addressed slab storage, when the model's experts are uniformly
        # shaped (see enable_slabs). None => the per-expert mx.array path.
        self.slab = None
        # Set while a prefill holds the slab's memory, so leave_prefill knows
        # to rebuild it (and to what size).
        self._slab_pending: tuple | None = None
        # Set while a release is draining in-flight slot reads (see
        # _release_slab). Blocks new ones so the drain terminates.
        self._slab_frozen = False
        self._ram_budget_bytes = 0
        self._slab_per_expert = 0
        self._slab_budget = 0
        self._slot_of: dict[tuple, int] = {}
        # GPU-resident expert->slot tables (see enable_flow). None = flow off,
        # and every call site below is a single None-check in that case.
        self._flow_rows: dict[str, np.ndarray] | None = None
        self._flow_tables: dict[str, mx.array] = {}
        self._flow_dirty: set[str] = set()
        self._flow_n = 0
        self._free_slots: list[int] = []
        # Slots the in-progress fetch depends on: they must survive eviction
        # for the duration of that fetch (see _alloc_slot).
        self._pinned: set[int] = set()
        self.slot_skipped_installs = 0
        self.slot_starved = 0
        # EXPERT_STREAM_SLAB_VERIFY: writes in flight per slot, so compute can
        # assert nobody is overwriting what it is about to read.
        self._verify = bool(config.SLAB_VERIFY)
        self._writing: dict[int, int] = {}
        self.slab_conflicts = 0
        self._evicted_since_clear = 0
        self._pending_clear = False
        # Set while a big prefill pass runs on a shrunk budget (see
        # enter_prefill); holds the decode-time budget to restore.
        self._decode_budget: int | None = None

        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.resident_bytes = 0
        self.peak_resident_bytes = 0
        self.bytes_read = 0
        # Bytes read to satisfy a *blocking* demand fetch, as opposed to
        # speculation. This is the quantity a prefetch head's budget has to be
        # measured against: it says how disk-bound this model actually is, and
        # therefore how much idle bandwidth (if any) there is to spend.
        self.demand_miss_bytes = 0
        # MoE layers that ran a prefill (whole-layer) fetch. Divided by the
        # model's MoE layer count this is the number of passes the prompt made
        # over the expert mass - the quantity prefill time is proportional to.
        self.prefill_layers = 0
        self.disk_wait_s = 0.0
        # Host-side copy of read bytes into MLX arrays, on the calling thread.
        # Tracked separately because it is neither disk wait nor GPU work, and
        # at prefill volumes (a full pass over the expert mass) it is large.
        self.materialize_s = 0.0
        # Route-prediction accounting (see PrefetchRing): how much speculative
        # I/O was actually used, and how much of the real demand it covered.
        self.pred_issued = 0
        self.pred_used = 0
        self.route_total = 0
        self.heur_used = 0
        # Layer calls that reused the block's own router output instead of
        # re-running the router (streaming._tapped_route). Anything below the
        # MoE layer count per token means some family fell back to the re-run.
        self.route_tapped = 0
        self.route_calls = 0
        # Decode pruning accounting (see config.PRUNE / streaming.__call__).
        self.pruned_slots = 0
        self.demand_slots = 0
        # Slots a weight test dropped and residency put back (KEEP_FREE).
        self.kept_free = 0
        # Experts skipped rather than waited on (WAIT_ABOVE).
        self.skipped_waits = 0
        self.staging_drops = 0
        self.spec_wasted_bytes = 0
        # Optional sidecar residency advisor (set by PrefetchRing.attach_sidecar).
        self._residency_advisor = None
        self.spec_skipped = 0
        # Wall time spent predicting + queueing speculative reads. This is the
        # cost side of prediction: it wins whenever it saves more disk_wait_s
        # than it spends here, which is a property of how fast the disk is.
        self.spec_s = 0.0
        # Speculation may only use spare read capacity. Past this many reads
        # in flight, prefetch is a no-op (would delay a blocking demand read).
        # Default mult is 3x: prune-aware prediction is ~77% precise, and a
        # big-expert demand burst alone holds ~30 reads - a 2x cap skipped
        # hundreds of correct predictions/token when prefetch mattered most.
        self._spec_inflight_cap = max(4, int(read_threads * config.SPEC_INFLIGHT_MULT))

    def register_layer(self, layer_key: str, components: dict[str, ExpertLocator]):
        self._layouts[layer_key] = components
        self._expert_nbytes[layer_key] = sum(
            loc.loc(0).nbytes for loc in components.values()
        )

    def expert_nbytes(self, layer_key: str) -> int:
        """Approximate bytes for one expert of this layer (for group sizing)."""
        return self._expert_nbytes[layer_key]

    # ----------------------------------------------------- flow slot tables

    def enable_flow(self, n_experts: int) -> bool:
        """Publish expert->slot as a GPU table per layer. Returns True if on.

        This is the mapping the per-layer sync exists to deliver: with it on the
        device, a layer can build its gather indices and decide what is resident
        without the routing ever reaching the CPU (see config.FLOW).

        The table only ever names experts that are installed in the LRU, never
        ones with a read in flight - a slot is assigned when the read is
        submitted, so trusting `_slot_of` alone would compute with whatever the
        slot held before.
        """
        if self.slab is None or n_experts <= 0 or not self._layouts:
            return False
        with self._lock:
            self._flow_n = int(n_experts)
            self._flow_rows = {
                key: np.full(self._flow_n, -1, dtype=np.int32)
                for key in self._layouts
            }
            self._flow_tables = {}
            self._flow_dirty = set(self._flow_rows)
            for (layer_key, eid), entry in self._lru.items():
                slot = entry.get("__slot__")
                row = self._flow_rows.get(layer_key)
                if slot is not None and row is not None and eid < self._flow_n:
                    row[eid] = slot
        return True

    def _flow_mark(self, key: tuple, slot: int | None) -> None:
        """Assumes lock held. Record (or clear) one expert's slot for the table."""
        if self._flow_rows is None:
            return
        layer_key, eid = key
        row = self._flow_rows.get(layer_key)
        if row is None or not (0 <= eid < self._flow_n):
            return
        row[eid] = -1 if slot is None else int(slot)
        self._flow_dirty.add(layer_key)

    def _flow_reset(self) -> None:
        """Assumes lock held. Nothing is resident any more (slab released)."""
        if self._flow_rows is None:
            return
        for layer_key, row in self._flow_rows.items():
            row.fill(-1)
            self._flow_dirty.add(layer_key)

    def flow_table(self, layer_key: str):
        """This layer's expert->slot table on the GPU, or None when flow is off.

        Call on the MLX thread. Re-uploads only after residency changed; the row
        is one int32 per expert (640 B on a 160-expert layer), so a rebuild is
        cheaper than the sync it replaces even when it happens every layer.
        """
        if self._flow_rows is None:
            return None
        with self._lock:
            row = self._flow_rows.get(layer_key)
            if row is None:
                return None
            if layer_key in self._flow_dirty:
                self._flow_dirty.discard(layer_key)
                self._flow_tables[layer_key] = mx.array(row)
            return self._flow_tables.get(layer_key)

    # --------------------------------------------------------- slab storage

    def enable_slabs(
        self,
        slab_bytes: int | None = None,
        reserve_frac: float | None = None,
        ram_budget_bytes: int = 0,
    ):
        """Switch expert residency to slot-addressed slabs. Returns True if on.

        Call once on the MLX thread, after every MoE layer has registered.
        Requires all layers to share one component layout - true for every
        stacked-checkpoint MoE we run, since expert shape is a property of the
        architecture, not of the layer. Anything unexpected (odd dtype, mixed
        shapes, too little room, a future MLX that stops handing out writeable
        buffers) returns False and leaves the per-expert path exactly as it
        was, so this can never make an unsupported model worse.

        Slabs are allocated once and never freed, which is the point: reads
        land in their final location and compute batches across slots. It also
        means the memory cannot be lent to prefill activations the way the
        per-expert cache could, so `slab_bytes` is a hard, up-front commitment
        rather than the soft ceiling `budget_bytes` was.
        """
        if self.slab is not None:
            return True
        if not self._layouts:
            return False

        specs: dict[str, tuple[tuple, str, int]] | None = None
        for comps in self._layouts.values():
            layer: dict[str, tuple[tuple, str, int]] = {}
            for name, locator in comps.items():
                loc = locator.loc(0)
                if not supported_dtype(loc.dtype):
                    return False
                layer[name] = (tuple(loc.shape), loc.dtype, int(loc.nbytes))
            if specs is None:
                specs = layer
            elif layer != specs:
                return False  # non-uniform experts: slots would not line up
        if not specs:
            return False

        per_expert = sum(spec[2] for spec in specs.values())
        want = int(slab_bytes if slab_bytes is not None else self.budget_bytes)
        slots = want // max(1, per_expert)
        # Never reserve more slots than the model has experts: on a model whose
        # whole expert mass fits the budget that would commit the full cache
        # size to hold nothing.
        total = 0
        for comps in self._layouts.values():
            locator = next(iter(comps.values()))
            if isinstance(locator, StackedLocator):
                total += int(locator.stacked.shape[0])
            elif isinstance(locator, DirectLocator):
                total += len(locator.per_expert)
            else:
                total = 0
                break
        if total:
            slots = min(slots, total)
        if slots < 32:
            # Fewer slots than a couple of layers' top-k plus reserve: the
            # per-expert path's soft byte budget degrades more gracefully.
            return False
        try:
            store = SlabStore(specs, slots)
        except Exception:
            return False

        self.slab = store
        self._build_read_plans()
        self._ram_budget_bytes = int(ram_budget_bytes)
        self._slab_per_expert = per_expert
        # Highest slot first so early allocations walk up from 0 (nicer to
        # read in traces); order is otherwise irrelevant.
        self._free_slots = list(range(slots - 1, -1, -1))
        # Hold back slots so in-flight reads and unclaimed prefetch guesses
        # have somewhere to land. Unlike the per-expert path (staging is
        # numpy outside the cache budget), staged guesses occupy slots, so
        # reserve and staging cap share one pool or prefetch starves reads.
        if reserve_frac is None:
            reserve_frac = config.SLAB_RESERVE_FRAC
        reserve = max(4, int(slots * reserve_frac))
        if config.STAGING_BYTES is None:  # an explicit setting still wins
            inflight = min(reserve // 2, self._spec_inflight_cap)
            self.staging_bytes = max(1, reserve - inflight) * per_expert
        self.budget_bytes = (slots - reserve) * per_expert
        self._slab_budget = self.budget_bytes
        self._slab_pending = None
        self._decode_budget = None
        return True

    def _build_read_plans(self) -> None:
        """Resolve every (layer, component) to (fd, base offset, stride) once.

        The miss path used to rebuild this per read: nine locator.loc() calls
        and nine slab.dest() slices. At ~250 misses/token that is ~2000
        dataclasses and numpy slices of GIL-held python (gil_probe.py: busy
        main thread costs up to 74% of read bandwidth).

        Ordered to match slab.names so a read zips this against
        slab.slot_dests[slot] with no name lookups. Stride/layout checks that
        used to run per-read happen here so mismatches fail at load.
        """
        plans: dict[str, tuple[tuple[int, int, int], ...]] = {}
        for layer_key, comps in self._layouts.items():
            entries = []
            for i, name in enumerate(self.slab.names):
                locator = comps[name]
                loc = locator.loc(0)
                stride = self.slab.strides[i]
                if loc.nbytes != stride:
                    raise RuntimeError(
                        f"{layer_key}.{name}: {loc.nbytes} B on disk, "
                        f"{stride} B slot"
                    )
                if isinstance(locator, StackedLocator):
                    base = locator.stacked.offset
                    per = locator.stacked.nbytes // locator.stacked.shape[0]
                    if per != stride:
                        raise RuntimeError(
                            f"{layer_key}.{name}: stacked stride {per} != "
                            f"slot stride {stride}"
                        )
                    entries.append((self.pool._fd(loc.file), base, stride))
                else:
                    # Per-expert tensors have no arithmetic stride between
                    # experts; leave the plan absent and use the locator path.
                    entries = []
                    break
            if entries:
                plans[layer_key] = tuple(entries)
        self._read_plan = plans if len(plans) == len(self._layouts) else {}
        if self._read_plan and self._reads is None and self._read_pool_threads > 0:
            self._reads = _ReadPool(self._read_pool_threads)

    def _evict_key(self, key: tuple) -> None:
        """Assumes lock held. Drop one resident entry, freeing its slot."""
        self._lru.pop(key, None)
        freed = self._entry_bytes.pop(key, 0)
        self._freq.pop(key, None)
        self.resident_bytes -= freed
        self.evictions += 1
        self._evicted_since_clear += freed
        slot = self._slot_of.pop(key, None)
        if slot is not None:
            self._free_slots.append(slot)
        self._flow_mark(key, None)

    def _release_slot(self, key: tuple, entry: dict) -> None:
        """Assumes lock held. Hand back the slot of an entry we won't install.

        A slot the resident copy of `key` is already using stays put; anything
        else (a duplicate read, or a soft install with no room) is orphaned and
        goes straight back on the free list.
        """
        slot = entry.get("__slot__")
        if slot is None:
            return
        if self._slot_of.get(key) == slot:
            if key in self._lru:
                return
            self._slot_of.pop(key, None)
        self._pinned.discard(slot)
        self._free_slots.append(slot)

    def _alloc_slot(self, evict_ok: bool = True) -> int | None:
        """Assumes lock held. A slot no live expert is using.

        `evict_ok=False` restricts the caller to slots that are already free -
        speculation must never evict a resident expert to make room for a
        guess. The reserve enable_slabs() holds back is what keeps that from
        starving prefetch.

        Reuse is destructive - the next read overwrites these bytes in place -
        so this must never return a slot some pending graph still reads. Two
        properties of the decode path guarantee that:

        * Every MoE layer syncs on its own router *before* it fetches (it
          cannot know what to read otherwise), and that sync forces every
          earlier layer's expert matmuls. So by the time any fetch can evict,
          all previously fetched experts have already been consumed.
        * The experts the in-progress fetch itself depends on are pinned, and
          slots holding reads in flight or unclaimed staging are in neither
          the LRU nor the free list, so they are never candidates.
        """
        if self._free_slots:
            return self._free_slots.pop()
        if not evict_ok:
            return None
        for key in list(self._lru):  # least recently used first
            if self._slot_of.get(key) in self._pinned:
                continue
            self._evict_key(key)
            if self._free_slots:
                return self._free_slots.pop()
        # Nothing evictable: give up unclaimed prefetch guesses instead.
        while self._raw_ready:
            oldest = next(iter(self._raw_ready))
            dropped = self._drop_raw(oldest)
            self.staging_drops += 1
            if dropped is not None:
                self.spec_wasted_bytes += dropped["__nbytes__"]
            if self._free_slots:
                return self._free_slots.pop()
        self.slot_starved += 1
        return None

    def _read_expert_slab(self, layer_key: str, expert_id: int, slot: int) -> dict:
        """Read one expert straight into its slab slot.

        Runs on an expert-read worker and touches no MLX API at all: the
        destinations are plain numpy views of slab memory captured at load
        time, so this is a pread into pages that already exist. That removes
        both the staging buffer and the host->device copy the per-expert path
        pays on the GPU thread. Only leaf preads go to the slice pool (nested
        waits inside one pool would deadlock it).
        """
        if self._verify:
            with self._lock:
                self._writing[slot] = self._writing.get(slot, 0) + 1
        try:
            return self._read_slab_inner(layer_key, expert_id, slot)
        finally:
            if self._verify:
                with self._lock:
                    self._writing[slot] -= 1

    def check_slots(self, layer_key: str, slots: dict) -> None:
        """Debug hook: the slots a layer is about to compute on must be quiet.

        A write in flight on a slot this layer owns, or bookkeeping that
        disagrees about who owns it, means the slot was handed out twice -
        which silently corrupts output rather than raising.
        """
        if not self._verify:
            return
        with self._lock:
            for eid, slot in slots.items():
                if self._writing.get(slot):
                    self.slab_conflicts += 1
                    print(
                        f"[slab] WRITE-WHILE-COMPUTE layer={layer_key} "
                        f"expert={eid} slot={slot}",
                        flush=True,
                    )
                owner = self._slot_of.get((layer_key, eid))
                if owner != slot:
                    self.slab_conflicts += 1
                    print(
                        f"[slab] OWNER-MISMATCH layer={layer_key} expert={eid} "
                        f"computing slot={slot} but slot_of={owner}",
                        flush=True,
                    )

    def _read_slab_inner(self, layer_key: str, expert_id: int, slot: int) -> dict:
        plan = self._read_plan.get(layer_key)
        if plan is None:
            return self._read_slab_legacy(layer_key, expert_id, slot)
        dests = self.slab.slot_dests[slot]

        # Fan only the big weight tensors; scales/biases stay inline.
        # Qwen3-235B: three ~3 MB weights + six ~0.2 MB scales - fanning all
        # nine bought queue depth at ~12 Futures/expert (~2950 Futures and
        # ~10700 lock acqs/token in mainthread_profile.py). Big requests
        # need ~qd16 to saturate (disk_ceiling.py); 0.2 MB ones never do.
        jobs = None
        fan = _FANOUT_BYTES
        for (fd, base, stride), mv in zip(plan, dests):
            off = base + expert_id * stride
            if stride >= fan:
                if jobs is None:
                    jobs = []
                jobs.append(self._slice_executor.submit(_pread_full, fd, off, mv))
            else:
                _pread_full(fd, off, mv)
        if jobs is not None:
            for j in jobs:
                j.result()
        return {"__nbytes__": self.slab.expert_bytes, "__slot__": slot}

    def _read_slab_legacy(self, layer_key: str, expert_id: int, slot: int) -> dict:
        """Locator-driven slot read, for layouts `_build_read_plans` declined
        (per-expert tensors, where expert e's offset is not base + e*stride)."""
        comps = self._layouts[layer_key]
        jobs = []
        nbytes = 0
        for name, locator in comps.items():
            loc = locator.loc(expert_id)
            dest = self.slab.dest(name, slot)
            if loc.nbytes != dest.size:
                raise RuntimeError(
                    f"{layer_key}.{name}: {loc.nbytes} B on disk, "
                    f"{dest.size} B slot"
                )
            nbytes += loc.nbytes
            for start in range(0, loc.nbytes, _SLICE_BYTES):
                end = min(start + _SLICE_BYTES, loc.nbytes)
                jobs.append(
                    self._slice_executor.submit(
                        self.pool.read_into,
                        loc.file,
                        loc.offset + start,
                        dest[start:end],
                    )
                )
        for j in jobs:
            j.result()
        return {"__nbytes__": nbytes, "__slot__": slot}

    def _submit_slab(
        self, layer_key: str, expert_id: int, evict_ok: bool = True
    ) -> Future | None:
        """Assumes lock held. Start a slot read, or None if no slot is free."""
        if self._slab_frozen:
            # A release is draining the reads that are already writing into
            # slab memory; adding one now would race the free.
            return None
        slot = self._alloc_slot(evict_ok=evict_ok)
        if slot is None:
            return None
        key = (layer_key, expert_id)
        self._slot_of[key] = slot
        self._pinned.add(slot)
        plan = self._read_plan.get(layer_key) if self._reads is not None else None
        if plan is None:
            fut = self._executor.submit(
                self._read_expert_slab, layer_key, expert_id, slot
            )
            self._inflight[key] = fut
            return fut

        # Planned path: push component reads straight onto the read pool.
        # No expert-level worker in between - it would only submit and block.
        fut: Future = Future()
        fut.set_running_or_notify_cancel()
        entry = {"__nbytes__": self.slab.expert_bytes, "__slot__": slot}
        on_done = None
        if self._verify:
            self._writing[slot] = self._writing.get(slot, 0) + 1

            def on_done(slot=slot):
                with self._lock:
                    self._writing[slot] -= 1

        latch = _SlotLatch(len(plan), fut, entry, on_done)
        dests = self.slab.slot_dests[slot]
        submit = self._reads.submit
        for (fd, base, stride), mv in zip(plan, dests):
            submit(fd, base + expert_id * stride, mv, latch)
        self._inflight[key] = fut
        return fut

    # ------------------------------------------------------------- reads

    def _read_expert(self, layer_key: str, expert_id: int) -> dict:
        """Read one expert with all components (and 2 MB slices of the big
        ones) in flight on the slice pool at once.

        Runs on an expert-read worker; only *leaf* preads go to the slice
        pool (no nested waits there, so no pool deadlock). Latency drops from
        the sum of ~9 serial preads to roughly one slice's worth - and decode
        blocks on exactly this latency for every unpredicted miss.
        """
        comps = self._layouts[layer_key]
        out = {}
        bufs = []
        nbytes = 0
        jobs = []
        for name, locator in comps.items():
            loc = locator.loc(expert_id)
            nbytes += loc.nbytes
            buf = self._buffers.acquire(loc.nbytes)
            bufs.append(buf)
            for start in range(0, loc.nbytes, _SLICE_BYTES):
                end = min(start + _SLICE_BYTES, loc.nbytes)
                jobs.append(
                    self._slice_executor.submit(
                        self.pool.read_into,
                        loc.file,
                        loc.offset + start,
                        buf[start:end],
                    )
                )
            np_dtype = _LOC_DTYPES[loc.dtype]
            out[name] = (buf.view(np_dtype).reshape(loc.shape), loc.is_bf16)
        for j in jobs:
            j.result()
        out["__nbytes__"] = nbytes
        out["__bufs__"] = bufs
        return out

    def _submit(self, key: tuple) -> Future:
        fut = self._inflight.get(key)
        if fut is None:
            fut = self._executor.submit(self._read_expert, key[0], key[1])
            self._inflight[key] = fut
        return fut

    # ------------------------------------------------------------- run reads

    def _read_expert_run(self, layer_key: str, ids: list[int]) -> dict[int, dict]:
        """Read several *consecutive* experts of one layer.

        For stacked checkpoints, experts first..last of each component tensor
        are one contiguous byte range, so the whole run costs a few large
        sequential reads instead of len(ids) × the component count of small
        scattered ones. This is what lets prefill - which wants nearly every
        expert of every layer - run at the SSD's sequential bandwidth.

        Every component's slices are submitted before anything is waited on.
        Draining them component by component instead left only one component's
        worth of reads (a few MB) in flight at a time and idled the drive at
        each of the ~9 barriers per run.
        """
        if len(ids) == 1:
            # Single-expert "runs" are the common decode case; use the pooled,
            # component-parallel read path (much lower latency).
            return {ids[0]: self._read_expert(layer_key, ids[0])}
        comps = self._layouts[layer_key]
        first, count = ids[0], len(ids)
        out: dict[int, dict] = {eid: {} for eid in ids}
        per_expert_bytes: dict[int, int] = {eid: 0 for eid in ids}
        jobs = []
        for name, locator in comps.items():
            if isinstance(locator, StackedLocator):
                span = locator.span(first, count)
                arr = self._submit_span(span, jobs)
                stride = span.nbytes // count
                for i, eid in enumerate(ids):
                    out[eid][name] = (arr[i], span.is_bf16)
                    per_expert_bytes[eid] += stride
            else:
                for eid in ids:
                    loc = locator.loc(eid)
                    out[eid][name] = (self._submit_span(loc, jobs), loc.is_bf16)
                    per_expert_bytes[eid] += loc.nbytes
        for j in jobs:
            j.result()
        for eid in ids:
            out[eid]["__nbytes__"] = per_expert_bytes[eid]
        return out

    def _submit_span(self, span, jobs: list) -> np.ndarray:
        """Start reads for one contiguous range; return its (unfilled) view.

        Appends the slice futures to `jobs` for the caller to drain once. The
        returned numpy view is only valid after that drain.

        Not buffer-pooled: span buffers are shared by several experts' raw
        entries (each views one row), so their lifetime isn't per-expert.
        Span sizes also vary with run length, which would fragment the pool.
        """
        buf = np.empty(span.nbytes, dtype=np.uint8)
        step = _PREFILL_SLICE_BYTES
        for start in range(0, span.nbytes, step):
            end = min(start + step, span.nbytes)
            jobs.append(
                self._slice_executor.submit(
                    self.pool.read_into, span.file, span.offset + start, buf[start:end]
                )
            )
        return buf.view(_LOC_DTYPES[span.dtype]).reshape(span.shape)

    def _submit_run(self, layer_key: str, ids: list[int]) -> list[tuple[int, Future]]:
        """Assumes lock held; `ids` are consecutive, uncached, not in flight.

        Registers one per-expert Future each (so prefetch/fetch dedup keeps
        working) resolved from a single run read.
        """
        futs = {eid: Future() for eid in ids}
        for eid, f in futs.items():
            self._inflight[(layer_key, eid)] = f
        run_fut = self._executor.submit(self._read_expert_run, layer_key, ids)

        def _split(rf):
            try:
                raws = rf.result()
            except Exception as e:
                for f in futs.values():
                    f.set_exception(e)
                return
            for eid, f in futs.items():
                f.set_result(raws[eid])

        run_fut.add_done_callback(_split)
        return list(futs.items())

    def _max_run_bytes(self, layer_key: str, prefill: bool = False) -> int:
        """Split big runs so reads still spread across the reader threads.

        Decode keeps runs short: it misses a handful of experts and wants each
        one back as soon as possible. Prefill is the opposite - it wants the
        whole layer and cares only about aggregate bandwidth, so the cap is
        raised until a run covers tens of experts. At the decode cap a single
        DeepSeek expert (~25 MB) already fills a run, which silently disabled
        coalescing for exactly the workload it was written for.
        """
        per = max(1, self.expert_nbytes(layer_key))
        floor = config.PREFILL_RUN_BYTES if prefill else (32 << 20)
        return max(per, floor)

    # ------------------------------------------------------------- memory

    def _pin(self, key: tuple) -> None:
        """Assumes lock held. Protect this key's slot for the current fetch."""
        slot = self._slot_of.get(key)
        if slot is not None:
            self._pinned.add(slot)

    def free_ids(self, layer_key: str, expert_ids) -> set[int]:
        """Which of these experts need no *new* disk read right now.

        Resident, already staged, or already in flight - in all three cases the
        bytes are either here or on their way, so computing the expert costs
        nothing extra on the disk that is this engine's bottleneck. Decode-time
        expert selection uses this to avoid the one trade that is never worth
        making: dropping an expert that was free, which spends output quality
        and buys no speed at all.
        """
        out: set[int] = set()
        with self._lock:
            for eid in expert_ids:
                eid = int(eid)
                key = (layer_key, eid)
                if (
                    key in self._lru
                    or key in self._raw_ready
                    or key in self._inflight
                ):
                    out.add(eid)
        return out

    def _drop_raw(self, key: tuple, claim: bool = False) -> dict | None:
        """Assumes lock held. Remove staged data for `key`, keeping the byte
        count honest (it is the cap that bounds speculative RAM).

        `claim=True` means a fetch is taking this entry and will install it, so
        it keeps its slot. Otherwise the guess is being thrown away and the slot
        goes back on the free list.
        """
        raw = self._raw_ready.pop(key, None)
        if raw is not None:
            self._raw_bytes -= raw["__nbytes__"]
            if not claim:
                slot = self._slot_of.get(key)
                # A pinned slot belongs to a fetch that is already waiting on
                # this read and will install it (prefetch's completion callback
                # runs on a worker and can reach this overflow path first).
                # Freeing it here hands a live slot to the next read, which
                # then overwrites the expert in place.
                if slot is not None and slot not in self._pinned:
                    self._slot_of.pop(key, None)
                    self._free_slots.append(slot)
        return raw

    def _log_memory(self, where: str) -> None:
        """What Metal is holding, for diagnosing prefill OOMs (MEM_DEBUG)."""
        if not config.MEM_DEBUG:
            return
        print(
            f"[paged-moe mem] {where}: "
            f"active={mx.get_active_memory() / 1e9:.2f} GB "
            f"cached={mx.get_cache_memory() / 1e9:.2f} GB "
            f"peak={mx.get_peak_memory() / 1e9:.2f} GB "
            f"slab={(self.slab.nbytes / 1e9) if self.slab else 0:.2f} GB "
            f"resident={self.resident_bytes / 1e9:.2f} GB",
            flush=True,
        )

    def _flush_metal_if_needed(self, force: bool = False):
        """Return freed Metal buffers to the OS. Call outside the lock."""
        if not (self._pending_clear or force):
            return
        self._pending_clear = False
        mx.clear_cache()

    # --------------------------------------------------- prefill/decode mode

    def enter_prefill(self, budget_bytes: int) -> None:
        """Shrink the residency budget for a big prefill pass.

        A prefill chunk touches ~every expert of every layer, so it reads the
        whole uncached expert mass once per chunk. The way to make prefill fast
        is therefore to process the prompt in as few chunks as possible - and a
        big chunk needs memory for its activations, which is memory the expert
        cache would otherwise hold. Within a single pass a large cache buys
        nothing (nothing is revisited), so trading it for one-pass prefill is
        strictly better: measured 31.5k tokens at 787 tok/s reading 40 GB
        (one pass, 4 GB cache) versus 541 tok/s reading 85 GB (four passes,
        18 GB cache) - and with a *lower* memory peak.
        """
        # Slab memory cannot be *lent*, only returned: a 32k-token chunk's
        # activations want >10 GB, which no 48 GB machine has spare next to a
        # 24 GB slab (measured: it OOMs Metal). So give the whole allocation
        # back and rebuild it when decode resumes. Residency is lost either
        # way - the per-expert path shrinks to PREFILL_CACHE_GB here for the
        # same reason.
        self._release_slab()
        with self._lock:
            first = self._decode_budget is None
            if first:
                self._decode_budget = self.budget_bytes
            self.budget_bytes = min(self.budget_bytes, int(budget_bytes))
            self._evict_to_budget()
        self._flush_metal_if_needed(force=True)
        if first:  # every MoE layer calls this; only the transition is news
            self._log_memory("enter_prefill")

    def make_room_for(self, nbytes: int, freeing: int = 0) -> bool:
        """Give the slab back if `nbytes` will not otherwise fit. Returns True
        if the cache had to yield.

        `freeing` is memory the caller releases as it allocates, and crediting it
        is the difference between yielding every turn and almost never: a
        resumed GLM-4.7 turn allocates 4.9 GB of fp16 KV but drops the 2.5 GB
        quantized cache it came from, so the slab only has to make room for the
        2.4 GB difference. Ignoring that released a 16 GB slab - and with it
        every resident expert - on turns that had 1.5 GB to spare.

        Slab memory is committed up front and cannot be lent, only returned
        (see enter_prefill), so this is all-or-nothing: either the allocation
        fits alongside the slab or the slab goes and `leave_prefill` rebuilds it
        to whatever the new tenant left (`_slots_that_fit`).

        The caller is the fp16 KV dequantize a resumed turn needs
        (`kvmem.unquantize`), which is the largest allocation the server makes
        outside load: 9.0 GB on GLM-4.7 at 24k tokens, against a 19 GB slab and
        a 9.6 GB backbone on a 35 GB budget. `PREFILL_SHRINK_TOKENS` cannot
        catch it, because that trigger looks at the *chunk* size and an agent
        turn appends a few hundred tokens to a long context - the memory comes
        from the context it resumes, not from the chunk.

        Yielding costs residency, so it is worth doing only when the alternative
        is an asynchronous Metal command-buffer failure (i.e. a dead request).
        Hence the check rather than an unconditional shrink.
        """
        if self.slab is None or not self._ram_budget_bytes or nbytes <= 0:
            return False
        # Both the transient peak and the settled figure matter: the conversion
        # is incremental, so at every step it holds some of the new allocation
        # and the not-yet-released remainder of the old one. Charging the full
        # allocation while crediting the full release bounds both.
        headroom = self._ram_budget_bytes - mx.get_active_memory() - _SLAB_MARGIN
        if nbytes - max(0, int(freeing)) <= headroom:
            return False
        self.enter_prefill(int(config.PREFILL_CACHE_GB * (1 << 30)))
        return True

    def _slots_that_fit(self, want: int) -> int:
        """Trim the slab to the room the KV cache left us.

        The slab is sized at load, when the only other tenant is the backbone.
        By the time it is rebuilt after a prefill, the conversation's KV cache
        is resident too, and that grows without bound: a 28k-token context on
        Qwen3-235B put the rebuild at 40.3 GB against a 35.05 GB budget, which
        Metal reports as an *asynchronous* command-buffer failure - the server
        dies rather than raising. A slab that is too small only costs a lower
        hit rate, so give the KV cache right of way and take what is left.
        """
        if not (self._ram_budget_bytes and self._slab_per_expert):
            return want
        headroom = self._ram_budget_bytes - mx.get_active_memory() - _SLAB_MARGIN
        fits = int(headroom) // self._slab_per_expert
        if fits >= want:
            return want
        return max(32, min(want, fits))

    def _release_slab(self) -> None:
        """Hand the slab's memory back to Metal, waiting out its writers.

        A slot read is a `pread` from a worker thread straight into slab
        memory (`_read_expert_slab`), and decode starts those speculatively -
        so when a long prompt arrives right after a decode turn, reads are
        still in flight. Freeing the slab under them is a use-after-free:
        Metal hands the pages to the prefill's activations and KV, the worker
        finishes its pread into what is now someone else's memory, and the
        model prefills a corrupted context. It reads as fluent, confidently
        off-topic output, and occasionally as an allocator-level abort.

        Freeze, drain, then free. The drain has to happen with the lock
        *released*: verification mode takes it inside the worker.
        """
        with self._lock:
            if self.slab is None:
                return
            self._slab_frozen = True
            pending = list(self._inflight.values())
        for fut in pending:
            try:
                fut.result()
            except Exception:
                pass  # a failed read has nothing in flight either way
        with self._lock:
            try:
                if self.slab is not None:
                    self._release_slab_locked()
            finally:
                self._slab_frozen = False

    def _release_slab_locked(self) -> None:
        """Assumes lock held. Hand the slab's memory back to Metal.

        Every slot reference has to go with it: resident entries and staged
        guesses both point into slabs that are about to stop existing. What
        comes back is the plain per-expert path, which is what prefill uses
        anyway.
        """
        self._slab_pending = (self.slab.specs, self.slab.slots)
        for key in list(self._lru):
            self._evict_key(key)
        for key in list(self._raw_ready):
            self._drop_raw(key)
        # Drained reads landed in slots that are about to stop existing, so
        # their results can't be claimed. (Reads that predate the slab are
        # plain buffers and stay.)
        for key, fut in list(self._inflight.items()):
            if not fut.done():
                continue
            try:
                raw = fut.result()
            except Exception:
                self._inflight.pop(key, None)
                continue
            if isinstance(raw, dict) and "__slot__" in raw:
                self._inflight.pop(key, None)
        self.slab.release()
        self.slab = None
        self._slot_of.clear()
        self._flow_reset()
        self._free_slots.clear()
        self._pinned.clear()

    def leave_prefill(self) -> None:
        """Restore the decode-time budget (decode re-warms as it reads).

        Also rebuilds the slab if prefill gave it back. Runs on the MLX thread
        (the decode path calls it), which is where slabs must be allocated.
        """
        with self._lock:
            pending = self._slab_pending
            if pending is not None and self.slab is None:
                specs, slots = pending
                # Nothing resident can survive: those entries are loose mx
                # arrays from prefill's read path and have no slot to live in.
                for key in list(self._lru):
                    self._evict_key(key)
                for key in list(self._raw_ready):
                    self._drop_raw(key)
                # The prefill that just finished leaves GBs sitting in Metal's
                # pool. Rebuilding the slab is the single largest allocation
                # the process makes after load, and it fails *asynchronously*
                # (an uncaught command-buffer error kills the server), so hand
                # those pages back first rather than asking for new ones.
                mx.clear_cache()
                slots = self._slots_that_fit(slots)
                try:
                    self.slab = SlabStore(specs, slots)
                except Exception:
                    # Out of room to rebuild: stay on the per-expert path
                    # rather than failing the request.
                    self._slab_pending = None
                else:
                    self._free_slots = list(range(slots - 1, -1, -1))
                    reserve = max(4, int(slots * 0.1))
                    self.budget_bytes = min(
                        self._slab_budget, (slots - reserve) * self._slab_per_expert
                    )
                    self._decode_budget = None
                    self._log_memory("leave_prefill (slab rebuilt)")
                    return
            if self._decode_budget is None:
                return
            self.budget_bytes = self._decode_budget
            self._decode_budget = None

    def _evict_to_budget(self) -> None:
        """Assumes lock held.

        Pure LRU is the wrong policy for MoE routing: expert use is heavily
        skewed, so a handful of experts per layer are wanted by almost every
        token while the long tail is wanted once. One sweep of tail traffic
        (a prefill chunk, or a topic change) is enough to evict the hot set
        under LRU, and then every subsequent token pays disk for experts it
        will want again immediately.

        So: grant a second chance. If the entry at the LRU tail has been used
        repeatedly, halve its counter and move it to MRU instead of evicting;
        the scan is bounded so a cache of uniformly hot entries still makes
        progress. Cold entries are evicted on sight, exactly as before.
        Eviction only affects what has to be re-read, never any result.
        """
        scan = config.LFRU_SCAN if config.LFRU else 0
        while self.resident_bytes > self.budget_bytes and len(self._lru) > 1:
            for _ in range(scan):
                if len(self._lru) <= 1:
                    break
                key = next(iter(self._lru))
                freq = self._freq.get(key, 0)
                if freq <= 0:
                    break
                self._freq[key] = freq >> 1
                self._lru.move_to_end(key)
            # Slots the in-progress fetch needs are not candidates; with slabs
            # off nothing is pinned and this picks the LRU tail as before.
            victim = None
            # Residency advisor (optional): skip hot keys within a bounded
            # scan so cold tails die first. Quality-safe - only changes which
            # expert is re-read later. Reprieved keys are collected and moved
            # after the walk: move_to_end during iteration raises
            # "OrderedDict mutated during iteration".
            advisor = getattr(self, "_residency_advisor", None)
            reprieved: list[tuple] = []
            for key in self._lru:
                if self._slot_of.get(key) in self._pinned:
                    continue
                if advisor is not None and len(reprieved) < _PROTECT_SCAN:
                    try:
                        advisor.on_scan()
                        if advisor.should_protect(key[0], key[1]):
                            advisor.on_protect()
                            reprieved.append(key)
                            continue
                    except Exception:
                        pass
                victim = key
                break
            for key in reprieved:
                self._lru.move_to_end(key)
            if victim is None:
                break
            self._evict_key(victim)

    def _install(self, key: tuple, entry: dict, mode: str):
        """Assumes lock held. Insert a materialized (mx) entry into the LRU."""
        if key in self._lru:
            self._release_slot(key, entry)
            return
        if self.slab is not None and "__slot__" not in entry:
            # Prefill's run reads produce loose mx arrays. Installing them
            # would leave a layer's experts split between slab slots and loose
            # arrays, which gather_qmm cannot address in one call. Decode
            # refills slots itself, so skipping these costs only a re-read.
            self.slot_skipped_installs += 1
            return
        nbytes = entry["__nbytes__"]
        if mode == "soft" and self.resident_bytes + nbytes > self.budget_bytes:
            self._release_slot(key, entry)
            return  # no room; don't evict for prefill traffic
        self._lru[key] = entry
        self._flow_mark(key, entry.get("__slot__"))
        self._entry_bytes[key] = nbytes
        self.resident_bytes += nbytes
        self.peak_resident_bytes = max(self.peak_resident_bytes, self.resident_bytes)
        self._evict_to_budget()
        if self._evicted_since_clear >= self.clear_bytes:
            # Dropping Python refs isn't enough - MLX's buffer pool keeps the
            # pages until clear_cache(). Without this the machine freezes.
            self._evicted_since_clear = 0
            self._pending_clear = True

    # ------------------------------------------------------------- fetch

    def _begin(
        self,
        layer_key: str,
        expert_ids,
        use_slab: bool = False,
        prefill: bool = False,
    ) -> _Begun:
        """Classify each id as LRU hit / prefetched raw / disk read (async).

        Consecutive missing ids are coalesced into run reads (one large
        sequential pread per component instead of many small ones).

        `use_slab` routes misses into slab slots (decode). Slot reads are
        per-expert rather than run-coalesced: each expert lands in its own
        slot, so a run has no single contiguous destination. Neighbouring
        experts still read at neighbouring file offsets, which is the locality
        the SSD actually cares about.
        """
        b = _Begun()
        misses: list[int] = []
        slab = use_slab and self.slab is not None
        with self._lock:
            if slab:
                # The sync this layer just did on its own router forced every
                # earlier layer's expert compute, so the previous fetch's
                # experts are consumed and their slots may now be reused.
                self._pinned.clear()
            for eid in expert_ids:
                eid = int(eid)
                key = (layer_key, eid)
                entry = self._lru.get(key)
                if entry is not None:
                    self.hits += 1
                    self._lru.move_to_end(key)
                    # Reuse is what earns eviction protection; a first install
                    # stays unproven (see _evict_to_budget).
                    self._freq[key] = min(_FREQ_MAX, self._freq.get(key, 0) + 1)
                    b.hits[eid] = entry
                    if slab:
                        self._pin(key)
                    continue
                raw = self._drop_raw(key, claim=True)
                if raw is not None:
                    self.hits += 1
                    b.raw.append((eid, raw))
                    if slab:
                        self._pin(key)
                    continue
                self.misses += 1
                fut = self._inflight.get(key)
                if fut is not None:
                    b.futs.append((eid, fut))
                    if slab:
                        self._pin(key)
                    continue
                misses.append(eid)

            if slab:
                # Anything that cannot get a slot falls through to the run-read
                # path below as a loose mx array. That makes the layer's experts
                # mixed, which costs the gather_qmm batching for this one layer
                # (streaming falls back to per-expert) but is always correct.
                starved: list[int] = []
                for eid in misses:
                    fut = self._submit_slab(layer_key, eid)
                    if fut is None:
                        starved.append(eid)
                    else:
                        b.futs.append((eid, fut))
                misses = starved

            if misses:
                per = max(1, self.expert_nbytes(layer_key))
                max_run = max(1, self._max_run_bytes(layer_key, prefill) // per)
                run: list[int] = [misses[0]]
                for eid in misses[1:]:
                    if eid == run[-1] + 1 and len(run) < max_run:
                        run.append(eid)
                    else:
                        b.futs.extend(self._submit_run(layer_key, run))
                        run = [eid]
                b.futs.extend(self._submit_run(layer_key, run))
        return b

    def _finish(self, layer_key: str, b: _Begun, install: str) -> dict[int, dict]:
        """Wait for reads, materialize on this (GPU) thread, install, return.

        Reads are consumed in completion order, and each one is materialized
        (numpy -> Metal copy) while the remaining reads are still in flight.
        On big-expert models a layer's miss mass is hundreds of MB, so doing
        the copies inside the disk-wait window instead of after it removes a
        serial memcpy pass from every decode layer.
        """
        result = dict(b.hits)
        raws_done = [raw for _, raw in b.raw]
        materialized = [(eid, _materialize(raw)) for eid, raw in b.raw]

        if b.futs:
            by_fut = {fut: eid for eid, fut in b.futs}
            t0 = time.perf_counter()
            busy = 0.0
            for fut in as_completed(by_fut):
                eid = by_fut[fut]
                raw = fut.result()
                key = (layer_key, eid)
                with self._lock:
                    self._inflight.pop(key, None)
                    # claim=True: if prefetch's callback already staged this
                    # very read, the staged entry is the same object with the
                    # same slot, and we are about to install it. Dropping it
                    # as an abandoned guess would put that slot back on the
                    # free list while the resident entry still points at it -
                    # the next read then overwrites a live expert in place.
                    self._drop_raw(key, claim=True)
                m0 = time.perf_counter()
                raws_done.append(raw)
                materialized.append((eid, _materialize(raw)))
                busy += time.perf_counter() - m0
            # Only the un-overlapped remainder counts as waiting on disk.
            self.disk_wait_s += max(0.0, time.perf_counter() - t0 - busy)
            # Materialize is a host-side copy of every byte read, on this
            # thread. Excluding it from disk_wait is right, but leaving it
            # unattributed made the prefill log call it compute.
            self.materialize_s += busy

        if materialized:
            _eval_experts([e for _, e in materialized])
            # Copies are on the GPU now; recycle the read buffers.
            for raw in raws_done:
                _release_raw(self._buffers, raw)
            with self._lock:
                for eid, entry in materialized:
                    # A route-predicted read that demand raced was already
                    # counted and installed by its arrival callback.
                    if not entry.pop("__counted__", False):
                        self.bytes_read += entry["__nbytes__"]
                        if install == "lru":
                            self.demand_miss_bytes += entry["__nbytes__"]
                    self._install((layer_key, eid), entry, install)
                    result[eid] = entry
            self._flush_metal_if_needed()

        return result

    def fetch(self, layer_key: str, expert_ids, install: str = "lru") -> dict[int, dict]:
        """Blocking fetch of a set of experts for one layer.

        Decode (install="lru") reads into slab slots when slabs are on; prefill
        keeps the run-read path, whose big sequential preads have no single
        contiguous slot to land in.
        """
        b = self._begin(layer_key, expert_ids, use_slab=install == "lru")
        return self._finish(layer_key, b, install)

    def fetch_groups(
        self,
        layer_key: str,
        expert_ids: list[int],
        group_bytes: int,
        install: str = "soft",
    ):
        """Yield (ids, experts) in bounded-size groups, pipelining disk reads.

        Used for prefill, where one layer can want hundreds of experts (more
        than fits in memory at once).  While group i computes on the GPU, the
        reads for group i+1 are already in flight on the worker threads.

        `expert_ids` arrives sorted, so a group is a near-contiguous id range
        and its reads coalesce into a few long sequential spans - which is why
        the group wants to be large (see config.PREFILL_RUN_BYTES).
        """
        per = max(1, self.expert_nbytes(layer_key))
        per_group = max(1, group_bytes // per)
        groups = [
            expert_ids[i : i + per_group]
            for i in range(0, len(expert_ids), per_group)
        ]
        pending = self._begin(layer_key, groups[0], prefill=True)
        for i, ids in enumerate(groups):
            current = pending
            if i + 1 < len(groups):
                pending = self._begin(layer_key, groups[i + 1], prefill=True)
            yield ids, self._finish(layer_key, current, install)

    # ------------------------------------------------------------- prefetch

    def _spec_blocked(self, staging: bool = True) -> bool:
        """Assumes lock held. True when speculation must stand down.

        `staging=False` skips the staging-occupancy check: route-predicted
        slab reads install straight into the LRU on arrival (see
        _attach_staging), so a staging area full of stale heuristic guesses
        must not gate them.
        """
        if staging and self._raw_bytes >= self.staging_bytes:
            # Staging is already full of unclaimed guesses.
            self.spec_skipped += 1
            return True
        if len(self._inflight) >= self._spec_inflight_cap:
            # The readers are busy with work someone is blocked on; adding queue
            # depth would delay a read a layer is waiting for.
            self.spec_skipped += 1
            return True
        return False

    def read_slack(self) -> int:
        """How many more reads may be started without delaying a demand miss.

        Prefetching is only free while the drive has capacity nobody is blocked
        on. Past that it is a transfer, not a gain: the speculative read sits in
        the queue ahead of the blocking read some layer is waiting for. A decode
        token on a big-expert model reads hundreds of MB, so a drive that is
        already busy has nothing to lend.

        Returned as a read count rather than a bool so a caller can size its
        batch to the capacity that exists instead of issuing a fixed top-k and
        hoping. Zero means stand down entirely.
        """
        with self._lock:
            return max(0, self._spec_inflight_cap - len(self._inflight))

    def _prefetch_locked(self, layer_key: str, expert_ids, submitted: list) -> None:
        """Assumes lock held. Queue reads for ids not already known.

        Consecutive survivors are coalesced into run reads - the same trick
        prefill uses, which matters more here because a predicted top-k is often
        several neighbouring experts, and per-expert reads would cost one pread
        per component each.
        """
        wanted: list[int] = []
        for eid in expert_ids:
            eid = int(eid)
            key = (layer_key, eid)
            if key in self._lru or key in self._raw_ready or key in self._inflight:
                continue
            wanted.append(eid)
        if not wanted:
            return
        wanted.sort()
        if self.slab is not None:
            # Speculation gets free slots only (evict_ok=False): trading a
            # resident expert for a guess is how prefetching starts losing.
            for eid in wanted:
                fut = self._submit_slab(layer_key, eid, evict_ok=False)
                if fut is None:
                    break
                submitted.append(((layer_key, eid), fut))
            return
        per = max(1, self.expert_nbytes(layer_key))
        max_run = max(1, self._max_run_bytes(layer_key) // per)
        run: list[int] = [wanted[0]]
        for eid in wanted[1:]:
            if eid == run[-1] + 1 and len(run) < max_run:
                run.append(eid)
            else:
                submitted.extend(
                    ((layer_key, e), f) for e, f in self._submit_run(layer_key, run)
                )
                run = [eid]
        submitted.extend(
            ((layer_key, e), f) for e, f in self._submit_run(layer_key, run)
        )

    def prefetch_many(self, batch) -> None:
        """prefetch() for several layers under one lock acquisition.

        Route prediction produces several layers' worth of ids at once, and this
        runs once per layer per token - so taking and dropping the lock per layer
        is overhead on exactly the path that has to stay nearly free.
        """
        # Opt-in (EXPERT_STREAM_PREDICT_INSTALL): measured on Qwen3-235B this
        # is only safe when prediction precision is high - wrong guesses that
        # install evict resident experts, and at precision ~0.5 that cache
        # pollution doubled per-token reads (2.2 GB/token) and cost 15%
        # against the staging path. Keep off unless precision is ~0.8+.
        install_now = self.slab is not None and config.PREDICT_INSTALL
        submitted: list[tuple[tuple, Future]] = []
        with self._lock:
            if self._spec_blocked(staging=not install_now):
                return
            for layer_key, ids in batch:
                self._prefetch_locked(layer_key, ids, submitted)
        self._attach_staging(submitted, install_now=install_now)

    def prefetch(self, layer_key: str, expert_ids):
        """Start reads for experts we expect to want, without waiting."""
        submitted: list[tuple[tuple, Future]] = []
        with self._lock:
            if self._spec_blocked():
                return
            self._prefetch_locked(layer_key, expert_ids, submitted)
        self._attach_staging(submitted)

    def _attach_staging(self, submitted: list, install_now: bool = False) -> None:
        """Stage each read's result for whoever asks for it later.

        `install_now` (route-predicted slab reads only): insert into the LRU
        the moment the read lands, instead of parking it in staging. Measured
        on Qwen3-235B, prune-aware prediction is ~77% precise - these reads
        *are* demand, three layers early - while staging was designed for the
        36%-precision guess stream and capped accordingly, so correct
        predictions were dropped before their layer ran (53/token) and then
        re-read from disk as blocking misses. A slab entry is just a slot
        number (no mx call needed), so installing from the reader callback is
        safe; the wrong ~23% age out of the LRU like any cold entry.
        """
        # Attach callbacks OUTSIDE the lock: if a future is already complete,
        # add_done_callback runs the callback inline on this thread, and the
        # callback takes the lock (deadlock if we still held it).
        for key, fut in submitted:

            def _done(f, key=key):
                try:
                    raw = f.result()
                except Exception:
                    with self._lock:
                        self._inflight.pop(key, None)
                    return
                with self._lock:
                    self._inflight.pop(key, None)
                    if key in self._lru:
                        return
                    if install_now and "__slot__" in raw:
                        # Count the read here; a later fetch that finds this
                        # entry sees a plain LRU hit (see _finish, which skips
                        # double-counting via __counted__).
                        raw["__counted__"] = True
                        self.bytes_read += raw["__nbytes__"]
                        self._install(key, raw, "lru")
                        return
                    self._raw_ready[key] = raw
                    self._raw_bytes += raw["__nbytes__"]
                    self._raw_ready.move_to_end(key)
                    while self._raw_ready and (
                        self._raw_bytes > self.staging_bytes
                        or len(self._raw_ready) > _MAX_RAW_READY
                    ):
                        oldest = next(iter(self._raw_ready))
                        dropped = self._drop_raw(oldest)
                        self.staging_drops += 1
                        if dropped is not None:
                            self.spec_wasted_bytes += dropped["__nbytes__"]
                            _release_raw(self._buffers, dropped)

            fut.add_done_callback(_done)

    def relieve_pressure(self):
        """Drop prefetch staging + Metal buffer cache between generations.

        Keeps the expert LRU (so the next turn stays warm) but returns spare
        Metal pages and abandons speculative numpy copies. Call this after
        each chat turn / HTTP response.
        """
        with self._lock:
            self.spec_wasted_bytes += self._raw_bytes
            # Via _drop_raw so slab slots held by unclaimed guesses come back.
            for key in list(self._raw_ready):
                raw = self._drop_raw(key)
                if raw is not None:
                    _release_raw(self._buffers, raw)
            self._raw_bytes = 0
            self._evicted_since_clear = 0
            self._pending_clear = False
        # Between turns, hand the recycled read buffers back to the OS too.
        self._buffers.clear()
        mx.clear_cache()

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "resident_experts": len(self._lru),
            "resident_gb": round(self.resident_bytes / 1e9, 3),
            "peak_resident_gb": round(self.peak_resident_bytes / 1e9, 3),
            "budget_gb": round(self.budget_bytes / 1e9, 3),
            "evictions": self.evictions,
            "slab_gb": round(self.slab.nbytes / 1e9, 3) if self.slab else 0.0,
            "slab_slots": self.slab.slots if self.slab else 0,
            "slab_free": len(self._free_slots),
            "slot_starved": self.slot_starved,
            "read_gb": round(self.bytes_read / 1e9, 3),
            # Whole-layer prefill fetches. Divide by the model's MoE layer
            # count for the number of passes the prompt made over the expert
            # mass - prefill time is proportional to that.
            "prefill_layers": self.prefill_layers,
            # Of those bytes, the ones a layer was blocked on. The gap between
            # this and read_gb is what speculation bought or wasted.
            "demand_gb": round(self.demand_miss_bytes / 1e9, 3),
            "disk_wait_s": round(self.disk_wait_s, 3),
            "materialize_s": round(self.materialize_s, 3),
            "prefetch_pending": len(self._raw_ready),
            "staging_gb": round(self._raw_bytes / 1e9, 3),
            "staging_drops": self.staging_drops,
            # Bytes read speculatively and thrown away unused - the cost side
            # of prediction, against read_gb as the useful side.
            "spec_wasted_gb": round(self.spec_wasted_bytes / 1e9, 3),
            "spec_skipped": self.spec_skipped,
            "spec_s": round(self.spec_s, 3),
            # Of the experts route prediction asked for, how many the model then
            # actually used (precision), and how much of real demand it saw
            # coming (coverage).
            "pred_precision": (
                round(self.pred_used / self.pred_issued, 4) if self.pred_issued else 0.0
            ),
            "pred_coverage": (
                round(self.pred_used / self.route_total, 4) if self.route_total else 0.0
            ),
            # Same denominator, for the previous-token heuristic alone.
            "heur_coverage": (
                round(self.heur_used / self.route_total, 4) if self.route_total else 0.0
            ),
            "pred_issued": self.pred_issued,
            # Fraction of routed (token, expert) slots skipped by decode
            # pruning (0 unless EXPERT_STREAM_PRUNE is set).
            "pruned_frac": (
                round(self.pruned_slots / self.demand_slots, 4)
                if self.demand_slots
                else 0.0
            ),
            # Experts a weight test dropped but residency rescued: quality
            # recovered for free (see config.KEEP_FREE).
            "kept_free": self.kept_free,
            # Experts skipped instead of blocking the GPU on a read, and
            # prefetched for later tokens instead (see config.WAIT_ABOVE).
            "skipped_waits": self.skipped_waits,
            # Share of layer calls that read the router's weights off the
            # block's own gate call rather than re-running it.
            "route_tap_rate": (
                round(self.route_tapped / self.route_calls, 4)
                if self.route_calls
                else 0.0
            ),
            "mx_active_gb": round(mx.get_active_memory() / 1e9, 3),
            "mx_cache_gb": round(mx.get_cache_memory() / 1e9, 3),
            "mx_peak_gb": round(mx.get_peak_memory() / 1e9, 3),
        }
