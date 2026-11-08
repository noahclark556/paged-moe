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
import weakref
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)

import numpy as np

import mlx.core as mx

from . import activity, config, gpu_trail, mempressure, pagecache, turn_trace
from .config import _env
from .slab import SlabStore, supported_dtype
from .safetensors_index import (
    _DTYPES,
    MAX_IO_BYTES,
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
# (PAGED_MOE_SLICE_MB): small slices minimize one cold expert's latency,
# but on a saturated pipe (big-expert decode) every slice is a Future plus a
# GIL window, and the scheduling overhead competes with the GPU thread.
_SLICE_BYTES = int(_env("PAGED_MOE_SLICE_MB", 2, float) * (1 << 20))

# Prefill reads whole layers, so it has no latency to minimize and plenty of
# parallelism already (many spans, many components, two groups in flight). Its
# scarce resource is bytes per syscall / per GIL window, so it slices coarsely.
_PREFILL_SLICE_BYTES = config.PREFILL_SLICE_BYTES

# Slot reads only fan a component to the slice pool at or above this size;
# smaller ones read inline on the expert worker. See _read_slab_inner.
# 0 = fan everything; a huge value = fully inline. Both kept for A/B.
_FANOUT_BYTES = int(_env("PAGED_MOE_FANOUT_MB", 1.0, float) * (1 << 20))


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
        """Stop the workers. Unused by the server (one process per model), but
        tests that build caches in a loop need it to avoid leaks."""
        for _ in range(self.threads):
            self._q.put(None)


def _pread_full(fd: int, offset: int, mv: memoryview) -> None:
    """pread `mv`'s full length at `offset`. One syscall in the normal case.

    Separate from FilePool.read_into, which resolves path->fd and rebuilds a
    memoryview each call. The miss path already has both from the read plan,
    and this runs thousands of times per token.

    Prefers the native helper when loaded (GIL released for the short-read
    loop); otherwise os.preadv. Same bytes either way. I/O errors always
    propagate; only a failed native *load* falls back to Python.
    """
    if config.NATIVE_READ != "off":
        try:
            from . import native_read as _nr

            if _nr.pread_full(fd, offset, mv):
                return
        except OSError:
            raise
        except Exception:  # noqa: BLE001 - native load/API failure only
            pass
    while True:
        n = os.preadv(fd, [mv[:MAX_IO_BYTES]], offset)
        if n <= 0:
            raise OSError(f"short read at {offset} (fd {fd})")
        if n >= len(mv):
            return
        mv = mv[n:]
        offset += n


# Left free when sizing a slab rebuild: the next token's activations, the KV
# the rest of this turn will append, and Metal's own recycled-buffer pool.
_SLAB_MARGIN = 3 << 30
# What streaming asks make_room_for for even a tiny prefill chunk.
_PREFILL_NEED_FLOOR = 512 << 20
# Per-token prefill need before any chunk has been measured. Flash
# (hyper-connections, 4 residual streams) measured ~1.3 MB/token at 1300 tok;
# _SLAB_MARGIN covers the gap until the first measurement replaces this.
_PREFILL_TOKEN_PRIOR = 1 << 20
# On top of a measured peak, for the context the next chunk lands on.
_PREFILL_NEED_MARGIN = 1.1
# A rebuilt slab keeps the last prefill's need free for the next one, up to
# about a 2.5k-token tool result on Flash. One big result must not shrink
# every later slab, and the slab is only rebuilt after it was given up.
_PREFILL_NEED_CAP = 4 << 30
# Smallest regrow worth dropping residency for (with 10% of the slab).
_REGROW_MIN_BYTES = 1 << 30
# PREFILL_AHEAD stands down while the share of its reads the next layer routed
# is under this, over a window that decays per layer (so it re-probes once the
# bad samples age out). Measured 0.86-0.97 on the shipped models; a wrong
# guess only costs a read in otherwise idle disk time, so the bar is low.
_AHEAD_MIN_PRECISION = 0.5
_AHEAD_MIN_SAMPLE = 512
_AHEAD_DECAY = 0.98


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


_PEAK_SEEN = 0


def note_peak() -> int:
    """Highest Metal peak this process has reached.

    begin_prefill_measure resets MLX's own counter, so stats read this.
    """
    global _PEAK_SEEN
    _PEAK_SEEN = max(_PEAK_SEEN, int(mx.get_peak_memory()))
    return _PEAK_SEEN


def _stage_result(fut: Future) -> bool:
    """True when `fut` has landed a stage-slot entry."""
    if not fut.done() or fut.cancelled() or fut.exception() is not None:
        return False
    raw = fut.result()
    return isinstance(raw, dict) and "__stage__" in raw


def _read_comps(comps, entry: dict) -> dict:
    """Plain preads into prepared destinations (no read pool configured)."""
    for fd, off, mv in comps:
        _pread_full(fd, off, mv)
    return entry


def _release_raw(pool: BufferPool, raw: dict) -> None:
    """Return a raw entry's pooled buffers. Idempotent (pop-once)."""
    bufs = raw.pop("__bufs__", None)
    if bufs:
        pool.release(bufs)


def _materialize(raw: dict) -> dict:
    """numpy -> mx. MUST run on the MLX/GPU thread (usually main).

    Only the run-read path lands here. A prefill that keeps the slab reads its
    misses into slots instead (config.PREFILL_SLAB), which removes this copy
    for real because compute takes range-slice views of the slots.

    Getting rid of it any other way has been tried. MLX-backed run spans made
    the copy vanish from the accounting (16.7s -> 0.2s) and total time did not
    move: `span[i]` is a gather in MLX, so the same copy came back at eval
    under compute. An expert view has to be a range slice (see
    SlabStore.expert) or the copy is still there, just not counted.
    """
    if "__slot__" in raw or "__stage__" in raw:
        # Slab or stage read: the bytes were pread straight into their final
        # slot in Metal-backed memory, so there is no copy to make and nothing
        # to evaluate. The entry is just the slot number; compute addresses it
        # by index (streaming._run_slab) or takes slot views (_loose).
        return raw
    out = {"__nbytes__": raw["__nbytes__"]}
    for name, value in raw.items():
        # __need_at__ and the other bookkeeping fields are not weights.
        if name.startswith("__"):
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


_ping: list = []


def _gpu_ping():
    """Tiny eval that keeps the GPU out of its idle clock state."""
    if not _ping:
        z = mx.zeros((1024,), mx.float32)
        mx.eval(z)
        _ping.append(z)
    mx.eval(_ping[0] + 1)


def _completed_warm(futs, every_s: float):
    """as_completed, but pings the GPU every `every_s` while it waits.

    Call from the thread that runs the model (where _finish runs). A failed
    ping turns pinging off and the wait carries on as plain as_completed.
    """
    pending = set(futs)
    last = time.perf_counter()
    while pending:
        done, pending = wait(pending, timeout=every_s, return_when=FIRST_COMPLETED)
        yield from done
        if pending and time.perf_counter() - last >= every_s:
            try:
                _gpu_ping()
            except Exception:
                yield from as_completed(pending)
                return
            last = time.perf_counter()


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
    # Prefill stage defaults, for caches built without __init__ (tests).
    stage = None
    reads_promoted = 0
    _stage_frozen = False
    _stage_refused = False
    _slab_per_expert = 0
    _stage_group = 0
    slab = None
    _slab_pending = None

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
        # Decode layer counter. Prefetch stamps the clock it expects to be
        # consumed at so a full staging area drops the furthest guess, not
        # the one the next layer is about to block on.
        self._prefetch_clock = 0
        self._prefetch_need: dict[tuple, int] = {}
        self.reads_promoted = 0
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
        # May be _ReadPool or native_read.NativeReadPool (same submit surface
        # plus optional submit_expert).
        self._reads = None
        self._native_reads = False
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
        # Experts stitch_load pinned into fixed slots. Eviction skips them.
        # Empty unless the slab holds every routed expert (config.STITCH).
        self._stitch_keys: set[tuple] = set()
        self._stitch_tables: dict[str, mx.array] = {}
        self.stitched = False
        self.stitch_forwards = 0
        self.full_stitch_tokens = 0
        self.closed_ok = 0
        self.closed_replay = 0
        # Slots of dropped guesses that were still pinned, by slot. Freed when
        # the pins clear if nothing has claimed them (see _clear_pins_locked).
        self._unfreed: dict[int, tuple] = {}
        self.slots_reclaimed = 0
        self.slot_skipped_installs = 0
        self.slot_starved = 0
        # Prefill misses read into free slab slots vs. left to run reads.
        self.prefill_slot_reads = 0
        self.prefill_slot_starved = 0
        # Next-layer prefill reads started from a router prediction, by layer
        # key until that layer settles them (see prefill_ahead).
        self._ahead: dict[str, set[int]] = {}
        self._ahead_orphans: dict[tuple, Future] = {}
        self.prefill_ahead_reads = 0
        self.prefill_ahead_used = 0
        self.prefill_ahead_wasted_bytes = 0
        self.prefill_ahead_skipped = 0
        self._ahead_n = 0.0
        self._ahead_u = 0.0
        # PAGED_MOE_SLAB_VERIFY: writes in flight per slot, so compute can
        # assert nobody is overwriting what it is about to read.
        self._verify = bool(config.SLAB_VERIFY)
        self._writing: dict[int, int] = {}
        self.slab_conflicts = 0
        self._evicted_since_clear = 0
        self._pending_clear = False
        # Set while a big prefill pass runs on a shrunk budget (see
        # enter_prefill); holds the decode-time budget to restore.
        self._decode_budget: int | None = None
        # Slot count the slab was sized for at load. Rebuilds scale from this,
        # not from the last slab, so a trimmed slab can grow back.
        self._slab_full_slots = 0
        self._pressure = mempressure.SlabGovernor()
        self._rebuild_worst = 0
        self._regrow_after = 0.0
        # KV the live conversation can still grow by (server sets it per
        # request from numCtx, kvmem.kv_room) and the last prefill's own need
        # (make_room_for). A slab built or regrown leaves both free.
        self.kv_room_bytes = 0
        self._turn_need = 0
        # Measured Metal peak above the start of recent kept-slab prefill chunks,
        # as (tokens, bytes); prefill_need prices the next chunk from these.
        self._prefill_peaks: deque[tuple[int, int]] = deque(maxlen=16)
        self._measure: tuple[int, int, int] | None = None
        self.chunk_need = _PREFILL_NEED_FLOOR
        self.prefill_layers_skipped = 0
        # Prefill stage (config.PREFILL_STAGE): slot store for prefill misses,
        # alive from the first prefill read to decode. `_stage_gen` tells an
        # entry from a stage that has since been released apart.
        self.stage: SlabStore | None = None
        self._stage_group = 0
        self._stage_free: list[int] = []
        self._stage_futs: set[Future] = set()
        self._stage_gen = 0
        self._stage_frozen = False
        self._stage_refused = False
        self.stage_reads = 0
        self.stage_starved = 0
        self.stage_builds = 0

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
        # Stitch applies the same mask on the GPU. The drop count stays a
        # graph until the token finishes, then one async eval, so the stat
        # updates without a per-layer sync.
        self._stitch_drop = None
        self._stitch_demand = 0
        self._stitch_pending: list[tuple] = []
        # Slots a weight test dropped and residency put back (KEEP_FREE).
        self.kept_free = 0
        # Experts skipped rather than waited on (WAIT_ABOVE).
        self.skipped_waits = 0
        # Decode misses on experts WAIT_ABOVE says are worth blocking for, per
        # MoE layer step. The per-layer path waits on these; flow drops them.
        self.crit_misses = 0
        self.crit_steps = 0
        # Share of the router's mixture each decode path actually computed,
        # summed over MoE layer steps (flow's handover compares the two).
        self.mass_sync = 0.0
        self.rows_sync = 0
        self.mass_flow = 0.0
        self.rows_flow = 0
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
        ref = weakref.ref(self)
        gpu_trail.probe("cache", lambda: ref().trail_state() if ref() is not None else "cache: gone")

    def trail_state(self) -> str:
        """One line of read and slab state for a stall report.

        Runs on the watchdog thread while decode may be hung holding the
        lock, so it only reads counters and container sizes, never the lock.
        """
        reads = list(self._inflight.values()) + list(self._ahead_orphans.values())
        open_reads = sum(1 for f in reads if not f.done())
        slab = self.slab
        slab_s = (f"slab {slab.slots} slots, {len(self._free_slots)} free, {len(self._pinned)} pinned"
                  if slab is not None else
                  f"no slab ({'released, rebuild pending' if self._slab_pending else 'per-expert path'})")
        stage = self.stage
        return (f"cache: {open_reads} of {len(reads)} reads still running, {slab_s}"
                f"{', frozen' if self._slab_frozen else ''}"
                f"{f', stage {stage.slots} slots' if stage is not None else ''}, "
                f"resident {self.resident_bytes / 1e9:.1f}/{self.budget_bytes / 1e9:.1f} GB, "
                f"flow tables {len(self._flow_tables)}, clear pending {self._pending_clear}")

    def register_layer(self, layer_key: str, components: dict[str, ExpertLocator]):
        self._layouts[layer_key] = components
        self._expert_nbytes[layer_key] = sum(
            loc.loc(0).nbytes for loc in components.values()
        )

    def expert_nbytes(self, layer_key: str) -> int:
        """Approximate bytes for one expert of this layer (for group sizing)."""
        return self._expert_nbytes[layer_key]

    def _expert_counts(self) -> dict[str, int] | None:
        """Experts per layer, or None if a layer's layout is not indexable."""
        counts: dict[str, int] = {}
        for layer_key, comps in self._layouts.items():
            locator = next(iter(comps.values()))
            if isinstance(locator, StackedLocator):
                counts[layer_key] = int(locator.stacked.shape[0])
            elif isinstance(locator, DirectLocator):
                counts[layer_key] = len(locator.per_expert)
            else:
                return None
            if counts[layer_key] <= 0:
                return None
        return counts or None

    def stitch_table(self, layer_key: str):
        """Static expert->slot table for a stitched layer, or None."""
        if not self.stitched:
            return None
        return self._stitch_tables.get(layer_key)

    def load_stitch(self) -> bool:
        """Read every routed expert into a fixed slot. True if decode can
        gather without a per-layer sync.

        Declines, leaving the cache as it was, when the slab cannot hold the
        whole set or a read fails. The slider's prune still applies in the
        forward when KEEP_FREE is off. Call on the MLX thread after enable_slabs.
        """
        if not config.STITCH or self.slab is None or self.stitched:
            return bool(self.stitched)
        counts = self._expert_counts()
        if counts is None:
            return False
        total = sum(counts.values())
        if self.slab.slots < total:
            return False
        jobs: list[tuple[tuple, Future]] = []
        with self._lock:
            for layer_key, n in counts.items():
                for eid in range(n):
                    fut = self._submit_slab(layer_key, eid, evict_ok=False)
                    if fut is None:
                        jobs.clear()
                        break
                    jobs.append(((layer_key, eid), fut))
                else:
                    continue
                break
        if len(jobs) != total:
            return False
        t0 = time.perf_counter()
        loaded: list[tuple[tuple, dict]] = []
        try:
            for key, fut in jobs:
                loaded.append((key, fut.result()))
        except Exception as e:
            print(f"[paged-moe] stitch declined: {e}", flush=True)
            return False
        tables: dict[str, np.ndarray] = {
            layer_key: np.full(n, -1, dtype=np.int32) for layer_key, n in counts.items()
        }
        with self._lock:
            for key, entry in loaded:
                self._inflight.pop(key, None)
                self._stitch_keys.add(key)
                self._install(key, entry, "lru")
                layer_key, eid = key
                slot = entry.get("__slot__")
                if slot is None:
                    self._stitch_keys.clear()
                    return False
                tables[layer_key][eid] = int(slot)
        if any(int(row.min()) < 0 for row in tables.values()):
            self._stitch_keys.clear()
            return False
        # The reserve keeps some slots empty for prefetch. A complete set has
        # nothing left to prefetch, and the install above may have landed over
        # the old budget. Cover every slot so a later eviction pass does not
        # try to undo the set.
        self.budget_bytes = self.slab.slots * max(1, self._slab_per_expert)
        self._slab_budget = self.budget_bytes
        self._stitch_tables = {k: mx.array(v) for k, v in tables.items()}
        self.stitched = True
        gb = self.resident_bytes / 1e9
        dt = time.perf_counter() - t0
        print(
            f"[paged-moe] stitch: {total} experts, {gb:.2f} GB in {dt:.1f}s. "
            "Decode gathers from the slab in one graph.",
            flush=True,
        )
        return True

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

    def flow_on(self) -> bool:
        """Slot tables are published (FLOW or DRAFT_FLOW enabled them)."""
        return self._flow_rows is not None

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

    def flow_admit(self, batch) -> list[Future]:
        """Flow decode's residency update, once per token.

        `batch` is last token's real routing, [(layer_key, ids)]. Resident ids
        get the same recency and reuse credit a fetch hit would; missing ones
        are read straight into slots and installed the moment they land.

        This has to install and has to be allowed to evict. Flow never calls
        fetch, and fetch is the only thing that promotes staged reads, so
        routing through prefetch left every read in staging until it was
        dropped: the resident set froze at whatever the warmup tokens loaded
        (465 experts of a 6126-slot slab on qwen38-coder-flash, hit rate 0.04).

        The caller must have fenced the previous token's graph first. Evicting
        frees a slot and the read below overwrites it, so nothing still pending
        may read it. In flow mode this is the only place reads are submitted,
        and the slots it frees are cleared from the tables before the next
        token snapshots them.

        Returns the reads it issued (nothing waits on them in decode).
        """
        if self.slab is None:
            return []
        submitted: list[tuple[tuple, Future]] = []
        with self._lock:
            self._clear_pins_locked()
            for layer_key, ids in batch:
                for eid in ids:
                    eid = int(eid)
                    key = (layer_key, eid)
                    if key in self._lru:
                        self.hits += 1
                        self._lru.move_to_end(key)
                        self._freq[key] = min(_FREQ_MAX, self._freq.get(key, 0) + 1)
                        continue
                    self.misses += 1
                    if key in self._inflight:
                        continue
                    raw = self._drop_raw(key, claim=True)
                    if raw is not None:
                        if not raw.pop("__counted__", False):
                            self.bytes_read += raw["__nbytes__"]
                        self._install(key, raw, "lru")
                        continue
                    fut = self._submit_slab(layer_key, eid, evict_ok=True)
                    if fut is None:
                        break
                    submitted.append((key, fut))
        self._attach_staging(submitted, install_now=True)
        return [fut for _, fut in submitted]

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
        self._slab_full_slots = slots
        mempressure.start()
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
            self._reads = self._make_read_pool(self._read_pool_threads)

    def _make_read_pool(self, threads: int):
        """Prefer the native C pool; fall back to Python _ReadPool.

        Never raises for a missing extension: decode must keep working on any
        Mac that can run the rest of the engine.
        """
        self._native_reads = False
        if config.NATIVE_READ != "off":
            try:
                from . import native_read as _nr

                mod = _nr.get_module()
                if mod is not None:
                    pool = _nr.NativeReadPool(threads, mod=mod)
                    self._native_reads = True
                    state, detail = _nr.status()
                    print(
                        f"[paged-moe] native read pool: {state} ({detail}), "
                        f"threads={pool.threads}",
                        flush=True,
                    )
                    return pool
                state, detail = _nr.status()
                print(
                    f"[paged-moe] native read pool unavailable "
                    f"({state}: {detail}); using Python pool",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001 - hard fallback
                print(
                    f"[paged-moe] native read pool error ({type(e).__name__}: {e}); "
                    f"using Python pool",
                    flush=True,
                )
        return _ReadPool(threads)

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
            if key in self._stitch_keys or self._slot_of.get(key) in self._pinned:
                continue
            self._evict_key(key)
            if self._free_slots:
                return self._free_slots.pop()
        # Nothing evictable: give up unclaimed prefetch guesses instead.
        # Furthest first. The oldest staged expert is usually the next
        # layer's, and taking that slot turns a finished prefetch into a miss.
        while self._raw_ready:
            victim = self._staged_victim()
            if victim is None:
                break
            self._discard_staged(victim)
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
        self, layer_key: str, expert_id: int, evict_ok: bool = True, urgent: bool = False
    ) -> Future | None:
        """Assumes lock held. Start a slot read, or None if no slot is free.

        `urgent` = a layer is blocked on this read; the native pool serves it
        ahead of queued speculation.
        """
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

        # Planned path: push reads onto the slot-read pool. Native path
        # submits one job for the whole expert (GIL released for every
        # component pread, single latch.done). Python path submits one job
        # per component into _ReadPool. Same bytes either way.
        fut: Future = Future()
        fut.set_running_or_notify_cancel()
        entry = {"__nbytes__": self.slab.expert_bytes, "__slot__": slot}
        on_done = None
        if self._verify:
            self._writing[slot] = self._writing.get(slot, 0) + 1

            def on_done(slot=slot):
                with self._lock:
                    self._writing[slot] -= 1

        dests = self.slab.slot_dests[slot]
        if self._native_reads and hasattr(self._reads, "submit_expert"):
            latch = _SlotLatch(1, fut, entry, on_done)
            comps = [
                (fd, base + expert_id * stride, mv)
                for (fd, base, stride), mv in zip(plan, dests)
            ]
            try:
                if urgent:
                    self._reads.submit_expert(comps, latch, urgent=True)
                else:
                    self._reads.submit_expert(comps, latch)
                    fut._es_latch = latch
            except Exception as e:  # noqa: BLE001 - never drop a miss
                # Last-resort: finish on the calling thread so fetch still
                # resolves instead of wedging the inflight table.
                latch.done(e)
        else:
            latch = _SlotLatch(len(plan), fut, entry, on_done)
            submit = self._reads.submit
            for (fd, base, stride), mv in zip(plan, dests):
                submit(fd, base + expert_id * stride, mv, latch)
        self._inflight[key] = fut
        return fut

    # ------------------------------------------------------- prefill stage

    def _stage_specs(self):
        if self.slab is not None:
            return self.slab.specs
        if self._slab_pending is not None:
            return self._slab_pending[0]
        return None

    def _stage_group_bytes(self) -> int:
        """Group size the stage was built for (configured size before that)."""
        if self.stage is not None and self._stage_group:
            return self._stage_group
        per = max(1, self._slab_per_expert)
        return max(int(config.PREFILL_STAGE_GROUP_MB) << 20, 8 * per)

    def ensure_stage(self) -> bool:
        """Bring the prefill stage up if it isn't. MLX thread. False = the
        run-read path for this prefill (no layout, no room, or turned off)."""
        if self.stage is not None:
            return True
        if not config.PREFILL_STAGE or self._stage_refused or self._stage_frozen:
            return False
        per = self._slab_per_expert
        specs = self._stage_specs()
        if not (specs and per and self._read_plan):
            return False
        if self.slab is not None and mempressure.level() >= mempressure.WARN:
            return False
        # Four groups (see config.PREFILL_STAGE_GROUP_MB), halving the group
        # until they fit what is free, never under 8 experts a group.
        group = max(int(config.PREFILL_STAGE_GROUP_MB) << 20, 8 * per)
        room = None
        if self._ram_budget_bytes:
            # Next to the slab, the chunk's own activations come first.
            need = self.chunk_need if self.slab is not None else 0
            room = self._ram_budget_bytes - mx.get_active_memory() - _SLAB_MARGIN - need
            while 4 * group > room and group // 2 >= 8 * per:
                group //= 2
            if 4 * group > room:
                return False
        slots = max(32, (4 * group) // per)
        try:
            store = SlabStore(specs, slots)
        except Exception as e:  # noqa: BLE001 - the run-read path still works
            self._stage_refused = True
            print(f"[paged-moe] prefill stage off: {type(e).__name__}: {e}", flush=True)
            return False
        with self._lock:
            self.stage = store
            self._stage_group = group
            self._stage_free = list(range(slots - 1, -1, -1))
            self._stage_futs = []
            self._stage_gen += 1
        self.stage_builds += 1
        if self.stage_builds == 1:
            print(
                f"[paged-moe] prefill stage: {slots} slots ({store.nbytes / 1e9:.2f} GB), "
                f"groups of {group >> 20} MB, native reads={self._native_reads}",
                flush=True,
            )
        return True

    def _submit_stage(self, layer_key: str, expert_id: int, urgent: bool = False) -> Future | None:
        """Assumes lock held. Read one expert into a free stage slot, or None."""
        stage = self.stage
        if stage is None or self._stage_frozen or not self._stage_free:
            return None
        plan = self._read_plan.get(layer_key)
        if plan is None:
            return None
        slot = self._stage_free.pop()
        key = (layer_key, expert_id)
        entry = {"__nbytes__": stage.expert_bytes, "__stage__": slot, "__gen__": self._stage_gen}
        dests = stage.slot_dests[slot]
        comps = [(fd, base + expert_id * stride, mv) for (fd, base, stride), mv in zip(plan, dests)]
        if self._reads is None:
            fut = self._executor.submit(_read_comps, comps, entry)
        else:
            fut = Future()
            fut.set_running_or_notify_cancel()
            if self._native_reads and hasattr(self._reads, "submit_expert"):
                latch = _SlotLatch(1, fut, entry)
                try:
                    self._reads.submit_expert(comps, latch, urgent=urgent)
                except Exception as e:  # noqa: BLE001 - never drop a miss
                    latch.done(e)
            else:
                latch = _SlotLatch(len(comps), fut, entry)
                for fd, off, mv in comps:
                    self._reads.submit(fd, off, mv, latch)
        # Waited on by release_stage. No done-callback: a read that fails
        # inside this call would run it under the lock we hold.
        self._stage_futs.append(fut)
        self._inflight[key] = fut
        self.stage_reads += 1
        return fut

    def _stage_return_locked(self, entry) -> None:
        """Assumes lock held. A stage slot nothing reads any more goes back."""
        if (
            isinstance(entry, dict)
            and "__stage__" in entry
            and self.stage is not None
            and entry.get("__gen__") == self._stage_gen
        ):
            self._stage_free.append(entry["__stage__"])

    def _stage_return(self, entries) -> None:
        if self.stage is None:
            return
        with self._lock:
            for entry in entries:
                self._stage_return_locked(entry)
            if len(self._stage_futs) > 4 * self.stage.slots:
                self._stage_futs = [f for f in self._stage_futs if not f.done()]

    def release_stage(self) -> None:
        """Give the stage's memory back once its reads have landed. MLX thread,
        no pending graph on it (each prefill group was evaluated)."""
        if self.stage is None:
            return
        with self._lock:
            self._stage_frozen = True
            pending = list(self._stage_futs)
        for fut in pending:
            try:
                fut.result()
            except Exception:
                pass
        with self._lock:
            try:
                # Every read that landed in the stage, not just the tracked
                # ones: _stage_return prunes finished futures from the list,
                # and an unclaimed lookahead read is one of those.
                done = set(pending)
                for key, fut in list(self._inflight.items()):
                    if fut in done or _stage_result(fut):
                        self._inflight.pop(key, None)
                if self.stage is not None:
                    self.stage.release()
                self.stage = None
                self._stage_free = []
                self._stage_futs = []
                self._stage_gen += 1
            finally:
                self._stage_frozen = False

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
        sequential reads instead of len(ids) x the component count of small
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

    def resident_ids(self, layer_key: str, expert_ids) -> set[int]:
        """Which of these experts are installed now (what flow can compute)."""
        with self._lock:
            return {
                int(e) for e in expert_ids if (layer_key, int(e)) in self._lru
            }

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
                # then overwrites the expert in place. But every slab read is
                # pinned at submit, so a guess dropped before the pins clear
                # looks the same, and skipping it for good lost the slot until
                # the next slab rebuild (5% of a 250-slot slab on the first
                # token: slot_leak_probe.py). Those wait in _unfreed.
                if slot is not None:
                    if slot in self._pinned:
                        self._unfreed[slot] = key
                    else:
                        self._slot_of.pop(key, None)
                        self._free_slots.append(slot)
        return raw

    def _staged_victim(self, only_after: int | None = None):
        """Assumes lock held. Which unclaimed guess to throw away.

        A tagged guess records the prefetch clock of the layer that will want
        it. Drop the furthest one. The soonest is about to be claimed, and
        dropping it re-reads those bytes as a blocking miss. Untagged guesses
        keep FIFO order. `only_after` will not drop a guess due at or before
        that clock, so a near prefetch can take a slot from a far one only.
        """
        best_key = None
        best_need = None
        untagged = None
        for key, raw in self._raw_ready.items():
            need = raw.get("__need_at__")
            if need is None:
                if untagged is None:
                    untagged = key
                continue
            if only_after is not None and need <= only_after:
                continue
            if best_need is None or need > best_need:
                best_need = need
                best_key = key
        if best_key is not None:
            return best_key
        if only_after is not None:
            return untagged
        return untagged

    def _discard_staged(self, key) -> None:
        """Assumes lock held. Forget one unclaimed guess and count it as waste."""
        dropped = self._drop_raw(key)
        if dropped is None:
            return
        self.staging_drops += 1
        self.spec_wasted_bytes += dropped["__nbytes__"]
        _release_raw(self._buffers, dropped)
        self._prefetch_need.pop(key, None)

    def _trim_staging(self, only_after: int | None = None) -> None:
        """Assumes lock held. Bring staging back under its cap."""
        while self._raw_ready and (
            self._raw_bytes > self.staging_bytes
            or len(self._raw_ready) > _MAX_RAW_READY
        ):
            victim = self._staged_victim(only_after)
            if victim is None:
                break
            self._discard_staged(victim)

    def _clear_pins_locked(self) -> None:
        """Assumes lock held, and no fetch in progress: the previous fetch has
        installed what it read. A dropped guess's slot that nothing holds now
        is free. Anything still holding one frees it through its own path."""
        self._pinned.clear()
        if not self._unfreed:
            return
        held = set(self._free_slots)
        held.update(e["__slot__"] for e in self._lru.values() if "__slot__" in e)
        held.update(r["__slot__"] for r in self._raw_ready.values() if "__slot__" in r)
        for key in list(self._inflight) + list(self._ahead_orphans):
            slot = self._slot_of.get(key)
            if slot is not None:
                held.add(slot)
        for slot, key in self._unfreed.items():
            if slot in held:
                continue
            if self._slot_of.get(key) == slot:
                del self._slot_of[key]
            self._free_slots.append(slot)
            self.slots_reclaimed += 1
        self._unfreed.clear()

    def _log_memory(self, where: str) -> None:
        """What Metal is holding, for diagnosing prefill OOMs (MEM_DEBUG)."""
        if not config.MEM_DEBUG:
            return
        print(
            f"[paged-moe mem] {where}: "
            f"active={mx.get_active_memory() / 1e9:.2f} GB "
            f"cached={mx.get_cache_memory() / 1e9:.2f} GB "
            f"peak={note_peak() / 1e9:.2f} GB "
            f"slab={(self.slab.nbytes / 1e9) if self.slab else 0:.2f} GB "
            f"resident={self.resident_bytes / 1e9:.2f} GB "
            f"footprint={mempressure.footprint_gb():.2f} GB "
            f"filecache={(pagecache.file_cache_bytes() or 0) / 1e9:.2f} GB "
            f"trims={pagecache.stats()['trims']}",
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
        to whatever the new tenant left, less the room the next turn will want
        (`_slots_that_fit`, `_turn_room`).

        Callers: streaming, for a small prefill chunk's activations, and the
        full fp16 KV dequantize (`kvmem.unquantize`) where a resumed cache
        still takes that path (hybrid layouts, KV_LAZY=0). That one is the
        largest allocation the server makes outside load: 9.0 GB on GLM-4.7 at
        24k tokens, against a 19 GB slab and a 9.6 GB backbone on a 35 GB
        budget. `PREFILL_SHRINK_TOKENS` cannot catch it, because that trigger
        looks at the *chunk* size and an agent turn appends a few hundred
        tokens to a long context - the memory comes from the context it
        resumes, not from the chunk.

        Yielding costs residency, so it is worth doing only when the alternative
        is an asynchronous Metal command-buffer failure (i.e. a dead request).
        Hence the check rather than an unconditional shrink.
        """
        if nbytes > 0:
            self._turn_need = max(0, int(nbytes) - max(0, int(freeing)))
        if self.slab is None or not self._ram_budget_bytes or nbytes <= 0:
            return False
        # Both the transient peak and the settled figure matter: the conversion
        # is incremental, so at every step it holds some of the new allocation
        # and the not-yet-released remainder of the old one. Charging the full
        # allocation while crediting the full release bounds both.
        headroom = self._ram_budget_bytes - mx.get_active_memory() - _SLAB_MARGIN
        if self.stage is not None:
            # ensure_stage admitted it with this chunk's need set aside.
            headroom += self.stage.nbytes
        if nbytes - max(0, int(freeing)) <= headroom:
            return False
        self.enter_prefill(int(config.PREFILL_CACHE_GB * (1 << 30)))
        return True

    def prefill_need(self, n: int) -> int:
        """Metal memory a kept-slab prefill chunk of `n` tokens will add.

        From the peaks recent chunks measured (begin/end_prefill_measure).
        Two or more sizes: a fixed + per-token line through them, raised until
        it covers every sample. One size: that peak for anything up to its
        length, scaled by length past it (over-counts the fixed part, the safe
        side). None yet: a per-token prior. Recent only, since attention
        scratch grows with the context a chunk lands on.
        """
        n = max(1, int(n))
        pts = list(self._prefill_peaks)
        sizes = {m for m, _ in pts}
        if len(sizes) >= 2:
            ms = np.array([m for m, _ in pts], dtype=np.float64)
            bs = np.array([b for _, b in pts], dtype=np.float64)
            per, fixed = np.polyfit(ms, bs, 1)
            per = max(0.0, float(per))
            fixed = float(np.max(bs - per * ms))
            est = fixed + per * n
        elif pts:
            m, b = max(pts)
            est = b * max(1.0, n / m)
        else:
            est = n * _PREFILL_TOKEN_PRIOR
        return max(_PREFILL_NEED_FLOOR, int(est * _PREFILL_NEED_MARGIN))

    def begin_prefill_measure(self, n: int) -> None:
        """Start timing a kept-slab chunk's memory peak (layer 0 of the chunk)."""
        note_peak()
        mx.reset_peak_memory()
        slab = id(self.slab) if self.slab is not None else 0
        self._measure = (int(n), int(mx.get_active_memory()), slab, self.stage is not None)

    def end_prefill_measure(self) -> None:
        """Record the chunk's peak if the slab stayed up for all of it.

        A stage brought up inside the chunk is left out: ensure_stage prices
        it on its own, against what is left once the chunk's need is set aside.
        """
        m = self._measure
        self._measure = None
        if m is None or self.slab is None or id(self.slab) != m[2] or m[0] < 64:
            return
        peak = max(0, int(mx.get_peak_memory()) - m[1])
        if self.stage is not None and not m[3]:
            peak = max(0, peak - self.stage.nbytes)
        self._prefill_peaks.append((m[0], peak))

    def _slots_that_fit(self, want: int) -> int:
        """Trim the slab to the room the KV cache left us.

        The slab is sized at load, when the only other tenant is the backbone.
        By the time it is rebuilt after a prefill, the conversation's KV cache
        is resident too, and that grows without bound: a 28k-token context on
        Qwen3-235B put the rebuild at 40.3 GB against a 35.05 GB budget, which
        Metal reports as an *asynchronous* command-buffer failure - the server
        dies rather than raising. A slab that is too small only costs a lower
        hit rate, so give the KV cache right of way and take what is left.

        Memory pressure from outside the process scales `want` first. Once the
        kernel has reported warn or worse since the last rebuild, it is also
        capped to what the kernel can hand out now (see mempressure.py). Not
        at normal pressure: macOS keeps little memory free on purpose, so
        free + file cache undercounts what it can give and the cap would
        shrink the slab on every rebuild.
        """
        asked = self._pressure_slots(want)
        if getattr(self, "_rebuild_worst", 0) >= mempressure.WARN:
            want = self._system_slots(asked)
        else:
            want = asked
        if want < asked:
            turn_trace.event("slab_capped", asked=asked, want=want)
            print(
                f"[paged-moe] system memory: "
                f"{(mempressure.available_bytes() or 0) / 1e9:.1f} GB available, "
                f"slab {asked} -> {want} slots",
                flush=True,
            )
        if not (self._ram_budget_bytes and self._slab_per_expert):
            return want
        headroom = (
            self._ram_budget_bytes - mx.get_active_memory() - _SLAB_MARGIN - self._turn_room()
        )
        fits = int(headroom) // self._slab_per_expert
        if fits >= want:
            return want
        return max(32, min(want, fits))

    def _turn_room(self, need: int = 0) -> int:
        """Memory a rebuilt slab leaves free: the next prefill's need (at least
        the floor streaming asks for) plus the KV still to come.

        Without it a slab rebuilt around a long conversation takes exactly what
        is free, and the next turn's make_room_for finds no headroom and
        releases it again: every turn pays a full re-read of the expert mass.
        """
        last = min(self._turn_need, _PREFILL_NEED_CAP)
        need = max(int(need), last, _PREFILL_NEED_FLOOR)
        return need + max(0, int(self.kv_room_bytes))

    def _system_slots(self, want: int, held: int = 0) -> int:
        """Cap `want` to what the kernel can hand out plus `held` (memory of
        ours the new slab replaces), less _SLAB_MARGIN. Never below FLOOR of
        `want`: a bad reading costs some hit rate, not most of the slab.
        """
        if not (config.MEM_PRESSURE and self._slab_per_expert):
            return want
        avail = mempressure.available_bytes()
        if avail is None:
            return want
        room = avail + max(0, int(held)) - _SLAB_MARGIN
        fits = max(0, room) // self._slab_per_expert
        if fits >= want:
            return want
        floor = min(1.0, max(0.0, float(config.MEM_PRESSURE_FLOOR)))
        return max(32, min(want, max(int(fits), round(want * floor))))

    def slab_oversized(self) -> bool:
        """True when the slab no longer fits next to everyone else and a
        rebuild would come out at least 10% smaller. For a turn too short to
        prefill, which would otherwise keep the slab as it is.
        """
        slab = self.slab
        if slab is None or not config.MEM_PRESSURE:
            return False
        if mempressure.level() < mempressure.WARN:
            return False
        full = self._slab_full_slots or slab.slots
        fit = self._system_slots(full, held=slab.nbytes)
        return fit < slab.slots * 0.9

    def _pressure_slots(self, want: int) -> int:
        if not config.MEM_PRESSURE:
            return want
        worst = mempressure.take_worst()
        self._rebuild_worst = worst
        if worst == 0:
            return want
        old = self._pressure.scale
        scale = self._pressure.step(worst)
        slots = min(want, max(32, round(want * scale)))
        before = min(want, max(32, round(want * old)))
        if slots > before:
            # Grow into free memory only. A step up the kernel has to compress
            # for is the same slab as before, just slower.
            fit = max(before, self._system_slots(slots))
            if fit < slots:
                slots = fit
                self._pressure.scale = slots / want
        if slots != before:
            turn_trace.event("pressure_" + mempressure.name(worst), slots_from=before, slots_to=slots)
            print(
                f"[paged-moe] memory pressure {mempressure.name(worst)}: slab "
                f"{before} -> {slots} slots "
                f"({self._pressure.scale:.0%} of budget), compressor "
                f"{mempressure.compressor_gb():.1f} GB",
                flush=True,
            )
        return slots

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
            gpu_trail.mark(f"slab release ({self.slab.slots} slots)")
            self._slab_frozen = True
            pending = list(self._inflight.values()) + list(self._ahead_orphans.values())
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
        self._ahead.clear()
        self._ahead_orphans.clear()
        self._slot_of.clear()
        self._flow_reset()
        self._free_slots.clear()
        self._pinned.clear()
        self._unfreed.clear()
        self._stitch_keys = set()
        self._stitch_tables = {}
        self.stitched = False

    def leave_prefill(self) -> None:
        """Restore the decode-time budget (decode re-warms as it reads).

        Also rebuilds the slab if prefill gave it back. Runs on the MLX thread
        (the decode path calls it), which is where slabs must be allocated.
        """
        self.release_stage()
        rebuilt = False
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
                # In-flight command buffers still hold the old slab, and the
                # free-memory reading below would count it as taken.
                mx.synchronize()
                mx.clear_cache()
                slots = self._slots_that_fit(self._slab_full_slots or slots)
                if self._build_slab_locked(specs, slots):
                    self._log_memory("leave_prefill (slab rebuilt)")
                    rebuilt = True
            elif self._decode_budget is not None:
                self.budget_bytes = self._decode_budget
                self._decode_budget = None
        # A release cleared the stitched set. If the rebuilt slab is still
        # big enough, fill it again. Declines immediately when it is not.
        if rebuilt:
            self.load_stitch()

    def _build_slab_locked(self, specs, slots: int) -> bool:
        """Assumes lock held, old slab released, on the MLX thread. False (and
        the per-expert path) when there's no room, rather than failing the
        request."""
        gpu_trail.mark(f"slab build ({slots} slots)")
        try:
            self.slab = SlabStore(specs, slots)
        except Exception:
            self._slab_pending = None
            return False
        self._slab_pending = None
        self._free_slots = list(range(slots - 1, -1, -1))
        self._pinned.clear()
        self._unfreed.clear()
        reserve = max(4, int(slots * 0.1))
        self.budget_bytes = min(
            self._slab_budget, (slots - reserve) * self._slab_per_expert
        )
        self._decode_budget = None
        return True

    def _regrow_target(self, need: int) -> int:
        """Slots to regrow to, or 0. See config.SLAB_REGROW."""
        slab = self.slab
        per = self._slab_per_expert
        full = self._slab_full_slots
        if slab is None or not (config.SLAB_REGROW and config.MEM_PRESSURE):
            return 0
        if not (per and full and self._ram_budget_bytes) or slab.slots >= full:
            return 0
        if time.monotonic() < self._regrow_after:
            return 0
        if mempressure.calm_s() < config.SLAB_REGROW_CALM_S:
            return 0
        avail = mempressure.available_bytes()
        if avail is None:
            return 0
        held = slab.nbytes
        others = mx.get_active_memory() - held
        room = self._turn_room(need)
        budget_fit = (self._ram_budget_bytes - others - _SLAB_MARGIN - room) // per
        system_fit = (avail + held - _SLAB_MARGIN - room) // per
        target = int(min(full, budget_fit, system_fit))
        if (target - slab.slots) * per < max(_REGROW_MIN_BYTES, held // 10):
            return 0
        return target

    def regrow_slab(self, need: int = 0) -> bool:
        """Rebuild an undersized slab at the start of a turn. True if it grew.

        `need` is what the coming prefill allocates (the caller's
        make_room_for figure), kept out of the target so the bigger slab
        doesn't get handed straight back. Residency is dropped, not copied:
        growing an MLX buffer means holding old and new at once, and on a
        235B that transient alone is a third of the slab. The prefill that
        follows reads its experts into the new slab anyway.
        """
        target = self._regrow_target(need)
        if not target:
            return False
        before = self.slab.slots
        self._regrow_after = time.monotonic() + config.SLAB_REGROW_CALM_S
        self._release_slab()
        with self._lock:
            pending = self._slab_pending
            if pending is None or self.slab is not None:
                return False
            specs, _ = pending
            mx.synchronize()
            mx.clear_cache()
            grew = self._build_slab_locked(specs, target)
            if not grew:
                # Same size as before, so a failed grow costs residency only.
                self._slab_pending = pending
                if not self._build_slab_locked(specs, before):
                    return False
        if grew:
            full = self._slab_full_slots
            self._pressure.scale = max(self._pressure.scale, target / full)
            turn_trace.event("slab_regrow", slots_from=before, slots_to=target)
            print(
                f"[paged-moe] memory calm {mempressure.calm_s():.0f}s: slab "
                f"{before} -> {target} slots ({target / full:.0%} of budget), "
                f"{(mempressure.available_bytes() or 0) / 1e9:.1f} GB available",
                flush=True,
            )
            self._log_memory("regrow (slab rebuilt)")
        # The release cleared any stitched set. Declines at once if the
        # rebuilt slab cannot hold it.
        self.load_stitch()
        return grew

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
                if key in self._stitch_keys or self._slot_of.get(key) in self._pinned:
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
        if "__stage__" in entry:
            # Stage slots belong to the group being computed and go back to
            # the stage after it (fetch_groups); nothing resident lives there.
            return
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
        activity.touch()  # watchdog heartbeat: one per MoE layer step
        gpu_trail.layer(layer_key, prefill)
        b = _Begun()
        misses: list[int] = []
        slab = use_slab and self.slab is not None
        # A prefill that kept the slab reads its misses into free slots: pages
        # that already exist, no host copy, and compute takes the same slot
        # views it already uses for hits. Never evicts, so decode residency
        # survives. Out of free slots, the rest take the run-read path.
        prefill_slots = (
            prefill and not slab and self.slab is not None and config.PREFILL_SLAB
        )
        with self._lock:
            if slab:
                # The sync this layer just did on its own router forced every
                # earlier layer's expert compute, so the previous fetch's
                # experts are consumed and their slots may now be reused.
                self._clear_pins_locked()
                self._prefetch_clock += 1
            for eid in expert_ids:
                eid = int(eid)
                key = (layer_key, eid)
                entry = self._lru.get(key)
                if entry is not None:
                    self.hits += 1
                    if not prefill or config.PREFILL_TOUCH:
                        self._lru.move_to_end(key)
                        # Reuse is what earns eviction protection; a first
                        # install stays unproven (see _evict_to_budget).
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
                if fut is not None and _stage_result(fut) and (
                    not prefill or fut.result().get("__gen__") != self._stage_gen
                ):
                    # A stage read decode must not compute from (the stage is
                    # prefill-only and may be gone): read it again.
                    self._inflight.pop(key, None)
                    fut = None
                if fut is not None:
                    b.futs.append((eid, fut))
                    if slab:
                        self._pin(key)
                    latch = getattr(fut, "_es_latch", None)
                    if latch is not None and config.READ_PROMOTE and not fut.done():
                        fut._es_latch = None
                        if self._reads.promote(latch):
                            self.reads_promoted += 1
                    continue
                misses.append(eid)

            if slab:
                # Anything that cannot get a slot falls through to the run-read
                # path below as a loose mx array. That makes the layer's experts
                # mixed, which costs the gather_qmm batching for this one layer
                # (streaming falls back to per-expert) but is always correct.
                starved: list[int] = []
                for eid in misses:
                    fut = self._submit_slab(layer_key, eid, urgent=True)
                    if fut is None:
                        starved.append(eid)
                    else:
                        b.futs.append((eid, fut))
                misses = starved
            elif prefill_slots and misses:
                for i, eid in enumerate(misses):
                    fut = self._submit_slab(layer_key, eid, evict_ok=False, urgent=True)
                    if fut is None:
                        self.prefill_slot_starved += len(misses) - i
                        misses = misses[i:]
                        break
                    b.futs.append((eid, fut))
                    self.prefill_slot_reads += 1
                else:
                    misses = []
            if prefill and misses and self.stage is not None:
                for i, eid in enumerate(misses):
                    fut = self._submit_stage(layer_key, eid, urgent=True)
                    if fut is None:
                        self.stage_starved += len(misses) - i
                        misses = misses[i:]
                        break
                    b.futs.append((eid, fut))
                else:
                    misses = []

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

    def _finish(
        self,
        layer_key: str,
        b: _Begun,
        install: str,
    ) -> dict[int, dict]:
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
            ka = int(config.GPU_KEEPALIVE_US)
            if ka > 0 and install == "lru":
                arrived = _completed_warm(by_fut, ka / 1e6)
            else:
                arrived = as_completed(by_fut)
            for fut in arrived:
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
        ahead=None,
    ):
        """Yield (ids, experts) in bounded-size groups, pipelining disk reads.

        `ahead`, if given, is called once the last group's reads are queued
        (see prefill_ahead).

        Used for prefill, where one layer can want hundreds of experts (more
        than fits in memory at once).  While group i computes on the GPU, the
        reads for group i+1 are already in flight on the worker threads.

        `expert_ids` arrives sorted, so a group is a near-contiguous id range
        and its reads coalesce into a few long sequential spans - which is why
        the group wants to be large (see config.PREFILL_RUN_BYTES).
        """
        per = max(1, self.expert_nbytes(layer_key))
        if self.ensure_stage():
            group_bytes = min(group_bytes, self._stage_group_bytes())
        per_group = max(1, group_bytes // per)
        groups = [
            expert_ids[i : i + per_group]
            for i in range(0, len(expert_ids), per_group)
        ]
        self._settle_ahead(layer_key, expert_ids)
        pending = self._begin(layer_key, groups[0], prefill=True)
        if len(groups) == 1 and ahead is not None:
            ahead()
        for i, ids in enumerate(groups):
            current = pending
            if i + 1 < len(groups):
                pending = self._begin(layer_key, groups[i + 1], prefill=True)
                if i + 2 == len(groups) and ahead is not None:
                    ahead()
            got = self._finish(layer_key, current, install)
            yield ids, got
            # The caller has evaluated this group (streaming._run, eval_groups),
            # so no pending graph reads its stage slots any more.
            self._stage_return(got.values())

    def prefill_ahead(self, layer_key: str, expert_ids) -> int:
        """Start a later prefill layer's predicted reads into free slots.

        A prefill layer can't name its experts until its router runs, and its
        router needs the previous layer's output, so without this the drive
        idles at every layer boundary (prefill_gap_probe.py: 10-22% of a small
        prefill on the 235B). The caller predicts with the next layer's real router on
        this layer's hidden state and calls this once this layer's own reads
        are all queued, so the guesses sit behind demand and never ahead of
        it. Free slots first, never an eviction, and a floor of slots is left
        for the next layer's unpredicted misses. Past that, up to
        config.PREFILL_AHEAD_LOOSE_MB go as run reads: a warm slab has few
        free slots (qwen38-coder-flash: 5-10% of the next layer's reads fit).

        `expert_ids` in priority order; both caps keep the front of it.

        With no slab (a chunk past PREFILL_SHRINK_TOKENS, or memory pressure
        gave it back) every miss is a run read anyway and the slab's memory is
        free, so the guesses cover the next layer's first group instead: lowest
        ids first, the order that layer's groups are read in. That is the read
        the layer boundary would otherwise wait on; later groups already
        overlap the group before them.

        Prefetch only. The next layer still fetches exactly what its router
        picks; a read it wanted is claimed as in flight, and the rest go back
        in `_settle_ahead`. Returns reads started.
        """
        slab = self.slab
        loose_mb = int(config.PREFILL_AHEAD_LOOSE_MB)
        if slab is None and loose_mb <= 0:
            return 0
        if (
            self._ahead_n >= _AHEAD_MIN_SAMPLE
            and self._ahead_u < _AHEAD_MIN_PRECISION * self._ahead_n
        ):
            self.prefill_ahead_skipped += 1
            self._ahead_n *= _AHEAD_DECAY
            self._ahead_u *= _AHEAD_DECAY
            return 0
        floor = max(8, slab.slots // 50) if slab is not None else 0
        loose_cap = (loose_mb << 20) if slab is not None else max(loose_mb << 20, config.GROUP_BYTES)
        if loose_cap > 0:
            # Host RAM, so never more than half of what the kernel can spare
            # past the margin a slab rebuild keeps.
            if mempressure.level() >= mempressure.WARN:
                loose_cap = 0
            else:
                avail = mempressure.available_bytes()
                if avail is not None:
                    loose_cap = min(loose_cap, max(0, (avail - _SLAB_MARGIN) // 2))
        started = 0
        loose: list[int] = []
        # Stage slots past one demand group's worth: the next layer's first
        # group must still find room for what it routes and nobody guessed.
        stage_floor = (
            self._stage_group_bytes() // max(1, self.expert_nbytes(layer_key))
            if self.stage is not None else 0
        )
        with self._lock:
            if self._slab_frozen:
                return 0
            mine = self._ahead.setdefault(layer_key, set())
            for eid in expert_ids:
                eid = int(eid)
                key = (layer_key, eid)
                if key in self._lru or key in self._raw_ready or key in self._inflight:
                    continue
                if (
                    slab is not None
                    and len(self._free_slots) > floor
                    and self._submit_slab(layer_key, eid, evict_ok=False) is not None
                ):
                    mine.add(eid)
                    started += 1
                elif (
                    self.stage is not None
                    and len(self._stage_free) > stage_floor
                    and self._submit_stage(layer_key, eid) is not None
                ):
                    mine.add(eid)
                    started += 1
                elif loose_cap and self.stage is None:
                    loose.append(eid)
                else:
                    break
            if loose:
                per = max(1, self.expert_nbytes(layer_key))
                keep = max(1, loose_cap // per)
                loose = sorted(loose[:keep] if slab is not None else sorted(loose)[:keep])
                max_run = max(1, self._max_run_bytes(layer_key, True) // per)
                run = [loose[0]]
                for eid in loose[1:] + [None]:
                    if eid is not None and eid == run[-1] + 1 and len(run) < max_run:
                        run.append(eid)
                        continue
                    for e, _ in self._submit_run(layer_key, run):
                        mine.add(e)
                        started += 1
                    run = [eid]
            self.prefill_ahead_reads += started
        return started

    def _settle_ahead(self, layer_key: str, wanted) -> None:
        """This layer's real ids are known: hand back the guesses it didn't
        route, once their reads land (a slot is never reused mid-write)."""
        if not self._ahead:  # only this (prefill) thread writes it
            return
        drop: list[tuple[tuple, Future]] = []
        with self._lock:
            mine = self._ahead.pop(layer_key, None)
            if not mine:
                return
            want = {int(e) for e in wanted}
            used = len(mine & want)
            self.prefill_ahead_used += used
            self._ahead_n = self._ahead_n * _AHEAD_DECAY + len(mine)
            self._ahead_u = self._ahead_u * _AHEAD_DECAY + used
            slab = self.slab
            for eid in mine - want:
                key = (layer_key, eid)
                fut = self._inflight.pop(key, None)
                if fut is not None:
                    # Out of _inflight so nobody claims it, but still drained
                    # by _release_slab: the pread may be writing slab memory.
                    self._ahead_orphans[key] = fut
                    drop.append((key, fut))

        for key, fut in drop:
            def _done(f, key=key, slab=slab):
                try:
                    raw = f.result()
                except Exception:
                    raw = None
                with self._lock:
                    if self._ahead_orphans.get(key) is f:
                        del self._ahead_orphans[key]
                    if raw is None:
                        return
                    self.bytes_read += raw["__nbytes__"]
                    self.prefill_ahead_wasted_bytes += raw["__nbytes__"]
                    _release_raw(self._buffers, raw)
                    self._stage_return_locked(raw)
                    if self.slab is slab and slab is not None:
                        self._release_slot(key, raw)

            fut.add_done_callback(_done)

    # ------------------------------------------------------------- prefetch

    def _spec_blocked(self, staging: bool = True, need_at: int | None = None) -> bool:
        """Assumes lock held. True when speculation must stand down.

        `staging=False` skips the staging-occupancy check: route-predicted
        slab reads install straight into the LRU on arrival (see
        _attach_staging), so a staging area full of stale heuristic guesses
        must not gate them.

        A full staging area skips the new read. Dropping a finished guess to
        start another one re-reads those bytes and measured slower on MiniMax.
        Which finished guess dies on overflow is _trim_staging, not this.
        `need_at` is accepted so callers can pass it; admission does not use it.
        """
        del need_at
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

    def _prefetch_locked(
        self, layer_key: str, expert_ids, submitted: list, need_at: int | None = None
    ) -> None:
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
                if need_at is not None and key in self._raw_ready:
                    prev = self._raw_ready[key].get("__need_at__")
                    if prev is None or need_at < prev:
                        self._raw_ready[key]["__need_at__"] = need_at
                continue
            wanted.append(eid)
        if not wanted:
            return
        wanted.sort()

        def _note(eid: int) -> None:
            if need_at is None:
                return
            key = (layer_key, eid)
            prev = self._prefetch_need.get(key)
            if prev is None or need_at < prev:
                self._prefetch_need[key] = need_at

        if self.slab is not None:
            # Speculation gets free slots only (evict_ok=False): trading a
            # resident expert for a guess is how prefetching starts losing.
            for eid in wanted:
                fut = self._submit_slab(layer_key, eid, evict_ok=False)
                if fut is None:
                    break
                _note(eid)
                submitted.append(((layer_key, eid), fut))
            return
        per = max(1, self.expert_nbytes(layer_key))
        max_run = max(1, self._max_run_bytes(layer_key) // per)
        run: list[int] = [wanted[0]]
        for eid in wanted[1:]:
            if eid == run[-1] + 1 and len(run) < max_run:
                run.append(eid)
            else:
                for e, f in self._submit_run(layer_key, run):
                    _note(e)
                    submitted.append(((layer_key, e), f))
                run = [eid]
        for e, f in self._submit_run(layer_key, run):
            _note(e)
            submitted.append(((layer_key, e), f))

    def prefetch_many(self, batch) -> None:
        """prefetch() for several layers under one lock acquisition.

        Route prediction produces several layers' worth of ids at once, and this
        runs once per layer per token - so taking and dropping the lock per layer
        is overhead on exactly the path that has to stay nearly free.
        """
        # Opt-in (PAGED_MOE_PREDICT_INSTALL): measured on Qwen3-235B this
        # is only safe when prediction precision is high - wrong guesses that
        # install evict resident experts, and at precision ~0.5 that cache
        # pollution doubled per-token reads (2.2 GB/token) and cost 15%
        # against the staging path. Keep off unless precision is ~0.8+.
        install_now = self.slab is not None and config.PREDICT_INSTALL
        submitted: list[tuple[tuple, Future]] = []
        needs = [item[2] for item in batch if len(item) >= 3]
        need_at = min(needs) if needs else None
        with self._lock:
            if self._spec_blocked(staging=not install_now, need_at=need_at):
                return
            for item in batch:
                layer_key, ids = item[0], item[1]
                item_need = item[2] if len(item) >= 3 else None
                self._prefetch_locked(layer_key, ids, submitted, need_at=item_need)
        self._attach_staging(submitted, install_now=install_now)

    def prefetch(self, layer_key: str, expert_ids, need_at: int | None = None):
        """Start reads for experts we expect to want, without waiting.

        `need_at` is the decode-layer clock of the consumer. When staging is
        full, a sooner guess displaces a later one instead of being skipped.
        """
        submitted: list[tuple[tuple, Future]] = []
        with self._lock:
            if self._spec_blocked(need_at=need_at):
                return
            self._prefetch_locked(layer_key, expert_ids, submitted, need_at=need_at)
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
                        self._prefetch_need.pop(key, None)
                    return
                with self._lock:
                    self._inflight.pop(key, None)
                    need = self._prefetch_need.pop(key, None)
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
                    if need is not None:
                        raw["__need_at__"] = need
                    self._raw_ready[key] = raw
                    self._raw_bytes += raw["__nbytes__"]
                    self._raw_ready.move_to_end(key)
                    self._trim_staging()

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

    def note_stitch_keep(self, compute, last: bool) -> None:
        """Count slots the stitch mask left out, without syncing this layer."""
        n = int(compute.shape[0]) * int(compute.shape[1])
        dropped = mx.sum(1 - compute.astype(mx.int32))
        if self._stitch_drop is None:
            self._stitch_drop = dropped
        else:
            self._stitch_drop = self._stitch_drop + dropped
        self._stitch_demand += n
        if not last:
            return
        pending, demand = self._stitch_drop, self._stitch_demand
        self._stitch_drop = None
        self._stitch_demand = 0
        self._stitch_pending.append((pending, demand))
        mx.async_eval(pending)

    def _flush_stitch_counts(self) -> None:
        if self._stitch_drop is not None:
            pending, demand = self._stitch_drop, self._stitch_demand
            self._stitch_drop = None
            self._stitch_demand = 0
            self._stitch_pending.append((pending, demand))
        if not self._stitch_pending:
            return
        pending = self._stitch_pending
        self._stitch_pending = []
        for drop, demand in pending:
            mx.eval(drop)
            self.pruned_slots += int(drop.item())
            self.demand_slots += int(demand)

    def stats(self) -> dict:
        self._flush_stitch_counts()
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
            "slots_reclaimed": self.slots_reclaimed,
            "prefill_slot_reads": self.prefill_slot_reads,
            "prefill_slot_starved": self.prefill_slot_starved,
            "stage_reads": self.stage_reads,
            "stage_starved": self.stage_starved,
            "stage_builds": self.stage_builds,
            "prefill_ahead_reads": self.prefill_ahead_reads,
            "prefill_ahead_used": self.prefill_ahead_used,
            "prefill_ahead_wasted_gb": round(self.prefill_ahead_wasted_bytes / 1e9, 3),
            "prefill_ahead_skipped": self.prefill_ahead_skipped,
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
            "reads_promoted": self.reads_promoted,
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
            # pruning (0 unless PAGED_MOE_PRUNE is set).
            "pruned_frac": (
                round(self.pruned_slots / self.demand_slots, 4)
                if self.demand_slots
                else 0.0
            ),
            # Experts a weight test dropped but residency rescued: quality
            # recovered for free (see config.KEEP_FREE).
            "kept_free": self.kept_free,
            "closed_ok": self.closed_ok,
            "closed_replay": self.closed_replay,
            # Experts skipped instead of blocking the GPU on a read, and
            # prefetched for later tokens instead (see config.WAIT_ABOVE).
            "skipped_waits": self.skipped_waits,
            "crit_per_layer": round(self.crit_misses / self.crit_steps, 4)
            if self.crit_steps
            else 0.0,
            "mass_sync": round(self.mass_sync / self.rows_sync, 4)
            if self.rows_sync
            else None,
            "mass_flow": round(self.mass_flow / self.rows_flow, 4)
            if self.rows_flow
            else None,
            # Share of layer calls that read the router's weights off the
            # block's own gate call rather than re-running it.
            "route_tap_rate": (
                round(self.route_tapped / self.route_calls, 4)
                if self.route_calls
                else 0.0
            ),
            "mx_active_gb": round(mx.get_active_memory() / 1e9, 3),
            "mx_cache_gb": round(mx.get_cache_memory() / 1e9, 3),
            "mx_peak_gb": round(note_peak() / 1e9, 3),
            "prefill_layers_skipped": self.prefill_layers_skipped,
        }
