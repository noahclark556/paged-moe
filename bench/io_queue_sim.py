#!/usr/bin/env python3
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""How much can a cancelable / priority IO queue buy?

The ExpertCache today submits every read into a ThreadPoolExecutor FIFO.
Speculation is issued early; demand misses are submitted when a layer blocks.
A demand Future therefore sits *behind* every not-yet-started speculative job
already in the executor queue. Canceling those pending specs (or letting
demand jump them) is the only preemption available: os.pread cannot be
interrupted once a worker has entered the kernel, and a dedicated speculative
lane was already measured 6-9% slower (docs/improvements.md).

This sim asks the quantitative question without rewriting the cache:

  Given the engine's real issue pattern (near-band prediction + last_ids
  heuristic at prefetch_depth, capped inflight), how much of demand's
  disk_wait is "queued behind cancelable speculative work" versus "the drive
  is just busy with useful bytes"?

Policies compared on the same trace:
  fifo       - today's ThreadPoolExecutor semantics
  priority   - demand jobs always pop before pending spec
  cancel     - on a demand miss, drop all pending (not running) speculative
               jobs that are NOT the demanded key; demand runs next

If cancel ~ fifo, the remaining disk wait is capacity/bandwidth, not queue
policy, and a preemptive queue is not the lever. If cancel wins big, it is.

Usage:
  python bench/io_queue_sim.py --bank ~/.paged_moe/sidecar/<slot>_wrap_bank.npz
  python bench/io_queue_sim.py --synthetic --layers 94 --experts 128 --k 8
"""

from __future__ import annotations

import argparse
import heapq
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field


@dataclass(order=True)
class Job:
    prio: int
    ready_at: float
    seq: int
    key: tuple = field(compare=False)
    kind: str = field(compare=False)
    bytes: int = field(compare=False)
    cancelable: bool = field(compare=False, default=True)
    remaining: float = field(compare=False, default=0.0)


class Drive:
    """N workers pulling from a priority / FIFO ready queue."""

    def __init__(self, workers: int, bw_bytes_s: float, policy: str):
        self.workers = workers
        self.bw = bw_bytes_s
        self.policy = policy
        self.t = 0.0
        self.seq = 0
        self.ready: list[Job] = []
        self.running: list[Job] = []
        self.running_fin: list[tuple[float, Job]] = []
        self.pending_by_key: dict[tuple, Job] = {}
        self.inflight_keys: set[tuple] = set()
        self.wait_demand = 0.0
        self.bytes_done = 0
        self.bytes_canceled = 0
        self.n_demand = 0
        self.n_canceled = 0
        self.demand_queue_wait = 0.0

    def _push(self, job: Job) -> None:
        if self.policy == "fifo":
            heapq.heappush(
                self.ready,
                Job(0, job.ready_at, job.seq, job.key, job.kind, job.bytes,
                    job.cancelable, job.remaining),
            )
        else:
            heapq.heappush(self.ready, job)
        self.pending_by_key[job.key] = job
        self.inflight_keys.add(job.key)

    def submit(self, key: tuple, kind: str, nbytes: int, at: float | None = None) -> None:
        if key in self.inflight_keys:
            return  # dedup: demand claims the in-flight spec
        at = self.t if at is None else at
        self.seq += 1
        prio = 0 if kind == "demand" else 1
        job = Job(prio, at, self.seq, key, kind, nbytes,
                  cancelable=(kind == "spec"), remaining=float(nbytes))
        self._push(job)

    def cancel_pending_spec(
        self,
        keep: set[tuple] | None = None,
        *,
        current_layer: int | None = None,
        far_after: int = 2,
        n_layers: int = 94,
    ) -> None:
        """Drop not-yet-started speculative jobs.

        `cancel`     - drop every pending spec (the blunt instrument; mostly
                       equivalent to turning speculation off under miss load).
        `cancel_far` - drop only specs whose layer is more than `far_after`
                       steps ahead of the demanding layer. Near predicted
                       work keeps its place; far heuristic / far-lead work
                       yields the pipe. This is the preemption worth testing.
        """
        if self.policy not in ("cancel", "cancel_far"):
            return
        keep = keep or set()
        kept: list[Job] = []
        while self.ready:
            j = heapq.heappop(self.ready)
            drop = False
            if j.cancelable and j.key not in keep and j.kind == "spec":
                if self.policy == "cancel":
                    drop = True
                elif current_layer is not None:
                    tgt = j.key[0]
                    # Forward distance in the layer walk (no wrap): wrap
                    # last_ids for the *next* token are the farthest work
                    # and should yield first.
                    if tgt < current_layer:
                        drop = True  # next-token wrap = far
                    elif tgt - current_layer > far_after:
                        drop = True
            if drop:
                self.n_canceled += 1
                self.bytes_canceled += j.bytes
                self.inflight_keys.discard(j.key)
                self.pending_by_key.pop(j.key, None)
            else:
                kept.append(j)
        self.ready = []
        for j in kept:
            heapq.heappush(self.ready, j)

    def _start_ready(self) -> None:
        while len(self.running) < self.workers and self.ready:
            if self.ready[0].ready_at > self.t + 1e-12:
                break
            j = heapq.heappop(self.ready)
            self.pending_by_key.pop(j.key, None)
            if j.kind == "demand":
                self.demand_queue_wait += self.t - j.ready_at
            if j.remaining <= 0:
                j.remaining = float(j.bytes)
            self.running.append(j)
        self._stamp_finishes()

    def _stamp_finishes(self) -> None:
        """Assign finish times from remaining bytes under fair-share BW."""
        n = max(1, len(self.running))
        share = self.bw / n
        self.running_fin = [
            (self.t + j.remaining / share, j) for j in self.running
        ]

    def _advance_to(self, t_next: float) -> None:
        """Consume fair-share bandwidth for every running job up to t_next."""
        dt = t_next - self.t
        if dt <= 0 or not self.running:
            self.t = max(self.t, t_next)
            return
        n = len(self.running)
        share = self.bw / n
        for j in self.running:
            j.remaining = max(0.0, j.remaining - share * dt)
        self.t = t_next

    def _pop_finished(self):
        self._stamp_finishes()
        self.running_fin.sort(key=lambda x: x[0])
        fin, j = self.running_fin[0]
        self._advance_to(fin)
        self.running = [x for x in self.running if x is not j]
        self.bytes_done += j.bytes
        self.inflight_keys.discard(j.key)
        return j

    def pump(self) -> None:
        self._start_ready()

    def drain_until(self, t_target: float, on_complete=None) -> None:
        while self.t < t_target:
            self._start_ready()
            if not self.running:
                self.t = t_target
                break
            self._stamp_finishes()
            self.running_fin.sort(key=lambda x: x[0])
            fin = self.running_fin[0][0]
            if fin > t_target:
                self._advance_to(t_target)
                break
            j = self._pop_finished()
            if on_complete is not None:
                on_complete(j.key)
            self._start_ready()


    def wait_until_key(
        self,
        key: tuple,
        nbytes: int,
        *,
        current_layer: int | None = None,
        on_complete=None,
    ) -> float:
        """Block until `key` is done; kept for unit tests, replay uses batching."""
        self.n_demand += 1
        t0 = self.t
        if key not in self.inflight_keys:
            if self.policy in ("cancel", "cancel_far"):
                self.cancel_pending_spec(keep={key}, current_layer=current_layer)
            self.submit(key, "demand", nbytes)
        elif self.policy in ("priority", "cancel", "cancel_far"):
            if self.policy in ("cancel", "cancel_far"):
                self.cancel_pending_spec(keep={key}, current_layer=current_layer)
            j = self.pending_by_key.get(key)
            if j is not None and j.kind == "spec":
                self.ready = [x for x in self.ready if x.key != key]
                heapq.heapify(self.ready)
                self.inflight_keys.discard(key)
                self.pending_by_key.pop(key, None)
                self.submit(key, "demand", nbytes)
        while key in self.inflight_keys:
            self._start_ready()
            if not self.running:
                break
            j = self._pop_finished()
            if on_complete is not None:
                on_complete(j.key)
            self._start_ready()
        self.wait_demand += self.t - t0
        return self.t - t0



def synthetic_trace(layers: int, n_experts: int, k: int, tokens: int, seed: int = 0):
    import random
    rng = random.Random(seed)
    # Mild locality: each layer has a hot set of 2k experts that rotate slowly.
    hot = {
        L: [rng.randrange(n_experts) for _ in range(2 * k)]
        for L in range(layers)
    }
    for t in range(tokens):
        if t % 8 == 0:
            for L in hot:
                hot[L] = [rng.randrange(n_experts) for _ in range(2 * k)]
        frame = {}
        for L in range(layers):
            ids = list(dict.fromkeys(
                hot[L][i % len(hot[L])] for i in range(k - 1)
            ))
            while len(ids) < k:
                e = rng.randrange(n_experts)
                if e not in ids:
                    ids.append(e)
            frame[L] = ids
        yield frame


def load_bank(path: str, max_tokens: int):
    import numpy as np
    z = np.load(path, allow_pickle=True)
    # wrap bank: trajectories of per-layer id lists
    trajs = z["trajectories"] if "trajectories" in z.files else z[z.files[0]]
    n = 0
    for traj in trajs:
        for frame in traj:
            if isinstance(frame, dict):
                yield {int(L): [int(e) for e in ids] for L, ids in frame.items()}
            else:
                # ndarray [layers, k]
                yield {L: [int(e) for e in row if int(e) >= 0]
                       for L, row in enumerate(frame)}
            n += 1
            if n >= max_tokens:
                return


def replay(trace, *, layers: int, policy: str, workers: int, bw: float,
           expert_bytes: int, cache_slots: int, predict_depth: int,
           prefetch_depth: int, spec_cap: int, predict_recall: float):
    """Approximate the engine's issue pattern on one policy."""
    import random
    rng = random.Random(1)
    drive = Drive(workers, bw, policy)
    cache: OrderedDict = OrderedDict()
    staged: set = set()  # speculative completions not yet demanded
    last_ids = {L: [] for L in range(layers)}

    def touch(key):
        staged.discard(key)
        cache.pop(key, None)
        cache[key] = True
        while len(cache) > cache_slots:
            cache.popitem(last=False)

    def stage(key):
        # Speculative landing: available for demand, but does NOT occupy an
        # LRU slot. Matches the engine's staging path (PREDICT_INSTALL off).
        if key not in cache:
            staged.add(key)

    compute_s = 0.0003

    for frame in trace:
        for L in range(layers):
            ids = frame.get(L) or frame.get(str(L)) or []
            if not ids:
                continue
            if predict_depth > 0 and L % predict_depth == 0:
                pending = len(drive.inflight_keys)
                for d in range(2, predict_depth + 2):
                    if pending >= spec_cap:
                        break
                    tgt = L + d
                    if tgt >= layers:
                        break
                    true = frame.get(tgt) or []
                    guess = [e for e in true if rng.random() < predict_recall]
                    n_fp = max(0, int(len(guess) * 0.43))
                    for _ in range(n_fp):
                        guess.append(rng.randrange(128 if not true else max(true) + 8))
                    for e in guess:
                        key = (tgt, int(e))
                        if key in cache or key in staged or key in drive.inflight_keys:
                            continue
                        if len(drive.inflight_keys) >= spec_cap:
                            break
                        drive.submit(key, "spec", expert_bytes)
                        pending += 1
            for d in range(1, prefetch_depth + 1):
                if len(drive.inflight_keys) >= spec_cap:
                    break
                tgt = L + d
                if tgt >= layers:
                    break
                for e in last_ids.get(tgt, []):
                    key = (tgt, int(e))
                    if key in cache or key in staged or key in drive.inflight_keys:
                        continue
                    if len(drive.inflight_keys) >= spec_cap:
                        break
                    drive.submit(key, "spec", expert_bytes)
            drive.pump()
            drive.drain_until(drive.t + compute_s, on_complete=stage)

            misses = []
            for e in ids:
                key = (L, int(e))
                if key in cache or key in staged:
                    touch(key)
                else:
                    misses.append(key)
            if not misses:
                last_ids[L] = [int(e) for e in ids]
                continue
            # Batch like ExpertCache._begin: submit every miss, then wait.
            # Prediction wins by shrinking this set, not by reordering a
            # single-expert critical path.
            t0 = drive.t
            if drive.policy in ("cancel", "cancel_far"):
                drive.cancel_pending_spec(
                    keep=set(misses), current_layer=L
                )
            for key in misses:
                if key not in drive.inflight_keys:
                    drive.submit(key, "demand", expert_bytes)
                elif drive.policy == "priority":
                    j = drive.pending_by_key.get(key)
                    if j is not None and j.kind == "spec":
                        drive.ready = [x for x in drive.ready if x.key != key]
                        heapq.heapify(drive.ready)
                        drive.inflight_keys.discard(key)
                        drive.pending_by_key.pop(key, None)
                        drive.submit(key, "demand", expert_bytes)
            pending = set(misses)
            while pending:
                drive._start_ready()
                if not drive.running:
                    break
                j = drive._pop_finished()
                stage(j.key)
                if j.key in pending:
                    touch(j.key)
                    pending.discard(j.key)
                drive._start_ready()
            drive.wait_demand += drive.t - t0
            drive.n_demand += len(misses)
            last_ids[L] = [int(e) for e in ids]

    return drive


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--layers", type=int, default=94)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--bw-gbs", type=float, default=10.0)
    ap.add_argument("--expert-mb", type=float, default=10.6)
    ap.add_argument("--cache-slots", type=int, default=2270)
    ap.add_argument("--predict-depth", type=int, default=3)
    ap.add_argument("--prefetch-depth", type=int, default=8)
    ap.add_argument("--spec-cap", type=int, default=48)
    ap.add_argument("--predict-recall", type=float, default=0.87)
    args = ap.parse_args()

    if args.bank:
        frames = list(load_bank(args.bank, args.tokens))
        # infer layers
        layers = max(max(f) for f in frames) + 1 if frames else args.layers
    else:
        frames = list(synthetic_trace(args.layers, args.experts, args.k,
                                      args.tokens))
        layers = args.layers

    bw = args.bw_gbs * 1e9
    expert_bytes = int(args.expert_mb * 1e6)
    print(f"{len(frames)} tokens, {layers} layers, policy comparison\n")
    print(f"{'policy':<10} {'demand_wait_s':>14} {'q_wait_s':>10} "
          f"{'canceled_GB':>12} {'vs fifo':>10}")

    base = None
    for policy in ("fifo", "priority", "cancel_far", "cancel"):
        d = replay(
            frames, layers=layers, policy=policy, workers=args.workers,
            bw=bw, expert_bytes=expert_bytes, cache_slots=args.cache_slots,
            predict_depth=args.predict_depth, prefetch_depth=args.prefetch_depth,
            spec_cap=args.spec_cap, predict_recall=args.predict_recall,
        )
        if base is None:
            base = d.wait_demand
        delta = (d.wait_demand - base) / base if base else 0.0
        print(f"{policy:<10} {d.wait_demand:14.3f} {d.demand_queue_wait:10.3f} "
              f"{d.bytes_canceled/1e9:12.3f} {delta:+10.1%}")
    print("\nIf cancel/priority are within a few % of fifo, the remaining wait "
          "is drive capacity - not queue policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
