#!/usr/bin/env python3
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Acceptance harness: does the wrap head cost time, in bytes, on real demand?

Why this is separate from sidecar_sim.py
----------------------------------------
`sidecar_sim.py` ranks *policies* - it answers "how good could an eviction or
prefetch rule be". This answers a narrower and more important question: does
the code we are about to ship make decode faster or slower than not running it,
using the weights that are actually on this machine and the demand that
actually came out of the router?

It does that by driving the real `WrapHead` - not a reimplementation of it -
through a cache simulator. Every gate under test (probability threshold, byte
budget, read slack, governor veto, byte-ledger demotion) executes here exactly
as it would in production. The only thing faked is the disk.

Why bytes and not recall
------------------------
The live tick log scores this head in recall (`gain`), which is >= 0 by
construction: the head only ever issues reads the engine's baseline missed, so
the metric cannot express harm. Decode on a streamed MoE is disk-bandwidth
bound, so a speculative read competes 1:1 with a demand miss and a wasted one
is a demand miss delayed. Time is the only honest objective.

Configurations compared
-----------------------
  off       no sidecar: the engine's own ring replay of the previous token
  shipped   the old policy, reconstructed: issue every prediction, every token,
            with no threshold, no byte budget and no slack gate
  gated     the head as it now stands, governor forced on (measures the head)
  governed  the head as it now stands with the governor live (measures the
            guarantee - this is what a user gets)

`shipped` uses the *same weights* as `gated`, so the difference reported is
purely the issuance policy and not a better model.

Usage
-----
  python bench/sidecar_verify.py --slot ~/.paged_moe/sidecar/<id>
  python bench/sidecar_verify.py --all          # every slot with a bank
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pagedmoe import config  # noqa: E402
from pagedmoe.sidecar.bank import load_bank  # noqa: E402


class SimCache:
    """LRU expert residency over a single-queue drive with explicit time.

    The timing model is the part that has to be right, and it took three tries
    to get there, so the two wrong versions are worth recording:

    1. `time = max(compute, all_bytes / bandwidth)`. Under this, a *correct*
       prefetch is byte-neutral (it moves a byte from the demand column to the
       speculative column) and a wrong one is pure cost, so no speculation can
       ever win and the harness rejects everything. Wrong because it assumes
       the drive is perfectly packed - it has no notion of a layer sitting idle
       waiting for a read that could have been started earlier.

    2. `time = compute + demand_bytes/bw + max(0, spec_bytes/bw - compute)`.
       This treats the whole compute window as idle drive capacity, so waste is
       free until it exceeds ~90 MB per token. Under it the old unbounded
       policy scores *better* than a gated one, which contradicts the -24%
       measured on the machine. Wrong because on a model streaming ~1.2 GB per
       token the drive is busy for 250 ms of a 270 ms token: there is no idle
       window to hide in.

    What actually happens is a pipeline. The engine walks layers in order; at
    each MoE layer it needs experts, and it blocks only if they are not there
    yet. Reads go to one queue served at the drive's bandwidth. So a prefetch
    is worth something when it gets its read *started early enough to finish
    before the layer arrives*, and it costs something whenever it occupies the
    queue ahead of a read a layer is about to block on. Both effects are
    timing, not volume, which is why they cannot be seen without a clock.

    So this keeps one: `drive` is when the queue next goes free, `now` is where
    the layer walk has got to. Everything else follows from those two.
    """

    def __init__(
        self,
        slots: int,
        expert_bytes: int,
        read_cap: int,
        *,
        bandwidth_gb_s: float,
        layer_compute_s: float,
    ):
        self.slots = int(slots)
        self._expert_bytes = int(expert_bytes)
        self.read_cap = int(read_cap)
        self._read_s = expert_bytes / (bandwidth_gb_s * 1e9)
        self._layer_s = float(layer_compute_s)
        self.resident: OrderedDict = OrderedDict()
        self.demand_miss_bytes = 0
        self.ring_bytes = 0
        self.spec_bytes = 0
        self.spec_used = 0
        self.spec_issued = 0
        self.hits = 0
        self.misses = 0
        # Reads queued at a token boundary, landing in the next token's window.
        self._queued: list[tuple[tuple[str, int], bool]] = []
        # Keys whose read is in flight, and when it completes. A prefetch is
        # only worth something if this is <= the moment its layer is reached.
        self._ready: dict[tuple[str, int], float] = {}
        # Keys that arrived speculatively and have not yet been demanded, so a
        # later demand hit can be credited. That credit is the head's value.
        self._speculative: dict[tuple[str, int], bool] = {}
        self.now = 0.0      # where the layer walk has reached
        self.drive = 0.0    # when the read queue next goes free
        self.stall_s = 0.0  # time layers spent blocked on reads
        self.spec_s = 0.0   # drive time spent on speculation
        self.waste_s = 0.0  # ...of which nothing ever wanted

    # ---- the API the head consults ------------------------------------

    def read_slack(self) -> int:
        """Reads that may still be queued without piling up behind work a
        layer is about to block on."""
        return max(0, self.read_cap - len(self._queued))

    def expert_nbytes(self, layer_key: str) -> int:
        return self._expert_bytes

    def prefetch_many(self, batch) -> None:
        self._queue(batch, speculative=True)

    def ring_prefetch(self, batch) -> None:
        """The engine's own replay. Happens in every configuration, including
        `off`, so it is charged to all of them equally and never to the head."""
        self._queue(batch, speculative=False)

    def _queue(self, batch, *, speculative: bool) -> None:
        for layer_key, ids in batch:
            for e in ids:
                key = (layer_key, int(e))
                if key in self.resident or key in self._ready:
                    continue  # already warm or already coming: no read
                self._queued.append((key, speculative))
                if speculative:
                    self.spec_issued += 1

    # ---- the token walk -----------------------------------------------

    def begin_token(self) -> None:
        """Start the queued reads, in order, on the drive.

        They run ahead of this token's demand, which is the point - and also
        the risk, because a read queued here that nothing wants still holds the
        drive for as long as a useful one would.
        """
        for key, speculative in self._queued:
            if key in self.resident or key in self._ready:
                continue
            start = max(self.drive, self.now)
            self.drive = start + self._read_s
            self._ready[key] = self.drive
            if speculative:
                self.spec_bytes += self._expert_bytes
                self.spec_s += self._read_s
                self._speculative[key] = False
            else:
                self.ring_bytes += self._expert_bytes
        self._queued.clear()

    def run_layer(self, layer_key: str, ids) -> None:
        """Compute one MoE layer, blocking on whatever is not there yet."""
        done = self.now + self._layer_s
        for e in ids:
            key = (layer_key, int(e))
            if key in self.resident:
                self.resident.move_to_end(key)
                self.hits += 1
                continue
            arrival = self._ready.pop(key, None)
            if arrival is not None:
                # Prefetched. Free if it landed in time, a partial stall if the
                # walk caught up with it - which is what happens when the queue
                # is deep with speculation.
                self.hits += 1
                self._admit(key)
                if self._speculative.pop(key, None) is False:
                    self.spec_used += 1
                done = max(done, arrival)
                continue
            # Nobody saw this coming: read it now, with the layer waiting.
            self.misses += 1
            self.demand_miss_bytes += self._expert_bytes
            start = max(self.drive, self.now)
            self.drive = start + self._read_s
            done = max(done, self.drive)
            self._admit(key)
        self.stall_s += max(0.0, done - (self.now + self._layer_s))
        self.now = done

    def end_token(self) -> float:
        """Finish the token; returns its wall time."""
        # Speculation still in flight at the boundary was never wanted by this
        # token. It may be wanted by the next one, so it is not written off
        # here - but the drive time it took is already spent.
        self.now = max(self.now, 0.0)
        return self.now

    def write_off_stale(self) -> None:
        """At a prompt boundary, in-flight speculation is dead weight."""
        for key, was_used in list(self._speculative.items()):
            if was_used is False:
                self.waste_s += self._read_s
        self._speculative.clear()
        self._ready.clear()
        self._queued.clear()

    def _admit(self, key) -> None:
        self.resident[key] = True
        while len(self.resident) > self.slots:
            old, _ = self.resident.popitem(last=False)
            self._speculative.pop(old, None)


def run(
    trajectories,
    n_layers: int,
    n_experts: int,
    slot_dir: Path,
    slot_id: str,
    mode: str,
    *,
    expert_mb: float,
    cache_gb: float,
    bandwidth_gb_s: float,
    compute_ms: float,
    read_cap: int,
) -> dict:
    """Replay every trajectory in one configuration and price the result."""
    from pagedmoe.sidecar.heads.wrap import WrapHead

    expert_bytes = int(expert_mb * 1e6)
    slots = max(1, int(cache_gb * 1e9 // expert_bytes))
    layer_keys = [f"layers.{i}" for i in range(n_layers)]

    saved = (
        config.SIDECAR_HEAD_WRAP,
        config.SIDECAR_GOVERNOR,
        config.SIDECAR_MIN_PROB,
        config.SIDECAR_SPEC_BYTE_FRAC,
    )
    config.SIDECAR_HEAD_WRAP = mode != "off"
    config.SIDECAR_GOVERNOR = mode == "governed"
    if mode == "shipped":
        # Reconstruct the old behaviour from the same weights: no confidence
        # bar, an effectively unbounded byte budget, and no slack gate.
        config.SIDECAR_MIN_PROB = 0.0
        config.SIDECAR_SPEC_BYTE_FRAC = 1e9
    try:
        from pagedmoe.sidecar.governor import NetGovernor

        head = None
        gov = NetGovernor(name=mode) if mode == "governed" else None
        if mode != "off":
            head = WrapHead(
                slot_id, n_layers, n_experts, layer_keys, slot_dir, governor=gov
            )
            if mode == "shipped":
                head._min_prob = 0.0
                head._byte_frac = 1e9
            # These runs measure the issuance policy, so give every
            # configuration the same standing permission to act. Whether that
            # permission should have been granted is what `governed` tests.
            head.slot.prefetch_enabled = True

        # `shipped` had no slack gate at all, so give it unlimited capacity -
        # that absence is exactly the behaviour under test.
        cache = SimCache(
            slots,
            expert_bytes,
            10**9 if mode == "shipped" else read_cap,
            bandwidth_gb_s=bandwidth_gb_s,
            layer_compute_s=compute_ms / 1000.0 / max(1, n_layers),
        )
        wrap = head.wrap if head is not None else _auto_wrap(n_layers)
        tokens = 0
        seconds = 0.0
        for traj in trajectories:
            cache.write_off_stale()
            for frame in traj:
                if not frame:
                    continue
                t0 = cache.now
                cache.begin_token()
                for i in range(n_layers):
                    ids = frame.get(i)
                    if ids:
                        cache.run_layer(layer_keys[i], ids)
                dt = cache.end_token() - t0
                seconds += dt
                tokens += 1
                # Order matters: the engine's ring replays this token's experts
                # into the next token's early layers, and the head speculates
                # on top of whatever queue depth the ring left. Letting the
                # head go first would credit it with capacity the ring had
                # already committed.
                cache.ring_prefetch(
                    [
                        (layer_keys[i], ids)
                        for i, ids in frame.items()
                        if i < wrap
                    ]
                )
                if head is not None:
                    if gov is not None:
                        gov.note_token(dt)
                    head.observe_end_token(frame, [], cache)
    finally:
        (
            config.SIDECAR_HEAD_WRAP,
            config.SIDECAR_GOVERNOR,
            config.SIDECAR_MIN_PROB,
            config.SIDECAR_SPEC_BYTE_FRAC,
        ) = saved

    total_bytes = cache.demand_miss_bytes + cache.ring_bytes + cache.spec_bytes
    return {
        "mode": mode,
        "tokens": tokens,
        "demand_misses": cache.misses,
        "hit_rate": round(cache.hits / max(1, cache.hits + cache.misses), 4),
        "demand_gb": round(cache.demand_miss_bytes / 1e9, 3),
        "ring_gb": round(cache.ring_bytes / 1e9, 3),
        "spec_gb": round(cache.spec_bytes / 1e9, 3),
        "spec_issued": cache.spec_issued,
        "spec_precision": round(cache.spec_used / max(1, cache.spec_issued), 4),
        "total_gb": round(total_bytes / 1e9, 3),
        # Time layers spent blocked on a read. This is what prefetch is for,
        # and the only place a win can come from.
        "stall_s": round(cache.stall_s, 1),
        # Drive time spent on speculation. Every second of it is a second the
        # demand path could have had.
        "spec_s": round(cache.spec_s, 1),
        "tok_s": round(tokens / seconds, 3) if seconds > 0 else 0.0,
        "head": head.stats() if head is not None else None,
        "governor": gov.stats() if gov is not None else None,
    }


def _auto_wrap(n_layers: int) -> int:
    from pagedmoe.sidecar.features import auto_wrap

    return auto_wrap(n_layers)


# Expert size is the single biggest lever on whether speculation can pay, and
# it is not in the bank - the bank records routing, not weights. These are the
# two checkpoints the banks came from, measured; override with --expert-mb.
_EXPERT_MB = {
    (94, 128): 1.6,   # Qwen3-235B-A22B, 4-bit
    (89, 160): 10.6,  # GLM-4.7, 4-bit
}


def verify_slot(path: Path, args) -> tuple[bool, dict]:
    bank = path.parent / f"{path.name}_wrap_bank.npz"
    if not bank.exists():
        bank = Path(str(path) + "_wrap_bank.npz")
    trajectories, meta = load_bank(bank)
    if not trajectories:
        raise SystemExit(f"no trajectories in {bank}")
    trajectories = [t for t in trajectories if len(t) >= args.min_traj]
    if not trajectories:
        raise SystemExit(f"no trajectory of length >= {args.min_traj}")
    n_layers = int(meta.get("n_layers") or max(
        max(f.keys(), default=0) for t in trajectories for f in t
    ) + 1)
    n_experts = int(meta.get("n_experts") or 128)
    expert_mb = args.expert_mb or _EXPERT_MB.get((n_layers, n_experts))
    if not expert_mb:
        raise SystemExit(
            f"unknown expert size for {n_layers}L/{n_experts}E - pass --expert-mb"
        )

    out = {}
    for mode in ("off", "shipped", "gated", "governed"):
        out[mode] = run(
            trajectories,
            n_layers,
            n_experts,
            path.parent,
            path.name,
            mode,
            expert_mb=expert_mb,
            cache_gb=args.cache_gb,
            bandwidth_gb_s=args.bandwidth_gb_s,
            compute_ms=args.compute_ms,
            read_cap=args.read_cap,
        )

    base = out["off"]["tok_s"]
    print(f"\n=== {path.name}  ({n_layers} layers, {n_experts} experts, "
          f"{len(trajectories)} trajectories, {out['off']['tokens']} tokens)")
    print(f"    cache {args.cache_gb} GB = "
          f"{int(args.cache_gb * 1e9 // (expert_mb * 1e6))} experts, "
          f"{expert_mb} MB each, {args.bandwidth_gb_s} GB/s, "
          f"read cap {args.read_cap}")
    print(f"    {'mode':10s} {'tok/s':>8s} {'vs off':>8s} {'misses':>9s} "
          f"{'stall s':>8s} {'spec s':>7s} {'issued':>8s} {'spec p':>7s}")
    for mode in ("off", "shipped", "gated", "governed"):
        r = out[mode]
        delta = (r["tok_s"] / base - 1.0) * 100 if base else 0.0
        print(f"    {mode:10s} {r['tok_s']:8.3f} {delta:+7.1f}% "
              f"{r['demand_misses']:9d} {r['stall_s']:8.1f} "
              f"{r['spec_s']:7.1f} {r['spec_issued']:8d} "
              f"{r['spec_precision']:7.3f}")

    # The acceptance criterion, and the whole point of the exercise: whatever
    # the head's model quality turns out to be, shipping it must not make the
    # machine slower than not shipping it.
    ok = True
    for mode in ("gated", "governed"):
        if out[mode]["tok_s"] < base * (1.0 - args.tolerance):
            print(f"    FAIL: {mode} is {(1 - out[mode]['tok_s'] / base) * 100:.1f}% "
                  f"slower than no sidecar")
            ok = False
    if ok:
        print("    PASS: no configuration regresses decode rate")
    g = out["governed"].get("governor") or {}
    if g:
        print(f"    governor: decision={g.get('decision')} "
              f"on={g.get('tok_s_on')} off={g.get('tok_s_off')} tok/s, "
              f"flips={g.get('flips')}, "
              f"actuated {g.get('tokens_on')}/{g.get('tokens_on', 0) + g.get('tokens_off', 0)} sampled tokens")
    return ok, out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--slot", help="path prefix, e.g. ~/.paged_moe/sidecar/<id>")
    ap.add_argument("--all", action="store_true", help="every slot with a wrap bank")
    ap.add_argument(
        "--root",
        default="~/.global_agent/expert_sidecar",
        help="where sidecar slots live",
    )
    ap.add_argument("--expert-mb", type=float, default=0.0,
                    help="MB per expert; 0 = look up by model geometry")
    ap.add_argument("--cache-gb", type=float, default=8.0)
    ap.add_argument("--bandwidth-gb-s", type=float, default=5.0)
    ap.add_argument("--compute-ms", type=float, default=18.0)
    ap.add_argument("--read-cap", type=int, default=48,
                    help="reads queueable per token window; ExpertCache uses "
                         "read_threads*3 = 48 by default")
    ap.add_argument("--min-traj", type=int, default=8)
    ap.add_argument("--tolerance", type=float, default=0.005,
                    help="allowed shortfall vs no sidecar (0.005 = 0.5%%)")
    ap.add_argument("--json", help="write full results here")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    if args.all:
        slots = sorted(
            {p.name[: -len("_wrap_bank.npz")] for p in root.glob("*_wrap_bank.npz")}
        )
        targets = [root / s for s in slots]
    elif args.slot:
        targets = [Path(args.slot).expanduser()]
    else:
        raise SystemExit("pass --slot <prefix> or --all")
    if not targets:
        raise SystemExit(f"no wrap banks under {root}")

    results = {}
    failures = []
    for t in targets:
        ok, out = verify_slot(t, args)
        results[t.name] = out
        if not ok:
            failures.append(t.name)

    if args.json:
        Path(args.json).expanduser().write_text(json.dumps(results, indent=2))
    print()
    if failures:
        raise SystemExit(f"REGRESSION in: {', '.join(failures)}")
    print(f"OK: {len(targets)} slot(s), no decode-rate regression")


if __name__ == "__main__":
    main()
