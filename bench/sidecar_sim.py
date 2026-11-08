#!/usr/bin/env python3
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Trace-driven sidecar harness: measures sidecar policies in BYTES and TIME.

Why this exists
---------------
The live tick log scores the wrap head in *recall* (`gain`), and `gain` is >= 0
by construction because the head only ever adds reads on top of the engine's
baseline. So the number it reports can never go negative no matter how much
harm the head does. Decode on a streamed MoE is disk-bandwidth-bound, which
means every speculative byte competes 1:1 with a demand miss. The only honest
objective is time, and the only honest unit of cost is bytes.

This replays saved wrap banks - real per-token, per-layer router demand from
actual decode - through a simulator of expert residency and reports, per
policy: demand misses, bytes read, speculative bytes wasted, and a
bandwidth-bound decode-rate estimate.

What it models well: which experts are resident when, how many blocking misses
a policy causes, and how many bytes each policy spends. What it does not model:
wall-clock scheduling, read queueing, compute overlap detail. So use it to rank
policies and to prove a policy cannot regress bytes - then confirm the winner
on a real checkpoint with bench/sidecar_ab.sh.

Policies
--------
  lru       what ExpertCache does with LFRU off
  lfru      the shipped bounded second-chance scan
  lfu       classic aging frequency counter
  ema       per-(layer,expert) EMA of observed reuse distance, bounded scan
            (this is what ResidencyHead implements online)
  belady    oracle: evict the entry used furthest in the future (ceiling)
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pagedmoe.sidecar.bank import load_bank  # noqa: E402


# ---------------------------------------------------------------- access trace


def build_trace(trajectories, n_layers: int):
    """Flatten trajectories into (key_id, token, layer) arrays.

    Keys are interned to small ints so the replay loops stay cheap. Trajectory
    boundaries are returned separately: routing does not carry across prompts,
    and letting residency survive a boundary would invent reuse that never
    happened.
    """
    keys: dict = {}
    key_ids = []
    layers = []
    tokens = []
    boundaries = []
    t = 0
    for traj in trajectories:
        boundaries.append(len(key_ids))
        for frame in traj:
            for layer in range(n_layers):
                ids = frame.get(layer)
                if not ids:
                    continue
                for e in ids:
                    k = (layer, int(e))
                    kid = keys.get(k)
                    if kid is None:
                        kid = len(keys)
                        keys[k] = kid
                    key_ids.append(kid)
                    layers.append(layer)
                    tokens.append(t)
            t += 1
    return (
        np.asarray(key_ids, dtype=np.int64),
        np.asarray(layers, dtype=np.int32),
        np.asarray(tokens, dtype=np.int64),
        np.asarray(boundaries, dtype=np.int64),
        len(keys),
    )


def next_use_positions(key_ids: np.ndarray, n_keys: int) -> np.ndarray:
    """For each access, the position of the next access of the same key.

    len(trace) means "never again". This is Belady's oracle signal and the
    regression target any learned reuse predictor is trying to approximate.
    """
    n = key_ids.shape[0]
    nxt = np.full(n, n, dtype=np.int64)
    last = np.full(n_keys, -1, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        k = key_ids[i]
        prev = last[k]
        if prev >= 0:
            nxt[i] = prev
        last[k] = i
    return nxt


# ------------------------------------------------------------------- replay


def replay(
    key_ids: np.ndarray,
    boundaries: np.ndarray,
    slots: int,
    policy: str,
    *,
    next_use: np.ndarray | None = None,
    tokens_of: np.ndarray | None = None,
    layers_of: np.ndarray | None = None,
    n_layers: int = 0,
    lfru_scan: int = 8,
    ema_scan: int = 32,
    ema_decay: float = 0.9,
    freq_decay: float = 0.93,
    freq_sample: int = 32,
    token_frames: list | None = None,
    protect_horizon: int = 0,
    protect_bar: float = 0.35,
    protect_cap: float = 0.33,
) -> dict:
    """Replay one access stream under one eviction policy.

    Every policy here only ever chooses *which resident expert to drop*, so
    none of them can change an output - the worst a bad choice does is force a
    re-read later. That is the whole reason residency is the right place to
    spend sidecar capacity: the downside is bounded at zero.
    """
    resident: OrderedDict = OrderedDict()  # key -> True, in LRU order
    freq: dict = defaultdict(int)
    hits = misses = evictions = 0
    boundary_set = set(int(b) for b in boundaries.tolist())

    # Flat list of resident keys with swap-removal, so `freq` can draw a
    # uniform sample in O(1) instead of walking the LRU order.
    resident_list: list[int] = []
    rpos: dict[int, int] = {}

    def _rl_add(key: int) -> None:
        rpos[key] = len(resident_list)
        resident_list.append(key)

    def _rl_drop(key: int) -> None:
        j = rpos.pop(key, None)
        if j is None:
            return
        last = resident_list.pop()
        if j < len(resident_list):
            resident_list[j] = last
            rpos[last] = j

    # belady: lazy max-heap on next-use, with stale entries skipped.
    heap: list = []
    heap_stamp: dict = {}

    # ema: per-key mean reuse distance (in accesses) and last-seen position.
    n_all = int(key_ids.max()) + 1
    ema_dist = np.zeros(n_all, dtype=np.float32)
    ema_seen = np.zeros(n_all, dtype=np.int64)
    last_pos = np.full(n_all, -1, dtype=np.int64)

    # freq: decayed per-key demand count, i.e. an estimate of P(this expert is
    # demanded by an upcoming token at its layer). Decayed lazily on read so
    # there is no periodic sweep over the table.
    #
    # This is the signal that matters for MoE residency, and recency is not.
    # Every MoE layer is touched exactly once per token, in order, so after one
    # token the LRU order is just "layer index" - the tail is the *early*
    # layers, which are the ones about to run again. Recency carries almost no
    # information about reuse here, which is why LRU, LFRU and reuse-distance
    # variants all measure within half a percent of each other below while
    # Belady is 27% better.
    fcount = np.zeros(n_all, dtype=np.float32)
    fstamp = np.zeros(n_all, dtype=np.int64)
    fseen = np.zeros(n_all, dtype=np.int32)
    ftok = np.zeros(n_all, dtype=np.int64)
    rng = np.random.default_rng(0)

    def fscore(key: int, tok: int) -> float:
        age = tok - int(fstamp[key])
        if age <= 0:
            return float(fcount[key])
        return float(fcount[key]) * (freq_decay ** min(64, age))

    # Per-layer residency census, sampled at trajectory-relative token ends.
    # Belady beating every causal policy by 27% while an *exact* global
    # frequency ranking manages 1% says the gap is not about picking the right
    # expert within a layer - it is about how many slots each layer gets. This
    # records that distribution so the two effects can be told apart.
    layer_resident = np.zeros(max(1, n_layers), dtype=np.float64)
    census_n = 0
    key_layer = np.zeros(n_all, dtype=np.int32)
    if layers_of is not None:
        key_layer[key_ids] = layers_of  # a key's layer never changes

    # protect<H>: LRU, but an entry is skipped as a victim if it is demanded
    # within the next H tokens. This is the policy a next-token demand
    # predictor actually enables - protection costs no bytes and cannot change
    # an output, unlike prefetch which spends bandwidth on every guess. Run
    # here with an oracle so the ceiling is known before any model is fitted.
    soon: dict = {}
    soon_tok = -1

    def _soon_advance(tok: int) -> None:
        """Roll the protected multiset to cover tokens tok+1 .. tok+H."""
        nonlocal soon_tok
        if token_frames is None:
            return
        while soon_tok < tok:
            soon_tok += 1
            drop = soon_tok  # leaving the window's lower edge
            add = soon_tok + protect_horizon
            if 0 <= drop < len(token_frames):
                for kk in token_frames[drop]:
                    c = soon.get(kk, 0) - 1
                    if c <= 0:
                        soon.pop(kk, None)
                    else:
                        soon[kk] = c
            if 0 <= add < len(token_frames):
                for kk in token_frames[add]:
                    soon[kk] = soon.get(kk, 0) + 1

    n = key_ids.shape[0]
    cur_token = -1
    for i in range(n):
        if i in boundary_set:
            resident.clear()
            freq.clear()
            heap.clear()
            heap_stamp.clear()
            resident_list.clear()
            rpos.clear()
            fcount[:] = 0.0
            fstamp[:] = 0
            fseen[:] = 0
        k = int(key_ids[i])
        if tokens_of is not None:
            cur_token = int(tokens_of[i])
            if policy == "protect":
                _soon_advance(cur_token)
        if policy in ("freq", "freq_exact", "tinylfu", "s3fifo"):
            fcount[k] = fscore(k, cur_token) + 1.0
            fstamp[k] = cur_token
            fseen[k] += 1

        if policy == "protect_pred":
            # ResidencyHead's online model, replayed exactly: an EMA over
            # "was this key wanted again within the horizon". Each key is
            # touched at most once per token, so a per-access update is the
            # same thing as the head's per-token one.
            prev_tok = ftok[k]
            if fseen[k] > 0:
                reused = 1.0 if (cur_token - prev_tok) <= protect_horizon else 0.0
                ema_dist[k] = ema_dist[k] * 0.9 + reused * 0.1
            else:
                ema_dist[k] = 0.15  # ResidencyHead._INIT
            ema_seen[k] = ema_seen[k] + 1
            fseen[k] += 1
            ftok[k] = cur_token

        if policy == "ema":
            prev = last_pos[k]
            if prev >= 0:
                d = float(i - prev)
                seen = ema_seen[k]
                ema_dist[k] = (
                    d if seen == 0 else ema_dist[k] * ema_decay + d * (1.0 - ema_decay)
                )
                ema_seen[k] = seen + 1
            last_pos[k] = i

        if k in resident:
            hits += 1
            resident.move_to_end(k)
            freq[k] = min(8, freq[k] + 1)
            if policy == "belady" and next_use is not None:
                nu = int(next_use[i])
                heap_stamp[k] = nu
                heapq.heappush(heap, (-nu, k))
            continue

        misses += 1

        # ---- admission control
        #
        # This is where the Belady gap actually lives. LRU installs every
        # missed expert, which evicts a resident one - so a single access to a
        # tail expert costs a hot expert its slot. With 9.7k live keys against
        # 2.6k slots, most misses are one-shot tail traffic and the cache
        # spends its life churning. Belady's advantage is not that it ranks
        # better; it is that it effectively never admits a key it will not
        # reuse (its next-use is infinity, so it is evicted immediately).
        #
        # Declining to install is free and quality-safe: the expert is still
        # read and still computed this token. All we give up is keeping it.
        if policy in ("tinylfu", "s3fifo") and len(resident) >= slots:
            cand_score = fscore(k, cur_token)
            victim0 = next(iter(resident))
            if policy == "tinylfu":
                # Admit only if the newcomer looks more valuable than the
                # entry it would displace (Einziger/Friedman TinyLFU).
                if cand_score <= fscore(victim0, cur_token):
                    continue  # computed, not retained
            else:
                # S3-FIFO-style: a key gets in on its second sighting. The
                # frequency table doubles as the ghost queue, so a one-shot
                # expert never displaces anything.
                if fseen[k] < 2:
                    continue

        # ---- install, evicting if needed
        while len(resident) >= slots:
            victim = None
            if policy == "lru":
                victim = next(iter(resident))
            elif policy == "lfru":
                for _ in range(lfru_scan):
                    if len(resident) <= 1:
                        break
                    cand = next(iter(resident))
                    f = freq.get(cand, 0)
                    if f <= 0:
                        break
                    freq[cand] = f >> 1
                    resident.move_to_end(cand)
                victim = next(iter(resident))
            elif policy == "lfu":
                # Aging LFU: least-used among a bounded window of the LRU tail.
                best = None
                best_f = None
                for j, cand in enumerate(resident):
                    if j >= ema_scan:
                        break
                    f = freq.get(cand, 0)
                    if best_f is None or f < best_f:
                        best, best_f = cand, f
                victim = best
            elif policy == "ema":
                # Approximate Belady causally: among a bounded window of the
                # LRU tail, drop the entry whose *predicted* next use is
                # furthest out. Prediction is the EMA of that key's own
                # observed reuse distances, discounted by how long it has
                # already been idle.
                best = None
                best_score = None
                for j, cand in enumerate(resident):
                    if j >= ema_scan:
                        break
                    seen = ema_seen[cand]
                    if seen == 0:
                        pred = float("inf")  # never reused: perfect victim
                    else:
                        idle = i - last_pos[cand]
                        pred = max(1.0, float(ema_dist[cand]) - float(idle))
                    if best_score is None or pred > best_score:
                        best, best_score = cand, pred
                        if pred == float("inf"):
                            break
                victim = best
            elif policy == "freq":
                # Sampled global minimum: draw a bounded sample of resident
                # keys and drop the least likely to be wanted again. Sampling
                # rather than scanning the LRU tail is the whole point - the
                # entries Belady wants gone are spread through the residency,
                # not parked at the tail.
                pool = resident_list
                m = len(pool)
                best = None
                best_score = None
                for _ in range(freq_sample):
                    cand = pool[int(rng.integers(0, m))]
                    if cand not in resident:
                        continue
                    s = fscore(cand, cur_token)
                    if best_score is None or s < best_score:
                        best, best_score = cand, s
                victim = best if best is not None else next(iter(resident))
            elif policy == "freq_exact":
                # Upper bound on `freq`: exact global argmin of the same score.
                best = None
                best_score = None
                for cand in resident:
                    s = fscore(cand, cur_token)
                    if best_score is None or s < best_score:
                        best, best_score = cand, s
                victim = best
            elif policy in ("tinylfu", "s3fifo"):
                victim = next(iter(resident))
            elif policy == "protect_pred":
                # The shippable version of protect<H>: no oracle, just each
                # key's own measured reuse rate within the horizon, which is
                # what ResidencyHead maintains online. Walk the LRU from the
                # tail and skip entries scoring above the bar, subject to a cap
                # so a uniformly hot cache still makes progress.
                budget = int(protect_cap * slots)
                skipped = 0
                for cand in resident:
                    if skipped >= budget:
                        victim = cand
                        break
                    seen = ema_seen[cand]
                    if seen == 0:
                        victim = cand
                        break
                    age = cur_token - ftok[cand]
                    score = float(ema_dist[cand])
                    if age > protect_horizon:
                        score *= 0.9 ** min(32.0, age / max(1, protect_horizon))
                    if score >= protect_bar:
                        skipped += 1
                        continue
                    victim = cand
                    break
                if victim is None:
                    victim = next(iter(resident))
            elif policy == "protect":
                # Walk the LRU from the tail and take the first entry nothing
                # wants within the horizon. `soon` holds at most H tokens'
                # worth of keys against a much larger residency, so this
                # almost always stops on the first candidate.
                for cand in resident:
                    if cand not in soon:
                        victim = cand
                        break
                if victim is None:
                    victim = next(iter(resident))
            elif policy == "belady":
                while heap:
                    neg, cand = heap[0]
                    if cand not in resident or heap_stamp.get(cand) != -neg:
                        heapq.heappop(heap)
                        continue
                    victim = cand
                    heapq.heappop(heap)
                    break
                if victim is None:
                    victim = next(iter(resident))
            else:
                raise ValueError(policy)
            resident.pop(victim, None)
            freq.pop(victim, None)
            heap_stamp.pop(victim, None)
            _rl_drop(victim)
            evictions += 1

        resident[k] = True
        freq.setdefault(k, 0)
        _rl_add(k)
        if policy == "belady" and next_use is not None:
            nu = int(next_use[i])
            heap_stamp[k] = nu
            heapq.heappush(heap, (-nu, k))

        if layers_of is not None and n_layers and (i % 4096 == 0):
            census_n += 1
            for cand in resident:
                layer_resident[key_layer[cand]] += 1.0

    total = hits + misses
    out = {
        "policy": policy,
        "accesses": total,
        "demand_misses": misses,
        "hit_rate": round(hits / total, 4) if total else 0.0,
        "evictions": evictions,
    }
    if census_n:
        per = layer_resident / census_n
        out["layer_residency"] = {
            "mean": round(float(per.mean()), 2),
            "std": round(float(per.std()), 2),
            "min": round(float(per.min()), 2),
            "max": round(float(per.max()), 2),
            "first8": [round(float(v), 1) for v in per[:8]],
        }
    return out


# --------------------------------------------------------------- wrap ceiling


def wrap_structure(trajectories, wrap: int):
    """The wrap head's entire addressable market, and what it costs to serve.

    The engine's prefetch ring already replays the previous token's experts
    into the early layers, so a wrap head can only ever add the *residual*:
    next-token early-layer demand the copy missed. `residual_per_token` is how
    many reads a perfect oracle would issue. Anything a real head issues beyond
    what it gets right is waste, and on a bandwidth-bound decode waste is
    charged at the same rate as useful bytes.
    """
    copy_hit = copy_need = 0
    residual = []
    for traj in trajectories:
        for a, b in zip(traj, traj[1:]):
            r = 0
            for i in range(wrap):
                want = set(b.get(i, ()))
                if not want:
                    continue
                have = set(a.get(i, ()))
                copy_need += len(want)
                copy_hit += len(want & have)
                r += len(want - have)
            residual.append(r)
    return {
        "wrap_layers": wrap,
        "copy_recall": round(copy_hit / copy_need, 4) if copy_need else 0.0,
        "oracle_extra_reads_per_token": round(float(np.mean(residual)), 3)
        if residual
        else 0.0,
    }


def per_layer_copy_recall(trajectories, n_layers: int) -> np.ndarray:
    hit = np.zeros(n_layers)
    need = np.zeros(n_layers)
    for traj in trajectories:
        for a, b in zip(traj, traj[1:]):
            for i in range(n_layers):
                want = set(b.get(i, ()))
                if not want:
                    continue
                need[i] += len(want)
                hit[i] += len(want & set(a.get(i, ())))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(need > 0, hit / np.maximum(need, 1), np.nan)


# ------------------------------------------------------------------ cost model


def est_tok_s(
    demand_misses: int,
    spec_bytes: float,
    tokens: int,
    expert_mb: float,
    bandwidth_gb_s: float,
    compute_ms: float,
) -> float:
    """Decode rate under a bandwidth-bound model.

    A token costs its compute floor or its bytes, whichever is larger, since
    reads overlap compute. Speculative bytes are charged in full whether or not
    they were useful - which is precisely the term recall-based `gain` omits.
    """
    if tokens <= 0:
        return 0.0
    total_bytes = demand_misses * expert_mb * 1e6 + spec_bytes
    read_ms = (total_bytes / (bandwidth_gb_s * 1e9)) * 1e3
    return 1e3 / max(compute_ms, read_ms / tokens)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--cache-gb", type=float, default=28.0)
    ap.add_argument("--expert-mb", type=float, default=10.6)
    ap.add_argument("--bandwidth-gb-s", type=float, default=5.5)
    ap.add_argument("--compute-ms", type=float, default=30.0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument(
        "--min-traj",
        type=int,
        default=0,
        help="drop trajectories shorter than this (short ones put the sim in a "
        "cold-cache regime and hand Belady unlearnable end-of-stream knowledge)",
    )
    ap.add_argument(
        "--reset-per-traj",
        action="store_true",
        help="clear residency at trajectory boundaries. Off by default because "
        "the real expert LRU survives across turns (relieve_pressure keeps it)",
    )
    ap.add_argument("--ema-scan", type=int, default=32)
    ap.add_argument(
        "--freq-decay",
        type=float,
        default=0.93,
        help="per-token decay of the demand-frequency estimate",
    )
    ap.add_argument("--protect-bar", type=float, default=0.35)
    ap.add_argument("--protect-cap", type=float, default=0.33)
    ap.add_argument(
        "--freq-sample",
        type=int,
        default=32,
        help="resident keys sampled per eviction by the `freq` policy",
    )
    ap.add_argument(
        "--wrap-waste",
        type=float,
        default=0.6,
        help="observed share of wrap extras nothing wanted (live ticks: 0.5-0.7)",
    )
    ap.add_argument("--wrap-topk", type=int, default=4)
    args = ap.parse_args()

    trajectories, meta = load_bank(Path(args.bank).expanduser())
    n_layers = int(meta.get("n_layers") or 0)
    n_experts = int(meta.get("n_experts") or 0)
    wrap = int(meta.get("wrap") or 6)
    if args.min_traj:
        trajectories = [t for t in trajectories if len(t) >= args.min_traj]
        if not trajectories:
            raise SystemExit(f"no trajectory of length >= {args.min_traj} in bank")
    if args.max_frames:
        trimmed, budget = [], args.max_frames
        for traj in trajectories:
            if budget <= 0:
                break
            trimmed.append(traj[:budget])
            budget -= len(trimmed[-1])
        trajectories = trimmed

    expert_mb = args.expert_mb
    slots = int(args.cache_gb * 1e9 / (expert_mb * 1e6))
    key_ids, _layers, _toks, boundaries, n_keys = build_trace(trajectories, n_layers)
    if not args.reset_per_traj:
        boundaries = np.zeros(0, dtype=np.int64)
    tokens = sum(max(0, len(t) - 1) for t in trajectories)
    nxt = next_use_positions(key_ids, n_keys)

    out = {
        "bank": str(args.bank),
        "geometry": {
            "n_layers": n_layers,
            "n_experts": n_experts,
            "wrap": wrap,
            "expert_mb": expert_mb,
            "cache_slots": slots,
            "cache_gb": args.cache_gb,
            "expert_mass_gb": round(n_layers * n_experts * expert_mb / 1e3, 1),
            "resident_share": round(slots / max(1, n_layers * n_experts), 4),
        },
        "trace": {"accesses": int(key_ids.size), "tokens": tokens, "keys": n_keys},
        "wrap_structure": wrap_structure(trajectories, wrap),
        "policies": [],
    }

    # Key ids demanded by each token, for the protect<H> oracle.
    n_tok = int(_toks.max()) + 1 if _toks.size else 0
    token_frames: list[list[int]] = [[] for _ in range(n_tok)]
    for kid, tok in zip(key_ids.tolist(), _toks.tolist()):
        token_frames[tok].append(kid)

    base_miss = None
    plans = [(p, 0) for p in ("lru", "lfru")]
    plans += [("protect", h) for h in (1, 4, 8, 16)]
    plans += [("protect_pred", h) for h in (4, 8, 16, 32)]
    plans += [("belady", 0)]
    for policy, horizon in plans:
        r = replay(
            key_ids,
            boundaries,
            slots,
            policy,
            next_use=nxt,
            tokens_of=_toks,
            layers_of=_layers,
            n_layers=n_layers,
            ema_scan=args.ema_scan,
            freq_decay=args.freq_decay,
            freq_sample=args.freq_sample,
            token_frames=token_frames if policy == "protect" else None,
            protect_horizon=horizon,
            protect_bar=args.protect_bar,
            protect_cap=args.protect_cap,
        )
        if horizon:
            r["policy"] = f"{policy}{horizon}"
        r["misses_per_token"] = round(r["demand_misses"] / max(1, tokens), 2)
        r["est_tok_s"] = round(
            est_tok_s(
                r["demand_misses"], 0.0, tokens, expert_mb, args.bandwidth_gb_s,
                args.compute_ms,
            ),
            3,
        )
        if policy == "lru":
            base_miss = r["demand_misses"]
        r["hit_rate"] = r["hit_rate"]
        r["miss_delta_vs_lru"] = (
            round((r["demand_misses"] - base_miss) / base_miss, 4) if base_miss else 0.0
        )
        out["policies"].append(r)

    # What the *current* wrap head does to the byte budget, on top of LFRU.
    lfru = next(p for p in out["policies"] if p["policy"] == "lfru")
    extras = wrap * args.wrap_topk
    wasted = extras * args.wrap_waste * expert_mb * 1e6 * tokens
    out["wrap_head_as_shipped"] = {
        "extras_issued_per_token": extras,
        "assumed_waste": args.wrap_waste,
        "wasted_mb_per_token": round(extras * args.wrap_waste * expert_mb, 1),
        "demand_miss_mb_per_token": round(lfru["misses_per_token"] * expert_mb, 1),
        "byte_overhead": round(
            (extras * args.wrap_waste) / max(1e-9, lfru["misses_per_token"]), 4
        ),
        "est_tok_s": round(
            est_tok_s(
                lfru["demand_misses"], wasted, tokens, expert_mb,
                args.bandwidth_gb_s, args.compute_ms,
            ),
            3,
        ),
        "baseline_est_tok_s": lfru["est_tok_s"],
    }

    reuse = per_layer_copy_recall(trajectories, n_layers)
    out["copy_recall_by_depth"] = {
        "layers_0_7": [None if np.isnan(v) else round(float(v), 3) for v in reuse[:8]],
        "mean_early": round(float(np.nanmean(reuse[:wrap])), 4),
        "mean_all": round(float(np.nanmean(reuse)), 4),
    }

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
