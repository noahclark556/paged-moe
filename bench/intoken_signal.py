#!/usr/bin/env python3
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Is there anything to learn about *this* token's later layers?

The engine already has two in-token prefetch sources, and they cover different
distances from the issuing layer:

    layers i+1 .. i+depth        previous token's ids at that layer (last_ids)
    layers i+lead+1 .. +predict  that layer's REAL router, run on layer i's
                                 hidden state (exact modulo drift)

Route prediction is the one that measures +79% (simulated read delay), and it is
strong because it has real lead time: the reads for layer i+4 start while layer
i is still computing, and on a saturated drive that head start is the whole
game. But it cannot be extended indefinitely - it feeds layer i's hidden state
to a router that expects layer i+d's input, and those diverge with every
attention block and FFN in between.

So the interesting question is not "can a learned head beat the router" (it
cannot, the router is the ground truth) but "can a learned head beat *last_ids*
at distances the router can no longer reach". That is a much lower bar, and if
it clears it the payoff is lead time nothing else can buy.

This script answers three things from the demand banks alone, before any
engine work:

  1. How good is the last_ids baseline as a function of layer depth? If the
     previous token's ids at layer L already cover this token's demand at
     layer L, there is nothing here and we stop.
  2. Is demand at layer L conditionally predictable from *this* token's
     earlier-layer demand - i.e. does the prefix of a token tell you where it
     is going? This is the signal a learned head would live on, and it is
     information the router-on-stale-hidden-state does not use at all.
  3. What is the oracle ceiling - if you knew this token's layer-L ids at
     layer i, how many misses would that remove?

Usage:
    python bench/intoken_signal.py --all
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pagedmoe.sidecar.bank import load_bank  # noqa: E402


def _frames(trajectories, min_layers: int):
    """Flatten to (traj_index, token_index, frame) for usable frames."""
    for ti, traj in enumerate(trajectories):
        for t, frame in enumerate(traj):
            if len(frame) >= min_layers:
                yield ti, t, frame


def baseline_recall(trajectories, n_layers: int) -> dict:
    """Recall of frame[t-1][L] against frame[t][L], bucketed by depth."""
    hit = np.zeros(n_layers, dtype=np.int64)
    tot = np.zeros(n_layers, dtype=np.int64)
    for traj in trajectories:
        for t in range(1, len(traj)):
            prev, cur = traj[t - 1], traj[t]
            for layer, ids in cur.items():
                if layer >= n_layers or not ids:
                    continue
                p = prev.get(layer)
                if not p:
                    continue
                s = set(p)
                hit[layer] += sum(1 for e in ids if e in s)
                tot[layer] += len(ids)
    return {"hit": hit, "tot": tot}


def prefix_conditional(trajectories, n_layers: int, *, split: float = 0.7) -> dict:
    """Does this token's early-layer demand predict its later-layer demand?

    The test is deliberately the dumbest thing that could work: for each
    (early layer a, late layer b) pair, count how often expert j at b follows
    expert i at a, and ask whether the resulting conditional ranking beats the
    unconditional popularity ranking at b. If a co-occurrence table cannot
    beat marginal popularity, a linear head on the same information will not
    either, and we have saved ourselves the implementation.

    Trained on the first `split` of trajectories, scored on the rest, so this
    is a real generalization number and not a memorization one.
    """
    cut = max(1, int(len(trajectories) * split))
    train, test = trajectories[:cut], trajectories[cut:]
    if not test:
        test = train[-1:]

    # Probe a few representative (source, target) distances rather than all
    # pairs: the table is O(n_layers^2 * n_experts) and the shape of the curve
    # is what matters.
    probes = [(4, 4 + d) for d in (2, 4, 8, 16, 32) if 4 + d < n_layers]
    probes += [(n_layers // 2, n_layers // 2 + d) for d in (2, 4, 8, 16)
               if n_layers // 2 + d < n_layers]

    results = {}
    for src, tgt in probes:
        # Unconditional popularity at tgt (the thing to beat).
        marginal: Counter = Counter()
        joint: dict[int, Counter] = defaultdict(Counter)
        for traj in train:
            for frame in traj:
                a, b = frame.get(src), frame.get(tgt)
                if not a or not b:
                    continue
                marginal.update(b)
                for i in a:
                    joint[i].update(b)
        if not marginal:
            continue

        # Score on held-out tokens: rank experts at tgt two ways, take the top
        # K (K = that token's real fan-out), and compare recall.
        m_rank = [e for e, _ in marginal.most_common()]
        cond_hit = marg_hit = tot = 0
        for traj in test:
            for frame in traj:
                a, b = frame.get(src), frame.get(tgt)
                if not a or not b:
                    continue
                k = len(b)
                truth = set(b)
                scores: Counter = Counter()
                for i in a:
                    scores.update(joint.get(i, {}))
                cond = [e for e, _ in scores.most_common(k)]
                cond_hit += sum(1 for e in cond if e in truth)
                marg_hit += sum(1 for e in m_rank[:k] if e in truth)
                tot += k
        if tot:
            results[(src, tgt)] = {
                "cond": cond_hit / tot,
                "marg": marg_hit / tot,
                "tot": tot,
            }
    return results


def prev_token_same_layer_vs_prefix(trajectories, n_layers: int) -> dict:
    """Head-to-head at equal budget: last_ids vs a prefix-conditioned guess.

    Both get exactly K guesses for layer tgt (K = real fan-out). last_ids uses
    frame[t-1][tgt]; the prefix model uses co-occurrence from frame[t][src].
    A union of the two is also scored, because in the engine they are not
    exclusive - prefetch drops resident ids, so agreement is free and the
    practical question is what the union covers.
    """
    cut = max(1, int(len(trajectories) * 0.7))
    train, test = trajectories[:cut], trajectories[cut:] or trajectories[-1:]
    src = 4
    out = {}
    for d in (4, 8, 16, 32):
        tgt = src + d
        if tgt >= n_layers:
            continue
        joint: dict[int, Counter] = defaultdict(Counter)
        for traj in train:
            for frame in traj:
                a, b = frame.get(src), frame.get(tgt)
                if a and b:
                    for i in a:
                        joint[i].update(b)
        prev_hit = pre_hit = union_hit = tot = 0
        union_cost = 0
        for traj in test:
            for t in range(1, len(traj)):
                frame, prev = traj[t], traj[t - 1]
                a, b = frame.get(src), frame.get(tgt)
                p = prev.get(tgt)
                if not a or not b or not p:
                    continue
                k = len(b)
                truth = set(b)
                scores: Counter = Counter()
                for i in a:
                    scores.update(joint.get(i, {}))
                pre = [e for e, _ in scores.most_common(k)]
                prev_hit += sum(1 for e in p if e in truth)
                pre_hit += sum(1 for e in pre if e in truth)
                union = set(p) | set(pre)
                union_hit += len(union & truth)
                union_cost += len(union)
                tot += k
        if tot:
            out[d] = {
                "prev": prev_hit / tot,
                "prefix": pre_hit / tot,
                "union": union_hit / tot,
                "union_reads": union_cost / tot,
                "tot": tot,
            }
    return out


def report(slot: str, trajectories, meta: dict) -> None:
    n_layers = int(meta.get("n_layers") or 0)
    if not n_layers:
        n_layers = 1 + max(
            (max(f) for traj in trajectories for f in traj if f), default=0
        )
    tokens = sum(len(t) for t in trajectories)
    print(f"\n=== {slot}  ({n_layers} layers, {len(trajectories)} traj, "
          f"{tokens} tokens)")

    base = baseline_recall(trajectories, n_layers)
    hit, tot = base["hit"], base["tot"]
    seen = tot > 0
    overall = hit[seen].sum() / max(1, tot[seen].sum())
    print(f"  last_ids recall (prev token, same layer): {overall:.3f} overall")
    # Bucket by depth to see whether early layers are the hard ones.
    edges = np.linspace(0, n_layers, 5).astype(int)
    for lo, hi in itertools.pairwise(edges):
        band = slice(lo, hi)
        t = tot[band].sum()
        if t:
            print(f"    layers {lo:>3}-{hi - 1:<3} {hit[band].sum() / t:.3f}")

    print("  prefix-conditional vs unconditional popularity at layer tgt:")
    for (src, tgt), r in sorted(prefix_conditional(trajectories, n_layers).items()):
        lift = r["cond"] - r["marg"]
        print(f"    {src:>3} -> {tgt:<3} (d={tgt - src:>2})  "
              f"cond {r['cond']:.3f}  popularity {r['marg']:.3f}  "
              f"lift {lift:+.3f}")

    print("  equal-budget head to head for one target layer:")
    for d, r in sorted(prev_token_same_layer_vs_prefix(trajectories, n_layers).items()):
        print(f"    d={d:>2}  last_ids {r['prev']:.3f}  prefix {r['prefix']:.3f}  "
              f"union {r['union']:.3f} at {r['union_reads']:.2f}x reads")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--root", default="~/.global_agent/expert_sidecar")
    ap.add_argument("--min-traj", type=int, default=16)
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    if args.slot:
        banks = [root / f"{args.slot}_wrap_bank.npz"]
    else:
        banks = sorted(root.glob("*_wrap_bank.npz"))
    if not banks:
        print(f"no banks under {root}")
        return 1

    for path in banks:
        trajectories, meta = load_bank(path)
        if len(trajectories) < args.min_traj:
            continue
        report(path.name.replace("_wrap_bank.npz", ""), trajectories, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
