#!/usr/bin/env python3
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""How much signal is actually in the expert demand stream?

Everything a sidecar can do rests on one question: given the past, how much
better than chance can you guess which experts a layer will want next? This
measures that directly on saved wrap banks, before any model is fitted, so we
know the ceiling instead of discovering it after shipping.

Reports, per bank:

  concentration   per layer, the share of demand taken by the top-k experts.
                  Flat means residency ranking has little to rank.
  persistence     P(expert demanded at t+1 | demanded at t) vs the base rate.
                  This is the lift the "copy the last token" baseline already
                  captures, and the upper bound on any Markov-order-1 head.
  horizon         P(demanded at t+h | demanded at t) as h grows: how long a
                  demand observation stays informative. Sets the useful
                  residency horizon and the decay constant.
  belady_split    of the experts Belady keeps that LRU evicts, how many were
                  predictable from marginal popularity alone. Separates
                  "oracle advantage" (not learnable) from "better ranking"
                  (learnable).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pagedmoe.sidecar.bank import load_bank  # noqa: E402


def demand_matrix(trajectories, n_layers: int, n_experts: int):
    """Dense [tokens, n_layers, n_experts] boolean demand, plus trajectory ids."""
    frames = []
    traj_id = []
    for t, traj in enumerate(trajectories):
        for frame in traj:
            row = np.zeros((n_layers, n_experts), dtype=bool)
            for layer, ids in frame.items():
                li = int(layer)
                if 0 <= li < n_layers:
                    arr = np.asarray(ids, dtype=np.int64)
                    arr = arr[(arr >= 0) & (arr < n_experts)]
                    row[li, arr] = True
            frames.append(row)
            traj_id.append(t)
    return np.asarray(frames), np.asarray(traj_id)


def concentration(D: np.ndarray, ks=(4, 8, 16, 32)):
    """Share of each layer's demand carried by its k most popular experts.

    A residency policy can only pay off if popularity is skewed: with a flat
    distribution every resident expert is equally likely to be wanted and any
    ranking is noise.
    """
    counts = D.sum(axis=0).astype(np.float64)  # [layers, experts]
    total = counts.sum(axis=1, keepdims=True)
    srt = -np.sort(-counts, axis=1)
    cum = np.cumsum(srt, axis=1)
    out = {}
    for k in ks:
        k = min(k, counts.shape[1])
        share = cum[:, k - 1] / np.maximum(total[:, 0], 1)
        out[f"top{k}_share"] = round(float(np.mean(share)), 4)
    # Effective number of experts per layer (perplexity of the popularity dist).
    p = counts / np.maximum(total, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    out["effective_experts"] = round(float(np.mean(np.exp(ent))), 2)
    out["n_experts"] = int(counts.shape[1])
    return out


def persistence(D: np.ndarray, traj_id: np.ndarray, horizons=(1, 2, 4, 8, 16, 32)):
    """P(demanded at t+h | demanded at t) against the base rate.

    The ratio is the lift a demand observation gives you h tokens later. When
    it falls to 1.0 the observation is worthless and nothing built on demand
    history can help at that horizon.
    """
    out = {}
    base = float(D.mean())
    for h in horizons:
        same = traj_id[:-h] == traj_id[h:]
        if not same.any():
            continue
        a = D[:-h][same]
        b = D[h:][same]
        joint = float((a & b).sum())
        cond = joint / max(1.0, float(a.sum()))
        out[f"h{h}"] = {
            "p_given_demanded": round(cond, 4),
            "lift_vs_base": round(cond / max(1e-9, base), 3),
        }
    out["base_rate"] = round(base, 4)
    return out


def next_token_predictability(D: np.ndarray, traj_id: np.ndarray, wrap: int):
    """Best achievable next-token recall from three causal predictors.

    * copy: the previous token's experts (what the engine already prefetches)
    * popular: the k globally most popular experts of that layer
    * union: both together

    If `popular` adds little over `copy`, then a learned head - which can at
    best interpolate between "what just fired" and "what usually fires" - has
    almost nothing to add, and every read it issues is pure cost.
    """
    same = traj_id[:-1] == traj_id[1:]
    A, B = D[:-1][same], D[1:][same]
    counts = D.sum(axis=0)
    out = {}
    for k in (2, 4, 8):
        rec_copy = rec_pop = rec_union = 0.0
        need = 0.0
        for layer in range(wrap):
            want = B[:, layer, :]
            have = A[:, layer, :]
            top = np.argpartition(-counts[layer], k - 1)[:k]
            pop = np.zeros(D.shape[2], dtype=bool)
            pop[top] = True
            n = want.sum()
            need += n
            rec_copy += (want & have).sum()
            rec_pop += (want & pop[None, :]).sum()
            rec_union += (want & (have | pop[None, :])).sum()
        out[f"k{k}"] = {
            "copy_recall": round(float(rec_copy / max(1, need)), 4),
            "popular_recall": round(float(rec_pop / max(1, need)), 4),
            "union_recall": round(float(rec_union / max(1, need)), 4),
            "popular_adds": round(float((rec_union - rec_copy) / max(1, need)), 4),
            "extra_reads_per_token": k * wrap,
        }
    return out


def belady_vs_popularity(D: np.ndarray, traj_id: np.ndarray, slots: int):
    """Is Belady's edge a better *ranking*, or knowledge of the future?

    At each token, for every layer, look at the experts demanded in the next
    token. Ask how well marginal popularity ranks them against the experts not
    demanded. AUC near 0.5 means popularity cannot separate them at all - so
    the Belady gap is oracle advantage and no causal policy, learned or not,
    can close it.
    """
    counts = D.sum(axis=0).astype(np.float64)
    same = traj_id[:-1] == traj_id[1:]
    B = D[1:][same]
    n_layers, n_experts = D.shape[1], D.shape[2]
    aucs = []
    rng = np.random.default_rng(0)
    idx = rng.choice(B.shape[0], size=min(512, B.shape[0]), replace=False)
    for layer in range(n_layers):
        c = counts[layer]
        for i in idx:
            pos = np.nonzero(B[i, layer])[0]
            if pos.size == 0 or pos.size == n_experts:
                continue
            neg = np.nonzero(~B[i, layer])[0]
            # Mann-Whitney U via rank comparison on a bounded sample.
            ns = min(64, neg.size)
            neg_s = neg[rng.integers(0, neg.size, ns)]
            wins = (c[pos][:, None] > c[neg_s][None, :]).mean()
            ties = (c[pos][:, None] == c[neg_s][None, :]).mean()
            aucs.append(float(wins + 0.5 * ties))
    return {
        "popularity_auc_next_token": round(float(np.mean(aucs)), 4),
        "note": "0.5 = popularity is uninformative about next-token demand",
    }


def reuse_within_horizon_auc(D: np.ndarray, traj_id: np.ndarray, horizon: int = 16):
    """Can any causal feature rank "wanted again within H tokens"?

    This is the decisive question for the residency head. Oracle protection of
    the next H tokens' demand is worth 17-32% of demand misses, and protection
    costs no bytes and cannot change an output - so if the target is
    predictable, it is the best thing a sidecar can possibly do. If it is not,
    residency is oracle-only and the head should not exist.

    Scored as AUC over the population that matters: (layer, expert) pairs that
    were just demanded, i.e. exactly the entries sitting in the cache competing
    to survive eviction.
    """
    T, L, E = D.shape
    # Target: demanded at some point in t+1 .. t+H, for keys demanded at t.
    fwd = np.zeros_like(D)
    acc = np.zeros((L, E), dtype=np.int32)
    # Rolling OR over the next H tokens, respecting trajectory boundaries.
    for t in range(T - 1, -1, -1):
        acc = acc * 0
        hi = min(T, t + horizon + 1)
        for u in range(t + 1, hi):
            if traj_id[u] != traj_id[t]:
                break
            acc |= D[u]
        fwd[t] = acc.astype(bool)

    # Causal features, all computable online for a few bytes per key.
    pop = np.zeros((L, E), dtype=np.float32)  # decayed demand count
    last = np.full((L, E), -10_000, dtype=np.int32)
    recent = np.zeros((L, E), dtype=np.float32)  # count in last H tokens
    ring = np.zeros((horizon, L, E), dtype=bool)

    feats: dict[str, list] = {
        "decayed_popularity": [],
        "count_last_H": [],
        "tokens_since_prev_use": [],
        "run_length": [],
    }
    labels: list = []
    runlen = np.zeros((L, E), dtype=np.float32)

    for t in range(T):
        cur = D[t]
        sel = np.nonzero(cur)
        if t > horizon and sel[0].size:
            labels.append(fwd[t][sel])
            feats["decayed_popularity"].append(pop[sel])
            feats["count_last_H"].append(recent[sel])
            feats["tokens_since_prev_use"].append(
                np.minimum(1000, t - last[sel]).astype(np.float32)
            )
            feats["run_length"].append(runlen[sel])
        # advance state
        slot = t % horizon
        recent -= ring[slot]
        ring[slot] = cur
        recent += cur
        pop *= 0.97
        pop += cur
        runlen = np.where(cur, runlen + 1.0, 0.0)
        last = np.where(cur, t, last)

    y = np.concatenate(labels).astype(np.float64)
    out = {"horizon": horizon, "n": int(y.size), "base_rate": round(float(y.mean()), 4)}
    for name, parts in feats.items():
        x = np.concatenate(parts).astype(np.float64)
        # AUC via rank statistic (handles ties correctly).
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, x.size + 1)
        # average ranks within tie groups
        xs = x[order]
        i = 0
        while i < xs.size:
            j = i
            while j + 1 < xs.size and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                ranks[order[i : j + 1]] = (i + j + 2) / 2.0
            i = j + 1
        n1 = float(y.sum())
        n0 = float(y.size - n1)
        if n1 == 0 or n0 == 0:
            out[name] = None
            continue
        auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        # Sign-free: report the better of the feature and its negation, and say
        # which direction carries the information.
        out[name] = {
            "auc": round(float(max(auc, 1 - auc)), 4),
            "direction": "higher=reused" if auc >= 0.5 else "lower=reused",
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--max-frames", type=int, default=6000)
    args = ap.parse_args()

    trajectories, meta = load_bank(Path(args.bank).expanduser())
    n_layers = int(meta.get("n_layers") or 0)
    n_experts = int(meta.get("n_experts") or 0)
    wrap = int(meta.get("wrap") or 6)
    if args.max_frames:
        trimmed, budget = [], args.max_frames
        for traj in trajectories:
            if budget <= 0:
                break
            trimmed.append(traj[:budget])
            budget -= len(trimmed[-1])
        trajectories = trimmed

    D, traj_id = demand_matrix(trajectories, n_layers, n_experts)
    out = {
        "bank": Path(args.bank).name,
        "shape": {"tokens": int(D.shape[0]), "layers": n_layers, "experts": n_experts},
        "concentration": concentration(D),
        "persistence": persistence(D, traj_id),
        "next_token": next_token_predictability(D, traj_id, wrap),
        "belady_vs_popularity": belady_vs_popularity(D, traj_id, 0),
        "reuse_auc": {
            str(h): reuse_within_horizon_auc(D, traj_id, h) for h in (4, 16)
        },
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
