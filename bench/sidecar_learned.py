#!/usr/bin/env python3
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fit the best *shippable* residency predictor and replay it as a policy.

Oracle protection of the next H tokens' demand is worth 17-32% of demand
misses (bench/sidecar_sim.py), and protection spends no bytes and cannot change
an output - so it is the most valuable thing a sidecar could possibly do. The
question this answers is whether the oracle's signal survives being predicted.

Method, deliberately generous to the learned side:

  * features are the cheap per-(layer, expert) counters a head can maintain
    online for a few bytes per key
  * the model is fitted offline on the *same* trace it is then replayed on, so
    there is no train/test penalty at all - this is an upper bound
  * the policy is given a full global ranking, not the bounded LRU-tail scan
    the online head would be limited to

If an upper bound constructed this favourably still does not beat LRU by a
worthwhile margin, then residency prediction is oracle-only and a residency
head should not ship.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pagedmoe.sidecar.bank import load_bank  # noqa: E402

FEATURES = (
    "decayed_popularity",
    "count_last_h",
    "tokens_since_prev",
    "run_length",
    "layer_frac",
    "bias",
)


def frames_of(trajectories, n_layers, n_experts):
    out = []
    tid = []
    for i, traj in enumerate(trajectories):
        for frame in traj:
            row = np.zeros((n_layers, n_experts), dtype=bool)
            for layer, ids in frame.items():
                li = int(layer)
                if 0 <= li < n_layers:
                    a = np.asarray(ids, dtype=np.int64)
                    a = a[(a >= 0) & (a < n_experts)]
                    row[li, a] = True
            out.append(row)
            tid.append(i)
    return np.asarray(out), np.asarray(tid)


class State:
    """The counters a residency head would keep. A few bytes per (layer, expert)."""

    def __init__(self, n_layers, n_experts, horizon):
        self.h = horizon
        self.pop = np.zeros((n_layers, n_experts), dtype=np.float32)
        self.last = np.full((n_layers, n_experts), -10_000, dtype=np.int32)
        self.recent = np.zeros((n_layers, n_experts), dtype=np.float32)
        self.ring = np.zeros((horizon, n_layers, n_experts), dtype=bool)
        self.runlen = np.zeros((n_layers, n_experts), dtype=np.float32)
        self.layer_frac = (
            np.arange(n_layers, dtype=np.float32)[:, None] / max(1, n_layers - 1)
        ) * np.ones((1, n_experts), dtype=np.float32)
        self.t = 0

    def feat(self, sel):
        return np.stack(
            [
                self.pop[sel],
                self.recent[sel],
                np.minimum(64.0, self.t - self.last[sel]).astype(np.float32),
                self.runlen[sel],
                self.layer_frac[sel],
                np.ones(sel[0].shape[0], dtype=np.float32),
            ],
            axis=1,
        )

    def feat_all(self):
        n = self.pop.size
        return np.stack(
            [
                self.pop.reshape(n),
                self.recent.reshape(n),
                np.minimum(64.0, self.t - self.last.reshape(n)).astype(np.float32),
                self.runlen.reshape(n),
                self.layer_frac.reshape(n),
                np.ones(n, dtype=np.float32),
            ],
            axis=1,
        )

    def advance(self, cur):
        slot = self.t % self.h
        self.recent -= self.ring[slot]
        self.ring[slot] = cur
        self.recent += cur
        self.pop *= 0.97
        self.pop += cur
        self.runlen = np.where(cur, self.runlen + 1.0, 0.0)
        self.last = np.where(cur, self.t, self.last)
        self.t += 1


def forward_reuse(D, tid, horizon):
    """Label: demanded at least once in t+1 .. t+horizon (same trajectory)."""
    T = D.shape[0]
    fwd = np.zeros_like(D)
    for t in range(T):
        acc = None
        for u in range(t + 1, min(T, t + horizon + 1)):
            if tid[u] != tid[t]:
                break
            acc = D[u] if acc is None else (acc | D[u])
        if acc is not None:
            fwd[t] = acc
    return fwd


def fit_logistic(X, y, epochs=12, lr=0.5):
    """Plain logistic regression with standardized features."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-6] = 1.0
    mu[-1], sd[-1] = 0.0, 1.0  # keep the bias column intact
    Z = (X - mu) / sd
    w = np.zeros(Z.shape[1], dtype=np.float64)
    n = Z.shape[0]
    rng = np.random.default_rng(0)
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, 8192):
            b = idx[s : s + 8192]
            p = 1.0 / (1.0 + np.exp(-(Z[b] @ w)))
            g = Z[b].T @ (p - y[b]) / b.size
            w -= lr * g
    p = 1.0 / (1.0 + np.exp(-(Z @ w)))
    auc = roc_auc(y, p)
    return w, mu, sd, auc


def roc_auc(y, s):
    order = np.argsort(s, kind="mergesort")
    r = np.empty(s.size, dtype=np.float64)
    r[order] = np.arange(1, s.size + 1)
    xs = s[order]
    i = 0
    while i < xs.size:
        j = i
        while j + 1 < xs.size and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n1 = float(y.sum())
    n0 = float(y.size - n1)
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def replay(D, tid, slots, policy, *, w=None, mu=None, sd=None, horizon=16, oracle=None):
    """Replay residency over the dense demand stream.

    `policy`:
      lru       evict the least recently used entry
      learned   evict the entry with the lowest predicted P(reuse within H),
                ranked globally over the whole residency
      oracle    evict the least recently used entry not demanded within H
    """
    T, L, E = D.shape
    st = State(L, E, horizon)
    resident: OrderedDict = OrderedDict()
    hits = misses = 0
    scores_flat = None

    for t in range(T):
        cur = D[t]
        if policy == "learned":
            Z = (st.feat_all() - mu) / sd
            scores_flat = Z @ w  # monotone in probability; no sigmoid needed
        sel = np.nonzero(cur)
        for li, ei in zip(sel[0].tolist(), sel[1].tolist()):
            key = li * E + ei
            if key in resident:
                hits += 1
                resident.move_to_end(key)
                continue
            misses += 1
            while len(resident) >= slots:
                if policy == "lru":
                    victim = next(iter(resident))
                elif policy == "learned":
                    # Global argmin over residency. The online head would use a
                    # bounded scan, so this is the favourable case.
                    keys = np.fromiter(resident.keys(), dtype=np.int64, count=len(resident))
                    victim = int(keys[int(np.argmin(scores_flat[keys]))])
                elif policy == "oracle":
                    victim = None
                    for cand in resident:
                        if not oracle[t].reshape(-1)[cand]:
                            victim = cand
                            break
                    if victim is None:
                        victim = next(iter(resident))
                else:
                    raise ValueError(policy)
                resident.pop(victim, None)
            resident[key] = True
        st.advance(cur)

    total = hits + misses
    return {
        "policy": policy,
        "hit_rate": round(hits / total, 4) if total else 0.0,
        "demand_misses": misses,
        "misses_per_token": round(misses / max(1, T), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--cache-gb", type=float, default=28.0)
    ap.add_argument("--expert-mb", type=float, default=10.6)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--min-traj", type=int, default=150)
    ap.add_argument("--max-frames", type=int, default=1500)
    args = ap.parse_args()

    trajectories, meta = load_bank(Path(args.bank).expanduser())
    n_layers = int(meta["n_layers"])
    n_experts = int(meta["n_experts"])
    trajectories = [t for t in trajectories if len(t) >= args.min_traj]
    trimmed, budget = [], args.max_frames
    for traj in trajectories:
        if budget <= 0:
            break
        trimmed.append(traj[:budget])
        budget -= len(trimmed[-1])
    trajectories = trimmed

    D, tid = frames_of(trajectories, n_layers, n_experts)
    slots = int(args.cache_gb * 1e9 / (args.expert_mb * 1e6))
    fwd = forward_reuse(D, tid, args.horizon)

    # ---- assemble the training set over just-demanded keys
    st = State(n_layers, n_experts, args.horizon)
    Xs, ys = [], []
    for t in range(D.shape[0]):
        sel = np.nonzero(D[t])
        if t > args.horizon and sel[0].size:
            Xs.append(st.feat(sel))
            ys.append(fwd[t][sel])
        st.advance(D[t])
    X = np.concatenate(Xs).astype(np.float64)
    y = np.concatenate(ys).astype(np.float64)
    w, mu, sd, auc = fit_logistic(X, y)

    out = {
        "bank": Path(args.bank).name,
        "tokens": int(D.shape[0]),
        "slots": slots,
        "horizon": args.horizon,
        "fit": {
            "n": int(y.size),
            "base_rate": round(float(y.mean()), 4),
            "train_auc": round(auc, 4),
            "weights": {k: round(float(v), 4) for k, v in zip(FEATURES, w)},
        },
        "policies": [],
    }
    for policy in ("lru", "learned", "oracle"):
        r = replay(
            D, tid, slots, policy,
            w=w, mu=mu, sd=sd, horizon=args.horizon, oracle=fwd,
        )
        out["policies"].append(r)
    base = out["policies"][0]["demand_misses"]
    for r in out["policies"]:
        r["miss_delta_vs_lru"] = round((r["demand_misses"] - base) / base, 4)

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
