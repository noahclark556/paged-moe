#!/usr/bin/env python3
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A Hawkes intensity as the causal estimator of Belady's key.

WHY THIS AND NOT MORE FEATURES. bench/belady_anatomy.py narrowed the residency
problem to one question. Coarsened Belady - evict any entry that will not be
touched for 32+ tokens rather than the furthest one - captures 95% of the
oracle's win, so the target is a classifier. But the payoff in classifier
accuracy is violently nonlinear (right 90% of the time buys -5.8% misses, right
100% buys -31.6%), a linear model on 10 causal features saturates at AUC 0.774,
and gradient boosting on the SAME features scored *worse* (0.672) because it
overfits one segment of the trace and does not transfer. So the ceiling is not
model capacity, it is model specification.

Expert demand is a self-exciting point process, and not approximately: 88% of
just-demanded experts are demanded again within 16 tokens. That is the defining
property of a Hawkes process,

    lambda_k(t) = mu_k + sum_s alpha_s * sum_{t_i < t} exp(-beta_s (t - t_i))

and it matters here for a specific reason. The expected time to the next event
of a point process is ~1/lambda, so a Hawkes intensity is a direct causal
estimator of *exactly the quantity Belady sorts on*. The shipped head
approximates this with a single-timescale EMA of "was it reused within H",
which is why it saturates near 1 and why a threshold on it protects everything.
A multi-timescale kernel can separate "bursting right now" from "persistently
popular", which one EMA provably cannot.

Three things are tested here, in increasing order of how much they could
matter:

  1. raw Hawkes intensity as a ranking, with no fitting at all
  2. a logistic fit on the decayed counters - a Hawkes-shaped LINEAR model,
     which keeps the transfer property that the tree lost
  3. layer-relative intensity, because eviction compares candidates across
     different layers and an absolute rate is not comparable between a layer
     whose demand is concentrated and one whose demand is flat

Usage:
    python bench/hawkes_cold.py --bank <bank.npz> --slots 754
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from belady_anatomy import (
    _true_gap,
    auc,
    build_trace,
    fit_logistic,
    study_cold,
)

from pagedmoe.sidecar.bank import load_bank

# Exponential kernel decay rates, in units of 1/token. Half-lives of roughly
# 1.4, 4.6, 14 and 46 tokens - chosen to bracket the 16-token reuse horizon
# from both sides, since the whole point is to tell timescales apart.
BETAS = (0.5, 0.15, 0.05, 0.015)

HAWKES_FEATURES = (
    [f"exc_{b}" for b in BETAS]
    + [
        "mu",            # long-run baseline rate for this key
        "intensity",     # mu + sum of excitations (the Hawkes lambda)
        "log_next",      # log(1 / lambda): estimated tokens to next event
        "rel_layer",     # lambda / mean lambda over live keys at this layer
        "rank_layer",    # within-layer rank of lambda, normalized
        "age",           # tokens since last demand (LRU's key, for reference)
        "phase",         # (layer - current layer) mod N, normalized
    ]
)


class Hawkes:
    """Multi-timescale self-excitation, O(1) per event per timescale.

    The recursion is the standard one: an exponentially-decayed counter only
    needs the elapsed time since it was last touched, so there is no event
    history to store and no sum to recompute. Four timescales over a 14k-key
    table is four numpy multiplies - affordable on the decode path, which is
    the only reason this is worth studying as a shipping candidate rather than
    as an offline curiosity.
    """

    def __init__(self, n_keys, n_layers, key_layer, betas=BETAS):
        self.betas = np.asarray(betas, dtype=np.float32)
        self.exc = np.zeros((len(betas), n_keys), dtype=np.float32)
        self.stamp = np.zeros(n_keys, dtype=np.int64)
        self.hits = np.zeros(n_keys, dtype=np.float32)
        self.first = np.full(n_keys, -1, dtype=np.int64)
        self.last = np.full(n_keys, -10_000, dtype=np.int64)
        self.n_layers = n_layers
        # Keys are interned in first-seen order, so the layer of a key is a
        # lookup, not arithmetic on the key id.
        self.key_layer = np.asarray(key_layer, dtype=np.int32)
        self.tok = 0
        # Per-layer mean intensity, refreshed lazily; makes scores comparable
        # across layers without a full table sweep per eviction.
        self._layer_mean = np.full(n_layers, 1e-3, dtype=np.float32)

    def _decayed(self, keys, tok):
        """Excitation counters for `keys`, brought forward to token `tok`."""
        dt = np.maximum(0, tok - self.stamp[keys]).astype(np.float32)
        # exp(-beta * dt) for each timescale: [n_betas, len(keys)]
        return self.exc[:, keys] * np.exp(-self.betas[:, None] * dt[None, :])

    def observe(self, keys, tok):
        keys = np.asarray(keys, dtype=np.int64)
        if keys.size == 0:
            return
        self.exc[:, keys] = self._decayed(keys, tok) + 1.0
        self.stamp[keys] = tok
        self.hits[keys] += 1.0
        new = self.first[keys] < 0
        if new.any():
            self.first[keys[new]] = tok
        self.last[keys] = tok
        self.tok = tok

    def refresh_layer_means(self, tok):
        live = self.hits > 0
        if not live.any():
            return
        keys = np.nonzero(live)[0]
        lam = self.intensity(keys, tok)
        layers = np.clip(self.key_layer[keys], 0, self.n_layers - 1).astype(
            np.int64
        )
        s = np.zeros(self.n_layers, dtype=np.float64)
        c = np.zeros(self.n_layers, dtype=np.float64)
        np.add.at(s, layers, lam)
        np.add.at(c, layers, 1.0)
        self._layer_mean = np.maximum(1e-3, s / np.maximum(1.0, c)).astype(
            np.float32
        )

    def mu(self, keys, tok):
        life = np.maximum(1.0, tok - self.first[keys]).astype(np.float32)
        return self.hits[keys] / life

    def intensity(self, keys, tok):
        exc = self._decayed(keys, tok)
        return self.mu(keys, tok) + exc.sum(axis=0)

    def features(self, keys, tok, cur_layer):
        keys = np.asarray(keys, dtype=np.int64)
        exc = self._decayed(keys, tok)               # [n_betas, k]
        mu = self.mu(keys, tok)
        lam = np.maximum(1e-4, mu + exc.sum(axis=0))
        layers = np.clip(self.key_layer[keys], 0, self.n_layers - 1)
        rel = lam / self._layer_mean[layers]
        age = np.maximum(0, tok - self.last[keys]).astype(np.float32)
        phase = ((layers - cur_layer) % self.n_layers) / self.n_layers
        cols = [exc[i] for i in range(exc.shape[0])]
        cols += [
            mu,
            lam,
            np.log(1.0 / lam),
            rel,
            # Within-layer rank is not exactly computable per-candidate without
            # the layer's full score vector; rel is its cheap monotone proxy,
            # and the log keeps the tail from dominating the fit.
            np.log1p(rel),
            np.log1p(age),
            phase.astype(np.float32),
        ]
        return np.stack(cols, axis=1).astype(np.float64)


def study_hawkes(
    key_ids,
    layers_of,
    boundaries,
    slots,
    *,
    n_layers,
    key_layer,
    per_token,
    cold_tokens=32.0,
    sample=16,
    seed=0,
):
    """Same sampling protocol as study_cold, with Hawkes features instead.

    Sampled at real LRU eviction decisions so the two feature sets are scored
    on identical candidate distributions; anything else would compare models on
    different questions.
    """
    rng = np.random.default_rng(seed)
    n_keys = int(key_ids.max()) + 1
    hk = Hawkes(n_keys, n_layers, key_layer)
    resident: OrderedDict = OrderedDict()
    rlist: list[int] = []
    rpos: dict = {}
    bset = {int(b) for b in boundaries.tolist()}
    X, y = [], []
    cur_token = -1
    pending: list[int] = []

    def _add(k):
        rpos[k] = len(rlist)
        rlist.append(k)

    def _drop(k):
        j = rpos.pop(k, None)
        if j is None:
            return
        last = rlist.pop()
        if j < len(rlist):
            rlist[j] = last
            rpos[last] = j

    for i in range(key_ids.shape[0]):
        if i in bset:
            resident.clear()
            rlist.clear()
            rpos.clear()
        k = int(key_ids[i])
        cur_layer = int(layers_of[i])
        tok = int(i // max(1, per_token))
        if tok != cur_token:
            # Fold the previous token's demand in one vectorized update, which
            # is how this would run in the engine (end_token, not per layer).
            if pending:
                hk.observe(np.asarray(pending, dtype=np.int64), cur_token)
                pending = []
            cur_token = tok
            if tok % 32 == 0:
                hk.refresh_layer_means(tok)

        if k in resident:
            resident.move_to_end(k)
        else:
            if len(resident) >= slots:
                m = len(rlist)
                idx = rng.integers(0, m, size=min(sample, m))
                cands = np.asarray([rlist[int(j)] for j in idx], dtype=np.int64)
                X.append(hk.features(cands, tok, cur_layer))
                y.extend(
                    1.0 if _true_gap(c, i, key_ids, per_token) >= cold_tokens
                    else 0.0
                    for c in cands.tolist()
                )
                victim = next(iter(resident))
                resident.pop(victim, None)
                _drop(victim)
            resident[k] = True
            _add(k)
        pending.append(k)

    if not X:
        return np.zeros((0, len(HAWKES_FEATURES))), np.zeros(0)
    return np.concatenate(X, axis=0), np.asarray(y, dtype=np.float64)


def coverage_at_precision(p, y, target):
    order = np.argsort(-p)
    ys = y[order]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    ok = np.nonzero(prec >= target)[0]
    if ok.size == 0:
        return 0.0
    return (int(ok[-1]) + 1) / len(ys)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--slots", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--min-traj", type=int, default=16)
    ap.add_argument("--cold-tokens", type=float, default=32.0)
    args = ap.parse_args()

    trajectories, meta = load_bank(Path(args.bank).expanduser())
    trajectories = [t for t in trajectories if len(t) >= args.min_traj]
    n_layers = int(meta.get("n_layers") or 0) or 1 + max(
        (max(f) for traj in trajectories for f in traj if f), default=0
    )
    n_experts = int(meta.get("n_experts") or 0) or 1 + max(
        (max(v) for traj in trajectories for f in traj for v in f.values() if v),
        default=0,
    )
    kept, total = [], 0
    for traj in trajectories:
        if total >= args.max_tokens:
            break
        kept.append(traj)
        total += len(traj)

    key_ids, layers_of, _t, boundaries, n_keys = build_trace(kept, n_layers)
    key_layer = np.zeros(n_keys, dtype=np.int32)
    key_layer[key_ids] = layers_of
    per_token = len(key_ids) / max(1, total)
    print(f"trace {total} tokens, {len(key_ids)} accesses, {n_keys} keys, "
          f"{n_layers}L x {n_experts}E, cache {args.slots} slots "
          f"({args.slots / per_token:.2f} tokens of demand)")
    print(f"label: cold = next use >= {args.cold_tokens:.0f} tokens\n")

    # Baseline: the 10 generic causal features from belady_anatomy.
    Xb, yb = study_cold(
        key_ids, layers_of, boundaries, args.slots,
        n_layers=n_layers, key_layer=key_layer, per_token=per_token,
        cold_tokens=args.cold_tokens,
    )
    Xh, yh = study_hawkes(
        key_ids, layers_of, boundaries, args.slots,
        n_layers=n_layers, key_layer=key_layer, per_token=per_token,
        cold_tokens=args.cold_tokens,
    )

    rows = []
    for name, X, y in (("generic (10 feats)", Xb, yb), ("hawkes", Xh, yh)):
        cut = int(len(X) * 0.7)
        w, mu, sd = fit_logistic(X[:cut], y[:cut])
        Z = np.concatenate(
            [(X[cut:] - mu) / sd, np.ones((len(X) - cut, 1))], axis=1
        )
        p = 1.0 / (1.0 + np.exp(-(Z @ w)))
        rows.append((name, auc(p, y[cut:]), p, y[cut:], w))

    # And the unfitted Hawkes ranking: 1/lambda with no learning at all.
    cut = int(len(Xh) * 0.7)
    li = HAWKES_FEATURES.index("log_next")
    rows.append(
        ("hawkes 1/lambda (unfit)", auc(Xh[cut:, li], yh[cut:]),
         Xh[cut:, li], yh[cut:], None)
    )

    print(f"{'model':<26} {'AUC':>6} {'cov@0.6p':>9} {'cov@0.7p':>9} "
          f"{'cov@0.8p':>9}")
    for name, a, p, y, _w in rows:
        print(f"{name:<26} {a:>6.3f} "
              f"{coverage_at_precision(p, y, 0.6):>9.3f} "
              f"{coverage_at_precision(p, y, 0.7):>9.3f} "
              f"{coverage_at_precision(p, y, 0.8):>9.3f}")

    print("\nhawkes weights (standardized, + = predicts cold):")
    w = rows[1][4]
    for nm, val in zip(HAWKES_FEATURES, w[:-1]):
        print(f"  {nm:<11} {val:+.3f}")
    print("\nsingle-feature AUC:")
    for i, nm in enumerate(HAWKES_FEATURES):
        a = auc(Xh[cut:, i], yh[cut:])
        print(f"  {nm:<11} {a:.3f}" + ("  (inverted)" if a < 0.5 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
