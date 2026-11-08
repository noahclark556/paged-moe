#!/usr/bin/env python3
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What is Belady actually doing that LRU is not?

Belady cuts misses 40-48% on these traces and every causal policy tried so far
lands at or behind LRU. Two very different worlds produce that result:

  (a) Belady's edge is exact future knowledge of *which* equally-plausible
      expert gets touched next. Unreachable - stop here.
  (b) Belady's edge is mostly STRUCTURAL, and LRU is losing to a pathology
      rather than to ignorance. MoE decode walks layers 0..N-1 in order, once
      per token, forever. That is a cyclic scan, and a cyclic scan is the
      textbook worst case for LRU: right after layer i runs, its experts are
      the most-recently-used and layer i+1's are among the LEAST recent (they
      last ran a full token ago), so LRU preferentially evicts the experts
      about to be demanded milliseconds from now.

If (b) holds, a policy can capture part of the gap with no prediction at all,
because the distance from the current layer to layer L's next execution is
    d = (L - i) mod N
which is EXACT - it comes from the shape of the forward pass, not from a model.
The only uncertain part is whether layer L will want this particular expert
when it runs, and that is a per-(layer, expert) rate the sidecar already counts.
Expected next use in layer-steps is then

    E[next use] ~= d + N * (1 - q) / q

with q the expert's demand rate at its layer: exact structure times learned
probability. Neither LRU (recency only) nor the freq policies (q only) use both.

This script settles which world we are in, by instrumenting the evictions
themselves rather than the miss totals:

  1. the cyclic layer phase of each victim, LRU vs Belady vs phase
  2. the victim's true next-use distance, so "how wrong was this choice" is
     measured in the same units Belady optimizes
  3. per-layer slot allocation, since a scan pathology shows up as a tilt
  4. the phase policy's actual miss count against both bounds

Usage:
    python bench/belady_anatomy.py --bank ~/.paged_moe/sidecar/<id>_wrap_bank.npz \\
        --slots 754
"""

from __future__ import annotations

import argparse
import heapq
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pagedmoe.sidecar.bank import load_bank


def build_trace(trajectories, n_layers):
    keys: dict = {}
    key_ids, layers, tokens, boundaries = [], [], [], []
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


def next_use_positions(key_ids, n_keys):
    n = key_ids.shape[0]
    nxt = np.full(n, n, dtype=np.int64)
    last = np.full(n_keys, -1, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        k = key_ids[i]
        p = last[k]
        if p >= 0:
            nxt[i] = p
        last[k] = i
    return nxt


class Recorder:
    """Per-eviction bookkeeping, in the units Belady optimizes."""

    def __init__(self, n_layers):
        self.phase = np.zeros(n_layers, dtype=np.int64)
        self.next_use_gap = []
        self.evictions = 0
        self.n_layers = n_layers

    def note(self, victim_layer, cur_layer, next_use_gap):
        d = (int(victim_layer) - int(cur_layer)) % self.n_layers
        self.phase[d] += 1
        self.next_use_gap.append(next_use_gap)
        self.evictions += 1

    def summary(self):
        if not self.evictions:
            return {}
        gaps = np.asarray(self.next_use_gap, dtype=np.float64)
        finite = gaps[np.isfinite(gaps)]
        ph = self.phase / self.phase.sum()
        # "Soon" = the victim's layer runs again within a fifth of a token.
        soon = int(max(1, self.n_layers // 5))
        return {
            "evictions": self.evictions,
            "victim_runs_again_within_20pct_of_token": float(ph[1 : soon + 1].sum()),
            "victim_layer_just_ran": float(ph[-soon:].sum()),
            "median_next_use_accesses": float(np.median(finite)) if finite.size else 0.0,
            "pct_reused_within_1_token": float(
                (finite < 1.0).mean() if finite.size else 0.0
            ),
        }


def replay(
    key_ids,
    layers_of,
    boundaries,
    slots,
    policy,
    *,
    next_use=None,
    n_layers=0,
    key_layer=None,
    per_token=1.0,
    record=False,
    lfru_scan=8,
    phase_scan=32,
    q_decay=0.97,
    coarse_gap=32.0,
    cold_model=None,
    cold_bar=0.5,
    aci_alpha=0.3,
    aci_eta=0.05,
):
    """One eviction policy, with optional per-eviction instrumentation."""
    cold_w = cold_mu = cold_sd = None
    cold_stats = None
    if cold_model is not None:
        cold_w, cold_mu, cold_sd = cold_model
        cold_stats = ColdStats(int(key_ids.max()) + 1)

    # Hawkes + adaptive conformal state (policy "hawkes_aci").
    hk = None
    aci_bar = [0.0]
    aci_pending: list[tuple[int, int]] = []
    aci_fired = aci_wrong = 0
    aci_last_seen: dict[int, int] = {}
    cur_token_hk = [-1]
    aci_on = policy in ("hawkes_aci", "learned_aci")
    if policy == "hawkes_aci":
        from hawkes_cold import Hawkes

        hk = Hawkes(int(key_ids.max()) + 1, n_layers, key_layer)
    hk_pending: list[int] = []
    if policy == "learned_aci":
        # ACI works in probability space here, and is clipped: a bar below 0.3
        # fires on candidates the detector is not confident about, which the
        # noisy sweep shows is where the policy turns harmful, and a bar above
        # 0.95 is indistinguishable from switching the head off.
        aci_bar[0] = 0.5
        aci_lo, aci_hi = 0.3, 0.95
    else:
        aci_lo, aci_hi = -1e9, 1e9
    resident: OrderedDict = OrderedDict()
    hits = misses = 0
    bset = {int(b) for b in boundaries.tolist()}
    rec = Recorder(n_layers) if record else None

    n_keys = int(key_ids.max()) + 1
    # q: decayed per-key demand rate, i.e. P(this expert is wanted when its
    # layer next runs. Counted causally - this is the only learned quantity
    # in the phase policy, and the sidecar already maintains its equivalent.
    q_count = np.zeros(n_keys, dtype=np.float32)
    q_tok = np.zeros(n_keys, dtype=np.int64)
    q_seen = np.zeros(n_keys, dtype=np.uint8)

    heap: list = []
    stamp: dict = {}
    rng = np.random.default_rng(0)
    rlist: list[int] = []
    rpos: dict[int, int] = {}

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

    def q_of(k, tok):
        seen = int(q_tok[k])
        c = float(q_count[k])
        if tok > seen:
            c *= q_decay ** min(64, tok - seen)
        # A key with no history yet is assumed average rather than cold, so a
        # fresh expert is not thrown out before it has had a chance to repeat.
        return c if q_seen[k] else 0.35

    n = key_ids.shape[0]
    cur_layer = 0
    cur_token = -1
    for i in range(n):
        if i in bset:
            resident.clear()
            rlist.clear()
            rpos.clear()
            heap.clear()
            stamp.clear()
        k = int(key_ids[i])
        cur_layer = int(layers_of[i])
        tok = int(i // max(1, per_token))
        if tok != cur_token:
            cur_token = tok
        if cold_stats is not None:
            cold_stats.roll(tok)
        if aci_on and tok != cur_token_hk[0]:
            # One vectorized fold per token, matching how the engine would run
            # this from end_token rather than per MoE layer.
            if hk is not None:
                if hk_pending:
                    hk.observe(
                        np.asarray(hk_pending, dtype=np.int64), cur_token_hk[0]
                    )
                    hk_pending.clear()
                if tok % 32 == 0:
                    hk.refresh_layer_means(tok)
            cur_token_hk[0] = tok
            # Retire ACI verifications that have aged out: a key evicted more
            # than `coarse_gap` tokens ago and not seen since was genuinely
            # cold, and one seen since was a mistake. Both are observable
            # without an oracle, which is what makes this shippable.
            still = []
            for key, when in aci_pending:
                if tok - when < coarse_gap:
                    still.append((key, when))
                    continue
                aci_fired += 1
                if aci_last_seen.get(key, -1) > when:
                    aci_wrong += 1
                # Adaptive conformal update: push the bar up when the realized
                # error rate exceeds the target, down when it is beating it.
                err = 1.0 if aci_last_seen.get(key, -1) > when else 0.0
                aci_bar[0] = min(
                    aci_hi, max(aci_lo, aci_bar[0] + aci_eta * (err - aci_alpha))
                )
            aci_pending = still

        if k in resident:
            hits += 1
            resident.move_to_end(k)
        else:
            misses += 1
            if len(resident) >= slots:
                victim = None
                if policy == "lru":
                    victim = next(iter(resident))
                elif policy == "lfru":
                    # LRU tail scan: of the `lfru_scan` oldest entries, drop
                    # the least demanded. This is what the engine ships.
                    best = None
                    for seen_n, cand in enumerate(resident):
                        s = q_of(cand, tok)
                        if best is None or s < best[0]:
                            best = (s, cand)
                        if seen_n + 1 >= lfru_scan:
                            break
                    victim = best[1]
                elif policy == "phase":
                    # Exact structural distance to the victim's next execution,
                    # discounted by how likely that execution wants it. Sampled
                    # rather than exhaustive so the cost stays O(scan).
                    best = None
                    m = len(rlist)
                    take = min(phase_scan, m)
                    for j in rng.integers(0, m, size=take):
                        cand = rlist[int(j)]
                        d = (int(key_layer[cand]) - cur_layer) % n_layers
                        q = max(1e-3, min(1.0, q_of(cand, tok)))
                        est = d + n_layers * (1.0 - q) / q
                        if best is None or est > best[0]:
                            best = (est, cand)
                    victim = best[1]
                elif policy == "wide_lfu":
                    # Width without learning. LFRU looks only at the LRU tail,
                    # so it can only pick the least-used of the oldest few -
                    # and the cold entries are not concentrated there. Sample
                    # the whole resident set instead and drop the least
                    # demanded. If this alone captures the win, no model and
                    # no sidecar are needed for it.
                    best = None
                    m = len(rlist)
                    for j in rng.integers(0, m, size=min(phase_scan, m)):
                        cand = rlist[int(j)]
                        s = q_of(cand, tok)
                        if best is None or s < best[0]:
                            best = (s, cand)
                    victim = best[1]
                elif policy == "learned":
                    # Same width, but ranked by the fitted cold detector.
                    best = None
                    m = len(rlist)
                    for j in rng.integers(0, m, size=min(phase_scan, m)):
                        cand = rlist[int(j)]
                        f = cold_stats.features(
                            cand, tok, cur_layer, key_layer, n_layers
                        )
                        z = (np.asarray(f) - cold_mu) / cold_sd
                        s = float(z @ cold_w[:-1] + cold_w[-1])
                        if best is None or s > best[0]:
                            best = (s, cand)
                    victim = best[1]
                elif policy == "learned_gated":
                    # The same detector, but only ACTED ON when confident.
                    # Taking the argmax of a noisy score unconditionally turns
                    # the policy into "random among plausible" on every slate
                    # where nothing is clearly cold - and random is 28% worse
                    # than LRU here. Falling back to LRU when no candidate
                    # clears the bar is what makes a weak detector usable:
                    # the downside is pinned at LRU by construction.
                    best = None
                    m = len(rlist)
                    for j in rng.integers(0, m, size=min(phase_scan, m)):
                        cand = rlist[int(j)]
                        f = cold_stats.features(
                            cand, tok, cur_layer, key_layer, n_layers
                        )
                        z = (np.asarray(f) - cold_mu) / cold_sd
                        s = float(z @ cold_w[:-1] + cold_w[-1])
                        if best is None or s > best[0]:
                            best = (s, cand)
                    p_cold = 1.0 / (1.0 + np.exp(-best[0]))
                    victim = best[1] if p_cold >= cold_bar else next(iter(resident))
                elif policy.startswith("noisy"):
                    # How precise does a cold detector have to BE? Emulate one
                    # that is right `p` of the time: with probability p it
                    # finds a genuinely cold entry in the sample, otherwise it
                    # picks blind. Sweeping p turns "is 0.78 AUC enough" into
                    # a number instead of an argument.
                    p_right = float(policy[5:]) / 100.0
                    m = len(rlist)
                    cands = [rlist[int(j)] for j in rng.integers(0, m, size=min(phase_scan, m))]
                    if rng.random() < p_right:
                        for cand in cands:
                            if _true_gap(cand, i, key_ids, per_token) > coarse_gap:
                                victim = cand
                                break
                    if victim is None:
                        victim = cands[int(rng.integers(0, len(cands)))]
                elif policy == "learned_aci":
                    # The version that actually composes. Conformal cannot
                    # create precision, so it must sit on top of a score that
                    # HAS a high-precision head - the fitted detector reaches
                    # 70% precision on its top 6% of candidates, the unfitted
                    # Hawkes intensity reaches 60% on none of them. What ACI
                    # then adds over the fixed 0.5 bar is the thing a constant
                    # cannot give: the firing rate self-corrects when the
                    # workload shifts, so the precision floor that makes this
                    # policy safe holds without being re-tuned per model.
                    best = None
                    m = len(rlist)
                    for j in rng.integers(0, m, size=min(phase_scan, m)):
                        cand = rlist[int(j)]
                        f = cold_stats.features(
                            cand, tok, cur_layer, key_layer, n_layers
                        )
                        z = (np.asarray(f) - cold_mu) / cold_sd
                        s = float(z @ cold_w[:-1] + cold_w[-1])
                        if best is None or s > best[0]:
                            best = (s, cand)
                    p_cold = 1.0 / (1.0 + np.exp(-best[0]))
                    if p_cold >= aci_bar[0]:
                        victim = best[1]
                        aci_pending.append((victim, tok))
                    else:
                        victim = next(iter(resident))
                elif policy == "hawkes_aci":
                    # Score: a multi-timescale Hawkes intensity, which needs no
                    # training at all - E[time to next event] ~ 1/lambda is
                    # exactly what Belady sorts on, so the estimator is
                    # closed-form rather than fitted. Measured AUC 0.730 for
                    # 1/lambda alone vs 0.767 for a fitted 10-feature logistic.
                    #
                    # Bar: adaptive conformal. The threshold is not a constant
                    # and not a servo on a hand-rolled error signal - it is
                    # updated so the long-run share of WRONG evictions stays at
                    # the target, which is the property that matters because
                    # the payoff curve is a cliff (below ~50% precision this
                    # policy is worse than doing nothing). The feedback is free
                    # and causal: whether an evicted key is re-demanded within
                    # `coarse_gap` tokens is observable a few tokens later, so
                    # nothing here needs an oracle.
                    m = len(rlist)
                    idx = rng.integers(0, m, size=min(phase_scan, m))
                    cands = np.asarray(
                        [rlist[int(j)] for j in idx], dtype=np.int64
                    )
                    lam = hk.intensity(cands, tok)
                    j = int(np.argmin(lam))       # coldest = lowest intensity
                    score = float(np.log(1.0 / max(1e-4, lam[j])))
                    if score >= aci_bar[0]:
                        victim = int(cands[j])
                        # Pending verification: did this key come back soon?
                        aci_pending.append((victim, tok))
                    else:
                        victim = next(iter(resident))
                elif policy == "random":
                    # The control that has to be run before believing any of
                    # this. If Belady's victims are long-gap and LRU's are
                    # short-gap, it is worth knowing whether a coin flip is
                    # already better than LRU on this trace.
                    victim = rlist[int(rng.integers(0, len(rlist)))]
                elif policy == "coarse":
                    # Coarsened Belady: evict ANY resident entry whose true
                    # next use is more than `coarse_gap` tokens away, taking
                    # the first such found in a bounded sample - NOT the
                    # furthest. This is the feasibility question. If a coin
                    # flip among "cold enough" entries captures most of the
                    # oracle's win, then the target is a CLASSIFIER ("is this
                    # one cold?") and not a regression onto exact next-use
                    # order, and a classifier is a far lower bar.
                    m = len(rlist)
                    take = min(phase_scan, m)
                    fallback = None
                    for j in rng.integers(0, m, size=take):
                        cand = rlist[int(j)]
                        g = _true_gap(cand, i, key_ids, per_token)
                        if g > coarse_gap:
                            victim = cand
                            break
                        if fallback is None or g > fallback[0]:
                            fallback = (g, cand)
                    if victim is None:
                        victim = fallback[1]
                elif policy == "belady":
                    while heap:
                        neg, cand, s = heapq.heappop(heap)
                        if cand in resident and stamp.get(cand) == s:
                            victim = cand
                            break
                    if victim is None:
                        victim = next(iter(resident))
                else:
                    raise ValueError(policy)

                if rec is not None and next_use is not None:
                    # How soon was the victim actually needed, in tokens.
                    nu = stamp.get(victim)
                    gap = (
                        (abs(nu) - i) / max(1.0, per_token)
                        if policy == "belady" and nu is not None
                        else _true_gap(victim, i, key_ids, per_token)
                    )
                    rec.note(key_layer[victim], cur_layer, gap)

                resident.pop(victim, None)
                _drop(victim)
            resident[k] = True
            _add(k)

        # update causal demand rate
        seen = int(q_tok[k])
        if tok > seen:
            q_count[k] *= q_decay ** min(64, tok - seen)
        q_count[k] += 1.0
        q_tok[k] = tok
        q_seen[k] = 1
        if cold_stats is not None:
            cold_stats.observe(k, tok)
        if hk is not None:
            hk_pending.append(k)
        if aci_on:
            aci_last_seen[k] = tok

        if policy == "belady" and next_use is not None:
            nu = int(next_use[i])
            stamp[k] = -nu
            heapq.heappush(heap, (-nu, k, -nu))

    total = hits + misses
    out = {
        "policy": policy,
        "misses": misses,
        "hit_rate": round(hits / total, 4) if total else 0.0,
    }
    if rec is not None:
        out.update(rec.summary())
    if policy in ("hawkes_aci", "learned_aci"):
        out["aci_fired"] = aci_fired
        out["aci_precision"] = (
            1.0 - aci_wrong / aci_fired if aci_fired else 0.0
        )
        out["aci_bar"] = round(aci_bar[0], 3)
    return out


# ------------------------------------------------------- is "cold" learnable?

# Causal features for one resident candidate at eviction time. Every one is
# something the engine already has or can keep for a few bytes per slot.
COLD_FEATURES = [
    "age",         # tokens since last demanded  (this is LRU's entire key)
    "hits",        # log1p total demands
    "rate",        # decayed demand count (LFU's key)
    "spread",      # distinct tokens that demanded it / tokens since first seen
    "recent1",     # demanded in the last token
    "recent4",     # demands in the last 4 tokens
    "recent16",    # demands in the last 16 tokens
    "oneshot",     # seen exactly once, ever
    "newborn",     # first seen within the last 4 tokens
    "phase",       # (layer - current layer) mod N, normalized
]


class ColdStats:
    """Per-key causal history, cheap enough to keep for every resident slot."""

    def __init__(self, n_keys, decay=0.97):
        self.hits = np.zeros(n_keys, dtype=np.int32)
        self.last = np.full(n_keys, -10_000, dtype=np.int64)
        self.first = np.full(n_keys, -1, dtype=np.int64)
        self.rate = np.zeros(n_keys, dtype=np.float32)
        self.rate_tok = np.zeros(n_keys, dtype=np.int64)
        self.decay = decay
        # Ring of the last 16 tokens' demand sets, for the recentN features.
        self.ring: list[set] = [set() for _ in range(16)]
        self.tok = 0

    def observe(self, key, tok):
        if self.first[key] < 0:
            self.first[key] = tok
        self.hits[key] += 1
        self.last[key] = tok
        seen = int(self.rate_tok[key])
        if tok > seen:
            self.rate[key] *= self.decay ** min(64, tok - seen)
        self.rate[key] += 1.0
        self.rate_tok[key] = tok
        self.ring[tok % 16].add(int(key))

    def roll(self, tok):
        if tok != self.tok:
            for t in range(self.tok + 1, tok + 1):
                self.ring[t % 16] = set()
            self.tok = tok

    def features(self, key, tok, cur_layer, key_layer, n_layers):
        age = tok - int(self.last[key])
        hits = int(self.hits[key])
        seen = int(self.rate_tok[key])
        rate = float(self.rate[key]) * (
            self.decay ** min(64, max(0, tok - seen))
        )
        life = max(1, tok - int(self.first[key]))
        r1 = 1.0 if key in self.ring[tok % 16] else 0.0
        r4 = sum(
            1.0 for d in range(4) if key in self.ring[(tok - d) % 16]
        )
        r16 = sum(
            1.0 for d in range(16) if key in self.ring[(tok - d) % 16]
        )
        return (
            np.log1p(max(0, age)),
            np.log1p(hits),
            rate,
            hits / life,
            r1,
            r4,
            r16,
            1.0 if hits <= 1 else 0.0,
            1.0 if (tok - int(self.first[key])) <= 4 else 0.0,
            ((int(key_layer[key]) - cur_layer) % n_layers) / n_layers,
        )


def study_cold(
    key_ids,
    layers_of,
    boundaries,
    slots,
    *,
    n_layers,
    key_layer,
    per_token,
    sample=16,
    cold_tokens=32.0,
    seed=0,
):
    """Collect (causal features -> "cold for >= cold_tokens") at eviction time.

    Sampled at the moments that matter: the decision points of a real LRU
    replay, over the entries LRU could have chosen. A model fit on uniformly
    sampled accesses would be answering a different question.
    """
    rng = np.random.default_rng(seed)
    resident: OrderedDict = OrderedDict()
    rlist: list[int] = []
    rpos: dict = {}
    bset = {int(b) for b in boundaries.tolist()}
    st = ColdStats(int(key_ids.max()) + 1)
    X, y = [], []

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
        st.roll(tok)
        if k in resident:
            resident.move_to_end(k)
        else:
            if len(resident) >= slots:
                m = len(rlist)
                for j in rng.integers(0, m, size=min(sample, m)):
                    cand = rlist[int(j)]
                    X.append(
                        st.features(cand, tok, cur_layer, key_layer, n_layers)
                    )
                    gap = _true_gap(cand, i, key_ids, per_token)
                    y.append(1.0 if gap >= cold_tokens else 0.0)
                victim = next(iter(resident))
                resident.pop(victim, None)
                _drop(victim)
            resident[k] = True
            _add(k)
        st.observe(k, tok)
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)


def fit_logistic(X, y, *, epochs=200, lr=0.3, l2=1e-5, seed=0):
    rng = np.random.default_rng(seed)
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-6
    Z = (X - mu) / sd
    Z = np.concatenate([Z, np.ones((len(Z), 1))], axis=1)
    w = np.zeros(Z.shape[1])
    idx = np.arange(len(Z))
    for _ in range(epochs):
        rng.shuffle(idx)
        for s in range(0, len(idx), 16384):
            b = idx[s : s + 16384]
            zb, yb = Z[b], y[b]
            p = 1.0 / (1.0 + np.exp(-zb @ w))
            w -= lr * (zb.T @ (p - yb) / len(b) + l2 * w)
    return w, mu, sd


def auc(scores, y):
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = y > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if not n_pos or not n_neg:
        return float("nan")
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


_FUTURE: dict = {}


def _index_future(key_ids):
    """positions[key] -> sorted array of access positions, for true gaps."""
    if _FUTURE:
        return _FUTURE
    pos = defaultdict(list)
    for i, k in enumerate(key_ids.tolist()):
        pos[k].append(i)
    for k, v in pos.items():
        _FUTURE[k] = np.asarray(v, dtype=np.int64)
    return _FUTURE


def _true_gap(key, i, key_ids, per_token):
    """Tokens until `key` is next accessed after position i (inf if never)."""
    pos = _index_future(key_ids).get(int(key))
    if pos is None:
        return float("inf")
    j = int(np.searchsorted(pos, i, side="right"))
    if j >= pos.size:
        return float("inf")
    return (int(pos[j]) - i) / max(1.0, per_token)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--slots", type=int, required=True)
    ap.add_argument("--min-traj", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--cold-tokens", type=float, default=32.0,
                    help="a candidate is 'cold' if unused for this many tokens")
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="only LRU and the gated detector, over (confidence bar, sample "
             "width). Detector precision at the TOP of the slate is what binds,"
             " and both knobs move it.",
    )
    args = ap.parse_args()

    trajectories, meta = load_bank(Path(args.bank).expanduser())
    trajectories = [t for t in trajectories if len(t) >= args.min_traj]
    n_layers = int(meta.get("n_layers") or 0) or 1 + max(
        (max(f) for traj in trajectories for f in traj if f), default=0
    )
    # Cap the trace: the instrumentation is O(evictions) with a python loop.
    kept, total = [], 0
    for traj in trajectories:
        if total >= args.max_tokens:
            break
        kept.append(traj)
        total += len(traj)
    trajectories = kept

    key_ids, layers_of, tokens, boundaries, n_keys = build_trace(
        trajectories, n_layers
    )
    nxt = next_use_positions(key_ids, n_keys)
    key_layer = np.zeros(n_keys, dtype=np.int32)
    key_layer[key_ids] = layers_of
    per_token = len(key_ids) / max(1, total)

    print(f"trace: {total} tokens, {len(key_ids)} accesses, {n_keys} distinct "
          f"experts, {per_token:.0f} accesses/token, {n_layers} layers")
    print(f"cache: {args.slots} slots "
          f"({args.slots / per_token:.2f} tokens of demand)\n")

    # Fit the cold detector before replaying, so the learned policy can be
    # scored alongside the bounds. Fit on the first 70% of eviction decisions.
    X, y = study_cold(
        key_ids,
        layers_of,
        boundaries,
        args.slots,
        n_layers=n_layers,
        key_layer=key_layer,
        per_token=per_token,
        cold_tokens=args.cold_tokens,
    )
    cut = int(len(X) * 0.7)
    cold_model = fit_logistic(X[:cut], y[:cut])

    if args.sweep:
        base = replay(
            key_ids, layers_of, boundaries, args.slots, "lru",
            next_use=nxt, n_layers=n_layers, key_layer=key_layer,
            per_token=per_token,
        )["misses"]
        print(f"{'bar':>5} {'scan':>5} {'misses':>9} {'vs lru':>8}")
        print(f"{'lru':>5} {'-':>5} {base:>9} {0.0:>+7.1f}%")
        for bar in (0.5, 0.7, 0.9):
            for scan in (32, 128):
                r = replay(
                    key_ids, layers_of, boundaries, args.slots,
                    "learned_gated", next_use=nxt, n_layers=n_layers,
                    key_layer=key_layer, per_token=per_token,
                    cold_model=cold_model, cold_bar=bar, phase_scan=scan,
                )
                print(f"{bar:>5.1f} {scan:>5} {r['misses']:>9} "
                      f"{(r['misses'] / base - 1) * 100:>+7.1f}%")
        return 0

    rows = []
    policies = (
        "lru", "lfru", "random", "phase", "wide_lfu",
        "learned", "learned_gated", "hawkes_aci", "learned_aci",
        "noisy30", "noisy50", "noisy70", "noisy90",
        "coarse", "belady",
    )
    for policy in policies:
        r = replay(
            key_ids,
            layers_of,
            boundaries,
            args.slots,
            policy,
            next_use=nxt,
            n_layers=n_layers,
            key_layer=key_layer,
            per_token=per_token,
            record=True,
            cold_model=(
                cold_model if policy.startswith("learned") else None
            ),
        )
        if "aci_fired" in r:
            print(f"  [{policy}] fired {r['aci_fired']} evictions at "
                  f"{r['aci_precision']:.3f} realized precision, "
                  f"final bar {r['aci_bar']}")
        rows.append(r)

    base = rows[0]["misses"]
    print(f"{'policy':<8} {'misses':>9} {'vs lru':>8} {'hit':>7} "
          f"{'victim runs soon':>17} {'victim just ran':>16} "
          f"{'median gap (tok)':>17}")
    for r in rows:
        print(f"{r['policy']:<8} {r['misses']:>9} "
              f"{(r['misses'] / base - 1) * 100:>+7.1f}% {r['hit_rate']:>7.3f} "
              f"{r.get('victim_runs_again_within_20pct_of_token', 0):>17.3f} "
              f"{r.get('victim_layer_just_ran', 0):>16.3f} "
              f"{r.get('median_next_use_accesses', 0):>17.2f}")
    print("\n'victim runs soon' = share of evictions whose layer re-executes "
          "within 20% of a token.\n'median gap' = tokens until the victim was "
          "actually needed again (higher is better).")

    # How good is the cold detector that the `learned` row above is using?
    # `coarse` shows that finding ANY entry cold for 32+ tokens is worth ~95%
    # of the oracle, so detection quality is what the whole idea rests on.
    w, mu, sd = cold_model
    Zte = np.concatenate(
        [(X[cut:] - mu) / sd, np.ones((len(X) - cut, 1))], axis=1
    )
    s = Zte @ w
    print(f"\ncold detector: label = next use >= {args.cold_tokens:.0f} tokens")
    print(f"  samples {len(X)}  base rate {y.mean():.3f}  "
          f"holdout AUC {auc(s, y[cut:]):.3f}")
    p = 1.0 / (1.0 + np.exp(-s))
    for th in (0.5, 0.7, 0.9):
        pick = p >= th
        if pick.sum():
            print(f"  at p>={th}: {pick.mean():.3f} of candidates, "
                  f"{y[cut:][pick].mean():.3f} of them truly cold")
    print("  weights (standardized, + = predicts cold):")
    for nm, val in zip(COLD_FEATURES, w[:-1]):
        print(f"    {nm:<9} {val:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
