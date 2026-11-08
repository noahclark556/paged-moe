#!/usr/bin/env python3
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Can a calibrated per-candidate model clear the net-win bar at depth?

bench/intoken_signal.py established the opening: the previous-token heuristic
that covers the far layers of a token only recalls ~0.46-0.49, and this token's
early-layer demand carries real, distance-flat information about its late-layer
demand. What it did NOT establish is that the signal is worth reading bytes
for - top-K co-occurrence reached 0.69 union recall but spent 1.66x the reads
to get there, and on a saturated drive that is how wrap lost.

THE BAR. For in-token prefetch of a layer that will run later in the SAME
token, the accounting is unusually clean. Let D be the layer's demand, C what
the baseline already covers, S what the head issues (disjoint from C), H = S
hits inside D\\C, W = S \\ D the waste. Total bytes are:

    baseline:  |D\\C|
    with head: |S| + (|D\\C| - |H|)  =  |D\\C| + |W|

so a correct prefetch is exactly byte-NEUTRAL - that expert was going to be
read either way - and it converts a blocking read into one with layers of lead
time. Only waste adds bytes. On a saturated drive an extra read costs about its
transfer time and an early read saves about its transfer time, so:

    net win  <=>  |H| > |W|  <=>  incremental precision > 0.5

That is the number this script measures, and it is a hard bar, not a tuned one.
Top-K co-occurrence sits at 0.26 and is therefore disqualified.

WHY A PER-CANDIDATE MODEL. The shipped heads learn one big linear map from a
hashed feature vector to every (layer, expert) output - 12-14k outputs here. It
has to see a given layer-expert pair many times before that column means
anything, which is exactly the "slowly climbing, never arriving" behaviour in
the logs. Scoring *candidates* instead shares every parameter across all layers
and all experts: 8 features, 9 weights, and evidence from one layer teaches the
model about every other layer. It also produces a probability rather than a
rank, which is what a byte budget and a net-win threshold need.

Usage:
    python bench/intoken_learn.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pagedmoe.sidecar.bank import load_bank  # noqa: E402

# Feature names, in vector order. Kept as a list so the fitted weights can be
# printed against them - a per-candidate model is small enough that the weights
# are readable, and a sign that disagrees with intuition is usually a bug.
FEATURES = [
    "prev_same",      # demanded at this layer by the previous token
    "prev_near",      # demanded at a neighbouring layer by the previous token
    "cooc",           # co-occurrence with this token's prefix, normalized
    "cooc_top",       # is the argmax of cooc for this layer
    "freq",           # long-run demand rate at this layer
    "recency",        # exp(-age/16) since last demanded at this layer
    "streak",         # demanded by both of the last two tokens
    "depth",          # normalized target layer index
]


class Stats:
    """Causal per-layer demand statistics, updated token by token.

    Everything here is available in the engine at the moment of prediction:
    counts of what has been demanded, when it was last demanded, and a
    co-occurrence table between an early "anchor" layer and everything else.
    Nothing reads the future, and nothing reads the current token's target
    layer - the whole point is that the target has not run yet.
    """

    def __init__(self, n_layers: int, n_experts: int, anchor: int = 4):
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.anchor = anchor
        self.count = np.zeros((n_layers, n_experts), dtype=np.float32)
        self.last = np.full((n_layers, n_experts), -10_000, dtype=np.int32)
        self.tokens = 0
        # co-occurrence: anchor expert -> (layer, expert) demand counts
        self.cooc = np.zeros((n_experts, n_layers, n_experts), dtype=np.float32)
        self.anchor_count = np.zeros(n_experts, dtype=np.float32)
        self.prev: dict[int, list[int]] = {}
        self.prev2: dict[int, list[int]] = {}

    def observe(self, frame: dict[int, list[int]]) -> None:
        anchor_ids = frame.get(self.anchor) or []
        for layer, ids in frame.items():
            if layer >= self.n_layers:
                continue
            arr = np.asarray(ids, dtype=np.int64)
            arr = arr[(arr >= 0) & (arr < self.n_experts)]
            if arr.size == 0:
                continue
            self.count[layer, arr] += 1.0
            self.last[layer, arr] = self.tokens
            for a in anchor_ids:
                if 0 <= a < self.n_experts:
                    self.cooc[a, layer, arr] += 1.0
        for a in anchor_ids:
            if 0 <= a < self.n_experts:
                self.anchor_count[a] += 1.0
        self.prev2 = self.prev
        self.prev = {int(k): list(v) for k, v in frame.items()}
        self.tokens += 1

    def build(self, target: int, anchor_ids: list[int]) -> np.ndarray:
        """Feature matrix [n_experts, len(FEATURES)] for one target layer."""
        n = self.n_experts
        f = np.zeros((n, len(FEATURES)), dtype=np.float32)

        prev_same = np.zeros(n, dtype=np.float32)
        for e in self.prev.get(target, ()):
            if 0 <= e < n:
                prev_same[e] = 1.0
        near = np.zeros(n, dtype=np.float32)
        for d in (-2, -1, 1, 2):
            for e in self.prev.get(target + d, ()):
                if 0 <= e < n:
                    near[e] = 1.0
        streak = np.zeros(n, dtype=np.float32)
        p2 = set(self.prev2.get(target, ()))
        for e in self.prev.get(target, ()):
            if e in p2 and 0 <= e < n:
                streak[e] = 1.0

        cooc = np.zeros(n, dtype=np.float32)
        for a in anchor_ids:
            if 0 <= a < n and self.anchor_count[a] > 0:
                cooc += self.cooc[a, target] / self.anchor_count[a]
        if anchor_ids:
            cooc /= len(anchor_ids)
        cooc_top = np.zeros(n, dtype=np.float32)
        if cooc.max() > 0:
            cooc_top[int(cooc.argmax())] = 1.0

        freq = self.count[target] / max(1.0, float(self.tokens))
        age = np.clip(self.tokens - self.last[target], 0, 10_000).astype(np.float32)

        f[:, 0] = prev_same
        f[:, 1] = near
        f[:, 2] = cooc
        f[:, 3] = cooc_top
        f[:, 4] = freq
        f[:, 5] = np.exp(-age / 16.0)
        f[:, 6] = streak
        f[:, 7] = target / max(1, self.n_layers - 1)
        return f


class LRU:
    """The cache the head is actually competing with.

    This is the correction that decides the whole question. The first pass of
    this study labelled a candidate positive if the target layer would demand
    it, and found a thin net win. But prefetching an expert that is already
    resident is a no-op - cache.prefetch drops resident ids - so those rows
    were being credited for work that costs nothing and saves nothing. The
    label that matters is "will be demanded AND will miss", and since the
    model's strongest feature is recency, while the cache's eviction order is
    *also* recency, the two are competing for the same signal.
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.order: dict[tuple[int, int], int] = {}
        self.t = 0

    def resident(self, layer: int, expert: int) -> bool:
        return (layer, expert) in self.order

    def resident_mask(self, layer: int, n_experts: int) -> np.ndarray:
        m = np.zeros(n_experts, dtype=bool)
        for (l, e) in self.order:
            if l == layer and e < n_experts:
                m[e] = True
        return m

    def touch(self, frame: dict[int, list[int]]) -> None:
        for layer, ids in frame.items():
            for e in ids:
                self.t += 1
                key = (int(layer), int(e))
                self.order.pop(key, None)
                self.order[key] = self.t
        while len(self.order) > self.capacity:
            self.order.pop(next(iter(self.order)))


def collect(
    trajectories,
    n_layers,
    n_experts,
    targets,
    *,
    anchor=4,
    capacity: int = 0,
):
    """Causal (features, label, baseline_covered) rows over the bank.

    `targets` are the layers the head would be responsible for: far enough
    ahead of the anchor that route prediction cannot reach them.

    With `capacity`, an LRU cache of that many experts runs alongside and the
    label becomes "demanded by the target layer and not resident" - i.e. a read
    the head would actually save. Resident experts are also removed from the
    eligible set, since issuing for them is free and pointless.
    """
    X, y, cov = [], [], []
    for traj in trajectories:
        st = Stats(n_layers, n_experts, anchor=anchor)
        lru = LRU(capacity) if capacity else None
        for frame in traj:
            anchor_ids = frame.get(anchor) or []
            if anchor_ids and st.tokens > 0:
                for target in targets:
                    truth = frame.get(target)
                    if not truth:
                        continue
                    f = st.build(target, anchor_ids)
                    lab = np.zeros(n_experts, dtype=np.float32)
                    t = np.asarray(truth, dtype=np.int64)
                    t = t[(t >= 0) & (t < n_experts)]
                    lab[t] = 1.0
                    base = np.zeros(n_experts, dtype=np.float32)
                    for e in st.prev.get(target, ()):
                        if 0 <= e < n_experts:
                            base[e] = 1.0
                    if lru is not None:
                        res = lru.resident_mask(target, n_experts)
                        # Resident: not a saveable read, and not worth issuing.
                        lab[res] = 0.0
                        base[res] = 1.0
                    X.append(f)
                    y.append(lab)
                    cov.append(base)
            st.observe(frame)
            if lru is not None:
                lru.touch(frame)
    if not X:
        return None
    return (
        np.concatenate(X).reshape(-1, len(FEATURES)),
        np.concatenate(y),
        np.concatenate(cov),
    )


def fit(X, y, *, epochs=60, lr=0.5, l2=1e-4, seed=0):
    """Plain logistic regression with a bias. Small enough to be exact-ish."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = np.zeros(d + 1, dtype=np.float64)
    Xb = np.concatenate([X, np.ones((n, 1), dtype=np.float32)], axis=1).astype(
        np.float64
    )
    # Class imbalance: ~k of n_experts are positive, so weight positives up to
    # keep the fit from collapsing to "never". Calibration is restored after
    # the fit by Platt-scaling on the training set.
    pos = y > 0.5
    wpos = float((~pos).sum()) / max(1.0, float(pos.sum()))
    sw = np.where(pos, wpos, 1.0)
    idx = np.arange(n)
    for _ in range(epochs):
        rng.shuffle(idx)
        for start in range(0, n, 8192):
            b = idx[start : start + 8192]
            xb, yb, sb = Xb[b], y[b], sw[b]
            p = 1.0 / (1.0 + np.exp(-xb @ w))
            g = xb.T @ (sb * (p - yb)) / len(b) + l2 * w
            w -= lr * g
    return w


def platt(scores, y):
    """Fit a 2-parameter calibration so the threshold means a probability.

    Without this, the positive class weighting above leaves the scores
    monotone but badly scaled, and a 0.5 net-win threshold would be a
    meaningless place to cut.
    """
    a, b = 1.0, 0.0
    for _ in range(200):
        z = a * scores + b
        p = 1.0 / (1.0 + np.exp(-z))
        r = p - y
        ga = float(np.mean(r * scores))
        gb = float(np.mean(r))
        a -= 2.0 * ga
        b -= 2.0 * gb
    return a, b


def evaluate(w, cal, X, y, cov, *, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)):
    Xb = np.concatenate([X, np.ones((len(X), 1), dtype=np.float32)], axis=1)
    z = Xb @ w
    a, b = cal
    p = 1.0 / (1.0 + np.exp(-(a * z + b)))
    covered = cov > 0.5
    truth = y > 0.5
    # The head only ever proposes what the baseline is not already fetching.
    eligible = ~covered
    uncovered_demand = int((truth & eligible).sum())
    out = {
        "baseline_recall": float((truth & covered).sum() / max(1, truth.sum())),
        "uncovered": uncovered_demand,
        "rows": len(X),
        "points": {},
    }
    for th in thresholds:
        issue = eligible & (p >= th)
        hits = int((issue & truth).sum())
        waste = int((issue & ~truth).sum())
        n_issue = hits + waste
        out["points"][th] = {
            "issued": n_issue,
            "precision": hits / max(1, n_issue),
            "recovered": hits / max(1, uncovered_demand),
            "net": hits - waste,
        }
    return out


def run_slot(name, trajectories, meta, *, capacity=0, stride=7):
    n_layers = int(meta.get("n_layers") or 0) or 1 + max(
        (max(f) for traj in trajectories for f in traj if f), default=0
    )
    n_experts = int(meta.get("n_experts") or 0) or 1 + max(
        (max(v) for traj in trajectories for f in traj for v in f.values() if v),
        default=0,
    )
    anchor = 4
    # Targets: layers route prediction cannot reach (it runs the real router on
    # a hidden state that is only valid for a layer or two) but which still run
    # later in this same token, so a hit has the whole intervening walk as lead.
    targets = [t for t in range(anchor + 6, n_layers, stride)]
    cut = max(1, int(len(trajectories) * 0.7))
    train_t, test_t = trajectories[:cut], trajectories[cut:] or trajectories[-1:]

    tr = collect(
        train_t, n_layers, n_experts, targets, anchor=anchor, capacity=capacity
    )
    te = collect(
        test_t, n_layers, n_experts, targets, anchor=anchor, capacity=capacity
    )
    if tr is None or te is None:
        print(f"=== {name}: not enough usable frames")
        return
    Xtr, ytr, _ = tr
    Xte, yte, cte = te
    w = fit(Xtr, ytr)
    Xb = np.concatenate([Xtr, np.ones((len(Xtr), 1), dtype=np.float32)], axis=1)
    cal = platt(Xb @ w, ytr)
    res = evaluate(w, cal, Xte, yte, cte)

    label = "demanded & missing" if capacity else "demanded"
    print(f"\n=== {name}  ({n_layers}L x {n_experts}E, {len(targets)} target layers)")
    if capacity:
        print(f"  LRU cache of {capacity} experts running alongside; "
              f"label = {label}")
    print(f"  train rows {len(Xtr)}  test rows {len(Xte)}")
    print(f"  baseline coverage on target layers: {res['baseline_recall']:.3f}")
    print(f"  uncovered {label} slots in test: {res['uncovered']}")
    print("  thr  issued   precision  recovered  net(H-W)")
    for th, pt in res["points"].items():
        flag = "  <-- net win" if pt["net"] > 0 else ""
        print(f"  {th:.1f} {pt['issued']:>8}     {pt['precision']:.3f}      "
              f"{pt['recovered']:.3f}    {pt['net']:>+7}{flag}")
    print("  weights:")
    for nm, val in zip(FEATURES, w[:-1]):
        print(f"    {nm:<10} {val:+.3f}")
    print(f"    {'bias':<10} {w[-1]:+.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--root", default="~/.global_agent/expert_sidecar")
    ap.add_argument("--min-traj", type=int, default=16)
    ap.add_argument(
        "--cache-experts",
        type=int,
        default=0,
        help="run an LRU cache of this size alongside and score only the reads "
             "the head would actually save (0 = demand-only labels)",
    )
    ap.add_argument("--stride", type=int, default=7,
                    help="sample every Nth layer as a target")
    args = ap.parse_args()
    root = Path(args.root).expanduser()
    banks = (
        [root / f"{args.slot}_wrap_bank.npz"]
        if args.slot
        else sorted(root.glob("*_wrap_bank.npz"))
    )
    for path in banks:
        trajectories, meta = load_bank(path)
        if len(trajectories) < args.min_traj:
            continue
        run_slot(
            path.name.replace("_wrap_bank.npz", ""),
            trajectories,
            meta,
            capacity=args.cache_experts,
            stride=args.stride,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
