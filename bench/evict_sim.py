# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Replay a decode demand trace (decode_profile.py --trace) through eviction
policies at a fixed slot count. Demand misses only, no prefetch: what each
policy would leave for the drive.

  python bench/evict_sim.py /tmp/esb/trace235_code.json --slots 2043
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict, defaultdict


def load(path):
    with open(path) as fh:
        tr = json.load(fh)
    layers = {}
    seq = []
    for lk, ids in tr:
        li = layers.setdefault(lk, len(layers))
        seq.append((li, ids))
    return seq, len(layers)


def run_lru(seq, cap, warm, scan=0, fmax=8):
    lru: OrderedDict = OrderedDict()
    freq: dict = {}
    miss = 0
    for i, (li, ids) in enumerate(seq):
        for e in ids:
            k = (li, e)
            if k in lru:
                lru.move_to_end(k)
                freq[k] = min(fmax, freq.get(k, 0) + 1)
                continue
            if i >= warm:
                miss += 1
            lru[k] = None
            while len(lru) > cap:
                for _ in range(scan):
                    t = next(iter(lru))
                    f = freq.get(t, 0)
                    if f <= 0:
                        break
                    freq[t] = f >> 1
                    lru.move_to_end(t)
                v = next(iter(lru))
                if v == k:
                    lru.move_to_end(v)
                    v = next(iter(lru))
                del lru[v]
                freq.pop(v, None)
    return miss


def run_belady(seq, cap, warm):
    import heapq
    nxt_use = [None] * sum(len(ids) for _, ids in seq)
    last = {}
    flat = []
    for i, (li, ids) in enumerate(seq):
        for e in ids:
            flat.append(((li, e), i))
    INF = 1 << 60
    for j in range(len(flat) - 1, -1, -1):
        k, i = flat[j]
        nxt_use[j] = last.get(k, INF)
        last[k] = flat[j][1]
    res = {}
    heap = []
    miss = 0
    for j, (k, i) in enumerate(flat):
        nu = nxt_use[j]
        if k in res:
            res[k] = nu
            heapq.heappush(heap, (-nu, k))
            continue
        if i >= warm:
            miss += 1
        res[k] = nu
        heapq.heappush(heap, (-nu, k))
        while len(res) > cap:
            negnu, v = heapq.heappop(heap)
            if v in res and res[v] == -negnu:
                del res[v]
    return miss


def run_s3fifo(seq, cap, warm, small_frac=0.1):
    from collections import deque
    small: deque = deque()
    main: deque = deque()
    ghost: OrderedDict = OrderedDict()
    where: dict = {}
    freq: dict = {}
    s_cap = max(1, int(cap * small_frac))
    m_cap = cap - s_cap
    miss = 0

    def evict_main():
        while main:
            v = main.popleft()
            if freq.get(v, 0) > 0:
                freq[v] -= 1
                main.append(v)
            else:
                del where[v]
                freq.pop(v, None)
                return

    def evict_small():
        while small:
            v = small.popleft()
            if freq.get(v, 0) > 0:
                freq[v] = 0
                if len(main) >= m_cap:
                    evict_main()
                main.append(v)
                where[v] = "m"
                if len(small) < s_cap:
                    return
            else:
                del where[v]
                freq.pop(v, None)
                ghost[v] = None
                if len(ghost) > m_cap:
                    ghost.popitem(last=False)
                return

    for i, (li, ids) in enumerate(seq):
        for e in ids:
            k = (li, e)
            if k in where:
                freq[k] = min(3, freq.get(k, 0) + 1)
                continue
            if i >= warm:
                miss += 1
            if k in ghost:
                del ghost[k]
                if len(main) >= m_cap:
                    evict_main()
                main.append(k)
                where[k] = "m"
            else:
                while len(small) >= s_cap:
                    evict_small()
                small.append(k)
                where[k] = "s"
            freq[k] = 0
    return miss


def run_partition(seq, cap, n_layers, warm, period=94 * 8):
    """Per-layer LRU partitions sized by marginal utility (UCP-style).

    Each layer keeps a shadow LRU stack of its recent distinct keys and counts
    hits per stack depth. Every `period` accesses, slots are handed out
    greedily by the next slot's hit count across layers.
    """
    stacks = [OrderedDict() for _ in range(n_layers)]
    depth_hits = [defaultdict(float) for _ in range(n_layers)]
    quota = [cap // n_layers] * n_layers
    res = [OrderedDict() for _ in range(n_layers)]
    miss = 0
    acc = 0
    shadow = cap // n_layers * 4
    for i, (li, ids) in enumerate(seq):
        st = stacks[li]
        for e in ids:
            if e in st:
                d = list(reversed(st)).index(e)
                depth_hits[li][d] += 1
                st.move_to_end(e)
            else:
                st[e] = None
                if len(st) > shadow:
                    st.popitem(last=False)
            r = res[li]
            if e in r:
                r.move_to_end(e)
            else:
                if i >= warm:
                    miss += 1
                r[e] = None
                while len(r) > quota[li]:
                    r.popitem(last=False)
            acc += 1
        if acc >= period:
            acc = 0
            import heapq
            q = [0] * n_layers
            h = [(-depth_hits[l].get(0, 0.0), 0, l) for l in range(n_layers)]
            heapq.heapify(h)
            for _ in range(cap):
                _, _, l = heapq.heappop(h)
                q[l] += 1
                heapq.heappush(h, (-depth_hits[l].get(q[l], 0.0), q[l], l))
            quota = q
            for l in range(n_layers):
                for d in list(depth_hits[l]):
                    depth_hits[l][d] *= 0.5
                while len(res[l]) > quota[l]:
                    res[l].popitem(last=False)
    return miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--slots", type=int, default=2043)
    ap.add_argument("--warm-tokens", type=int, default=32)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    seq, nl = load(args.trace)
    warm = args.warm_tokens * nl
    total = sum(len(ids) for i, (_, ids) in enumerate(seq) if i >= warm)
    toks = (len(seq) - warm) / nl
    cap = args.slots
    pol = {
        "lru": lambda: run_lru(seq, cap, warm),
        "lfru": lambda: run_lru(seq, cap, warm, scan=8),
        "s3fifo": lambda: run_s3fifo(seq, cap, warm),
        "partition": lambda: run_partition(seq, cap, nl, warm),
        "belady": lambda: run_belady(seq, cap, warm),
    }
    only = [p for p in args.only.split(",") if p]
    base = None
    for name, fn in pol.items():
        if only and name not in only:
            continue
        m = fn()
        base = base or m
        print(f"{name:10s} misses/tok {m / toks:7.1f}  hit {1 - m / total:.3f}  vs first {m / base - 1:+.1%}")


if __name__ == "__main__":
    main()
