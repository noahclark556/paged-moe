# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""How far ahead does a real router stay right? The full (source, target) map.

Route prediction is the only mechanism in this engine that accelerates decode
(+79% with simulated read delay), and it works by running layer t's real router on layer s's hidden
state. The engine ships it at depth 2-3 on the belief that accuracy decays
past that, but the decay has never actually been mapped - and the answer
decides the next architecture:

  * if a long-lived predictable component exists, prefetch can be driven many
    layers ahead from one hidden state, which is enormous: lead time is the
    entire game on a saturated drive.
  * if it collapses fast, then the shipped ~+3 real-router predictor is
    provably the maximum causal lead available WITHOUT doing expert
    computation, and the only way further out is a pass that computes
    something (see bench/shadow_router.py).

Either answer closes a question, so this is measured before anything is built.

The metrics escalate from "is there statistical signal" to "would this save a
read", because the first three can look healthy while the last two are zero:

  spearman    rank correlation of the 512 router logits. Sees the whole
              distribution, not just the winners.
  topk        recall of the true top-k. The prefetch-relevant version.
  kl          KL(true || predicted) over the softmax. Penalises confident
              wrongness, which recall does not.
  inc         incremental recall BEYOND last-token replay: of the true demand
              the ring's free last_ids fallback did not already cover, how much
              does this predict? Anything the baseline covers is worth zero.
  miss-prec   precision among CACHE MISSES, against a simulated LRU. This is
              the one that decides viability, because a prefetch for a resident
              expert is a no-op and a prefetch for an expert that is never
              demanded is a wasted read on a saturated drive. The net-win bar
              is 0.5.
  miss-rec    recall of those same misses - the size of the prize at that
              distance.

Usage:
    python bench/router_matrix.py --model ~/mlx-models/qwen3-next --tokens 24
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROMPTS = [
    "Write a Python function that merges two sorted lists.",
    "Explain why a cyclic access pattern is the worst case for LRU caching.",
    "Summarize the tradeoffs between mmap and direct IO for large reads.",
    "def quicksort(a):\n    if len(a) <= 1:\n        return a\n",
    "What is the difference between a mutex and a semaphore?",
]


class Capture:
    """Records each MoE block's input hidden state and true routing.

    The block input is what the router actually sees, so capturing there means
    the cross-layer experiment needs no assumptions about which norm feeds the
    router - the same reason paged-moe patches this seam.
    """

    def __init__(self):
        self.blocks: list = []
        self.x: dict[int, np.ndarray] = {}
        self.ids: dict[int, list[int]] = {}
        self._orig = None
        self._cls = None
        self._index: dict[int, int] = {}

    def attach(self, model) -> int:
        blocks = [
            (p, m)
            for p, m in model.named_modules()
            if hasattr(m, "gate") and hasattr(m, "switch_mlp")
        ]

        def layer_num(path):
            for part in path.split("."):
                if part.isdigit():
                    return int(part)
            return 0

        blocks.sort(key=lambda kv: layer_num(kv[0]))
        if not blocks:
            return 0
        self.blocks = [m for _p, m in blocks]
        self._index = {id(m): i for i, m in enumerate(self.blocks)}
        cls = type(self.blocks[0])
        cap = self
        original = cls.__call__

        def patched(self_mod, x, *a, **kw):
            idx = cap._index.get(id(self_mod))
            if idx is not None:
                arr = np.asarray(x.astype(mx.float32)).reshape(-1, x.shape[-1])
                cap.x[idx] = arr[-1]           # the continuing position
            return original(self_mod, x, *a, **kw)

        cls.__call__ = patched
        self._cls, self._orig = cls, original

        # Top-k ids come from the same SwitchGLU seam the engine fetches on.
        from pagedmoe.streaming import StreamedSwitchGLU

        glus = [
            (p, m)
            for p, m in model.named_modules()
            if isinstance(m, StreamedSwitchGLU)
        ]
        glus.sort(key=lambda kv: layer_num(kv[0]))
        self._gidx = {id(m): i for i, (_p, m) in enumerate(glus)}
        gcls = type(glus[0][1])
        gorig = gcls.__call__

        def gpatched(self_mod, xx, indices, *a, **kw):
            gi = cap._gidx.get(id(self_mod))
            if gi is not None:
                ind = np.asarray(indices)
                k = int(ind.shape[-1])
                cap.ids[gi] = [int(e) for e in ind.reshape(-1)[-k:]]
            return gorig(self_mod, xx, indices, *a, **kw)

        gcls.__call__ = gpatched
        self._gcls, self._gorig = gcls, gorig
        return len(self.blocks)

    def detach(self):
        if self._cls is not None:
            self._cls.__call__ = self._orig
        if getattr(self, "_gcls", None) is not None:
            self._gcls.__call__ = self._gorig

    def logits(self, target: int, x: np.ndarray) -> np.ndarray:
        """Layer `target`'s real router, evaluated on the given hidden state."""
        gate = self.blocks[target].gate
        out = gate(mx.array(x[None, :].astype(np.float32)))
        return np.asarray(out.astype(mx.float32)).reshape(-1)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d > 0 else 0.0


def kl(true_logits: np.ndarray, pred_logits: np.ndarray) -> float:
    def soft(z):
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    p, q = soft(true_logits), soft(pred_logits)
    return float(np.sum(p * np.log((p + 1e-12) / (q + 1e-12))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/mlx-models/qwen3-next")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--slots", type=int, default=3000,
                    help="simulated cache slots for the miss-precision columns")
    ap.add_argument("--dists", default="1,2,4,8,16,32")
    args = ap.parse_args()

    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    path = str(Path(args.model).expanduser())
    print(f"loading {path} ...", flush=True)
    model, tokenizer = load(path)
    cap = Capture()
    n = cap.attach(model)
    print(f"tapped {n} MoE layers")
    if n == 0:
        return 1

    dists = [int(d) for d in args.dists.split(",")]
    sources = [s for s in (0, 4, 8, 16, 24, 32, 40) if s < n]

    # Per-position records, so all pair math happens once at the end.
    recs: list[dict] = []
    lru: OrderedDict = OrderedDict()
    resident_snapshots: list[set] = []

    for prompt in PROMPTS:
        ids = tokenizer.encode(prompt)
        cache = make_prompt_cache(model)
        cap.x, cap.ids = {}, {}
        logits = model(mx.array([ids]), cache=cache)
        tok = int(mx.argmax(logits[:, -1, :]).item())
        prev: dict[int, list[int]] | None = None

        for _step in range(args.tokens):
            cap.x, cap.ids = {}, {}
            logits = model(mx.array([[tok]]), cache=cache)
            mx.eval(logits)
            if len(cap.x) == n and len(cap.ids) == n:
                recs.append({
                    "x": dict(cap.x),
                    "ids": dict(cap.ids),
                    "prev": prev,
                })
                resident_snapshots.append(set(lru))
                prev = dict(cap.ids)
                for layer, ee in cap.ids.items():
                    for e in ee:
                        key = (layer, int(e))
                        lru.pop(key, None)
                        lru[key] = True
                while len(lru) > args.slots:
                    lru.pop(next(iter(lru)))
            tok = int(mx.argmax(logits[:, -1, :]).item())

    cap_blocks = cap.blocks
    print(f"{len(recs)} decode positions\n")

    # Per-channel gain ratio between layers, estimated from the run itself:
    # both hidden states are norm outputs differing mostly by per-channel
    # gains, and a linear router's top-k is invariant to positive scaling, so
    # removing the gain mismatch is the only systematic correction available
    # without reaching into a specific architecture's norm weights.
    scale = np.zeros((n, recs[0]["x"][0].shape[0]), dtype=np.float64)
    for r in recs:
        for layer in range(n):
            scale[layer] += np.abs(r["x"][layer])
    scale = np.maximum(scale / max(1, len(recs)), 1e-6)

    print(f"{'src':>4} {'tgt':>4} {'d':>3} {'spearman':>9} {'topk':>6} "
          f"{'kl':>6} {'inc':>6} {'miss-prec':>10} {'miss-rec':>9}")
    for s in sources:
        for d in dists:
            t = s + d
            if t >= n:
                continue
            sp, tk, kls, inc_hit, inc_tot = [], [], [], 0, 0
            mp_hit = mp_tot = mr_hit = mr_tot = 0
            for r, res in zip(recs, resident_snapshots):
                truth = r["ids"][t]
                k = len(truth)
                xs = r["x"][s] * (scale[t] / scale[s])
                pred_logits = cap.logits(t, xs)
                true_logits = cap.logits(t, r["x"][t])
                sp.append(spearman(true_logits, pred_logits))
                kls.append(kl(true_logits, pred_logits))
                pred = set(np.argsort(-pred_logits)[:k].tolist())
                tset = set(truth)
                tk.append(len(pred & tset) / max(1, k))
                # incremental over the free last-token replay
                pv = set((r["prev"] or {}).get(t, ()))
                residual = tset - pv
                if residual:
                    inc_tot += len(residual)
                    inc_hit += len(residual & pred)
                # among cache misses: a prefetch for a resident expert is a
                # no-op, so both sides drop residency before being scored.
                miss = {e for e in tset if (t, e) not in res}
                issue = {e for e in pred if (t, e) not in res}
                mp_tot += len(issue)
                mp_hit += len(issue & miss)
                mr_tot += len(miss)
                mr_hit += len(miss & issue)
            print(f"{s:>4} {t:>4} {d:>3} {np.mean(sp):>9.3f} "
                  f"{np.mean(tk):>6.3f} {np.mean(kls):>6.2f} "
                  f"{(inc_hit / inc_tot if inc_tot else float('nan')):>6.3f} "
                  f"{(mp_hit / mp_tot if mp_tot else float('nan')):>10.3f} "
                  f"{(mr_hit / mr_tot if mr_tot else float('nan')):>9.3f}")

    cap.detach()
    del cap_blocks
    print("\nnet-win bar for miss-prec is 0.50: a correct "
          "in-token prefetch is byte-neutral, only waste adds bytes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
