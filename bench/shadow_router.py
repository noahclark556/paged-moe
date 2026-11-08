#!/usr/bin/env python3
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Can a shadow pass with no expert reads predict the real routers?

Every learned prefetch head in this tree measures at or below zero, for a
reason that is now well established: the cache has already consumed the
learnable part of the demand signal, so a predictor is left fishing in the
residue. The exception is route prediction (+79% on a 64-expert MoE with
simulated read delay), and it is not a learned predictor at all. It runs the
layers' real routers, which are small resident matvecs, so it does not have
to beat recency; it only has to arrive early.

WHAT LIMITS IT. Route prediction feeds layer i's hidden state to layer j's
router. That is only a good approximation of layer j's input while j is close
to i, because every intervening layer edits the residual stream. Getting layer
j's true input means computing layers i+1..j-1, which needs their experts,
which is what we were trying to prefetch. Circular - so the lead time is
capped at a layer or two, and a 10 MB expert read needs more cover than that.

THE WAY OUT. An MoE layer is attention + norms + shared experts + a sum of
routed experts. Everything except the routed experts is resident backbone. So
run a forward pass that simply SKIPS the routed experts: zero disk reads, no
extra weights (it is a second activation pass over weights already in memory,
batch 1 at decode), and it produces an approximate hidden state at every layer.
Then ask each layer's real router what it wants, given that state. If the
answers agree with the true routing, we get exact router decisions for all N
layers with a full token of lead time, at the cost of one backbone pass - on a
model that spends 93% of a token stalled on disk.

The whole idea rests on one number, which this script measures: how much does
the routing change when the routed-expert contribution is dropped?

Two references are measured alongside, because "78% agreement" means nothing
without knowing what is already achieved for free:

  * prev-token routing at the same layer - what the prefetch ring's last_ids
    fallback covers today (measured at 0.46-0.49 on the demand banks)
  * layer i's hidden state into layer j's router at distance d - what route
    prediction does today, as a function of d, which shows the drift wall the
    shadow pass is meant to climb over

Usage:
    python bench/shadow_router.py --model ~/mlx-models/qwen3-next --tokens 32
"""

from __future__ import annotations

import argparse
import itertools
import sys
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
]


class Tap:
    """Records each MoE layer's top-k indices, and can blank the expert sum.

    Hooking SwitchGLU rather than the router means the recorded indices are
    exactly what the engine would fetch - the same seam paged-moe itself
    patches - so nothing here depends on guessing the router's arithmetic.
    """

    def __init__(self):
        self.ids: dict[int, list[int]] = {}
        self.blank = False
        self.keep = 0
        # When set, a set of (layer, expert) keys standing in for the cache.
        self.resident: set | None = None
        self.res_hit = 0.0
        self.res_tot = 0.0
        self._originals: list = []
        self._index: dict[int, int] = {}

    def attach(self, model) -> int:
        """Patch the CLASS, index by instance.

        `mod(x, idx)` resolves __call__ on the type, so assigning it on an
        instance is silently ignored - the measurement would have reported the
        real routing twice and "proved" perfect agreement.
        """
        from pagedmoe.streaming import find_switch_glus

        glus = find_switch_glus(model)
        if not glus:
            # The installed mlx auto-hook swaps SwitchGLU for
            # StreamedSwitchGLU at load, so on a machine with paged-moe
            # installed the engine's own class is what is in the tree.
            from pagedmoe.streaming import StreamedSwitchGLU

            glus = [
                (p, m)
                for p, m in model.named_modules()
                if isinstance(m, StreamedSwitchGLU)
            ]

            def _layer_num(path):
                for part in path.split("."):
                    if part.isdigit():
                        return int(part)
                return 0

            glus.sort(key=lambda kv: _layer_num(kv[0]))
        if not glus:
            return 0

        self._index = {id(m): i for i, (_p, m) in enumerate(glus)}
        cls = type(glus[0][1])
        tap = self
        original = cls.__call__

        def patched(self_mod, x, indices, *a, **kw):
            idx = tap._index.get(id(self_mod))
            if idx is not None:
                arr = np.asarray(indices)
                # Decode routes one position; a prefill chunk routes many. The
                # last position is the one continuing the sequence and the only
                # one the prefetch ring acts on, so record that one.
                k = int(arr.shape[-1])
                tap.ids[idx] = [int(e) for e in arr.reshape(-1)[-k:]]
            out = original(self_mod, x, indices, *a, **kw)
            if tap.blank and idx is not None:
                # Shadow pass: keep only the strongest `tap.keep` of the k
                # routed experts and drop the rest. keep=0 is the pure
                # "no expert reads at all" shadow; keep=k is the real pass.
                #
                # This is the right knob rather than an arbitrary ablation,
                # because mlx's top-k returns experts in descending router
                # weight and the high-weight experts are also the frequently
                # demanded ones - which is to say, the ones already resident.
                # So "keep the top m" approximates "compute the shadow pass
                # with whatever is warm in cache and skip only the misses",
                # which is a shadow pass that still reads nothing from disk.
                arr = np.asarray(indices)
                k = out.shape[-2]
                if tap.resident is not None:
                    # The real proposal: compute the shadow pass with whatever
                    # is already RESIDENT and skip only the misses. Still zero
                    # disk reads, but the retained experts are chosen by what
                    # the cache happens to hold rather than by router weight,
                    # which is a weaker selection than "top m" - residency
                    # correlates with weight but does not track it.
                    want = arr.reshape(-1)[-k:]
                    mask = np.array(
                        [
                            1.0 if (idx, int(e)) in tap.resident else 0.0
                            for e in want
                        ],
                        dtype=np.float32,
                    )
                    tap.res_hit += float(mask.sum())
                    tap.res_tot += float(mask.size)
                    return out * mx.array(mask)[:, None]
                if tap.keep <= 0:
                    return mx.zeros_like(out)
                if tap.keep >= k:
                    return out
                mask = np.zeros(k, dtype=np.float32)
                mask[: tap.keep] = 1.0
                return out * mx.array(mask)[:, None]
            return out

        cls.__call__ = patched
        self._originals.append((cls, original))
        return len(glus)

    def detach(self):
        for cls, original in self._originals:
            cls.__call__ = original
        self._originals = []


def overlap(a: list[int], b: list[int]) -> float:
    """Fraction of the TRUE set `a` that appears in the predicted set `b`.

    Recall, not symmetric agreement: a prefetcher is judged on how much of the
    demand it covered, and an over-large prediction is charged in bytes
    elsewhere, not here.
    """
    if not a:
        return float("nan")
    sb = set(b)
    return sum(1 for e in a if e in sb) / len(a)


def run(model_path: str, n_tokens: int, verbose: bool,
        keep_list: str, slots: int) -> int:
    from mlx_lm import load

    print(f"loading {model_path} ...", flush=True)
    model, tokenizer = load(model_path)

    tap = Tap()
    n_moe = tap.attach(model)
    print(f"tapped {n_moe} MoE layers")
    if n_moe == 0:
        print("no SwitchGLU layers found - not an MoE checkpoint?")
        return 1

    keeps = [int(v) for v in keep_list.split(",")]
    # recall[keep] -> list of (layer, recall) over all measured positions
    recall: dict[int, list[np.ndarray]] = {k: [] for k in keeps}
    prev_recall: list[np.ndarray] = []
    n_layers = n_moe
    # Stand-in for the expert cache. Small checkpoints are fully resident in
    # reality, which would make a resident-only shadow pass look perfect and
    # tell us nothing, so residency is simulated at a chosen slot budget.
    from collections import OrderedDict
    lru: OrderedDict = OrderedDict()

    for pi, prompt in enumerate(PROMPTS):
        from mlx_lm.models.cache import make_prompt_cache

        ids = tokenizer.encode(prompt)
        prev_true: dict[int, list[int]] | None = None
        cache_true = make_prompt_cache(model)
        # Prime the cache with the prompt, then step token by token so each
        # decode position is measured in the regime that matters.
        tap.blank = False
        tap.ids = {}
        logits = model(mx.array([ids]), cache=cache_true)
        tok = int(mx.argmax(logits[:, -1, :]).item())
        seq = list(ids)

        for step in range(n_tokens):
            # --- true pass: the routing the engine must actually serve
            tap.blank = False
            tap.ids = {}
            logits = model(mx.array([[tok]]), cache=cache_true)
            mx.eval(logits)
            true_ids = dict(tap.ids)
            seq.append(tok)

            # --- shadow passes at each `keep` level. A separate cache each
            # time, so the shadow's (different) K/V never corrupt the real
            # sequence. In the engine this would be a scratch KV lane for the
            # same reason - the shadow's attention output differs from the
            # real one the moment a layer's expert sum changes.
            for kp in keeps:
                tap.keep = kp
                # keep = -1 means "resident only", driven by the LRU below,
                # whose state is as of BEFORE this token - the shadow pass runs
                # first, so it cannot see this token's own installs.
                tap.resident = set(lru) if kp < 0 else None
                tap.blank = True
                tap.ids = {}
                out = model(mx.array([seq]), cache=make_prompt_cache(model))
                mx.eval(out)
                sh = dict(tap.ids)
                tap.blank = False
                tap.resident = None
                recall[kp].append(
                    np.array([
                        overlap(true_ids.get(l, []), sh.get(l, []))
                        for l in range(n_layers)
                    ])
                )

            # Fold this token's real demand into the LRU, oldest evicted first.
            for layer, ids_l in true_ids.items():
                for e in ids_l:
                    key = (layer, int(e))
                    lru.pop(key, None)
                    lru[key] = True
            while len(lru) > slots:
                lru.pop(next(iter(lru)))

            if prev_true is not None:
                prev_recall.append(
                    np.array([
                        overlap(true_ids.get(l, []), prev_true.get(l, []))
                        for l in range(n_layers)
                    ])
                )
            prev_true = true_ids
            tok = int(mx.argmax(logits[:, -1, :]).item())
            if verbose:
                shown = np.nanmean(recall[keeps[0]][-1])
                print(f"  prompt {pi} step {step}: keep={keeps[0]} "
                      f"recall {shown:.3f}", flush=True)

    tap.detach()

    print(f"\n{len(prev_recall)} decode positions, {n_layers} MoE layers")
    print(f"\n{'shadow pass':<34} {'recall of true routing':>22}")
    for kp in keeps:
        rows = np.vstack(recall[kp])
        label = ("resident only" if kp < 0
                 else f"keep top {kp} of k routed experts")
        print(f"  {label:<32} {np.nanmean(rows):>22.3f}")
    prev = np.vstack(prev_recall) if prev_recall else np.zeros((1, n_layers))
    print(f"{'previous token, same layer':<34} {np.nanmean(prev):>22.3f}")

    print("\nby depth:")
    edges = np.linspace(0, n_layers, 5).astype(int)
    header = "  layers      " + "".join(f"keep{k:<3}" for k in keeps) + "  prev"
    print(header)
    for lo, hi in itertools.pairwise(edges):
        if hi <= lo:
            continue
        cells = "".join(
            f"{np.nanmean(np.vstack(recall[k])[:, lo:hi]):<7.3f}" for k in keeps
        )
        print(f"  {lo:>3}-{hi - 1:<7} {cells} {np.nanmean(prev[:, lo:hi]):.3f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/mlx-models/qwen3-next")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--keep",
        default="0,2,4,6",
        help="how many of the k routed experts the shadow pass keeps; "
             "0 reads nothing, k is the real pass, -1 is resident-only",
    )
    ap.add_argument(
        "--slots",
        type=int,
        default=3000,
        help="simulated cache slots for the resident-only shadow (keep=-1)",
    )
    args = ap.parse_args()
    return run(
        str(Path(args.model).expanduser()),
        args.tokens,
        not args.quiet,
        args.keep,
        args.slots,
    )


if __name__ == "__main__":
    raise SystemExit(main())
