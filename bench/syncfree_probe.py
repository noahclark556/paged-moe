# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
How much of the per-token cost is the *sync itself*?

A streamed decode blocks once per MoE layer (94x on Qwen3-235B) because the
expert->slot mapping lives in a Python dict: `_run_slab` cannot build its
`rhs_indices` until the router's choices have reached the CPU. Each of those
blocking evals costs a Metal commit + completion wait, and - more importantly  - 
it stops graph encode from running ahead of the GPU, so encode lands on the
critical path 94 times instead of once.

This probe measures what removing them would buy, with everything else held
fixed. Both modes run the real layers, the real slab, and the same
`gather_qmm` shapes; neither touches the disk (slot indices are synthetic, so
the weights read are whatever the slab happens to hold - timing is unaffected
and the output is discarded):

  sync    one blocking mx.eval per MoE layer, as production does today
  nosync  expert->slot resolved on the GPU from a resident table, so the whole
          token is one graph with one eval

The gap between them is the removable sync tax. `nosync` computes all K experts
per layer, which is the pessimistic case - the real engine computes fewer.

    python bench/syncfree_probe.py --model ~/mlx-models/qwen3-235-4bit --mode both
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm.models.cache import make_prompt_cache

from pagedmoe import load
from pagedmoe.streaming import StreamedSwitchGLU


def moe_layers(model) -> list[StreamedSwitchGLU]:
    found: list[StreamedSwitchGLU] = []

    def walk(mod):
        for child in mod.children().values():
            if isinstance(child, StreamedSwitchGLU):
                found.append(child)
            elif isinstance(child, (list, tuple)):
                for c in child:
                    if isinstance(c, StreamedSwitchGLU):
                        found.append(c)
                    elif hasattr(c, "children"):
                        walk(c)
            elif hasattr(child, "children"):
                walk(child)

    walk(model)
    return found


def patch(layers: list[StreamedSwitchGLU], mode: str) -> None:
    """Replace the MoE call with a disk-free variant in the requested mode."""
    for layer in layers:
        slab = layer.cache.slab
        if slab is None:
            raise SystemExit("slab is off; this probe measures the slab path")
        n_slots = slab.slots
        # Resident expert->slot table. The real thing would be maintained by the
        # cache as experts install and evict; here it is a fixed permutation, so
        # every routed expert resolves to *some* slot and the kernel shapes match
        # production exactly.
        # Sized past any real expert count so any routed id indexes it.
        table = mx.array((np.arange(4096) % n_slots).astype(np.uint32))
        layer._probe_table = table

        if mode == "sync":
            def call(self, x, indices, _slab=slab):
                lead = indices.shape
                K = lead[-1]
                x_flat = x.reshape(-1, x.shape[-1])
                ind2 = indices.reshape(-1, K)
                N = ind2.shape[0]
                # The sync production pays: router choices must reach the CPU
                # before a dict lookup can produce slot numbers.
                mx.eval(ind2)
                inds = np.asarray(ind2)
                slot_ids = (inds.reshape(-1) % _slab.slots).astype(np.uint32)
                rhs = mx.array(slot_ids)
                lhs = mx.array((np.arange(N * K) // K).astype(np.uint32))
                xe = x_flat[:, None, :]
                x_up = self._gather_proj(xe, "up_proj", rhs, lhs)
                x_gate = self._gather_proj(xe, "gate_proj", rhs, lhs)
                y = self._gather_proj(self.activation(x_up, x_gate), "down_proj", rhs)
                return y.reshape(*lead, -1)
        else:
            def call(self, x, indices):
                lead = indices.shape
                K = lead[-1]
                x_flat = x.reshape(-1, x.shape[-1])
                ind2 = indices.reshape(-1, K)
                N = ind2.shape[0]
                # No sync: the table is an mx.array, so the gather that turns
                # expert ids into slots is just another node in this token's
                # graph.
                rhs = self._probe_table[ind2].reshape(-1)
                lhs = (mx.arange(N * K) // K).astype(mx.uint32)
                xe = x_flat[:, None, :]
                x_up = self._gather_proj(xe, "up_proj", rhs, lhs)
                x_gate = self._gather_proj(xe, "gate_proj", rhs, lhs)
                y = self._gather_proj(self.activation(x_up, x_gate), "down_proj", rhs)
                return y.reshape(*lead, -1)

        layer.__class__ = type(
            f"Probe{mode.title()}SwitchGLU", (StreamedSwitchGLU,), {"__call__": call}
        )


def run(model, tokenizer, mode: str, layers, prompt_tokens: int, steps: int) -> float:
    patch(layers, mode)
    ids = list(range(10, 10 + prompt_tokens))
    cache = make_prompt_cache(model)
    logits = model(mx.array([ids]), cache=cache)
    mx.eval(logits)

    tok = mx.array([[11]])
    for _ in range(3):  # warm
        logits = model(tok, cache=cache)
        mx.eval(logits)

    t0 = time.perf_counter()
    for _ in range(steps):
        logits = model(tok, cache=cache)
        mx.eval(logits)
    dt = time.perf_counter() - t0
    return steps / dt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="both", choices=["sync", "nosync", "both"])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--cache-gb", type=float, default=None)
    args = ap.parse_args()

    out = {}
    for mode in (["sync", "nosync"] if args.mode == "both" else [args.mode]):
        # Fresh load per mode: patching rewrites the class.
        model, tokenizer = load(
            args.model, mode="streamed", cache_gb=args.cache_gb, verbose=False
        )
        layers = moe_layers(model)
        tps = run(model, tokenizer, mode, layers, args.prompt_tokens, args.steps)
        out[mode] = round(tps, 3)
        out[f"{mode}_ms"] = round(1000.0 / tps, 2)
        print(f"{mode:>7}: {tps:6.2f} tok/s  ({1000.0 / tps:6.2f} ms/token, "
              f"{len(layers)} MoE layers)")
        del model

    if "sync" in out and "nosync" in out:
        saved = out["sync_ms"] - out["nosync_ms"]
        print(f"\nremovable sync tax: {saved:.1f} ms/token "
              f"({saved / out['sync_ms'] * 100:.0f}% of the sync-mode token), "
              f"{out['sync']:.2f} -> {out['nosync']:.2f} tok/s "
              f"({out['nosync'] / out['sync']:.2f}x) with disk excluded")
    print("JSON:" + json.dumps(out))


if __name__ == "__main__":
    main()
