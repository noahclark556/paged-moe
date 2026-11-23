# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Differentiable streamed experts.

`StreamedSwitchGLU` is built for decode: slab slots, flow, prune, prefetch
learners. None of that belongs in a backward pass, and slab slots are
overwritten in place, which a graph that is evaluated later cannot survive.

`TrainSwitch` takes its place while training. Per call it:

  1. syncs the router ids (MLX allows this inside grad / vjp / checkpoint),
  2. fetches the unique experts through the plain LRU (ordinary arrays, so a
     pending backward keeps them alive even if the cache evicts them),
  3. stacks them and runs the same gather_qmm calls as mlx-lm's SwitchGLU.

Step 3 is the resident math on the same weights, so a streamed layer and a
resident layer give the same output and the same gradient. The only thing
streaming changes is where the bytes come from.

Reads ahead are timing only. `expect()` starts the reads for a set of ids
before the layer runs (the backward pass knows them exactly, the forward pass
guesses from the previous step); the layer still computes with the ids its
router produced, and anything it did not expect is fetched on the spot.
"""

from __future__ import annotations

import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import fastgrad

_PROJS = ("gate_proj", "up_proj", "down_proj")


class Counters:
    """Per-run expert traffic, reset by the trainer each step."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.calls = 0
        self.experts = 0
        self.expected = 0
        self.expected_used = 0
        self.fetch_s = 0.0

    def snapshot(self) -> dict:
        return {
            "calls": self.calls,
            "experts": self.experts,
            "ahead_hit": (
                round(self.expected_used / self.expected, 3) if self.expected else None
            ),
            "fetch_s": round(self.fetch_s, 3),
        }


COUNTERS = Counters()


class TrainSwitch(nn.Module):
    def __init__(self, streamed):
        super().__init__()
        # Plain attribute (not a child module): the streamed layer carries
        # decode state that must not show up in parameter trees.
        object.__setattr__(self, "_src", streamed)
        object.__setattr__(self, "_ahead", None)
        object.__setattr__(self, "last_ids", np.zeros((0,), np.int64))
        self.layer_key = streamed.layer_key
        self.activation = streamed.activation

    @property
    def cache(self):
        return self._src.cache

    @property
    def specs(self):
        return self._src.specs

    # ------------------------------------------------------------ reads ahead

    def expect(self, ids) -> None:
        """Start reading `ids` now; the next call consumes them."""
        self._drain_ahead()
        ids = sorted({int(e) for e in ids})
        if not ids:
            return
        begun = self.cache._begin(self.layer_key, ids, use_slab=False)
        object.__setattr__(self, "_ahead", (set(ids), begun))
        COUNTERS.expected += len(ids)

    def _drain_ahead(self) -> dict:
        ahead = self._ahead
        if ahead is None:
            return {}
        object.__setattr__(self, "_ahead", None)
        return self.cache._finish(self.layer_key, ahead[1], "lru")

    # ---------------------------------------------------------------- fetch

    def _fetch(self, unique: np.ndarray) -> dict:
        t0 = time.perf_counter()
        want = [int(e) for e in unique]
        ahead = self._ahead
        got: dict = {}
        if ahead is not None:
            COUNTERS.expected_used += len(ahead[0].intersection(want))
            got = self._drain_ahead()
        rest = [e for e in want if e not in got]
        if rest:
            begun = self.cache._begin(self.layer_key, rest, use_slab=False)
            got.update(self.cache._finish(self.layer_key, begun, "lru"))
        COUNTERS.fetch_s += time.perf_counter() - t0
        COUNTERS.experts += len(want)
        return {e: got[e] for e in want}

    def _stacked(self, experts: dict, unique: np.ndarray) -> dict:
        out = {}
        first = experts[int(unique[0])]
        for name in first:
            if name.startswith("__"):
                continue
            out[name] = mx.stack([experts[int(e)][name] for e in unique])
        return out

    # -------------------------------------------------------------- compute

    def _proj(self, x, w: dict, proj: str, idx, sort: bool):
        spec = self.specs[proj]
        if spec.quantized and fastgrad.active():
            y = fastgrad.gather_qmm(
                x,
                w[f"{proj}.weight"],
                w[f"{proj}.scales"],
                w.get(f"{proj}.biases"),
                idx,
                group_size=spec.group_size,
                bits=spec.bits,
                mode=spec.mode,
                sorted_indices=sort,
            )
        elif spec.quantized:
            y = mx.gather_qmm(
                x,
                w[f"{proj}.weight"],
                w[f"{proj}.scales"],
                w.get(f"{proj}.biases"),
                rhs_indices=idx,
                transpose=True,
                group_size=spec.group_size,
                bits=spec.bits,
                mode=spec.mode,
                sorted_indices=sort,
            )
        else:
            y = mx.gather_mm(
                x,
                mx.swapaxes(w[f"{proj}.weight"], -1, -2),
                rhs_indices=idx,
                sorted_indices=sort,
            )
        if spec.has_bias:
            y = y + mx.expand_dims(w[f"{proj}.bias"][idx], -2)
        return y

    def _pair(self, x, w: dict, idx, sort: bool):
        """up and gate in one op, so their backward holds one dense copy at a time."""
        def args(proj):
            s = self.specs[proj]
            return (w[f"{proj}.weight"], w[f"{proj}.scales"], w.get(f"{proj}.biases"),
                    s.group_size, s.bits, s.mode)

        ys = fastgrad.gather_qmm2(x, args("up_proj"), args("gate_proj"), idx, sorted_indices=sort)
        out = []
        for proj, y in zip(("up_proj", "gate_proj"), ys):
            if self.specs[proj].has_bias:
                y = y + mx.expand_dims(w[f"{proj}.bias"][idx], -2)
            out.append(y)
        return out

    def __call__(self, x, indices):
        COUNTERS.calls += 1
        mx.eval(indices)
        ids = np.asarray(indices)
        unique = np.unique(ids)
        object.__setattr__(self, "last_ids", unique)
        w = self._stacked(self._fetch(unique), unique)
        local = mx.array(np.searchsorted(unique, ids).astype(np.uint32))

        x = mx.expand_dims(x, (-2, -3))
        sort = local.size >= 64
        idx = local
        inv = None
        if sort:
            M = local.shape[-1]
            flat = local.flatten()
            order = mx.argsort(flat)
            inv = mx.argsort(order)
            fast = fastgrad.active()
            x = fastgrad.spread(x.flatten(0, -3), order, inv) if fast else x.flatten(0, -3)[order // M]
            idx = flat[order]
        if fastgrad.active() and self.specs["up_proj"].quantized and self.specs["gate_proj"].quantized:
            x_up, x_gate = self._pair(x, w, idx, sort)
        else:
            x_up = self._proj(x, w, "up_proj", idx, sort)
            x_gate = self._proj(x, w, "gate_proj", idx, sort)
        y = self._proj(self.activation(x_up, x_gate), w, "down_proj", idx, sort)
        if sort:
            y = fastgrad.unsort(y, inv, order) if fast else y[inv]
            y = mx.unflatten(y, 0, local.shape)
        return y.squeeze(-2)


def _parent(model, path: str):
    obj = model
    parts = path.split(".")
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj, parts[-1]


def install(model) -> list[TrainSwitch]:
    """Swap every StreamedSwitchGLU for a TrainSwitch. Returns them in layer order."""
    from ..streaming import StreamedSwitchGLU

    found = [
        (name, m)
        for name, m in model.named_modules()
        if isinstance(m, StreamedSwitchGLU)
    ]
    found.sort(key=lambda t: t[1].ring_index)
    swaps = []
    for name, m in found:
        parent, attr = _parent(model, name)
        ts = TrainSwitch(m)
        setattr(parent, attr, ts)
        swaps.append((parent, attr, m, ts))
    # A list set through Module.__setattr__ would join the parameter tree.
    object.__setattr__(model, "_pagedmoe_train_swaps", swaps)
    return [s[3] for s in swaps]


def uninstall(model) -> None:
    """Put the StreamedSwitchGLU modules back (for generation)."""
    for parent, attr, streamed, ts in model.__dict__.get("_pagedmoe_train_swaps", []):
        ts._drain_ahead()
        setattr(parent, attr, streamed)


def reinstall(model) -> None:
    for parent, attr, _streamed, ts in model.__dict__.get("_pagedmoe_train_swaps", []):
        setattr(parent, attr, ts)
