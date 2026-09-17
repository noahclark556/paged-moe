# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
StreamedSwitchGLU: a drop-in replacement for mlx-lm's SwitchGLU that pulls
expert weights from the ExpertCache (disk-backed) instead of holding them all
in memory.

How mlx-lm MoE models work, in one paragraph
--------------------------------------------
Every MoE model in mlx-lm (GLM-4.x-MoE, Qwen3-MoE, OLMoE, Mixtral, DeepSeek,
and future ones like glm5_next) computes a router score per token, picks the
top-k expert ids, and then calls `self.switch_mlp(x, indices)` where
switch_mlp is a `SwitchGLU` holding *all* experts stacked in big tensors.
That one class is the only place expert weights are ever touched, which makes
it the perfect seam: we swap each SwitchGLU instance for a StreamedSwitchGLU
and the rest of the model (attention, norms, routers, shared experts - the
"backbone") runs completely unmodified.

What happens per forward call
-----------------------------
1. `indices` (which experts each token wants) is forced to CPU - this is the
   one unavoidable sync point, because we cannot know what to read from disk
   until the router has spoken.
2. Decode (a handful of tokens): the needed experts are fetched (LRU cache /
   parallel disk reads), compute runs per expert directly on the cached
   arrays - no stacking, no weight copies - and speculative prefetch warms
   the next layers using the previous token's routing.
3. Prefill (many tokens): a long prompt activates nearly *every* expert of
   every layer, which can never fit in memory at once.  Experts stream
   through in bounded groups (config.GROUP_BYTES): while the GPU computes
   group i, worker threads read group i+1.  Groups are cached "softly"
   (installed only while there's free room - never evicting), so prefill
   cannot thrash the cache that decode depends on, and each group's weights
   are dropped as soon as its matmuls have run.
"""

from __future__ import annotations

import re
import time

import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU

from . import config
from .cache import ExpertCache
from .safetensors_index import DirectLocator, ExpertLocator, StackedLocator, TensorLoc

_PROJS = ("gate_proj", "up_proj", "down_proj")

# At most this many tokens counts as "decode" (cache + prefetch path);
# anything bigger is treated as prefill (grouped streaming path).
#
# 12, not 4: speculative decoding verifies num_draft_tokens+1 positions in one
# forward call, and that batch must stay on the decode path - it wants the
# full cache, route prediction, and per-expert compute on resident weights.
# (This is also the whole reason speculative decoding pays on a streamed MoE:
# the batch reads each layer's expert *union* once - consecutive tokens share
# roughly half their experts - and reads every weight once for k tokens
# instead of k times.) Real prefill chunks are hundreds of tokens minimum, so
# there is no ambiguity in between.
_DECODE_MAX_TOKENS = 12


class _ProjSpec:
    """Shape/quantization info for one of gate/up/down, captured from the
    original (Quantized)SwitchLinear before we discard its weights."""

    def __init__(self, module):
        self.quantized = isinstance(module, QuantizedSwitchLinear)
        if self.quantized:
            self.group_size = module.group_size
            self.bits = module.bits
            self.mode = module.mode
        self.has_bias = "bias" in module


class _RoutePredictor:
    """One MoE layer's router, callable on another layer's hidden state.

    Plain object on purpose: holding an nn.Module or an mx.array as an
    attribute of StreamedSwitchGLU would register it a second time in the
    model's parameter tree (double eval, double reported size). Nothing here
    owns weights - they are borrowed references into the resident backbone.
    """

    __slots__ = ("gate", "norm_weight")

    def __init__(self, gate, norm_weight):
        self.gate = gate
        self.norm_weight = norm_weight


# The router output of the most recent gate call: (id(gate), id(x), out).
#
# Every MoE block runs `self.gate(x)` and then `self.switch_mlp(x, inds)` with
# the same `x`, so the weights the selectors need are already in the graph by
# the time the streamed layer runs - re-deriving them means a second full
# router pass (on GLM-4.7 that is ~10 extra kernels to encode inside a
# blocking eval, 89 times per token). One slot is enough because the gate call
# immediately precedes the switch_mlp call; route prediction also runs gates,
# but only from the *previous* layer, so its write is always overwritten by the
# real one before anybody reads it. Both ids are checked anyway, and a mismatch
# just falls back to re-running the router.
_LAST_ROUTE: tuple | None = None
_TAPPED_ROUTERS: dict[type, type] = {}


def _tap_router(gate) -> None:
    """Record `gate`'s output on the way past, for _tapped_route to pick up.

    Retypes the instance instead of wrapping it: `block.gate(x)` looks
    __call__ up on the class, and swapping the module for a wrapper object
    would take the router's weights out of the model's parameter tree.
    """
    cls = type(gate)
    if cls in _TAPPED_ROUTERS.values():
        return
    tapped = _TAPPED_ROUTERS.get(cls)
    if tapped is None:
        base_call = cls.__call__

        def __call__(self, x):  # noqa: N807 - mirrors the wrapped signature
            out = base_call(self, x)
            global _LAST_ROUTE
            _LAST_ROUTE = (id(self), id(x), out)
            return out

        tapped = type(f"Tapped{cls.__name__}", (cls,), {"__call__": __call__})
        _TAPPED_ROUTERS[cls] = tapped
    try:
        gate.__class__ = tapped
    except TypeError:
        pass  # exotic router class; the re-run path still works


def _route_rank_mx(gate, x, ind2):
    """A monotone score per routed slot as an mx.array, without a sync.

    Flow decode only ever *ranks* the router's own top-k, so any monotone
    transform of the mixture weight will do: MoEGate hands its scores back
    already, and for a logits router the selected logits order exactly the way
    a softmax over them would. Returns None when the router output was not
    captured, which sends the layer down the ordinary per-layer path.
    """
    if gate is None:
        return None
    out = _tapped_route(gate, x)
    if out is None:
        return None
    K = ind2.shape[-1]
    if isinstance(out, (tuple, list)):
        if len(out) < 2 or not isinstance(out[1], mx.array):
            return None
        return out[1].reshape(-1, K).astype(mx.float32)
    if isinstance(out, mx.array) and out.ndim >= 2:
        logits = out if out.ndim == 2 else out.reshape(-1, out.shape[-1])
        return mx.take_along_axis(logits, ind2, axis=-1).astype(mx.float32)
    return None


def _tapped_route(gate, x):
    """This layer's own router output, or None if it was not the last call.

    `x` is the *unreshaped* activation the block passed to both the gate and
    switch_mlp, so identity on it is what proves the recorded output belongs to
    this forward rather than to a prediction or a previous layer.
    """
    rec = _LAST_ROUTE
    if rec is None or rec[0] != id(gate) or rec[1] != id(x):
        return None
    return rec[2]


class StreamedSwitchGLU(nn.Module):
    def __init__(
        self,
        layer_key: str,
        original: SwitchGLU,
        cache: ExpertCache,
        ring: "PrefetchRing",
        predictor: _RoutePredictor | None = None,
    ):
        super().__init__()
        self.layer_key = layer_key
        self.cache = cache
        self.ring = ring
        self.pred = predictor
        self.ring_index = ring.register(self)

        self.activation = original.activation
        self.specs = {p: _ProjSpec(original[p]) for p in _PROJS}

        # Routing of the previous decode step (or the last prompt token after
        # prefill); used by PrefetchRing to guess what this layer will want
        # for the next token.
        self.last_ids: list[int] = []
        # Experts an earlier layer predicted this layer would want on this
        # step (accuracy accounting only - never used to pick experts).
        self.predicted: set[int] = set()
        # Gather row indices per (N, M), built once (flow decode).
        self._flow_lhs_cache: dict[tuple, mx.array] = {}

    # ------------------------------------------------------------- compute

    def _proj(self, xe: mx.array, expert: dict, proj: str) -> mx.array:
        """One expert's gate/up/down projection on [M, D] activations."""
        spec = self.specs[proj]
        if spec.quantized:
            y = mx.quantized_matmul(
                xe,
                expert[f"{proj}.weight"],
                expert[f"{proj}.scales"],
                expert.get(f"{proj}.biases"),
                transpose=True,
                group_size=spec.group_size,
                bits=spec.bits,
                mode=spec.mode,
            )
        else:
            y = xe @ expert[f"{proj}.weight"].T
        if spec.has_bias:
            y = y + expert[f"{proj}.bias"]
        return y

    def _expert_ffn(self, xe: mx.array, expert: dict) -> mx.array:
        x_up = self._proj(xe, expert, "up_proj")
        x_gate = self._proj(xe, expert, "gate_proj")
        return self._proj(self.activation(x_up, x_gate), expert, "down_proj")

    # --------------------------------------------------- slab (gather) path

    def _gather_proj(self, xe, proj: str, rhs, lhs=None) -> mx.array:
        """One projection for *every* expert of this layer, in one call.

        `rhs` holds the slab slot of the expert each output row needs and `lhs`
        which row of `xe` it reads, so the whole layer is a single kernel
        instead of three per expert.
        """
        spec = self.specs[proj]
        slab = self.cache.slab
        w = slab.compute(f"{proj}.weight")
        if spec.quantized:
            biases = (
                slab.compute(f"{proj}.biases")
                if slab.has(f"{proj}.biases")
                else None
            )
            y = mx.gather_qmm(
                xe,
                w,
                slab.compute(f"{proj}.scales"),
                biases,
                lhs_indices=lhs,
                rhs_indices=rhs,
                transpose=True,
                group_size=spec.group_size,
                bits=spec.bits,
                mode=spec.mode,
            )
        else:
            y = mx.matmul(xe if lhs is None else xe[lhs], mx.swapaxes(w[rhs], -1, -2))
        if spec.has_bias:
            y = y + slab.compute(f"{proj}.bias")[rhs][:, None, :]
        return y

    def _loose(self, entry: dict) -> dict:
        """A slot-backed entry as per-expert arrays (views, not copies).

        Only for the fallback path: a layer whose experts did not all get
        slots cannot be addressed by one gather, and `_run` wants arrays.
        """
        slot = entry.get("__slot__")
        if slot is None:
            return entry
        slab = self.cache.slab
        out = {"__nbytes__": entry["__nbytes__"]}
        for name in slab.names:
            out[name] = slab.expert(name, slot)
        return out

    def _run_slab(self, x_flat, inds: np.ndarray, slots: dict, keep=None):
        """Decode compute over slab slots: one gather_qmm per projection.

        Every output *row* is its own batch element, indexed into the slab by
        the slot of the expert it needs. That makes a variable number of rows
        per expert free (positions of a verify batch route differently), and
        results come back already in (token, k) order - so unlike `_run` there
        is no sort, no unsort, and no per-expert Python loop. Feeding `x` by
        `lhs_indices` also avoids materializing the gathered activations.

        Bit-identical to `_run` on the same weights: same kernel, same
        quantization parameters, just batched (verified in tests/test_stream.py).
        """
        N, K = inds.shape
        flat = inds.reshape(-1)
        if keep is None:
            kept_pos = None
            rows = np.arange(N * K, dtype=np.uint32) // K
        else:
            kept_pos = np.nonzero(keep.reshape(-1))[0]
            flat = flat[kept_pos]
            rows = (kept_pos // K).astype(np.uint32)
        if flat.size == 0:
            return mx.zeros((N, K, x_flat.shape[-1]), dtype=x_flat.dtype)

        self.cache.check_slots(self.layer_key, slots)
        slot_ids = np.fromiter(
            (slots[int(e)] for e in flat), dtype=np.uint32, count=flat.size
        )
        lhs = mx.array(rows)
        rhs = mx.array(slot_ids)
        xe = x_flat[:, None, :]  # [N, 1, D]: batch dim is the token

        x_up = self._gather_proj(xe, "up_proj", rhs, lhs)
        x_gate = self._gather_proj(xe, "gate_proj", rhs, lhs)
        y = self._gather_proj(self.activation(x_up, x_gate), "down_proj", rhs)
        y = y.reshape(y.shape[0], y.shape[-1])  # [n_kept, D]

        if keep is None:
            return y.reshape(N, K, -1)
        out = mx.zeros((N * K, y.shape[-1]), dtype=y.dtype)
        out[mx.array(kept_pos)] = y
        return out.reshape(N, K, -1)

    # ------------------------------------------------------ flow (no sync)

    def _flow_lhs(self, n_rows: int, m: int):
        """Row-of-x for each gather row. Static per (N, M), so build it once."""
        key = (n_rows, m)
        lhs = self._flow_lhs_cache.get(key)
        if lhs is None:
            lhs = mx.array((np.arange(n_rows * m, dtype=np.uint32) // m))
            self._flow_lhs_cache[key] = lhs
        return lhs

    def _run_flow(self, x_flat, ind2, rank, table, topm: int):
        """Decode compute with no host round trip.

        `table` maps expert id -> slab slot (-1 when not resident) and lives on
        the GPU, so residency, selection and the gather indices are all mx ops.
        Nothing here needs the routing on the CPU, which is what lets a whole
        token be one graph instead of one graph per MoE layer.

        Returns [N, K, D]: computed rows sit in their own router slots and every
        other slot is zero, so the parent block's weighted sum simply omits what
        was not computed - the same shape contract `_run_slab` honors under a
        prune mask.
        """
        N, K = ind2.shape
        m = K if topm <= 0 else min(topm, K)
        slots = mx.take(table, ind2)  # [N, K]
        resident = slots >= 0
        if m < K:
            # Rank among *resident* experts only: one we would have to read
            # cannot displace one we can compute now. Every id considered is
            # still one this layer's router chose, so this picks among the
            # router's experts and never substitutes for them.
            sel = mx.argpartition(
                -mx.where(resident, rank, mx.array(-mx.inf, mx.float32)),
                kth=m - 1,
                axis=-1,
            )[..., :m]
            sel_slots = mx.take_along_axis(slots, sel, axis=-1)
            valid = mx.take_along_axis(resident, sel, axis=-1)
        else:
            sel = None
            sel_slots = slots
            valid = resident

        rhs = mx.maximum(sel_slots, 0).reshape(-1).astype(mx.uint32)
        lhs = self._flow_lhs(N, m)
        xe = x_flat[:, None, :]
        x_up = self._gather_proj(xe, "up_proj", rhs, lhs)
        x_gate = self._gather_proj(xe, "gate_proj", rhs, lhs)
        y = self._gather_proj(self.activation(x_up, x_gate), "down_proj", rhs)
        y = y.reshape(N, m, -1)
        # A slot with no resident expert contributes nothing. Zeroing the row is
        # the graph-only equivalent of leaving it out of the gather, which would
        # need a host-side count of what survived.
        y = y * valid[..., None].astype(y.dtype)
        if sel is None:
            return y, valid
        out = mx.zeros((N, K, y.shape[-1]), dtype=y.dtype)
        idx = mx.broadcast_to(sel[..., None], y.shape)
        return mx.put_along_axis(out, idx, y, axis=-2), valid

    def _flow_forward(self, x, x_flat, ind2):
        """One MoE layer of a flow-decode token, or None to use the sync path."""
        table = self.cache.flow_table(self.layer_key)
        if table is None:
            return None
        rank = _route_rank_mx(self.pred.gate if self.pred else None, x, ind2)
        if rank is None:
            return None
        if self.ring_index == 0:
            # The previous token's routing is materialized by now (the sampler
            # synced it), so this is where the reads for this token get issued -
            # a whole token of lead time instead of a few layers.
            self.ring.flow_pump()
            self.cache.leave_prefill()
        out, valid = self._run_flow(x_flat, ind2, rank, table, config.FLOW_TOPM)
        self.ring.flow_note(self.ring_index, ind2, valid)
        if self.ring_index + 1 >= len(self.ring.layers):
            self.ring.flow_note_hidden(x_flat[-1])
        return out

    def _run(self, x_flat, inds: np.ndarray, groups, eval_groups: bool, keep=None):
        """Shared compute: iterate experts in ascending-id order, run the FFN
        directly on the cached weights (no stacking/copying), then unsort back
        to (token, k) order.

        `groups` yields (ids, {id: expert_tensors}) with ids ascending across
        the whole iteration and covering np.unique(inds) exactly (np.unique of
        the *kept* slots when a prune mask is given).

        `keep` (optional, decode pruning) is a bool [N, K] mask; slots pruned
        by the router-weight threshold produce a zero row, which the calling
        MoE block then weighs by that expert's (small) router weight and sums -
        i.e. the dropped expert's contribution is simply omitted.

        Rows are gathered **once** for the whole layer, in expert order, so
        each expert's activations are a contiguous slice of `x_sorted`. Doing
        it per expert instead costs one host->device index copy plus one gather
        kernel per expert - during prefill that is tens of thousands of tiny
        dispatches per prompt chunk, which is the difference between being
        disk-bound (what we want) and launch-bound.
        """
        N, K = inds.shape
        flat = inds.reshape(-1)
        if keep is not None:
            kept_pos = np.nonzero(keep.reshape(-1))[0]
            flat = flat[kept_pos]
        order = np.argsort(flat, kind="stable")  # positions grouped by expert
        sorted_e = flat[order]

        if keep is None:
            rows = order // K
        else:
            rows = kept_pos[order] // K
        x_sorted = x_flat[mx.array(rows)]  # [n_kept, D], grouped by expert

        ys: list[mx.array] = []
        for ids, experts in groups:
            group_ys: list[mx.array] = []
            for e in ids:
                lo = np.searchsorted(sorted_e, e, "left")
                hi = np.searchsorted(sorted_e, e, "right")
                if lo == hi:
                    continue
                # Cache hits may be slot-backed (decode installs those), and a
                # slot entry carries a slot number, not arrays. Prefill sees
                # them whenever a prompt arrives after decode has populated
                # slots - i.e. every turn after the first.
                entry = experts[e]
                if "__slot__" in entry:
                    entry = self._loose(entry)
                group_ys.append(self._expert_ffn(x_sorted[lo:hi], entry))
            if eval_groups and group_ys:
                # Force this group's matmuls to run NOW, so its (possibly
                # uncached) weights can be freed before the next group lands.
                # Reads for the next group are already in flight, so the GPU
                # and the SSD stay busy simultaneously.
                mx.eval(*group_ys)
            ys.extend(group_ys)

        y_sorted = mx.concatenate(ys, axis=0)  # [n_kept, D] in expert order

        if keep is None:
            inv = np.empty_like(order)
            inv[order] = np.arange(order.size)
            return y_sorted[mx.array(inv)].reshape(N, K, -1)

        # Scatter kept rows back to their (token, k) slots; pruned slots stay 0.
        y = mx.zeros((N * K, y_sorted.shape[-1]), dtype=y_sorted.dtype)
        y[mx.array(kept_pos[order])] = y_sorted
        return y.reshape(N, K, -1)

    # ------------------------------------------------------------- forward

    def __call__(self, x, indices) -> mx.array:
        lead_shape = indices.shape  # (..., K)
        K = lead_shape[-1]
        x_flat = x.reshape(-1, x.shape[-1])
        ind2 = indices.reshape(-1, K)
        N = ind2.shape[0]  # static shape: known without a sync

        # Flow decode: keep the whole layer on the GPU so the token needs no
        # per-layer sync at all. Returns None (and falls through to the path
        # below) whenever its preconditions do not hold - no slab, an
        # unrecognized router, a table that does not exist yet - so enabling it
        # can never break a model, only leave it on the ordinary path.
        if config.FLOW and N <= _DECODE_MAX_TOKENS and self.cache.slab is not None:
            if self.ring_index == 0:
                self.ring.flow_token_start()
            if self.ring.flow_ready():
                flowed = self._flow_forward(x, x_flat, ind2)
                if flowed is not None:
                    return flowed.reshape(*lead_shape, -1)

        # Opt-in approximate routing. Re-derive the router's weights for the
        # selected experts (the router is a resident matvec - the same decision
        # the block just made), then decide which slots to actually compute:
        # PRUNE (weight vs the winner), ROUTE_TOP_P (mixture mass),
        # ROUTE_TOP_K (rank), all composing by intersection. Dropped slots
        # output zeros, so the block's weighted sum simply omits those
        # contributions.
        #
        # Works for both router families: nn.Linear logits (Qwen3-MoE /
        # Qwen3-Next) and MoEGate (inds, scores) (GLM / DeepSeek). See
        # `_route_weights_from_gate`.
        #
        # Decode only. Prefill computes the full mixture, so the hidden states a
        # prompt establishes are exact and the approximation never compounds
        # through the KV cache.
        #
        # Whatever the selectors drop, KEEP_FREE then puts back anything that
        # needs no disk read (below) - that is what keeps the quality cost
        # proportional to the bytes actually saved. WAIT_ABOVE then skips any
        # surviving miss whose weight is too small to be worth blocking on.
        prune = float(config.PRUNE)
        cap = int(config.ROUTE_TOP_K)
        cap_on = 0 < cap < K
        top_p = float(config.ROUTE_TOP_P)
        top_p_on = 0.0 < top_p < 1.0
        wait_above = float(config.WAIT_ABOVE)
        wait_on = wait_above > 0.0
        need_weights = (
            0.0 < prune <= 1.0 or cap_on or top_p_on or wait_on
            or (
                self.ring.sidecar is not None
                and getattr(self.ring.sidecar, "prune", None) is not None
                and self.ring.sidecar.prune.enabled
            )
        )
        keep = None
        renorm = None
        wnp = None
        if (
            need_weights
            and N <= _DECODE_MAX_TOKENS
            and self.pred is not None
            and self.pred.gate is not None
        ):
            try:
                tapped = _tapped_route(self.pred.gate, x)
                self.cache.route_calls += 1
                self.cache.route_tapped += tapped is not None
                wnp = _route_weights_from_gate(
                    self.pred.gate,
                    x_flat,
                    ind2,
                    pending=self.ring.pending_sync(),
                    tapped=tapped,
                )
                if wnp is not None:
                    # Adaptive prune/wait from sidecar when head is live.
                    if (
                        self.ring.sidecar is not None
                        and self.ring.sidecar.prune.enabled
                    ):
                        prune, wait_above = self.ring.sidecar.prune_params(wnp)
                        wait_on = wait_above > 0.0
                    keep = _selector_mask(wnp, prune=prune, cap=cap, top_p=top_p)
                    if keep is None and wait_on:
                        keep = np.ones(wnp.shape, dtype=bool)
                    if self.ring.sidecar is not None:
                        self.ring.sidecar.observe_prune(wnp, prune)
            except Exception:
                keep = None
                wnp = None

        # Sync point: router decisions must reach the CPU before we can read
        # the right experts from disk. (With the weight block above this is a
        # plain copy; without it, carry the deferred prediction here instead.)
        if keep is None and N <= _DECODE_MAX_TOKENS:
            pending = self.ring.pending_sync()
            if pending:
                mx.eval(ind2, *pending)
        inds = np.asarray(ind2)

        if keep is not None and wnp is not None:
            free_arr = None
            if config.KEEP_FREE or wait_on:
                # One residency query serves both KEEP_FREE and WAIT_ABOVE.
                need = np.unique(inds if wait_on else inds[~keep])
                free = self.cache.free_ids(self.layer_key, need)
                free_arr = np.array(sorted(free), inds.dtype)
            if config.KEEP_FREE and free_arr is not None and not keep.all():
                # Put back every dropped expert that needs no disk read. This
                # is the one part of the trade that is strictly one-sided:
                # those bytes are already here, so the only thing dropping them
                # bought was a worse mixture.
                rescued = np.isin(inds, free_arr) & ~keep
                if rescued.any():
                    keep = keep | rescued
                    self.cache.kept_free += int(rescued.sum())
            if wait_on and free_arr is not None:
                # Skip - and prefetch for later tokens - any surviving expert
                # that would block on a read and isn't worth blocking for.
                # This is what takes the disk off the critical path.
                stall = keep & ~np.isin(inds, free_arr)
                skip = stall & (wnp < wait_above)
                if skip.any():
                    keep = keep & ~skip
                    late = np.unique(inds[skip]).tolist()
                    self.cache.skipped_waits += len(late)
                    self.cache.prefetch(self.layer_key, late)
            if config.ROUTE_RENORM and not keep.all():
                # Mixture mass that survived. The block sums score_i * y_i over
                # kept slots, so scaling the kept rows by 1/mass turns that into
                # a properly normalized mixture over the computed experts.
                # Measured worse than leaving it attenuated - see config.
                mass = (wnp * keep).sum(axis=-1, keepdims=True)
                renorm = np.reciprocal(np.maximum(mass, 1e-6))

        if keep is not None:
            unique = np.unique(inds.reshape(-1)[keep.reshape(-1)])
            self.cache.pruned_slots += int(keep.size - keep.sum())
            self.cache.demand_slots += int(keep.size)
        else:
            unique = np.unique(inds)

        if N <= _DECODE_MAX_TOKENS:
            ids = [int(e) for e in unique]
            trace = getattr(self.cache, "demand_trace", None)
            if trace is not None:
                trace.append((self.layer_key, ids))
            # Decode wants the full cache back: a big resident set is what
            # keeps per-token disk reads down (prefill shrinks it, below).
            self.cache.leave_prefill()
            if self.predicted:
                self.cache.pred_issued += len(self.predicted)
                self.cache.pred_used += len(self.predicted.intersection(ids))
                self.predicted.clear()
            self.cache.route_total += len(ids)
            # What the previous-token heuristic would have covered on its own,
            # measured on the same traffic: the baseline prediction has to beat.
            if self.last_ids:
                self.cache.heur_used += len(set(self.last_ids).intersection(ids))
            # Sidecar observes real demand (no-op when disabled).
            self.ring.note_demand(self.layer_key, ids)
            experts = self.cache.fetch(self.layer_key, ids, install="lru")
            # Issue the reads for the previous stride's prediction - its
            # merged array was materialized by this layer's router sync above,
            # so this is host work only. After the fetch, not before: even on
            # a separate executor lane the drive is shared, and speculative
            # reads running during this layer's blocking miss window measured
            # 12 ms/token slower than letting them start in the compute gap.
            self.ring.flush()
            # Prefetch state is per *position*, not per batch. A speculative
            # verify batch (N = num_draft+1) demands the union of its
            # positions - up to N*K ids - but feeding that union to the
            # predictors and the heuristic ring multiplies every speculative
            # read by N for no gain in accuracy: measured on a reasoning
            # prompt at N=5, prefetch read and threw away 228 GB per 96
            # tokens (vs 51 GB at N=1), nearly doubling total SSD traffic and
            # making speculation 40% SLOWER than plain decode. Both sources
            # below therefore look at the last position only - the one that
            # actually continues the sequence.
            last = inds[-1] if keep is None else inds[-1][keep[-1]]
            self.last_ids = [int(e) for e in np.unique(last)]
            # Warm upcoming layers while this layer's matmuls run. x_flat is
            # already materialized (the sync above forced it), so the routers
            # of the next layers can be run on it for the price of a matvec.
            # Predict against the *effective* k: with a cap on, no layer will
            # ever demand more than `cap` experts, so predicting the router's
            # full K would spend the disk on slots that cannot be requested.
            self.ring.speculate(
                self.ring_index, x_flat[-1:], cap if cap_on else K, positions=N
            )
            # Last MoE layer of the token: close the sidecar's token window so
            # it can train and prefetch the next token's early layers. Cheap
            # None-check when sidecar is off.
            if self.ring_index + 1 >= len(self.ring.layers):
                # Hand over this position's hidden state too: it is what the LM
                # head turns into the next token, so it is by far the strongest
                # predictor of that token's early-layer routing. Already
                # materialized, so this is a 1-row host copy.
                self.ring.end_decode_token(
                    hidden=x_flat[-1] if self.ring.sidecar is not None else None
                )
            slots = None
            if self.cache.slab is not None:
                # Every expert must be slot-backed for one gather to address
                # them all; a slot-starved fetch leaves loose arrays behind
                # and that layer falls back to the per-expert loop.
                slots = {e: ent["__slot__"] for e, ent in experts.items()
                         if "__slot__" in ent}
                if len(slots) != len(experts):
                    slots = None
            if slots is not None:
                out = self._run_slab(x_flat, inds, slots, keep=keep)
            else:
                # Mixed layer (something missed a slot): _run converts the
                # slot-backed entries itself.
                out = self._run(
                    x_flat, inds, [(ids, experts)], eval_groups=False, keep=keep
                )
            if renorm is not None:
                scale = mx.array(renorm.astype(np.float32)).astype(out.dtype)
                out = out * scale[:, :, None]
        else:
            # A prediction left over from the previous decode token is stale
            # (routing came from a hidden state this prompt replaces).
            self.ring.drop_pending()
            self.ring.flow_drop()
            # Clear decode-token sidecar state once at the start of the prefill
            # pass - NOT on every MoE layer. drop_token() aborts the prefill
            # chunk window; calling it per-layer left prefill_r stuck at 0/0.
            if self.ring.sidecar is not None and self.ring_index == 0:
                self.ring.sidecar.drop_token()
            if N >= config.PREFILL_SHRINK_TOKENS:
                self.cache.enter_prefill(int(config.PREFILL_CACHE_GB * (1 << 30)))
            # Prefill-union head: start-of-chunk prefetch (no-op when off).
            if self.ring.sidecar is not None and self.ring_index == 0:
                self.ring.sidecar.begin_prefill_chunk(self.cache)
            if self.ring.sidecar is not None:
                self.ring.sidecar.note_prefill_demand(
                    self.layer_key, [int(e) for e in unique]
                )
            groups = self.cache.fetch_groups(
                self.layer_key,
                [int(e) for e in unique],
                config.GROUP_BYTES,
                install="soft",
            )
            out = self._run(x_flat, inds, groups, eval_groups=True)
            self.cache.prefill_layers += 1
            # Seed decode prefetch with the last prompt token's routing.
            self.last_ids = [int(e) for e in np.unique(inds[-1])]
            self.predicted.clear()  # stale across a prefill (accounting only)
            # Last MoE layer of a prefill chunk: close the union window.
            if (
                self.ring.sidecar is not None
                and self.ring_index + 1 >= len(self.ring.layers)
            ):
                self.ring.sidecar.end_prefill_chunk()

        return out.reshape(*lead_shape, -1)


class PrefetchRing:
    """Coordinates speculative prefetch across MoE layers during decode.

    Layers register in construction order (== execution order). When layer i
    runs, we start reads for the layers about to run, so their experts arrive
    while the GPU is still busy with layer i. Two sources of guesses, in
    decreasing order of value:

    1. Route prediction. Every layer's router is a small matrix that lives in
       the resident backbone, so we can simply *ask* layer i+1 which experts it
       wants - feeding it layer i's hidden state, which is a good estimate of
       its own input (the two differ by one attention block plus one FFN on a
       residual stream). This is the same routing decision the layer will make
       a millisecond later, not a guess about token similarity, so it holds up
       exactly where the previous-token heuristic breaks down: the first token
       of a reply, a topic change mid-generation, and long generations that
       wander. The result feeds cache.prefetch() only; the layer still computes
       with whatever its real router selects, so output is unaffected.

    2. Previous-token routing (last_ids), for the layers beyond prediction
       depth, where the hidden state has drifted too far to be informative.

    Both go through prefetch(), which drops ids that are already resident, so
    agreement between the two sources costs nothing.
    """

    def __init__(
        self,
        cache: ExpertCache,
        depth: int,
        predict_depth: int = 0,
        slack: int = 0,
        renorm: bool = True,
        predict_mode: str = "auto",
        lead: int = 0,
    ):
        self.cache = cache
        self.depth = depth
        self.predict_depth = predict_depth
        # Extra layers of head start: layer i predicts layers
        # i+lead+1 .. i+lead+predict_depth instead of i+1 .. i+predict_depth.
        # Window size (read burst per stride) and lead time are different
        # knobs: a 10 MB expert read needs more than the ~2 ms/layer of
        # compute that depth alone provides as cover, but simply deepening
        # the window (depth 8) was measured slower - the 8-layer burst
        # crowds the disk queue right when a real miss needs it, and
        # prediction accuracy at distance 7-8 decays under the prune test.
        self.lead = lead
        self.slack = slack
        self.renorm = renorm
        self.layers: list[StreamedSwitchGLU] = []
        self._ratios: dict[tuple[int, int], mx.array | None] = {}
        # "auto" governor state (see _token_boundary).
        self._configured_depth = predict_depth
        self._auto = predict_mode == "auto" and predict_depth > 0
        self._probing = self._auto
        # Per-state samples: [windows, sum(tok/s), sum(tok/s^2), tokens]. The
        # spread matters as much as the mean here, so keep enough to estimate it.
        self._acc: dict[bool, list[float]] = {
            True: [0.0, 0.0, 0.0, 0.0],
            False: [0.0, 0.0, 0.0, 0.0],
        }
        self._decision: bool | None = None
        self._hold_left = 0
        self._last_token_t: float | None = None
        self._win_tokens = 0
        self._win_time = 0.0
        # Deferred prediction awaiting its ride-along sync (see speculate).
        self._pending: tuple | None = None
        # Online expert-prefetch sidecar (None when EXPERT_STREAM_SIDECAR=0).
        # Attached by patch_model / loader only when enabled - every call site
        # below is gated on this so disabled models pay a single None-check.
        self.sidecar = None
        self._token_open = False
        # Flow decode: one token's lazy routing, read at the next token's start.
        self._flow_recs: list[tuple] = []
        self._flow_hidden = None
        self._flow_tokens = 0

    def attach_sidecar(self, sidecar) -> None:
        self.sidecar = sidecar
        # Optional eviction advisor - None when residency head is off.
        if (
            sidecar is not None
            and getattr(sidecar, "residency", None) is not None
            and sidecar.residency.enabled
        ):
            self.cache._residency_advisor = sidecar
        else:
            self.cache._residency_advisor = None

    def note_demand(self, layer_key: str, expert_ids: list[int]) -> None:
        if self.sidecar is None:
            return
        self.sidecar.note_demand(layer_key, expert_ids)
        self._token_open = True

    # --------------------------------------------------------- flow decode

    def flow_token_start(self) -> None:
        """Count decode tokens since the last prefill (flow warmup gate)."""
        self._flow_tokens += 1
        if self._flow_tokens == config.FLOW_WARMUP + 1:
            # Handing over from the per-layer path: its last deferred prediction
            # has no sync left to ride on, so let it go rather than leak a graph.
            self.drop_pending()

    def flow_ready(self) -> bool:
        return self._flow_tokens > config.FLOW_WARMUP

    def flow_note(self, index: int, ind2, valid) -> None:
        """Hold a layer's routing for the next token to read.

        These stay lazy on purpose: touching them here is the sync flow decode
        exists to avoid. The sampler materializes them at the end of the token
        (they are ancestors of the sampled id), and `flow_pump` reads them at
        the start of the next one, by which time it is a host copy.
        """
        self._flow_recs.append((index, ind2, valid))

    def flow_note_hidden(self, hidden) -> None:
        self._flow_hidden = hidden

    def flow_pump(self) -> None:
        """Start of a flow-decode token: learn the last one, read for this one.

        This is the whole host side of flow decode, and it runs once per token
        rather than once per layer. Last token's routing is the best available
        statement of what this token wants, and issuing it here gives the reads
        a full token of cover instead of the few layers a prefetch stride buys.
        """
        recs = self._flow_recs
        if not recs:
            return
        self._flow_recs = []
        hidden = self._flow_hidden
        self._flow_hidden = None
        cache = self.cache
        batch: list[tuple[str, list[int]]] = []
        for index, ind2, valid in recs:
            layer = self.layers[index]
            inds = np.asarray(ind2)
            ids = [int(e) for e in np.unique(inds)]
            layer.last_ids = ids
            cache.route_total += len(ids)
            # Slots the router asked for vs slots that had a resident expert:
            # this is flow's whole quality story, so keep it on the same
            # counters PRUNE reports through (`pruned_frac`).
            vnp = np.asarray(valid)
            cache.demand_slots += int(inds.size)
            cache.pruned_slots += int(inds.size - vnp.sum())
            self.note_demand(layer.layer_key, ids)
            batch.append((layer.layer_key, ids))
        if batch:
            cache.prefetch_many(batch)
        if self.sidecar is not None:
            if hidden is not None:
                try:
                    if hidden.dtype != mx.float32:
                        hidden = hidden.astype(mx.float32)
                    self.sidecar.note_hidden(np.asarray(hidden))
                except Exception:
                    pass
            self.sidecar.end_token(cache)
            self._token_open = False

    def flow_drop(self) -> None:
        """Forget held routing: a prefill replaces the state it came from."""
        self._flow_recs = []
        self._flow_hidden = None
        self._flow_tokens = 0

    def end_decode_token(self, hidden=None) -> None:
        """Close a decode token for the sidecar."""
        if self.sidecar is None or not self._token_open:
            return
        self._token_open = False
        if hidden is not None:
            try:
                # bfloat16 has no numpy equivalent, so cast on device first.
                if hidden.dtype != mx.float32:
                    hidden = hidden.astype(mx.float32)
                self.sidecar.note_hidden(np.asarray(hidden))
            except Exception:
                pass
        mode = self.sidecar.end_token(self.cache)
        # Breadcrumb every 64 tokens - only when sidecar debug is on.
        if getattr(self, "_sidecar_breadcrumb", 0) % 64 == 0:
            from .sidecar.slot import _log

            _log(f"decode-token mode={mode}")
        self._sidecar_breadcrumb = getattr(self, "_sidecar_breadcrumb", 0) + 1

    def _token_boundary(self, positions: int = 1) -> None:
        """Measure decode throughput per window; used by the auto governor.

        `positions` is how many token positions the pass just covered, which is
        1 for ordinary decode and up to LOOKUP_TOKENS+1 under lookup decode.
        Counting passes instead would stretch every window by that factor: a
        96-token reply is only ~22 passes at a 4.3 tok/pass accept rate, so the
        governor would never reach PREDICT_MIN_TOKENS inside a normal reply and
        would sit in its probe, spending half of it in the losing state. It
        slightly overstates the rate (rejected positions are counted), but the
        accept rate does not depend on prediction, so both sides are inflated
        equally and the comparison is unaffected.

        Prefetching is only free when the read is the bottleneck. On unified
        memory the reader threads and the GPU share memory bandwidth, so
        overlapping reads with compute is a *trade*, not a pure win: when
        experts already arrive at near-RAM speed (a small model, or a working
        set the drive caches), or when the drive is already saturated, moving
        reads into the compute window contends with the kernels instead of
        hiding behind them. Both regimes were measured here - hit rate 0.73 to
        0.90 with disk wait halved, and still 7% slower overall - while a
        genuinely disk-bound model gains 44% from the identical change.

        No single static signal separates those cases (cost per miss puts the
        bandwidth-saturated case on the wrong side of any threshold), so don't
        model it: alternate prediction on and off in short windows, pool
        tokens/s each way until there is enough evidence to call it, keep the
        winner for a long hold, and re-check occasionally in case conditions
        drift.

        Window size and evidence threshold are separate knobs on purpose.
        Windows stay short so the two states interleave and therefore sample the
        same content (tool-call JSON and prose decode at visibly different
        rates); the decision waits for PREDICT_MIN_TOKENS on *each* side because
        agent traffic's window-to-window spread is wider than the effect being
        measured. Judging one window against one window made this flip on nearly
        every probe -- nine reversals in four minutes on real traffic, over
        samples ranging 9.2 to 19.2 tok/s -- which left it running in the losing
        configuration a good fraction of the time.
        """
        now = time.perf_counter()
        prev, self._last_token_t = self._last_token_t, now
        if prev is None:
            return
        dt = now - prev
        if dt > 1.0:
            # A gap between requests, not a decode step.
            return
        self._win_time += dt
        self._win_tokens += max(1, positions)
        if self._win_tokens < max(4, config.PREDICT_WINDOW):
            return
        tokens, seconds = self._win_tokens, self._win_time
        self._win_tokens, self._win_time = 0, 0.0
        if seconds > 0:
            self._window_done(tokens, seconds)

    @staticmethod
    def _mean_and_var(acc: list[float]) -> tuple[float, float, float]:
        n, total, total_sq = acc[0], acc[1], acc[2]
        if n < 2:
            return (total / n if n else 0.0), float("inf"), n
        mean = total / n
        var = max(0.0, (total_sq - total * mean) / (n - 1))
        return mean, var, n

    def _settle(self, on: bool, reason: str, hold: int) -> None:
        self.predict_depth = self._configured_depth if on else 0
        self._probing = False
        self._hold_left = hold
        if on != self._decision:
            self._decision = on
            print(f"[paged-moe] route prediction {'on' if on else 'off'}: {reason}",
                  flush=True)

    def _window_done(self, tokens: float, seconds: float) -> None:
        if not self._auto:
            return  # an explicit EXPERT_STREAM_PREDICT is not up for revision
        active = self.predict_depth > 0
        if not self._probing:
            self._hold_left -= 1
            if self._hold_left <= 0:
                # Re-probe, but keep half the evidence. Conditions drift slowly
                # (model, cache pressure, prompt length), so resuming from what
                # was already learned reaches the next decision sooner and
                # without discarding a well-sampled result.
                for totals in self._acc.values():
                    for i in range(len(totals)):
                        totals[i] *= 0.5
                self._probing = True
                self.predict_depth = self._configured_depth
            return

        rate = tokens / seconds
        acc = self._acc[active]
        acc[0] += 1.0
        acc[1] += rate
        acc[2] += rate * rate
        acc[3] += tokens
        # Alternate every window so the two states see the same mix of work.
        self.predict_depth = 0 if active else self._configured_depth

        floor = max(config.PREDICT_WINDOW, config.PREDICT_MIN_TOKENS)
        on_tokens, off_tokens = self._acc[True][3], self._acc[False][3]
        if min(on_tokens, off_tokens) < floor:
            return

        on_tps, on_var, on_n = self._mean_and_var(self._acc[True])
        off_tps, off_var, off_n = self._mean_and_var(self._acc[False])
        if on_n < 2 or off_n < 2:
            return

        # A fixed percentage margin is not enough on its own. Agent decode rates
        # swing by tens of percent window to window (tool-call JSON vs prose, cache
        # state, prefill landing mid-window), so at realistic sample sizes the
        # error bar on the comparison is wider than the margin and a tie resolves
        # essentially at random -- which is what made the first version oscillate.
        # Require the gap to clear both the margin and the measured error bar, and
        # otherwise keep gathering rather than guessing.
        stderr = ((on_var / on_n) + (off_var / off_n)) ** 0.5
        gap = on_tps - off_tps
        need = max((config.PREDICT_MARGIN - 1.0) * off_tps, config.PREDICT_Z * stderr)
        sample = f"{on_tokens:.0f}/{off_tokens:.0f} tokens, +/-{config.PREDICT_Z * stderr:.1f} tok/s"

        if gap > need:
            self._settle(True, f"measured {on_tps:.1f} tok/s with, {off_tps:.1f} without "
                               f"({sample})", config.PREDICT_HOLD_WINDOWS)
            return
        if -gap > need:
            self._settle(False, f"measured {on_tps:.1f} tok/s with, {off_tps:.1f} without "
                                f"({sample}); EXPERT_STREAM_PREDICT=1 to force on",
                         config.PREDICT_HOLD_WINDOWS)
            return

        # Inconclusive. Give it room, but not unbounded room: if the difference is
        # still lost in the noise after a few times the minimum sample, there is
        # little to win either way, so take the cheaper side and hold much longer.
        if min(on_tokens, off_tokens) < floor * config.PREDICT_MAX_PROBE:
            return
        self._settle(False, f"no measurable difference ({on_tps:.1f} vs {off_tps:.1f} tok/s "
                            f"over {sample}); EXPERT_STREAM_PREDICT=1 to force on",
                     config.PREDICT_HOLD_WINDOWS * 4)

    def register(self, layer: StreamedSwitchGLU) -> int:
        self.layers.append(layer)
        return len(self.layers) - 1

    # --------------------------------------------------------- prediction

    def _ratio(self, src: int, tgt: int):
        """Per-channel rescale taking layer `src`'s MoE input to layer `tgt`'s.

        Both are RMSNorm outputs: x_src = w_src * h_hat, and the router of tgt
        expects w_tgt * h_hat. Since h_hat barely rotates across one block, and
        a linear router's top-k is invariant to positive scaling, multiplying by
        w_tgt / w_src removes the only systematic mismatch (the per-channel
        gains) for the cost of one elementwise multiply.
        """
        key = (src, tgt)
        if key in self._ratios:
            return self._ratios[key]
        ratio = None
        if self.renorm:
            ps, pt = self.layers[src].pred, self.layers[tgt].pred
            ws = ps.norm_weight if ps is not None else None
            wt = pt.norm_weight if pt is not None else None
            if ws is not None and wt is not None and ws.shape == wt.shape:
                mag = mx.maximum(mx.abs(ws), 1e-3)
                sign = mx.where(ws < 0, -1.0, 1.0)
                ratio = (wt / (sign * mag)).astype(ws.dtype)
                mx.eval(ratio)
        self._ratios[key] = ratio
        return ratio

    def _distances(self, src: int, n: int) -> tuple[int, ...]:
        """Which layer distances to predict from `src`.

        The near band is the shipped behaviour. The far entry is the payoff
        from measured router agreement over every (source, target) pair: the
        decay is NOT a function of
        distance but of SOURCE DEPTH: on Qwen3-235B, precision among cache
        misses from layer 0 is gone by distance 4 (0.28), while from layer 24
        it is still 0.73 at distance 32 - against a net-win bar of 0.5. Early
        layers rewrite the residual stream wholesale, later layers refine it,
        so a router's view of a deep hidden state stops going stale.

        The far band is deliberately ONE target per source, not a widened
        window. Widening uniformly was already measured slower (see the
        cap-4/slack-2 note below: 22.6 GB/token of speculative waste), because
        a wide burst sits in the disk queue ahead of real demand misses. One
        far target per source costs the same order of bytes as a single extra
        depth step while buying FAR_LEAD layers of cover instead of one, and
        every layer still gets predicted once at long lead as src walks down.
        """
        if self.predict_depth <= 0:
            # Prediction off (the governor's OFF phase sets depth 0). The far
            # target must not sneak a read in behind that.
            return ()
        near = range(2 + self.lead, self.predict_depth + self.lead + 2)
        far = config.PREDICT_FAR_LEAD
        if far <= 0 or src < int(config.PREDICT_FAR_FROM * n):
            return tuple(d for d in near if src + d < n)
        # Skip a far target the near band already covers, so the extra read is
        # never a duplicate of a prediction the token was making anyway.
        out = [d for d in near if src + d < n]
        if far not in out and src + far < n:
            out.append(far)
        return tuple(out)

    def _predict_graph(self, src: int, x: mx.array, top_k: int):
        """Build the prediction graph for layers past `src`. Returns
        (merged lazy mx array, targets, lengths, weight_kind) - no sync.

        The caller decides when to materialize `merged`. During decode that
        happens inside the *next* layer's router eval (see __call__), so the
        prediction rides a sync the token was paying anyway instead of adding
        its own - measured ~0.35 ms x ~31 strides = ~10 ms/token on
        Qwen3-235B. Targets start one layer further out to compensate for
        ids becoming available one layer later.
        """
        n = len(self.layers)
        parts: list[mx.array] = []
        targets: list[int] = []
        lengths: list[int] = []
        # Per-target: None = ids only; "logits" = Linear (softmax on host);
        # "scores" = MoEGate mixture weights (unit-mass on host).
        weight_kind: list[str | None] = []
        prune = float(config.PRUNE)
        for d in self._distances(src, n):
            tgt = src + d
            if tgt >= n:
                continue
            pred = self.layers[tgt].pred
            if pred is None or pred.gate is None:
                continue
            ratio = self._ratio(src, tgt)
            xin = x if ratio is None else x * ratio
            out = pred.gate(xin)
            kind: str | None = None
            if isinstance(out, (tuple, list)):
                # Router modules of the DeepSeek/GLM family already return the
                # selected indices (grouped top-k, correction bias and all),
                # plus the mixture scores. When any weight-based selector is
                # on, carry the scores so the host can apply the same filter
                # the demand path will - without them prediction over-fetches
                # every slot the gate returns.
                inds = out[0]
                scores = out[1] if len(out) > 1 else None
                approx = (
                    prune > 0.0
                    or config.ROUTE_TOP_K > 0
                    or config.ROUTE_TOP_P > 0.0
                )
                if approx and scores is not None:
                    flat = mx.concatenate(
                        [
                            inds.reshape(-1).astype(mx.float32),
                            scores.reshape(-1).astype(mx.float32),
                        ]
                    )
                    kind = "scores"
                else:
                    flat = inds.reshape(-1).astype(mx.float32)
            else:
                # nn.Linear routers (Qwen3-MoE / Qwen3-Next) return logits, and
                # the block would softmax + argpartition them. Softmax is
                # monotonic, so top-k of the logits is the same set. `slack`
                # widens the guess a little: the experts just outside the cut
                # are the ones a one-block-old hidden state most often gets
                # wrong, and they are the cheapest to be right about.
                #
                # With decode pruning on, the demand path is about to drop the
                # weak half of the top-k anyway (see __call__), so predicting
                # the full top-k+slack reads experts the token will never use.
                # Measured on Qwen3-235B (prune 0.5): prediction issued 930
                # experts/token against ~350 of post-prune demand - 0.57 GB of
                # the 1.2 GB read per token was thrown away unused, and those
                # junk reads sat in the disk queue ahead of real demand misses.
                # So: no slack when pruning, and carry the top-k logits along
                # to apply the same weight threshold the demand path will.
                #
                # A TOP_K cap makes the same argument absolutely rather than on
                # average: no layer can demand more than `cap` experts, so
                # `top_k` here is already the cap and slack would predict slots
                # that are unreachable by construction. Measured cap-4 with
                # slack 2: 34,776 experts issued/token at precision 0.58
                # against 23,688 of demand - 22.6 GB of speculative waste, and
                # cap-4 came out *slower* than no cap at all despite computing
                # half the experts.
                approx = (
                    prune > 0.0
                    or config.ROUTE_TOP_K > 0
                    or config.ROUTE_TOP_P > 0.0
                )
                slack = 0 if approx else self.slack
                m = min(max(top_k, 1) + slack, out.shape[-1])
                inds = mx.argpartition(out, kth=-m, axis=-1)[..., -m:]
                if 0.0 < prune <= 1.0 or 0.0 < config.ROUTE_TOP_P < 1.0:
                    # Weight-based tests need the logits, not just the ids, so
                    # the host can apply the same filter the demand path will.
                    vals = mx.take_along_axis(out, inds, axis=-1)
                    # ids and logits ride the same sync (ids are small ints,
                    # exactly representable in f32).
                    flat = mx.concatenate(
                        [inds.reshape(-1).astype(mx.float32), vals.reshape(-1)]
                    )
                    kind = "logits"
                else:
                    flat = inds.reshape(-1).astype(mx.float32)
            parts.append(flat)
            targets.append(tgt)
            lengths.append(flat.size)
            weight_kind.append(kind)

        if not parts:
            return None
        return mx.concatenate(parts), targets, lengths, weight_kind

    def _decode_predictions(self, merged: np.ndarray, targets, lengths, weight_kind):
        """Host side of _predict_graph: threshold + dedupe the merged sync."""
        prune = float(config.PRUNE)
        cap = int(config.ROUTE_TOP_K)
        out_ids: dict[int, list[int]] = {}
        off = 0
        for tgt, length, kind in zip(targets, lengths, weight_kind):
            chunk = merged[off : off + length]
            off += length
            if kind is not None:
                k = length // 2
                ids = chunk[:k].astype(np.int64)
                vals = chunk[k:]
                if kind == "logits":
                    # Softmax over the selected experts' logits - same math as
                    # the Linear demand path.
                    w = np.exp(vals - vals.max())
                    w /= w.sum()
                else:
                    # MoEGate scores already encode the mixture (possibly
                    # scaled by routed_scaling_factor); unit-mass them.
                    w = _unit_mass(vals[None, :])[0]
                keep = np.ones(w.shape, dtype=bool)
                if prune > 0.0:
                    # Softer threshold than the demand path: these weights come
                    # from a one-block-old hidden state, so experts near the
                    # cut flip either way. Predicting with the demand threshold
                    # measured coverage 0.73 (vs 0.96 unpruned) - every
                    # borderline expert wrongly dropped is a blocking miss a
                    # few layers later, which costs far more than the ~10 MB
                    # read it saved.
                    keep &= w >= prune * config.PREDICT_PRUNE_SOFT * w.max()
                if 0.0 < config.ROUTE_TOP_P < 1.0:
                    # Same softening argument: aim at a little more mass than
                    # the demand path will keep, so a borderline expert is
                    # prefetched rather than turned into a blocking miss.
                    p = min(
                        1.0,
                        config.ROUTE_TOP_P
                        + (1.0 - config.ROUTE_TOP_P)
                        * (1.0 - config.PREDICT_PRUNE_SOFT),
                    )
                    keep &= _nucleus_mask(w[None, :], p)[0]
                if 0 < cap < w.shape[0]:
                    # Soften the cap by one when possible so a near-miss at the
                    # cut still gets prefetched.
                    soft_cap = min(w.shape[0], cap + (1 if cap < w.shape[0] else 0))
                    top = np.argpartition(w, w.shape[0] - soft_cap)[w.shape[0] - soft_cap :]
                    cap_mask = np.zeros(w.shape, dtype=bool)
                    cap_mask[top] = True
                    keep &= cap_mask
                ids = ids[keep]
            else:
                ids = chunk.astype(np.int64)
            # np.unique + tolist rather than a Python set: this runs once per
            # layer per token, so it is on the critical path, and it hands
            # prefetch what it wants anyway (sorted, deduped, native ints).
            out_ids[tgt] = np.unique(ids).tolist()
        return out_ids

    # --------------------------------------------------------- scheduling

    def drop_pending(self) -> None:
        self._pending = None

    def pending_sync(self) -> tuple:
        """Lazy prediction arrays for the next layer's router eval to carry.

        Materializing the prediction inside a sync the token already pays
        (the per-layer router eval) is what makes deferred prediction free;
        this returns () when there is nothing pending.
        """
        if self._pending is None:
            return ()
        return (self._pending[0],)

    def flush(self) -> None:
        """Issue the reads for a prediction whose sync has landed.

        Called right after the router eval that carried the merged array. If
        for any reason it was not carried (a layer without the prune block),
        np.asarray below performs the sync itself - same behavior as the old
        inline path, just paid here.
        """
        pending = self._pending
        if pending is None:
            return
        self._pending = None
        merged_lazy, targets, lengths, weight_kind = pending
        t0 = time.perf_counter()
        try:
            predicted = self._decode_predictions(
                np.asarray(merged_lazy), targets, lengths, weight_kind
            )
        except Exception as e:  # pragma: no cover - arch safety valve
            print(f"[paged-moe] route prediction disabled: {e}", flush=True)
            self.predict_depth = 0
            predicted = {}
        batch = []
        for tgt, ids in predicted.items():
            self.layers[tgt].predicted.update(ids)
            batch.append((self.layers[tgt].layer_key, ids))
        if batch:
            self.cache.prefetch_many(batch)
        self.cache.spec_s += time.perf_counter() - t0

    def speculate(
        self,
        current_index: int,
        x: mx.array | None = None,
        top_k: int = 0,
        positions: int = 1,
    ):
        n = len(self.layers)
        if n < 2:
            return

        if self._auto and current_index == 0:
            self._token_boundary(positions)

        # Predict on a stride, not every layer. Predicting `depth` layers ahead
        # at every layer means each layer is predicted `depth` times, so the
        # marginal predictions cost a GPU sync and a pass of host work to
        # mostly re-derive ids already queued. On a stride each layer is
        # predicted exactly once, with a lead of 1..depth layers - which is all
        # the lead time the reads need - for a third of the overhead. That
        # matters because when decode is *not* disk-bound there is no wait to
        # hide, and speculation has to be nearly free rather than merely useful.
        #
        # The graph is built here but *not* synced: the next layer's router
        # eval materializes it (see pending_sync/flush), and the reads are
        # queued immediately after. Targets are shifted one layer further out
        # so the lead time relative to the issuing layer is unchanged.
        if (
            self.predict_depth > 0
            and x is not None
            and current_index % self.predict_depth == 0
        ):
            t0 = time.perf_counter()
            try:
                self._pending = self._predict_graph(current_index, x, top_k)
            except Exception as e:  # pragma: no cover - arch safety valve
                # A router shape we don't understand must never break the
                # forward pass; fall back to the previous-token heuristic.
                print(f"[paged-moe] route prediction disabled: {e}", flush=True)
                self.predict_depth = 0
                self._pending = None
            self.cache.spec_s += time.perf_counter() - t0

        if self.depth <= 0:
            return
        for d in range(1, self.depth + 1):
            nxt_i = current_index + d
            if nxt_i >= n:
                # Past the last layer: wrap to the front for the *next* token.
                # Those layers run again within a few milliseconds and their
                # ids are usually still resident, so this is close to free -
                # it only does work in the case that used to be a guaranteed
                # stall, the first layers of a token after an eviction sweep.
                nxt_i -= n
                if nxt_i >= current_index:
                    break
            nxt = self.layers[nxt_i]
            if nxt.last_ids:
                self.cache.prefetch(nxt.layer_key, nxt.last_ids)


# --------------------------------------------------------------------------
# Model surgery
# --------------------------------------------------------------------------


def _nucleus_mask(w: np.ndarray, p: float) -> np.ndarray:
    """Smallest set of experts per row whose weights sum to >= p.

    `w` is [N, K] mixture weights (rows sum to 1). Always keeps at least the
    strongest expert, so a row can never end up empty no matter how flat or
    how peaked it is.
    """
    order = np.argsort(-w, axis=-1)
    cum = np.cumsum(np.take_along_axis(w, order, axis=-1), axis=-1)
    # Rank of the expert that first reaches p; keep everything up to and
    # including it. `< p` counts the ones that fall short, hence the +1.
    n_keep = np.minimum((cum < p).sum(axis=-1) + 1, w.shape[-1])
    rank = np.empty(w.shape, dtype=np.int32)
    np.put_along_axis(
        rank, order, np.broadcast_to(np.arange(w.shape[-1]), w.shape), axis=-1
    )
    return rank < n_keep[:, None]


def _unit_mass(w: np.ndarray) -> np.ndarray:
    """Normalize mixture weights so each row sums to 1.

    Linear routers already produce a unit-mass softmax. MoEGate (GLM /
    DeepSeek) returns scores that sum to `routed_scaling_factor` (e.g. 2.5)
    after norm_topk_prob. Selectors and WAIT_ABOVE are defined in *share of
    the mixture*, so both families have to speak the same units.
    """
    mass = w.sum(axis=-1, keepdims=True)
    return w / np.maximum(mass, 1e-6)


def _align_scores_to_indices(
    gate_inds: np.ndarray, gate_scores: np.ndarray, ind2: np.ndarray
) -> np.ndarray:
    """Map MoEGate (inds, scores) onto the slot order `switch_mlp` received.

    The parent block does `inds, scores = gate(x); y = switch_mlp(x, inds)`,
    so our `indices` are those inds. Re-running the gate for the weight
    recomputation can permute the top-k (argpartition is not ordered), so we
    align by expert id rather than by position.
    """
    N, K = ind2.shape
    out = np.empty((N, K), dtype=np.float32)
    for n in range(N):
        by_id = {
            int(e): float(s) for e, s in zip(gate_inds[n], gate_scores[n])
        }
        for k in range(K):
            out[n, k] = by_id.get(int(ind2[n, k]), 0.0)
    return out


def _selector_mask(
    w: np.ndarray,
    prune: float = 0.0,
    cap: int = 0,
    top_p: float = 0.0,
) -> np.ndarray | None:
    """Boolean keep mask from unit-mass mixture weights. None = keep all."""
    N, K = w.shape
    keep = None
    if 0.0 < prune <= 1.0:
        keep = w >= prune * w.max(axis=-1, keepdims=True)
    if 0.0 < top_p < 1.0:
        p_mask = _nucleus_mask(w, top_p)
        keep = p_mask if keep is None else (keep & p_mask)
    if 0 < cap < K:
        top = np.argpartition(w, K - cap, axis=-1)[:, K - cap :]
        cap_mask = np.zeros(w.shape, dtype=bool)
        np.put_along_axis(cap_mask, top, True, axis=-1)
        keep = cap_mask if keep is None else (keep & cap_mask)
    return keep


def _route_weights_from_gate(
    gate,
    x_flat: mx.array,
    ind2: mx.array,
    pending: tuple = (),
    tapped=None,
) -> np.ndarray | None:
    """Mixture weights for `ind2`, from either router family.

    * `nn.Linear` (Qwen3-MoE / Qwen3-Next): gate returns logits; softmax over
      the selected logits is the parent block's normalized top-k scores.
    * `MoEGate` (GLM / DeepSeek): gate returns `(inds, scores)` already; those
      scores are in `ind2` order when they come from the block itself, and are
      aligned by expert id when the router had to be re-run.

    `tapped` is the output the parent block's own gate call produced for this
    same activation (see _tap_router). Reusing it is not just cheaper: the
    re-run path has to align by expert id because argpartition does not order
    ties, and the block's scores are already the ones it will weigh with.

    Returns None if the gate shape is unrecognized (caller leaves routing
    exact). Syncs `ind2` + weights, and any deferred prediction in `pending`.
    """
    K = ind2.shape[-1]
    out = tapped if tapped is not None else gate(x_flat)
    if isinstance(out, (tuple, list)):
        if len(out) < 2:
            return None
        g_inds, g_scores = out[0], out[1]
        if not isinstance(g_inds, mx.array) or not isinstance(g_scores, mx.array):
            return None
        if tapped is not None:
            # Straight from the block: same order as ind2, no alignment needed.
            g_scores = g_scores.reshape(-1, K)
            mx.eval(ind2, g_scores, *pending)
            return _unit_mass(np.asarray(g_scores).astype(np.float32))
        mx.eval(ind2, g_inds, g_scores, *pending)
        w = _align_scores_to_indices(
            np.asarray(g_inds), np.asarray(g_scores), np.asarray(ind2)
        )
        return _unit_mass(w)
    if isinstance(out, mx.array) and out.ndim >= 2:
        logits = out if out.ndim == 2 else out.reshape(-1, out.shape[-1])
        sel = mx.take_along_axis(logits, ind2, axis=-1)
        w = mx.softmax(sel.astype(mx.float32), axis=-1)
        mx.eval(ind2, w, *pending)
        return np.asarray(w)
    return None


def _resolve_parent(model, path: str):
    """Walk 'model.layers.7.mlp' style paths to (parent_object, last_name)."""
    parts = path.split(".")
    obj = model
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj, parts[-1]


def _find_router(model, path: str) -> _RoutePredictor | None:
    """The router and pre-MoE norm weight for the MoE block owning `path`.

    Every mlx-lm MoE block keeps its router next to the SwitchGLU it feeds
    ('.gate', or '.router' on a few architectures) and the decoder layer
    normalizes with '.post_attention_layernorm' before calling the block. Both
    are backbone tensors, so they are already resident and free to re-run.
    Returns None (prediction off for this layer) if the layout is unfamiliar.
    """
    try:
        block, _ = _resolve_parent(model, path)
        gate = getattr(block, "gate", None)
        if gate is None:
            gate = getattr(block, "router", None)
        if gate is None or not callable(gate):
            return None
        decoder, _ = _resolve_parent(model, path.rsplit(".", 1)[0])
        norm = getattr(decoder, "post_attention_layernorm", None)
        return _RoutePredictor(gate, getattr(norm, "weight", None))
    except (AttributeError, IndexError, KeyError, ValueError):
        return None


def find_switch_glus(model) -> list[tuple[str, SwitchGLU]]:
    """All SwitchGLU modules with their tree paths (== checkpoint prefixes)."""
    found = []
    for path, module in model.named_modules():
        if isinstance(module, SwitchGLU):
            found.append((path, module))

    def layer_num(path):
        for part in path.split("."):
            if part.isdigit():
                return int(part)
        return 0

    found.sort(key=lambda kv: layer_num(kv[0]))
    return found


def _layer_layout(
    path: str, tensor_index: dict[str, TensorLoc]
) -> dict[str, ExpertLocator]:
    """Locate one MoE layer's expert tensors in the checkpoint.

    Handles both on-disk layouts (no conversion needed for either):
      stacked:     "<path>.gate_proj.weight" with shape [E, ...]
      per-expert:  "<parent>.experts.<e>.gate_proj.weight" (HF-style; mlx-lm's
                   sanitize() stacks these at load time, we read them directly)
    """
    components: dict[str, ExpertLocator] = {}

    for proj in _PROJS:
        for comp in ("weight", "scales", "biases", "bias"):
            name = f"{path}.{proj}.{comp}"
            if name in tensor_index:
                components[f"{proj}.{comp}"] = StackedLocator(tensor_index[name])
    if components:
        return components

    parent = path.rsplit(".", 1)[0]
    pat = re.compile(
        re.escape(parent) + r"\.experts\.(\d+)\.(\w+)\.(weight|scales|biases|bias)$"
    )
    grouped: dict[str, dict[int, TensorLoc]] = {}
    for name, loc in tensor_index.items():
        m = pat.match(name)
        if m and m.group(2) in _PROJS:
            grouped.setdefault(f"{m.group(2)}.{m.group(3)}", {})[
                int(m.group(1))
            ] = loc
    for comp_name, per_expert in grouped.items():
        components[comp_name] = DirectLocator(per_expert)
    return components


def patch_model(
    model,
    cache: ExpertCache,
    tensor_index: dict[str, TensorLoc],
    prefetch_depth: int,
    predict_mode: str | None = None,
    model_path: str | None = None,
) -> PrefetchRing:
    """Replace every SwitchGLU with a StreamedSwitchGLU. Returns the ring.

    predict_mode overrides config.PREDICT when given (the loader forces "on"
    for models whose expert mass dwarfs the cache: those are disk-bound by
    construction, so the governor's probe would just spend minutes at reduced
    speed re-discovering that).

    When EXPERT_STREAM_SIDECAR=1, attaches an online expert-prefetch sidecar
    to the ring. When 0 (default), the ring's sidecar stays None and the
    decode path pays only a None-check.
    """
    mode = predict_mode or config.PREDICT
    glus = find_switch_glus(model)
    if not glus:
        raise ValueError(
            "no SwitchGLU modules found - this model has no MoE layers "
            "mlx-lm knows how to stream (dense model, or unsupported arch)"
        )

    ring = PrefetchRing(
        cache,
        prefetch_depth,
        predict_depth=0 if mode == "off" else config.PREDICT_DEPTH,
        slack=config.PREDICT_SLACK,
        renorm=config.PREDICT_RENORM,
        predict_mode=mode,
        lead=config.PREDICT_LEAD,
    )

    for path, module in glus:
        components = _layer_layout(path, tensor_index)
        if not components:
            raise ValueError(f"checkpoint has no tensors for MoE module {path}")
        cache.register_layer(path, components)

        predictor = _find_router(model, path)
        if predictor is not None:
            _tap_router(predictor.gate)
        streamed = StreamedSwitchGLU(path, module, cache, ring, predictor)
        parent, attr = _resolve_parent(model, path)
        if attr.isdigit():
            parent[int(attr)] = streamed
        else:
            setattr(parent, attr, streamed)

    if config.SIDECAR:
        # Import only when enabled so a disabled model never loads the learner.
        from .sidecar import ExpertSidecar

        n_experts = max_expert_count(cache)
        if n_experts <= 0:
            from .sidecar.slot import _log

            _log("enabled but could not infer expert count - leaving off")
        else:
            layer_keys = [glu.layer_key for glu in ring.layers]
            ring.attach_sidecar(
                ExpertSidecar(
                    model_path or "unknown",
                    layer_keys,
                    n_experts,
                )
            )

    return ring


def max_expert_count(cache: ExpertCache) -> int:
    """Routed experts per layer, from the registered layouts (0 if unknown)."""
    n_experts = 0
    for comps in cache._layouts.values():
        locator = next(iter(comps.values()))
        if isinstance(locator, StackedLocator):
            n_experts = max(n_experts, int(locator.stacked.shape[0]))
        elif isinstance(locator, DirectLocator):
            n_experts = max(n_experts, len(locator.per_expert))
    return n_experts


def expert_tensor_names(
    model, tensor_index: dict[str, TensorLoc]
) -> set[str]:
    """Names of all checkpoint tensors holding routed-expert weights,
    in either layout (stacked under the SwitchGLU path, or HF-style
    per-expert tensors under '<parent>.experts.<e>.')."""
    prefixes = []
    for path, _ in find_switch_glus(model):
        prefixes.append(f"{path}.")
        prefixes.append(f"{path.rsplit('.', 1)[0]}.experts.")
    prefixes = tuple(prefixes)
    return {name for name in tensor_index if name.startswith(prefixes)}
