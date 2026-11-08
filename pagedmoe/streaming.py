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
1. `indices` (which experts each token wants) is forced to CPU before a read.
   That sync is what a miss costs. When the slab already holds every expert
   (config.STITCH), nothing is read and the gather stays on the GPU, so a
   decode token is one graph.
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

import contextlib
import re
import time

import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU

from . import activity, config, fidelity, gpu_crumbs, prune_curve, route_log, stage_gentle
from .cache import ExpertCache
from .checkpoint_keys import lookup_tensor, tensor_key_candidates
from .safetensors_index import DirectLocator, ExpertLocator, StackedLocator, TensorLoc

_PROJS = ("gate_proj", "up_proj", "down_proj")

# At most this many tokens counts as "decode" (cache + prefetch path);
# anything bigger is treated as prefill (grouped streaming path).
#
# 20, not 4: speculative decoding verifies num_draft_tokens+1 positions in one
# forward call, and that batch must stay on the decode path - it wants the
# full cache, route prediction, and per-expert compute on resident weights.
# (This is also the whole reason speculative decoding pays on a streamed MoE:
# the batch reads each layer's expert *union* once - consecutive tokens share
# roughly half their experts - and reads every weight once for k tokens
# instead of k times.) A long verbatim copy drafts 16 tokens (17 positions).
# Real prefill chunks are hundreds of tokens minimum, so there is no
# ambiguity in between.
_DECODE_MAX_TOKENS = 20


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

    `score_bias` is set for logits routers whose block selects by
    sigmoid(logits) + correction bias and mixes with the normalized sigmoid
    (MiniMax). None means the block softmaxes the logits.
    """

    __slots__ = ("gate", "norm_weight", "score_bias")

    def __init__(self, gate, norm_weight, score_bias=None):
        self.gate = gate
        self.norm_weight = norm_weight
        self.score_bias = score_bias


# Families whose MoE block keeps an nn.Linear router plus a block-level
# e_score_correction_bias and always scores with sigmoid. kimi_linear has the
# same layout but picks softmax or sigmoid from config, so it is not listed.
_SIGMOID_BIAS_ROUTERS = frozenset({"minimax", "minimax_m3"})


def _router_select(pred, out):
    """What the block ranks a logits router's experts by, for its top-k."""
    if pred.score_bias is None:
        return out
    return mx.sigmoid(out.astype(mx.float32)) + pred.score_bias


def _pred_input(pred, x):
    """Router input as the block would feed it (MiniMax casts to fp32)."""
    if pred.score_bias is not None:
        x = x.astype(mx.float32)
    return _gate_input(pred.gate, x)


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
#
# Some blocks hand the gate a cast copy (`self.gate(x.astype(mx.float32))` on
# qwen4_exp), so id(x) never matches there. For those the record still belongs
# to this forward when the gate matches and the router saw the same rows: the
# engine's own gate calls (route prediction, re-runs) go through
# _untapped_call and never write here, so the last recorded call of a gate is
# always its block's real one.
_LAST_ROUTE: tuple | None = None
_TAPPED_ROUTERS: dict[type, type] = {}
_TAP_OFF = False


def _untapped_call(gate, x):
    """Run a router without recording it (engine-side calls only)."""
    global _TAP_OFF
    prev = _TAP_OFF
    _TAP_OFF = True
    try:
        return gate(x)
    finally:
        _TAP_OFF = prev


def _rows(shape) -> int:
    n = 1
    for d in shape[:-1]:
        n *= int(d)
    return n


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
            if not _TAP_OFF:
                global _LAST_ROUTE
                _LAST_ROUTE = (id(self), id(x), out, _rows(x.shape))
            return out

        tapped = type(f"Tapped{cls.__name__}", (cls,), {"__call__": __call__})
        _TAPPED_ROUTERS[cls] = tapped
    try:
        gate.__class__ = tapped
    except TypeError:
        pass  # exotic router class; the re-run path still works


def _unit_mass_mx(pred, x, ind2):
    """Mixture weights for the routed slots, lazy, same definition as the
    per-layer selector. None if this router was not tapped."""
    if pred is None or pred.gate is None:
        return None
    out = _tapped_route(pred.gate, x)
    if out is None:
        return None
    k = ind2.shape[-1]
    if isinstance(out, (tuple, list)):
        if len(out) < 2 or not isinstance(out[1], mx.array):
            return None
        w = out[1].reshape(-1, k).astype(mx.float32)
        return w / mx.maximum(mx.sum(w, axis=-1, keepdims=True), 1e-8)
    if isinstance(out, mx.array) and out.ndim >= 2:
        logits = out if out.ndim == 2 else out.reshape(-1, out.shape[-1])
        sel = mx.take_along_axis(logits, ind2, axis=-1).astype(mx.float32)
        if pred.score_bias is not None:
            w = mx.sigmoid(sel)
            return w / mx.maximum(mx.sum(w, axis=-1, keepdims=True), 1e-8)
        return mx.softmax(sel, axis=-1)
    return None


def _active_prune_wait() -> tuple[float, float]:
    """Thresholds the per-layer path would start from, before the prune head.

    The head only raises them. Using the lower value makes a closed miss a
    superset of the experts the head would still have waited for, so a live
    head can force a replay and cannot silently drop one.
    """
    if _EXACT_STEP or fidelity.rescuing():
        return 0.0, 0.0
    if _DRAFT_STEP is not None:
        prune, wait = _DRAFT_STEP
    else:
        prune, wait = float(config.PRUNE), float(config.WAIT_ABOVE)
    if fidelity.strict() and _DRAFT_STEP is None:
        prune, wait = fidelity.adjust(prune, wait)
    return float(prune), float(wait)


def _nucleus_mx(w, p: float):
    """True where the slot is inside the smallest prefix of `w` that sums to p."""
    order = mx.argsort(-w, axis=-1)
    ranked = mx.take_along_axis(w, order, axis=-1)
    n_keep = mx.minimum(mx.sum(mx.cumsum(ranked, axis=-1) < p, axis=-1) + 1, w.shape[-1])
    rank = mx.argsort(order, axis=-1)
    return rank < n_keep[:, None]


def _closed_keep(pred, x, ind2, resident):
    """(compute, missing) masks matching the slider's per-layer keep set.

    compute: this row's expert output is part of the mixture.
    missing: the per-layer path would have blocked on a disk read for it.
    None, None when the router weights are not available; the caller then
    treats every non-resident slot as a miss.
    """
    w = _unit_mass_mx(pred, x, ind2)
    if w is None:
        return None, None
    prune, wait = _active_prune_wait()
    exact = _EXACT_STEP or fidelity.rescuing()
    cap = 0 if exact else int(config.ROUTE_TOP_K)
    top_p = 0.0 if exact else float(config.ROUTE_TOP_P)
    k = ind2.shape[-1]
    keep = w >= 0
    if prune > 0.0:
        top = mx.maximum(mx.max(w, axis=-1, keepdims=True), 1e-20)
        keep = keep & (w >= prune * top)
    if 0.0 < top_p < 1.0:
        keep = keep & _nucleus_mx(w, top_p)
    if 0 < cap < k:
        order = mx.argsort(-w, axis=-1)
        keep = keep & (mx.argsort(order, axis=-1) < cap)
    if config.KEEP_FREE:
        keep = keep | resident
    if wait > 0.0:
        keep = keep & (resident | (w >= wait))
    return keep & resident, keep & ~resident


def _route_rank_mx(pred, x, ind2):
    """A monotone score per routed slot as an mx.array, without a sync.

    Flow decode only ever *ranks* the router's own top-k, so any monotone
    transform of the mixture weight will do: MoEGate hands its scores back
    already, and for a logits router the selected logits order exactly the way
    a softmax (or sigmoid) over them would. Returns None when the router output
    was not captured, which sends the layer down the ordinary per-layer path.

    Also returns each slot's weight as a fraction of the row's strongest
    expert (the quantity PRUNE thresholds), still lazy.
    """
    if pred is None or pred.gate is None:
        return None
    out = _tapped_route(pred.gate, x)
    if out is None:
        return None
    K = ind2.shape[-1]
    if isinstance(out, (tuple, list)):
        if len(out) < 2 or not isinstance(out[1], mx.array):
            return None
        rank = out[1].reshape(-1, K).astype(mx.float32)
        top = mx.maximum(mx.max(rank, axis=-1, keepdims=True), 1e-20)
        return rank, rank / top
    if isinstance(out, mx.array) and out.ndim >= 2:
        logits = out if out.ndim == 2 else out.reshape(-1, out.shape[-1])
        rank = mx.take_along_axis(logits, ind2, axis=-1).astype(mx.float32)
        if pred.score_bias is not None:
            w = mx.sigmoid(rank)
            return w, w / mx.maximum(mx.max(w, axis=-1, keepdims=True), 1e-20)
        return rank, mx.exp(rank - mx.max(rank, axis=-1, keepdims=True))
    return None


_EXACT_STEP = False


@contextlib.contextmanager
def exact_step():
    """Decode inside this block computes the router's full top-k.

    No flow, PRUNE, ROUTE_TOP_K / TOP_P, WAIT_ABOVE or prune head: every layer
    waits for every expert it picked. Lets a bench compare the exact model with
    a tuned env on one load (bench/early_stop_probe.py --exact). These
    steps stay out of the sidecar's demand trajectories.
    """
    global _EXACT_STEP
    prev = _EXACT_STEP
    _EXACT_STEP = True
    try:
        yield
    finally:
        _EXACT_STEP = prev


_MATCHED_STEP = False


@contextlib.contextmanager
def matched_step():
    """A multi-position verify pass that routes each row like a plain step.

    Same selector, PRUNE / WAIT_ABOVE and fidelity span scale as the plain
    decode step at this point, applied per row, on the sync path (flow keeps
    per-token state). Stays out of the sidecar, early-warm and ttw token
    windows like exact_step does: a pass is several positions, not one token.
    """
    global _MATCHED_STEP
    prev = _MATCHED_STEP
    _MATCHED_STEP = True
    try:
        yield
    finally:
        _MATCHED_STEP = prev


_CACHE_ONLY = False


@contextlib.contextmanager
def cache_only():
    """Model calls inside this block are read for the KV they write only.

    The server's prefill loops: the last prompt token always goes through the
    decode step, so nothing reads a prefill chunk's logits. See
    config.PREFILL_SKIP_FINAL.
    """
    global _CACHE_ONLY
    prev = _CACHE_ONLY
    _CACHE_ONLY = True
    try:
        yield
    finally:
        _CACHE_ONLY = prev


_DRAFT_STEP: tuple[float, float] | None = None


@contextlib.contextmanager
def draft_step(prune: float, wait_above: float):
    """Decode inside this block uses this PRUNE / WAIT_ABOVE instead of config's.

    For a self-drafter on an exact decode config: the draft tokens come from a
    cheap approximate step and an exact_step pass verifies them, so the drafts
    only decide how many tokens a pass emits, never which ones. Decode flow
    stays out (it keys off config.PRUNE); DRAFT_FLOW runs these steps on the
    flow graph instead (StreamedSwitchGLU._draft_flow).
    """
    global _DRAFT_STEP
    prev = _DRAFT_STEP
    _DRAFT_STEP = (float(prune), float(wait_above))
    try:
        yield
    finally:
        _DRAFT_STEP = prev


def draft_flow_on(model) -> bool:
    """Self-draft steps on this model run on the flow graph (config.DRAFT_FLOW)."""
    if not config.DRAFT_FLOW:
        return False
    ec = getattr(model, "_pagedmoe_cache", None)
    return bool(
        ec is not None
        and getattr(ec, "slab", None) is not None
        and getattr(ec, "flow_on", lambda: False)()
    )


def decode_is_exact() -> bool:
    """True when a plain decode step already computes the router's full top-k."""
    return (
        float(config.PRUNE) <= 0.0
        and float(config.WAIT_ABOVE) <= 0.0
        and not 0 < int(config.ROUTE_TOP_K)
        and not 0.0 < float(config.ROUTE_TOP_P) < 1.0
        and not (config.SIDECAR and config.SIDECAR_HEAD_PRUNE)
    )


def _critical_slots(ratio: np.ndarray) -> np.ndarray:
    """Slots the per-layer path would block on if they missed.

    Kept by PRUNE (weight vs the row's winner) and carrying at least WAIT_ABOVE
    of the mixture. Flow drops any of these that are not resident, which is
    where its quality goes on a model whose cache cannot hold the working set.
    """
    share = ratio / np.maximum(ratio.sum(axis=-1, keepdims=True), 1e-20)
    return (ratio >= float(config.PRUNE)) & (share >= float(config.WAIT_ABOVE))


def _tapped_route(gate, x):
    """This layer's own router output, or None if it was not the last call.

    `x` is the *unreshaped* activation the block passed to both the gate and
    switch_mlp, so identity on it is what proves the recorded output belongs to
    this forward rather than to a prediction or a previous layer.
    """
    rec = _LAST_ROUTE
    if rec is None or rec[0] != id(gate):
        return None
    if rec[1] != id(x) and rec[3] != _rows(x.shape):
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

    def _ahead_graph(self, x_flat, k: int):
        """(next layer, lazy routed ids) predicted from this layer's MoE input,
        or None. See ExpertCache.prefill_ahead.

        Same prediction decode uses (the next layer's real router, norm gains
        corrected by ring._ratio), batched over the prompt chunk. Never reaches
        compute: it only decides which reads start early.
        """
        nxt = self.ring_index + 1
        layers = self.ring.layers
        if nxt >= len(layers):
            return None
        if layers[nxt]._skips_final():
            return None
        tgt = layers[nxt]
        pred = tgt.pred
        if pred is None or pred.gate is None:
            return None
        try:
            ratio = self.ring._ratio(self.ring_index, nxt)
            xin = x_flat if ratio is None else x_flat * ratio
            out = _untapped_call(pred.gate, _pred_input(pred, xin))
            if isinstance(out, (tuple, list)):
                ids = out[0]
            else:
                m = min(k, out.shape[-1])
                ids = mx.argpartition(_router_select(pred, out), kth=-m, axis=-1)[..., -m:]
            return tgt, ids.reshape(-1).astype(mx.int32)
        except Exception:
            return None

    def _loose(self, entry: dict) -> dict:
        """A slot-backed entry as per-expert arrays (views, not copies).

        Only for the fallback path: a layer whose experts did not all get
        slots cannot be addressed by one gather, and `_run` wants arrays.
        """
        slot = entry.get("__slot__")
        slab = self.cache.slab
        if slot is None:
            slot = entry.get("__stage__")
            slab = self.cache.stage
            if slot is None:
                return entry
        out = {"__nbytes__": entry["__nbytes__"]}
        for name in slab.names:
            out[name] = slab.expert(name, slot)
        return out

    def _run_slab(self, x_flat, inds: np.ndarray, slots: dict, keep=None, early=None):
        """Decode compute over slab slots: one gather_qmm per projection.

        Every output *row* is its own batch element, indexed into the slab by
        the slot of the expert it needs. That makes a variable number of rows
        per expert free (positions of a verify batch route differently), and
        results come back already in (token, k) order - so unlike `_run` there
        is no sort, no unsort, and no per-expert Python loop. Feeding `x` by
        `lhs_indices` also avoids materializing the gathered activations.

        Bit-identical to `_run` on the same weights: same kernel, same
        quantization parameters, just batched (verified in tests/test_stream.py
        and bench/slab_probe.py).
        """
        N, K = inds.shape
        flat = inds.reshape(-1)
        if keep is None:
            kept_pos = None
        else:
            kept_pos = np.nonzero(keep.reshape(-1))[0]
            flat = flat[kept_pos]
        if flat.size == 0:
            return mx.zeros((N, K, x_flat.shape[-1]), dtype=x_flat.dtype)

        self.cache.check_slots(self.layer_key, slots)
        if early is not None:
            pos = np.arange(N * K) if kept_pos is None else kept_pos
            done, y_done = early
            rest = np.setdiff1d(pos, done, assume_unique=True)
            out = mx.zeros((N * K, y_done.shape[-1]), dtype=y_done.dtype)
            out[mx.array(done)] = y_done
            if rest.size:
                ids = inds.reshape(-1)[rest]
                out[mx.array(rest)] = self._slab_rows(
                    x_flat, rest // K, [slots[int(e)] for e in ids]
                )
            return out.reshape(N, K, -1)
        pos = np.arange(N * K) if kept_pos is None else kept_pos
        y = self._slab_rows(x_flat, pos // K, [slots[int(e)] for e in flat])

        if keep is None:
            return y.reshape(N, K, -1)
        out = mx.zeros((N * K, y.shape[-1]), dtype=y.dtype)
        out[mx.array(kept_pos)] = y
        return out.reshape(N, K, -1)

    def _slab_rows(self, x_flat, rows, slot_ids) -> mx.array:
        """[len(rows), D]: row r of x through the expert in slot slot_ids[r]."""
        lhs = mx.array(np.asarray(rows, dtype=np.uint32))
        rhs = mx.array(np.asarray(slot_ids, dtype=np.uint32))
        xe = x_flat[:, None, :]  # [N, 1, D]: batch dim is the token
        x_up = self._gather_proj(xe, "up_proj", rhs, lhs)
        x_gate = self._gather_proj(xe, "gate_proj", rhs, lhs)
        y = self._gather_proj(self.activation(x_up, x_gate), "down_proj", rhs)
        return y.reshape(y.shape[0], y.shape[-1])

    def _run_hits_early(self, x_flat, inds: np.ndarray, keep, begun):
        """Start the rows whose experts are already in the slab, before the
        layer blocks on its misses. Returns (flat positions, lazy rows) for
        _run_slab, or None.

        Hits stay pinned until the next layer's fetch, so no read this layer
        waits on can land in their slots while the GPU uses them.
        """
        hit = {e: ent["__slot__"] for e, ent in begun.hits.items() if "__slot__" in ent}
        if not hit:
            return None
        K = inds.shape[1]
        flat = inds.reshape(-1)
        pos = np.arange(flat.size) if keep is None else np.nonzero(keep.reshape(-1))[0]
        done = pos[np.isin(flat[pos], np.fromiter(hit, dtype=flat.dtype))]
        if done.size == 0 or done.size == pos.size:
            return None
        y = self._slab_rows(x_flat, done // K, [hit[int(e)] for e in flat[done]])
        mx.async_eval(y)
        return done, y

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
        if self.cache.flow_table(self.layer_key) is None:
            return None
        ranked = _route_rank_mx(self.pred, x, ind2)
        if ranked is None:
            return None
        rank, ratio = ranked
        if self.ring_index == 0:
            # Last token's routing is the read list for this one: a whole token
            # of lead time instead of a few layers. Runs before this token takes
            # any table, so a slot it evicts is already gone from every table
            # this token will read.
            self.cache.leave_prefill()
            self.ring.flow_pump()
        table = self.cache.flow_table(self.layer_key)
        out, valid = self._run_flow(x_flat, ind2, rank, table, config.FLOW_TOPM)
        K = ind2.shape[-1]
        resident = valid if not 0 < config.FLOW_TOPM < K else mx.take(table, ind2) >= 0
        self.ring.flow_note(self.ring_index, ind2, valid, ratio, resident)
        if self.ring_index + 1 >= len(self.ring.layers):
            self.ring.flow_note_hidden(x_flat[-1])
            self.ring.flow_note_fence(out)
        return out

    def _draft_flow(self, x, x_flat, ind2):
        """One MoE layer of a self-draft token on the flow graph, or None.

        Resident experts only, no reads, no ring or sidecar state: a draft is a
        guess the verify pass checks, so it only has to be cheap. A background
        read can retarget a slot the table still names while this runs; that
        costs a worse guess, never a wrong output token.
        """
        table = self.cache.flow_table(self.layer_key)
        if table is None:
            return None
        ranked = _route_rank_mx(self.pred, x, ind2)
        if ranked is None:
            return None
        out, _ = self._run_flow(x_flat, ind2, ranked[0], table, int(config.DRAFT_FLOW_TOPM))
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
                if "__slot__" in entry or "__stage__" in entry:
                    entry = self._loose(entry)
                group_ys.append(self._expert_ffn(x_sorted[lo:hi], entry))
            if eval_groups and group_ys:
                # Force this group's matmuls to run NOW, so its (possibly
                # uncached) weights can be freed before the next group lands.
                # Reads for the next group are already in flight, so the GPU
                # and the SSD stay busy simultaneously.
                mx.eval(*group_ys)
                stage_gentle.wait_for_pause()
            ys.extend(group_ys)

        if not ys:
            # Every slot was dropped (WAIT_ABOVE on a cold layer); same as _run_slab.
            return mx.zeros((N, K, x_flat.shape[-1]), dtype=x_flat.dtype)
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

    def _stitch_selectors_off(self) -> bool:
        """Approximate selectors change which experts run. The one graph does not."""
        if _DRAFT_STEP is not None:
            return False
        if (
            float(config.PRUNE) > 0.0
            or float(config.WAIT_ABOVE) > 0.0
            or int(config.ROUTE_TOP_K) > 0
            or float(config.ROUTE_TOP_P) > 0.0
        ):
            return False
        sc = self.ring.sidecar
        prune = getattr(sc, "prune", None) if sc is not None else None
        if prune is not None and getattr(prune, "enabled", False):
            return False
        return True

    def _stitch_ok(self, n_tokens: int) -> bool:
        """True when every routed expert is in a fixed slot."""
        if not config.STITCH or _DRAFT_STEP is not None:
            return False
        return bool(getattr(self.cache, "stitched", False)) and n_tokens <= _DECODE_MAX_TOKENS

    def _stitch_forward(self, x, x_flat, ind2):
        """One MoE layer of a fully stitched token. None to use the per-layer path."""
        # FLOW_TOPM asks flow to compute fewer than the router picked. That
        # reduction is the knob; the static full-set gather would undo it.
        if (
            config.FLOW
            and int(config.FLOW_TOPM) > 0
            and not _EXACT_STEP
            and not fidelity.rescuing()
        ):
            return None
        table = self.cache.stitch_table(self.layer_key)
        if table is None or self.cache.slab is None:
            return None
        # topm 0: every id the router picked. Rank is unused on that path.
        out, resident = self._run_flow(x_flat, ind2, None, table, 0)
        # KEEP_FREE puts every resident expert back, and here every expert is
        # resident, so the slider's prune does not change the mixture. With
        # KEEP_FREE off the same mask the per-layer path uses still applies.
        last = self.ring_index + 1 >= len(self.ring.layers)
        compute = None
        if not (config.KEEP_FREE or self._stitch_selectors_off()):
            compute, _missing = _closed_keep(self.pred, x, ind2, resident)
            if compute is None:
                return None
            self.cache.note_stitch_keep(compute, last)
            out = out * compute.astype(out.dtype)[..., None]
        activity.touch()
        self.cache.stitch_forwards += 1
        # These experts are resident; fetch() is not called, so count them
        # here or a stitched token looks like it routed nothing.
        self.cache.hits += int(ind2.shape[0]) * int(ind2.shape[1])
        self.cache.route_total += int(ind2.shape[0]) * int(ind2.shape[1])
        if last:
            self.cache.full_stitch_tokens += 1
        return out

    def _closed_forward(self, x, x_flat, ind2):
        """One MoE layer using whatever the residency table already holds.

        The keep set is the slider's: prune, top-k, top-p, KEEP_FREE, and the
        wait gate, including a strict fidelity span. A miss is an expert that
        set would have waited on disk for. Resident experts the slider keeps
        are computed; the rest are zeros. The token wrapper keeps the logits
        only when that miss sum is zero.
        """
        table = (
            self.cache.stitch_table(self.layer_key)
            if self.cache.stitched
            else self.cache.flow_table(self.layer_key)
        )
        if table is None or self.cache.slab is None:
            self.ring.closed_broke = True
            return None
        activity.touch()
        self.cache.stitch_forwards += 1
        out, resident = self._run_flow(x_flat, ind2, None, table, 0)
        compute, missing = _closed_keep(self.pred, x, ind2, resident)
        if compute is not None:
            out = out * compute.astype(out.dtype)[..., None]
            miss = mx.sum(missing.astype(mx.int32))
        else:
            miss = mx.sum(1 - resident.astype(mx.int32))
        prev = self.ring._closed_miss
        self.ring._closed_miss = miss if prev is None else prev + miss
        return out

    def _skips_final(self) -> bool:
        """Final decoder layer of a cache-only prefill (config.PREFILL_SKIP_FINAL)."""
        layers = self.ring.layers
        return (
            _CACHE_ONLY
            and config.PREFILL_SKIP_FINAL
            and self.ring.final_is_moe
            and len(layers) > 1
            and self.ring_index + 1 == len(layers)
        )

    def __call__(self, x, indices) -> mx.array:
        if not gpu_crumbs.LAYERS:
            return self._forward(x, indices)
        gpu_crumbs.crumb(x, f"{self.layer_key} input (attention done)")
        y = self._forward(x, indices)
        gpu_crumbs.crumb(y, f"{self.layer_key} output (experts done)")
        return y

    def _forward(self, x, indices) -> mx.array:
        lead_shape = indices.shape  # (..., K)
        K = lead_shape[-1]
        x_flat = x.reshape(-1, x.shape[-1])
        ind2 = indices.reshape(-1, K)
        N = ind2.shape[0]  # static shape: known without a sync

        if self._stitch_ok(N):
            # Every routed expert is already in a fixed slot. The gather uses
            # the router's ids directly, so this token never syncs per layer.
            out = self._stitch_forward(x, x_flat, ind2)
            if out is not None:
                return out.reshape(*lead_shape, -1)
        if self.ring.closed_active and N <= _DECODE_MAX_TOKENS and _DRAFT_STEP is None:
            out = self._closed_forward(x, x_flat, ind2)
            if out is not None:
                return out.reshape(*lead_shape, -1)

        if N > _DECODE_MAX_TOKENS and self._skips_final():
            # Nothing reads this layer's output but logits nobody evaluates.
            if hasattr(self.cache, "end_prefill_measure"):
                self.cache.prefill_layers_skipped += 1
                self.cache.end_prefill_measure()
            if self.ring.sidecar is not None:
                self.ring.sidecar.end_prefill_chunk()
            return mx.zeros((*lead_shape, x.shape[-1]), dtype=x.dtype)

        # Flow decode: keep the whole layer on the GPU so the token needs no
        # per-layer sync at all. Returns None (and falls through to the path
        # below) whenever its preconditions do not hold - no slab, an
        # unrecognized router, a table that does not exist yet - so enabling it
        # can never break a model, only leave it on the ordinary path.
        # PRUNE 0 means exact output was asked for, so flow stays out of it.
        # The fidelity guard holds a degenerate approximate decode exact.
        rescue = fidelity.rescuing()
        if rescue and self.ring_index == 0:
            self.ring.flow_suspend()
        exact = _EXACT_STEP or rescue
        # One token's worth of decode, for the per-token learners and windows.
        quiet = exact or _MATCHED_STEP
        if (
            config.FLOW
            and config.PRUNE > 0
            and not quiet
            and N <= _DECODE_MAX_TOKENS
            and self.cache.slab is not None
        ):
            if self.ring_index == 0:
                self.ring.flow_token_start()
            if self.ring.flow_ready():
                flowed = self._flow_forward(x, x_flat, ind2)
                if flowed is not None:
                    return flowed.reshape(*lead_shape, -1)
        if (
            _DRAFT_STEP is not None
            and not exact
            and config.DRAFT_FLOW
            and N <= _DECODE_MAX_TOKENS
            and self.cache.slab is not None
        ):
            drafted = self._draft_flow(x, x_flat, ind2)
            if drafted is not None:
                return drafted.reshape(*lead_shape, -1)

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
        if _DRAFT_STEP is not None and not exact:
            prune, wait_above = _DRAFT_STEP
        wait_on = wait_above > 0.0
        need_weights = not exact and (
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
                    score_bias=self.pred.score_bias,
                )
                if wnp is not None:
                    # Adaptive prune/wait from sidecar when head is live. Not
                    # for a matched pass: the head reads one token's routing,
                    # and it only ever raises both, so the configured values
                    # are what every row can share without going looser.
                    if (
                        self.ring.sidecar is not None
                        and self.ring.sidecar.prune.enabled
                        and not _MATCHED_STEP
                    ):
                        prune, wait_above = self.ring.sidecar.prune_params(wnp)
                        wait_on = wait_above > 0.0
                    if fidelity.strict() and _DRAFT_STEP is None:
                        prune, wait_above = fidelity.adjust(prune, wait_above)
                        wait_on = wait_above > 0.0
                    keep = _selector_mask(wnp, prune=prune, cap=cap, top_p=top_p)
                    if keep is None and wait_on:
                        keep = np.ones(wnp.shape, dtype=bool)
                    if self.ring.sidecar is not None and not quiet:
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
        routes = self.ring.routes if route_log._FWD is not None else None
        rank_mx = None
        if routes is not None and N > _DECODE_MAX_TOKENS:
            # Prefill has no selector weights; the router's own scores order
            # the record, evaluated with the ids below (no extra sync).
            rk = _route_rank_mx(self.pred, x, ind2)
            rank_mx = rk[0] if rk is not None else None
        ahead = None
        if (
            N > _DECODE_MAX_TOKENS
            and config.PREFILL_AHEAD
            and hasattr(self.cache, "prefill_ahead")
        ):
            ahead = self._ahead_graph(x_flat, K)
            if ahead is not None:
                # Rides the routing sync this layer pays anyway.
                if rank_mx is not None:
                    mx.eval(ind2, ahead[1], rank_mx)
                else:
                    mx.eval(ind2, ahead[1])
        if rank_mx is not None and ahead is None:
            mx.eval(ind2, rank_mx)
        inds = np.asarray(ind2)
        if N > _DECODE_MAX_TOKENS:
            # The router sync just drained the GPU; a background stage leaves
            # it idle a moment here so the rest of the Mac gets a turn.
            stage_gentle.pace()
        if routes is not None:
            try:
                w = wnp if wnp is not None else (
                    np.asarray(rank_mx) if rank_mx is not None else None
                )
                route_log.record_forward(routes, self.layer_key, inds, w)
            except Exception:
                pass

        if config.FLOW and wnp is not None and N <= _DECODE_MAX_TOKENS:
            # Same count flow keeps, so the handover compares like with like.
            ratio = wnp / np.maximum(wnp.max(axis=-1, keepdims=True), 1e-20)
            crit = _critical_slots(ratio)
            if crit.any():
                want = np.unique(inds[crit])
                have = self.cache.resident_ids(self.layer_key, want)
                self.cache.crit_misses += len(want) - len(have)
            self.cache.crit_steps += int(N)

        if keep is not None and wnp is not None:
            free_arr = None
            if config.KEEP_FREE or wait_on:
                # One residency query serves both KEEP_FREE and WAIT_ABOVE.
                need = np.unique(inds if wait_on else inds[~keep])
                free = self.cache.free_ids(self.layer_key, need)
                free_arr = np.array(sorted(free), inds.dtype)
            free_mask = None
            if wait_on and free_arr is not None:
                free_mask = np.isin(inds, free_arr)
                if not quiet and _DRAFT_STEP is None:
                    prune_curve.note(wnp, free_mask)
            if config.KEEP_FREE and free_arr is not None and not keep.all():
                # Put back every dropped expert that needs no disk read. This
                # is the one part of the trade that is strictly one-sided:
                # those bytes are already here, so the only thing dropping them
                # bought was a worse mixture.
                rescued = np.isin(inds, free_arr) & ~keep
                if rescued.any():
                    keep = keep | rescued
                    self.cache.kept_free += int(rescued.sum())
            if free_mask is not None:
                # Skip - and prefetch for later tokens - any surviving expert
                # that would block on a read and isn't worth blocking for.
                # This is what takes the disk off the critical path.
                stall = keep & ~free_mask
                skip = stall & (wnp < wait_above)
                if skip.any():
                    keep = keep & ~skip
                    late = np.unique(inds[skip]).tolist()
                    self.cache.skipped_waits += len(late)
                    # Next time this layer runs is a full token away. Tag it
                    # so a nearer last-token prefetch can take the slot.
                    self.cache.prefetch(
                        self.layer_key,
                        late,
                        need_at=self.cache._prefetch_clock + len(self.ring.layers),
                    )
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
        if config.FLOW and wnp is not None and N <= _DECODE_MAX_TOKENS:
            kept = wnp if keep is None else wnp * keep
            self.cache.mass_sync += float(kept.sum())
            self.cache.rows_sync += int(N)

        if N <= _DECODE_MAX_TOKENS:
            ids = [int(e) for e in unique]
            rlog = self.ring.routes
            if rlog is not None:
                if rlog.armed:
                    rlog.score(self.layer_key, ids)
                if not _EXACT_STEP and _DRAFT_STEP is None:
                    rlog.note_kept(
                        self.layer_key,
                        float(keep.sum(axis=-1).mean()) if keep is not None else float(K),
                    )
            # Decode wants the full cache back: a big resident set is what
            # keeps per-token disk reads down (prefill shrinks it, below).
            self.cache.leave_prefill()
            # First MoE layer of the first decode token after prefill: seed
            # the rebuilt slab from the early-decode warm pool (prefetch only)
            # and start the greppable [ttw] timer for A/B.
            if self.ring_index == 0:
                self.ring.mark_decode_warm_start()
                if self.ring.tool_warm is not None:
                    self.ring.tool_warm.maybe_warm(self.cache)
            # Submit the reads before the accounting below. None of it changes
            # which experts this layer computes, and on a miss the drive should
            # already be moving while it runs.
            begun = self.cache._begin(self.layer_key, ids, use_slab=True)
            early = None
            if config.HITS_EARLY and begun.futs and self.cache.slab is not None:
                early = self._run_hits_early(x_flat, inds, keep, begun)
            trace = getattr(self.cache, "demand_trace", None)
            if trace is not None:
                trace.append((self.layer_key, ids))
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
            if not quiet:
                self.ring.note_demand(self.layer_key, ids)
            experts = self.cache._finish(self.layer_key, begun, "lru")
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
            # Last MoE layer of the token: close the sidecar / early-warm / ttw
            # / mitm token window. Cheap None-check when all are off.
            if self.ring_index + 1 >= len(self.ring.layers) and not quiet:
                if (
                    self.ring.sidecar is not None
                    or self.ring.early_warm is not None
                    or self.ring.tool_warm is not None
                    or self.ring.ttw is not None
                    or self.ring.mitm is not None
                ):
                    need_hidden = (
                        self.ring.sidecar is not None or self.ring.mitm is not None
                    )
                    self.ring.end_decode_token(
                        hidden=(x_flat[-1] if need_hidden else None)
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
                out = self._run_slab(x_flat, inds, slots, keep=keep, early=early)
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
            if self.ring.early_warm is not None and self.ring_index == 0:
                self.ring.early_warm.drop_token()
            if self.ring.tool_warm is not None and self.ring_index == 0:
                self.ring.tool_warm.drop_token()
            if self.ring.ttw is not None and self.ring_index == 0:
                self.ring.ttw.reset()
            if N >= config.PREFILL_SHRINK_TOKENS or (
                self.ring_index == 0
                and N > 1
                and getattr(self.cache, "slab_oversized", None) is not None
                and self.cache.slab_oversized()
            ):
                # Second case: another app grew since the slab was built and
                # the kernel is compressing; leave_prefill rebuilds it to fit.
                self.cache.enter_prefill(int(config.PREFILL_CACHE_GB * (1 << 30)))
            elif getattr(self.cache, "slab", None) is not None and N > 1:
                # Below SHRINK_TOKENS the decode slab normally stays up so
                # tool results keep residency. If active memory is already
                # near the RAM envelope, even a few hundred tokens of hyper
                # residuals (hc_count>1) will Metal-OOM and freeze the Mac.
                # make_room_for is a no-op when there is real headroom; the
                # need is what chunks this size measured (prefill_need).
                if hasattr(self.cache, "prefill_need"):
                    if self.ring_index == 0:
                        self.cache.chunk_need = self.cache.prefill_need(int(N))
                    need = self.cache.chunk_need
                else:
                    need = max(int(N) << 20, int(0.5 * (1 << 30)))
                if self.ring_index == 0 and hasattr(self.cache, "regrow_slab"):
                    self.cache.regrow_slab(need)
                yielded = self.cache.make_room_for(need)
                if self.ring_index == 0 and not yielded and hasattr(self.cache, "begin_prefill_measure"):
                    self.cache.begin_prefill_measure(int(N))
            # Prefill-union head: start-of-chunk prefetch (no-op when off).
            if self.ring.sidecar is not None and self.ring_index == 0:
                self.ring.sidecar.begin_prefill_chunk(self.cache)
            if self.ring.sidecar is not None:
                self.ring.sidecar.note_prefill_demand(
                    self.layer_key, [int(e) for e in unique]
                )
            ahead_fn = None
            if ahead is not None:
                tgt_key = ahead[0].layer_key
                # Most-predicted first: whatever cap the cache applies keeps
                # the guesses the most tokens agree on.
                tgt_ids, votes = np.unique(np.asarray(ahead[1]), return_counts=True)
                tgt_ids = tgt_ids[np.argsort(-votes, kind="stable")]
                ahead_fn = lambda: self.cache.prefill_ahead(tgt_key, tgt_ids)  # noqa: E731
            groups = self.cache.fetch_groups(
                self.layer_key,
                [int(e) for e in unique],
                config.GROUP_BYTES,
                install="soft",
                ahead=ahead_fn,
            )
            out = self._run(x_flat, inds, groups, eval_groups=True)
            self.cache.prefill_layers += 1
            hook = getattr(self.cache, "prefill_layer_hook", None)
            if hook is not None:
                try:
                    hook(self.ring_index, len(self.ring.layers))
                except Exception:
                    pass
            # Seed decode prefetch with the last prompt token's routing.
            self.last_ids = [int(e) for e in np.unique(inds[-1])]
            self.predicted.clear()  # stale across a prefill (accounting only)
            # Last MoE layer of a prefill chunk: close the union window.
            if self.ring_index + 1 >= len(self.ring.layers):
                if hasattr(self.cache, "end_prefill_measure"):
                    self.cache.end_prefill_measure()
                if self.ring.sidecar is not None:
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
        # The last layer in `layers` is the model's last decoder layer (set by
        # patch_model). Only then may a cache-only prefill skip it.
        self.final_is_moe = False
        # route_log.RouteLog when verify decode may replay copied routing.
        self.routes = None
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
        # Closed stitch: one graph for this token, accepted only if every
        # routed expert was already resident. See install_closed_stitch.
        self.closed_active = False
        self.closed_live = True
        self.closed_armed = False
        self.closed_broke = False
        self.warm_streak = 0
        # After a replay, ignore this many decode tokens before arming again.
        # A miss means the keep set was not resident; trying again immediately
        # just pays a second forward and thrashes the slab.
        self.closed_cooldown = 0
        self._closed_miss = None
        # Online expert-prefetch sidecar (None when PAGED_MOE_SIDECAR=0).
        # Attached by patch_model / loader only when enabled - every call site
        # below is gated on this so disabled models pay a single None-check.
        self.sidecar = None
        self._token_open = False
        # Early-decode warm pool (None unless EARLY_DECODE or TRAIN). Prefetch
        # only; independent of the sidecar master switch.
        self.early_warm = None
        # Tool-body expert prefetch (None unless TOOL_WARM). Independent of
        # the early-decode pool: that one is the first tokens after prefill.
        self.tool_warm = None
        # Greppable time-to-warm A/B ([ttw] lines). Independent of early_warm
        # so disable/enable EARLY_DECODE still produces comparable warm lines.
        self.ttw = None
        if bool(config.TTW_LOG):
            from .time_to_warm import TimeToWarm

            win = int(config.TTW_TOKENS) or int(config.EARLY_DECODE_TOKENS)
            self.ttw = TimeToWarm(win)
        # Dev-only activation debugger (None unless PAGED_MOE_MITM and the module is present).
        self.mitm = None
        # Flow decode: one token's lazy routing, read at the next token's start.
        self._flow_recs: list[tuple] = []
        self._flow_hidden = None
        self._flow_fence = None
        self._flow_tokens = 0
        self._flow_live = False
        self._flow_hm: tuple[int, int] | None = None
        self._flow_hit_ema: float | None = None
        self._flow_crit_ema: float | None = None
        self._flow_backoff = False

    def attach_mitm(self, mitm) -> None:
        self.mitm = mitm

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

    def attach_early_warm(self, early) -> None:
        self.early_warm = early

    def attach_tool_warm(self, warm) -> None:
        self.tool_warm = warm

    def mark_decode_warm_start(self) -> None:
        """First MoE layer of first decode after prefill: start [ttw] + warm."""
        early_on = (
            self.early_warm is not None and bool(getattr(self.early_warm, "enabled", False))
        )
        if self.ttw is not None:
            self.ttw.on_first_decode(self.cache, early=early_on)
        self.maybe_early_warm()

    def maybe_early_warm(self) -> None:
        """Once per turn after leave_prefill: prefetch the early-decode pool."""
        if self.early_warm is None:
            return
        info = self.early_warm.maybe_warm(self.cache) or {}
        if self.ttw is not None:
            self.ttw.note_prefetch(
                float(info.get("pref_ms", 0.0) or 0.0),
                int(info.get("plan_n", 0) or 0),
                actuated=bool(info.get("actuated", False)),
            )

    def note_demand(self, layer_key: str, expert_ids: list[int]) -> None:
        if self.early_warm is not None:
            self.early_warm.note_demand(layer_key, expert_ids)
        if self.tool_warm is not None:
            self.tool_warm.note_demand(layer_key, expert_ids)
        if self.sidecar is not None:
            self.sidecar.note_demand(layer_key, expert_ids)
            self._token_open = True
        elif self.early_warm is not None or self.ttw is not None:
            self._token_open = True

    # --------------------------------------------------------- flow decode

    def flow_token_start(self) -> None:
        """Count decode tokens since the last prefill (flow warmup gate)."""
        self._flow_tokens += 1
        if self.tool_warm is not None:
            self.tool_warm.maybe_warm(self.cache)
        c = self.cache
        hm = (c.hits, c.misses, c.crit_misses, c.crit_steps)
        prev = self._flow_hm
        self._flow_hm = hm
        if prev is not None:
            dh, dm = hm[0] - prev[0], hm[1] - prev[1]
            if not self._flow_live and dh + dm > 0:
                r = dh / (dh + dm)
                ema = self._flow_hit_ema
                self._flow_hit_ema = r if ema is None else 0.7 * ema + 0.3 * r
            dc, ds = hm[2] - prev[2], hm[3] - prev[3]
            if ds > 0:
                # Per token, summed over MoE layers: the sampled token carries
                # every layer's error, so deep models get less room per layer.
                r = dc / ds * len(self.layers)
                ema = self._flow_crit_ema
                self._flow_crit_ema = r if ema is None else 0.7 * ema + 0.3 * r
        max_crit = float(config.FLOW_MAX_CRIT)
        crit = self._flow_crit_ema
        if self._flow_live:
            if max_crit > 0 and crit is not None and crit > max_crit:
                # The cache stopped holding what the router wants: flow would
                # now be dropping experts the per-layer path waits for. Finish
                # the last flow token's reads and go back to waiting.
                self.flow_pump()
                self._flow_live = False
                self._flow_backoff = True
                print(
                    f"[paged-moe] flow off after {self._flow_tokens - 1} tokens "
                    f"(critical misses {crit:.1f}/token)",
                    flush=True,
                )
            return
        if self._flow_tokens <= config.FLOW_WARMUP:
            return
        warm = (
            config.FLOW_WARM_HIT <= 0
            or (self._flow_hit_ema or 0.0) >= config.FLOW_WARM_HIT
        )
        if max_crit > 0:
            # Re-entry needs clear room under the bar, so a cache sitting right
            # at it does not flip every few tokens.
            bar = max_crit / 2 if self._flow_backoff else max_crit
            warm = warm and crit is not None and crit <= bar
        if warm:
            self._flow_live = True
            print(
                f"[paged-moe] flow on after {self._flow_tokens - 1} tokens "
                f"(hit {self._flow_hit_ema or 0.0:.2f}, critical misses "
                f"{crit or 0.0:.1f}/token)",
                flush=True,
            )
            # Handing over from the per-layer path: its last deferred prediction
            # has no sync left to ride on, so let it go rather than leak a graph.
            self.drop_pending()

    def flow_ready(self) -> bool:
        return self._flow_live

    def flow_suspend(self) -> None:
        """Hand back to the per-layer path for exact tokens (fidelity guard).

        Same exit as the critical-miss backoff, minus the backoff: flow
        re-warms through flow_token_start once the guard lets go.
        """
        if not self._flow_live:
            return
        self.flow_pump()
        self._flow_live = False
        print("[paged-moe] flow paused: fidelity guard holds decode exact", flush=True)

    def flow_note(self, index: int, ind2, valid, ratio=None, resident=None) -> None:
        """Hold a layer's routing for the next token to read.

        These stay lazy on purpose: touching them here is the sync flow decode
        exists to avoid. The sampler materializes them at the end of the token
        (they are ancestors of the sampled id), and `flow_pump` reads them at
        the start of the next one, by which time it is a host copy.
        """
        self._flow_recs.append(
            (index, ind2, valid, ratio, valid if resident is None else resident)
        )

    def flow_note_hidden(self, hidden) -> None:
        self._flow_hidden = hidden

    def flow_note_fence(self, out) -> None:
        """Last MoE layer's output: every expert this token computed feeds it."""
        self._flow_fence = out

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
        fence = self._flow_fence
        self._flow_fence = None
        # mlx-lm builds this token while the last one may still be running
        # (async_eval pipelining). flow_admit below can evict a slot and read
        # over it, so the last token's expert matmuls must be finished first.
        # The weight ratios are the only piece not already an ancestor of the
        # sampled token, so they ride on the same eval.
        admit = float(config.PRUNE)
        ratios = [r[3] for r in recs if r[3] is not None] if admit > 0 else []
        ratios += [r[4] for r in recs if admit > 0 and r[4] is not r[2]]
        if fence is not None or ratios:
            mx.eval(*([fence] if fence is not None else []), *ratios)
        cache = self.cache
        batch: list[tuple[str, list[int]]] = []
        for index, ind2, valid, ratio, resident in recs:
            layer = self.layers[index]
            inds = np.asarray(ind2)
            # Read only what PRUNE would keep. Flow computes whatever of the
            # router's top-k is resident, so reading the whole top-k would
            # double the working set on the experts PRUNE exists to skip.
            if admit > 0 and ratio is not None:
                rnp = np.asarray(ratio)
                ids = [int(e) for e in np.unique(inds[rnp >= admit])]
                rnp_res = np.asarray(resident)
                crit = _critical_slots(rnp) & ~rnp_res
                cache.crit_misses += len(np.unique(inds[crit]))
                cache.crit_steps += int(inds.shape[0])
                share = rnp / np.maximum(rnp.sum(axis=-1, keepdims=True), 1e-20)
                prune_curve.note(share, rnp_res.astype(bool))
                cache.mass_flow += float((share * rnp_res).sum())
                cache.rows_flow += int(inds.shape[0])
            else:
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
            cache.flow_admit(batch)
        if self.mitm is not None and hidden is not None:
            try:
                self.mitm.note_hidden(hidden)
            except Exception:
                pass
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
        if self.tool_warm is not None:
            self.tool_warm.end_token(cache)

    def flow_drop(self) -> None:
        """Forget held routing: a prefill replaces the state it came from."""
        self._flow_recs = []
        self._flow_hidden = None
        self._flow_fence = None
        self._flow_tokens = 0
        self._flow_live = False
        self._flow_hm = None
        self._flow_hit_ema = None
        self._flow_crit_ema = None
        self._flow_backoff = False

    def end_decode_token(self, hidden=None) -> None:
        """Close a decode token for the sidecar / early-warm / ttw / mitm collectors."""
        if self.early_warm is not None:
            self.early_warm.end_token()
        if self.tool_warm is not None:
            self.tool_warm.end_token(self.cache)
        if self.ttw is not None:
            self.ttw.end_token(self.cache)
        if self.mitm is not None and hidden is not None:
            try:
                self.mitm.note_hidden(hidden)
            except Exception:
                pass
        if self.sidecar is None:
            self._token_open = False
            return
        if not self._token_open:
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

    def prior(self, on: bool, reason: str, hold: int) -> bool:
        """Start the auto governor settled, before any evidence. It re-probes
        after `hold` windows as usual. False if not auto or already decided."""
        if not self._auto or self._decision is not None:
            return False
        self._settle(on, reason, hold)
        return True

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
            return  # an explicit PAGED_MOE_PREDICT is not up for revision
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
                                f"({sample}); PAGED_MOE_PREDICT=1 to force on",
                         config.PREDICT_HOLD_WINDOWS)
            return

        # Inconclusive. Give it room, but not unbounded room: if the difference is
        # still lost in the noise after a few times the minimum sample, there is
        # little to win either way, so take the cheaper side and hold much longer.
        if min(on_tokens, off_tokens) < floor * config.PREDICT_MAX_PROBE:
            return
        self._settle(False, f"no measurable difference ({on_tps:.1f} vs {off_tps:.1f} tok/s "
                            f"over {sample}); PAGED_MOE_PREDICT=1 to force on",
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
        from bench/router_matrix.py, which mapped router agreement over every
        (source, target) pair and found the decay is NOT a function of
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
            out = _untapped_call(pred.gate, _pred_input(pred, xin))
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
                inds = mx.argpartition(_router_select(pred, out), kth=-m, axis=-1)[..., -m:]
                if 0.0 < prune <= 1.0 or 0.0 < config.ROUTE_TOP_P < 1.0:
                    # Weight-based tests need the logits, not just the ids, so
                    # the host can apply the same filter the demand path will.
                    vals = mx.take_along_axis(out, inds, axis=-1)
                    kind = "logits"
                    if pred.score_bias is not None:
                        vals = mx.sigmoid(vals.astype(mx.float32))
                        kind = "scores"
                    # ids and logits ride the same sync (ids are small ints,
                    # exactly representable in f32).
                    flat = mx.concatenate(
                        [
                            inds.reshape(-1).astype(mx.float32),
                            vals.reshape(-1).astype(mx.float32),
                        ]
                    )
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
                # `d` layers of lead, including the wrap to the next token.
                # Staging keeps the soonest of these and drops the furthest
                # when the reserve is full (large experts overflow it).
                self.cache.prefetch(
                    nxt.layer_key,
                    nxt.last_ids,
                    need_at=self.cache._prefetch_clock + d,
                )


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


def _gate_input(gate, x: mx.array) -> mx.array:
    """DeepSeek-V2's grouped router reads (batch, seq) from x.shape[:2].

    A flat [tokens, hidden] input makes it reshape by the hidden size, so give
    that router one batch row. Every other gate gets x unchanged.
    """
    if x.ndim == 2 and getattr(gate, "topk_method", None) == "group_limited_greedy":
        return x[None]
    return x


def _route_weights_from_gate(
    gate,
    x_flat: mx.array,
    ind2: mx.array,
    pending: tuple = (),
    tapped=None,
    score_bias=None,
) -> np.ndarray | None:
    """Mixture weights for `ind2`, from either router family.

    * `nn.Linear` (Qwen3-MoE / Qwen3-Next): gate returns logits; softmax over
      the selected logits is the parent block's normalized top-k scores.
      With `score_bias` set (MiniMax) the block mixes with the selected
      sigmoids instead, normalized to sum 1.
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
    if tapped is not None:
        out = tapped
    else:
        xin = x_flat.astype(mx.float32) if score_bias is not None else x_flat
        out = _untapped_call(gate, _gate_input(gate, xin))
    if isinstance(out, (tuple, list)):
        if len(out) < 2:
            return None
        g_inds, g_scores = out[0], out[1]
        if not isinstance(g_inds, mx.array) or not isinstance(g_scores, mx.array):
            return None
        if tapped is None and g_inds.ndim == 3:
            g_inds = g_inds.reshape(-1, g_inds.shape[-1])
            g_scores = g_scores.reshape(-1, g_scores.shape[-1])
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
        sel = mx.take_along_axis(logits, ind2, axis=-1).astype(mx.float32)
        if score_bias is not None:
            w = mx.sigmoid(sel)
            w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)
        else:
            w = mx.softmax(sel, axis=-1)
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


def _is_final_layer(model, path: str) -> bool:
    """True when `path` sits in the last entry of its decoder-layer list."""
    parts = path.split(".")
    idx = [i for i, p in enumerate(parts) if p.isdigit()]
    if not idx:
        return False
    i = idx[0]
    try:
        layers, _ = _resolve_parent(model, ".".join(parts[: i + 1]))
        return int(parts[i]) == len(layers) - 1
    except Exception:
        return False


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
        norm_weight = getattr(norm, "weight", None)
        if norm_weight is not None and "gemma" in type(norm).__name__.lower():
            # Gemma-style RMSNorm scales by (1 + weight).
            norm_weight = 1.0 + norm_weight
        score_bias = None
        mt = str(getattr(getattr(model, "args", None), "model_type", "") or "")
        if mt in _SIGMOID_BIAS_ROUTERS:
            bias = getattr(block, "e_score_correction_bias", None)
            if isinstance(bias, mx.array):
                score_bias = bias
        return _RoutePredictor(gate, norm_weight, score_bias)
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

    Also resolves `language_model.`-prefixed keys (qwen4_exp / Flash).
    """
    components: dict[str, ExpertLocator] = {}

    for proj in _PROJS:
        for comp in ("weight", "scales", "biases", "bias"):
            loc = lookup_tensor(tensor_index, f"{path}.{proj}.{comp}")
            if loc is not None:
                components[f"{proj}.{comp}"] = StackedLocator(loc)
    if components:
        return components

    parents = [path.rsplit(".", 1)[0]]
    for cand in tensor_key_candidates(parents[0]):
        if cand not in parents:
            parents.append(cand)
    pats = [
        re.compile(
            re.escape(parent) + r"\.experts\.(\d+)\.(\w+)\.(weight|scales|biases|bias)$"
        )
        for parent in parents
    ]
    grouped: dict[str, dict[int, TensorLoc]] = {}
    for name, loc in tensor_index.items():
        for pat in pats:
            m = pat.match(name)
            if m and m.group(2) in _PROJS:
                grouped.setdefault(f"{m.group(2)}.{m.group(3)}", {})[
                    int(m.group(1))
                ] = loc
                break
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

    When PAGED_MOE_SIDECAR=1, attaches an online expert-prefetch sidecar
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
    ring.final_is_moe = _is_final_layer(model, glus[-1][0])

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

    if config.EARLY_DECODE or config.EARLY_DECODE_TRAIN:
        from .early_decode import EarlyDecodeWarm

        n_experts = max_expert_count(cache)
        if n_experts > 0:
            layer_keys = [glu.layer_key for glu in ring.layers]
            expert_bytes = 0
            if cache.slab is not None:
                expert_bytes = int(getattr(cache.slab, "expert_bytes", 0) or 0)
            if expert_bytes <= 0:
                expert_bytes = int(getattr(cache, "_slab_per_expert", 0) or 0)
            ring.attach_early_warm(
                EarlyDecodeWarm(
                    model_path or "unknown",
                    layer_keys,
                    n_experts,
                    expert_bytes=expert_bytes,
                )
            )

    if config.TOOL_WARM:
        from .tool_warm import ToolWarm

        n_experts = max_expert_count(cache)
        if n_experts > 0:
            layer_keys = [glu.layer_key for glu in ring.layers]
            expert_bytes = 0
            if cache.slab is not None:
                expert_bytes = int(getattr(cache.slab, "expert_bytes", 0) or 0)
            if expert_bytes <= 0:
                expert_bytes = int(getattr(cache, "_slab_per_expert", 0) or 0)
            ring.attach_tool_warm(
                ToolWarm(
                    model_path or "unknown",
                    layer_keys,
                    n_experts,
                    expert_bytes=expert_bytes,
                )
            )

    return ring


def install_closed_stitch(model) -> bool:
    """One-graph decode when this token's experts are already resident.

    The full-set stitch covers a model the slab can hold outright. This covers
    the rest: run the same gather from the live residency table, and keep the
    token only when every selected expert was in it. A miss rewinds the cache
    and runs the ordinary per-layer path, so a wrong table never changes a logit.
    """
    if not (config.STITCH and config.STITCH_PARTIAL):
        return False
    ring = getattr(model, "_pagedmoe_ring", None)
    cache = getattr(model, "_pagedmoe_cache", None)
    if ring is None or cache is None or cache.slab is None:
        return False
    if not cache.flow_on() and not cache.enable_flow(max_expert_count(cache)):
        return False
    base = type(model)
    orig = base.__call__

    def __call__(self, *args, **kwargs):
        if self is not model:
            return orig(self, *args, **kwargs)
        if not _closed_attempt(self, args, kwargs):
            return _note_warm(orig, self, args, kwargs)
        return _closed_call(orig, self, args, kwargs)

    model.__class__ = type(base.__name__, (base,), {"__call__": __call__})
    return True


def _decode_inputs(args, kwargs):
    inputs = args[0] if args else kwargs.get("inputs")
    cache = kwargs.get("cache", None)
    if cache is None and len(args) > 1:
        cache = args[1]
    return inputs, cache


def _closed_attempt(model, args, kwargs) -> bool:
    ring = model._pagedmoe_ring
    cache = model._pagedmoe_cache
    # Full stitch owns a model whose experts all sit in fixed slots.
    # FLOW_TOPM is a reduction below the router's top-k. Closed stitch would
    # compute the slider's keep set and undo it, so flow keeps that token.
    if not (config.STITCH and config.STITCH_PARTIAL) or not ring.layers or cache.stitched:
        return False
    # A prefill can give the slab back. The first decode token after it has to
    # take the per-layer path, which is where leave_prefill rebuilds it.
    if cache.slab is None or cache._slab_pending is not None:
        ring.closed_armed = False
        ring.warm_streak = 0
        return False
    if (
        config.FLOW
        and int(config.FLOW_TOPM) > 0
        and not _EXACT_STEP
        and not fidelity.rescuing()
    ):
        return False
    # Flow is already one graph. A flow token also does not record a fetch
    # when it skips an expert, so it looks warm and then the stitch replay
    # runs the token twice. Drafts are checked later; they must not stitch.
    if _DRAFT_STEP is not None or ring.flow_ready():
        return False
    if not ring.closed_live or not ring.closed_armed or ring.closed_active:
        return False
    inputs, prompt_cache = _decode_inputs(args, kwargs)
    if inputs is None or prompt_cache is None:
        return False
    if getattr(inputs, "ndim", 0) < 2 or int(inputs.shape[0]) != 1 or int(inputs.shape[1]) != 1:
        return False
    from .lookup_decode import can_rewind

    return can_rewind(prompt_cache)


def _rewind_one(prompt_cache, snap) -> None:
    from mlx_lm.models.cache import trim_prompt_cache

    if snap is None:
        got = trim_prompt_cache(prompt_cache, 1)
        if got != 1:
            raise RuntimeError(f"closed stitch could not rewind: trimmed {got}")
        return
    for c, (kind, saved) in zip(prompt_cache, snap):
        if kind == "arrays":
            c.cache = list(saved)
            continue
        got = c.trim(1)
        if got != 1:
            raise RuntimeError(f"closed stitch could not rewind: trimmed {got}")
        for side, state in saved:
            side.state = state


def _closed_disarm(ring) -> None:
    ring.closed_armed = False
    ring.warm_streak = 0
    ring.closed_cooldown = 32


def _note_warm(orig, model, args, kwargs):
    """Arm closed stitch after a few per-layer decode tokens that read nothing.

    A cold token that misses would otherwise pay a speculative forward and
    then the real one. Flow tokens do not count: flow skips an expert without
    recording a fetch, so a flat miss counter there is not a resident keep set.
    While flow is on it already runs one graph, and stitch stays out of it.
    """
    ring = model._pagedmoe_ring
    cache = model._pagedmoe_cache
    inputs, _prompt_cache = _decode_inputs(args, kwargs)
    decode = (
        inputs is not None
        and getattr(inputs, "ndim", 0) >= 2
        and int(inputs.shape[0]) == 1
        and int(inputs.shape[1]) == 1
        and _DRAFT_STEP is None
    )
    before_m = cache.misses
    before_c = cache.crit_misses
    flow_before = ring.flow_ready()
    logits = orig(model, *args, **kwargs)
    if cache.stitched:
        return logits
    if not decode:
        # A prompt reshuffles residency (and may release the slab), so the
        # warm streak before it says nothing about the next token.
        if inputs is not None and getattr(inputs, "ndim", 0) >= 2 and int(inputs.shape[1]) > 1:
            ring.closed_armed = False
            ring.warm_streak = 0
        return logits
    if flow_before or ring.flow_ready():
        ring.closed_armed = False
        ring.warm_streak = 0
        return logits
    if ring.closed_armed:
        return logits
    if ring.closed_cooldown > 0:
        ring.closed_cooldown -= 1
        ring.warm_streak = 0
        return logits
    if cache.misses == before_m and cache.crit_misses == before_c:
        ring.warm_streak += 1
        if ring.warm_streak >= 4:
            ring.closed_armed = True
            print("[paged-moe] stitch partial: on", flush=True)
    else:
        ring.warm_streak = 0
    return logits


def _closed_call(orig, model, args, kwargs):
    from .lookup_decode import _needs_snapshot, _snapshot

    ring = model._pagedmoe_ring
    cache = model._pagedmoe_cache
    _, prompt_cache = _decode_inputs(args, kwargs)
    snap = _snapshot(prompt_cache) if _needs_snapshot(prompt_cache) else None
    ring.closed_active = True
    ring.closed_broke = False
    ring._closed_miss = None
    try:
        logits = orig(model, *args, **kwargs)
    finally:
        ring.closed_active = False
    miss = ring._closed_miss
    if miss is None or ring.closed_broke:
        if miss is not None or ring.closed_broke:
            cache.closed_replay += 1
            _closed_disarm(ring)
            _rewind_one(prompt_cache, snap)
            return orig(model, *args, **kwargs)
        return logits
    mx.eval(logits, miss)
    if int(miss.item()) == 0:
        cache.closed_ok += 1
        return logits
    cache.closed_replay += 1
    _closed_disarm(ring)
    _rewind_one(prompt_cache, snap)
    return orig(model, *args, **kwargs)


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
    per-expert tensors under '<parent>.experts.<e>.').

    Matches both module-tree paths and `language_model.`-prefixed on-disk keys.
    """
    prefixes: list[str] = []
    for path, _ in find_switch_glus(model):
        for base in tensor_key_candidates(path):
            prefixes.append(f"{base}.")
            prefixes.append(f"{base.rsplit('.', 1)[0]}.experts.")
    prefixes_t = tuple(prefixes)
    return {name for name in tensor_index if name.startswith(prefixes_t)}
