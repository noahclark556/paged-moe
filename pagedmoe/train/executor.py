# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Forward / backward over a model whose experts may live on disk.

Two executors, one interface.

FusedExecutor
    One value_and_grad per micro-batch over the whole model, optionally with
    per-layer gradient checkpointing. Fastest when everything is resident.
    Not usable on a streamed model: MLX builds the whole backward graph
    before evaluating it, so every layer's experts would be held at once.

LayeredExecutor
    Layer-major. The forward pass runs layer 0 for every micro-batch, then
    layer 1, and so on, keeping each layer's input (one hidden state per
    token per layer). The backward pass walks the layers in reverse and
    recomputes each one under its own vjp. Each layer's experts are needed
    for one short window, so a step reads the expert mass about twice no
    matter how many micro-batches it holds: the prefill lesson again, where
    cost is priced in expert bytes and the trick is to touch them as few
    times as possible.

    Because the head runs after every micro-batch's forward pass, the loss
    sees the whole step's token logprobs at once. Global normalizers (token
    means across the step, group statistics) are exact rather than an
    accumulation approximation.

Both are architecture agnostic. A dry call with every decoder layer swapped
for a tap records what the model passes each layer (hidden state, masks,
per-layer inputs); lazy evaluation means the dry call computes nothing. The
head is reached the same way, by injecting a hidden state at layer 0 and
letting identity taps carry it to the model's own norm and lm_head, so
softcapping, logit scaling and tied embeddings all come for free.

Losses are written against per-token logprobs of the targets. The layered
executor computes them in chunks (a 250k vocabulary is 1 MB of logits per
token in fp32), asks the objective for d loss / d logprob, and backprops the
head chunk by chunk with those weights. The full logits never exist at once.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map_with_path, tree_unflatten

from . import experts as train_experts


@dataclass
class Batch:
    """One micro-batch, already on device.

    tokens   [B, T] int32  input ids (placeholders where embeds go)
    targets  [B, T] int32  next-token targets per position
    mask     [B, T] f32    which positions count toward the loss
    embeds   [B, P, H]     optional inputs injected at `embeds_at`
    meta     objective extras: advantages, pair layout, reference logprobs...
    rows     the source rows (for encoders and reward functions)
    """

    tokens: mx.array
    targets: mx.array
    mask: mx.array
    embeds: mx.array | None = None
    embeds_at: int = 0
    meta: dict = field(default_factory=dict)
    rows: list | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.tokens.shape[0]), int(self.tokens.shape[1])

    @property
    def n_tokens(self) -> int:
        return int(self.mask.sum().item())


LossFn = Callable[[mx.array, Batch], "tuple[mx.array, dict]"]


class UnsupportedModel(RuntimeError):
    pass


class _Tap:
    """Stands in for a decoder layer during a dry or head call."""

    def __init__(self, layer, fn):
        self.__dict__["_layer"] = layer
        self.__dict__["_fn"] = fn

    def __getattr__(self, name):
        return getattr(self._layer, name)

    def __call__(self, *args, **kwargs):
        return self._fn(args, kwargs)


def _live_layers(model) -> list:
    """The list the forward pass reads its decoder layers from.

    Layers are swapped in place (taps), so this has to be the list itself, not
    a copy. mlx-lm's pipeline models (GLM-4, DeepSeek V2 / V3, Kimi-K2) return
    a fresh slice of model.model.layers on every access; their forward slices
    it again each call, so the inner list is the live one as long as the
    slice is all of it (no pipeline split).
    """
    outer = model.layers
    if model.layers is outer:
        return outer
    inner = getattr(getattr(model, "model", None), "layers", None)
    if (isinstance(inner, list) and len(inner) == len(outer)
            and all(a is b for a, b in zip(inner, outer))):
        return inner
    raise UnsupportedModel("model.layers must return the live layer list")


def _hidden_arg(args, kwargs):
    if args:
        return args[0]
    for k in ("x", "hidden_states", "h", "inputs"):
        if k in kwargs:
            return kwargs[k]
    raise UnsupportedModel("decoder layer was not called with a hidden state first")


def _with_hidden(args, kwargs, x):
    if args:
        return (x,) + tuple(args[1:]), kwargs
    kw = dict(kwargs)
    for k in ("x", "hidden_states", "h", "inputs"):
        if k in kw:
            kw[k] = x
            return args, kw
    raise UnsupportedModel("decoder layer was not called with a hidden state first")


def _logp_of(logits: mx.array, targets: mx.array) -> mx.array:
    logits = logits.astype(mx.float32)
    picked = mx.take_along_axis(logits, targets[..., None], axis=-1).squeeze(-1)
    return picked - mx.logsumexp(logits, axis=-1)


def _add_into(acc: dict, flat: list, prefix: str = "") -> None:
    for k, v in flat:
        key = prefix + k
        acc[key] = v if key not in acc else acc[key] + v


class _Base:
    def __init__(
        self,
        model,
        *,
        encoder: nn.Module | None = None,
        head_chunk: int = 512,
        on_layer: Callable[[str, int, int], None] | None = None,
    ):
        self.model = model
        self.encoder = encoder
        self.head_chunk = max(16, int(head_chunk))
        self.on_layer = on_layer
        self.layers = _live_layers(model)
        self.L = len(self.layers)
        self.prefix = self._layers_prefix()
        self.switch_of = self._switches_by_layer()
        self.timing: dict[str, float] = {}
        self.boundary_bytes: int | None = None

    # ----------------------------------------------------------- structure

    def _layers_prefix(self) -> str:
        first = self.layers[0]
        for name, m in self.model.named_modules():
            if m is first:
                return name[: -len("0")]
        raise UnsupportedModel("could not locate the decoder layers in the module tree")

    def _switches_by_layer(self) -> list[list]:
        out = []
        for layer in self.layers:
            found = [
                m for _, m in layer.named_modules()
                if isinstance(m, train_experts.TrainSwitch)
            ]
            out.append(found)
        return out

    def _taps(self, fns):
        saved = list(self.layers)
        for i, fn in enumerate(fns):
            self.layers[i] = _Tap(saved[i], fn)
        return saved

    def _untap(self, saved):
        for i, layer in enumerate(saved):
            self.layers[i] = layer

    # ------------------------------------------------------------- capture

    def capture(self, batch: Batch):
        """(x0, [(args, kwargs) per layer]) for this batch. Computes nothing."""
        rec: list = []

        def tap(args, kwargs):
            rec.append((args, kwargs))
            return _hidden_arg(args, kwargs)

        saved = self._taps([tap] * self.L)
        try:
            self.model(batch.tokens)
        finally:
            self._untap(saved)
        if len(rec) != self.L:
            raise UnsupportedModel(
                f"model called {len(rec)} of {self.L} decoder layers in a dry pass"
            )
        x0 = _hidden_arg(*rec[0])
        # bytes per token at a layer boundary, as the model really passes it
        self.boundary_bytes = int(x0.shape[-1]) * x0.dtype.size
        return x0, rec

    def inputs(self, batch: Batch, x0: mx.array, enc_params=None) -> mx.array:
        """x0 with the batch's embeds (or the encoder's output) spliced in."""
        emb = batch.embeds
        if self.encoder is not None:
            if enc_params is not None:
                self.encoder.update(enc_params)
            emb = self.encoder(batch)
        if emb is None:
            return x0
        emb = emb.astype(x0.dtype)
        a = int(batch.embeds_at)
        p = int(emb.shape[1])
        parts = [x0[:, :a], emb, x0[:, a + p :]]
        return mx.concatenate([q for q in parts if q.shape[1]], axis=1)

    def head(self, h: mx.array) -> mx.array:
        """Logits for hidden states h [B, T, H], through the model's own head."""
        B, T = int(h.shape[0]), int(h.shape[1])

        def inject(args, kwargs):
            return h

        def carry(args, kwargs):
            return _hidden_arg(args, kwargs)

        saved = self._taps([inject] + [carry] * (self.L - 1))
        try:
            return self.model(mx.zeros((B, T), dtype=mx.int32))
        finally:
            self._untap(saved)

    def run_layer(self, i: int, x: mx.array, rec) -> mx.array:
        args, kwargs = _with_hidden(*rec[i], x)
        y = self.layers[i](*args, **kwargs)
        if not isinstance(y, mx.array):
            raise UnsupportedModel(
                f"decoder layer {i} returned {type(y).__name__}, expected one array"
            )
        return y

    def outer_params(self) -> list:
        """Trainable parameters outside the decoder layers (head / embed LoRA)."""
        flat = tree_flatten(self.model.trainable_parameters())
        return [(k, v) for k, v in flat if not k.startswith(self.prefix)]

    def grads_tree(self, flat: dict):
        def pick(path, v):
            g = flat.get(path)
            return mx.zeros_like(v) if g is None else g

        return tree_map_with_path(pick, self.model.trainable_parameters())

    # --------------------------------------------------------- no-grad pass

    def logps(self, batch: Batch) -> mx.array:
        """Per-token logprobs of the targets, layer by layer (bounded memory)."""
        x0, rec = self.capture(batch)
        x = self.inputs(batch, x0)
        mx.eval(x)
        for i in range(self.L):
            x = self.run_layer(i, x, rec)
            mx.eval(x)
        return self.head_logps(x, batch.targets)

    def logps_many(self, batches: list[Batch]) -> list[mx.array]:
        """Target logprobs for many batches in one layer-major pass.

        Only the current layer's activations are alive, so the group can be
        large, and a streamed model reads each layer's experts once for the
        whole group instead of once per batch.
        """
        caps = [self.capture(b) for b in batches]
        xs = []
        for b, (x0, _rec) in zip(batches, caps):
            x = self.inputs(b, x0)
            mx.eval(x)
            xs.append(x)
        for i in range(self.L):
            for j in range(len(batches)):
                xs[j] = self.run_layer(i, xs[j], caps[j][1])
                mx.eval(xs[j])
            if self.on_layer:
                self.on_layer("score", i, self.L)
        return [self.head_logps(x, b.targets) for x, b in zip(xs, batches)]

    def head_logps(self, h: mx.array, targets: mx.array) -> mx.array:
        out = []
        T = int(h.shape[1])
        for s in range(0, T, self.head_chunk):
            e = min(T, s + self.head_chunk)
            lp = _logp_of(self.head(h[:, s:e]), targets[:, s:e])
            mx.eval(lp)
            out.append(lp)
        return mx.concatenate(out, axis=1) if len(out) > 1 else out[0]

    def _tick(self, key: str, t0: float) -> float:
        now = time.perf_counter()
        self.timing[key] = self.timing.get(key, 0.0) + (now - t0)
        return now


class FusedExecutor(_Base):
    """One graph per micro-batch. Resident models only."""

    def __init__(self, model, *, checkpoint: bool = False, **kw):
        super().__init__(model, **kw)
        if any(self.switch_of):
            raise UnsupportedModel(
                "the fused executor needs every expert resident; use executor=layered"
            )
        self.checkpoint = checkpoint

    def _layer(self, i, x, rec):
        if not self.checkpoint:
            return self.run_layer(i, x, rec)
        layer = self.layers[i]

        def inner(p, x):
            layer.update(p)
            return self.run_layer(i, x, rec)

        return mx.checkpoint(inner)(layer.trainable_parameters(), x)

    def _head_graph(self, x, targets, plain: bool):
        """Target logprobs inside the graph. Chunks are checkpointed, so only
        one chunk of [tokens, vocab] logits exists at a time in either pass."""
        if plain:
            return _logp_of(self.head(x), targets)
        T = int(x.shape[1])
        parts = []
        for s in range(0, T, self.head_chunk):
            e = min(T, s + self.head_chunk)
            tc = targets[:, s:e]

            def chunk(hc, tc=tc):
                return _logp_of(self.head(hc), tc)

            parts.append(mx.checkpoint(chunk)(x[:, s:e]))
        return mx.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

    def step(self, batches: list[Batch], loss_fn: LossFn):
        self.timing = {}
        grads: dict = {}
        enc_grads: dict = {}
        total = 0.0
        metrics: dict[str, float] = {}
        model = self.model
        enc = self.encoder
        # Head LoRA would be a closure inside the checkpointed chunks, which
        # mx.checkpoint does not differentiate; keep the plain graph then.
        plain_head = bool(self.outer_params())

        for mb in batches:
            t0 = time.perf_counter()
            x0, rec = self.capture(mb)

            def f(params, enc_params):
                model.update(params)
                x = self.inputs(mb, x0, enc_params)
                for i in range(self.L):
                    x = self._layer(i, x, rec)
                lp = self._head_graph(x, mb.targets, plain_head)
                loss, m = loss_fn(lp, mb)
                return loss, m

            enc_p = enc.trainable_parameters() if enc is not None else {}
            (loss, m), (g, ge) = mx.value_and_grad(f, argnums=(0, 1))(
                model.trainable_parameters(), enc_p
            )
            mx.eval(loss, g, ge, m)
            _add_into(grads, tree_flatten(g))
            if enc is not None:
                _add_into(enc_grads, tree_flatten(ge))
            total += float(loss.item())
            for k, v in m.items():
                metrics[k] = metrics.get(k, 0.0) + float(v.item())
            self._tick("fused", t0)
        return total, self.grads_tree(grads), enc_grads, metrics


class LayeredExecutor(_Base):
    """Layer-major forward, reverse layer-major recompute + vjp."""

    def __init__(self, model, *, seed: int = 0, **kw):
        super().__init__(model, **kw)
        self.seed = int(seed)
        self.step_index = 0
        self._union_prev: list[set] = [set() for _ in range(self.L)]

    def _rng(self, i: int, j: int) -> None:
        # Dropout must draw the same mask in the forward pass and the recompute.
        mx.random.seed((self.seed * 1_000_003 + self.step_index * 10_007 + i * 101 + j) % (1 << 31))

    def _expect(self, i: int, ids: set) -> None:
        if 0 <= i < self.L and ids:
            for sw in self.switch_of[i]:
                sw.expect(ids)

    def _union(self, i: int) -> set:
        u: set = set()
        for sw in self.switch_of[i]:
            u.update(int(e) for e in sw.last_ids)
        return u

    def step(self, batches: list[Batch], loss_fn: LossFn):
        self.timing = {}
        M = len(batches)
        enc = self.encoder
        t0 = time.perf_counter()

        caps = [self.capture(mb) for mb in batches]
        xs = []
        for mb, (x0, _rec) in zip(batches, caps):
            x = self.inputs(mb, x0)
            mx.eval(x)
            xs.append(x)
        t0 = self._tick("capture", t0)

        # ---- forward, layer-major. hs[i][j] = input of layer i, micro-batch j.
        hs: list[list[mx.array]] = [xs]
        union_fwd: list[set] = [set() for _ in range(self.L)]
        for i in range(self.L):
            outs = []
            for j in range(M):
                self._rng(i, j)
                y = self.run_layer(i, hs[i][j], caps[j][1])
                if j == 0 and i + 1 < self.L:
                    # Reads for the next layer ride this layer's compute.
                    self._expect(i + 1, self._union_prev[i + 1])
                mx.eval(y)
                union_fwd[i] |= self._union(i)
                outs.append(y)
            hs.append(outs)
            if self.on_layer:
                self.on_layer("forward", i, self.L)
        t0 = self._tick("forward", t0)

        # ---- head + loss over the whole step.
        lps = [self.head_logps(hs[self.L][j], batches[j].targets) for j in range(M)]
        total = 0.0
        metrics: dict[str, float] = {}
        weights = []
        for lp, mb in zip(lps, batches):
            def f(lp_):
                return loss_fn(lp_, mb)

            (loss, m), w = mx.value_and_grad(f)(lp)
            mx.eval(loss, w, m)
            total += float(loss.item())
            for k, v in m.items():
                metrics[k] = metrics.get(k, 0.0) + float(v.item())
            weights.append(w)
        t0 = self._tick("loss", t0)

        grads: dict = {}
        outer = self.outer_params()
        gs = []
        for j in range(M):
            g, og = self._head_backward(hs[self.L][j], batches[j].targets, weights[j], outer)
            _add_into(grads, og)
            gs.append(g)
        hs[self.L] = None
        t0 = self._tick("head", t0)

        # ---- backward, reverse layer-major.
        for i in reversed(range(self.L)):
            layer = self.layers[i]
            params = layer.trainable_parameters()
            has_params = bool(tree_flatten(params))
            for j in range(M):
                self._rng(i, j)
                rec = caps[j][1]
                g_out = gs[j]

                def f(p, x):
                    layer.update(p)
                    y = self.run_layer(i, x, rec)
                    return (y.astype(mx.float32) * g_out.astype(mx.float32)).sum()

                if has_params:
                    _, (gp, gx) = mx.value_and_grad(f, argnums=(0, 1))(params, hs[i][j])
                else:
                    gp = None
                    gx = mx.grad(lambda x: f(params, x))(hs[i][j])
                if j == 0:
                    self._expect(i - 1, union_fwd[i - 1] if i > 0 else set())
                if gp is not None:
                    flat = tree_flatten(gp)
                    mx.eval(gx, [v for _, v in flat])
                    _add_into(grads, flat, prefix=f"{self.prefix}{i}.")
                else:
                    mx.eval(gx)
                gs[j] = gx
            if has_params:
                layer.update(params)
            hs[i + 1] = None
            if self.on_layer:
                self.on_layer("backward", i, self.L)
        t0 = self._tick("backward", t0)

        enc_grads: dict = {}
        if enc is not None:
            for j, mb in enumerate(batches):
                x0 = caps[j][0]
                gx0 = gs[j]

                def fe(ep):
                    x = self.inputs(mb, x0, ep)
                    return (x.astype(mx.float32) * gx0.astype(mx.float32)).sum()

                ge = mx.grad(fe)(enc.trainable_parameters())
                mx.eval(ge)
                _add_into(enc_grads, tree_flatten(ge))
            self._tick("encoder", t0)

        self._union_prev = union_fwd
        self.step_index += 1
        return total, self.grads_tree(grads), enc_grads, metrics

    def _head_backward(self, h, targets, w, outer):
        T = int(h.shape[1])
        parts = []
        acc: dict = {}
        model = self.model
        keys = [k for k, _ in outer]
        for s in range(0, T, self.head_chunk):
            e = min(T, s + self.head_chunk)
            hc = h[:, s:e]
            tc = targets[:, s:e]
            wc = w[:, s:e]

            def f(op, hc_):
                if op:
                    model.update(tree_unflatten(list(zip(keys, op))))
                return (_logp_of(self.head(hc_), tc) * wc).sum()

            if outer:
                vals = [v for _, v in outer]
                _, (gop, ghc) = mx.value_and_grad(f, argnums=(0, 1))(vals, hc)
                mx.eval(ghc, gop)
                _add_into(acc, list(zip(keys, gop)))
            else:
                ghc = mx.grad(lambda x: f(None, x), argnums=0)(hc)
                mx.eval(ghc)
            parts.append(ghc)
        if outer:
            model.update(tree_unflatten(outer))
        g = mx.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        return g, list(acc.items())
