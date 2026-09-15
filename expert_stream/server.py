# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Portions adapted from mlx-lm's HTTP server
# (Copyright MLX Contributors, MIT License). See NOTICE.
"""
PagedMoE OpenAI-compatible HTTP server (experts streamed from disk).

This is mlx-lm's production server with a few changes:
  1. model loader is ours (experts stream from disk)
  2. after each generation we call relieve_pressure() so Metal returns
     recycled pages - without that, request #2 freezes a 48 GB Mac
  3. streamed models generate with an 8-bit quantized KV cache (a 32k-token
     GLM-4.5-Air KV cache is ~10 GB at fp16 - memory we don't have)
  4. memory-safe defaults for the server's own knobs (see _DEFAULT_ARGS):
     bounded prompt-cache, one prompt prefilled at a time, bigger prefill
     chunks so long prompts make fewer passes over the expert mass
  5. user-segment prompt-cache snapshots for thinking models (see
     _snapshot_user_segment): without them, any client that does not replay
     reasoning verbatim (the host app does not) breaks the cached prefix at
     the generation prompt's `<think>` tail every turn, and hybrid-attention
     models (qwen3-next) cannot trim their cache - so every agent turn would
     re-prefill the whole 30k-token conversation from zero.
  6. speculative decoding via EXPERT_STREAM_DRAFT_MODEL (path to a small
     same-tokenizer model) + EXPERT_STREAM_DRAFT_TOKENS. On a streamed MoE
     this is a *disk* optimization as much as a compute one: verifying k
     drafted tokens in one forward reads each layer's expert union once
     (consecutive tokens share ~half their experts) and reads every resident
     weight once for k tokens instead of k times. Greedy output is identical
     to non-speculative decoding (acceptance is exact token match).

Every default can still be overridden by passing the flag explicitly.

Run:

    python -m expert_stream.server --model <path-or-hf-repo> --port 8080
"""

from __future__ import annotations

import copy
import os
import sys
import threading
from collections import deque

import mlx.core as mx
import mlx_lm.server as _mlx_server
from mlx_lm.generate import generation_stream
from mlx_lm.models.cache import LRUPromptCache

from . import config, kvmem
from .loader import load as _streamed_load
from .loader import relieve_pressure

# Injected only when the user didn't pass the flag themselves.
_DEFAULT_ARGS = {
    # A 48 GB machine cannot hold 10 stale KV caches for 32k-context models.
    # 6 (not 2): each request stores up to two prefix snapshots plus its final
    # cache, and evicting this turn's snapshot would defeat the point. The byte
    # bound is the real guard - but mlx-lm only applies --prompt-cache-bytes on
    # the batched request path, which streamed models are never on, so we
    # enforce it ourselves against a model-aware budget (kvmem.bound_store).
    "--prompt-cache-size": "6",
    "--prompt-cache-bytes": "3G",
    # Prefill of a long prompt reads ~the whole expert mass once per chunk, so
    # a 32k prompt in one chunk reads it once instead of four times. The expert
    # cache shrinks itself for the duration (ExpertCache.enter_prefill) to pay
    # for the activation memory a chunk this big needs.
    "--prefill-step-size": str(config.PREFILL_CHUNK),
    # Never prefill two prompts at once (each one streams GBs of experts).
    "--prompt-concurrency": "1",
    "--decode-concurrency": "4",
}

# Speculative decoding knobs, settable via env so the host app's per-model
# env overrides can turn them on without touching the CLI.
_DRAFT_MODEL = os.environ.get("EXPERT_STREAM_DRAFT_MODEL", "").strip()
_DRAFT_TOKENS = os.environ.get("EXPERT_STREAM_DRAFT_TOKENS", "").strip()
if _DRAFT_MODEL:
    _DEFAULT_ARGS["--draft-model"] = os.path.expanduser(_DRAFT_MODEL)
    _DEFAULT_ARGS["--num-draft-tokens"] = _DRAFT_TOKENS or "3"

# Set by the wrapped fetch_nearest_cache; read (once) by the wrapped
# stream_generate on the same generation thread.
_req_ctx = threading.local()

# The streamed model, once loaded. One per server process; the prompt-cache
# patches need it (they are methods on mlx-lm's store, which is model-agnostic
# and is constructed before any model exists).
_model = None

# Don't snapshot a prompt shorter than this - a re-prefill is cheap anyway.
# Env override: EXPERT_STREAM_MIN_SNAPSHOT (set huge to disable snapshots).
_MIN_SNAPSHOT_TOKENS = int(os.environ.get("EXPERT_STREAM_MIN_SNAPSHOT", "1024") or 1024)
# ...but once we're past that, a snapshot is worth taking for even a few
# hundred new tokens: copying a KV cache costs ~0.1 s, while re-prefilling
# 300 tokens of a streamed MoE costs ~2 s. Without incremental snapshots the
# reuse point freezes and every later turn re-prefills everything appended
# since - which is what makes a long agent session get slower and slower.
_MIN_SNAPSHOT_GAIN = 128
# Cap copies per request (each one costs its own KV bytes).
_MAX_SNAPSHOTS = 2

# EXPERT_STREAM_PREFIX_DEBUG=1 logs, per request, how much of the prompt the
# cache could reuse and - when reuse is partial - the decoded text on both
# sides of the divergence point. That is how you find a client that rewrites
# its own history (the difference between a 1 s and a 60 s agent turn).
_PREFIX_DEBUG = bool(int(os.environ.get("EXPERT_STREAM_PREFIX_DEBUG", "0") or 0))
_recent_prompts: deque = deque(maxlen=6)


def _log_prefix_diagnostics(tokenizer, tokens: list, cached: int) -> None:
    best, best_n = None, -1
    for prev in _recent_prompts:
        n = 0
        for a, b in zip(prev, tokens):
            if a != b:
                break
            n += 1
        if n > best_n:
            best, best_n = prev, n
    _recent_prompts.append(list(tokens))

    print(
        f"[prefix] prompt={len(tokens)} reused={cached} "
        f"best_match_vs_recent={max(best_n, 0)}",
        flush=True,
    )
    if best is None or best_n <= 0 or best_n >= len(tokens):
        return
    try:
        head = tokenizer.decode(tokens[max(0, best_n - 60) : best_n])
        old = tokenizer.decode(best[best_n : best_n + 40])
        new = tokenizer.decode(tokens[best_n : best_n + 40])
    except Exception:
        return
    print(f"[prefix]   diverges after: ...{head!r}", flush=True)
    print(f"[prefix]   previously:     {old!r}", flush=True)
    print(f"[prefix]   now:            {new!r}", flush=True)


def _find_think_tail(tokenizer, tokens: list) -> int:
    """Index of an unmatched think-open token in the prompt's last 11 tokens.

    Mirrors mlx-lm's own segment logic (_tokenize): thinking templates end the
    generation prompt with e.g. `<|im_start|>assistant\\n<think>\\n`. Returns
    -1 when there is no such tail.
    """
    if not getattr(tokenizer, "has_thinking", False):
        return -1
    try:
        ts = tokenizer.think_start_id
        te = tokenizer.think_end_id
    except ValueError:
        return -1  # multi-token think markers (gpt-oss style); skip
    for i in range(len(tokens) - 1, max(len(tokens) - 12, -1), -1):
        if tokens[i] == te:
            return -1
        if tokens[i] == ts:
            return i
    return -1


def _last_message_boundary(tokenizer, tokens: list, min_back: int = 512) -> int:
    """Latest message boundary that is at least `min_back` tokens from the end.

    Agent clients park volatile blocks (refreshed task state, project memory,
    reminders) as the last messages before the generation prompt. Snapshotting
    before them stores the part that survives into the next request - the
    cold-start stand-in for the learned divergence point, so turn 2 of a
    session already gets to reuse its prefix instead of re-prefilling.

    `min_back` is the assumed size of that volatile tail: overshooting only
    costs re-prefilling a few hundred tokens.
    """
    eos = getattr(tokenizer, "eos_token_ids", None) or set()
    if not eos:
        return -1
    for i in range(len(tokens) - 1 - min_back, -1, -1):
        if tokens[i] in eos:
            return i + 1
    return -1


def _snapshot_boundaries(tokenizer, tokens: list, cached: int, prompt_len: int) -> list:
    """Token offsets (absolute) at which to snapshot the KV cache.

    Three candidate boundaries, because different clients break prefix reuse
    in different places:

    * **last divergence point** - where the previous request's prompt stopped
      matching this one. Clients that rewrite their own history do it at the
      same structural place every turn, so this *learns* the stable/volatile
      split from actual traffic. Best signal when we have it.
    * **last message boundary** - heuristic version of the above for the
      first turn, when there is no previous prompt to compare against.
    * **think tail** - thinking templates end the prompt with an open
      `<think>`; the cached sequence then continues with generated reasoning
      the client won't replay. Right boundary for plain chat clients that
      append and never rewrite.

    They coincide for simple clients, and dedupe below collapses them.
    """
    tail = _find_think_tail(tokenizer, tokens)
    if tail <= 0:
        # No think-open tail: stop one token short of the end so
        # stream_generate still has input to process.
        tail = len(tokens) - 1
    bounds = {tail}

    prev = _recent_prompts[-1] if _recent_prompts else None
    if prev is not None:
        n = 0
        for a, b in zip(prev, tokens):
            if a != b:
                break
            n += 1
        if n < len(tokens):
            bounds.add(n)
    else:
        msg = _last_message_boundary(tokenizer, tokens)
        if msg > 0:
            bounds.add(msg)

    out = []
    for b in sorted(bounds):
        if b > cached + prompt_len or b < _MIN_SNAPSHOT_TOKENS:
            continue
        gain = b - (out[-1] if out else cached)
        if gain < _MIN_SNAPSHOT_GAIN:
            continue
        out.append(b)
    # Keep the extremes: the earliest is the robust one, the latest saves the
    # most when the client turns out not to rewrite anything.
    if len(out) > _MAX_SNAPSHOTS:
        out = out[: _MAX_SNAPSHOTS - 1] + out[-1:]
    return out


def _dequantize_kv_cache(cache: list, model=None) -> None:
    """Put a reused cache back in fp16 so prefill can use the fused kernel.

    The conversion itself lives in `kvmem.unquantize` (it is a memory problem,
    not a serving one). What belongs here is the sequencing: the fp16 cache is
    the largest allocation this process makes after load, and on a
    full-attention model it does not fit next to a committed expert slab, so the
    expert cache is asked to yield first. Doing it in the other order is the
    asynchronous Metal failure that kills the request.
    """
    expert_cache = getattr(model, "_expert_stream_cache", None) if model else None
    if expert_cache is not None:
        held = sum(getattr(c, "nbytes", 0) for c in cache)
        # fp16 is 2 B/element against the 8-bit cache's ~1.06, so the fp16 cache
        # is a little under twice what this one holds - and `unquantize` releases
        # the quantized bytes as it goes, so the slab only has to cover the
        # difference.
        need = int(held * 2.0)
        if expert_cache.make_room_for(need, freeing=held):
            print(
                f"[paged-moe] expert cache yielded for a {need / 1e9:.1f} GB "
                "fp16 KV dequantize (resumed long context)",
                flush=True,
            )
    kvmem.unquantize(cache)


def _snapshot_user_segment(model, kwargs) -> None:
    """Prefill up to each snapshot boundary, store the cache there, and hand
    only the remainder to stream_generate.

    Storing the cache keyed by a prefix the *next* request will repeat is what
    turns a 30k-token agent turn from a full re-prefill into a few seconds.
    Hybrid-attention models (qwen3-next) can't trim their cache backwards, so
    a stored sequence is useless unless it is an exact prefix - hence storing
    at boundaries rather than relying on trimming.
    """
    ctx_tokens = getattr(_req_ctx, "tokens", None)
    _req_ctx.tokens = None  # single-shot; never reuse across requests
    if ctx_tokens is None:
        return
    tokenizer = kwargs.get("tokenizer")
    prompt = kwargs.get("prompt")
    cache = kwargs.get("prompt_cache")
    if tokenizer is None or prompt is None or cache is None:
        return

    # With speculative decoding the cache list is model layers + draft layers,
    # and a stored snapshot is only useful if BOTH advanced over the same
    # tokens: speculative_generate_step prefills each model with whatever
    # prompt remainder it gets, so a draft cache that skipped the snapshotted
    # span would be conditioned on a hole (correctness is unaffected - the
    # target validates every token - but acceptance, and thus speed, craters).
    draft_model = kwargs.get("draft_model")
    n_model_layers = len(model.layers)
    draft_cache = cache[n_model_layers:] if draft_model is not None else []

    cached = _req_ctx.cached
    bounds = _snapshot_boundaries(tokenizer, ctx_tokens, cached, len(prompt))
    if _PREFIX_DEBUG:
        _log_prefix_diagnostics(tokenizer, ctx_tokens, cached)
        print(f"[prefix]   snapshot boundaries: {[b - cached for b in bounds]}", flush=True)
    else:
        _recent_prompts.append(list(ctx_tokens))
    if not bounds:
        return

    step = kwargs.get("prefill_step_size") or 2048
    progress = kwargs.get("prompt_progress_callback") or (lambda *_: None)
    total = len(prompt)
    store = _req_ctx.store
    budget = getattr(store, "max_bytes", 1 << 62)

    processed = 0
    for bound in bounds:
        target = bound - cached
        while processed < target:
            n = min(step, target - processed)
            chunk = mx.array(prompt[processed : processed + n])
            # Same (thread-local) stream speculative_generate_step runs on.
            # Touching the draft model on the default stream here and on
            # generation_stream inside sgs reliably ended in a Metal
            # "GPU Timeout Error" on the draft's first decode step.
            with mx.stream(generation_stream):
                model(chunk[None], cache=cache)
                mx.eval([c.state for c in cache[:n_model_layers]])
                if draft_model is not None:
                    # Tiny + resident: this costs milliseconds. Bounded evals,
                    # separate from the main model's command buffers.
                    for i in range(0, n, 2048):
                        draft_model(chunk[i : i + 2048][None], cache=draft_cache)
                        mx.eval([c.state for c in draft_cache])
            processed += n
            progress(processed, total)
            mx.clear_cache()

        # A snapshot costs its own KV bytes. Skip (rather than thrash the
        # prompt cache) when one copy would eat over half the byte budget -
        # full-attention models at long context land here.
        nbytes = sum(c.nbytes for c in cache)
        if nbytes * 2 > budget:
            if _PREFIX_DEBUG:
                print(f"[prefix]   snapshot skipped: {nbytes/1e9:.2f} GB vs budget", flush=True)
            continue
        store.insert_cache(
            _req_ctx.model_key,
            list(ctx_tokens[:bound]),
            copy.deepcopy(cache),
            cache_type="user",
        )
        mx.clear_cache()

    kwargs["prompt"] = prompt[processed:]
    if _PREFIX_DEBUG:
        # What the generator actually starts from. A remainder of 0, or an
        # offset that disagrees with cached+processed, means the model never
        # sees part of its own prompt - which reads as fluent, off-topic output.
        print(
            f"[prefix]   handoff: cache_offset={getattr(cache[0], 'offset', None)} "
            f"expected={cached + processed} remainder={len(kwargs['prompt'])} "
            f"tail={tokenizer.decode(ctx_tokens[-24:])!r}",
            flush=True,
        )


def install_patches(model=None):
    """Apply every mlx-lm patch a streamed model needs, and return nothing.

    Split out of `main()` so `bench/session_memory.py` can replay the real
    serving path - prompt-cache handover, KV conversion, snapshotting, the
    memory guards - instead of an approximation of it. A memory benchmark that
    measures a different code path than production is worse than no benchmark.

    `model` short-circuits the ModelProvider hook for callers that loaded the
    model themselves.
    """
    global _model
    if model is not None:
        _model = model

    # Streamed models must use the single-request path: the batch engine
    # bypasses stream_generate (so no KV quantization, no relieve_pressure)
    # and would prefill multiple prompts concurrently - each of which streams
    # gigabytes of experts.
    _orig_provider_load = _mlx_server.ModelProvider.load

    def _provider_load(self, *args, **kwargs):
        global _model
        result = _orig_provider_load(self, *args, **kwargs)
        if getattr(self.model, "_expert_stream_cache", None) is not None:
            self.is_batchable = False
            _model = self.model
        return result

    _mlx_server.ModelProvider.load = _provider_load

    # Record which stored prefix the request resumed from, so the
    # stream_generate wrapper knows where the user segment ends in `rest`
    # coordinates and where to store the snapshot.
    _orig_fetch = LRUPromptCache.fetch_nearest_cache

    def _fetch(self, model_key, tokens):
        if config.PROMPT_CACHE_MOVE and _model is not None:
            cache, rest = kvmem.take_nearest_cache(self, model_key, tokens)
        else:
            cache, rest = _orig_fetch(self, model_key, tokens)
        if cache is not None and len(rest) > 1:
            _dequantize_kv_cache(cache, _model)
        _req_ctx.store = self
        _req_ctx.model_key = model_key
        _req_ctx.tokens = list(tokens)
        _req_ctx.cached = len(tokens) - len(rest)
        return cache, rest

    LRUPromptCache.fetch_nearest_cache = _fetch

    # Enforce the byte budget mlx-lm only applies to batched requests. Without
    # this the store is bounded by --prompt-cache-size alone, i.e. by a *count*
    # of KV caches: six of GLM-4.7's at 24k tokens is 29 GB.
    _orig_insert = LRUPromptCache.insert_cache

    def _insert_cache(self, model_key, tokens, prompt_cache, **kwargs):
        _orig_insert(self, model_key, tokens, prompt_cache, **kwargs)
        freed = kvmem.bound_store(self)
        if freed and config.MEM_DEBUG:
            print(
                f"[paged-moe mem] prompt cache trimmed {freed / 1e9:.2f} GB "
                f"-> {self.nbytes / 1e9:.2f} GB",
                flush=True,
            )

    LRUPromptCache.insert_cache = _insert_cache

    # Convert the KV cache one layer at a time instead of building both formats
    # of the whole thing as one lazy graph (13.8 GB on GLM-4.7 at 24k, where
    # 4.8 GB is the steady state) - and immediately after a long prefill, next
    # to a slab the decode path has just rebuilt.
    # generate_step re-resolves this global into a functools.partial on every
    # call, so replacing it here covers the speculative path too.
    import mlx_lm.generate as _mlx_generate

    _mlx_generate.maybe_quantize_kv_cache = kvmem.requantize

    # Wrap stream_generate so every HTTP completion (a) uses a quantized KV
    # cache when the model is streamed, (b) snapshots the user segment for
    # thinking models, and (c) returns Metal pages afterwards while keeping
    # the expert LRU warm for the next request.
    _orig_stream = _mlx_server.stream_generate

    def _stream_generate(model, *args, **kwargs):
        if getattr(model, "_expert_stream_cache", None) is not None:
            kwargs.setdefault("kv_bits", config.KV_BITS)
            kwargs.setdefault("kv_group_size", config.KV_GROUP_SIZE)
            # Quantize only once the prompt is in: a quantized KV cache drops
            # attention onto mlx-lm's Python fallback, which materializes the
            # whole [heads, chunk, keys] score matrix. That is fine for decode
            # (one query row) and fatal for prefill - a 32k chunk against a 64k
            # context asks Metal for ~68 GB against a ~28 GB max buffer size,
            # which aborts the process. Prefill therefore stays on the fused
            # causal kernel, and decode still gets the smaller cache.
            # kwargs["prompt"] is only the uncached remainder, so the threshold
            # has to be the whole context length.
            prompt = kwargs.get("prompt")
            ctx_len = max(
                len(prompt) if prompt is not None else 0,
                len(getattr(_req_ctx, "tokens", None) or ()),
            )
            if ctx_len:
                # Below KV_FP16_CTX the cache stays fp16: quantized-KV
                # attention adds per-layer graph ops whose encode cost lands
                # on every one of the model's per-layer syncs (+8% decode on
                # Qwen3-235B at short context). Past the threshold it
                # quantizes mid-decode, one-time, and memory wins again.
                kwargs.setdefault(
                    "quantized_kv_start", max(ctx_len + 1, config.KV_FP16_CTX)
                )
        try:
            _snapshot_user_segment(model, kwargs)
        except Exception as e:  # snapshots are an optimization, never fatal
            print(f"[paged-moe] user-segment snapshot skipped: {e!r}", flush=True)
        # Speculative decoding: hand sgs a fully-materialized uint32 prompt.
        # A Python list becomes a lazy int64->uint32 astype rooted on the
        # default stream; when the remainder is tiny (post-snapshot) nothing
        # evaluates it until it enters the draft's async decode pipeline on
        # generation_stream - that cross-stream dependency, interleaved with
        # the expert-streaming worker evals, deterministically stalls a Metal
        # command buffer past the watchdog ("GPU Timeout Error").
        if kwargs.get("draft_model") is not None:
            prompt = kwargs.get("prompt")
            if prompt is not None and not isinstance(prompt, mx.array):
                p = mx.array(prompt, mx.uint32)
                mx.eval(p)
                kwargs["prompt"] = p
        try:
            yield from _orig_stream(model, *args, **kwargs)
        except RuntimeError as e:
            # Metal OOM mid-decode/prefill: free everything we can so the
            # *next* request has a chance, then surface a clear message.
            # (mlx-lm's request thread otherwise just dies with the raw
            # kIOGPUCommandBufferCallbackErrorOutOfMemory string.)
            msg = str(e)
            if "Insufficient Memory" in msg or "OutOfMemory" in msg:
                cache = getattr(model, "_expert_stream_cache", None)
                if cache is not None:
                    # Hand back speculative staging and the slab, so KV and
                    # activations have somewhere to live next turn. It has to be
                    # the *slab*: with slabs on, lowering budget_bytes and
                    # evicting frees no Metal memory at all - the slots are
                    # committed either way - so the obvious recovery is a no-op
                    # exactly when recovery matters. leave_prefill rebuilds it
                    # around whatever the KV cache ended up needing.
                    try:
                        cache.relieve_pressure()
                        cache.enter_prefill(int(config.PREFILL_CACHE_GB * (1 << 30)))
                    except Exception:
                        pass
                print(
                    "[paged-moe] Metal OOM during generation - released the "
                    "expert slab; the next turn rebuilds it around the KV cache. "
                    "If this keeps happening, lower numCtx (which is also the KV "
                    "reserve) or EXPERT_STREAM_CACHE_GB for this model.",
                    flush=True,
                )
            raise
        finally:
            relieve_pressure(model)

    _mlx_server.stream_generate = _stream_generate


def main():
    _mlx_server.load = _streamed_load

    for flag, value in _DEFAULT_ARGS.items():
        if flag not in sys.argv:
            sys.argv.extend([flag, value])

    install_patches()
    _mlx_server.main()


if __name__ == "__main__":
    main()
