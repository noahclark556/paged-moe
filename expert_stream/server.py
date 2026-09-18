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
     the generation prompt's `<think>` tail every turn. Trimmable caches
     (DeepSeek / Qwen3-MoE / GLM) prefill once then fork prefixes via
     deepcopy+trim; hybrid-attention models (qwen3-next) still stop mid-prefill
     because they cannot trim backwards.
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
import json
import os
import re
import sys
import threading
import time
from collections import deque
from typing import Any, Optional

import mlx.core as mx
import mlx_lm.server as _mlx_server
from mlx_lm.generate import generation_stream
from mlx_lm.models.cache import (
    LRUPromptCache,
    can_trim_prompt_cache,
    trim_prompt_cache,
)

from . import adaptive_prefill, config, kvmem, lookup_decode, prefill_fused
from .loader import load as _streamed_load
from .loader import relieve_pressure

# DeepSeek enable_thinking -> thinking_mode shim installs via loader import.

# mlx-lm has no deepseek_v32 tool_parser yet, so has_tool_calling stays false
# and the server only warns + skips structured tool_calls. The chat template
# still renders tool schemas; wire DSML parsing so act-mode tools work.
_DSML_TOKEN = "｜DSML｜"
_DSML_TOOL_CALL_START = f"<{_DSML_TOKEN}function_calls>"
_DSML_TOOL_CALL_END = f"</{_DSML_TOKEN}function_calls>"
_DSML_INVOKE_RE = re.compile(
    rf"<{re.escape(_DSML_TOKEN)}invoke\s+name=\"([^\"]+)\">(.*?)"
    rf"</{re.escape(_DSML_TOKEN)}invoke>",
    re.DOTALL,
)
_DSML_PARAM_RE = re.compile(
    rf"<{re.escape(_DSML_TOKEN)}parameter\s+name=\"([^\"]+)\""
    rf"(?:\s+string=\"(true|false)\")?\s*>(.*?)"
    rf"</{re.escape(_DSML_TOKEN)}parameter>",
    re.DOTALL,
)


def _parse_deepseek_dsml_tool_call(
    text: str, tools: Optional[list[Any]] = None
) -> list[dict]:
    del tools  # schema-guided casting not required; JSON/literal decode below
    body = text
    if _DSML_TOOL_CALL_START in body:
        start = body.find(_DSML_TOOL_CALL_START) + len(_DSML_TOOL_CALL_START)
        end = body.find(_DSML_TOOL_CALL_END, start)
        body = body[start : end if end >= 0 else None]
    out: list[dict] = []
    for match in _DSML_INVOKE_RE.finditer(body):
        name = match.group(1).strip()
        args: dict[str, Any] = {}
        for pm in _DSML_PARAM_RE.finditer(match.group(2)):
            key = pm.group(1).strip()
            is_str = (pm.group(2) or "true") == "true"
            raw = pm.group(3).strip()
            if is_str:
                args[key] = raw
            else:
                try:
                    args[key] = json.loads(raw)
                except json.JSONDecodeError:
                    args[key] = raw
        out.append({"name": name, "arguments": args})
    if not out:
        raise ValueError("no DSML invoke blocks found")
    return out


def _json_tool_body_candidates(text: str) -> list[str]:
    """Bodies mlx-lm json_tools should try before giving up.

    Qwen2.5's shipped chat template prints the example as
    `{{"name": ..., "arguments": ...}}`. json.loads then fails at column 2
    (`Expecting property name`) and the server drops the call, so ga sees an
    empty turn. Strip that extra brace pair; also try the outermost JSON
    object if the model wrapped it in junk.
    """
    s = (text or "").strip()
    out = [s]
    if s.startswith("{{") and s.endswith("}}"):
        out.append(s[1:-1].strip())
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        slice_ = s[i : j + 1]
        out.append(slice_)
        if slice_.startswith("{{") and slice_.endswith("}}"):
            out.append(slice_[1:-1].strip())
    # Dedup while keeping order.
    seen: set[str] = set()
    uniq: list[str] = []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def _repair_json_noise(s: str) -> str:
    """Cheap fixes for common model JSON mistakes (not a full JSON5 parser)."""
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace(
        "\u2019", "'"
    )
    # Trailing commas before } or ].
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    return s


def _loads_tool_obj(s: str) -> dict | None:
    for cand in (s, _repair_json_noise(s)):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            return obj
        # First complete object if there is trailing junk.
        try:
            obj, _end = json.JSONDecoder().raw_decode(cand.lstrip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    # Python-literal shape: {'name': '...', 'arguments': {...}}
    try:
        import ast

        obj = ast.literal_eval(_repair_json_noise(s))
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None


def _salvage_tool_call(text: str) -> dict | None:
    """Pull name (+ arguments when possible) out of broken tool JSON.

    mlx-lm's tool state machine never puts the raw markup into `content`, so a
    failed parse is an empty turn for ga. Returning a name with best-effort
    args is better than dropping the call.
    """
    s = (text or "").strip()
    m = re.search(r'"name"\s*:\s*"((?:\\.|[^"\\])*)"', s)
    if not m:
        m = re.search(r"'name'\s*:\s*'([^']*)'", s)
    if not m:
        return None
    name = m.group(1)
    try:
        name = json.loads(f'"{name}"')
    except json.JSONDecodeError:
        pass
    if not isinstance(name, str) or not name.strip():
        return None
    name = name.strip()
    args: dict = {}
    am = re.search(r'["\']arguments["\']\s*:\s*', s)
    if am:
        rest = s[am.end() :].lstrip()
        for attempt in (rest, _repair_json_noise(rest)):
            try:
                parsed, _ = json.JSONDecoder().raw_decode(attempt)
                if isinstance(parsed, dict):
                    args = parsed
                    break
            except json.JSONDecodeError:
                continue
    return {"name": name, "arguments": args}


def _normalize_tool_obj(obj: dict) -> dict | None:
    name = obj.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    args = obj.get("arguments", {})
    if isinstance(args, str):
        loaded = _loads_tool_obj(args)
        args = loaded if isinstance(loaded, dict) else {}
    elif not isinstance(args, dict):
        args = {}
    return {"name": name.strip(), "arguments": args}


def _wrap_json_tools_parser(orig):
    """Retry / repair / salvage after stock json_tools.parse_tool_call fails."""

    def parse_tool_call(text, tools=None):
        try:
            return orig(text, tools)
        except (json.JSONDecodeError, ValueError, TypeError) as first:
            last: Exception = first
            for cand in _json_tool_body_candidates(text):
                obj = _loads_tool_obj(cand)
                if obj is None:
                    continue
                norm = _normalize_tool_obj(obj)
                if norm is not None:
                    return norm
                last = ValueError("tool JSON missing name")
            salvaged = _salvage_tool_call(text)
            if salvaged is not None:
                preview = (text or "").replace("\n", "\\n")
                if len(preview) > 400:
                    preview = preview[:400] + "..."
                print(
                    f"[paged-moe] tool JSON repaired via salvage "
                    f"({type(first).__name__}: {first}); body={preview!r}",
                    flush=True,
                )
                return salvaged
            preview = (text or "").replace("\n", "\\n")
            if len(preview) > 600:
                preview = preview[:600] + "..."
            print(
                f"[paged-moe] tool JSON unrecoverable ({type(first).__name__}: {first}); "
                f"body={preview!r}",
                flush=True,
            )
            raise last

    parse_tool_call._paged_moe_json_wrap = True  # type: ignore[attr-defined]
    return parse_tool_call


def _install_json_tools_unwrap(tokenizer) -> None:
    """Harden mlx-lm's json_tools parser. No-op for qwen3_coder / glm / DSML."""
    orig = getattr(tokenizer, "_tool_parser", None)
    if orig is None or getattr(orig, "_paged_moe_json_wrap", False):
        return
    mod = getattr(orig, "__module__", "") or ""
    if not mod.endswith("json_tools"):
        return
    tokenizer._tool_parser = _wrap_json_tools_parser(orig)
    print(
        "[paged-moe] json_tools: unwrap/repair/salvage Qwen2.5 tool bodies",
        flush=True,
    )


def _install_deepseek_tool_parser(tokenizer) -> None:
    if getattr(tokenizer, "has_tool_calling", False):
        return
    # TokenizerWrapper stores these privately; mirror mlx_lm.tokenizer_utils.load.
    tokenizer._tool_parser = _parse_deepseek_dsml_tool_call
    tokenizer._tool_call_start = _DSML_TOOL_CALL_START
    tokenizer._tool_call_end = _DSML_TOOL_CALL_END
    try:
        tokenizer._tool_call_start_tokens = tuple(
            tokenizer.encode(_DSML_TOOL_CALL_START, add_special_tokens=False)
        )
        tokenizer._tool_call_end_tokens = tuple(
            tokenizer.encode(_DSML_TOOL_CALL_END, add_special_tokens=False)
        )
    except Exception as e:
        print(f"[paged-moe] deepseek DSML tool tokens skipped: {e!r}", flush=True)
        return
    print(
        "[paged-moe] deepseek_v32: installed DSML tool parser "
        f"(start={_DSML_TOOL_CALL_START!r})",
        flush=True,
    )


def _looks_like_kimi_tools(tokenizer) -> bool:
    """Kimi chat templates declare tools via tool_declare; response markers may
    be absent from the jinja, so mlx-lm's template heuristic never attaches
    kimi_k2."""
    ct = getattr(tokenizer, "chat_template", None) or ""
    if not isinstance(ct, str):
        ct = str(ct)
    return "tool_declare" in ct or "<|tool_calls_section_begin|>" in ct


def _install_kimi_tool_parser(tokenizer) -> None:
    if getattr(tokenizer, "has_tool_calling", False):
        return
    if not _looks_like_kimi_tools(tokenizer):
        return
    try:
        from mlx_lm.tool_parsers import kimi_k2 as _kimi
    except ImportError as e:
        print(f"[paged-moe] kimi_k2 tool parser unavailable: {e!r}", flush=True)
        return
    start = getattr(_kimi, "tool_call_start", None)
    end = getattr(_kimi, "tool_call_end", None)
    if not start or not end:
        return
    tokenizer._tool_parser = _kimi.parse_tool_call
    tokenizer._tool_call_start = start
    tokenizer._tool_call_end = end
    try:
        tokenizer._tool_call_start_tokens = tuple(
            tokenizer.encode(start, add_special_tokens=False)
        )
        tokenizer._tool_call_end_tokens = tuple(
            tokenizer.encode(end, add_special_tokens=False)
        )
    except Exception as e:
        print(f"[paged-moe] kimi_k2 tool tokens skipped: {e!r}", flush=True)
        return
    print(
        f"[paged-moe] kimi_k2: installed tool parser (start={start!r})",
        flush=True,
    )


def _plain_mla_kv_quantize_unsafe(model) -> bool:
    """Plain deepseek_v3 / kimi MLA does pe_scores on update_and_fetch output.

    QuantizedKVCache returns quantized (pack, scale, bias) tuples there, so
    ``k_pe.swapaxes`` crashes after the first decode token. deepseek_v32 uses
    CacheList via make_cache and is fine. MLA KV is already small; stay fp16.
    """
    if callable(getattr(model, "make_cache", None)):
        return False
    mt = getattr(getattr(model, "args", None), "model_type", None)
    if mt in ("deepseek_v3", "kimi_k2", "joyai_llm_flash"):
        return True
    layers = getattr(model, "layers", None) or []
    if not layers:
        return False
    attn = getattr(layers[0], "self_attn", None)
    return type(attn).__name__ == "DeepseekV3Attention"


def _clamp_prefill_step(provider, model) -> None:
    """Bound --prefill-step-size to what this model can actually run.

    The value is a *ceiling*: our generate_step wrapper resizes per step, and
    stock mlx-lm paths that still read the raw int must not exceed it. Fused
    prefill makes the ceiling an activation bound for every architecture;
    without it, a DSA model is capped by its dense score matrix instead.
    """
    cap = adaptive_prefill.default_cli_prefill_step(model)
    cur = int(getattr(provider.cli_args, "prefill_step_size", 0) or 0)
    if 0 < cur <= cap:
        return
    provider.cli_args.prefill_step_size = cap
    why = (
        "activation bound; attention sub-chunked per layer"
        if prefill_fused.active()
        else "score matrix under Metal max"
    )
    print(
        f"[paged-moe] prefill-step-size {cur or 'unset'} -> {cap} ({why})",
        flush=True,
    )


def _install_adaptive_prefill_generate() -> None:
    """Recompute n_to_process each prefill step (mlx-lm uses a fixed int)."""
    global _adaptive_generate_installed
    if _adaptive_generate_installed:
        return
    _adaptive_generate_installed = True

    import functools
    import importlib

    _mlx_gen = importlib.import_module("mlx_lm.generate")
    from mlx_lm.models import cache as _mlx_cache

    _orig_gs = _mlx_gen.generate_step
    _orig_sgs = _mlx_gen.speculative_generate_step

    def _timed_decode(gen):
        """One overall decode tok/s line for the generation, not a per-tick EMA.

        Clock starts after prefill (caller only wraps the decode generator), so
        this is tokens out / wall decode time - the number that answers "how
        fast was this reply", as opposed to the sidecar tick's rolling average.
        """
        t0 = time.perf_counter()
        n = 0
        try:
            for item in gen:
                n += 1
                yield item
        finally:
            dt = time.perf_counter() - t0
            if n > 0 and dt > 0:
                print(
                    f"[paged-moe] decode: {n} tokens in {dt:.1f}s = "
                    f"{n / dt:.2f} tok/s",
                    flush=True,
                )

    def _quantize_fn(kv_bits, kv_group_size, quantized_kv_start):
        return functools.partial(
            _mlx_gen.maybe_quantize_kv_cache,
            quantized_kv_start=quantized_kv_start,
            kv_group_size=kv_group_size,
            kv_bits=kv_bits,
        )

    def _dense_model(model) -> bool:
        info = getattr(model, "_expert_stream_info", None) or {}
        return info.get("mode") == "dense"

    def _dense_lookup_on(model) -> bool:
        if not _dense_model(model):
            return False
        try:
            from .dense import lookup_enabled

            return bool(lookup_enabled())
        except Exception:
            return False

    def _dense_adapt(model, ctx_tokens: int) -> None:
        if not _dense_model(model):
            return
        cache = getattr(model, "_expert_stream_cache", None)
        adapt = getattr(cache, "adapt_to_context", None)
        if adapt is None:
            return
        try:
            adapt(int(ctx_tokens))
        except Exception as e:
            print(f"[dense] adaptive pin failed: {e!r}", flush=True)

    def generate_step(
        prompt,
        model,
        *,
        max_tokens: int = 256,
        sampler=None,
        logits_processors=None,
        max_kv_size=None,
        prompt_cache=None,
        prefill_step_size: int = 2048,
        kv_bits=None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
        prompt_progress_callback=None,
        input_embeddings=None,
    ):
        # Lookup decode replaces the decode loop further down, so it needs this
        # wrapper even when adaptive prefill has nothing to contribute.
        # Dense multi-token verify is a separate switch (dense.lookup); MoE
        # LOOKUP stays alone so agent traffic on streamed MoE is unchanged.
        if kv_bits is not None and _plain_mla_kv_quantize_unsafe(model):
            kv_bits = None
        if (
            not adaptive_prefill.enabled()
            and not config.LOOKUP
            and not _dense_lookup_on(model)
            and not _dense_model(model)
        ):
            yield from _orig_gs(
                prompt,
                model,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                max_kv_size=max_kv_size,
                prompt_cache=prompt_cache,
                prefill_step_size=prefill_step_size,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
                quantized_kv_start=quantized_kv_start,
                prompt_progress_callback=prompt_progress_callback,
                input_embeddings=input_embeddings,
            )
            return

        if input_embeddings is not None:
            # Keep mlx-lm's embedding path untouched - rare for our servers.
            yield from _orig_gs(
                prompt,
                model,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                max_kv_size=max_kv_size,
                prompt_cache=prompt_cache,
                prefill_step_size=prefill_step_size,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
                quantized_kv_start=quantized_kv_start,
                prompt_progress_callback=prompt_progress_callback,
                input_embeddings=input_embeddings,
            )
            return

        if prompt_cache is None:
            prompt_cache = _mlx_cache.make_prompt_cache(
                model, max_kv_size=max_kv_size
            )

        progress = prompt_progress_callback or (lambda *_: None)
        quantize_cache_fn = _quantize_fn(kv_bits, kv_group_size, quantized_kv_start)

        # Mirror mlx-lm: leave one token for the decode _step.
        if not isinstance(prompt, mx.array):
            prompt = mx.array(prompt)
        # The prefill loop below slices `prompt` down to its remainder; lookup
        # decode needs the part it consumed as drafting context.
        full_prompt = prompt
        total = int(prompt.size)
        processed = 0
        progress(processed, total)
        mode = adaptive_prefill.resolve_mode(model)
        cap = int(prefill_step_size or config.ADAPTIVE_PREFILL_MAX)

        # Dense: spend unused KV reserve on pinned MLP pages for this turn's
        # context (+ decode budget) before the first weight pass.
        if _dense_model(model):
            kv0 = adaptive_prefill.cache_offset(prompt_cache)
            _dense_adapt(model, kv0 + total + int(max_tokens))

        with mx.stream(_mlx_gen.generation_stream):
            while total - processed > 1:
                remaining = (total - processed) - 1
                kv_len = adaptive_prefill.cache_offset(prompt_cache)
                n = adaptive_prefill.next_chunk(
                    kv_len, remaining, model, prefill_cap=cap
                )
                if config.ADAPTIVE_PREFILL_DEBUG:
                    print(
                        f"[paged-moe] adaptive-prefill mode={mode} "
                        f"chunk={n} kv={kv_len} rem={remaining} cap={cap}",
                        flush=True,
                    )
                chunk = prompt[:n]
                model(chunk[None], cache=prompt_cache)
                quantize_cache_fn(prompt_cache)
                mx.eval([c.state for c in prompt_cache])
                processed += n
                progress(processed, total)
                prompt = prompt[n:]
                mx.clear_cache()

        if _dense_lookup_on(model):
            from .dense import lookup as dense_lookup

            yield from _timed_decode(
                dense_lookup.generate_step(
                    prompt,
                    model,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    logits_processors=logits_processors,
                    prompt_cache=prompt_cache,
                    history=_drafting_history(full_prompt, int(prompt.size)),
                    quantize_cache_fn=quantize_cache_fn,
                    generation_stream=_mlx_gen.generation_stream,
                )
            )
            return

        if config.LOOKUP:
            # N-gram self-draft: verify several tokens per pass so they share
            # their expert reads. Quality-neutral, so it needs no gate beyond
            # the flag. Same (token, logprobs) contract as _orig_gs.
            yield from _timed_decode(
                lookup_decode.generate_step(
                    prompt,
                    model,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    logits_processors=logits_processors,
                    prompt_cache=prompt_cache,
                    history=_drafting_history(full_prompt, int(prompt.size)),
                    quantize_cache_fn=quantize_cache_fn,
                    generation_stream=_mlx_gen.generation_stream,
                )
            )
            return

        yield from _timed_decode(
            _orig_gs(
                prompt,
                model,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                max_kv_size=max_kv_size,
                prompt_cache=prompt_cache,
                # Exactly one token is left (loop condition above), so this only
                # has to be >= 1; keep it at the model's ceiling regardless.
                prefill_step_size=cap,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
                quantized_kv_start=quantized_kv_start,
                # Prefill is already reported complete above. Forwarding `progress`
                # would let mlx-lm re-report the 1-token remainder as (0, 1) and
                # (1, 1), so any TTFT / percent readout jumps backwards at the end.
                prompt_progress_callback=lambda *_: None,
                input_embeddings=None,
            )
        )

    def speculative_generate_step(
        prompt,
        model,
        draft_model,
        *,
        num_draft_tokens: int = 2,
        max_tokens: int = 256,
        sampler=None,
        logits_processors=None,
        prompt_cache=None,
        prefill_step_size: int = 512,
        kv_bits=None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
    ):
        if kv_bits is not None and _plain_mla_kv_quantize_unsafe(model):
            kv_bits = None
        if not adaptive_prefill.enabled():
            yield from _orig_sgs(
                prompt,
                model,
                draft_model,
                num_draft_tokens=num_draft_tokens,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                prompt_cache=prompt_cache,
                prefill_step_size=prefill_step_size,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
                quantized_kv_start=quantized_kv_start,
            )
            return

        # Prefill both caches with adaptive chunks, then hand the remainder
        # (one token) to stock speculative_generate_step.
        y = prompt.astype(mx.uint32) if isinstance(prompt, mx.array) else mx.array(
            prompt, dtype=mx.uint32
        )
        if prompt_cache is None:
            model_cache = _mlx_cache.make_prompt_cache(model)
            draft_cache = _mlx_cache.make_prompt_cache(draft_model)
            prompt_cache = model_cache + draft_cache
        else:
            model_cache = prompt_cache[: len(model.layers)]
            draft_cache = prompt_cache[len(model.layers) :]

        quantize_cache_fn = _quantize_fn(kv_bits, kv_group_size, quantized_kv_start)
        cap = int(prefill_step_size or config.ADAPTIVE_PREFILL_MAX)

        def _prefill_one(m, c, tokens):
            with mx.stream(_mlx_gen.generation_stream):
                while tokens.size > 1:
                    remaining = int(tokens.size) - 1
                    kv_len = adaptive_prefill.cache_offset(c)
                    n = adaptive_prefill.next_chunk(
                        kv_len, remaining, m, prefill_cap=cap
                    )
                    m(tokens[:n][None], cache=c)
                    quantize_cache_fn(c)
                    mx.eval([e.state for e in c])
                    tokens = tokens[n:]
                    mx.clear_cache()
            return tokens

        # Both loops stop at one token, so the two caches land at the same
        # offset and the remainder below is the same for either model.
        _prefill_one(draft_model, draft_cache, y)
        y = _prefill_one(model, model_cache, y)

        yield from _orig_sgs(
            y,
            model,
            draft_model,
            num_draft_tokens=num_draft_tokens,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prompt_cache=prompt_cache,
            prefill_step_size=max(cap, int(y.size) + 1),
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
        )

    _mlx_gen.generate_step = generate_step
    _mlx_gen.speculative_generate_step = speculative_generate_step


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
_adaptive_generate_installed = False


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


def _materialize_trimmed_prefix(cache: list) -> None:
    """After ``trim_prompt_cache``, physically drop the unused suffix arrays.

    ``KVCache.trim`` only decrements ``offset``; the underlying keys/values keep
    their full length, so a deepcopy-then-trim snapshot of a 14k prefill would
    still cost 14k of KV bytes in the prompt store. Round-tripping ``.state``
    rebinds each layer to the sliced ``[..., :offset, :]`` view.
    """
    for c in cache:
        try:
            c.state = c.state
        except Exception:
            continue


def _fork_prefix_snapshots(
    store,
    model_key: str,
    ctx_tokens: list,
    cache: list,
    bounds: list[int],
    *,
    cached: int,
    processed: int,
    budget: int,
) -> int:
    """Insert deepcopy+trim snapshots at each absolute ``bound``. Returns count."""
    abs_off = cached + processed
    stored = 0
    # Longest first. ``insert_cache`` of a trimmable entry pops its own
    # prefixes, so short-then-long would leave only the latest bound; long-
    # then-short keeps both extremes (robust early + max-reuse late).
    for bound in sorted(bounds, reverse=True):
        if bound > abs_off or bound < max(cached, 1):
            continue
        to_trim = abs_off - bound
        snap = copy.deepcopy(cache)
        if to_trim > 0:
            trim_prompt_cache(snap, to_trim)
            _materialize_trimmed_prefix(snap)
        nbytes = sum(c.nbytes for c in snap)
        if nbytes * 2 > budget:
            if _PREFIX_DEBUG:
                print(
                    f"[prefix]   snapshot skipped: {nbytes / 1e9:.2f} GB vs budget",
                    flush=True,
                )
            continue
        store.insert_cache(
            model_key,
            list(ctx_tokens[:bound]),
            snap,
            cache_type="user",
        )
        stored += 1
        mx.clear_cache()
    return stored


def _prefill_remainder(
    model,
    prompt,
    cache,
    *,
    n_model_layers: int,
    draft_model,
    draft_cache,
    step_cap: int,
    progress,
    total: int,
    processed: int,
    target: int,
) -> int:
    """Drive ``model`` (and optional draft) from ``processed`` up to ``target``."""
    start, t0 = processed, time.perf_counter()
    expert_cache = getattr(model, "_expert_stream_cache", None)
    if expert_cache is not None:
        base = (
            expert_cache.bytes_read,
            expert_cache.disk_wait_s,
            expert_cache.prefill_layers,
            expert_cache.materialize_s,
        )
    while processed < target:
        remaining = target - processed
        kv_len = adaptive_prefill.cache_offset(cache[:n_model_layers])
        if adaptive_prefill.enabled():
            n = adaptive_prefill.next_chunk(
                kv_len, remaining, model, prefill_cap=step_cap
            )
        else:
            n = min(step_cap, remaining)
        # mlx-lm's progress callback only fires *after* a chunk completes, and a
        # fused chunk is the whole prompt, so the single tick lands at the very
        # end. Say up front what is about to happen, or minutes of expert
        # streaming are indistinguishable from a hang.
        print(
            f"[prefill] chunk {n} tok at kv={kv_len} ({processed}/{target} done)"
            " - one pass over the expert mass",
            flush=True,
        )
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
    _log_prefill_cost(model, expert_cache, base if expert_cache else None,
                      t0, processed - start)
    return processed


def _log_prefill_cost(model, expert_cache, base, t0, tokens: int) -> None:
    """One line per prefill, split into terms that can each be acted on.

    Three costs, not two, and the third one used to hide inside the second:

    * **blocked** - a layer sat waiting for a read that had not landed. Reduce
      by reading earlier (more concurrency, deeper readahead) or reading less.
    * **copy** - ``_materialize``: a host-side copy of every byte read, on the
      calling thread. Scales with bytes, not tokens, so a full pass over the
      expert mass pays it in full. Reduce by landing reads somewhere the GPU can
      use directly, not by touching the drive.
    * **rest** - everything else: attention, MoE matmuls, router syncs, Python.
      The only genuinely compute-shaped term.

    Reporting copy as compute is how a memcpy-bound prefill gets mistaken for a
    GPU-bound one, which is exactly the wrong half to optimize. ``GB/s`` is bytes
    over wall time, so it is a utilization figure, not the drive's rate: it can
    only approach the device's number when ``rest`` and ``copy`` are small.
    """
    elapsed = time.perf_counter() - t0
    if expert_cache is None or base is None or tokens <= 0 or elapsed < 1.0:
        return
    read_gb = (expert_cache.bytes_read - base[0]) / 1e9
    blocked = expert_cache.disk_wait_s - base[1]
    copy = expert_cache.materialize_s - base[3]
    moe_layers = int(getattr(model, "_expert_stream_info", {}).get("moe_layers") or 0)
    layers = expert_cache.prefill_layers - base[2]
    passes = f"{layers / moe_layers:.1f} pass" if moe_layers else f"{layers} layer"
    print(
        f"[prefill] {tokens} tok in {elapsed:.1f}s · {read_gb:.0f} GB experts "
        f"({passes}es over the mass, {read_gb / elapsed:.1f} GB/s wall) · "
        f"{blocked:.0f}s blocked, {copy:.0f}s copy "
        f"({read_gb / max(copy, 1e-6):.0f} GB/s), {elapsed - blocked - copy:.0f}s rest",
        flush=True,
    )


def _drafting_history(full_prompt, remaining: int) -> list[int]:
    """Tokens preceding the next decode step, prompt-cache prefix included.

    generate_step only sees the remainder the prompt cache could not serve,
    which on a warm agent turn is a couple of tokens. An n-gram drafter given
    only that finds nothing to match, so recover the served prefix here.
    """
    total = int(full_prompt.size)
    seq = getattr(_req_ctx, "draft_history", None)
    _req_ctx.draft_history = None  # single-shot; never reuse across requests
    tail = full_prompt.tolist()
    if seq is not None and len(seq) >= total and seq[len(seq) - total :] == tail:
        return seq[: len(seq) - remaining]
    # Not this request's prompt: draft off the remainder alone rather than
    # feeding the matcher a sequence the KV cache never saw.
    return tail[: total - remaining]


def _snapshot_user_segment(model, kwargs) -> None:
    """Prefill + store prefix KV snapshots for next-turn reuse.

    Storing the cache keyed by a prefix the *next* request will repeat is what
    turns a 30k-token agent turn from a full re-prefill into a few seconds.

    Two paths, same stored result:

    * **Trimmable caches** (DeepSeek-V3.2, Qwen3-MoE, GLM, ...): prefill the
      whole remainder in one fused pass, then fork each boundary with
      deepcopy + trim. Mid-prefill stops used to cost a full expert-mass read
      each; on a cold ~14k agent prompt that was a second ~368 GB pass for a
      snapshot that ``insert_cache`` of the final sequence would pop anyway
      (trimmable stores drop their own prefixes).
    * **Non-trimmable caches** (qwen3-next hybrid): still stop at each
      boundary. Those caches cannot recover a shorter prefix after overshooting,
      so the mid-prefill stop is load-bearing.
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

    step_cap = int(kwargs.get("prefill_step_size") or config.ADAPTIVE_PREFILL_MAX or 2048)
    progress = kwargs.get("prompt_progress_callback") or (lambda *_: None)
    total = len(prompt)
    store = _req_ctx.store
    budget = getattr(store, "max_bytes", 1 << 62)
    trimmable = can_trim_prompt_cache(cache)

    if trimmable:
        # One fused pass over the whole remainder (leave 1 token for decode,
        # same as generate_step), then fork prefixes. No mid-stop = no second
        # expert-mass read on a cold agent prompt.
        target = max(0, total - 1)
        processed = _prefill_remainder(
            model,
            prompt,
            cache,
            n_model_layers=n_model_layers,
            draft_model=draft_model,
            draft_cache=draft_cache,
            step_cap=step_cap,
            progress=progress,
            total=total,
            processed=0,
            target=target,
        )
        n_snap = _fork_prefix_snapshots(
            store,
            _req_ctx.model_key,
            ctx_tokens,
            cache,
            bounds,
            cached=cached,
            processed=processed,
            budget=budget,
        )
        if _PREFIX_DEBUG or n_snap:
            print(
                f"[prefix]   single-pass prefill {processed}/{total}; "
                f"forked {n_snap} prefix snapshot(s)",
                flush=True,
            )
    else:
        processed = 0
        for bound in bounds:
            target = bound - cached
            processed = _prefill_remainder(
                model,
                prompt,
                cache,
                n_model_layers=n_model_layers,
                draft_model=draft_model,
                draft_cache=draft_cache,
                step_cap=step_cap,
                progress=progress,
                total=total,
                processed=processed,
                target=target,
            )
            _fork_prefix_snapshots(
                store,
                _req_ctx.model_key,
                ctx_tokens,
                cache,
                [bound],
                cached=cached,
                processed=processed,
                budget=budget,
            )

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

    _install_adaptive_prefill_generate()

    # Streamed models must use the single-request path: the batch engine
    # bypasses stream_generate (so no KV quantization, no relieve_pressure)
    # and would prefill multiple prompts concurrently - each of which streams
    # gigabytes of experts.
    _orig_provider_load = _mlx_server.ModelProvider.load

    def _provider_load(self, *args, **kwargs):
        global _model
        result = _orig_provider_load(self, *args, **kwargs)
        model = self.model
        mt = getattr(getattr(model, "args", None), "model_type", None)
        if mt == "deepseek_v32":
            _install_deepseek_tool_parser(self.tokenizer)
        else:
            # No-op unless the chat template looks like Kimi (tool_declare).
            _install_kimi_tool_parser(self.tokenizer)
        _install_json_tools_unwrap(self.tokenizer)
        # Idempotent: the PagedMoE loader already did this, but a model that
        # arrived through mlx-lm's own loader still needs it.
        prefill_fused.install(model)
        _clamp_prefill_step(self, model)
        if getattr(model, "_expert_stream_cache", None) is not None:
            self.is_batchable = False
            _model = model
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
        seq = list(tokens)
        _req_ctx.tokens = seq
        # Same sequence, separate slot: the snapshot path clears .tokens before
        # generate_step runs, and the drafter needs the prefix the cache served.
        _req_ctx.draft_history = seq
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
    import importlib

    _mlx_gen = importlib.import_module("mlx_lm.generate")
    _mlx_gen.maybe_quantize_kv_cache = kvmem.requantize

    # Wrap stream_generate so every HTTP completion (a) uses a quantized KV
    # cache when the model is streamed, (b) snapshots the user segment for
    # thinking models, and (c) returns Metal pages afterwards while keeping
    # the expert LRU warm for the next request.
    _orig_stream = _mlx_server.stream_generate

    def _stream_generate(model, *args, **kwargs):
        if getattr(model, "_expert_stream_cache", None) is not None:
            # Plain deepseek_v3 / Kimi MLA cannot consume QuantizedKVCache
            # (pe_scores path). Keep fp16; MLA KV is already compact.
            if _plain_mla_kv_quantize_unsafe(model):
                kwargs["kv_bits"] = None
                if not getattr(_stream_generate, "_mla_kv_noted", False):
                    print(
                        "[paged-moe] plain MLA (deepseek_v3/kimi): "
                        "keeping fp16 KV (8-bit QuantizedKVCache breaks pe_scores)",
                        flush=True,
                    )
                    _stream_generate._mla_kv_noted = True  # type: ignore[attr-defined]
            else:
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

    # store_true flag (no value). Needed for Kimi / Laguna custom tokenizers.
    if "--trust-remote-code" not in sys.argv:
        sys.argv.append("--trust-remote-code")

    install_patches()
    _mlx_server.main()


if __name__ == "__main__":
    main()
