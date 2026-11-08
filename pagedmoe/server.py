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
     deepcopy+trim; hybrid-attention models (qwen3-next / qwen4_exp) single-pass too
     because they cannot trim backwards.
  6. speculative decoding via PAGED_MOE_DRAFT_MODEL (path to a small
     same-tokenizer model) + PAGED_MOE_DRAFT_TOKENS. On a streamed MoE
     this is a *disk* optimization as much as a compute one: verifying k
     drafted tokens in one forward reads each layer's expert union once
     (consecutive tokens share ~half their experts) and reads every resident
     weight once for k tokens instead of k times. Greedy output is identical
     to non-speculative decoding (acceptance is exact token match).
  7. warm_prefix=true on /v1/chat/completions: prefill + store that exact
     prompt in the LRU and skip decode (any client, e.g. right after a turn ends).

Every default can still be overridden by passing the flag explicitly.

Run:

    python -m pagedmoe.server --model <path-or-hf-repo> --port 8080
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import os
import re
import sys
import threading
import time
from collections import OrderedDict, deque
from typing import Any, Optional

import mlx.core as mx
import mlx_lm.server as _mlx_server
from mlx_lm.generate import generation_stream
from mlx_lm.models.cache import (
    ArraysCache,
    LRUPromptCache,
    can_trim_prompt_cache,
    trim_prompt_cache,
)

from . import (
    activity,
    adaptive_prefill,
    config,
    fidelity,
    gpu_trail,
    hybrid_ckpt,
    inpass_snap,
    kvmem,
    live_config,
    lookup_decode,
    minimax_tools,
    prefill_fused,
    prefix_cache,
    prefix_disk,
    prune_curve,
    replay_splice,
    rescue,
    stage_gentle,
    think_budget,
    turn_trace,
    verify_decode,
)
from .loader import load as _streamed_load
from .loader import relieve_pressure
from .streaming import cache_only, exact_step

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
    (`Expecting property name`) and the server drops the call, so the host sees an
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
    failed parse is an empty turn for the client. Returning a name with best-effort
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


def _stop_on_endoftext(tokenizer) -> None:
    """Treat <|endoftext|> as a stop token when the vocab has it as a special.

    Qwen and GLM ship it in generation_config.json's eos list, but a checkpoint
    without that file (qwen3-235-4bit) only stops on <|im_end|>. The model then
    runs past the end of its turn and writes a made-up "Human:" message into
    the reply.

    <|im_start|> is added too. A ChatML reply never contains it; when a model
    emits one it has skipped <|im_end|> and is opening a made-up next turn.
    MiniMax's message start ]~b] is the same case.
    """
    try:
        added = tokenizer.get_added_vocab()
    except Exception:
        return
    eos = getattr(tokenizer, "eos_token_ids", None)
    if eos is None:
        return
    for name in ("<|endoftext|>", "<|im_start|>", "]~b]"):
        tid = added.get(name)
        if tid is None or tid in eos:
            continue
        eos = set(eos) | {int(tid)}
        tokenizer.eos_token_ids = eos
        if name != "<|endoftext|>":
            tokenizer._pagedmoe_turn_opens = getattr(tokenizer, "_pagedmoe_turn_opens", set()) | {int(tid)}
        print(f"[paged-moe] stop tokens += {name} ({tid})", flush=True)


def _message_ends(tokenizer) -> set:
    """Stop tokens that close a message. The turn openers added above stop a
    reply too, but they start messages, and prefix boundaries must not land
    after them."""
    eos = getattr(tokenizer, "eos_token_ids", None) or set()
    return set(eos) - getattr(tokenizer, "_pagedmoe_turn_opens", set())


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


def _install_minimax_tool_parser(tokenizer) -> None:
    """MiniMax-M3 XML tool calls (see minimax_tools)."""
    if not minimax_tools.template_uses(getattr(tokenizer, "chat_template", None)):
        return
    try:
        start = tuple(tokenizer.encode(minimax_tools.TOOL_CALL_START, add_special_tokens=False))
        end = tuple(tokenizer.encode(minimax_tools.TOOL_CALL_END, add_special_tokens=False))
    except Exception as e:
        print(f"[paged-moe] minimax tool tokens skipped: {e!r}", flush=True)
        return
    tokenizer._tool_parser = minimax_tools.parse_tool_call
    tokenizer._tool_call_start = minimax_tools.TOOL_CALL_START
    tokenizer._tool_call_end = minimax_tools.TOOL_CALL_END
    tokenizer._tool_call_start_tokens = start
    tokenizer._tool_call_end_tokens = end
    print(
        f"[paged-moe] minimax: installed XML tool parser (start={start}, end={end})",
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
    """Plain MLA must stay on an fp16 cache.

    DeepSeek-V2 stores the compressed latent (see mla_v2). DeepSeek-V3 / Kimi
    run pe_scores on update_and_fetch output. QuantizedKVCache returns
    (pack, scale, bias) tuples, which breaks both. deepseek_v32 uses CacheList
    via make_cache and is fine. The latent is already small.
    """
    if callable(getattr(model, "make_cache", None)):
        return False
    mt = getattr(getattr(model, "args", None), "model_type", None)
    if mt in ("deepseek_v2", "deepseek_v3", "kimi_k2", "joyai_llm_flash"):
        return True
    layers = getattr(model, "layers", None) or []
    if not layers:
        return False
    attn = getattr(layers[0], "self_attn", None)
    return type(attn).__name__ in ("DeepseekV2Attention", "DeepseekV3Attention")


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


def _stitch_snap(model):
    """(full set pinned, full-set tokens, partial tokens kept, partial replays)."""
    cache = getattr(model, "_pagedmoe_cache", None) if model is not None else None
    if cache is None:
        return None
    return (
        bool(getattr(cache, "stitched", False)),
        int(getattr(cache, "full_stitch_tokens", 0) or 0),
        int(getattr(cache, "closed_ok", 0) or 0),
        int(getattr(cache, "closed_replay", 0) or 0),
    )


def _log_stitch(model, n: int, snap) -> None:
    """One line next to the decode tok/s: which stitch ran, and on how many tokens.

    That count is the gain. Those tokens skipped the per-layer stop. A replay
    ran the token twice and then took the normal path, so it is not a gain.
    """
    now = _stitch_snap(model)
    if now is None or snap is None or n <= 0:
        return
    full_now, full_tok, ok, replay = now
    _, full0, ok0, replay0 = snap
    full_n = max(0, full_tok - full0)
    ok_n = max(0, ok - ok0)
    replay_n = max(0, replay - replay0)
    if full_now:
        print(
            f"[paged-moe] stitch full: {full_n}/{n} tokens on the one graph",
            flush=True,
        )
        return
    if ok_n or replay_n:
        print(
            f"[paged-moe] stitch partial: {ok_n}/{n} tokens on the one graph, "
            f"{replay_n} replayed",
            flush=True,
        )
        return
    print(f"[paged-moe] stitch off: 0/{n} tokens", flush=True)


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
    # generate_step is defined below this helper, so the model it is decoding
    # is stashed here for the finally block. A nested lookup of `model` raises
    # and skips the decode summary.
    _decode_model = [None]

    def _timed_decode(gen):
        """One overall decode tok/s line for the generation, not a per-tick EMA.

        Clock starts after prefill (caller only wraps the decode generator), so
        this is tokens out / wall decode time - the number that answers "how
        fast was this reply", as opposed to the sidecar tick's rolling average.
        """
        t0 = time.perf_counter()
        n = 0
        last = None
        stitch_at = _stitch_snap(_decode_model[0])
        fidelity.begin_turn()
        turn_trace.mark("decode_start")
        try:
            for item in gen:
                n += 1
                if n == 1:
                    turn_trace.mark("first_token")
                last = item
                try:
                    fidelity.observe(item[0] if isinstance(item, tuple) else item)
                except (TypeError, ValueError, IndexError):
                    pass
                yield item
        finally:
            strict = fidelity.end_turn()
            mdl = _decode_model[0]
            ring = getattr(mdl, "_pagedmoe_ring", None) if mdl is not None else None
            warm = getattr(ring, "tool_warm", None) if ring is not None else None
            if warm is not None:
                try:
                    warm.end_turn()
                except Exception:
                    pass
            tool = fidelity.tool_tokens()
            trips, rescued = fidelity.guard_stats()
            loops = rescue.stats()
            dt = time.perf_counter() - t0
            turn_trace.mark("decode_end")
            if turn_trace.enabled():
                try:
                    last_tok = int(last[0] if isinstance(last, tuple) and last else last)
                except (TypeError, ValueError):
                    last_tok = None
                turn_trace.note(
                    decode_tokens=n,
                    decode_s=round(dt, 4),
                    strict_tokens=strict or 0,
                    tool_tokens=tool or 0,
                    last_token=last_tok,
                )
            if n > 0 and dt > 0:
                # Last token id says how the turn ended (eos, a stop token, or max_tokens).
                tok = last[0] if isinstance(last, tuple) and last else last
                try:
                    tok = int(tok)
                except (TypeError, ValueError):
                    tok = None
                print(
                    f"[paged-moe] decode: {n} tokens in {dt:.1f}s = "
                    f"{n / dt:.2f} tok/s"
                    + (f" last={tok}" if tok is not None else "")
                    + (f" strict={strict}" if strict else "")
                    + (f" tool={tool}" if tool else "")
                    + (f" guard={trips}x/{rescued}tok" if trips else "")
                    + (
                        f" rescue={loops['trips']}x/{loops['dropped']}drop"
                        f"/{loops['audits']}audit"
                        if loops.get("trips") or loops.get("audits")
                        else ""
                    ),
                    flush=True,
                )
                _log_stitch(mdl, n, stitch_at)
            try:
                info = getattr(mdl, "_pagedmoe_info", None) or {}
                prune_curve.flush(info.get("model_path"))
            except Exception:
                pass

    def _quantize_fn(kv_bits, kv_group_size, quantized_kv_start):
        return functools.partial(
            _mlx_gen.maybe_quantize_kv_cache,
            quantized_kv_start=quantized_kv_start,
            kv_group_size=kv_group_size,
            kv_bits=kv_bits,
        )

    def _dense_model(model) -> bool:
        info = getattr(model, "_pagedmoe_info", None) or {}
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
        cache = getattr(model, "_pagedmoe_cache", None)
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
        _decode_model[0] = model
        # Lookup decode replaces the decode loop further down, so it needs this
        # wrapper even when adaptive prefill has nothing to contribute.
        # Dense multi-token verify is a separate switch (dense.lookup); MoE
        # LOOKUP stays alone so agent traffic on streamed MoE is unchanged.
        if kv_bits is not None and _plain_mla_kv_quantize_unsafe(model):
            kv_bits = None
        if (
            not adaptive_prefill.enabled()
            and not config.LOOKUP
            and not verify_decode.wanted(model)
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
        # Realize it before generation_stream. A lazy prompt consumed on that
        # stream is the cross-stream wait that stalls a Metal completion for
        # the whole prefill (same failure as a draft prompt left on the
        # default stream).
        mx.eval(prompt)
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
                ec = getattr(model, "_pagedmoe_cache", None)
                tb = time.perf_counter()
                cb = (ec.bytes_read, ec.disk_wait_s, ec.materialize_s) if ec is not None else None
                with _layer_progress(model, progress, processed, n, total), exact_step(), cache_only():
                    model(chunk[None], cache=prompt_cache)
                quantize_cache_fn(prompt_cache)
                mx.eval([c.state for c in prompt_cache])
                if cb is not None:
                    turn_trace.prefill_chunk(
                        n, time.perf_counter() - tb, (ec.bytes_read - cb[0]) / 1e9,
                        ec.disk_wait_s - cb[1], ec.materialize_s - cb[2], kv_len,
                    )
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

        if verify_decode.wanted(model):
            # Plain steps are _orig_gs's; drafted passes run exact and only
            # where the governor measured them faster.
            history = _drafting_history(full_prompt, int(prompt.size))
            sources = _drafting_sources()

            def _verify(p, n, prefix, procs):
                return verify_decode.generate_step(
                    p,
                    model,
                    max_tokens=n,
                    sampler=sampler,
                    logits_processors=procs,
                    prompt_cache=prompt_cache,
                    history=history + prefix,
                    sources=sources,
                    quantize_cache_fn=quantize_cache_fn,
                    generation_stream=_mlx_gen.generation_stream,
                )

            yield from _timed_decode(
                rescue.guarded(
                    _verify, prompt, model, prompt_cache,
                    max_tokens=max_tokens, logits_processors=logits_processors,
                    stream=_mlx_gen.generation_stream,
                )
            )
            return

        def _plain(p, n, prefix, procs):
            return _orig_gs(
                p,
                model,
                max_tokens=n,
                sampler=sampler,
                logits_processors=procs,
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

        yield from _timed_decode(
            rescue.guarded(
                _plain, prompt, model, prompt_cache,
                max_tokens=max_tokens, logits_processors=logits_processors,
                stream=_mlx_gen.generation_stream,
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

    def _counted(gen):
        # Shutdown waits on activity.decoding(). The synchronize has to run
        # here, on the generation thread, because its stream is thread local.
        # Every decode path touches for the watchdog here. Resident and dense
        # decode never reach the expert fetch that touches, and the plain and
        # speculative paths skip _timed_decode.
        def run(*args, **kwargs):
            activity.decode_enter()
            inner = gen(*args, **kwargs)
            try:
                for item in inner:
                    activity.touch()
                    rest = activity.pace()
                    if rest:
                        time.sleep(rest)
                    yield item
            finally:
                inner.close()
                if activity.stopping():
                    try:
                        mx.synchronize(_mlx_gen.generation_stream)
                    except Exception:
                        pass
                activity.decode_leave()

        return run

    _mlx_gen.generate_step = _counted(generate_step)
    _mlx_gen.speculative_generate_step = _counted(speculative_generate_step)


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
_DRAFT_MODEL = os.environ.get("PAGED_MOE_DRAFT_MODEL", "").strip()
_DRAFT_TOKENS = os.environ.get("PAGED_MOE_DRAFT_TOKENS", "").strip()
if _DRAFT_MODEL:
    _DEFAULT_ARGS["--draft-model"] = os.path.expanduser(_DRAFT_MODEL)
    _DEFAULT_ARGS["--num-draft-tokens"] = _DRAFT_TOKENS or "3"

# Set by the wrapped fetch_nearest_cache; read (once) by the wrapped
# stream_generate on the same generation thread.
_req_ctx = threading.local()

# warm_prefix is set on the HTTP handler thread, but stream_generate runs on
# mlx-lm's ResponseGenerator worker. threading.local() does not cross that
# boundary, so the warm flag lives here for the duration of handle_completion
# (which blocks until the worker finishes). Single Metal owner => one gen at
# a time; no need for a per-request map.
_warm_prefix_active = False

# The streamed model, once loaded. One per server process; the prompt-cache
# patches need it (they are methods on mlx-lm's store, which is model-agnostic
# and is constructed before any model exists).
_model = None
# Whatever model the provider loaded, streamed or resident. The prefix disk
# files need it; nothing that assumes an expert cache may read it.
_served_model = None
# The loader's mode for whatever model the provider loaded, streamed or not.
_load_mode: Optional[str] = None

# Don't snapshot a prompt shorter than this - a re-prefill is cheap anyway.
# Env override: PAGED_MOE_MIN_SNAPSHOT (set huge to disable snapshots).
_MIN_SNAPSHOT_TOKENS = int(os.environ.get("PAGED_MOE_MIN_SNAPSHOT", "1024") or 1024)
# Separate, much smaller floor for the non-trimmable mid-prefill snapshot. On a
# cache that cannot be trimmed, that mid-snap is the only thing that keeps the
# reuse point moving forward, so skipping it is far more expensive than the
# deepcopy it saves: the fallback is an end-of-prefill entry whose key ends in
# generation-prompt tokens, which never recur at that offset on the next turn.
# Short turns (a brief reply plus a small tool result, or a stall/nudge cycle)
# add well under 1024 tokens, so reusing the big floor here silently froze the
# reuse point for exactly the traffic this path exists to serve.
# Env override: PAGED_MOE_MIN_MID_SNAPSHOT.
_MIN_MID_SNAPSHOT_TOKENS = int(
    os.environ.get("PAGED_MOE_MIN_MID_SNAPSHOT", "128") or 128
)
# ...but once we're past that, a snapshot is worth taking for even a few
# hundred new tokens: copying a KV cache costs ~0.1 s, while re-prefilling
# 300 tokens of a streamed MoE costs ~2 s. Without incremental snapshots the
# reuse point freezes and every later turn re-prefills everything appended
# since - which is what makes a long agent session get slower and slower.
_MIN_SNAPSHOT_GAIN = 128
# Cap copies per request (each one costs its own KV bytes).
_MAX_SNAPSHOTS = 2
# Longest repeated tail (reminder blocks + generation prompt) the mid-snap
# anchor will step back over. Past this it is new content, not a reminder.
_VOLATILE_TAIL_MAX = 4096

# PAGED_MOE_PREFIX_DEBUG=1 logs, per request, how much of the prompt the
# cache could reuse and - when reuse is partial - the decoded text on both
# sides of the divergence point. That is how you find a client that rewrites
# its own history (the difference between a 1 s and a 60 s agent turn).
_PREFIX_DEBUG = bool(int(os.environ.get("PAGED_MOE_PREFIX_DEBUG", "0") or 0))
_recent_prompts: deque = deque(maxlen=6)
# The last entry the store was told to keep (model_key, tokens).
_live_key = None
# Cold prefills while the store held entries. Each one is a full pass over the
# expert mass that a cache hit would have skipped.
_cold_misses = 0
_adaptive_generate_installed = False
# HTTP thread writes these; the generate worker reads them. Streamed models
# are single-request, so there is no overlap.
_pending_cache_breaks: list[int] = []
class _Pin(prefix_cache.StablePrefixPin):
    """Refuses KV from a request a live settings change landed in."""

    def update(self, *a, **k) -> bool:
        return False if live_config.kv_stale() else super().update(*a, **k)

    def put(self, *a, **k) -> bool:
        return False if live_config.kv_stale() else super().put(*a, **k)


_stable_pin = _Pin()


def _drop_settings_kv(store) -> None:
    """Live settings changed (live_config): nothing cached under the old ones
    may be served. On disk, the settings fingerprint keeps them apart."""
    n, gb = len(store), store.nbytes / 1e9
    store.trim_to(n_sequences=0)
    _stable_pin.clear()
    hybrid_ckpt.clear()
    prefix_disk.reset()
    print(
        f"[prefix] settings changed live: dropped {n} cached prompts ({gb:.2f} GB), "
        f"the front matter pin and hybrid checkpoints",
        flush=True,
    )


def _save_pin_to_disk(model, budget: int, ckpts: dict | None = None, replay_for=None) -> None:
    """Write the front matter pin just taken, for the next cold start.
    `ckpts` is recurrent state inside it by position (hybrid only).
    `replay_for(n)` is how this request prefilled the end of n tokens, if it did."""
    if not prefix_disk.enabled() or _served_model is None or model is not _served_model:
        return
    pin = _stable_pin.get(_req_ctx.model_key)
    if pin is not None:
        # System span only (no workspace blocks): it is the shared file, and
        # workspace front files must not evict it.
        if _pending_stable_client:
            kind = prefix_disk.KIND_CLIENT
        elif _pending_stable_all_shared:
            kind = prefix_disk.KIND_SHARED
        else:
            kind = prefix_disk.KIND_FRONT
        n = len(pin[0])
        ck = {p: s for p, s in (ckpts or {}).items() if p < n} or None
        rp = replay_for(n) if replay_for is not None else None
        gpu_trail.mark(f"prefix-disk save {kind} {n} tok")
        prefix_disk.save(
            _req_ctx.model_key, pin[0], pin[1], model, budget, kind=kind, ckpts=ck, replay=rp
        )


# A last chunk shorter than this runs at decode-like shapes, so a replay of it
# alone says little about the bulk of the file. The one before it goes too.
_REPLAY_MIN_CHUNK = 512


def _replay_spec(chunk_log: list, cached: int, prompt, n_abs: int) -> dict | None:
    """The chunk(s) of this request that wrote the end of the first `n_abs`
    tokens, for prefix_disk's engine check. None when they did not run here.
    `prompt` is what the chunks ran over (positions from `cached`)."""
    rel = n_abs - cached
    idx = next(
        (i for i in range(len(chunk_log) - 1, -1, -1)
         if chunk_log[i]["lo"] < rel <= chunk_log[i]["lo"] + chunk_log[i]["n"]),
        None,
    )
    if idx is None:
        return None
    j = idx
    if (
        chunk_log[idx]["n"] < _REPLAY_MIN_CHUNK
        and idx > 0
        and chunk_log[idx - 1]["lo"] + chunk_log[idx - 1]["n"] == chunk_log[idx]["lo"]
        and chunk_log[idx - 1]["slots"] is not None
    ):
        j = idx - 1
    slots = chunk_log[j]["slots"]
    if slots is None:
        return None
    hi = chunk_log[idx]["lo"] + chunk_log[idx]["n"]
    if hi > len(prompt):
        return None
    extra = prompt[rel:hi]
    extra = extra.tolist() if isinstance(extra, mx.array) else list(extra)
    return {
        "lo": cached + chunk_log[j]["lo"],
        "sizes": [chunk_log[k]["n"] for k in range(j, idx + 1)],
        "extra": [int(t) for t in extra],
        "slots": slots,
    }


def _replay_chunks(model, cache, ids: list, sizes: list) -> None:
    """prefix_disk's replay: `ids` through `cache` in exactly `sizes` chunks,
    down the same path a prefill takes."""
    gpu_trail.mark(f"prefix-disk replay check {len(ids)} tok in {len(sizes)} chunk(s)")
    done = _prefill_remainder(
        model, ids, cache, n_model_layers=len(cache), draft_model=None, draft_cache=None,
        step_cap=max(sizes), progress=lambda *_: None, total=len(ids), processed=0,
        target=len(ids), plan=sizes,
    )
    if done != len(ids):
        raise RuntimeError(f"replay stopped at {done} of {len(ids)}")


def _warm_keeps_stable_on_disk(model, budget: int, tokens: list) -> None:
    """A warm request whose pin is exactly its stable front: save it, or mark
    the saved file used (save does that for a file it already has). The host
    reads the index afterwards to learn which file it staged."""
    if not _warm_prefix_active or not tokens:
        return
    pin = _stable_pin.get(_req_ctx.model_key)
    if pin is not None and len(pin[0]) == len(tokens) and list(pin[0]) == list(tokens):
        _save_pin_to_disk(model, budget)

# `stable_prefix_messages: N` from the request body: the first N messages are
# front matter the host keeps byte-identical (system, tool schemas, pinned
# memory / map). Held as (messages[:N], tools, template kwargs) until the
# worker has a tokenizer to find the token offset.
_pending_stable: Optional[tuple] = None
# Same, for `stable_prefix_head_messages`: front matter up to the memory block.
_pending_stable_head: Optional[tuple] = None
# Same, for `stable_prefix_shared_messages`: the system span every workspace
# starts with, before the codebase map.
_pending_stable_shared: Optional[tuple] = None
# The host marked the whole stable prefix as shared (system and tools only).
_pending_stable_all_shared = False
# `stable_prefix_kind: "client"`: the stable prefix is another client's
# system span (host OpenAI API), saved under its own kind.
_pending_stable_client = False
# `stable_prefix_chat_messages: N` and `stable_prefix_chat_id`: the first N
# messages are a whole conversation to keep on disk under that id.
_pending_chat: Optional[tuple] = None
_pending_chat_tag: Optional[str] = None
# Shorter conversations are mostly the front matter file already.
_CHAT_MIN_TOKENS = 2048
# Rendered probe tokens keyed by the front matter json. Same front matter
# every turn, so the template only renders once per change.
_stable_probe_cache: dict = {}
_STABLE_PROBE_CACHE_MAX = 8

# JSON-escaped text of the request's recent tool results and the tokenizer to
# encode it with, for the verify drafter (config.VERIFY_COPY_SOURCES). Set by
# the handler, taken once by the worker.
_pending_copy_texts: list = []
_pending_copy_tok = None
# Encoded escapes by text. Every later request of a conversation sends the
# same tool results again, so each one is encoded once.
_copy_cache: "OrderedDict[str, list]" = OrderedDict()
_COPY_CACHE_MAX = 64
_COPY_TEXTS_MAX = 12
# Total escaped text per request, newest results first. The drafter indexes it
# on its first lookup in decode, so this bounds that one-time cost (~30k tok).
_COPY_CHARS_MAX = 120_000


def _copy_texts(body) -> list:
    """Escaped forms of host file bodies and recent tool results."""
    from .copy_texts import escaped_runs

    return escaped_runs(body, limit_chars=_COPY_CHARS_MAX, limit_runs=_COPY_TEXTS_MAX)


def _drafting_sources():
    """Lazy builder for the drafter's copy-only runs, or None."""
    global _pending_copy_texts, _pending_copy_tok
    texts, tok = _pending_copy_texts, _pending_copy_tok
    _pending_copy_texts, _pending_copy_tok = [], None
    if not texts or tok is None:
        return None

    def build():
        runs = []
        for text in texts:
            ids = _copy_cache.get(text)
            if ids is None:
                ids = list(tok.encode(text, add_special_tokens=False))
                _copy_cache[text] = ids
                while len(_copy_cache) > _COPY_CACHE_MAX:
                    _copy_cache.popitem(last=False)
            else:
                _copy_cache.move_to_end(text)
            runs.append(ids)
        return runs

    return build


def _shared_len(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _branch_keep(store, model_key, tokens) -> list:
    """The live entry, when `tokens` branched from it and it cannot trim.

    A hybrid entry is not a prefix of a prompt that re-renders history behind
    its end, so the fetch and the next insert would treat it as a finished
    chat and drop it under pressure. The checkpoints on its lineage then have
    nothing to restore onto, and the next request that comes back to that
    history prefills from the turn-one checkpoint (2026-10-12: 26.9k tokens).
    """
    live = _live_key
    if live is None or live[0] != model_key:
        return []
    try:
        entry = store._trie.get(*live)
    except (KeyError, TypeError):
        return []
    if entry is None or can_trim_prompt_cache(entry.prompt_cache):
        return []
    shared = _shared_len(live[1], tokens)
    if shared >= len(live[1]) or shared < max(int(config.HYBRID_CKPT_MIN_TOKENS), len(live[1]) // 4):
        return []
    return [live]


def _note_history_rewrite(tokenizer, tokens: list, stable_at: int = 0) -> None:
    """Say where a prompt stopped extending the one before it.

    Normal turns diverge only at the previous generation prompt. Anything
    earlier is the host rewriting history it already sent, and on a hybrid
    cache that costs a prefill from the nearest checkpoint behind it. A prompt
    that leaves right after the host's stable front matter is a new chat (or
    a warmup before one), not a rewrite.
    """
    prev = _recent_prompts[-1] if _recent_prompts else None
    _recent_prompts.append(list(tokens))
    if not prev or len(prev) < _MIN_SNAPSHOT_TOKENS:
        return
    n = _shared_len(prev, tokens)
    if n >= len(prev) - 16 or n < len(prev) // 4:
        return
    if stable_at and stable_at <= n <= stable_at + 16:
        return
    try:
        head = tokenizer.decode(tokens[max(0, n - 40) : n])
        old = tokenizer.decode(prev[n : n + 40])
        new = tokenizer.decode(tokens[n : n + 40])
    except Exception:
        return
    print(
        f"[prefix] history rewritten at token {n} of {len(prev)} (host changed text it already sent): "
        f"...{head[-80:]!r} | was {old[:80]!r} | now {new[:80]!r}",
        flush=True,
    )


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


def _first_message_boundary(tokenizer, tokens: list, max_at: int = 8192) -> int:
    """First EOS in the prompt - typically the end of the system message.

    That span is the constant head (system + often the tool schemas) that
    later turns should still hit after the live transcript rewrites.
    """
    eos = _message_ends(tokenizer)
    if not eos:
        return -1
    limit = min(len(tokens) - 1, max_at)
    for i in range(limit):
        if tokens[i] in eos:
            return i + 1
    return -1


def _stable_count(body: dict, snake: str, camel: str) -> int:
    try:
        return int(body.get(snake) or body.get(camel) or 0)
    except (TypeError, ValueError):
        return 0


def _stable_spec(handler, body: dict, head: bool = False, shared: bool = False) -> Optional[tuple]:
    """(messages[:N], tools, template kwargs) for `stable_prefix_messages: N`.

    With `head`, N is `stable_prefix_head_messages`: where the volatile part
    of the front matter (memory) starts. With `shared`, N is
    `stable_prefix_shared_messages`: where the workspace blocks start. Each
    is only used when it is shorter than the stable prefix.
    """
    n = _stable_count(body, "stable_prefix_messages", "stablePrefixMessages")
    if head or shared:
        h = (
            _stable_count(body, "stable_prefix_shared_messages", "stablePrefixSharedMessages")
            if shared
            else _stable_count(body, "stable_prefix_head_messages", "stablePrefixHeadMessages")
        )
        if not 0 < h < n:
            return None
        n = h
    return _spec_for(handler, body, n)


def _chat_spec(handler, body: dict) -> Optional[tuple]:
    """Same, for `stable_prefix_chat_messages: N`: the conversation to save."""
    return _spec_for(handler, body, _stable_count(body, "stable_prefix_chat_messages", "stablePrefixChatMessages"))


def _spec_for(handler, body: dict, n: int) -> Optional[tuple]:
    messages = body.get("messages") or []
    if n <= 0 or n >= len(messages):
        return None
    msgs = copy.deepcopy(messages[:n])
    try:
        _mlx_server.process_message_content(msgs)
    except Exception:
        return None
    try:
        kwargs = dict(handler.response_generator.cli_args.chat_template_args or {})
    except Exception:
        kwargs = {}
    kwargs.update(body.get("chat_template_kwargs") or {})
    return msgs, body.get("tools"), kwargs


def _stable_prefix_bound(tokenizer, tokens: list, spec="pending") -> int:
    """Token offset where the host's stable front matter ends, or 0.

    Same probe mlx-lm uses for its system segment: render the first N messages
    plus an empty user turn, and take the common prefix with the real prompt.
    Then back up to the last message end, so the key does not run into the
    role header of whatever comes next.

    Any offset is a correct place for a KV snapshot (KV at i depends only on
    tokens[:i]). A template quirk here can only make the pin shorter, never
    wrong.
    """
    if spec == "pending":
        spec = _pending_stable
    if not spec:
        return 0
    msgs, tools, kwargs = spec
    try:
        key = json.dumps([msgs, tools, kwargs], sort_keys=True, default=str)
    except Exception:
        return 0
    probe = _stable_probe_cache.get(key)
    if probe is None:
        try:
            out = tokenizer.apply_chat_template(
                msgs + [{"role": "user", "content": ""}],
                add_generation_prompt=False,
                tools=tools,
                tokenize=True,
                **kwargs,
            )
            if isinstance(out, dict) or hasattr(out, "input_ids"):
                out = out["input_ids"]
            probe = list(out)
        except Exception as e:
            print(f"[prefix]   stable prefix probe failed: {e!r}", flush=True)
            probe = []
        if len(_stable_probe_cache) >= _STABLE_PROBE_CACHE_MAX:
            _stable_probe_cache.pop(next(iter(_stable_probe_cache)))
        _stable_probe_cache[key] = probe
    n = prefix_cache.common_prefix_len(probe, tokens)
    eos = _message_ends(tokenizer)
    for i in range(n - 1, max(n - 16, 0) - 1, -1):
        if tokens[i] in eos:
            n = i + 1
            break
    # Leave at least one token for decode.
    n = min(n, len(tokens) - 1)
    return n if n >= _MIN_SNAPSHOT_TOKENS else 0


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
    eos = _message_ends(tokenizer)
    if not eos:
        return -1
    for i in range(len(tokens) - 1 - min_back, -1, -1):
        if tokens[i] in eos:
            return i + 1
    return -1


def _last_message_start(tokenizer, tokens: list) -> int:
    """Offset where the final message before the generation prompt begins.

    -1 when the prompt has fewer than two message ends.
    """
    eos = _message_ends(tokenizer)
    if not eos:
        return -1
    ends = 0
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i] in eos:
            ends += 1
            if ends == 2:
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
    for b in _pending_cache_breaks:
        if 0 < b < len(tokens):
            bounds.add(int(b))

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

    head = _first_message_boundary(tokenizer, tokens)
    if head > 0:
        bounds.add(head)

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

    Plain attention layouts don't convert at all: `kvmem.ResumedKVCache`
    hands attention fp16 one layer at a time, so there is nothing to make room
    for and the expert slab stays up. The rest (hybrid, or KV_LAZY=0) go
    through `kvmem.unquantize`, and there the sequencing matters: the fp16
    cache is the largest allocation this process makes after load, and on a
    full-attention model it does not fit next to a committed expert slab, so the
    expert cache is asked to yield first. Doing it in the other order is the
    asynchronous Metal failure that kills the request.
    """
    from mlx_lm.models.cache import QuantizedKVCache

    # MLA and short-context caches stay fp16, so there is nothing to convert
    # and evicting experts for it only costs the next prefill a re-read.
    if not any(isinstance(c, QuantizedKVCache) for c in cache):
        return
    if config.KV_LAZY and kvmem.lazy_resume_ok(cache):
        kvmem.resume_lazily(cache)
        return
    expert_cache = getattr(model, "_pagedmoe_cache", None) if model else None
    if expert_cache is not None:
        held = sum(getattr(c, "nbytes", 0) for c in cache)
        # fp16 is 2 B/element against the 8-bit cache's ~1.06, so the fp16 cache
        # is a little under twice what this one holds - and `unquantize` releases
        # the quantized bytes as it goes, so the slab only has to cover the
        # difference.
        need = int(held * 2.0)
        if expert_cache.make_room_for(need, freeing=held):
            turn_trace.event("slab_yield_kv", gb=round(need / 1e9, 2))
            print(
                f"[paged-moe] expert cache yielded for a {need / 1e9:.1f} GB "
                "fp16 KV dequantize (resumed long context)",
                flush=True,
            )
    kvmem.unquantize(cache)


def _note_kv_room(n_tokens: int) -> None:
    """Tell the expert cache how much KV this conversation can still add, so
    a slab it rebuilds this turn still fits the next ones."""
    expert_cache = getattr(_model, "_pagedmoe_cache", None) if _model is not None else None
    if expert_cache is not None and hasattr(expert_cache, "kv_room_bytes"):
        expert_cache.kv_room_bytes = kvmem.kv_room(n_tokens)


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


def _kv_copy_fits(cache: list) -> bool:
    """Whether a full deepcopy of `cache` fits under Metal's working set.

    Resident loads only. They have no expert slab to give memory back, and on
    a dense full-attention model each copy is the whole conversation's KV
    (KAT-Dev 32B: 0.26 MB/token, 4.5 GB at 17k). Skipping a snapshot costs a
    re-prefill if a request branches there; the OOM costs the request.
    """
    if _load_mode != "resident":
        return True
    from . import mempressure

    ws = mempressure.gpu_working_set_bytes()
    if not ws:
        return True
    need = sum(getattr(c, "nbytes", 0) for c in cache)
    margin = int(float(config.ACTIVATION_GB) * (1 << 30))
    if mx.get_active_memory() + need + margin <= ws:
        return True
    if _PREFIX_DEBUG:
        print(
            f"[prefix]   KV copy skipped: {need / 1e9:.2f} GB would pass the "
            f"{ws / 1e9:.1f} GB Metal working set",
            flush=True,
        )
    return False


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
        # Hybrid caches (qwen4_exp / qwen3-next ArraysCache) are not trimmable.
        # trim_prompt_cache is a silent no-op then, and inserting tokens[:bound]
        # against a longer KV would poison the prompt store.
        if to_trim > 0 and not can_trim_prompt_cache(cache):
            continue
        if not _kv_copy_fits(cache):
            break
        gpu_trail.mark(f"prefix snapshot fork, trim {to_trim}")
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


@contextlib.contextmanager
def _layer_progress(model, progress, done: int, n: int, total: int):
    """Report prompt progress per MoE layer while one chunk runs.

    A fused chunk is often the whole prompt, so the per-chunk callback only
    fires at the end. Each MoE layer is evaluated before the next one starts,
    so the fraction of layers done is real progress through the chunk.
    """
    cache = getattr(model, "_pagedmoe_cache", None)
    if cache is None or n < 256:
        yield
        return
    # mlx-lm logs every callback, so report about once a second.
    last = [time.monotonic()]

    def hook(i: int, layers: int) -> None:
        now = time.monotonic()
        if 0 < i + 1 < layers and now - last[0] >= 1.0:
            last[0] = now
            progress(done + (n * (i + 1)) // layers, total)

    cache.prefill_layer_hook = hook
    try:
        yield
    finally:
        cache.prefill_layer_hook = None


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
    snap_at: int = 0,
    snap_out: dict | None = None,
    ckpt_at: list[int] | None = None,
    ckpt_out: dict | None = None,
    plan: list[int] | None = None,
    chunk_log: list | None = None,
) -> int:
    """Drive ``model`` (and optional draft) from ``processed`` up to ``target``.

    With ``snap_out``, the recurrent state after ``snap_at`` (same coordinates
    as ``processed``) lands in ``snap_out["slots"]`` (see inpass_snap.run).
    With ``ckpt_out``, the state after each of ``ckpt_at`` that this run
    passes lands in ``ckpt_out[pos]`` (disk checkpoints, see prefix_disk).
    A position whose state cannot be read is left out, never guessed.
    ``plan`` fixes the chunk sizes (a prefix_disk replay runs a file's chunks
    again exactly). ``chunk_log`` gets one {"lo", "n", "slots"} per chunk, with
    the recurrent state it started from on the last two.
    """
    want_ck = sorted(p for p in (ckpt_at or ()) if processed <= p < target) if ckpt_out is not None else []
    plan = list(plan) if plan else None
    start, t0 = processed, time.perf_counter()
    kv0 = adaptive_prefill.cache_offset(cache[:n_model_layers])
    expert_cache = getattr(model, "_pagedmoe_cache", None)
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
        if plan is not None:
            n = min(plan.pop(0), remaining) if plan else remaining
        elif adaptive_prefill.enabled():
            n = adaptive_prefill.next_chunk(
                kv_len, remaining, model, prefill_cap=step_cap
            )
        else:
            n = min(step_cap, remaining)
        if chunk_log is not None:
            chunk_log.append({"lo": processed, "n": n, "slots": _live_slots(cache[:n_model_layers])})
            if len(chunk_log) > 2 and chunk_log[-3]["slots"]:
                chunk_log[-3]["slots"] = None
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
        gpu_trail.mark(f"prefill chunk {n} tok at kv={kv_len}")
        if snap_out is not None and processed == snap_at and "slots" not in snap_out:
            snap_out["slots"] = _live_slots(cache[:n_model_layers])
        if processed in want_ck and processed not in ckpt_out:
            ckpt_out[processed] = _live_slots(cache[:n_model_layers])
        inside = snap_out is not None and processed < snap_at < processed + n
        ck_inside = [p for p in want_ck if processed < p < processed + n]
        # Same (thread-local) stream speculative_generate_step runs on.
        # Touching the draft model on the default stream here and on
        # generation_stream inside sgs reliably ended in a Metal
        # "GPU Timeout Error" on the draft's first decode step.
        with mx.stream(generation_stream):
            # A chunk this small (the tail after a mid-snap is the generation
            # prompt, 3 tokens) is decode-sized, and decode-sized calls get
            # PRUNE / WAIT_ABOVE. Prompt tokens must stay exact.
            with _layer_progress(model, progress, processed, n, total), exact_step(), cache_only():
                if inside or ck_inside:
                    snaps = sorted(ck_inside + ([snap_at] if inside else []))
                    gpu_trail.mark(f"prefill chunk with in-pass snapshot(s) at {snaps}")
                    ids = chunk[None]
                    rels = [p - processed for p in ck_inside]
                    if inside:
                        rels.append(snap_at - processed)
                    got = inpass_snap.run_multi(
                        model, cache[:n_model_layers], ids, rels,
                        lambda: model(ids, cache=cache),
                        quiet=not inside,
                    )
                    by_rel = dict(zip(sorted(set(rels)), got)) if got is not None else {}
                    if inside:
                        snap_out["slots"] = by_rel.get(snap_at - processed)
                    for p in ck_inside:
                        if (p - processed) in by_rel:
                            ckpt_out[p] = by_rel[p - processed]
                else:
                    model(chunk[None], cache=cache)
            mx.eval([c.state for c in cache[:n_model_layers]])
            if draft_model is not None:
                # Tiny + resident: this costs milliseconds. Bounded evals,
                # separate from the main model's command buffers.
                for i in range(0, n, 2048):
                    draft_model(chunk[i : i + 2048][None], cache=draft_cache)
                    mx.eval([c.state for c in draft_cache])
        processed += n
        # Resident and dense prefill never reach the expert fetch that touches.
        activity.touch()
        progress(processed, total)
        mx.clear_cache()
    if snap_out is not None and processed == snap_at and "slots" not in snap_out:
        snap_out["slots"] = _live_slots(cache[:n_model_layers])
    _log_prefill_cost(model, expert_cache, base if expert_cache else None,
                      t0, processed - start, kv0)
    return processed


def _live_slots(cache) -> dict:
    """Recurrent state of a cache that sits exactly at the snapshot point."""
    return {i: list(c.cache) for i, c in enumerate(cache) if isinstance(c, ArraysCache)}


def _log_prefill_cost(model, expert_cache, base, t0, tokens: int, kv: int = -1) -> None:
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
    if expert_cache is None or base is None or tokens <= 0:
        return
    read_gb = (expert_cache.bytes_read - base[0]) / 1e9
    blocked = expert_cache.disk_wait_s - base[1]
    copy = expert_cache.materialize_s - base[3]
    turn_trace.prefill_chunk(tokens, elapsed, read_gb, blocked, copy, kv)
    if elapsed < 1.0:
        return
    moe_layers = int(getattr(model, "_pagedmoe_info", {}).get("moe_layers") or 0)
    layers = expert_cache.prefill_layers - base[2]
    passes = f"{layers / moe_layers:.1f} pass" if moe_layers else f"{layers} layer"
    print(
        f"[prefill] {tokens} tok in {elapsed:.1f}s, {read_gb:.0f} GB experts "
        f"({passes}es over the mass, {read_gb / elapsed:.1f} GB/s wall), "
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
    * **Non-trimmable caches** (qwen3-next / qwen4_exp hybrid): also single-
      pass. Mid-stop snapshots used to cost one expert-mass read each and are
      not recoverable by trim anyway; exact-offset forks + insert_cache cover
      append-mostly agent reuse.
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
    # Both notes below append this prompt to _recent_prompts. The mid-snap
    # anchor needs the one before it, or it matches itself and lands at the end.
    prev_prompt = _recent_prompts[-1] if _recent_prompts else None
    if _PREFIX_DEBUG:
        _log_prefix_diagnostics(tokenizer, ctx_tokens, cached)
        print(f"[prefix]   snapshot boundaries: {[b - cached for b in bounds]}", flush=True)
    else:
        _note_history_rewrite(tokenizer, ctx_tokens, _stable_prefix_bound(tokenizer, ctx_tokens))
    if not bounds:
        return

    step_cap = int(kwargs.get("prefill_step_size") or config.ADAPTIVE_PREFILL_MAX or 2048)
    progress = kwargs.get("prompt_progress_callback") or (lambda *_: None)
    total = len(prompt)
    store = _req_ctx.store
    budget = getattr(store, "max_bytes", 1 << 62)
    trimmable = can_trim_prompt_cache(cache)
    head = _first_message_boundary(tokenizer, ctx_tokens)
    max_pin = prefix_cache.pin_cap(len(ctx_tokens))
    # Host-declared front matter. Pinned so a host that rewrites its history
    # (compaction) resumes here instead of at kv=0.
    stable_at = _stable_prefix_bound(tokenizer, ctx_tokens)
    if stable_at > prefix_cache.PIN_CEILING_TOKENS:
        stable_at = 0
    # Front matter before the memory block. A layout that cannot trim a disk
    # file back to it gets a head file there, so a memory edit resumes from
    # disk instead of kv=0. The stop costs one more pass over the expert
    # mass, so it only happens while that file is missing.
    # The system span before the codebase map is the same in every workspace,
    # so a shared file there serves a new or changed workspace. Same rules as
    # the head: one stop, only while its file is missing.
    head_at = shared_at = 0
    if stable_at and _pending_stable_head:
        head_at = _stable_prefix_bound(tokenizer, ctx_tokens, _pending_stable_head)
    if stable_at and _pending_stable_shared:
        shared_at = _stable_prefix_bound(tokenizer, ctx_tokens, _pending_stable_shared)
    can_stop = (
        draft_model is None
        and _served_model is not None
        and model is _served_model
        and prefix_disk.supports(_req_ctx.model_key)
        and not prefix_disk.trims_exactly(cache)
    )
    head_ok = 0 < head_at < stable_at
    stops = []
    if (
        can_stop
        and 0 < shared_at < (head_at if head_ok else stable_at)
        and cached < shared_at
        and not prefix_disk.has(_req_ctx.model_key, ctx_tokens[:shared_at])
    ):
        stops.append((shared_at, prefix_disk.KIND_SHARED))
    if (
        can_stop
        and head_ok
        and cached < head_at
        and not prefix_disk.has(_req_ctx.model_key, ctx_tokens[:head_at])
    ):
        stops.append((head_at, prefix_disk.KIND_HEAD))
    # A whole conversation the host asked to keep. Any layout: a chat file is
    # only ever served whole. The cache already holding exactly the
    # conversation is saved as it is; one that holds more cannot be cut back.
    chat_at = _stable_prefix_bound(tokenizer, ctx_tokens, _pending_chat) if _pending_chat else 0
    if (
        chat_at >= _CHAT_MIN_TOKENS
        and draft_model is None
        and _served_model is not None
        and model is _served_model
        and prefix_disk.supports(_req_ctx.model_key)
        and cached <= chat_at < len(ctx_tokens) - 1
        and not prefix_disk.has(_req_ctx.model_key, ctx_tokens[:chat_at])
    ):
        if cached == chat_at:
            gpu_trail.mark(f"prefix-disk save {prefix_disk.KIND_CHAT} {chat_at} tok")
            prefix_disk.save(
                _req_ctx.model_key, ctx_tokens[:chat_at], cache, model, budget,
                kind=prefix_disk.KIND_CHAT, tag=_pending_chat_tag,
            )
        else:
            stops.append((chat_at, prefix_disk.KIND_CHAT))
    stops.sort()

    # Recurrent state at fixed positions inside the span a disk file will
    # hold, so a later prompt that changes part way in (an app update that
    # edits the system prompt) resumes from the last one before the change
    # instead of prefilling the file again. Taken inside the passes this
    # request already runs; a capture that fails just leaves them out.
    disk_ck: dict = {}
    ck_rel: list[int] = []
    save_end = max(stable_at, shared_at, head_at)
    if (
        can_stop
        and prefix_disk.ckpt_every() > 0
        and save_end > cached
        and inpass_snap.supports(model, cache[:n_model_layers])
    ):
        ck_rel = [p - cached for p in prefix_disk.ckpt_positions(cached, save_end)]

    def _disk_ckpts(end: int) -> dict:
        return {cached + r: s for r, s in disk_ck.items() if s and cached + r < end}

    # What each prefill chunk of this request ran, so a disk file records how
    # its end was written and a later engine can check it by replaying that.
    chunk_log: list = []

    def _replay_for(n_abs: int):
        return _replay_spec(chunk_log, cached, prompt, n_abs)

    def _stop_at(at: int, kind: str, processed: int) -> int:
        tgt = at - cached
        got = _prefill_remainder(
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
            target=tgt,
            ckpt_at=ck_rel,
            ckpt_out=disk_ck,
            chunk_log=chunk_log,
        )
        if got == tgt:
            chat = kind == prefix_disk.KIND_CHAT
            if kind == prefix_disk.KIND_HEAD or (chat and not trimmable):
                hybrid_ckpt.capture(_req_ctx.model_key, ctx_tokens[:at], cache)
            gpu_trail.mark(f"prefix-disk save {kind} {at} tok")
            prefix_disk.save(
                _req_ctx.model_key, ctx_tokens[:at], cache, model, budget, kind=kind,
                ckpts=None if chat else _disk_ckpts(at), replay=_replay_for(at),
                **({"tag": _pending_chat_tag} if chat else {}),
            )
        return got

    def _run_stops(limit: int) -> int:
        processed = 0
        for at, kind in stops:
            if at - cached <= processed or at >= cached + limit:
                continue
            processed = _stop_at(at, kind, processed)
        return processed

    if trimmable:
        # One fused pass over the whole remainder (leave 1 token for decode,
        # same as generate_step), then fork prefixes. No mid-stop = no second
        # expert-mass read on a cold agent prompt.
        target = max(0, total - 1)
        processed = _run_stops(target)
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
            chunk_log=chunk_log,
        )
        abs_off = cached + processed
        fork_bounds = list(bounds)
        if abs_off > cached:
            fork_bounds.append(abs_off)
        if cached < chat_at < abs_off and chat_at not in fork_bounds:
            fork_bounds.append(chat_at)
        # A fork of a resumed cache shares its 8-bit prefix, which would then
        # stay alive next to the re-quantized copy for all of decode. The end
        # of the turn drops every fork anyway (each is a prefix of what it
        # inserts), and a cancelled request still reaches that insert.
        n_snap = 0
        if not kvmem.has_resumed(cache):
            n_snap = _fork_prefix_snapshots(
                store,
                _req_ctx.model_key,
                ctx_tokens,
                cache,
                fork_bounds,
                cached=cached,
                processed=processed,
                budget=budget,
            )
        pin_at = prefix_cache.pick_pin_bound(
            bounds + [head] + ([stable_at] if stable_at else []),
            max_pin=max(max_pin, stable_at),
            prefer=stable_at or head,
        )
        pinned = False
        if pin_at and _kv_copy_fits(cache):
            pinned = _stable_pin.update(
                _req_ctx.model_key, ctx_tokens, cache, pin_at
            )
        if pinned and pin_at == stable_at:
            _save_pin_to_disk(model, budget, replay_for=_replay_for)
        elif stable_at and pin_at == stable_at:
            _warm_keeps_stable_on_disk(model, budget, ctx_tokens[:stable_at])
        if _PREFIX_DEBUG or n_snap or pinned:
            print(
                f"[prefix]   single-pass prefill {processed}/{total}; "
                f"forked {n_snap} prefix snapshot(s)"
                + (f"; pinned head {pin_at} tok" if pinned else ""),
                flush=True,
            )
    else:
        # Hybrid / linear-attention caches cannot trim, so we cannot fork a
        # KV snapshot at an arbitrary prefix length after the fact.  For
        # append-mostly agent traffic the key stable boundary is the point
        # where the previous turn's prompt diverges from this one: everything
        # before that point is identical across turns, so a KV snapshot there
        # is a valid starting point for every future turn.
        #
        # We find that boundary from _recent_prompts and do a single mid-
        # prefill stop there, deepcopy the KV (cheap ~2-3 GB memcpy), and
        # continue.  The extra cost vs a pure single pass is negligible: the
        # prefill chunks are the same size; we just record the state at one
        # chunk boundary instead of discarding it.
        # A pin that grows past the host's stable front matter covers history
        # the host may rewrite, and a hybrid pin cannot serve a partial match.
        if (
            head <= cached <= max_pin
            and (not stable_at or cached <= stable_at)
            and _kv_copy_fits(cache)
            and _stable_pin.update(_req_ctx.model_key, ctx_tokens, cache, cached)
        ):
            print(f"[prefix]   pinned served head {cached} tok", flush=True)
        # The state this turn resumed from. Its store entry is the first thing
        # eviction takes once longer entries land, and a request that branches
        # back to it (retry, recovery) would otherwise prefill cold.
        if cached:
            hybrid_ckpt.capture(_req_ctx.model_key, ctx_tokens[:cached], cache)

        target = max(0, total - 1)

        # One stop at the end of the front matter, once per change of it: the
        # next request that rewrites history resumes from this pin. On a
        # streamed MoE the stop costs one extra partial chunk (one more pass
        # over the expert mass); later requests start at or past it.
        pre = _run_stops(target)
        # Stopped at the end of a kept conversation: that is where the next
        # ask starts, so it is this turn's snapshot (the anchor is never past it).
        chat_snapped = False
        if chat_at and cached + pre == chat_at:
            cost = sum(getattr(c, "nbytes", 0) for c in cache)
            try:
                if cost * 2 <= budget:
                    store.insert_cache(
                        _req_ctx.model_key, list(ctx_tokens[:chat_at]), copy.deepcopy(cache), cache_type="user"
                    )
                    chat_snapped = True
                    print(f"[prefix]   non-trimmable snapshot @ {chat_at} tok (conversation end)", flush=True)
            except Exception as e:
                print(f"[prefix]   conversation snapshot failed: {e!r}", flush=True)
        if stable_at and cached + pre < stable_at < cached + target:
            pre_target = stable_at - cached
            pre = _prefill_remainder(
                model,
                prompt,
                cache,
                n_model_layers=n_model_layers,
                draft_model=draft_model,
                draft_cache=draft_cache,
                step_cap=step_cap,
                progress=progress,
                total=total,
                processed=pre,
                target=pre_target,
                ckpt_at=ck_rel,
                ckpt_out=disk_ck,
                chunk_log=chunk_log,
            )
            if pre == pre_target:
                hybrid_ckpt.capture(_req_ctx.model_key, ctx_tokens[:stable_at], cache)
                pin_cost = sum(getattr(c, "nbytes", 0) for c in cache)
                if pin_cost * 2 > budget:
                    print(
                        f"[prefix]   stable pin skipped: needs {pin_cost * 2 >> 20} MB "
                        f"of {budget >> 20} MB budget",
                        flush=True,
                    )
                elif _kv_copy_fits(cache) and _stable_pin.update(
                    _req_ctx.model_key, ctx_tokens, cache, stable_at
                ):
                    print(
                        f"[prefix]   pinned stable front matter @ {stable_at} tok "
                        f"({pin_cost / 1e9:.2f} GB)",
                        flush=True,
                    )
                    _save_pin_to_disk(model, budget, _disk_ckpts(stable_at), _replay_for)
        elif stable_at and cached == stable_at:
            _warm_keeps_stable_on_disk(model, budget, ctx_tokens[:stable_at])

        # Compute the stable-prefix length: how many tokens this prompt shares
        # with the previous one.  That position is where the generation prompt
        # from the previous turn ended and new conversation content begins in
        # the current prompt.  A KV snapshot here is reusable across turns.
        _mid_snap_abs = 0
        # Start from the reachable range: prefill holds the final token back so
        # stream_generate still has something to process.
        _anchor = cached + target
        _n = 0
        if prev_prompt is not None:
            for _a, _b in zip(prev_prompt, ctx_tokens):
                if _a != _b:
                    break
                _n += 1
        # A previous prompt from another conversation (a warmup, a side
        # request, a switched chat) shares little more than the template
        # header past what the cache already matched, so anchoring on it would
        # not snapshot anything useful. Treat this like a first turn instead.
        if prev_prompt is not None and _n - cached >= _MIN_MID_SNAPSHOT_TOKENS:
            # The shared span can reach or pass the end of the reachable range
            # when this whole prompt is a prefix of the previous one: a retried
            # turn, or one the client shortened. Clamp instead of skipping. The
            # shared region is real either way, only the offset has to be a
            # place prefill can actually stop at.
            _anchor = min(_anchor, _n)
            # A host that re-appends the same reminder block every turn drops
            # the old copy, so the previous prompt's tail after _n is volatile.
            # When this prompt ends in that same tail, anchor before it here
            # too: the next turn then shares everything up to it, instead of
            # only up to where the previous turn's copy sat.
            _vol = len(prev_prompt) - _n
            if (
                0 < _vol <= _VOLATILE_TAIL_MAX
                and len(ctx_tokens) - _vol > _anchor
                and ctx_tokens[-_vol:] == list(prev_prompt[-_vol:])
            ):
                _anchor = min(cached + target, len(ctx_tokens) - _vol)
            elif 0 < _vol <= _VOLATILE_TAIL_MAX:
                # The previous tail can carry one more host message than this
                # one (a nudge that has since moved into history), so the exact
                # match fails. Fall back to the suffix both prompts share and
                # anchor at the first whole message inside it.
                _sfx = 0
                _cap = min(len(prev_prompt), len(ctx_tokens)) - _n
                while (
                    _sfx < _cap
                    and _sfx <= _VOLATILE_TAIL_MAX
                    and ctx_tokens[-1 - _sfx] == prev_prompt[-1 - _sfx]
                ):
                    _sfx += 1
                if 0 < _sfx <= _VOLATILE_TAIL_MAX:
                    _eos_ids = _message_ends(tokenizer)
                    for _p in range(len(ctx_tokens) - _sfx, len(ctx_tokens)):
                        if ctx_tokens[_p - 1] in _eos_ids:
                            if _p > _anchor:
                                _anchor = min(cached + target, _p)
                            break
        else:
            # First turn, or nothing useful to diff against. The end snapshot
            # already covers a plain append, so put this one before a short
            # final message (a reminder the host re-appends and drops next
            # turn) instead of a token away from the end.
            _msg = _last_message_start(tokenizer, ctx_tokens)
            if 0 < _msg < _anchor and len(ctx_tokens) - _msg <= _VOLATILE_TAIL_MAX:
                _anchor = _msg
        # Never anchor at or past the think-open tail. The generation prompt
        # ends "<|im_start|>assistant\n<think>\n" with a lone newline, but that
        # same turn replayed inside the next prompt renders as
        # "<think>\n\n</think>\n\n" - a different token directly after <think>.
        # A key that includes <think> therefore diverges one token later on
        # every following turn, and a non-trimmable cache cannot be cut back to
        # the match, so the entry is dead weight that still costs a slot.
        # Stopping short of <think> is what makes the key replay-stable.
        _tail = _find_think_tail(tokenizer, ctx_tokens)
        if 0 < _tail < _anchor:
            _anchor = _tail
        # Same for the role header. The shared span often runs a few tokens
        # into "<|im_start|>assistant", but the next request may put a user or
        # tool message there instead, so the key must end at the message end.
        _eos = _message_ends(tokenizer)
        for _i in range(_anchor - 1, max(_anchor - 16, cached) - 1, -1):
            if ctx_tokens[_i] in _eos:
                _anchor = _i + 1
                break
        if (_anchor - cached) - pre >= _MIN_MID_SNAPSHOT_TOKENS:
            _mid_snap_abs = _anchor
        elif not chat_snapped:
            # Say why, so a run that quietly stops reusing its prefix is
            # diagnosable from the log instead of only from tok/s.
            print(
                f"[prefix]   no mid-snap: anchor {_anchor} vs cached {cached}, "
                f"limit {cached + target}, delta {_anchor - cached} "
                f"(need >= {_MIN_MID_SNAPSHOT_TOKENS})",
                flush=True,
            )

        n_snap = 0

        def _store_mid_snap(make_entry, snap_cost: int, how: str) -> int:
            if snap_cost * 2 > budget:
                print(
                    f"[prefix]   mid-snap skipped: needs {snap_cost * 2 >> 20} MB "
                    f"of {budget >> 20} MB budget",
                    flush=True,
                )
                return 0
            try:
                store.insert_cache(
                    _req_ctx.model_key,
                    list(ctx_tokens[:_mid_snap_abs]),
                    make_entry(),
                    cache_type="user",
                )
            except Exception as _e:
                print(f"[prefix]   mid-snap failed: {_e!r}", flush=True)
                return 0
            print(
                f"[prefix]   non-trimmable mid-snap @ {_mid_snap_abs} tok "
                f"({(_mid_snap_abs - cached) / total * 100:.0f}% of prefill{how})",
                flush=True,
            )
            return 1

        inpass = (
            _mid_snap_abs > 0
            and config.HYBRID_INPASS_SNAP
            and draft_model is None
            and inpass_snap.supports(model, cache[:n_model_layers])
        )
        if inpass:
            _mid_rel = _mid_snap_abs - cached
            snap: dict = {}
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
                processed=pre,
                target=target,
                snap_at=_mid_rel,
                snap_out=snap,
            )
            slots = snap.get("slots")
            ckpt = (
                hybrid_ckpt.from_slots(
                    _req_ctx.model_key, ctx_tokens[:_mid_snap_abs], cache, slots
                )
                if slots
                else None
            )
            if ckpt is None:
                inpass_snap.disable("no recurrent state captured at the anchor")
            else:

                def _entry():
                    e = hybrid_ckpt.restore(cache, ckpt)
                    if e is None:
                        inpass_snap.disable("could not rewind the cache to the anchor")
                        raise ValueError("in-pass restore refused")
                    return e

                n_snap = _store_mid_snap(
                    _entry, sum(getattr(c, "nbytes", 0) for c in cache), ", in-pass"
                )
        elif _mid_snap_abs > 0:
            # Two-phase prefill: run up to the stable boundary, store a snapshot,
            # then continue to the full target.
            _mid_rel = _mid_snap_abs - cached  # relative to prompt start
            processed_a = _prefill_remainder(
                model,
                prompt,
                cache,
                n_model_layers=n_model_layers,
                draft_model=draft_model,
                draft_cache=draft_cache,
                step_cap=step_cap,
                progress=progress,
                total=total,
                processed=pre,
                target=_mid_rel,
            )
            if processed_a == _mid_rel:
                # KV is exactly at the stable boundary: no trim needed.
                hybrid_ckpt.capture(
                    _req_ctx.model_key, ctx_tokens[:_mid_snap_abs], cache
                )
                n_snap = _store_mid_snap(
                    lambda: copy.deepcopy(cache),
                    sum(getattr(c, "nbytes", 0) for c in cache),
                    "",
                )
            else:
                print(
                    f"[prefix]   mid-snap skipped: prefill stopped at {processed_a}, "
                    f"wanted {_mid_rel}",
                    flush=True,
                )
            # Continue to full target
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
                processed=processed_a,
                target=target,
            )
        else:
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
                processed=pre,
                target=target,
            )

        # Record the exact end-of-prefill offset for truly append-only
        # follow-ups (same prefix, more tokens at the tail). Skip this when
        # we already have a mid-snap: the mid-snap at the stable boundary is
        # a better hit for every subsequent turn, and the end-of-prefill entry
        # (which ends with the generation-prompt tokens) is the one that causes
        # the 3-token divergence on the next turn. Storing both wastes a
        # deepcopy and a cache slot without any reuse benefit.
        abs_off = cached + processed
        fork_bounds = list(bounds)
        if abs_off > cached and n_snap == 0:
            fork_bounds.append(abs_off)
        n_snap += _fork_prefix_snapshots(
            store,
            _req_ctx.model_key,
            ctx_tokens,
            cache,
            fork_bounds,
            cached=cached,
            processed=processed,
            budget=budget,
        )
        print(
            f"[prefix]   non-trimmable prefill {processed}/{total}; "
            f"snapshots: {n_snap}"
            + (f" (mid @ {_mid_snap_abs}, end @ {abs_off})" if _mid_snap_abs else f" @ {abs_off}"),
            flush=True,
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


def _send_json(handler, status: int, payload: dict) -> None:
    handler._set_completion_headers(status)
    handler.end_headers()
    handler.wfile.write(json.dumps(payload).encode())
    handler.wfile.flush()


def _install_config_route() -> None:
    """GET/POST /config (live_config.py), plus quiet hang-ups on GET.

    Served on the HTTP thread, so a change lands mid-prefill; the next layer
    that reads the setting picks it up.
    """
    if getattr(_mlx_server.APIHandler, "_es_config_route", False):
        return
    from . import live_config

    _orig_do_get = _mlx_server.APIHandler.do_GET
    _orig_do_post = _mlx_server.APIHandler.do_POST

    # Debuggers and browser tooling probe local ports (GET /json/version) and
    # hang up before the 404 body is written.
    def _do_get(self):
        try:
            if self.path == "/config":
                return _send_json(
                    self, 200, {"live": live_config.live_keys(), "values": live_config.values()}
                )
            return _orig_do_get(self)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _do_post(self):
        if self.path == "/stage/boost":
            # Served on the HTTP thread while the stage runs on the worker.
            try:
                return _send_json(self, 200, {"ok": True, "boosted": stage_gentle.boost()})
            except (BrokenPipeError, ConnectionResetError):
                return None
        if self.path != "/config":
            return _orig_do_post(self)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n)) if n > 0 else {}
            env = body.get("env") if isinstance(body, dict) else None
            if not isinstance(env, dict):
                return _send_json(self, 400, {"error": 'expected {"env": {...}}'})
            result = live_config.apply(env)
            return _send_json(self, 409 if result["rejected"] else 200, result)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except ValueError as e:
            try:
                _send_json(self, 400, {"error": str(e)})
            except (BrokenPipeError, ConnectionResetError):
                pass

    _mlx_server.APIHandler.do_GET = _do_get
    _mlx_server.APIHandler.do_POST = _do_post
    _mlx_server.APIHandler._es_config_route = True


def install_patches(model=None):
    """Apply every mlx-lm patch a streamed model needs, and return nothing.

    Split out of `main()` so `bench/session_memory.py` can replay the real
    serving path - prompt-cache handover, KV conversion, snapshotting, the
    memory guards - instead of an approximation of it. A memory benchmark that
    measures a different code path than production is worse than no benchmark.

    `model` short-circuits the ModelProvider hook for callers that loaded the
    model themselves.
    """
    global _model, _served_model
    if model is not None:
        _model = _served_model = model

    from .mla_v2 import install as install_mla_v2
    from .vision_chat import install as install_vision_chat

    install_mla_v2()
    install_vision_chat()
    _install_adaptive_prefill_generate()

    # Streamed models must use the single-request path: the batch engine
    # bypasses stream_generate (so no KV quantization, no relieve_pressure)
    # and would prefill multiple prompts concurrently - each of which streams
    # gigabytes of experts.
    _orig_provider_load = _mlx_server.ModelProvider.load

    def _provider_load(self, *args, **kwargs):
        global _model, _served_model, _load_mode
        result = _orig_provider_load(self, *args, **kwargs)
        model = _served_model = self.model
        mt = getattr(getattr(model, "args", None), "model_type", None)
        if mt == "deepseek_v32":
            _install_deepseek_tool_parser(self.tokenizer)
        elif minimax_tools.template_uses(getattr(self.tokenizer, "chat_template", None)):
            _install_minimax_tool_parser(self.tokenizer)
        else:
            # No-op unless the chat template looks like Kimi (tool_declare).
            _install_kimi_tool_parser(self.tokenizer)
        _install_json_tools_unwrap(self.tokenizer)
        _stop_on_endoftext(self.tokenizer)
        fidelity.configure(self.tokenizer)
        rescue.configure(self.tokenizer)
        try:
            verify_decode.configure(self.tokenizer)
        except Exception as e:
            print(f"[verify] indent tokens skipped: {e!r}", flush=True)
        # Idempotent: the PagedMoE loader already did this, but a model that
        # arrived through mlx-lm's own loader still needs it.
        prefill_fused.install(model)
        _clamp_prefill_step(self, model)
        # The batch path skips stream_generate, so hybrid (non-trimmable) KV
        # never gets prefix snapshots and every turn re-prefills the prompt.
        self.is_batchable = False
        if getattr(model, "_pagedmoe_cache", None) is not None:
            _model = model
        _load_mode = (getattr(model, "_pagedmoe_info", None) or {}).get("mode")
        return result

    _mlx_server.ModelProvider.load = _provider_load

    # Record which stored prefix the request resumed from, so the
    # stream_generate wrapper knows where the user segment ends in `rest`
    # coordinates and where to store the snapshot.
    _orig_fetch = LRUPromptCache.fetch_nearest_cache

    def _fetch(self, model_key, tokens):
        turn_trace.bind_model(_model)
        turn_trace.mark("fetch")
        t_fetch = time.perf_counter()
        cache, rest = _fetch_inner(self, model_key, tokens)
        if cache is not None:
            # The prefill reads this cache on generation_stream; a lazy op
            # left in it from the default stream is a Metal hang.
            with mx.stream(generation_stream):
                prefix_disk.realize(cache)
        if turn_trace.enabled():
            turn_trace.note(
                prompt_tokens=len(tokens),
                cached_tokens=len(tokens) - len(rest),
                fetch_s=round(time.perf_counter() - t_fetch, 4),
            )
        return cache, rest

    def _fetch_inner(self, model_key, tokens):
        src = "none"
        if live_config.begin_request():
            _drop_settings_kv(self)
        if config.PROMPT_CACHE_MOVE and _model is not None:
            cache, rest = kvmem.take_nearest_cache(self, model_key, tokens)
        else:
            cache, rest = _orig_fetch(self, model_key, tokens)
        # insert keeps the live conversation even over budget. If this request
        # did not continue it (new chat), drop it now, before the prefill
        # allocates, not after. A hybrid entry this prompt branched from stays:
        # the checkpoint lookup below restores onto it.
        freed = kvmem.bound_store(self, keep=_branch_keep(self, model_key, tokens))
        if freed and config.MEM_DEBUG:
            print(f"[paged-moe mem] prompt cache trimmed {freed / 1e9:.2f} GB on fetch", flush=True)
        cached = 0 if cache is None else max(0, len(tokens) - len(rest))
        if cached:
            src = "store"
        pin_hit = _stable_pin.lookup(model_key, list(tokens))
        if _served_model is not None and prefix_disk.enabled():
            have = max(cached, pin_hit[2] if pin_hit is not None else 0)
            gpu_trail.mark(f"prefix-disk load ({len(tokens)} tok prompt, {have} cached)")
            got = prefix_disk.load(
                model_key, list(tokens), _served_model, have, getattr(self, "max_bytes", 1 << 62),
                replay=lambda c, ids, sizes: _replay_chunks(_served_model, c, ids, sizes),
            )
            if got is not None and prefix_disk.last_kind == prefix_disk.KIND_CHAT:
                # Served as this request's cache: a pin would hold a second copy.
                cache, rest, cached = got[1], tokens[len(got[0]):], len(got[0])
                src = "disk"
            elif got is not None and _stable_pin.put(model_key, got[0], got[1]):
                pin_hit = _stable_pin.lookup(model_key, list(tokens))
                src = "disk"
        if pin_hit is not None:
            pin_cache, pin_rest, pin_n = pin_hit
            if pin_n > cached:
                cache, rest = pin_cache, pin_rest
                if src != "disk":
                    src = "pin"
                if _PREFIX_DEBUG:
                    print(f"[prefix]   stable-pin hit {pin_n} tok (store had {cached})", flush=True)
                cached = pin_n
        if hybrid_ckpt.enabled() and (cache is None or not can_trim_prompt_cache(cache)):
            try:
                gpu_trail.mark(f"hybrid checkpoint resume ({cached} cached)")
                ck = hybrid_ckpt.resume(self, model_key, list(tokens), cached)
            except Exception as e:
                ck = None
                print(f"[prefix]   hybrid checkpoint skipped: {e!r}", flush=True)
            if ck is not None:
                print(
                    f"[prefix]   hybrid checkpoint hit {ck[2]} tok (store alone had "
                    f"{cached}); {hybrid_ckpt.count()} held, "
                    f"{hybrid_ckpt.nbytes() / 1e9:.2f} GB",
                    flush=True,
                )
                cache, rest, cached = ck
                src = "hybrid"
        if src in ("pin", "disk"):
            _note_short_reuse(self, model_key, tokens, cached)
        if cache is not None and len(rest) > 1:
            _dequantize_kv_cache(cache, _model)
        _note_kv_room(len(tokens))
        turn_trace.note(reuse_src=src if cache is not None else "none")
        _req_ctx.store = self
        _req_ctx.model_key = model_key
        seq = list(tokens)
        _req_ctx.tokens = seq
        # Same sequence, separate slot: the snapshot path clears .tokens before
        # generate_step runs, and the drafter needs the prefix the cache served.
        _req_ctx.draft_history = seq
        _req_ctx.cached = len(tokens) - len(rest)
        if (
            cache is None
            and getattr(self, "nbytes", 0) > 0
            and len(tokens) > _MIN_SNAPSHOT_TOKENS
        ):
            # Hybrid non-trimmable miss: store has entries but none is a prefix
            # of this prompt (assistant rewrite / mid-history digest) and we
            # cannot trim a longer hit.
            global _cold_misses
            _cold_misses += 1
            turn_trace.event("cold_miss", store_gb=round(self.nbytes / 1e9, 2), prompt=len(tokens))
            print(
                f"[prefix]   COLD MISS #{_cold_misses} with store "
                f"{self.nbytes / 1e9:.2f} GB / {len(self)} seq "
                f"(prompt {len(tokens)} tok) - cold prefill; checkpoints "
                f"{hybrid_ckpt.count()} held, {hybrid_ckpt.STATS['hits']} hits",
                flush=True,
            )
            try:
                lru = getattr(self, "_lru", None)
                ordering = getattr(lru, "_ordering", None) or []
                lrus = getattr(lru, "_lrus", None) or {}
                for cache_type in ordering:
                    for stored_model, key in list(lrus.get(cache_type) or []):
                        if stored_model != model_key:
                            continue
                        n = 0
                        for a, b in zip(key, tokens):
                            if a != b:
                                break
                            n += 1
                        diverged_mid = 0 < n < min(len(key), len(tokens))
                        print(
                            f"[prefix]     stored {cache_type} len={len(key)} "
                            f"common={n}"
                            + (
                                " (diverged mid-prefix)"
                                if diverged_mid
                                else (
                                    " (prompt shorter than snap; cannot trim)"
                                    if n == len(tokens) < len(key)
                                    else ""
                                )
                            ),
                            flush=True,
                        )
                        if diverged_mid:
                            # Decode a small window around the divergence point so
                            # we can trace which message content changed.
                            win = 8
                            lo = max(0, n - win)
                            stored_window = list(key[lo: n + win])
                            cur_window = list(tokens[lo: n + win])
                            try:
                                tok = getattr(_model, "tokenizer", None) or getattr(_model, "_tokenizer", None)
                                dec = getattr(tok, "decode", None) if tok else None
                                stored_text = dec(stored_window) if dec else str(stored_window)
                                cur_text = dec(cur_window) if dec else str(cur_window)
                            except Exception:
                                stored_text = str(stored_window)
                                cur_text = str(cur_window)
                            print(
                                f"[prefix]       diverge@{n}: stored={stored_text!r} cur={cur_text!r}",
                                flush=True,
                            )
            except Exception as e:
                print(f"[prefix]     miss-detail skipped: {e!r}", flush=True)
        return cache, rest

    LRUPromptCache.fetch_nearest_cache = _fetch

    def _note_short_reuse(store, model_key, tokens, cached):
        """Say where the store lost a turn that fell back to the front-matter pin."""
        try:
            lru = getattr(store, "_lru", None)
            best = 0
            best_len = 0
            for cache_type in getattr(lru, "_ordering", None) or []:
                for stored_model, key in list((getattr(lru, "_lrus", None) or {}).get(cache_type) or []):
                    if stored_model != model_key:
                        continue
                    n = 0
                    for a, b in zip(key, tokens):
                        if a != b:
                            break
                        n += 1
                    if n > best:
                        best, best_len = n, len(key)
            if best - cached < 1024:
                return
            turn_trace.event("short_reuse", served=cached, shared=best, key_len=best_len)
            print(
                f"[prefix]   short reuse: served {cached} tok, a stored entry "
                f"(len {best_len}) shares {best} but cannot trim and no checkpoint "
                f"covers {cached}..{best}",
                flush=True,
            )
        except Exception as e:
            print(f"[prefix]   short-reuse detail skipped: {e!r}", flush=True)

    # Explicit warm_prefix (OpenAI-compatible extension). Any client can POST
    # {"warm_prefix": true, "messages": [...], "max_tokens": 0} to prefill and
    # store that exact token sequence in the prompt-cache LRU without decoding.
    # Next request whose prompt starts with those tokens is delta-prefill only.
    _orig_handle_completion = _mlx_server.APIHandler.handle_completion

    def _handle_completion(self, request, stop_words):
        global _warm_prefix_active, _pending_copy_texts, _pending_copy_tok
        global _pending_cache_breaks, _pending_stable, _pending_stable_head, _pending_stable_shared
        global _pending_stable_all_shared, _pending_stable_client, _pending_chat, _pending_chat_tag
        body = getattr(self, "body", None) or {}
        activity.enter()
        turn_trace.begin(body, _model)
        trace_err = None
        _pending_cache_breaks = prefix_cache.explicit_bounds(body)
        _pending_stable = _stable_spec(self, body)
        _pending_stable_head = _stable_spec(self, body, head=True) if _pending_stable else None
        _pending_stable_shared = _stable_spec(self, body, shared=True) if _pending_stable else None
        stable_n = _stable_count(body, "stable_prefix_messages", "stablePrefixMessages")
        _pending_stable_all_shared = bool(_pending_stable) and stable_n == _stable_count(
            body, "stable_prefix_shared_messages", "stablePrefixSharedMessages"
        )
        _pending_stable_client = bool(_pending_stable) and body.get("stable_prefix_kind") == "client"
        chat_id = body.get("stable_prefix_chat_id")
        _pending_chat = _chat_spec(self, body) if isinstance(chat_id, str) and chat_id else None
        _pending_chat_tag = chat_id if _pending_chat else None
        warm = bool(body.get("warm_prefix") or body.get("warmPrefix"))
        _pending_copy_texts, _pending_copy_tok = [], None
        if config.VERIFY_COPY_SOURCES and not warm:
            try:
                _pending_copy_texts = _copy_texts(body)
                _pending_copy_tok = self.response_generator.model_provider.tokenizer
                host_n = len(body.get("copy_texts") or [])
                if host_n:
                    print(f"[verify] host copy texts={host_n}", flush=True)
            except Exception:
                _pending_copy_texts = []
        # Cross-thread: worker reads _warm_prefix_active, not _req_ctx.
        _warm_prefix_active = warm
        _req_ctx.warm_prefix = warm
        if warm and body.get("stage_gentle"):
            stage_gentle.begin()
        if warm:
            # 0 tokens: generate_step still runs prefill + the last-token KV
            # step, then yields nothing. mlx_lm.stream_generate's finalize
            # UnboundLocalError's on that empty stream; _stream_generate
            # catches it so insert_cache still stores the prompt key.
            self.max_tokens = 0
            self.stream = False
            n_msgs = len(body.get("messages") or [])
            print(
                f"[prefix] warm_prefix=1 messages={n_msgs} "
                f"(prefill + cache insert, no decode)",
                flush=True,
            )
        try:
            return _orig_handle_completion(self, request, stop_words)
        except BrokenPipeError:
            # Client (proxy) closed the connection mid-stream, not an error.
            # Letting this propagate would log a full traceback to stderr for
            # every OOM-killed or cancelled request. Silently discard it; the
            # generation loop already exited cleanly via its own finally block.
            trace_err = "client_closed"
        except Exception as e:
            trace_err = f"{type(e).__name__}: {str(e)[:200]}"
            raise
        finally:
            stage_gentle.end()
            turn_trace.end(trace_err)
            activity.leave()
            _warm_prefix_active = False
            _req_ctx.warm_prefix = False
            _pending_cache_breaks = []
            _pending_stable = None
            _pending_stable_head = None
            _pending_stable_shared = None
            _pending_stable_all_shared = False
            _pending_stable_client = False
            _pending_copy_texts, _pending_copy_tok = [], None

    _mlx_server.APIHandler.handle_completion = _handle_completion

    _install_config_route()

    # Enforce the byte budget mlx-lm only applies to batched requests. Without
    # this the store is bounded by --prompt-cache-size alone, i.e. by a *count*
    # of KV caches: six of GLM-4.7's at 24k tokens is 29 GB.
    _orig_insert = LRUPromptCache.insert_cache

    def _insert_cache(self, model_key, tokens, prompt_cache, **kwargs):
        if live_config.kv_stale():
            return
        t_ins = time.perf_counter()
        _orig_insert(self, model_key, tokens, prompt_cache, **kwargs)
        global _live_key
        keep = _branch_keep(self, model_key, tokens) + [(model_key, tokens)]
        _live_key = (model_key, tuple(tokens))
        freed = kvmem.bound_store(self, keep=keep)
        turn_trace.note(insert_s=round(time.perf_counter() - t_ins, 4), stored_tokens=len(tokens))
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
        global _pending_think_budget
        budget, _pending_think_budget = _pending_think_budget, None
        if budget is not None:
            kwargs["logits_processors"] = list(kwargs.get("logits_processors") or []) + [budget]
        if getattr(model, "_pagedmoe_cache", None) is not None:
            # Plain deepseek_v3 / Kimi MLA cannot consume QuantizedKVCache
            # (pe_scores path). Keep fp16; MLA KV is already compact.
            if _plain_mla_kv_quantize_unsafe(model):
                kwargs["kv_bits"] = None
                if not getattr(_stream_generate, "_mla_kv_noted", False):
                    print(
                        "[paged-moe] plain MLA (deepseek_v2/v3/kimi): "
                        "keeping fp16 KV (8-bit QuantizedKVCache breaks the latent cache)",
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
        # the paged-moeing worker evals, deterministically stalls a Metal
        # command buffer past the watchdog ("GPU Timeout Error").
        if kwargs.get("draft_model") is not None:
            prompt = kwargs.get("prompt")
            if prompt is not None and not isinstance(prompt, mx.array):
                p = mx.array(prompt, mx.uint32)
                mx.eval(p)
                kwargs["prompt"] = p
        try:
            # Warm / max_tokens=0: do NOT skip _orig_stream. generate_step must
            # still run the last prompt-token into the KV (snapshot leaves one).
            # mlx_lm.stream_generate then finalizes with `token=token` after
            # zero yields -> UnboundLocalError. Catch that so the worker's
            # insert_cache still runs. (_req_ctx.warm_prefix is HTTP-thread
            # local and is False here; use the module flag / max_tokens.)
            try:
                from . import dev_mitm

                prompt = kwargs.get("prompt")
                rem_n = len(prompt) if prompt is not None else 0
                dev_mitm.begin_response(
                    model,
                    rem=rem_n,
                    cached=getattr(_req_ctx, "cached", None),
                )
            except Exception:
                pass
            try:
                if budget is None:
                    yield from _orig_stream(model, *args, **kwargs)
                else:
                    # Logged as the close goes by: the server stops reading
                    # this generator at the stop token, so nothing after the
                    # loop runs.
                    toks, k = [], len(budget.close_ids)
                    for r in _orig_stream(model, *args, **kwargs):
                        toks.append(int(getattr(r, "token", -1)))
                        if toks[-1] == budget.end_id and budget.closed_in(toks[-k:]):
                            print(
                                f"[paged-moe] think budget: closed reasoning in-stream "
                                f"at {budget.chars}ch ({len(toks)} tokens in)",
                                flush=True,
                            )
                        yield r
            except UnboundLocalError:
                if not (
                    _warm_prefix_active or kwargs.get("max_tokens", -1) == 0
                ):
                    raise
                rem = kwargs.get("prompt")
                rem_n = len(rem) if rem is not None else 0
                print(
                    f"[prefix] warm complete remainder={rem_n} "
                    f"(decode skipped, max_tokens=0)",
                    flush=True,
                )
                return
        except RuntimeError as e:
            # Metal OOM mid-decode/prefill: free everything we can so the
            # *next* request has a chance, then surface a clear message.
            # (mlx-lm's request thread otherwise just dies with the raw
            # kIOGPUCommandBufferCallbackErrorOutOfMemory string.)
            msg = str(e)
            if "Insufficient Memory" in msg or "OutOfMemory" in msg:
                cache = getattr(model, "_pagedmoe_cache", None)
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
                turn_trace.event("metal_oom")
                print(
                    "[paged-moe] Metal OOM during generation - released the "
                    "expert slab; the next turn rebuilds it around the KV cache. "
                    "If this keeps happening, lower numCtx (which is also the KV "
                    "reserve) or PAGED_MOE_CACHE_GB for this model.",
                    flush=True,
                )
            raise
        finally:
            try:
                from . import dev_mitm

                dev_mitm.end_response(model)
            except Exception:
                pass
            t_rel = time.perf_counter()
            relieve_pressure(model)
            turn_trace.note(relieve_s=round(time.perf_counter() - t_rel, 4))

    _mlx_server.stream_generate = _stream_generate
    _install_replay_splice()
    _install_think_prefill()
    _install_tool_stream()
    _install_lenient_tool_parse()


_XML_FUNCTION_RE = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", re.DOTALL)
_XML_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def _lenient_xml_call(tool_text: str):
    m = _XML_FUNCTION_RE.search(tool_text)
    if not m:
        return None
    args = {}
    for name, value in _XML_PARAM_RE.findall(m.group(2)):
        try:
            args[name] = json.loads(value)
        except ValueError:
            args[name] = value
    return {"name": m.group(1), "arguments": args}


def _install_lenient_tool_parse():
    """Keep a tool call whose parameter values do not parse.

    mlx-lm's qwen3-coder parser falls back to ast.literal_eval for object and
    list params. A malformed value raises SyntaxError, which its formatter does
    not catch, so the request dies with nothing sent. Even the errors it does
    catch drop the call silently. Here a failed parse is retried leniently:
    values that are not valid JSON are passed through as raw strings and the
    client repairs or rejects them.
    """
    fmt_cls = getattr(_mlx_server, "ToolCallFormatter", None)
    if fmt_cls is None or getattr(fmt_cls.__call__, "_ga_lenient", False):
        return

    def __call__(self, tool_calls):
        result = []
        for tool_text in tool_calls or []:
            try:
                parsed = self._tool_parser(tool_text, self._tools)
            except Exception as e:
                parsed = _lenient_xml_call(tool_text)
                if parsed is None:
                    print(f"[paged-moe] dropped unparseable tool call ({type(e).__name__}: {e})", flush=True)
                    continue
                print(
                    f"[paged-moe] tool call {parsed['name']} kept with raw values "
                    f"({type(e).__name__}: {e})",
                    flush=True,
                )
            if not isinstance(parsed, list):
                parsed = [parsed]
            result.extend(self._format(tc) for tc in parsed)
        return result

    __call__._ga_lenient = True
    fmt_cls.__call__ = __call__


_TOOL_STREAM_EVERY = 8
_tool_stream_local = threading.local()


def _keep_open_tool_call(response):
    """Keep a tool call the model ended with a stop token instead of its close tag.

    The stop token comes out of mlx-lm's state machine with state None, and
    handle_completion only parses leftover tool text when the last state was
    "tool". So `<tool_call>{...}<|im_end|>` is thrown away without a log line
    and the client gets a reply with no text, no reasoning and no call: 569
    decoded tokens of an edit, gone, on qwen3-235b.
    """
    prev = None
    for gen in response:
        state = getattr(gen, "state", None)
        if state is None and prev == "tool" and dataclasses.is_dataclass(gen):
            print("[paged-moe] tool call ended on a stop token without its close tag; kept", flush=True)
            gen = dataclasses.replace(gen, state="tool")
            state = "tool"
        prev = state
        yield gen


def _install_tool_stream():
    """Stream tool-call text as SSE comments while the call is being written.

    mlx-lm writes nothing while the model is inside a tool call and sends the
    whole call at the end. A large file write then looks like a dead stream,
    the client times out, and because mlx-lm only notices a disconnect on
    write, decode keeps the GPU for the rest of max_tokens. Comment lines
    (": ga-tool <tokens> <json text>") keep the socket live, let a disconnect stop
    decode through the handler's ctx.stop(), and are ignored by plain SSE
    clients. The "ga-" prefix comes from Global Agent, the first client to
    read them.
    """
    handler_cls = getattr(_mlx_server, "APIHandler", None)
    gen_cls = getattr(_mlx_server, "ResponseGenerator", None)
    if handler_cls is None or gen_cls is None:
        return
    if getattr(gen_cls.generate, "_ga_tool_stream", False):
        return
    orig_handle = handler_cls.handle_completion
    orig_generate = gen_cls.generate

    def handle_completion(self, request, stop_words):
        _tool_stream_local.handler = self
        try:
            return orig_handle(self, request, stop_words)
        finally:
            _tool_stream_local.handler = None

    def _write(handler, pending):
        handler.wfile.write(f": ga-tool {len(pending)} {json.dumps(''.join(pending))}\n\n".encode())
        handler.wfile.flush()

    def _relay(response, handler):
        pending = []
        for gen in response:
            if getattr(gen, "state", None) == "tool":
                pending.append(gen.text)
                if len(pending) >= _TOOL_STREAM_EVERY:
                    _write(handler, pending)
                    pending = []
            elif pending:
                _write(handler, pending)
                pending = []
            yield gen
        if pending:
            _write(handler, pending)

    def generate(self, request, generation_args, progress_callback=None):
        handler = getattr(_tool_stream_local, "handler", None)
        started = {}

        # ": ga-cached <tokens>" goes out once, ahead of the first keepalive, so a
        # client can tell a cold read from the uncached tail of a warm prompt.
        def progress(processed, total):
            ctx = started.pop("ctx", None)
            if ctx is not None and handler is not None and getattr(handler, "stream", False):
                cached = int(getattr(ctx, "prompt_cache_count", 0) or 0)
                handler.wfile.write(f": ga-cached {cached}\n\n".encode())
            progress_callback(processed, total)

        ctx, response = orig_generate(
            self, request, generation_args, progress if progress_callback is not None else None
        )
        started["ctx"] = ctx
        response = _keep_open_tool_call(response)
        if handler is None or not getattr(handler, "stream", False):
            return ctx, response
        return ctx, _relay(response, handler)

    generate._ga_tool_stream = True
    handler_cls.handle_completion = handle_completion
    gen_cls.generate = generate


def _decoded_prefix(tokenizer, gen: list, text: str) -> tuple[int, int]:
    """Longest `gen[:k]` whose decoded text starts `text`: (k, chars it covers).

    Leading whitespace the model wrote before its reasoning is kept as its own
    tokens, since the client strips it from the text it sends back.
    """
    if not gen or not text:
        return 0, 0
    full = tokenizer.decode(gen)
    lead = 0 if text[:1].isspace() else len(full) - len(full.lstrip())
    common = 0
    for a, b in zip(full[lead:], text):
        if a != b:
            break
        common += 1
    lo, hi = 0, len(gen)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tokenizer.decode(gen[:mid])) - lead <= common:
            lo = mid
        else:
            hi = mid - 1
    for k in range(lo, 0, -1):
        got = tokenizer.decode(gen[:k])[lead:]
        if text.startswith(got):
            return k, len(got)
    return 0, 0


def _reuse_cut_tokens(gen_obj, tokenizer, prompt: list, text: str) -> tuple[list, str]:
    """Split `text` into the cut pass's own tokens and what still needs encoding.

    A think-budget cut stores prompt + the reasoning tokens decode got to. The
    resume sends that reasoning back as text, and encoding it again rarely
    gives the same ids (BPE has many spellings of one string, sampled text
    takes the odd ones), so the store matched 16 tokens in and re-prefilled
    the other 664 (9 s on the 235B). Same text either way; this only picks the
    spelling the KV cache already holds.
    """
    try:
        node = gen_obj.prompt_cache._trie._trie[gen_obj.model_provider.model_key]
        for tok in prompt:
            node = node[tok]
    except Exception:
        return [], text
    best, best_n = [], 0
    for gen in _stored_continuations(node):
        k, n = _decoded_prefix(tokenizer, gen, text)
        if n > best_n:
            best, best_n = gen[:k], n
    return best, text[best_n:]


def _stored_continuations(node: dict, limit: int = 4, budget: int = 1 << 16) -> list:
    """Token paths below a prompt-trie node that end at a stored entry, deepest
    per branch."""
    out, stack = [], [(node, [])]
    while stack and len(out) < limit and budget > 0:
        cur, path = stack.pop()
        budget -= 1
        kids = [(t, n) for t, n in cur.items() if t != "__value__"]
        if not kids:
            if path and "__value__" in cur:
                out.append(path)
            continue
        for t, n in kids:
            stack.append((n, path + [t]))
    return out


def _install_replay_splice():
    """Spell replayed assistant turns with their decoded ids (replay_splice.py)."""
    gen_cls = getattr(_mlx_server, "ResponseGenerator", None)
    orig = getattr(gen_cls, "_tokenize", None) if gen_cls else None
    if orig is None or getattr(orig, "_ga_splice", False):
        return

    def _tokenize(self, tokenizer, request, args):
        out = orig(self, tokenizer, request, args)
        if (
            not config.REPLAY_SPLICE
            or getattr(request, "request_type", "") != "chat"
            or not getattr(tokenizer, "has_chat_template", False)
        ):
            return out
        try:
            kw = dict(self.model_provider.cli_args.chat_template_args or {})
            if getattr(args, "chat_template_kwargs", None):
                kw.update(args.chat_template_kwargs)
            kw["tools"] = request.tools
            prompt = list(out[0])
            new = replay_splice.apply(
                tokenizer,
                self.prompt_cache,
                self.model_provider.model_key,
                request.messages,
                kw,
                prompt,
            )
        except Exception as e:
            print(f"[splice] skipped: {e!r}", flush=True)
            return out
        if new is None:
            return out
        from . import vision_chat

        if vision_chat.image_tokens(new) != vision_chat.image_tokens(prompt):
            print("[splice] skipped: image tokens differ from the vision prompt", flush=True)
            return out
        segs, kinds = replay_splice.resegment(prompt, out[1], out[2], new)
        return new, segs, kinds, out[3]

    _tokenize._ga_splice = True
    gen_cls._tokenize = _tokenize


def _install_think_prefill():
    """Resume a cut reasoning block instead of starting a new turn.

    A client sets chat_template_kwargs.ga_think_prefill to the reasoning it
    already has. The prompt is rendered exactly as before (same prefix, same
    cache hits), then the reasoning and the think close tag are appended so
    the model continues straight into its answer or tool call.
    """
    gen_cls = getattr(_mlx_server, "ResponseGenerator", None)
    orig = getattr(gen_cls, "_tokenize", None) if gen_cls else None
    if orig is None or getattr(orig, "_ga_prefill", False):
        return

    def _tokenize(self, tokenizer, request, args):
        global _pending_think_budget
        kw = getattr(args, "chat_template_kwargs", None)
        prefill = kw.pop("ga_think_prefill", None) if isinstance(kw, dict) else None
        budget = kw.pop("ga_think_budget", None) if isinstance(kw, dict) else None
        _pending_think_budget = _think_budget_for(tokenizer, budget)
        out = orig(self, tokenizer, request, args)
        if _pending_think_budget is not None:
            rendered = list(out[0])
            _pending_think_budget.open_in_prompt = (
                tokenizer.rfind_think_start(rendered) > tokenizer.rfind_think_end(rendered)
            )
        if (
            not isinstance(prefill, str)
            or not prefill.strip()
            or getattr(request, "request_type", "") != "chat"
            or not getattr(tokenizer, "has_thinking", False)
        ):
            return out
        prompt = list(out[0])
        text = prefill.strip() + "\n" + tokenizer.think_end + "\n\n"
        if tokenizer.rfind_think_start(prompt) <= tokenizer.rfind_think_end(prompt):
            text = tokenizer.think_start + "\n" + text
        own, rest = _reuse_cut_tokens(self, tokenizer, prompt, text)
        prompt += own + tokenizer.encode(rest, add_special_tokens=False)
        print(
            f"[paged-moe] think prefill: +{len(prompt) - len(out[0])} tokens"
            f" ({len(own)} reused from the cut pass)",
            flush=True,
        )
        return prompt, [prompt], ["assistant"], "normal"

    _tokenize._ga_prefill = True
    gen_cls._tokenize = _tokenize


# Set by _tokenize, taken by the next stream_generate (same worker thread).
_pending_think_budget = None


def _think_budget_for(tokenizer, spec):
    """ThinkBudget from chat_template_kwargs.ga_think_budget, or None.

    `spec` is {"chars": int, "close": str}; close must end with the think end
    tag. Anything malformed is ignored and the host's own cap still applies.
    """
    if not isinstance(spec, dict) or not getattr(tokenizer, "has_thinking", False):
        return None
    try:
        chars = int(spec.get("chars") or 0)
        close = str(spec.get("close") or "")
        if chars <= 0 or not close.strip():
            return None
        text = "\n\n" + close.strip() + "\n" + tokenizer.think_end
        return think_budget.ThinkBudget(tokenizer, chars, text)
    except Exception as e:
        print(f"[paged-moe] think budget skipped: {e!r}", flush=True)
        return None


_TERM_DRAIN_S = 4.0


def _install_graceful_term() -> None:
    """SIGTERM: let queued GPU work finish (bounded), then exit.

    The default SIGTERM ends the process with command buffers still in flight,
    which is the case a SIGKILL after a Metal OOM also hits; the driver has
    been seen keeping ~30 GB with no owner afterwards. generation_stream is
    thread local, so a synchronize from here would wait on an empty stream of
    this thread. Decode stops at its next token and drains on its own thread
    (activity.stop), and this waits for that. An idle engine exits at once.
    os._exit matches the old default (no atexit work either).
    """
    import signal

    def _on_term(signum, frame):
        activity.stop()
        deadline = time.monotonic() + _TERM_DRAIN_S
        while activity.decoding() and time.monotonic() < deadline:
            time.sleep(0.02)
        busy = activity.decoding()
        print(
            "[paged-moe] stop: "
            + (f"decode still running after {_TERM_DRAIN_S:.0f}s, exiting anyway" if busy
               else "GPU work drained, exiting"),
            flush=True,
        )
        if not busy:
            try:
                prefix_disk.stopped_cleanly()
            except Exception:
                pass
        os._exit(0)

    try:
        signal.signal(signal.SIGTERM, _on_term)
        # `kill -USR1 <pid>` writes every thread's Python stack to the log, for
        # a server that stops making progress.
        import faulthandler

        faulthandler.register(signal.SIGUSR1, all_threads=True)
    except Exception:
        pass


def main():
    _install_graceful_term()
    _mlx_server.load = _streamed_load

    for flag, value in _DEFAULT_ARGS.items():
        if flag not in sys.argv:
            sys.argv.extend([flag, value])

    # store_true flag (no value). Needed for Kimi / Laguna custom tokenizers.
    if "--trust-remote-code" not in sys.argv:
        sys.argv.append("--trust-remote-code")

    from .compat import install_transformers_compat

    install_transformers_compat()
    install_patches()
    prefix_disk.write_stamp()
    prefix_disk.housekeep()
    from . import mempressure, watchdog

    watchdog.start()
    # The slab starts the poller too, but a resident model has no slab, and
    # the prompt store reads it to shrink under pressure.
    mempressure.start()
    _mlx_server.main()


if __name__ == "__main__":
    main()
