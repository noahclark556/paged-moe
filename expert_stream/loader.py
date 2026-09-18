# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
load(): the public entry point.

    from expert_stream import load
    model, tokenizer = load("~/mlx-models/qwen3-235-4bit")
    # or an HF repo id, downloaded to MODELS_DIR on first use:
    model, tokenizer = load("mlx-community/Qwen3-30B-A3B-4bit")

What it does
------------
1. Resolve the model: local directory, or download the repo into MODELS_DIR
   (default ~/models/paged-moe).
2. Read the safetensors headers (cheap - no weight data) and size the model.
3. Decide the mode:
     - resident: the whole model fits in the RAM budget -> load it fully,
       exactly like plain mlx-lm. Fastest option; streaming machinery unused.
     - streamed: backbone (attention/routers/shared experts/embeddings) is
       loaded to RAM; routed experts stay on disk and are paged in on demand
       through the ExpertCache.
   Auto-picked by size, or forced with mode="resident"/"streamed" or the
   EXPERT_STREAM_MODE env var.
4. In streamed mode, swap every SwitchGLU for a StreamedSwitchGLU *before*
   evaluating weights, so expert tensors are never read during load.

The returned (model, tokenizer) behave exactly like mlx-lm's, so everything
in the mlx-lm ecosystem (stream_generate, prompt caches, the HTTP server,
chat templates) works unchanged.
"""

from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model, load_tokenizer

from . import config, kvmem
from .chat_compat import install_deepseek_v32_chat_template

# Any load() path (server, CLI, benches) needs the DeepSeek template shim.
install_deepseek_v32_chat_template()


def _shim_bytes_to_unicode() -> None:
    """Transformers 5 moved bytes_to_unicode off gpt2; Kimi remote code still imports it."""
    try:
        from transformers.models.gpt2 import tokenization_gpt2 as gpt2_tok

        if hasattr(gpt2_tok, "bytes_to_unicode"):
            return
        from transformers.convert_slow_tokenizer import bytes_to_unicode

        gpt2_tok.bytes_to_unicode = bytes_to_unicode
    except Exception:
        pass


_shim_bytes_to_unicode()
from .cache import ExpertCache
from .safetensors_index import read_headers
from .streaming import (
    expert_tensor_names,
    find_switch_glus,
    max_expert_count,
    patch_model,
)

# Reserved for KV cache, activations, and Metal scratch (on top of what the
# RAM_FRACTION budget already leaves for macOS + apps). Expert reads bypass
# the page cache (F_NOCACHE) so no extra allowance is needed for it.
#
# Used when we have no context length to size the KV cache from
# (EXPERT_STREAM_RESERVE_CTX unset). With one, the reserve is computed per model
# instead and this flat figure splits into activations-plus-scratch alone.
_HEADROOM_BYTES = 6 << 30

# Decode activations, Metal's recycled-buffer pool, and the sampler's scratch.
# Prefill needs far more, which is why it hands the slab back rather than
# budgeting for it here (see ExpertCache.enter_prefill).
_ACTIVATION_BYTES = 3 << 30


def _dense_enabled() -> bool:
    """Whether this build can stream a *dense* checkpoint's layers from disk.

    Two gates, both required. The env var is the switch; the spec lookup is
    because the dense package is a development-only part of the tree, so a
    build without it must degrade to the ordinary "does not fit" error rather
    than an ImportError.
    """
    import importlib.util
    import os

    if os.environ.get("EXPERT_STREAM_DENSE", "").strip().lower() not in (
        "1",
        "on",
        "true",
        "yes",
    ):
        return False
    try:
        return importlib.util.find_spec("expert_stream.dense") is not None
    except (ImportError, ValueError):
        return False


def resolve_model_path(path_or_repo: str) -> Path:
    """Local dir as-is; otherwise treat as an HF repo id and download it
    into MODELS_DIR (external SSD when mounted)."""
    p = Path(path_or_repo).expanduser()
    if p.exists():
        return p

    from huggingface_hub import snapshot_download

    dest = Path(config.MODELS_DIR) / path_or_repo.replace("/", "--")
    dest.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=path_or_repo,
        local_dir=str(dest),
        allow_patterns=[
            "*.json", "*.safetensors", "*.py", "tokenizer.model",
            "*.tiktoken", "*.txt", "*.jsonl", "*.jinja", "*.model",
        ],
    )
    return dest


# Budgets of every model load()ed into this process so far. The memory limit
# must cover their SUM: a speculative-decoding draft model (small, resident)
# is loaded after the streamed main model, and both are live at once. Sizing
# the limit to either one alone makes the Metal allocator stall waiting for
# memory that will never free - which surfaces as a "GPU Timeout Error" on
# whatever innocent op happens to be in flight (in practice: the draft's
# first decode step once prompt-cache snapshots have used up the headroom).
_applied_limits = {"budget": 0, "pool": 0}


def _set_memory_limits(
    active_budget_bytes: int,
    pool_bytes: int = 2 << 30,
    extra_bytes: int | None = None,
):
    """Cap Metal so recycled buffers can't silently eat the other half of RAM.

    active_budget_bytes ≈ backbone + expert cache. We allow a little overhead
    for KV/activations, but refuse to let the MLX buffer pool grow unbounded.

    extra_bytes is the Metal ceiling on top of the live weights. MoE leaves
    this at the flat 6 GB headroom (KV was already taken out of the expert
    cache, and enter_prefill can still hand the slab back). Dense residency
    cannot be handed back, so it passes its KV reserve + activation
    allowance instead of that flat figure.

    pool_bytes bounds the recycled-buffer pool. It must cover roughly one
    decode token's expert churn: expert tensors are uniform sizes, so a freed
    expert's buffers are recycled verbatim for the next miss - but only if the
    pool is big enough to still hold them. Too small and every miss pays
    zero-fill page faults for memory the process just released.
    """
    _applied_limits["budget"] += int(active_budget_bytes)
    extra = _HEADROOM_BYTES if extra_bytes is None else int(extra_bytes)
    memory = _applied_limits["budget"] + extra
    pool = max(_applied_limits["pool"], int(pool_bytes))
    _applied_limits["pool"] = pool
    try:
        if hasattr(mx, "set_cache_limit"):
            mx.set_cache_limit(pool)
        # Soft ceiling on live allocations.
        if hasattr(mx, "set_memory_limit"):
            mx.set_memory_limit(memory)
    except Exception:
        pass


def _neutralize_wired_limit():
    """Stop mlx-lm from pinning (wiring) most of physical RAM.

    mlx-lm raises the Metal *wired* limit to `max_recommended_working_set_size`
    both at server startup and inside a context manager around every
    generation.  Wired memory is unswappable; on this machine that limit is
    ~48 GB of 48 GB (the iogpu.wired_limit_mb sysctl was raised to 46000), so
    one heavy generation can pin essentially all RAM and hard-freeze macOS.

    A streaming engine deliberately cycles gigabytes of expert weights per
    token - the last thing it should do is wire them.  We (a) replace
    mlx_lm.generate.wired_limit with a no-op context manager and (b) set the
    process wired limit to 0 (nothing wired; Metal pages on demand).
    """
    import contextlib

    @contextlib.contextmanager
    def _noop_wired_limit(model=None, streams=None):
        yield

    try:
        import mlx_lm.generate as _gen

        _gen.wired_limit = _noop_wired_limit
    except Exception:
        pass
    try:
        mx.set_wired_limit(0)
    except Exception:
        pass


def load(
    path_or_hf_repo: str,
    tokenizer_config: dict | None = None,
    model_config: dict | None = None,
    adapter_path: str | None = None,
    lazy: bool = False,  # accepted for mlx-lm API compatibility; ignored
    return_config: bool = False,
    revision: str | None = None,
    *,
    mode: str | None = None,
    cache_gb: float | None = None,
    read_threads: int | None = None,
    prefetch_depth: int | None = None,
    verbose: bool = True,
):
    """Drop-in replacement for mlx_lm.load() with streaming support.

    Extra keyword args:
      mode            "auto" (default), "resident", or "streamed"
      cache_gb        expert cache budget in GB (streamed mode)
      read_threads    parallel SSD readers (streamed mode)
      prefetch_depth  speculative prefetch lookahead in layers, 0 = off
    """
    mode = mode or config.MODE
    read_threads = read_threads or config.READ_THREADS
    prefetch_depth = (
        prefetch_depth if prefetch_depth is not None else config.PREFETCH_DEPTH
    )
    if cache_gb is None:
        cache_gb = config.CACHE_GB

    t0 = time.perf_counter()
    model_path = resolve_model_path(path_or_hf_repo)

    # --- size the model from headers only (no weight data touched) ---------
    tensor_index = read_headers(model_path)
    total_bytes = sum(loc.nbytes for loc in tensor_index.values())
    ram = config.total_ram_bytes()
    budget = int(config.RAM_FRACTION * ram)

    # --- build the module tree with lazy (unmaterialized) weights ----------
    model, cfg = load_model(model_path, lazy=True, model_config=model_config)

    glus = find_switch_glus(model)
    expert_bytes = 0
    if glus:
        expert_bytes = sum(
            tensor_index[name].nbytes
            for name in expert_tensor_names(model, tensor_index)
        )
    backbone_bytes = total_bytes - expert_bytes

    # What the KV cache will want at the context this model is served with.
    # This comes straight out of the expert cache's budget, so guessing it with
    # a flat allowance is how a full-attention model ends up committing a slab
    # it cannot keep: GLM-4.7 at 24k tokens needs 4.8 GB for KV where
    # Qwen3-235B at the same length needs 2.5 GB, and _HEADROOM_BYTES has to
    # cover activations and Metal scratch out of the same 6 GB.
    kv_reserve = kvmem.kv_reserve_bytes(cfg, config.RESERVE_CTX)
    headroom = _HEADROOM_BYTES if kv_reserve <= 0 else _ACTIVATION_BYTES + kv_reserve

    if mode == "auto":
        fits = total_bytes + headroom <= budget
        if fits:
            mode = "resident"
        elif glus:
            mode = "streamed"
        else:
            # A dense checkpoint over budget. "dense" pages its layer weights;
            # without that module the resident branch raises the size error.
            mode = "dense" if _dense_enabled() else "resident"

    info = {
        "model_path": str(model_path),
        "mode": mode,
        "total_gb": round(total_bytes / 1e9, 2),
        "backbone_gb": round(backbone_bytes / 1e9, 2),
        "expert_gb": round(expert_bytes / 1e9, 2),
        "moe_layers": len(glus),
        "ram_budget_gb": round(budget / 1e9, 2),
    }
    if kv_reserve > 0:
        info["kv_reserve_gb"] = round(kv_reserve / 1e9, 2)
        info["reserve_ctx"] = config.RESERVE_CTX

    cache = None
    ring = None
    dense_plan = None
    dense_extra = None
    if mode == "dense":
        if not _dense_enabled():
            raise ValueError(
                "mode='dense' needs EXPERT_STREAM_DENSE=1 and the expert_stream"
                ".dense package (development build only)"
            )
        from . import dense

        dense_plan = dense.plan(model, cfg, tensor_index, total_bytes, ram)
        info.update(dense_plan.info)
        cache = dense.attach(dense_plan, verbose=verbose)
        _neutralize_wired_limit()
        cache_bytes = dense_plan.store_bytes
        pool_bytes = 2 << 30
        dense_extra = dense_plan.reserve_bytes + dense_plan.activation_bytes
    elif mode == "streamed":
        if not glus:
            hint = (
                "use mode='dense' to page its layer weights"
                if _dense_enabled()
                else f"a dense model this size ({info['total_gb']} GB) cannot "
                "run on this machine"
            )
            raise ValueError(
                f"{model_path} has no streamable MoE layers; {hint}"
            )
        if cache_gb is not None:
            requested = cache_bytes = int(cache_gb * 1e9)
            # An explicit CACHE_GB is a deliberate over-commitment: it buys
            # residency now and accepts that the slab gets shrunk when the KV
            # cache grows into it (ExpertCache._slots_that_fit on the next
            # rebuild, ExpertCache.make_room_for before a big dequantize). That
            # is a real strategy on a model whose KV is cheap - Qwen3-235B at 2k
            # context has 4 GB spare that auto mode would leave idle - so honor
            # it, and only clamp to what the backbone and decode activations
            # physically leave. Auto mode is the safe, KV-aware default.
            hard_cap = budget - backbone_bytes - _ACTIVATION_BYTES
            if hard_cap > 0:
                cache_bytes = min(cache_bytes, hard_cap)
            kv_aware_cap = budget - backbone_bytes - headroom
            if cache_bytes > kv_aware_cap > 0:
                info["overcommitted_gb"] = round((cache_bytes - kv_aware_cap) / 1e9, 2)
            # Only complain when the *cap* took it below a workable floor. A
            # deliberately tiny cache is a legitimate request (the test suite
            # asks for 1 GB to force misses), and rejecting it conflates "you
            # asked for too little" with "this machine has no room".
            if cache_bytes < (1 << 30) and cache_bytes < requested:
                raise ValueError(
                    f"expert cache budget would be {cache_bytes / 1e9:.1f} GB - "
                    f"backbone ({info['backbone_gb']} GB) leaves no room. "
                    "Lower EXPERT_STREAM_CACHE_GB or raise "
                    "EXPERT_STREAM_RAM_FRACTION."
                )
        else:
            cache_bytes = budget - backbone_bytes - headroom
            # Hard ceiling - auto mode must never grab 28 GB on a 48 GB box.
            max_cache = int(config.MAX_CACHE_GB * 1e9)
            if cache_bytes > max_cache:
                cache_bytes = max_cache
            if cache_bytes < (1 << 30):
                raise ValueError(
                    f"expert cache budget would be {cache_bytes / 1e9:.1f} GB - "
                    f"backbone ({info['backbone_gb']} GB) leaves no room. "
                    "Lower quantization bits or raise EXPERT_STREAM_RAM_FRACTION."
                )
        cache = ExpertCache(
            cache_bytes,
            read_threads=read_threads,
            nocache=config.NOCACHE,
            clear_bytes=(
                config.CLEAR_BYTES
                if config.CLEAR_BYTES is not None
                else config.clear_bytes_auto(cache_bytes)
            ),
        )
        # Route prediction is left to the governor even when the expert mass
        # dwarfs the cache. This used to force it on there, reasoning that a
        # disk-bound model can only gain from prefetching. Measured on GLM-4.7
        # (189 GB of experts against a 14.5 GB cache, so squarely in the case
        # the shortcut was written for) prediction *loses*, at both prompt
        # lengths tried and with or without lookup decode:
        #
        #   162-tok prompt   5.97 -> 7.30 tok/s with prediction off
        #   972-tok prompt   5.55 -> 7.48 tok/s with prediction off
        #
        # Predictor precision is only 0.31-0.43, so it spends 0.36-0.75 GB per
        # token on reads nothing asks for - and once the drive is saturated
        # (notes.md: tok/s x GB/token pins at ~5.8 GB/s) a wasted prefetch is
        # bandwidth taken directly from a read something is waiting on. Being
        # disk-bound is the reason prediction cannot pay here, not the reason it
        # must. The governor measures instead of assuming, so let it.
        predict_mode = None
        ring = patch_model(
            model,
            cache,
            tensor_index,
            prefetch_depth,
            predict_mode=predict_mode,
            model_path=str(model_path),
        )
        _neutralize_wired_limit()
        # Now that every layer is registered we know the expert size; scale
        # the buffers whose right size is "N experts", not "N bytes".
        per_expert = max(cache._expert_nbytes.values())
        if config.STAGING_BYTES is None:
            cache.staging_bytes = config.staging_bytes_auto(per_expert)
        pool_bytes = max(2 << 30, min(6 << 30, 384 * per_expert))
        info["cache_gb"] = round(cache_bytes / 1e9, 2)
        info["read_threads"] = read_threads
        info["prefetch_depth"] = prefetch_depth
        info["nocache"] = config.NOCACHE
        info["staging_gb"] = round(cache.staging_bytes / 1e9, 2)
        info["clear_gb"] = round(cache.clear_bytes / 1e9, 2)
        info["metal_pool_gb"] = round(pool_bytes / 1e9, 2)
        info["predict"] = predict_mode or config.PREDICT
        if config.PRUNE > 0:
            info["prune"] = config.PRUNE
        if config.WAIT_ABOVE > 0:
            info["wait_above"] = config.WAIT_ABOVE
        if config.ROUTE_TOP_K > 0:
            info["route_top_k"] = config.ROUTE_TOP_K
        if config.ROUTE_TOP_P > 0:
            info["route_top_p"] = config.ROUTE_TOP_P
        if config.ROUTE_TOP_K > 0 or config.ROUTE_TOP_P > 0 or config.PRUNE > 0:
            info["renorm"] = config.ROUTE_RENORM
        if config.KV_FP16_CTX != 8192:  # only note when it differs from default
            info["kv_fp16_ctx"] = config.KV_FP16_CTX
        if config.SIDECAR and ring is not None and ring.sidecar is not None:
            info["sidecar"] = ring.sidecar.stats()
    else:
        if total_bytes + _HEADROOM_BYTES > budget:
            raise ValueError(
                f"model is {info['total_gb']} GB but the RAM budget is "
                f"{info['ram_budget_gb']} GB; use mode='streamed'"
            )
        cache_bytes = 0
        pool_bytes = 2 << 30

    if mode == "dense":
        # Everything the dense engine actually holds: embeddings and norms, the
        # tensor groups left resident as ordinary modules, and the page stores.
        active_bytes = (
            dense_plan.other_bytes + dense_plan.native_bytes + dense_plan.store_bytes
        )
    else:
        active_bytes = backbone_bytes + (
            cache_bytes if mode == "streamed" else total_bytes
        )
    _set_memory_limits(active_bytes, pool_bytes, extra_bytes=dense_extra)

    # Materialize whatever is left in the tree: everything (resident) or
    # just the backbone (streamed - experts were dropped un-read above).
    mx.eval(model.parameters())

    if mode == "streamed" and config.SLAB != "off":
        # After the backbone is resident and the memory limit covers the slab.
        # "auto" == on wherever the architecture allows: measured +11% on
        # Qwen3-235B (disk-bound, so the win is partly hidden behind the SSD)
        # and +20% on qwen3-next, which is *less* disk-bound and therefore had
        # more per-expert dispatch overhead to give back. enable_slabs()
        # declines on anything it cannot address, so this needs no size test.
        if cache.enable_slabs(config.SLAB_BYTES, ram_budget_bytes=budget):
            info["slab_gb"] = round(cache.slab.nbytes / 1e9, 2)
            info["slab_slots"] = cache.slab.slots
            info["cache_gb"] = round(cache.budget_bytes / 1e9, 2)
            info["staging_gb"] = round(cache.staging_bytes / 1e9, 2)
            # Flow decode needs the slab (slot-addressed experts) plus the
            # expert count, which is only known once layers have registered.
            if config.FLOW and cache.enable_flow(max_expert_count(cache)):
                info["flow"] = True
                info["flow_topm"] = config.FLOW_TOPM or "router top-k"

    if mode == "streamed" and config.LOOKUP:
        info["lookup"] = True
        info["lookup_tokens"] = int(config.LOOKUP_TOKENS)
        # Opt-in, and both caveats are easy to forget: it needs a high accept
        # rate to beat plain decode, and it is only output-identical at PRUNE=0.
        if config.PRUNE:
            print(
                "[expert-stream] lookup decode is on with "
                f"PRUNE={config.PRUNE}: a k-token verify pass prunes on "
                "different mass than single-token steps, so output will not "
                "match a PRUNE-only run",
                flush=True,
            )

    if mode in ("streamed", "dense"):
        # What the prompt-cache store may hold - computed here because it is
        # whatever the slab did *not* take, and the slab's real size is only
        # known once it is allocated. mlx-lm never applies --prompt-cache-bytes
        # on the sequential path streamed models are pinned to, so this is the
        # only bound there is; without it the store grows until Metal fails.
        if mode == "dense":
            store_budget = (
                budget
                - dense_plan.other_bytes
                - dense_plan.native_bytes
                - dense_plan.store_bytes
                - dense_plan.activation_bytes
            )
        else:
            committed = (
                cache.slab.nbytes if cache.slab is not None else cache.budget_bytes
            )
            store_budget = budget - backbone_bytes - committed - _ACTIVATION_BYTES
        # Never below one conversation at the served context: evicting the prefix
        # the next turn resumes from trades a Metal OOM for a full re-prefill,
        # which on GLM-4.7 is 95 s. An over-committed slab can make the
        # subtraction negative, and 0 would read as "no bound at all" - the
        # unbounded store is the bug this exists to fix.
        store_budget = max(store_budget, kv_reserve, 2 << 30)
        kvmem.set_store_budget(
            store_budget, kvmem.kv_bytes_per_token(cfg, bits=config.KV_BITS)
        )
        info["prompt_cache_gb"] = round(kvmem.store_budget() / 1e9, 2)

    if adapter_path is not None:
        from mlx_lm.utils import load_adapters

        model = load_adapters(model, adapter_path)
        model.eval()

    # Match stock mlx-lm sharded_load: custom-code repos (Kimi tiktoken, etc.)
    # need this. Server passes trust_remote_code=None unless --trust-remote-code
    # is set; treat None as True so local checkpoints with tokenization_*.py load.
    tok_cfg = dict(tokenizer_config or {})
    if tok_cfg.get("trust_remote_code") is None:
        tok_cfg["trust_remote_code"] = True
    tokenizer = load_tokenizer(
        model_path,
        tok_cfg,
        eos_token_ids=cfg.get("eos_token_id", None),
    )

    info["load_s"] = round(time.perf_counter() - t0, 2)

    # Decouple the attention chunk from the expert-streaming chunk, so a long
    # prompt reads the expert mass once instead of once per attention chunk.
    try:
        from . import prefill_fused as _prefill_fused

        info["fused_prefill"] = bool(_prefill_fused.install(model))
    except Exception as e:
        info["fused_prefill"] = f"skip:{e!r}"

    # Stash handles for stats/introspection (underscore = not a model param).
    model._expert_stream_cache = cache
    model._expert_stream_ring = ring
    model._expert_stream_info = info

    if verbose:
        import json

        # flush=True: the host app redirects our stdout into a file, and Python block-
        # buffers a non-TTY stdout. Without this the startup line sits in the
        # buffer for the whole lifetime of the server and never appears in
        # the host's log file.
        print(f"[paged-moe] {json.dumps(info)}", flush=True)

    if return_config:
        return model, tokenizer, cfg
    return model, tokenizer


def relieve_pressure(model) -> None:
    """Call after each generation so request #2 doesn't freeze the Mac.

    Keeps the expert LRU warm (so the next turn is still fast) but:
      - drops speculative numpy staging
      - returns Metal recycled pages via mx.clear_cache()
    """
    cache = getattr(model, "_expert_stream_cache", None)
    if cache is not None:
        # A request aborted mid-prefill would otherwise leave the cache on its
        # shrunk prefill budget for the next one.
        cache.leave_prefill()
        cache.relieve_pressure()
    ring = getattr(model, "_expert_stream_ring", None)
    if ring is not None and getattr(ring, "sidecar", None) is not None:
        # Persist warm/locked state between turns so a restart isn't cold.
        try:
            ring.sidecar._persist()
        except Exception:
            pass
    if cache is None:
        mx.clear_cache()


def get_stats(model) -> dict:
    """Runtime stats: streaming cache + Metal memory."""
    out = dict(getattr(model, "_expert_stream_info", {}))
    cache = getattr(model, "_expert_stream_cache", None)
    if cache is not None:
        out["cache"] = cache.stats()
    ring = getattr(model, "_expert_stream_ring", None)
    if ring is not None and getattr(ring, "sidecar", None) is not None:
        out["sidecar"] = ring.sidecar.stats()
    out["mx_active_gb"] = round(mx.get_active_memory() / 1e9, 3)
    out["mx_cache_gb"] = round(mx.get_cache_memory() / 1e9, 3)
    out["mx_peak_gb"] = round(mx.get_peak_memory() / 1e9, 3)
    return out
