# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
load(): the public entry point.

    from pagedmoe import load
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
   PAGED_MOE_MODE env var.
4. In streamed mode, swap every SwitchGLU for a StreamedSwitchGLU *before*
   evaluating weights, so expert tensors are never read during load.

The returned (model, tokenizer) behave exactly like mlx-lm's, so everything
in the mlx-lm ecosystem (stream_generate, prompt caches, the HTTP server,
chat templates) works unchanged.
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model, load_tokenizer

from . import config, kvmem, pagecache
from .chat_compat import fix_think_tokens, install_deepseek_v32_chat_template

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
from .cache import ExpertCache, note_peak
from .safetensors_index import read_headers
from .streaming import (
    expert_tensor_names,
    find_switch_glus,
    install_closed_stitch,
    max_expert_count,
    patch_model,
)
from .ple_offload import patch_ple_ngram, skipped_resident_tensor_names

# Reserved for KV cache, activations, and Metal scratch (on top of what the
# RAM_FRACTION budget already leaves for macOS + apps). Expert reads bypass
# the page cache (F_NOCACHE) so no extra allowance is needed for it.
#
# Used when we have no context length to size the KV cache from
# (PAGED_MOE_RESERVE_CTX unset). With one, the reserve is computed per model
# instead and this flat figure splits into activations-plus-scratch alone.
# Values come from config so the machine provider can scale them; 48 GB stays
# at 6 GB / 3 GB.


def _headroom_bytes() -> int:
    return int(float(config.HEADROOM_GB) * (1 << 30))


def _drop_mtp(head_bytes: int, cache_bytes: int):
    print(
        f"[mtp] head ({head_bytes / 1e9:.2f} GB) is more than "
        f"{config.MTP_MAX_FRAC:.0%} of the {cache_bytes / 1e9:.1f} GB expert cache, "
        "loading without it",
        flush=True,
    )
    mx.clear_cache()
    return None, 0


def _activation_bytes() -> int:
    return int(float(config.ACTIVATION_GB) * (1 << 30))


def held_aware_budget(
    budget: int,
    *,
    gpu_held: int,
    held_excess: int,
    backbone_bytes: int,
    headroom: int,
    ram: int,
    name: str = "model",
) -> int:
    """Shrink the RAM budget to fit alongside memory already held outside this
    engine (a crashed model's orphaned wired pages, or another GPU app).

    The result keeps backbone + KV/activation headroom + expert cache + the
    Metal recycled-buffer pool under both the RAM budget (minus the held
    memory) and what Metal keeps resident (its working set, minus the held
    memory). The working-set ceiling is the one a flat floor ignores, and
    ignoring it is what lets a load commit past the resident limit: a command
    buffer then OOMs on Metal's async completion thread and the process aborts,
    orphaning this engine's wired pages as a fresh ghost.

    Raises EnvelopeOOM when even a 1 GB cache will not fit, so the engine
    refuses cleanly instead of loading into that crash.
    """
    from . import mempressure
    from .machine.oom import EnvelopeOOM

    floor = backbone_bytes + headroom + (1 << 30)
    working_set = mempressure.gpu_working_set_bytes() or ram
    scratch = int(config.STAGING_CAP_GB * (1 << 30))
    ceiling = min(working_set, ram) - (2 << 30)
    new_budget = min(budget - held_excess, ceiling - scratch - gpu_held)
    if new_budget < floor:
        msg = (
            f"{name} cannot load safely: {gpu_held / 1e9:.1f} GB of GPU memory is "
            f"already held outside this engine. A crashed model can orphan that much "
            f"wired memory until the GPU session is torn down, which leaves "
            f"{max(0, new_budget) / 1e9:.1f} GB against a {floor / 1e9:.1f} GB "
            f"minimum. Loading anyway would OOM and strand more memory. Log out or "
            f"restart the Mac to reclaim it, then try again."
        )
        print(f"[paged-moe] {msg}", flush=True)
        raise EnvelopeOOM(msg)
    return new_budget


def _kv_model_config(cfg: dict) -> dict:
    """Config slice used for KV sizing.

    Multimodal wrappers (qwen4_exp) put layer/head fields under text_config /
    a nested `text` object while the top-level num_hidden_layers is None.
    Without this fallback RESERVE_CTX silently reserves 0 bytes.
    """
    if int(cfg.get("num_hidden_layers") or 0) > 0:
        return cfg
    for key in ("text_config", "text", "llm_config", "language_config"):
        nested = cfg.get(key)
        if isinstance(nested, dict) and int(nested.get("num_hidden_layers") or 0) > 0:
            return nested
    return cfg


def _dense_enabled() -> bool:
    """Whether this build can stream a *dense* checkpoint's layers from disk.

    Two gates, both required. The env var is the switch; the spec lookup is
    because the dense package is a development-only part of the tree, so a
    build without it must degrade to the ordinary "does not fit" error rather
    than an ImportError.
    """
    import importlib.util
    import os

    if os.environ.get("PAGED_MOE_DENSE", "").strip().lower() not in (
        "1",
        "on",
        "true",
        "yes",
    ):
        return False
    try:
        return importlib.util.find_spec("pagedmoe.dense") is not None
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

    active_budget_bytes is about backbone + expert cache. We allow a little overhead
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
    extra = _headroom_bytes() if extra_bytes is None else int(extra_bytes)
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
    # Size capacity knobs for this Mac. No-op on the measured 48 GB class so
    # config.py defaults stay as measured. Explicit PAGED_MOE_* env wins.
    try:
        from .machine import apply_machine_defaults

        apply_machine_defaults()
    except Exception:
        pass

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
    gpu_held = None
    if config.GPU_HELD_CHECK:
        from . import mempressure

        gpu_held = mempressure.gpu_in_use_bytes()

    # --- build the module tree with lazy (unmaterialized) weights ----------
    from .dsa_shared import get_model_classes
    from .local_arch import register as register_local_arch

    # Match stock mlx-lm sharded_load: custom-code repos (Kimi tiktoken, etc.)
    # need this. Server passes trust_remote_code=None unless --trust-remote-code
    # is set; treat None as True so local checkpoints with tokenization_*.py load.
    tok_cfg = dict(tokenizer_config or {})
    if tok_cfg.get("trust_remote_code") is None:
        tok_cfg["trust_remote_code"] = True

    register_local_arch(model_path)
    model_kwargs = {}
    # mlx-lm 0.32 refuses config model_file (qwen4_exp.py etc.) without this.
    if "trust_remote_code" in inspect.signature(load_model).parameters:
        model_kwargs["trust_remote_code"] = tok_cfg["trust_remote_code"]
    model, cfg = load_model(
        model_path,
        lazy=True,
        model_config=model_config,
        get_model_classes=get_model_classes,
        **model_kwargs,
    )
    # DeepSeek-V2 caches expanded heads unless this is installed first.
    from .mla_v2 import install as install_mla_v2

    install_mla_v2()

    glus = find_switch_glus(model)
    expert_bytes = 0
    if glus:
        expert_bytes = sum(
            tensor_index[name].nbytes
            for name in expert_tensor_names(model, tensor_index)
        )
    # Vision / MTP are not loaded for text; PLE n-gram shards are paged from
    # disk (see ple_offload). Both must leave the resident-backbone figure or
    # Flash-class checkpoints report a negative expert cache on a 48 GB Mac.
    skip_names = skipped_resident_tensor_names(tensor_index)
    skip_bytes = sum(tensor_index[n].nbytes for n in skip_names)
    backbone_bytes = total_bytes - expert_bytes - skip_bytes

    # What the KV cache will want at the context this model is served with.
    # This comes straight out of the expert cache's budget, so guessing it with
    # a flat allowance is how a full-attention model ends up committing a slab
    # it cannot keep: GLM-4.7 at 24k tokens needs 4.8 GB for KV where
    # Qwen3-235B at the same length needs 2.5 GB, and headroom has to
    # cover activations and Metal scratch out of the same reserve.
    kv_cfg = _kv_model_config(cfg)
    kv_reserve = kvmem.kv_reserve_bytes(kv_cfg, config.RESERVE_CTX)
    headroom = _headroom_bytes() if kv_reserve <= 0 else _activation_bytes() + kv_reserve

    if mode == "auto":
        fits = total_bytes + headroom <= budget
        if fits:
            mode = "resident"
        elif glus:
            mode = "streamed"
        else:
            # Dense layer streaming is shelved and not in this repo, so this
            # falls through to the ordinary "does not fit" error.
            mode = "dense" if _dense_enabled() else "resident"

    # Streamed only: a smaller budget there costs cache slots, while resident
    # and dense loads would turn it into a refusal to start.
    held_excess = 0
    if gpu_held and mode == "streamed":
        from . import mempressure

        held_excess = mempressure.gpu_held_excess(gpu_held, ram)
    if held_excess:
        # Plan around memory that is really there. A smaller cache here only
        # costs slots (slower decode), so prefer that to a refusal; refuse only
        # when even a minimal load would push past the resident limit and crash.
        was = budget
        budget = held_aware_budget(
            budget,
            gpu_held=gpu_held,
            held_excess=held_excess,
            backbone_bytes=backbone_bytes,
            headroom=headroom,
            ram=ram,
            name=model_path.name,
        )
        print(
            f"[paged-moe] GPU memory: macOS reports {gpu_held / 1e9:.1f} GB already in use "
            f"outside this engine (another GPU app, or memory kept after a crashed process). "
            f"Budget {was / 1e9:.1f} -> {budget / 1e9:.1f} GB. Quitting GPU-heavy apps or "
            f"restarting the Mac returns it.",
            flush=True,
        )

    info = {
        "model_path": str(model_path),
        "mode": mode,
        "total_gb": round(total_bytes / 1e9, 2),
        "backbone_gb": round(backbone_bytes / 1e9, 2),
        "expert_gb": round(expert_bytes / 1e9, 2),
        "moe_layers": len(glus),
        "ram_budget_gb": round(budget / 1e9, 2),
        "ram_gb": round(ram / (1 << 30), 2),
        "max_cache_gb": float(config.MAX_CACHE_GB),
        "headroom_gb": round(headroom / 1e9, 2),
    }
    if held_excess:
        info["gpu_held_gb"] = round(gpu_held / 1e9, 2)
    if skip_bytes:
        info["paged_other_gb"] = round(skip_bytes / 1e9, 2)
    if kv_reserve > 0:
        info["kv_reserve_gb"] = round(kv_reserve / 1e9, 2)
        info["reserve_ctx"] = config.RESERVE_CTX

    cache = None
    ring = None
    dense_plan = None
    dense_extra = None
    mtp_prep = None
    head_bytes = 0
    if ((mode == "streamed" and glus) or mode == "resident") and config.MTP:
        from . import mtp as _mtp

        mtp_prep = _mtp.prepare(model, cfg, model_path)
        head_bytes = mtp_prep.nbytes if mtp_prep is not None else 0
    if head_bytes and mode == "resident" and total_bytes + head_bytes + _headroom_bytes() > budget:
        # Resident has no expert cache to give the head room, so it only rides
        # along when the whole model plus the head still clears the budget.
        print(
            f"[mtp] head ({head_bytes / 1e9:.2f} GB) does not fit next to the "
            f"{info['total_gb']} GB resident model on a {info['ram_budget_gb']} GB "
            "budget, loading without it",
            flush=True,
        )
        mtp_prep, head_bytes = None, 0
        mx.clear_cache()
    if mode == "dense":
        if not _dense_enabled():
            raise ValueError(
                "mode='dense' needs PAGED_MOE_DENSE=1 and the pagedmoe"
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
            if head_bytes and head_bytes > config.MTP_MAX_FRAC * cache_bytes:
                mtp_prep, head_bytes = _drop_mtp(head_bytes, cache_bytes)
            hard_cap = budget - backbone_bytes - _activation_bytes() - head_bytes
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
                    "Lower PAGED_MOE_CACHE_GB or raise "
                    "PAGED_MOE_RAM_FRACTION."
                )
        else:
            cache_bytes = budget - backbone_bytes - headroom
            # Hard ceiling - auto mode must never grab 28 GB on a 48 GB box.
            max_cache = int(config.MAX_CACHE_GB * 1e9)
            if cache_bytes > max_cache:
                cache_bytes = max_cache
            if head_bytes:
                # The head comes out of whatever MAX_CACHE_GB left unused first.
                room = budget - backbone_bytes - headroom
                if head_bytes > config.MTP_MAX_FRAC * cache_bytes:
                    mtp_prep, head_bytes = _drop_mtp(head_bytes, cache_bytes)
                else:
                    cache_bytes = min(cache_bytes, room - head_bytes)
            # If every routed expert fits in the RAM budget, hold them. The
            # usual MAX_CACHE_GB cap is for a working set, not a reason to
            # leave a 40 GB expert mass on the per-layer path on a 128 GB Mac.
            # A checkpoint that does not fit (235B on 48 GB) stays capped.
            # Only the cap is lifted: the KV and activation reserve stays.
            if config.STITCH and expert_bytes > cache_bytes:
                # One extra expert of padding so integer slot division does not
                # come up one short of the set.
                need = expert_bytes + (16 << 20)
                room = budget - backbone_bytes - headroom - head_bytes
                if need <= room:
                    info["stitch_grow_gb"] = round((need - cache_bytes) / 1e9, 2)
                    cache_bytes = need
            if cache_bytes < (1 << 30):
                msg = (
                    f"expert cache budget would be {cache_bytes / 1e9:.1f} GB - "
                    f"backbone ({info['backbone_gb']} GB) leaves no room on a "
                    f"{info['ram_budget_gb']} GB envelope. "
                    "This would OOM on the target Mac."
                )
                from .machine.oom import EnvelopeOOM, envelope_strict

                if envelope_strict():
                    raise EnvelopeOOM(msg)
                raise ValueError(
                    msg + " Lower quantization bits or raise "
                    "PAGED_MOE_RAM_FRACTION."
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
        # (tok/s x GB/token pins at ~5.8 GB/s) a wasted prefetch is
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
        from .fast_norm import install as install_fast_norm

        install_fast_norm(model)
        ple_pool = patch_ple_ngram(model, tensor_index)
        if ple_pool is not None:
            info["ple_offload"] = True
        info["pagecache_trim"] = pagecache.watch(
            sorted({loc.file for loc in tensor_index.values()})
        )
        _neutralize_wired_limit()
        # Now that every layer is registered we know the expert size; scale
        # the buffers whose right size is "N experts", not "N bytes".
        per_expert = max(cache._expert_nbytes.values())
        if getattr(ring, "early_warm", None) is not None:
            ring.early_warm.set_expert_bytes(per_expert)
        if getattr(ring, "tool_warm", None) is not None:
            ring.tool_warm.set_expert_bytes(per_expert)
        if config.STAGING_BYTES is None:
            cache.staging_bytes = config.staging_bytes_auto(per_expert)
        pool_bytes = max(1 << 30, min(int(config.STAGING_CAP_GB * (1 << 30)), 384 * per_expert))
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
        # Explicit resident (PAGED_MOE_MODE=resident / runsNative): load like
        # plain mlx-lm when the weights alone fit. The flat HEADROOM_GB reserve
        # exists for streamed expert-cache churn; requiring it on top of a full
        # resident load rejects models that mlx-openai-server would run fine.
        need = total_bytes + _headroom_bytes()
        if need > budget:
            if total_bytes <= budget:
                if verbose:
                    print(
                        f"[load] resident: model {info['total_gb']} GB fits the "
                        f"{info['ram_budget_gb']} GB budget with tight headroom "
                        f"({config.HEADROOM_GB} GB); loading anyway (explicit resident)",
                        flush=True,
                    )
            else:
                raise ValueError(
                    f"model is {info['total_gb']} GB but the RAM budget is "
                    f"{info['ram_budget_gb']} GB; use mode='streamed'"
                )
        cache_bytes = 0
        pool_bytes = 2 << 30
        # Leave Metal alone for resident: plain mlx-lm raises wired_limit and
        # does not put a 2 GB cache_limit on the process. Neutralizing those
        # (correct for streamed expert churn) is what made a 30 GB model that
        # runs fine under mlx-openai-server OOM on the first prefill here.

    if mode == "dense":
        # Everything the dense engine actually holds: embeddings and norms, the
        # tensor groups left resident as ordinary modules, and the page stores.
        active_bytes = (
            dense_plan.other_bytes + dense_plan.native_bytes + dense_plan.store_bytes
        )
        _set_memory_limits(active_bytes, pool_bytes, extra_bytes=dense_extra)
    elif mode == "streamed":
        active_bytes = backbone_bytes + cache_bytes + head_bytes
        _set_memory_limits(active_bytes, pool_bytes, extra_bytes=None)
    else:
        # Resident: no paged-moe Metal caps. mlx-lm owns the working set.
        pass

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
            if getattr(ring, "early_warm", None) is not None:
                ring.early_warm.set_expert_bytes(int(cache.slab.expert_bytes))
            if getattr(ring, "tool_warm", None) is not None:
                ring.tool_warm.set_expert_bytes(int(cache.slab.expert_bytes))
            # Flow decode needs the slab (slot-addressed experts) plus the
            # expert count, which is only known once layers have registered.
            if cache.load_stitch():
                info["stitch"] = True
                info["stitch_experts"] = len(cache._stitch_keys)
                info["cache_gb"] = round(cache.budget_bytes / 1e9, 2)
            if (config.FLOW or config.DRAFT_FLOW) and cache.enable_flow(max_expert_count(cache)):
                if config.FLOW:
                    info["flow"] = True
                    info["flow_topm"] = config.FLOW_TOPM or "router top-k"
                if config.DRAFT_FLOW:
                    info["draft_flow"] = True
            # Exact decode on a model the slab holds little of is disk-bound
            # by construction, and there prediction measured +15-20% every
            # time (235B: 2.5 vs 3.1 tok/s). The auto governor would spend
            # the first ~200 tokens after every load finding that out, half
            # of them in the losing state. Start it on; it still re-probes.
            frac = cache.slab.nbytes / max(1, expert_bytes)
            if (
                config.PREDICT_PRIOR_RESIDENT > 0
                and float(config.PRUNE) == 0.0
                and frac < config.PREDICT_PRIOR_RESIDENT
                and ring.prior(
                    True,
                    f"slab holds {frac:.0%} of the experts at PRUNE 0, "
                    "starting on (re-probes later)",
                    config.PREDICT_HOLD_WINDOWS // 2,
                )
            ):
                info["predict"] = "auto:on"

    if mtp_prep is not None:
        try:
            _mtp.attach(model, mtp_prep)
            info["mtp"] = {"gb": round(head_bytes / 1e9, 2), "from": mtp_prep.source}
        except Exception as e:
            print(f"[mtp] head not attached, decoding without it: {e!r}", flush=True)
            info["mtp"] = f"skip:{e!r}"

    if mode == "streamed" and config.LOOKUP:
        info["lookup"] = True
        info["lookup_tokens"] = int(config.LOOKUP_TOKENS)
        # Opt-in, and both caveats are easy to forget: it needs a high accept
        # rate to beat plain decode, and it is only output-identical at PRUNE=0.
        if config.PRUNE:
            print(
                "[paged-moe] lookup decode is on with "
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
            store_budget = budget - backbone_bytes - committed - head_bytes - _activation_bytes()
        # Never below one conversation at the served context: evicting the prefix
        # the next turn resumes from trades a Metal OOM for a full re-prefill,
        # which on GLM-4.7 is 95 s. An over-committed slab can make the
        # subtraction negative, and 0 would read as "no bound at all" - the
        # unbounded store is the bug this exists to fix.
        store_budget = max(store_budget, kv_reserve, 2 << 30)
        kvmem.set_store_budget(
            store_budget, kvmem.kv_bytes_per_token(kv_cfg, bits=config.KV_BITS)
        )
        kvmem.set_kv_reserve(kv_cfg, config.RESERVE_CTX)
        info["prompt_cache_gb"] = round(kvmem.store_budget() / 1e9, 2)
    elif mode == "resident":
        # Same missing bound, and no slab to give memory back. What the weights
        # leave under Metal's working set is split between the live
        # conversation and the store.
        from . import mempressure

        ceiling = min(budget, mempressure.gpu_working_set_bytes() or budget)
        room = ceiling - total_bytes - _activation_bytes()
        kvmem.set_store_budget(
            max(room // 2, 1 << 30),
            kvmem.kv_bytes_per_token(kv_cfg, bits=config.KV_BITS),
        )
        info["prompt_cache_gb"] = round(kvmem.store_budget() / 1e9, 2)

    if adapter_path is not None:
        from mlx_lm.utils import load_adapters

        model = load_adapters(model, adapter_path)
        model.eval()

    tokenizer = load_tokenizer(
        model_path,
        tok_cfg,
        eos_token_ids=cfg.get("eos_token_id", None),
    )
    fix_think_tokens(tokenizer)

    info["load_s"] = round(time.perf_counter() - t0, 2)

    # Decouple the attention chunk from the paged-moeing chunk, so a long
    # prompt reads the expert mass once instead of once per attention chunk.
    try:
        from . import prefill_fused as _prefill_fused

        info["fused_prefill"] = bool(_prefill_fused.install(model))
    except Exception as e:
        info["fused_prefill"] = f"skip:{e!r}"

    # Router ids by position, for verify passes that copy context.
    if ring is not None and config.VERIFY and config.VERIFY_REPLAY:
        try:
            from . import route_log as _route_log

            log = _route_log.RouteLog([glu.layer_key for glu in ring.layers])
            _route_log.install(model, log)
            ring.routes = log
            info["route_log_tokens"] = log.cap
        except Exception as e:
            info["route_log"] = f"skip:{e!r}"

    # Stash handles for stats/introspection (underscore = not a model param).
    model._pagedmoe_cache = cache
    model._pagedmoe_ring = ring
    model._pagedmoe_info = info
    if mode == "streamed" and cache is not None and getattr(cache, "slab", None) is not None:
        if install_closed_stitch(model):
            info["stitch_closed"] = True

    # Dev-only activation debugger (no-op when PAGED_MOE_MITM=0 or the module is missing).
    try:
        from . import dev_mitm

        mitm = dev_mitm.maybe_install(model, ring)
        if mitm is not None:
            info["mitm"] = True
    except Exception as e:
        info["mitm"] = f"skip:{e!r}"

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
    from . import turn_trace

    t0 = time.perf_counter()
    cache = getattr(model, "_pagedmoe_cache", None)
    if cache is not None:
        # A request aborted mid-prefill would otherwise leave the cache on its
        # shrunk prefill budget for the next one.
        cache.leave_prefill()
        t1 = time.perf_counter()
        cache.relieve_pressure()
        turn_trace.note(relieve_leave_s=round(t1 - t0, 4), relieve_cache_s=round(time.perf_counter() - t1, 4))
    ring = getattr(model, "_pagedmoe_ring", None)
    if ring is not None and getattr(ring, "sidecar", None) is not None:
        # Persist warm/locked state between turns so a restart isn't cold.
        # Queued, not inline: the response and the next request wait on this.
        t2 = time.perf_counter()
        try:
            ring.sidecar.persist_soon()
        except Exception:
            pass
        turn_trace.note(relieve_persist_s=round(time.perf_counter() - t2, 4))
    if cache is None:
        mx.clear_cache()


def get_stats(model) -> dict:
    """Runtime stats: streaming cache + Metal memory."""
    out = dict(getattr(model, "_pagedmoe_info", {}))
    cache = getattr(model, "_pagedmoe_cache", None)
    if cache is not None:
        out["cache"] = cache.stats()
    ring = getattr(model, "_pagedmoe_ring", None)
    if ring is not None and getattr(ring, "sidecar", None) is not None:
        out["sidecar"] = ring.sidecar.stats()
    out["mx_active_gb"] = round(mx.get_active_memory() / 1e9, 3)
    out["mx_cache_gb"] = round(mx.get_cache_memory() / 1e9, 3)
    out["mx_peak_gb"] = round(note_peak() / 1e9, 3)
    return out
