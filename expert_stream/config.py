# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Configuration.

Everything has a sane default and can be overridden by environment variables
(and most things also by CLI flags / load() kwargs, which win over env vars).

Env vars:
  EXPERT_STREAM_MODELS_DIR      where downloaded models are stored.
                                Default: ~/models/paged-moe.
  EXPERT_STREAM_MODE            "auto" | "resident" | "streamed".  Default auto:
                                fully load models that fit in RAM, stream the rest.
  EXPERT_STREAM_RAM_FRACTION    fraction of physical RAM the whole engine may use
                                (backbone + expert cache + KV headroom).
                                Default 0.68. Safe now that expert reads bypass
                                the OS page cache (see NOCACHE) and the Metal
                                wired limit is never raised.
  EXPERT_STREAM_CACHE_GB        explicit expert-cache budget in GB. Overrides the
                                auto budget. 32 is worth trying on a 48 GB
                                machine if little else is running.
  EXPERT_STREAM_MAX_CACHE_GB    hard ceiling on the auto cache budget. Default 26.
                                This is the main lever on the delay between agent
                                turns; see the table at its definition below.
  EXPERT_STREAM_NOCACHE         1 (default) = read expert weights with F_NOCACHE,
                                bypassing the macOS page cache. This makes RAM
                                usage deterministic: the expert LRU is the only
                                place expert bytes live. 0 = let the kernel
                                page-cache the checkpoint (can balloon tens of
                                GB during prefill and freeze a 48 GB machine).
  EXPERT_STREAM_READ_THREADS    parallel disk readers. Default 16.
  EXPERT_STREAM_READ_POOL_THREADS
                                decode slot-read pool queue depth.
                                0 = auto (2x READ_THREADS), -1 = old executor
                                path. Default 0.
  EXPERT_STREAM_PREFETCH_DEPTH  speculative prefetch lookahead in layers. Default 8.
  EXPERT_STREAM_PREDICT         "auto" (default), "1" (always) or "0" (never):
                                during decode, run the *next* layers' routers on
                                the current hidden state and prefetch what they
                                are about to want. The routers live in the
                                resident backbone, so this costs one small matvec
                                per predicted layer and turns blocking expert
                                reads into cache hits. Prefetch can only change
                                timing, never values.
                                "auto" exists because the win is conditional:
                                hiding a read behind compute is only free if the
                                read is the bottleneck. On unified memory the
                                reader threads and the GPU share bandwidth, so
                                when experts already arrive at near-RAM speed, or
                                the drive is already saturated, overlapping costs
                                ~7% instead of saving 40%. Rather than model
                                that, "auto" alternates on/off in short windows,
                                pools tok/s each way until there is enough
                                evidence to call it, and keeps the winner.
  EXPERT_STREAM_PREDICT_WINDOW  decode tokens per governor window. Default 8.
                                Short windows interleave the two states so they
                                sample the same content; the evidence needed for
                                a decision is set separately, below.
  EXPERT_STREAM_PREDICT_MIN_TOKENS
                                tokens to pool per state before the governor
                                decides. Default 96 (~24 windows total). One
                                window per state is far below the noise floor of
                                agent traffic and made the decision flip nine
                                times in four minutes, running in the losing
                                configuration much of the time.
  EXPERT_STREAM_PREDICT_MARGIN  how much faster prediction must measure before
                                it is kept. Default 1.03. Necessary but not
                                sufficient: the gap must also clear the measured
                                error bar (see PREDICT_Z).
  EXPERT_STREAM_PREDICT_Z       error bars the gap must clear before the
                                governor acts on it. Default 2.0. A percentage
                                margin alone is not enough - agent decode rates
                                swing tens of percent window to window, so at
                                realistic sample sizes the error bar is wider
                                than the margin and ties resolve at random.
  EXPERT_STREAM_PREDICT_MAX_PROBE
                                multiples of PREDICT_MIN_TOKENS to keep probing
                                while the result stays inconclusive. Default 6;
                                after that the cheaper side (off) wins by
                                default and is held 4x longer, since a
                                difference still lost in the noise is not worth
                                re-measuring often.
  EXPERT_STREAM_PREDICT_HOLD    windows to keep the governor's decision before
                                re-measuring. Default 128 (~1024 tokens), which
                                spans several agent turns; re-probing keeps half
                                the previous evidence so a settled result is not
                                relitigated from scratch.
  EXPERT_STREAM_PREDICT_FAR_LEAD
                                add ONE prediction per layer at this distance,
                                for source layers past FAR_FROM of the stack.
                                Default 0 (off). The window above is uniform
                                over depth; bench/router_matrix.py shows that
                                is wrong - agreement decays with source depth,
                                not distance, so the early layers that justify
                                a short window are not the deep ones that could
                                run 16-32 ahead at 0.6-0.8 miss-precision. One
                                target (not a wider window) because a wide
                                burst queues ahead of real demand misses.
  EXPERT_STREAM_PREDICT_FAR_FROM
                                fraction of the stack before FAR_LEAD engages.
                                Default 0.17 (~n/6), where the measured
                                early-layer collapse ends.
  EXPERT_STREAM_PREDICT_DEPTH   how many layers ahead to route-predict.
                                Default 3. Prediction accuracy is nearly flat
                                with distance (87-89% at depth 1-8 measured),
                                so depth is really a bandwidth knob: each extra
                                layer issues more speculative reads. Measured
                                decode tok/s on OLMoE at a 41% cache ratio:
                                20.8 off, 24.5 (d1), 29.5 (d2), 33.5 (d3),
                                34.5 (d4), 31.1 (d6), 29.3 (d8) - past ~4 the
                                wasted reads cost more than the hits gain.
                                Raise it if the disk has spare bandwidth.
  EXPERT_STREAM_PREDICT_SLACK   extra experts beyond top-k to prefetch per
                                predicted layer. Default 2 (0 measured 2% slower
                                and 5 points less coverage; 8 wasted 11 GB to
                                gain nothing).
  EXPERT_STREAM_PREDICT_RENORM  1 = rescale the hidden state by w_target/w_source
                                (the two RMSNorm weights) before applying the
                                target router. Default 0: measured neutral on
                                coverage (0.8824 vs 0.8821), so it is not worth
                                the extra multiply. Kept for architectures whose
                                norm scales vary more sharply across layers.
  EXPERT_STREAM_LFRU            1 (default) = frequency-protected eviction. A
                                cold entry at the LRU tail is evicted; a
                                frequently used one gets a second chance (its
                                counter halves and it moves to MRU). Stops one
                                sweep of prefill/decode traffic from flushing
                                the experts every token needs.
  EXPERT_STREAM_LFRU_SCAN       max second chances granted per eviction. Default 8.
  EXPERT_STREAM_STAGING_GB      cap on prefetched-but-unused numpy staging.
                                Default: auto (max(1 GB, 256 experts' worth,
                                capped at 6 GB)). Must comfortably hold one
                                prediction horizon (PREDICT_DEPTH layers' worth
                                of experts), otherwise predictions are dropped
                                before the layer that wanted them runs and get
                                re-read as misses - a fixed 1 GB caused ~100
                                such drops per decoded token on Qwen3-235B
                                (10.6 MB experts).
  EXPERT_STREAM_CLEAR_GB        bytes evicted between mx.clear_cache() calls.
                                Default: auto (cache budget / 3, floor 2 GB).
                                Big-expert models evict multiple GB per decoded
                                token; clearing too often flushes Metal's
                                recycled-buffer pool and every miss then pays
                                fresh zero-fill page faults.
  EXPERT_STREAM_SLAB            "auto" (default, = on wherever the model's
                                experts are uniformly shaped) | "on" | "off".
                                Holds resident experts as slots in contiguous
                                Metal-backed tensors, so reads pread into their
                                final location (no host->device copy) and a
                                layer computes with one gather_qmm per
                                projection instead of 3 calls per expert.
                                Measured +11% on Qwen3-235B and +20% on
                                qwen3-next. See docs/DESIGN.md "Slab storage".
  EXPERT_STREAM_SLAB_GB         hard slab reservation. Unlike the expert cache's
                                soft budget this is committed at load, so it is
                                given back for big prefills and rebuilt for
                                decode. Default: the expert-cache budget.
  EXPERT_STREAM_SLAB_VERIFY     1 = check at compute time that no slot a layer
                                reads is being written and that slot ownership
                                agrees. Slot reuse overwrites bytes in place,
                                so a double-owned slot means wrong output
                                rather than an exception - use this first when
                                debugging slab correctness.
  EXPERT_STREAM_PRUNE           opt-in decode "turbo": drop a routed expert
                                whose router weight is below this fraction of
                                the strongest selected expert's weight (0 =
                                off, the default; output stays bit-identical
                                to resident inference). Measured on
                                Qwen3-235B-A22B 4-bit: 0.3 prunes ~23-29% of
                                slots (2.6 -> 2.9-3.0 tok/s), 0.5 prunes ~55%
                                (-> ~4 tok/s). Output stays coherent but is no
                                longer bit-identical. Only affects decode, and
                                only Qwen3-MoE-style plain-logits routers.
  EXPERT_STREAM_GROUP_MB        max bytes of expert weights materialized per
                                prefill group (bounds transient memory while
                                whole layers stream through). Default 384.
  EXPERT_STREAM_PREFILL_CACHE_GB
                                expert-cache budget while a big prefill chunk is
                                being processed. Default 4. A prefill pass reads
                                the whole uncached expert mass, and within one
                                pass a big cache buys nothing - so giving that
                                memory to activations instead (which is what
                                allows one-pass prefill) is a large net win.
  EXPERT_STREAM_PREFILL_CHUNK   prompt tokens per prefill pass, used as the
                                default for the server and CLI. Default 32768:
                                a 32k-token agent prompt then streams the expert
                                mass exactly once instead of four times.
  EXPERT_STREAM_RESERVE_CTX     context length (tokens) to hold back KV memory
                                for when sizing the expert cache. The host app sets this
                                from the model's numCtx. Full-attention models
                                need it: GLM-4.7 spends 196 KB/token at 8 bits,
                                so 24k tokens is 4.8 GB the expert cache must
                                not have committed to a slab. 0 = use the flat
                                headroom allowance (right only for cheap-KV
                                models like Qwen3-235B at 102 KB/token).
  EXPERT_STREAM_KV_BITS         decode-time KV cache precision. Default 8.
  EXPERT_STREAM_KV_STORE_SLACK  multiplier on reserved KV bytes covering the
                                prompt-cache store next to the live cache.
                                Default 1.25.
  EXPERT_STREAM_PROMPT_CACHE_MOVE
                                1 (default) = hand a reused prompt cache to the
                                request instead of deep-copying it, which is a
                                whole extra KV cache per turn (4.8 GB on GLM-4.7
                                at 24k). 0 restores mlx-lm's copying, which
                                survives a request that dies mid-generation.
  EXPERT_STREAM_SIDECAR         0 (default) = offline. 1 = online expert-prefetch
                                sidecar: a tiny per-model learner that guesses
                                which experts the *next* token's early layers
                                will need and prefetches them. Quality-safe
                                (prefetch only; real router still decides).
                                When off, none of the sidecar code runs on the
                                decode path.
  EXPERT_STREAM_SIDECAR_DEBUG   0 (default) = quiet. 1 = print [sidecar] status
                                lines (also enabled by PAGED_MOE_DEBUG=1).
  EXPERT_STREAM_SIDECAR_WRAP    early MoE layers to prefetch for the next token.
                                Default 0 = scale from the MoE layer count.
  EXPERT_STREAM_SIDECAR_TOPK    experts predicted per wrap layer. Default 0 =
                                track the router's observed fan-out per layer.
  EXPERT_STREAM_SIDECAR_TOPK_FACTOR
                                multiple of that fan-out to aim for when TOPK
                                is auto. Default 1.0.
  EXPERT_STREAM_SIDECAR_MIN_SCORE
                                softmax probability floor; drop guesses below
                                this even inside top-k. Default 0.04.
  EXPERT_STREAM_SIDECAR_HISTORY how many prior tokens feed the feature vector.
                                Default 6.
  EXPERT_STREAM_SIDECAR_FEAT_DIM
                                feature width. Default 0 = scale from expert
                                and layer counts.
  EXPERT_STREAM_SIDECAR_HIDDEN  1 (default) = project the last hidden state into
                                the features. It is what the LM head turns into
                                the next token, so it is the strongest single
                                predictor of that token's routing; history alone
                                can only model expert co-occurrence.
  EXPERT_STREAM_SIDECAR_SKETCH_DIM
                                width of that projection. Default 64.
  EXPERT_STREAM_SIDECAR_ADAGRAD 1 (default) = per-parameter adaptive step. Sparse
                                hashed features converge far faster with it.
  EXPERT_STREAM_SIDECAR_WD      weight decay on the shadow weights. Default 1e-5.
  EXPERT_STREAM_SIDECAR_PRIOR_W blend weight for the per-layer expert-popularity
                                prior while the head is cold, decaying to 0 over
                                PRIOR_DECAY train steps. Default 0.5 / 20000.
  EXPERT_STREAM_SIDECAR_MIN_GAIN
                                marginal recall a head must add on top of the
                                engine's own last-token replay to go (or stay)
                                live. Default 0.02. This, not absolute recall,
                                is what says the head is earning its reads.
  EXPERT_STREAM_SIDECAR_EXTRA_BUDGET
                                cap on extra experts issued per token.
                                Default 0 = wrap * topk.
  EXPERT_STREAM_SIDECAR_TIME_BUDGET_MS / _MAX_OVERHEAD
                                the sidecar runs inline between tokens, so its
                                own time is token latency. Past either bound it
                                sheds work (shadow scoring, training stride,
                                top-k, then actuation). Default 4.0ms / 0.05.
  EXPERT_STREAM_SIDECAR_SAVE_EVERY
                                persist every N decode tokens so a kill never
                                costs a whole session. Default 512.
  EXPERT_STREAM_SIDECAR_LOCK_HITS
                                consecutive good end_token checks before lock.
                                Default 0 = never auto-lock (early lock froze
                                mediocre adapters). Set e.g. 64 to opt in.
  EXPERT_STREAM_SIDECAR_LOCK_MIN_TOKENS
                                refuse lock until this many tokens seen.
                                Default 4096.
  EXPERT_STREAM_SIDECAR_LOCK_OVER_CEILING
                                live recall must beat same-layer ceiling by
                                this margin to lock. Default 0.15.
  EXPERT_STREAM_SIDECAR_MIN_RECALL
                                rolling recall@topk to promote / lock. Default
                                0.40. Only used before MIN_GAIN has samples.
  EXPERT_STREAM_SIDECAR_MIN_PRECISION
                                rolling precision@topk to promote / lock /
                                avoid SSD thrash. Default 0.22.
  EXPERT_STREAM_SIDECAR_DIR     where per-model adapter slots are persisted.
                                Default ~/.paged_moe/sidecar.
                                Override to keep weights under another root.
  EXPERT_STREAM_SIDECAR_HEAD_WRAP
                                1 (default when SIDECAR on) = next-token wrap
                                prefetch head.
  EXPERT_STREAM_SIDECAR_HEAD_PREFILL
                                0 (default) = off. 1 = prefill hot-expert
                                prefetch / post-leave warm (quality-safe).
  EXPERT_STREAM_SIDECAR_PREFILL_HOT_FRAC
                                a chunk's expert union is nearly every expert,
                                so the head targets its HOT set instead: experts
                                whose per-chunk token count reaches this
                                fraction of the layer's busiest. Default 0.5.
  EXPERT_STREAM_SIDECAR_HEAD_RESIDENCY
                                0 (default) = off. 1 = eviction protect scores
                                (quality-safe).
  EXPERT_STREAM_SIDECAR_RESIDENCY_MAX_PROTECT
                                cap on the share of scanned eviction candidates
                                that may be protected, so a uniformly hot cache
                                still makes progress. Default 0.10 - at 0.33 the
                                head stopped selecting and merely perturbed
                                LRU's victim order, measured +55% misses.
  EXPERT_STREAM_SIDECAR_RESIDENCY_MIN_HITS
                                separate demands on a key before its reuse rate
                                may override LRU. Default 8: 88% of just-used
                                experts are reused inside the horizon, so a rate
                                from one or two observations is noise.
  EXPERT_STREAM_SIDECAR_GOVERNOR
                                1 (default) = actuation is A/B'd against itself
                                on measured decode rate and only stays on while
                                it wins. This is what makes "the sidecar cannot
                                make decode slower" a property rather than a
                                hope; recall-style metrics cannot deliver it
                                because they have no cost term. 0 = actuate
                                unconditionally (A/B rigs only).
  EXPERT_STREAM_SIDECAR_GOV_WINDOW / _MIN_TOKENS
                                tokens per A/B window, and tokens per arm before
                                a verdict. Default 16 / 256. The error bar
                                shrinks with the number of *windows*, so small
                                windows and a high token floor is the cheap way
                                to resolve a small effect.
  EXPERT_STREAM_SIDECAR_GOV_MARGIN / _Z
                                how much faster actuation must measure (ratio,
                                default 1.03) and how many error bars the gap
                                must clear (default 3.0) to switch ON. Switching
                                OFF needs only half the Z: stopping a
                                speculation is cheap to get wrong, starting one
                                is not.
  EXPERT_STREAM_SIDECAR_GOV_HOLD / _PROBE_DUTY / _MAX_PROBE
                                tokens a verdict holds (512), how often re-probes
                                sample the losing side (1 window in 8, widening
                                as the verdict is confirmed), and how far past
                                the minimum sample an inconclusive round runs
                                before calling a tie (24x). Ties settle OFF.
  EXPERT_STREAM_SIDECAR_MIN_PROB
                                minimum calibrated probability for a speculative
                                read. Default 0.35. Volume tracks confidence
                                rather than a fixed top-k, so an unsure head
                                ships nothing instead of k reads per layer.
  EXPERT_STREAM_SIDECAR_SPEC_BYTE_FRAC
                                speculative bytes per token as a share of the
                                demand-miss bytes the same token paid for.
                                Default 0.25. Bounds the worst case before any
                                learning has happened, and scales itself across
                                models whose experts differ by 6x in size.
  EXPERT_STREAM_SIDECAR_TARGET_PRECISION
                                precision the issue threshold servos toward.
                                Default 0.5.
  EXPERT_STREAM_SIDECAR_HEAD_PRUNE
                                0 (default) = off. 1 = adaptive prune/wait
                                (quality-affecting; shadow->promote).
  EXPERT_STREAM_SIDECAR_PRUNE_MIN_MASS
                                fraction of the *baseline* threshold's retained
                                mixture mass the shadow prune policy must match
                                before it may go live. Default 0.98.
  EXPERT_STREAM_SIDECAR_MAX_WASTE
                                share of the wrap head's extra reads that may go
                                unused before it shrinks its top-k. Default 0.85.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env(name: str, default, cast):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return cast(raw)


def _default_models_dir() -> str:
    return str(Path.home() / "models" / "paged-moe")


MODELS_DIR: str = _env("EXPERT_STREAM_MODELS_DIR", _default_models_dir(), str)

# "auto" | "resident" | "streamed"
MODE: str = _env("EXPERT_STREAM_MODE", "auto", str)

# With F_NOCACHE reads there is no hidden page-cache growth, so the engine can
# safely claim more than half of RAM for backbone + expert cache.
#
# 0.70 did OOM a 48 GB machine once - but that was before F_NOCACHE, when every
# expert byte landed in the kernel's page cache *as well as* the expert LRU. The
# pile-on that caused it is gone, and the wired limit is now forced to 0, so the
# budget is no longer fighting an invisible second copy of the checkpoint.
RAM_FRACTION: float = _env("EXPERT_STREAM_RAM_FRACTION", 0.68, float)

# None means "auto": min(RAM_FRACTION * RAM - backbone - headroom, MAX_CACHE_GB)
CACHE_GB: float | None = _env("EXPERT_STREAM_CACHE_GB", None, float)

# Hard ceiling so auto mode never grabs most of a 48 GB box.
#
# This ceiling is the single biggest lever on how an agent session *feels*, and
# 18 was leaving most of it on the table. Every agent turn appends a few hundred
# fresh tokens, and a short prefill chunk routes across nearly every expert, so
# each turn re-reads the whole non-resident expert mass. The delay before each
# edit is therefore (expert mass - cache) / bandwidth and barely depends on how
# many tokens arrived. Measured on qwen3-next (40.5 GiB of experts, 1.26 GiB
# backbone), 4 steady-state turns of ~750 fresh tokens each:
#
#   cache   read/turn   prefill/turn   hit rate   decode
#   18 GB     16.7 GB       2.87 s       0.52     14.3 tok/s
#   26 GB      9.9 GB       2.29 s       0.65     18.6 tok/s
#   32 GB      4.8 GB       1.87 s       0.75     23.0 tok/s
#
# 26 is the default: it takes ~70% of the available win while leaving ~15 GiB
# for macOS, the KV cache and other apps. 32 measured better still and is one
# env var away (EXPERT_STREAM_CACHE_GB=32), but on a 48 GB machine it leaves
# little room for a long KV cache plus an editor and a browser.
MAX_CACHE_GB: float = _env("EXPERT_STREAM_MAX_CACHE_GB", 26.0, float)

# Bypass the OS page cache for expert reads (macOS F_NOCACHE).
NOCACHE: bool = bool(_env("EXPERT_STREAM_NOCACHE", 1, int))

READ_THREADS: int = _env("EXPERT_STREAM_READ_THREADS", 16, int)

# Slot-read pool workers (ExpertCache._ReadPool): decode miss queue depth.
# Miss threads push reads here directly. disk_ceiling.py saturates ~qd16 on
# 3 MB requests and still gains at qd64 on 0.2 MB scales, so default wants
# deeper than READ_THREADS.
# 0 = auto (2x READ_THREADS); -1 disables and restores the ThreadPoolExecutor path.
READ_POOL_THREADS: int = _env("EXPERT_STREAM_READ_POOL_THREADS", 0, int)

# Speculative inflight cap as a multiple of READ_THREADS. Too low leaves the
# drive idle (drive_idle.py: ~19% idle while speculation skips ~200x/token).
# 0 = auto.
SPEC_INFLIGHT_MULT: float = _env("EXPERT_STREAM_SPEC_INFLIGHT_MULT", 3.0, float)

# Slab slots held back from residency so in-flight reads and unclaimed
# guesses have somewhere to land. Same pool as the cache, so this trades hit
# rate for prefetch runway. Raising STAGING_GB alone does not buy runway:
# staging past the reserve starves reads of slots (Qwen3-235B: drive
# occupancy 80% -> 66%).
SLAB_RESERVE_FRAC: float = _env("EXPERT_STREAM_SLAB_RESERVE_FRAC", 0.1, float)

# Deep enough to hide NVMe latency behind a few layers of attention+MoE
# on big models (GLM-4.5-Air has 46 MoE layers).
PREFETCH_DEPTH: int = _env("EXPERT_STREAM_PREFETCH_DEPTH", 8, int)

# Route prediction: ask the next layers' (resident) routers what they will want,
# instead of assuming they want what they wanted for the previous token.
def _predict_mode(raw) -> str:
    v = str(raw).strip().lower()
    if v in ("0", "off", "false", "no"):
        return "off"
    if v in ("1", "on", "true", "yes"):
        return "on"
    return "auto"


PREDICT: str = _predict_mode(_env("EXPERT_STREAM_PREDICT", "auto", str))
PREDICT_WINDOW: int = _env("EXPERT_STREAM_PREDICT_WINDOW", 8, int)
PREDICT_MIN_TOKENS: int = _env("EXPERT_STREAM_PREDICT_MIN_TOKENS", 96, int)
PREDICT_MARGIN: float = _env("EXPERT_STREAM_PREDICT_MARGIN", 1.03, float)
PREDICT_Z: float = _env("EXPERT_STREAM_PREDICT_Z", 2.0, float)
PREDICT_MAX_PROBE: int = _env("EXPERT_STREAM_PREDICT_MAX_PROBE", 6, int)
PREDICT_HOLD_WINDOWS: int = _env("EXPERT_STREAM_PREDICT_HOLD", 128, int)
PREDICT_DEPTH: int = _env("EXPERT_STREAM_PREDICT_DEPTH", 3, int)
# Aim the prediction window this many layers further ahead (layer i predicts
# layers i+lead+1 .. i+lead+depth). Lead time and burst size are separate
# knobs: see PrefetchRing.__init__.
PREDICT_LEAD: int = _env("EXPERT_STREAM_PREDICT_LEAD", 0, int)
# One extra prediction per layer, aimed this many layers ahead, for source
# layers past PREDICT_FAR_FROM of the stack. bench/router_matrix.py shows
# router agreement decays with SOURCE DEPTH rather than distance: on
# Qwen3-235B, precision among cache misses from layer 0 dies by distance 4
# (0.28) while from layer 24 it holds 0.73 at distance 32, against a net-win
# bar of 0.5. 0 (default) keeps the shipped uniform-window behaviour; try
# 16-32 on a disk-bound model, and let the prediction governor confirm it.
PREDICT_FAR_LEAD: int = _env("EXPERT_STREAM_PREDICT_FAR_LEAD", 0, int)
PREDICT_FAR_FROM: float = _env("EXPERT_STREAM_PREDICT_FAR_FROM", 0.17, float)
PREDICT_SLACK: int = _env("EXPERT_STREAM_PREDICT_SLACK", 2, int)
PREDICT_RENORM: bool = bool(_env("EXPERT_STREAM_PREDICT_RENORM", 0, int))
# When decode pruning is on, prediction prunes its guesses with the same
# router-weight test - but against threshold PRUNE * this factor, not PRUNE
# itself (1.0, the default, = the demand threshold; 0 = never prune guesses).
# Softer values keep borderline experts the demand path might rescue, but on
# Qwen3-235B every softening measured slower: 5.34 tok/s at 1.0 vs 5.14 at
# 0.8 vs 4.57 at 0.5 - extra coverage never paid for the extra queue traffic,
# because blocking reads already run near the drive's scattered ceiling and
# anything speculative just stands in their way.
PREDICT_PRUNE_SOFT: float = _env("EXPERT_STREAM_PREDICT_PRUNE_SOFT", 1.0, float)
# Install route-predicted slab reads straight into the LRU on arrival instead
# of staging them. Only pays when prediction precision is high (wrong installs
# evict resident experts - cache pollution); measured 15% slower at precision
# 0.5, so default off.
PREDICT_INSTALL: bool = bool(_env("EXPERT_STREAM_PREDICT_INSTALL", 0, int))

# Online expert-prefetch sidecar (v2). Off by default so models that don't
# opt in pay nothing. See expert_stream/sidecar.py.
SIDECAR: bool = bool(_env("EXPERT_STREAM_SIDECAR", 0, int))
# 0 = auto (scaled from the model's MoE geometry at attach time).
SIDECAR_WRAP: int = _env("EXPERT_STREAM_SIDECAR_WRAP", 0, int)
SIDECAR_TOPK: int = _env("EXPERT_STREAM_SIDECAR_TOPK", 0, int)
# When TOPK is auto, aim for this multiple of the observed experts/layer/token.
SIDECAR_TOPK_FACTOR: float = _env("EXPERT_STREAM_SIDECAR_TOPK_FACTOR", 1.0, float)
SIDECAR_MIN_SCORE: float = _env("EXPERT_STREAM_SIDECAR_MIN_SCORE", 0.04, float)
SIDECAR_HISTORY: int = _env("EXPERT_STREAM_SIDECAR_HISTORY", 6, int)
SIDECAR_LOCK_HITS: int = _env("EXPERT_STREAM_SIDECAR_LOCK_HITS", 0, int)
SIDECAR_LOCK_MIN_TOKENS: int = _env(
    "EXPERT_STREAM_SIDECAR_LOCK_MIN_TOKENS", 4096, int
)
SIDECAR_LOCK_OVER_CEILING: float = _env(
    "EXPERT_STREAM_SIDECAR_LOCK_OVER_CEILING", 0.15, float
)
SIDECAR_MIN_RECALL: float = _env("EXPERT_STREAM_SIDECAR_MIN_RECALL", 0.40, float)
SIDECAR_MIN_PRECISION: float = _env(
    "EXPERT_STREAM_SIDECAR_MIN_PRECISION", 0.22, float
)
SIDECAR_WINDOW: int = _env("EXPERT_STREAM_SIDECAR_WINDOW", 32, int)
SIDECAR_LR: float = _env("EXPERT_STREAM_SIDECAR_LR", 0.08, float)
# 0 = auto (scaled from expert count / layer count).
SIDECAR_FEAT_DIM: int = _env("EXPERT_STREAM_SIDECAR_FEAT_DIM", 0, int)
SIDECAR_LOG_EVERY: int = _env("EXPERT_STREAM_SIDECAR_LOG_EVERY", 32, int)

# --- v4 learner -------------------------------------------------------------
# Per-parameter adaptive step (AdaGrad). Sparse hashed features converge much
# faster with it; costs one extra float32 array per trainable head.
SIDECAR_ADAGRAD: bool = bool(_env("EXPERT_STREAM_SIDECAR_ADAGRAD", 1, int))
SIDECAR_WD: float = _env("EXPERT_STREAM_SIDECAR_WD", 1e-5, float)
# Blend a per-layer expert-frequency prior into scores while the model is cold;
# weight decays to 0 over this many train steps.
SIDECAR_PRIOR_W: float = _env("EXPERT_STREAM_SIDECAR_PRIOR_W", 0.5, float)
SIDECAR_PRIOR_DECAY: int = _env("EXPERT_STREAM_SIDECAR_PRIOR_DECAY", 20000, int)
# Signed random projection of the last hidden state, appended to the features.
# The single strongest signal for next-token routing. 0 disables the block.
SIDECAR_SKETCH_DIM: int = _env("EXPERT_STREAM_SIDECAR_SKETCH_DIM", 64, int)
SIDECAR_HIDDEN: bool = bool(_env("EXPERT_STREAM_SIDECAR_HIDDEN", 1, int))

# --- v4 actuation / gating --------------------------------------------------
# Prefetch only what the engine's own last-token heuristic does NOT already
# cover, so the sidecar is strictly additive and its marginal value is
# measurable. Promote on that marginal recall gain rather than absolute recall.
SIDECAR_MIN_GAIN: float = _env("EXPERT_STREAM_SIDECAR_MIN_GAIN", 0.02, float)
# Ceiling on the share of extra reads that go unused. The wrap head shrinks its
# top-k while it is above this and grows back toward the observed router fan-out
# below it, so the cost/benefit knob tunes itself per model.
SIDECAR_MAX_WASTE: float = _env("EXPERT_STREAM_SIDECAR_MAX_WASTE", 0.85, float)
# Max extra experts issued per token (0 = auto: wrap * topk).
SIDECAR_EXTRA_BUDGET: int = _env("EXPERT_STREAM_SIDECAR_EXTRA_BUDGET", 0, int)

# --- v4 cost guards ---------------------------------------------------------
# The sidecar runs inline between tokens, so its own cost is token latency.
# Above either bound it sheds work (topk, then heads) and logs once.
SIDECAR_TIME_BUDGET_MS: float = _env(
    "EXPERT_STREAM_SIDECAR_TIME_BUDGET_MS", 4.0, float
)
SIDECAR_MAX_OVERHEAD: float = _env("EXPERT_STREAM_SIDECAR_MAX_OVERHEAD", 0.05, float)
# Persist every N decode tokens so a kill never costs a whole session.
SIDECAR_SAVE_EVERY: int = _env("EXPERT_STREAM_SIDECAR_SAVE_EVERY", 512, int)


def _default_sidecar_dir() -> str:
    return str(Path.home() / ".paged_moe" / "sidecar")


SIDECAR_DIR: str = _env("EXPERT_STREAM_SIDECAR_DIR", _default_sidecar_dir(), str)


def sync_from_environ() -> None:
    """Re-bind module-level knobs from the current process environment.

    Used by the mlx_lm drop-in hook when a matched model carries per-model
    ``EXPERT_STREAM_*`` overrides. Safe to call repeatedly; unknown attrs are
    skipped. Bool knobs expect ``0`` / ``1``.
    """
    g = globals()
    for name, cur in list(g.items()):
        if not name.isupper() or name.startswith("_"):
            continue
        raw = os.environ.get(f"EXPERT_STREAM_{name}")
        if raw is None or raw == "":
            continue
        try:
            if isinstance(cur, bool):
                g[name] = bool(int(raw))
            elif isinstance(cur, int) and not isinstance(cur, bool):
                g[name] = int(raw)
            elif isinstance(cur, float):
                g[name] = float(raw)
            elif isinstance(cur, str):
                g[name] = str(raw)
            elif cur is None:
                # Optional floats (CACHE_GB, ...): prefer float when it parses.
                try:
                    g[name] = float(raw)
                except ValueError:
                    g[name] = raw
        except (TypeError, ValueError):
            pass

# Per-head enable bits. Master SIDECAR must be on; each head defaults off
# except wrap (the original behaviour). Disabled head = no weights, no
# hot-path work beyond a bool check on the façade.
SIDECAR_HEAD_WRAP: bool = bool(_env("EXPERT_STREAM_SIDECAR_HEAD_WRAP", 1, int))
SIDECAR_HEAD_PREFILL: bool = bool(
    _env("EXPERT_STREAM_SIDECAR_HEAD_PREFILL", 0, int)
)
SIDECAR_HEAD_RESIDENCY: bool = bool(
    _env("EXPERT_STREAM_SIDECAR_HEAD_RESIDENCY", 0, int)
)
SIDECAR_HEAD_PRUNE: bool = bool(_env("EXPERT_STREAM_SIDECAR_HEAD_PRUNE", 0, int))
# 0 = auto for both (scaled from MoE layer count / observed demand).
SIDECAR_PREFILL_LAYERS: int = _env("EXPERT_STREAM_SIDECAR_PREFILL_LAYERS", 0, int)
SIDECAR_PREFILL_TOPK: int = _env("EXPERT_STREAM_SIDECAR_PREFILL_TOPK", 0, int)
# A prefill chunk's expert *union* is nearly every expert, so predicting it is
# trivial and worthless. The head targets the chunk's HOT set instead: experts
# whose per-chunk token count is at least this fraction of the layer's max.
SIDECAR_PREFILL_HOT_FRAC: float = _env(
    "EXPERT_STREAM_SIDECAR_PREFILL_HOT_FRAC", 0.5, float
)
# --- net-time governor -----------------------------------------------------
# The sidecar's only unconditional promise is "not slower". Recall-style
# metrics cannot deliver it: a head can hold high recall while most of its
# speculative reads go unused, and on a disk-bandwidth-bound decode those
# unused reads come straight out of the demand path. Both benched models
# shipped in exactly that state (-24% and -13% tok/s at positive `gain`).
# So actuation is decided by an interleaved A/B on measured token time
# instead, and the heads only get to spend bandwidth while it is winning.
SIDECAR_GOVERNOR: bool = bool(_env("EXPERT_STREAM_SIDECAR_GOVERNOR", 1, int))
# Tokens per A/B window. Small, because the error bar on the comparison
# shrinks with the number of *windows*, not the number of tokens - and short
# windows also interleave the two states finely enough that both see the same
# mix of work. Matches the route-prediction governor's window for the same
# reason.
SIDECAR_GOV_WINDOW: int = _env("EXPERT_STREAM_SIDECAR_GOV_WINDOW", 16, int)
# Tokens per arm before a verdict is even considered. At the window above this
# is 16 windows per arm, enough for the error bar to resolve a ~20% effect,
# which is the size of the regressions actually measured here.
SIDECAR_GOV_MIN_TOKENS: int = _env("EXPERT_STREAM_SIDECAR_GOV_MIN_TOKENS", 256, int)
# How much faster actuation must measure before it is allowed to stay on, as a
# ratio (1.03 = 3%). Same convention as EXPERT_STREAM_PREDICT_MARGIN. Not 1.0:
# at 1.0, noise alone turns actuation on about half the time.
SIDECAR_GOV_MARGIN: float = _env("EXPERT_STREAM_SIDECAR_GOV_MARGIN", 1.03, float)
# Error bars the gap must clear to switch actuation ON. Switching OFF needs
# only half of this - stopping a speculation that may be costing throughput is
# cheap to get wrong, starting one is not.
SIDECAR_GOV_Z: float = _env("EXPERT_STREAM_SIDECAR_GOV_Z", 3.0, float)
# How far past the minimum sample an inconclusive A/B keeps probing before it
# is called a tie (a multiple of SIDECAR_GOV_MIN_TOKENS). Ties settle OFF: if
# the effect is still inside the noise after this much evidence, there is
# little to win and the configuration that spends no disk is the safe one.
SIDECAR_GOV_MAX_PROBE: float = _env("EXPERT_STREAM_SIDECAR_GOV_MAX_PROBE", 24.0, float)
# Tokens a verdict holds before the next probe round. Bounds the cost of being
# wrong after conditions change, and the cost of probing when they have not.
SIDECAR_GOV_HOLD: int = _env("EXPERT_STREAM_SIDECAR_GOV_HOLD", 512, int)
# After the first verdict, re-probes sample the losing configuration only one
# window in this many. An even re-probe would spend a quarter of all tokens in
# a state already measured slower, which is most of the regression this is
# supposed to remove.
SIDECAR_GOV_PROBE_DUTY: int = _env("EXPERT_STREAM_SIDECAR_GOV_PROBE_DUTY", 8, int)

# --- wrap issuance ---------------------------------------------------------
# Minimum calibrated probability for a speculative read. Volume tracks
# confidence rather than a fixed top-k, so an uncertain token issues little or
# nothing instead of always issuing k per layer.
SIDECAR_MIN_PROB: float = _env("EXPERT_STREAM_SIDECAR_MIN_PROB", 0.35, float)
# Speculative bytes per token as a share of the demand misses the same token
# paid for. Keeps speculation a bounded fraction of a cost we were already
# incurring, on the model's own current working set, at any cache size.
SIDECAR_SPEC_BYTE_FRAC: float = _env(
    "EXPERT_STREAM_SIDECAR_SPEC_BYTE_FRAC", 0.25, float
)
# Precision the issue threshold servos toward. At 1/2, a speculative read is
# expected to save more demand-path time than it costs.
SIDECAR_TARGET_PRECISION: float = _env(
    "EXPERT_STREAM_SIDECAR_TARGET_PRECISION", 0.5, float
)

SIDECAR_RESIDENCY_HORIZON: int = _env(
    "EXPERT_STREAM_SIDECAR_RESIDENCY_HORIZON", 32, int
)
SIDECAR_RESIDENCY_MIN_SCORE: float = _env(
    "EXPERT_STREAM_SIDECAR_RESIDENCY_MIN_SCORE", 0.35, float
)
# Never protect more than this share of scanned eviction candidates - a cache
# of uniformly "hot" keys must still make progress. Also sets the top slice of
# the observed score distribution the protect bar tracks. Lowered from 0.33:
# at a third of candidates the head stopped selecting and merely perturbed
# LRU's victim order, which measured +55% misses against leaving LRU alone.
SIDECAR_RESIDENCY_MAX_PROTECT: float = _env(
    "EXPERT_STREAM_SIDECAR_RESIDENCY_MAX_PROTECT", 0.10, float
)
# Minimum separate demands on a key before its reuse rate is allowed to
# override LRU. 88% of just-used experts are reused within the horizon, so a
# rate computed from one or two observations is noise, and LRU's recency
# ordering beats noise.
SIDECAR_RESIDENCY_MIN_HITS: int = _env(
    "EXPERT_STREAM_SIDECAR_RESIDENCY_MIN_HITS", 8, int
)
SIDECAR_PRUNE_MAX: float = _env("EXPERT_STREAM_SIDECAR_PRUNE_MAX", 0.95, float)
# Adaptive prune only goes live once the shadow policy retains at least this
# fraction of the mixture mass the shipped PRUNE threshold retains on the same
# tokens (quality guard, not just agreement with a regression target). Relative,
# not absolute: PRUNE=0.7 itself keeps well under half the mass on GLM and was
# measured fine, so an absolute bar would permanently veto the head.
SIDECAR_PRUNE_MIN_MASS: float = _env(
    "EXPERT_STREAM_SIDECAR_PRUNE_MIN_MASS", 0.98, float
)

# Opt-in decode-time expert pruning ("turbo" mode). 0 (default) = off:
# streamed output stays bit-identical to resident inference. A value t in
# (0, 1] drops a routed expert whenever its router weight is below t times the
# strongest selected expert's weight for that token - e.g. 0.1 skips experts
# the router itself scored at <10% of the winner. MoE routers concentrate
# most of the mixture weight in 2-4 of the top-8, so a conservative threshold
# removes a large fraction of the per-token disk *and* GPU cost at a small,
# bounded output perturbation (the dropped contribution's total mixture weight
# is at most K*t/(1+K*t)). Only applies to decode, only on architectures whose
# router returns plain logits (Qwen3-MoE family); everything else silently
# ignores it.
PRUNE: float = _env("EXPERT_STREAM_PRUNE", 0.0, float)

# Hard cap on experts computed per token during decode ("use 4 of 8").
#
# 0 (default) = off: every expert the router selected is computed. A value
# 1 <= c < K keeps only the c strongest of the router's K slots and zeroes the
# rest, which is the same mechanism as PRUNE (see above) with a rank test
# instead of a weight test. Where PRUNE adapts - a token whose router is
# confident keeps 2 experts, an ambiguous one keeps 8 - a cap is a *guarantee*:
# disk demand per layer can never exceed c, which is what makes the miss mass
# of a 235B-class model predictable instead of merely smaller on average.
#
# Only applies to decode (prefill computes the full mixture, so a prompt's
# hidden states are exact), and only on architectures whose router returns
# plain logits - the same requirement PRUNE has.
ROUTE_TOP_K: int = _env("EXPERT_STREAM_ROUTE_TOP_K", 0, int)

# Mass-based expert selection ("nucleus routing"): keep the fewest experts
# whose mixture weights sum to at least this fraction, drop the rest.
#
# 0 (default) = off. This is the most direct statement of the actual tradeoff.
# PRUNE thresholds each weight against the strongest one, which is scale-free
# but says nothing about how much of the mixture goes missing: on a token whose
# top weight is 0.30 with seven 0.10 siblings, PRUNE=0.5 keeps one expert and
# silently discards 70% of the mass. TOP_K fixes the *count*, which overspends
# on confident tokens (experts 3 and 4 carry almost nothing, yet each costs a
# ~10 MB read) and underspends on ambiguous ones (where the 5th expert still
# matters). TOP_P instead bounds the discarded mass at 1-p per layer, so reads
# go where the mixture is genuinely spread out and confident tokens get cheap.
ROUTE_TOP_P: float = _env("EXPERT_STREAM_ROUTE_TOP_P", 0.0, float)

# Never drop an expert that needs no disk read (resident, staged, or already
# in flight).
#
# Every approximation above trades output quality for disk bytes. On a resident
# expert there are no bytes to save, so that trade is pure loss: at a decode hit
# rate of ~0.82, most of what PRUNE/TOP_P discard were free. Keeping them moves
# the computed mixture back toward the full one at identical disk traffic, and
# costs only the expert's matmul (~20 ms/token covers all of them).
#
# Caveat: the mixture a token gets now depends on what happens to be cached, so
# the same prompt can decode slightly differently depending on session history.
# The output was already an approximation once PRUNE/TOP_K/TOP_P is on; this
# makes it a *better* approximation but a less reproducible one. Set to 0 for
# run-to-run determinism.
KEEP_FREE: bool = bool(_env("EXPERT_STREAM_KEEP_FREE", 1, int))

# Don't stall the GPU on the disk for an expert that barely matters.
#
# KEEP_FREE settled the cheap half of the trade (never drop a free expert).
# This is the other half: when a routed expert *is* missing, the layer either
# waits for a ~10 MB read - the single largest cost in a streamed decode, ~70
# ms/token on Qwen3-235B - or skips it and prefetches it for later tokens. A
# value of w means "only wait when the missing expert carries at least w of the
# token's mixture mass at this layer".
#
# 0 (default) = always wait, the historical behavior. 1.01 = never wait. The
# useful settings are small: at K=8 the strongest expert typically carries
# 0.25-0.45 and the tail 0.05-0.12, so a gate around 0.15-0.25 keeps the
# decisive reads on the critical path and lets the tail arrive whenever it
# arrives. Skipped experts are still prefetched, so they populate the cache for
# subsequent tokens rather than being lost.
WAIT_ABOVE: float = _env("EXPERT_STREAM_WAIT_ABOVE", 0.0, float)

# Rescale the surviving experts' contributions by 1/(their mixture mass) when
# PRUNE, TOP_K or TOP_P drops a slot.
#
# Default OFF, and that is a measured decision, not an oversight. Renormalizing
# is the mathematically "correct" reduced mixture - with TOP_K it makes the
# engine exactly equivalent to the model at num_experts_per_tok=cap (asserted in
# tests/test_stream.py) - yet it measures consistently *worse* on Qwen3-235B:
# prune 0.5 goes ppl 5.17 -> 7.21, cap 4 goes 5.34 -> 5.57. Attenuating the MoE
# block's contribution (what you get without renorm) is a small step toward
# skipping the block, which the residual stream tolerates well; renormalizing
# instead hands one surviving expert several times its trained weight, which is
# out of distribution.
#
# Without this, dropping half the mixture also drops half its magnitude, and
# the MoE block returns a residual contribution that is systematically too
# small - an error that compounds across 94 layers. With it, the block returns
# a correctly normalized mixture over the experts that *were* computed, which
# is exactly what norm_topk_prob does for the full top-k. Costs one broadcast
# multiply per MoE layer.
ROUTE_RENORM: bool = bool(_env("EXPERT_STREAM_ROUTE_RENORM", 0, int))

# Slot-addressed slab storage for resident experts (see slab.py). Reads land
# directly in Metal-backed memory and a layer's experts are computed with one
# gather_qmm per projection instead of three calls per expert - worth ~130
# ms/token on Qwen3-235B, where per-expert dispatch and the host->device copy
# together are a third of the decode budget.
#
#   "auto" (default) on when the model is disk-bound (expert mass >> cache),
#          which is exactly when per-token miss traffic and dispatch counts
#          are large enough for slabs to matter
#   "on"   whenever the architecture allows it (uniform expert shapes)
#   "off"  the per-expert mx.array path
#
# Unlike the per-expert cache's soft byte budget, a slab is committed at load
# and cannot be lent back to prefill activations, so SLAB_GB is a hard
# reservation. None = use the expert-cache budget.
SLAB: str = _env("EXPERT_STREAM_SLAB", "auto", str)
# Debug: check at compute time that no slot a layer is about to read is being
# written, and that slot bookkeeping agrees. Slot reuse overwrites bytes in
# place, so a double-owned slot corrupts output instead of raising.
SLAB_VERIFY: bool = bool(_env("EXPERT_STREAM_SLAB_VERIFY", 0, int))
# Debug: log Metal memory at the prefill/decode transitions. A prefill OOM is
# the one failure that kills the process outright (the command buffer fails
# asynchronously, so there is nothing to catch), which makes knowing what was
# actually committed at that moment the difference between a fix and a guess.
MEM_DEBUG: bool = bool(_env("EXPERT_STREAM_MEM_DEBUG", 0, int))
_slab_raw = _env("EXPERT_STREAM_SLAB_GB", None, float)
SLAB_BYTES: int | None = None if _slab_raw is None else int(_slab_raw * (1 << 30))

# Frequency-protected eviction (LFRU-style second chance).
LFRU: bool = bool(_env("EXPERT_STREAM_LFRU", 1, int))
LFRU_SCAN: int = _env("EXPERT_STREAM_LFRU_SCAN", 8, int)

# Ceiling on prefetched-but-not-yet-used numpy staging. None = auto-scale
# with the model's expert size (see staging_bytes_auto).
_staging_raw = _env("EXPERT_STREAM_STAGING_GB", None, float)
STAGING_BYTES: int | None = (
    None if _staging_raw is None else int(_staging_raw * (1 << 30))
)


def staging_bytes_auto(expert_nbytes: int) -> int:
    """Staging must hold the prefetch horizon in *experts*, not bytes.

    A fixed 1 GB holds ~400 qwen3-next experts (plenty) but <100 of a 235B's
    10.6 MB experts - measured on Qwen3-235B-A22B, that dropped ~100 staged
    experts per decoded token, many of them correct predictions that were then
    re-read from disk as blocking misses. ~256 experts of headroom covers the
    deepest prefetch horizon (PREFETCH_DEPTH + PREDICT_DEPTH layers of top-k
    plus slack) with room for timing skew.
    """
    return max(1 << 30, min(6 << 30, 256 * max(1, expert_nbytes)))


# Bytes evicted between mx.clear_cache() calls. None = auto-scale with the
# cache budget (see clear_bytes_auto).
_clear_raw = _env("EXPERT_STREAM_CLEAR_GB", None, float)
CLEAR_BYTES: int | None = None if _clear_raw is None else int(_clear_raw * (1 << 30))


def clear_bytes_auto(cache_budget_bytes: int) -> int:
    """clear_cache() cadence: rare, and proportional to the churn.

    Big-expert models evict multiple GB per decoded token; a fixed 2 GB
    threshold flushed Metal's recycled-buffer pool roughly once per token, so
    every token's miss mass paid fresh zero-fill page faults instead of
    recycling last token's buffers (expert tensors are uniform sizes, so
    recycling is perfect). The pool itself is bounded by set_cache_limit, so
    a large threshold here is safe.
    """
    return max(2 << 30, cache_budget_bytes // 3)

# Prefill processes each layer's experts in groups of at most this many bytes,
# so a 512-expert layer never sits in memory all at once.
GROUP_BYTES: int = _env("EXPERT_STREAM_GROUP_MB", 384, int) * (1 << 20)

# Residency budget while a large prefill chunk runs (see ExpertCache.enter_prefill).
PREFILL_CACHE_GB: float = _env("EXPERT_STREAM_PREFILL_CACHE_GB", 4.0, float)

# Prompt tokens per prefill pass (server/CLI default).
PREFILL_CHUNK: int = _env("EXPERT_STREAM_PREFILL_CHUNK", 32768, int)

# Contexts below this many tokens keep an fp16 KV cache; past it, the cache is
# quantized to 8-bit (kv_bits=8) as before. fp16 KV measured +8% decode
# throughput on Qwen3-235B at 2k context - quantized-KV attention adds
# dequant/quant ops to every layer's graph, and with 94 per-layer syncs the
# encode cost of those ops is paid on the critical path 94 times per token.
# fp16 KV costs ~192 KB/token on Qwen3-235B, so 8k tokens is ~1.6 GB - cheap.
# Past the threshold the memory argument wins. 0 = always quantize (old
# behavior).
KV_FP16_CTX: int = _env("EXPERT_STREAM_KV_FP16_CTX", 8192, int)

# Precision of the decode-time KV cache. 8-bit at group 64 costs 1.0625 B per
# element against fp16's 2, and quantized-KV attention is only used for decode
# (one query row), where its lack of a fused kernel does not matter.
KV_BITS: int = _env("EXPERT_STREAM_KV_BITS", 8, int)
KV_GROUP_SIZE: int = _env("EXPERT_STREAM_KV_GROUP_SIZE", 64, int)

# Context length to reserve KV memory for when sizing the expert cache.
#
# This is the single most important number for a full-attention model on a
# small machine, because KV and experts come out of the same pool: GLM-4.7
# spends 196 KB/token at 8 bits, so a 24k-token conversation is 4.8 GB that the
# expert cache must not have already committed to a slab. Sizing the cache
# without knowing it is what made GLM-4.7 Metal-OOM on turn two.
#
# The host app sets this from the model's `numCtx`. 0 falls back to the flat
# _HEADROOM_BYTES allowance in the loader, which is right only for models whose
# KV is cheap (Qwen3-235B: 4 KV heads, 102 KB/token).
RESERVE_CTX: int = _env("EXPERT_STREAM_RESERVE_CTX", 0, int)

# Multiplier on the reserved KV bytes, covering the prompt-cache store's steady
# state next to the live cache. 1.25 is one full conversation plus a partial
# user-segment snapshot (see server._snapshot_user_segment).
KV_STORE_SLACK: float = _env("EXPERT_STREAM_KV_STORE_SLACK", 1.25, float)

# Hand a reused prompt cache to the request instead of deep-copying it
# (see kvmem.take_nearest_cache). 0 restores mlx-lm's copying.
PROMPT_CACHE_MOVE: bool = bool(_env("EXPERT_STREAM_PROMPT_CACHE_MOVE", 1, int))

# Only chunks at least this big switch the cache to the shrunk budget.
#
# Reading the expert mass costs ~the same for a 500-token chunk as for a
# 32k-token one (either way, essentially every expert of every layer gets
# routed to). So shrinking the cache pays off only when it buys *fewer passes*:
# for a big cold prompt, yes; for the few-hundred-token delta of an agent turn
# that hits the prompt cache, shrinking would just evict the warm experts that
# make the delta cheap. 8192 is the largest chunk that still fits alongside a
# full cache, which makes it the natural break-even point.
PREFILL_SHRINK_TOKENS: int = _env("EXPERT_STREAM_PREFILL_SHRINK_TOKENS", 8192, int)


def total_ram_bytes() -> int:
    """Physical RAM (macOS + Linux)."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        import subprocess

        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
        )
        return int(out.stdout.strip())
