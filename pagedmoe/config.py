# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Configuration.

Everything has a sane default and can be overridden by environment variables
(and most things also by CLI flags / load() kwargs, which win over env vars).

Env vars:
  PAGED_MOE_DOWNLOAD_DIR      where downloaded models are stored.
                                Default: ~/models/paged-moe.
  PAGED_MOE_MODE            "auto" | "resident" | "streamed".  Default auto:
                                fully load models that fit in RAM, stream the rest.
  PAGED_MOE_RAM_FRACTION    fraction of physical RAM the whole engine may use
                                (backbone + expert cache + KV headroom).
                                Default 0.68. Safe now that expert reads bypass
                                the OS page cache (see NOCACHE) and the Metal
                                wired limit is never raised.
  PAGED_MOE_CACHE_GB        explicit expert-cache budget in GB. Overrides the
                                auto budget. 32 is worth trying on a 48 GB
                                machine if little else is running.
  PAGED_MOE_MAX_CACHE_GB    hard ceiling on the auto cache budget. Default 26.
                                This is the main lever on the delay between agent
                                turns; see the table at its definition below.
  PAGED_MOE_MEM_PRESSURE    1 (default) = shrink the slab at its next rebuild
                                while macOS reports memory pressure, and grow it
                                back when pressure clears. _STEP (0.15), _FLOOR
                                (0.5) and _POLL_S (0.5) tune it.
  PAGED_MOE_SLAB_REGROW     1 (default, needs MEM_PRESSURE) = rebuild a slab
                                left small once memory has been calm for
                                _CALM_S (60) seconds, at the start of a turn.
  PAGED_MOE_PREFILL_SLAB    1 (default) = short prefills read misses into
                                free slab slots instead of host buffers.
  PAGED_MOE_PREFILL_AHEAD   1 (default) = each prefill layer starts the next
                                layer's predicted reads. _LOOSE_MB (512) of them
                                may go as run reads once free slots run out.
  PAGED_MOE_NOCACHE         1 (default) = read expert weights with F_NOCACHE,
                                bypassing the macOS page cache. This makes RAM
                                usage deterministic: the expert LRU is the only
                                place expert bytes live. 0 = let the kernel
                                page-cache the checkpoint (can balloon tens of
                                GB during prefill and freeze a 48 GB machine).
                                Every pread is capped at 1 MB, past which macOS
                                ignores F_NOCACHE.
  PAGED_MOE_PAGECACHE_TRIM  1 (default, needs NOCACHE) = drop the
                                checkpoint's pages from the file cache when it
                                grows _MB (512) since the last drop, checked
                                every _POLL_S (0.25). F_NOCACHE still leaks
                                some; see pagecache.py.
  PAGED_MOE_READ_THREADS    parallel disk readers. Default 16.
  PAGED_MOE_READ_POOL_THREADS
                                decode slot-read pool queue depth.
                                0 = auto (2x READ_THREADS), -1 = old executor
                                path. Default 0.
  PAGED_MOE_NATIVE_READ     1 (default) = prefer the C pread pool (one
                                job per expert, GIL released for the latch).
                                Same bytes as Python; falls back automatically
                                if the extension cannot load or compile.
                                0 = force Python. Sample configs leave this on
  PAGED_MOE_READ_SPLIT_KB   native pool piece size. Default 1024: idle
                                workers share one expert's preads, so a single
                                miss runs at queue depth instead of QD1.
                                0 = one worker per expert.
                                for large streamed MoEs.
  PAGED_MOE_EARLY_DECODE    0 (package default) = off. 1 = prefetch the
                                early-decode warm pool after prefill
                                (quality-safe). Sample / large-model recipes
                                turn this on.
  PAGED_MOE_EARLY_DECODE_TRAIN
                                0 (default) = listed off on purpose. With
                                EARLY_DECODE=1 and no warm-pool weights yet,
                                cold-starts until a high cover/prec bar, then
                                freezes train. 1 = continuous online learning.
  PAGED_MOE_EARLY_DECODE_TOKENS
                                early window length (default 64).
  PAGED_MOE_EARLY_DECODE_GB max GB to prefetch from the warm pool
                                (default 4).
  PAGED_MOE_EARLY_DECODE_MIN_COVER / _MIN_PREC
                                Hold prefetch when cover/prec EMAs fall below
                                these after MIN_EPISODES (still trains). Defaults
                                0.12 / 0.15.
  PAGED_MOE_EARLY_DECODE_DECAY
                                Per-episode count decay (default 0.97). Softens
                                stale domains so the pool cannot lock onto an
                                old traffic pattern forever.
  PAGED_MOE_EARLY_DECODE_BOOTSTRAP_COVER / _PREC / _MIN_EP / _MAX_EP
                                When EARLY_DECODE=1, TRAIN=0, and no warm-pool
                                weights exist yet: train until cover/prec EMAs
                                clear these high bars (defaults 0.40 / 0.22,
                                min 48 episodes, max 256), then freeze train
                                and keep prefetch-only. Explicit TRAIN=1 never
                                auto-stops.
  PAGED_MOE_TTW_LOG         1 (default) = print greppable [ttw] lines for
                                time-to-warm A/B (first decode through early window).
                                Not gated on SIDECAR_DEBUG.
  PAGED_MOE_TTW_TOKENS      window length for [ttw] warm line; 0 = use
                                EARLY_DECODE_TOKENS.
  PAGED_MOE_PREFETCH_DEPTH  speculative prefetch lookahead in layers. Default 8.
  PAGED_MOE_PREDICT         "auto" (default), "1" (always) or "0" (never):
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
  PAGED_MOE_PREDICT_WINDOW  decode tokens per governor window. Default 8.
                                Short windows interleave the two states so they
                                sample the same content; the evidence needed for
                                a decision is set separately, below.
  PAGED_MOE_PREDICT_MIN_TOKENS
                                tokens to pool per state before the governor
                                decides. Default 96 (~24 windows total). One
                                window per state is far below the noise floor of
                                agent traffic and made the decision flip nine
                                times in four minutes, running in the losing
                                configuration much of the time.
  PAGED_MOE_PREDICT_MARGIN  how much faster prediction must measure before
                                it is kept. Default 1.03. Necessary but not
                                sufficient: the gap must also clear the measured
                                error bar (see PREDICT_Z).
  PAGED_MOE_PREDICT_Z       error bars the gap must clear before the
                                governor acts on it. Default 2.0. A percentage
                                margin alone is not enough - agent decode rates
                                swing tens of percent window to window, so at
                                realistic sample sizes the error bar is wider
                                than the margin and ties resolve at random.
  PAGED_MOE_PREDICT_MAX_PROBE
                                multiples of PREDICT_MIN_TOKENS to keep probing
                                while the result stays inconclusive. Default 6;
                                after that the cheaper side (off) wins by
                                default and is held 4x longer, since a
                                difference still lost in the noise is not worth
                                re-measuring often.
  PAGED_MOE_PREDICT_HOLD    windows to keep the governor's decision before
                                re-measuring. Default 128 (~1024 tokens), which
                                spans several agent turns; re-probing keeps half
                                the previous evidence so a settled result is not
                                relitigated from scratch.
  PAGED_MOE_PREDICT_FAR_LEAD
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
  PAGED_MOE_PREDICT_FAR_FROM
                                fraction of the stack before FAR_LEAD engages.
                                Default 0.17 (~n/6), where the measured
                                early-layer collapse ends.
  PAGED_MOE_PREDICT_DEPTH   how many layers ahead to route-predict.
                                Default 3. Prediction accuracy is nearly flat
                                with distance (87-89% at depth 1-8 measured),
                                so depth is really a bandwidth knob: each extra
                                layer issues more speculative reads. Measured
                                decode tok/s on OLMoE at a 41% cache ratio:
                                20.8 off, 24.5 (d1), 29.5 (d2), 33.5 (d3),
                                34.5 (d4), 31.1 (d6), 29.3 (d8) - past ~4 the
                                wasted reads cost more than the hits gain.
                                Raise it if the disk has spare bandwidth.
  PAGED_MOE_PREDICT_SLACK   extra experts beyond top-k to prefetch per
                                predicted layer. Default 2 (0 measured 2% slower
                                and 5 points less coverage; 8 wasted 11 GB to
                                gain nothing).
  PAGED_MOE_PREDICT_RENORM  1 = rescale the hidden state by w_target/w_source
                                (the two RMSNorm weights) before applying the
                                target router. Default 0: measured neutral on
                                coverage (0.8824 vs 0.8821), so it is not worth
                                the extra multiply. Kept for architectures whose
                                norm scales vary more sharply across layers.
  PAGED_MOE_LFRU            1 (default) = frequency-protected eviction. A
                                cold entry at the LRU tail is evicted; a
                                frequently used one gets a second chance (its
                                counter halves and it moves to MRU). Stops one
                                sweep of prefill/decode traffic from flushing
                                the experts every token needs.
  PAGED_MOE_LFRU_SCAN       max second chances granted per eviction. Default 8.
  PAGED_MOE_STAGING_GB      cap on prefetched-but-unused numpy staging.
                                Default: auto (max(1 GB, 256 experts' worth,
                                capped at 6 GB)). Must comfortably hold one
                                prediction horizon (PREDICT_DEPTH layers' worth
                                of experts), otherwise predictions are dropped
                                before the layer that wanted them runs and get
                                re-read as misses - a fixed 1 GB caused ~100
                                such drops per decoded token on Qwen3-235B
                                (10.6 MB experts).
  PAGED_MOE_CLEAR_GB        bytes evicted between mx.clear_cache() calls.
                                Default: auto (cache budget / 3, floor 2 GB).
                                Big-expert models evict multiple GB per decoded
                                token; clearing too often flushes Metal's
                                recycled-buffer pool and every miss then pays
                                fresh zero-fill page faults.
  PAGED_MOE_SLAB            "auto" (default, = on wherever the model's
                                experts are uniformly shaped) | "on" | "off".
                                Holds resident experts as slots in contiguous
                                Metal-backed tensors, so reads pread into their
                                final location (no host->device copy) and a
                                layer computes with one gather_qmm per
                                projection instead of 3 calls per expert.
                                Measured +11% on Qwen3-235B and +20% on
                                qwen3-next. See docs/DESIGN.md "Slab storage".
  PAGED_MOE_SLAB_GB         hard slab reservation. Unlike the expert cache's
                                soft budget this is committed at load, so it is
                                given back for big prefills and rebuilt for
                                decode. Default: the expert-cache budget.
  PAGED_MOE_SLAB_VERIFY     1 = check at compute time that no slot a layer
                                reads is being written and that slot ownership
                                agrees. Slot reuse overwrites bytes in place,
                                so a double-owned slot means wrong output
                                rather than an exception - use this first when
                                debugging slab correctness.
  PAGED_MOE_PRUNE           opt-in decode "turbo": drop a routed expert
                                whose router weight is below this fraction of
                                the strongest selected expert's weight (0 =
                                off, the default; output stays bit-identical
                                to resident inference). Measured on
                                Qwen3-235B-A22B 4-bit: 0.3 prunes ~23-29% of
                                slots (2.6 -> 2.9-3.0 tok/s), 0.5 prunes ~55%
                                (-> ~4 tok/s). Output stays coherent but is no
                                longer bit-identical. Only affects decode, and
                                only Qwen3-MoE-style plain-logits routers.
  PAGED_MOE_GROUP_MB        max bytes of expert weights materialized per
                                prefill group (bounds transient memory while
                                whole layers stream through). Default 1024.
  PAGED_MOE_PREFILL_RUN_MB / _SLICE_MB
                                prefill read shape: longest contiguous
                                multi-expert span, and the pread granularity
                                inside it. Defaults 512 / 8. The run cap must
                                exceed one expert or coalescing cannot happen.
  PAGED_MOE_PREFILL_CACHE_GB
                                expert-cache budget while a big prefill chunk is
                                being processed. Default 4. A prefill pass reads
                                the whole uncached expert mass, and within one
                                pass a big cache buys nothing - so giving that
                                memory to activations instead (which is what
                                allows one-pass prefill) is a large net win.
  PAGED_MOE_PREFILL_CHUNK   prompt tokens per prefill pass, used as the
                                default for the server and CLI. Default 32768:
                                a 32k-token agent prompt then streams the expert
                                mass exactly once instead of four times.
                                With adaptive prefill on, this is the *max*
                                chunk (per-step size shrinks with KV length).
  PAGED_MOE_FUSED_PREFILL   1 (default) = sub-chunk attention inside each
                                decoder layer and run the MoE once per chunk,
                                so a long prompt streams the expert mass once
                                instead of once per attention chunk.
  PAGED_MOE_ADAPTIVE_PREFILL
                                1 = resize the prefill step each pass instead
                                of using one fixed PREFILL_CHUNK.
  PAGED_MOE_ADAPTIVE_PREFILL_MODE
                                auto|dsa|fused|off. auto picks dsa when the
                                model has an indexer / deepseek_v32-class
                                attention, else fused.
  PAGED_MOE_ADAPTIVE_PREFILL_MAX / _SAFETY
                                chunk ceiling and Metal-buffer safety factor
                                (default PREFILL_CHUNK / 0.75).
  PAGED_MOE_RESERVE_CTX     context length (tokens) to hold back KV memory
                                for when sizing the expert cache. The host app sets this
                                from the model's numCtx. Full-attention models
                                need it: GLM-4.7 spends 196 KB/token at 8 bits,
                                so 24k tokens is 4.8 GB the expert cache must
                                not have committed to a slab. 0 = use the flat
                                headroom allowance (right only for cheap-KV
                                models like Qwen3-235B at 102 KB/token).
  PAGED_MOE_KV_BITS         decode-time KV cache precision. Default 8.
  PAGED_MOE_KV_STORE_SLACK  multiplier on reserved KV bytes covering the
                                prompt-cache store next to the live cache.
                                Default 1.25.
  PAGED_MOE_PROMPT_CACHE_MOVE
                                1 (default) = hand a reused prompt cache to the
                                request instead of deep-copying it, which is a
                                whole extra KV cache per turn (4.8 GB on GLM-4.7
                                at 24k). 0 restores mlx-lm's copying, which
                                survives a request that dies mid-generation.
  PAGED_MOE_KV_LAZY         1 (default) = resume a reused 8-bit KV cache
                                without an fp16 copy (dequantized a layer at a
                                time inside attention). Same logits. 0 = the
                                full dequantize, which makes the slab yield
                                every turn on a long conversation.
  PAGED_MOE_HYBRID_CKPT     1 (default) = keep the recurrent state at each
                                turn's resume point and mid-snap so a hybrid
                                cache can rewind a longer stored entry there.
                                _MAX (8) and _MB (1024) bound the count and
                                bytes; _MIN_TOKENS (1024) skips short prefixes.
  PAGED_MOE_HYBRID_INPASS_SNAP
                                1 (default) = take the mid-snap inside the
                                prefill pass, saving the second chunk for the
                                generation prompt. 0 = stop at it as before.
  PAGED_MOE_PREFIX_DISK     1 (default) = write the pinned front matter KV
                                to disk and load it on a cold start whose
                                prompt starts with it. _DIR (next to the
                                sidecar dir), _MODELS (3), _PER_MODEL (2) and
                                _MAX_GB (8) bound what is kept. On hybrid
                                models _CKPT_EVERY (2048) and _CKPT_MAX (4)
                                place recurrent checkpoints inside a file, so
                                an edited system prompt resumes part way.
                                _KEY_FD seals every file with a key the host
                                writes to that fd (seal.py).
  PAGED_MOE_VERIFY          1 (default) = exact-verify decode on streamed
                                MoE: n-gram and self drafts are checked with
                                one pass, and a timing governor only drafts
                                where it measured faster. 0 = mlx-lm's plain
                                loop. _SPANS, _TOKENS, _LONG_TOKENS, SELF_DRAFT,
                                DRAFT_PRUNE, DRAFT_WAIT_ABOVE tune it.
                                _TRUNCATE (1) sizes passes by row read cost,
                                _REPLAY (1) prefetches copied routing.
  PAGED_MOE_TOOL_WARM       1 (default) = prefetch experts seen in tool-call
                                bodies (quality-safe). _GB (2), _SKIP (24),
                                _MIN_TOKENS (32), _DECAY (0.85). 0 turns it off.
  PAGED_MOE_SIDECAR         0 (default) = offline. 1 = online expert-prefetch
                                sidecar: a tiny per-model learner that guesses
                                which experts the *next* token's early layers
                                will need and prefetches them. Quality-safe
                                (prefetch only; real router still decides).
                                When off, none of the sidecar code runs on the
                                decode path.
  PAGED_MOE_SIDECAR_DEBUG   0 (default) = quiet. 1 = print [sidecar] status
                                lines (also enabled by PAGED_MOE_DEBUG=1).
  PAGED_MOE_SIDECAR_WRAP    early MoE layers to prefetch for the next token.
                                Default 0 = scale from the MoE layer count.
  PAGED_MOE_SIDECAR_TOPK    experts predicted per wrap layer. Default 0 =
                                track the router's observed fan-out per layer.
  PAGED_MOE_SIDECAR_TOPK_FACTOR
                                multiple of that fan-out to aim for when TOPK
                                is auto. Default 1.0.
  PAGED_MOE_SIDECAR_MIN_SCORE
                                softmax probability floor; drop guesses below
                                this even inside top-k. Default 0.04.
  PAGED_MOE_SIDECAR_HISTORY how many prior tokens feed the feature vector.
                                Default 6.
  PAGED_MOE_SIDECAR_FEAT_DIM
                                feature width. Default 0 = scale from expert
                                and layer counts.
  PAGED_MOE_SIDECAR_HIDDEN  1 (default) = project the last hidden state into
                                the features. It is what the LM head turns into
                                the next token, so it is the strongest single
                                predictor of that token's routing; history alone
                                can only model expert co-occurrence.
  PAGED_MOE_SIDECAR_SKETCH_DIM
                                width of that projection. Default 64.
  PAGED_MOE_SIDECAR_ADAGRAD 1 (default) = per-parameter adaptive step. Sparse
                                hashed features converge far faster with it.
  PAGED_MOE_SIDECAR_WD      weight decay on the shadow weights. Default 1e-5.
  PAGED_MOE_SIDECAR_PRIOR_W blend weight for the per-layer expert-popularity
                                prior while the head is cold, decaying to 0 over
                                PRIOR_DECAY train steps. Default 0.5 / 20000.
  PAGED_MOE_SIDECAR_MIN_GAIN
                                marginal recall a head must add on top of the
                                engine's own last-token replay to go (or stay)
                                live. Default 0.02. This, not absolute recall,
                                is what says the head is earning its reads.
  PAGED_MOE_SIDECAR_EXTRA_BUDGET
                                cap on extra experts issued per token.
                                Default 0 = wrap * topk.
  PAGED_MOE_SIDECAR_TIME_BUDGET_MS / _MAX_OVERHEAD
                                the sidecar runs inline between tokens, so its
                                own time is token latency. Past either bound it
                                sheds work (shadow scoring, training stride,
                                top-k, then actuation). Default 4.0ms / 0.05.
  PAGED_MOE_SIDECAR_SAVE_EVERY
                                persist every N decode tokens so a kill never
                                costs a whole session. Default 512.
  PAGED_MOE_SIDECAR_PERSIST_IDLE_S
                                saves run on a background thread once the
                                engine has been idle this long, never on a
                                request. Default 2.0.
  PAGED_MOE_SIDECAR_LOCK_HITS
                                consecutive good end_token checks before lock.
                                Default 0 = never auto-lock (early lock froze
                                mediocre adapters). Set e.g. 64 to opt in.
  PAGED_MOE_SIDECAR_LOCK_MIN_TOKENS
                                refuse lock until this many tokens seen.
                                Default 4096.
  PAGED_MOE_SIDECAR_LOCK_OVER_CEILING
                                live recall must beat same-layer ceiling by
                                this margin to lock. Default 0.15.
  PAGED_MOE_SIDECAR_MIN_RECALL
                                rolling recall@topk to promote / lock. Default
                                0.40. Only used before MIN_GAIN has samples.
  PAGED_MOE_SIDECAR_MIN_PRECISION
                                rolling precision@topk to promote / lock /
                                avoid SSD thrash. Default 0.22.
  PAGED_MOE_SIDECAR_DIR     where per-model adapter slots are persisted.
                                Default ~/.paged_moe/sidecar.
                                Override to keep weights under another root.
  PAGED_MOE_SIDECAR_HEAD_WRAP
                                1 (default when SIDECAR on) = next-token wrap
                                prefetch head.
  PAGED_MOE_SIDECAR_HEAD_PREFILL
                                0 (default) = off. 1 = prefill hot-expert
                                prefetch / post-leave warm (quality-safe).
  PAGED_MOE_SIDECAR_PREFILL_HOT_FRAC
                                a chunk's expert union is nearly every expert,
                                so the head targets its HOT set instead: experts
                                whose per-chunk token count reaches this
                                fraction of the layer's busiest. Default 0.5.
  PAGED_MOE_SIDECAR_HEAD_RESIDENCY
                                0 (default) = off. 1 = eviction protect scores
                                (quality-safe).
  PAGED_MOE_SIDECAR_RESIDENCY_MAX_PROTECT
                                cap on the share of scanned eviction candidates
                                that may be protected, so a uniformly hot cache
                                still makes progress. Default 0.10 - at 0.33 the
                                head stopped selecting and merely perturbed
                                LRU's victim order, measured +55% misses.
  PAGED_MOE_SIDECAR_RESIDENCY_MIN_HITS
                                separate demands on a key before its reuse rate
                                may override LRU. Default 8: 88% of just-used
                                experts are reused inside the horizon, so a rate
                                from one or two observations is noise.
  PAGED_MOE_SIDECAR_GOVERNOR
                                1 (default) = actuation is A/B'd against itself
                                on measured decode rate and only stays on while
                                it wins. This is what makes "the sidecar cannot
                                make decode slower" a property rather than a
                                hope; recall-style metrics cannot deliver it
                                because they have no cost term. 0 = actuate
                                unconditionally (A/B rigs only).
  PAGED_MOE_SIDECAR_GOV_WINDOW / _MIN_TOKENS
                                tokens per A/B window, and tokens per arm before
                                a verdict. Default 16 / 256. The error bar
                                shrinks with the number of *windows*, so small
                                windows and a high token floor is the cheap way
                                to resolve a small effect.
  PAGED_MOE_SIDECAR_GOV_MARGIN / _Z
                                how much faster actuation must measure (ratio,
                                default 1.03) and how many error bars the gap
                                must clear (default 3.0) to switch ON. Switching
                                OFF needs only half the Z: stopping a
                                speculation is cheap to get wrong, starting one
                                is not.
  PAGED_MOE_SIDECAR_GOV_HOLD / _PROBE_DUTY / _MAX_PROBE
                                tokens a verdict holds (512), how often re-probes
                                sample the losing side (1 window in 8, widening
                                as the verdict is confirmed), and how far past
                                the minimum sample an inconclusive round runs
                                before calling a tie (24x). Ties settle OFF.
  PAGED_MOE_SIDECAR_MIN_PROB
                                minimum calibrated probability for a speculative
                                read. Default 0.35. Volume tracks confidence
                                rather than a fixed top-k, so an unsure head
                                ships nothing instead of k reads per layer.
  PAGED_MOE_SIDECAR_SPEC_BYTE_FRAC
                                speculative bytes per token as a share of the
                                demand-miss bytes the same token paid for.
                                Default 0.25. Bounds the worst case before any
                                learning has happened, and scales itself across
                                models whose experts differ by 6x in size.
  PAGED_MOE_SIDECAR_TARGET_PRECISION
                                precision the issue threshold servos toward.
                                Default 0.5.
  PAGED_MOE_SIDECAR_HEAD_PRUNE
                                0 (default) = off. 1 = adaptive prune/wait
                                (quality-affecting; shadow->promote).
  PAGED_MOE_SIDECAR_PRUNE_MIN_MASS
                                fraction of the *baseline* threshold's retained
                                mixture mass the shadow prune policy must match
                                before it may go live. Default 0.98.
  PAGED_MOE_SIDECAR_MAX_WASTE
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


MODELS_DIR: str = _env("PAGED_MOE_DOWNLOAD_DIR", _default_models_dir(), str)

# "auto" | "resident" | "streamed"
MODE: str = _env("PAGED_MOE_MODE", "auto", str)

# With F_NOCACHE reads there is no hidden page-cache growth, so the engine can
# safely claim more than half of RAM for backbone + expert cache.
#
# 0.70 did OOM a 48 GB machine once - but that was before F_NOCACHE, when every
# expert byte landed in the kernel's page cache *as well as* the expert LRU. The
# pile-on that caused it is gone, and the wired limit is now forced to 0, so the
# budget is no longer fighting an invisible second copy of the checkpoint.
RAM_FRACTION: float = _env("PAGED_MOE_RAM_FRACTION", 0.68, float)

# None means "auto": min(RAM_FRACTION * RAM - backbone - headroom, MAX_CACHE_GB)
CACHE_GB: float | None = _env("PAGED_MOE_CACHE_GB", None, float)

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
# env var away (PAGED_MOE_CACHE_GB=32), but on a 48 GB machine it leaves
# little room for a long KV cache plus an editor and a browser.
MAX_CACHE_GB: float = _env("PAGED_MOE_MAX_CACHE_GB", 26.0, float)

# That budget is fixed at load, and other apps grow. When the kernel reports
# memory pressure, macOS compresses the slab's idle slots, which costs every
# app on the machine CPU and leaves those slots no faster than disk. So each
# slab rebuild (after a prefill) steps the slab down a STEP under pressure and
# back up half a STEP once it has been calm for SLAB_REGROW_CALM_S with the
# compressor flat and free memory for the extra slots, never below FLOOR of
# the wanted size. See mempressure.py.
MEM_PRESSURE: bool = bool(_env("PAGED_MOE_MEM_PRESSURE", 1, int))
MEM_PRESSURE_STEP: float = _env("PAGED_MOE_MEM_PRESSURE_STEP", 0.15, float)
MEM_PRESSURE_FLOOR: float = _env("PAGED_MOE_MEM_PRESSURE_FLOOR", 0.5, float)
MEM_PRESSURE_POLL_S: float = _env("PAGED_MOE_MEM_PRESSURE_POLL_S", 0.5, float)

# A slab only changes size when it is rebuilt, and in an agent session with a
# prefix cache that is the first big prefill and then never. So a slab that
# pressure (or a crowded rebuild) left small stayed small all day after the
# other app was gone. At the start of a turn, once the kernel has reported
# normal for SLAB_REGROW_CALM_S, the slab is rebuilt to what fits now: the
# load-time size at most, the RAM budget next to everything else live, and
# what the kernel can hand out without compressing anyone, less the usual
# margin. Rebuilt before that turn's prefill, so the prefill refills it; the
# one-time cost is the hits the old slab would have served. Only when the gain
# is at least 10% of the slab and 1 GB.
SLAB_REGROW: bool = bool(_env("PAGED_MOE_SLAB_REGROW", 1, int))
SLAB_REGROW_CALM_S: float = _env("PAGED_MOE_SLAB_REGROW_CALM_S", 60.0, float)

# Bypass the OS page cache for expert reads (macOS F_NOCACHE).
NOCACHE: bool = bool(_env("PAGED_MOE_NOCACHE", 1, int))

# F_NOCACHE leaks (pagecache.py). With NOCACHE on, drop the checkpoint's
# cached pages whenever the system file cache has grown TRIM_MB since the
# last drop. Checked every POLL_S with one sysctl.
PAGECACHE_TRIM: bool = bool(_env("PAGED_MOE_PAGECACHE_TRIM", 1, int))
PAGECACHE_TRIM_MB: int = _env("PAGED_MOE_PAGECACHE_TRIM_MB", 512, int)
PAGECACHE_TRIM_POLL_S: float = _env("PAGED_MOE_PAGECACHE_TRIM_POLL_S", 0.25, float)

READ_THREADS: int = _env("PAGED_MOE_READ_THREADS", 16, int)

# Slot-read pool workers (ExpertCache._ReadPool): decode miss queue depth.
# Miss threads push reads here directly. disk_ceiling.py saturates ~qd16 on
# 3 MB requests and still gains at qd64 on 0.2 MB scales, so default wants
# deeper than READ_THREADS.
# 0 = auto (2x READ_THREADS); -1 disables and restores the ThreadPoolExecutor path.
READ_POOL_THREADS: int = _env("PAGED_MOE_READ_POOL_THREADS", 0, int)

# Native C pread pool (one job per expert, GIL released for the whole latch).
# Bit-identical to the Python _ReadPool: same fds/offsets into the same slab
# slots. 1 / auto = try native, silently fall back to Python on any load or
# compile failure. 0 = force the Python pool. Default on for streamed models.
def _native_read_mode(raw) -> str:
    v = str(raw).strip().lower()
    if v in ("0", "off", "false", "no"):
        return "off"
    if v in ("1", "on", "true", "yes", "auto"):
        return "on"
    return "on"

NATIVE_READ: str = _native_read_mode(_env("PAGED_MOE_NATIVE_READ", "1", str))

# Native pool piece size. Workers share one expert's reads in pieces of this
# many KB, so a single demand miss gets the drive's queue depth instead of one
# thread's QD1 (MiniMax 32 MB experts: ~9 ms per expert at QD1).
# 0 = one worker per expert (the old schedule).
READ_SPLIT_KB: int = _env("PAGED_MOE_READ_SPLIT_KB", 1024, int)
# 1 = a speculative read a layer starts waiting on moves to the native pool's
# urgent queue. Otherwise it waits behind every guess queued before it.
READ_PROMOTE: bool = bool(_env("PAGED_MOE_READ_PROMOTE", 1, int))
# While a decode layer waits on the drive, run a tiny GPU op this often (us).
# An idle Apple GPU drops clocks within about a ms, and the next layer's graph
# then runs ~2x slower (235B layer graph: 0.75 ms busy, 1.75 ms after a 1.5 ms
# idle gap). Values are untouched. 0 = off.
GPU_KEEPALIVE_US: int = _env("PAGED_MOE_GPU_KEEPALIVE_US", 250, int)
# 1 = a prefill hit refreshes the expert's recency and frequency like a decode
# hit. Prefill routes to nearly every resident expert, so that reorders the
# whole cache by prefill layer order and hands the last turn's experts
# protection the next turn's decode set never earned. Default 0.
PREFILL_TOUCH: bool = bool(_env("PAGED_MOE_PREFILL_TOUCH", 0, int))
# At load, read the GPU driver's in-use memory and, when far above normal
# (mempressure.gpu_held_excess), plan a streamed model's budget without it.
GPU_HELD_CHECK: bool = bool(_env("PAGED_MOE_GPU_HELD_CHECK", 1, int))
# Watchdog thread (watchdog.py): a request open with no layer progress
# for STALL_S exits 70 (hung Metal wait); kernel pressure CRITICAL for
# PRESSURE_S straight during a request exits 71 (compressor eating the
# slab). HEARTBEAT_S between "alive" log lines. 0 disables that check.
# Every streamed layer step touches, and a slow one is a few seconds, so
# 120 s of nothing is a wedge, not a slow drive.
WATCHDOG: bool = bool(_env("PAGED_MOE_WATCHDOG", 1, int))
WATCHDOG_STALL_S: float = _env("PAGED_MOE_WATCHDOG_STALL_S", 120.0, float)
WATCHDOG_PRESSURE_S: float = _env("PAGED_MOE_WATCHDOG_PRESSURE_S", 45.0, float)
WATCHDOG_HEARTBEAT_S: float = _env("PAGED_MOE_WATCHDOG_HEARTBEAT_S", 300.0, float)
# While macOS reports a thermal warning or a CPU speed limit, decode waits this
# long after each token so the GPU gets gaps to cool. Timing only. 0 turns it off.
THERMAL_PACE_MS: float = _env("PAGED_MOE_THERMAL_PACE_MS", 30.0, float)
# Before a stall exit, macOS `sample` writes the native stacks (the Metal
# wait the Python dump cannot see) here. Empty means the temp dir.
INCIDENT_DIR: str = _env("PAGED_MOE_INCIDENT_DIR", "", str)
WATCHDOG_SAMPLE_S: int = _env("PAGED_MOE_WATCHDOG_SAMPLE_S", 3, int)
# On a stall, also save the macOS unified log's GPU driver lines (AGX
# faults, restarts, context switch timeouts) from the minutes before it.
WATCHDOG_OSLOG: bool = bool(_env("PAGED_MOE_WATCHDOG_OSLOG", 1, int))
# GPU breadcrumbs (gpu_crumbs.py): a tiny kernel after a point of interest,
# so a stall names the last point the GPU finished. Bit-identical; a dispatch
# and a commit per crumb. 1 = the steps of each token (logits, processors,
# logprobs, sampled token), about 50 us a token. 2 = also every MoE layer's
# input and output, about 100 crumbs a token, for chasing one hang. 0 = off.
GPU_CRUMBS: int = _env("PAGED_MOE_GPU_CRUMBS", 1, int)
# At start, end other paged-moe servers whose parent process is gone.
LEFTOVER_KILL: bool = bool(_env("PAGED_MOE_LEFTOVER_KILL", 1, int))

# Paged PLE n-gram rows (ple_offload.py, qwen4_exp only). Raw quantized rows
# kept in host RAM, LRU. A row is ~140 bytes, so the default is ~37 MB.
# 0 turns the cache off (every lookup reads the disk).
PLE_ROW_CACHE: int = _env("PAGED_MOE_PLE_ROW_CACHE", 262144, int)
# Reader threads for PLE row misses. Prefill touches ~16 rows per prompt token.
PLE_READ_THREADS: int = _env("PAGED_MOE_PLE_READ_THREADS", 8, int)
# 0 = the old per-shard gather (serial reads, one sync per shard). Same rows,
# same output; kept for A/B.
PLE_FAST: bool = bool(_env("PAGED_MOE_PLE_FAST", 1, int))

# Early-decode warm pool (independent of PAGED_MOE_SIDECAR). Prefetch-only
# actuation; online frequency training from the first K decode tokens.
EARLY_DECODE: bool = bool(_env("PAGED_MOE_EARLY_DECODE", 0, int))
EARLY_DECODE_TRAIN: bool = bool(_env("PAGED_MOE_EARLY_DECODE_TRAIN", 0, int))
EARLY_DECODE_TOKENS: int = _env("PAGED_MOE_EARLY_DECODE_TOKENS", 64, int)
EARLY_DECODE_GB: float = _env("PAGED_MOE_EARLY_DECODE_GB", 4.0, float)
# Hold prefetch when EMAs fall below these after MIN_EPISODES (still trains).
EARLY_DECODE_MIN_COVER: float = _env("PAGED_MOE_EARLY_DECODE_MIN_COVER", 0.12, float)
EARLY_DECODE_MIN_PREC: float = _env("PAGED_MOE_EARLY_DECODE_MIN_PREC", 0.15, float)
EARLY_DECODE_MIN_EPISODES: int = _env("PAGED_MOE_EARLY_DECODE_MIN_EPISODES", 4, int)
# Consecutive below-bar / above-bar episodes to hold or resume actuation.
EARLY_DECODE_HOLD_STREAK: int = _env("PAGED_MOE_EARLY_DECODE_HOLD_STREAK", 2, int)
EARLY_DECODE_CLEAR_STREAK: int = _env(
    "PAGED_MOE_EARLY_DECODE_CLEAR_STREAK", 2, int
)
# Per-episode multiply on the count table (1 = no decay). Softens stale domains.
EARLY_DECODE_DECAY: float = _env("PAGED_MOE_EARLY_DECODE_DECAY", 0.97, float)
# Cold-start bootstrap (EARLY_DECODE on, TRAIN off, no existing pool weights):
# train until cover/prec EMAs hit these high bars, then freeze. Explicit TRAIN
# never auto-stops. Bars are intentionally high ("max gains") so a cold Mac
# gets a strong first plan before settling to prefetch-only.
EARLY_DECODE_BOOTSTRAP_COVER: float = _env(
    "PAGED_MOE_EARLY_DECODE_BOOTSTRAP_COVER", 0.40, float
)
EARLY_DECODE_BOOTSTRAP_PREC: float = _env(
    "PAGED_MOE_EARLY_DECODE_BOOTSTRAP_PREC", 0.22, float
)
EARLY_DECODE_BOOTSTRAP_MIN_EPISODES: int = _env(
    "PAGED_MOE_EARLY_DECODE_BOOTSTRAP_MIN_EPISODES", 48, int
)
EARLY_DECODE_BOOTSTRAP_MAX_EPISODES: int = _env(
    "PAGED_MOE_EARLY_DECODE_BOOTSTRAP_MAX_EPISODES", 256, int
)
# Plateau window: if cover EMA moves < this over the last N episodes *after*
# MIN_EPISODES and cover is already near the bootstrap bar, treat as maxed.
EARLY_DECODE_BOOTSTRAP_PLATEAU_N: int = _env(
    "PAGED_MOE_EARLY_DECODE_BOOTSTRAP_PLATEAU_N", 12, int
)
EARLY_DECODE_BOOTSTRAP_PLATEAU_EPS: float = _env(
    "PAGED_MOE_EARLY_DECODE_BOOTSTRAP_PLATEAU_EPS", 0.005, float
)
# Greppable time-to-warm A/B log: first decode through the early window.
# Always prints `[ttw] ...` (not gated on SIDECAR_DEBUG). Tail with:
#   grep '\[ttw\]' <server.log>
TTW_LOG: bool = bool(_env("PAGED_MOE_TTW_LOG", 1, int))
# 0 = same length as EARLY_DECODE_TOKENS (fair A/B window).
TTW_TOKENS: int = _env("PAGED_MOE_TTW_TOKENS", 0, int)

# Speculative inflight cap as a multiple of READ_THREADS. Too low leaves the
# drive idle (drive_idle.py: ~19% idle while speculation skips ~200x/token).
# 0 = auto.
SPEC_INFLIGHT_MULT: float = _env("PAGED_MOE_SPEC_INFLIGHT_MULT", 3.0, float)

# Slab slots held back from residency so in-flight reads and unclaimed
# guesses have somewhere to land. Same pool as the cache, so this trades hit
# rate for prefetch runway. Raising STAGING_GB alone does not buy runway:
# staging past the reserve starves reads of slots (Qwen3-235B: drive
# occupancy 80% -> 66%).
SLAB_RESERVE_FRAC: float = _env("PAGED_MOE_SLAB_RESERVE_FRAC", 0.1, float)

# Deep enough to hide NVMe latency behind a few layers of attention+MoE
# on big models (GLM-4.5-Air has 46 MoE layers).
PREFETCH_DEPTH: int = _env("PAGED_MOE_PREFETCH_DEPTH", 8, int)

# Route prediction: ask the next layers' (resident) routers what they will want,
# instead of assuming they want what they wanted for the previous token.
def _predict_mode(raw) -> str:
    v = str(raw).strip().lower()
    if v in ("0", "off", "false", "no"):
        return "off"
    if v in ("1", "on", "true", "yes"):
        return "on"
    return "auto"


PREDICT: str = _predict_mode(_env("PAGED_MOE_PREDICT", "auto", str))
PREDICT_WINDOW: int = _env("PAGED_MOE_PREDICT_WINDOW", 8, int)
PREDICT_MIN_TOKENS: int = _env("PAGED_MOE_PREDICT_MIN_TOKENS", 96, int)
PREDICT_MARGIN: float = _env("PAGED_MOE_PREDICT_MARGIN", 1.03, float)
PREDICT_Z: float = _env("PAGED_MOE_PREDICT_Z", 2.0, float)
PREDICT_MAX_PROBE: int = _env("PAGED_MOE_PREDICT_MAX_PROBE", 6, int)
PREDICT_HOLD_WINDOWS: int = _env("PAGED_MOE_PREDICT_HOLD", 128, int)
# "auto" starts settled ON (half a hold, then the normal probe) when the
# decode slab holds under this fraction of the expert mass and PRUNE is 0:
# the disk-bound exact case where prediction has always measured faster.
# 0 = always probe first.
PREDICT_PRIOR_RESIDENT: float = _env("PAGED_MOE_PREDICT_PRIOR_RESIDENT", 0.35, float)
PREDICT_DEPTH: int = _env("PAGED_MOE_PREDICT_DEPTH", 3, int)
# Aim the prediction window this many layers further ahead (layer i predicts
# layers i+lead+1 .. i+lead+depth). Lead time and burst size are separate
# knobs: see PrefetchRing.__init__.
PREDICT_LEAD: int = _env("PAGED_MOE_PREDICT_LEAD", 0, int)
# One extra prediction per layer, aimed this many layers ahead, for source
# layers past PREDICT_FAR_FROM of the stack. bench/router_matrix.py shows
# router agreement decays with SOURCE DEPTH rather than distance: on
# Qwen3-235B, precision among cache misses from layer 0 dies by distance 4
# (0.28) while from layer 24 it holds 0.73 at distance 32, against a net-win
# bar of 0.5. 0 (default) keeps the shipped uniform-window behaviour; try
# 16-32 on a disk-bound model, and let the prediction governor confirm it.
PREDICT_FAR_LEAD: int = _env("PAGED_MOE_PREDICT_FAR_LEAD", 0, int)
PREDICT_FAR_FROM: float = _env("PAGED_MOE_PREDICT_FAR_FROM", 0.17, float)
PREDICT_SLACK: int = _env("PAGED_MOE_PREDICT_SLACK", 2, int)
PREDICT_RENORM: bool = bool(_env("PAGED_MOE_PREDICT_RENORM", 0, int))
# When decode pruning is on, prediction prunes its guesses with the same
# router-weight test - but against threshold PRUNE * this factor, not PRUNE
# itself (1.0, the default, = the demand threshold; 0 = never prune guesses).
# Softer values keep borderline experts the demand path might rescue, but on
# Qwen3-235B every softening measured slower: 5.34 tok/s at 1.0 vs 5.14 at
# 0.8 vs 4.57 at 0.5 - extra coverage never paid for the extra queue traffic,
# because blocking reads already run near the drive's scattered ceiling and
# anything speculative just stands in their way.
PREDICT_PRUNE_SOFT: float = _env("PAGED_MOE_PREDICT_PRUNE_SOFT", 1.0, float)
# Install route-predicted slab reads straight into the LRU on arrival instead
# of staging them. Only pays when prediction precision is high (wrong installs
# evict resident experts - cache pollution); measured 15% slower at precision
# 0.5, so default off.
PREDICT_INSTALL: bool = bool(_env("PAGED_MOE_PREDICT_INSTALL", 0, int))

# Online expert-prefetch sidecar (v2). Off by default so models that don't
# opt in pay nothing. See pagedmoe/sidecar.py.
SIDECAR: bool = bool(_env("PAGED_MOE_SIDECAR", 0, int))
# 0 = auto (scaled from the model's MoE geometry at attach time).
SIDECAR_WRAP: int = _env("PAGED_MOE_SIDECAR_WRAP", 0, int)
SIDECAR_TOPK: int = _env("PAGED_MOE_SIDECAR_TOPK", 0, int)
# When TOPK is auto, aim for this multiple of the observed experts/layer/token.
SIDECAR_TOPK_FACTOR: float = _env("PAGED_MOE_SIDECAR_TOPK_FACTOR", 1.0, float)
SIDECAR_MIN_SCORE: float = _env("PAGED_MOE_SIDECAR_MIN_SCORE", 0.04, float)
SIDECAR_HISTORY: int = _env("PAGED_MOE_SIDECAR_HISTORY", 6, int)
SIDECAR_LOCK_HITS: int = _env("PAGED_MOE_SIDECAR_LOCK_HITS", 0, int)
SIDECAR_LOCK_MIN_TOKENS: int = _env(
    "PAGED_MOE_SIDECAR_LOCK_MIN_TOKENS", 4096, int
)
SIDECAR_LOCK_OVER_CEILING: float = _env(
    "PAGED_MOE_SIDECAR_LOCK_OVER_CEILING", 0.15, float
)
SIDECAR_MIN_RECALL: float = _env("PAGED_MOE_SIDECAR_MIN_RECALL", 0.40, float)
SIDECAR_MIN_PRECISION: float = _env(
    "PAGED_MOE_SIDECAR_MIN_PRECISION", 0.22, float
)
SIDECAR_WINDOW: int = _env("PAGED_MOE_SIDECAR_WINDOW", 32, int)
SIDECAR_LR: float = _env("PAGED_MOE_SIDECAR_LR", 0.08, float)
# 0 = auto (scaled from expert count / layer count).
SIDECAR_FEAT_DIM: int = _env("PAGED_MOE_SIDECAR_FEAT_DIM", 0, int)
SIDECAR_LOG_EVERY: int = _env("PAGED_MOE_SIDECAR_LOG_EVERY", 32, int)

# --- v4 learner -------------------------------------------------------------
# Per-parameter adaptive step (AdaGrad). Sparse hashed features converge much
# faster with it; costs one extra float32 array per trainable head.
SIDECAR_ADAGRAD: bool = bool(_env("PAGED_MOE_SIDECAR_ADAGRAD", 1, int))
SIDECAR_WD: float = _env("PAGED_MOE_SIDECAR_WD", 1e-5, float)
# Blend a per-layer expert-frequency prior into scores while the model is cold;
# weight decays to 0 over this many train steps.
SIDECAR_PRIOR_W: float = _env("PAGED_MOE_SIDECAR_PRIOR_W", 0.5, float)
SIDECAR_PRIOR_DECAY: int = _env("PAGED_MOE_SIDECAR_PRIOR_DECAY", 20000, int)
# Signed random projection of the last hidden state, appended to the features.
# The single strongest signal for next-token routing. 0 disables the block.
SIDECAR_SKETCH_DIM: int = _env("PAGED_MOE_SIDECAR_SKETCH_DIM", 64, int)
SIDECAR_HIDDEN: bool = bool(_env("PAGED_MOE_SIDECAR_HIDDEN", 1, int))

# --- v4 actuation / gating --------------------------------------------------
# Prefetch only what the engine's own last-token heuristic does NOT already
# cover, so the sidecar is strictly additive and its marginal value is
# measurable. Promote on that marginal recall gain rather than absolute recall.
SIDECAR_MIN_GAIN: float = _env("PAGED_MOE_SIDECAR_MIN_GAIN", 0.02, float)
# Ceiling on the share of extra reads that go unused. The wrap head shrinks its
# top-k while it is above this and grows back toward the observed router fan-out
# below it, so the cost/benefit knob tunes itself per model.
SIDECAR_MAX_WASTE: float = _env("PAGED_MOE_SIDECAR_MAX_WASTE", 0.85, float)
# Max extra experts issued per token (0 = auto: wrap * topk).
SIDECAR_EXTRA_BUDGET: int = _env("PAGED_MOE_SIDECAR_EXTRA_BUDGET", 0, int)

# --- v4 cost guards ---------------------------------------------------------
# The sidecar runs inline between tokens, so its own cost is token latency.
# Above either bound it sheds work (topk, then heads) and logs once.
SIDECAR_TIME_BUDGET_MS: float = _env(
    "PAGED_MOE_SIDECAR_TIME_BUDGET_MS", 4.0, float
)
SIDECAR_MAX_OVERHEAD: float = _env("PAGED_MOE_SIDECAR_MAX_OVERHEAD", 0.05, float)
# Persist every N decode tokens so a kill never costs a whole session.
SIDECAR_SAVE_EVERY: int = _env("PAGED_MOE_SIDECAR_SAVE_EVERY", 512, int)
# A save holds the GIL for ~0.5 s (json), which the read path would pay for.
SIDECAR_PERSIST_IDLE_S: float = _env(
    "PAGED_MOE_SIDECAR_PERSIST_IDLE_S", 2.0, float
)


def _default_sidecar_dir() -> str:
    return str(Path.home() / ".paged_moe" / "sidecar")


SIDECAR_DIR: str = _env("PAGED_MOE_SIDECAR_DIR", _default_sidecar_dir(), str)


def sync_from_environ() -> None:
    """Re-bind module-level knobs from the current process environment.

    Used by the mlx_lm drop-in hook when a matched model carries per-model
    ``PAGED_MOE_*`` overrides. Safe to call repeatedly; unknown attrs are
    skipped. Bool knobs expect ``0`` / ``1``.
    """
    g = globals()
    for name, cur in list(g.items()):
        if not name.isupper() or name.startswith("_"):
            continue
        env_name = "PAGED_MOE_DOWNLOAD_DIR" if name == "MODELS_DIR" else f"PAGED_MOE_{name}"
        raw = os.environ.get(env_name)
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
# hot-path work beyond a bool check on the facade.
SIDECAR_HEAD_WRAP: bool = bool(_env("PAGED_MOE_SIDECAR_HEAD_WRAP", 1, int))
SIDECAR_HEAD_PREFILL: bool = bool(
    _env("PAGED_MOE_SIDECAR_HEAD_PREFILL", 0, int)
)
SIDECAR_HEAD_RESIDENCY: bool = bool(
    _env("PAGED_MOE_SIDECAR_HEAD_RESIDENCY", 0, int)
)
SIDECAR_HEAD_PRUNE: bool = bool(_env("PAGED_MOE_SIDECAR_HEAD_PRUNE", 0, int))
# 0 = auto for both (scaled from MoE layer count / observed demand).
SIDECAR_PREFILL_LAYERS: int = _env("PAGED_MOE_SIDECAR_PREFILL_LAYERS", 0, int)
SIDECAR_PREFILL_TOPK: int = _env("PAGED_MOE_SIDECAR_PREFILL_TOPK", 0, int)
# A prefill chunk's expert *union* is nearly every expert, so predicting it is
# trivial and worthless. The head targets the chunk's HOT set instead: experts
# whose per-chunk token count is at least this fraction of the layer's max.
SIDECAR_PREFILL_HOT_FRAC: float = _env(
    "PAGED_MOE_SIDECAR_PREFILL_HOT_FRAC", 0.5, float
)
# --- net-time governor -----------------------------------------------------
# The sidecar's only unconditional promise is "not slower". Recall-style
# metrics cannot deliver it: a head can hold high recall while most of its
# speculative reads go unused, and on a disk-bandwidth-bound decode those
# unused reads come straight out of the demand path. Both benched models
# shipped in exactly that state (-24% and -13% tok/s at positive `gain`).
# So actuation is decided by an interleaved A/B on measured token time
# instead, and the heads only get to spend bandwidth while it is winning.
SIDECAR_GOVERNOR: bool = bool(_env("PAGED_MOE_SIDECAR_GOVERNOR", 1, int))
# Tokens per A/B window. Small, because the error bar on the comparison
# shrinks with the number of *windows*, not the number of tokens - and short
# windows also interleave the two states finely enough that both see the same
# mix of work. Matches the route-prediction governor's window for the same
# reason.
SIDECAR_GOV_WINDOW: int = _env("PAGED_MOE_SIDECAR_GOV_WINDOW", 16, int)
# Tokens per arm before a verdict is even considered. At the window above this
# is 16 windows per arm, enough for the error bar to resolve a ~20% effect,
# which is the size of the regressions actually measured here.
SIDECAR_GOV_MIN_TOKENS: int = _env("PAGED_MOE_SIDECAR_GOV_MIN_TOKENS", 256, int)
# How much faster actuation must measure before it is allowed to stay on, as a
# ratio (1.03 = 3%). Same convention as PAGED_MOE_PREDICT_MARGIN. Not 1.0:
# at 1.0, noise alone turns actuation on about half the time.
SIDECAR_GOV_MARGIN: float = _env("PAGED_MOE_SIDECAR_GOV_MARGIN", 1.03, float)
# Error bars the gap must clear to switch actuation ON. Switching OFF needs
# only half of this - stopping a speculation that may be costing throughput is
# cheap to get wrong, starting one is not.
SIDECAR_GOV_Z: float = _env("PAGED_MOE_SIDECAR_GOV_Z", 3.0, float)
# How far past the minimum sample an inconclusive A/B keeps probing before it
# is called a tie (a multiple of SIDECAR_GOV_MIN_TOKENS). Ties settle OFF: if
# the effect is still inside the noise after this much evidence, there is
# little to win and the configuration that spends no disk is the safe one.
SIDECAR_GOV_MAX_PROBE: float = _env("PAGED_MOE_SIDECAR_GOV_MAX_PROBE", 24.0, float)
# Tokens a verdict holds before the next probe round. Bounds the cost of being
# wrong after conditions change, and the cost of probing when they have not.
SIDECAR_GOV_HOLD: int = _env("PAGED_MOE_SIDECAR_GOV_HOLD", 512, int)
# After the first verdict, re-probes sample the losing configuration only one
# window in this many. An even re-probe would spend a quarter of all tokens in
# a state already measured slower, which is most of the regression this is
# supposed to remove.
SIDECAR_GOV_PROBE_DUTY: int = _env("PAGED_MOE_SIDECAR_GOV_PROBE_DUTY", 8, int)

# --- wrap issuance ---------------------------------------------------------
# Minimum calibrated probability for a speculative read. Volume tracks
# confidence rather than a fixed top-k, so an uncertain token issues little or
# nothing instead of always issuing k per layer.
SIDECAR_MIN_PROB: float = _env("PAGED_MOE_SIDECAR_MIN_PROB", 0.35, float)
# Speculative bytes per token as a share of the demand misses the same token
# paid for. Keeps speculation a bounded fraction of a cost we were already
# incurring, on the model's own current working set, at any cache size.
SIDECAR_SPEC_BYTE_FRAC: float = _env(
    "PAGED_MOE_SIDECAR_SPEC_BYTE_FRAC", 0.25, float
)
# Precision the issue threshold servos toward. At 1/2, a speculative read is
# expected to save more demand-path time than it costs.
SIDECAR_TARGET_PRECISION: float = _env(
    "PAGED_MOE_SIDECAR_TARGET_PRECISION", 0.5, float
)

SIDECAR_RESIDENCY_HORIZON: int = _env(
    "PAGED_MOE_SIDECAR_RESIDENCY_HORIZON", 32, int
)
SIDECAR_RESIDENCY_MIN_SCORE: float = _env(
    "PAGED_MOE_SIDECAR_RESIDENCY_MIN_SCORE", 0.35, float
)
# Never protect more than this share of scanned eviction candidates - a cache
# of uniformly "hot" keys must still make progress. Also sets the top slice of
# the observed score distribution the protect bar tracks. Lowered from 0.33:
# at a third of candidates the head stopped selecting and merely perturbed
# LRU's victim order, which measured +55% misses against leaving LRU alone.
SIDECAR_RESIDENCY_MAX_PROTECT: float = _env(
    "PAGED_MOE_SIDECAR_RESIDENCY_MAX_PROTECT", 0.10, float
)
# Minimum separate demands on a key before its reuse rate is allowed to
# override LRU. 88% of just-used experts are reused within the horizon, so a
# rate computed from one or two observations is noise, and LRU's recency
# ordering beats noise.
SIDECAR_RESIDENCY_MIN_HITS: int = _env(
    "PAGED_MOE_SIDECAR_RESIDENCY_MIN_HITS", 8, int
)
SIDECAR_PRUNE_MAX: float = _env("PAGED_MOE_SIDECAR_PRUNE_MAX", 0.95, float)
# Adaptive prune only goes live once the shadow policy retains at least this
# fraction of the mixture mass the shipped PRUNE threshold retains on the same
# tokens (quality guard, not just agreement with a regression target). Relative,
# not absolute: PRUNE=0.7 itself keeps well under half the mass on GLM and was
# measured fine, so an absolute bar would permanently veto the head.
SIDECAR_PRUNE_MIN_MASS: float = _env(
    "PAGED_MOE_SIDECAR_PRUNE_MIN_MASS", 0.98, float
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
PRUNE: float = _env("PAGED_MOE_PRUNE", 0.0, float)

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
ROUTE_TOP_K: int = _env("PAGED_MOE_ROUTE_TOP_K", 0, int)

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
ROUTE_TOP_P: float = _env("PAGED_MOE_ROUTE_TOP_P", 0.0, float)

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
KEEP_FREE: bool = bool(_env("PAGED_MOE_KEEP_FREE", 1, int))

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
WAIT_ABOVE: float = _env("PAGED_MOE_WAIT_ABOVE", 0.0, float)
# Record how much mixture mass approximate decode keeps by (PRUNE, WAIT_ABOVE),
# per model, under SIDECAR_DIR/prune_curves (prune_curve.py). Timing only.
PRUNE_CURVE: bool = bool(_env("PAGED_MOE_PRUNE_CURVE", 1, int))

# Start a decode layer's resident experts on the GPU before it blocks on its
# misses. The GPU otherwise idles for the whole read; the rows come out the
# same either way (one gather per row). GLM-5.2: 6.47 -> 6.74 tok/s.
HITS_EARLY: bool = bool(_env("PAGED_MOE_HITS_EARLY", 1, int))

# Spend the PRUNE / WAIT_ABOVE approximation where the reply tolerates it.
#
# The speed setting applies the same approximation to every decode token, but
# the tokens do not cost the same when they come out wrong. Inside a tool call
# one bad character fails the call, and the retry is a whole extra turn (a
# prefill pass over the expert mass plus decode). Right after a prefill the
# cache is cold, so WAIT_ABOVE skips the most experts exactly where the model
# decides how to start its turn. Everywhere else - reasoning, prose - a
# slightly rougher token costs next to nothing.
#
# With this on, tool-call spans (the tokenizer's own tool_call_start/end
# markers), the visible reply after a closing think marker (FIDELITY_REPLY; it
# is what the user reads, and dropped spaces or doubled letters show there),
# and the first FIDELITY_TURN_START decode tokens of a turn run with
# PRUNE and WAIT_ABOVE scaled by FIDELITY_SCALE, the live prune head's extra
# aggression is withheld there. Other tokens are unchanged, and so are tokens
# one-pass (flow) decode already owns. At PRUNE 0 and WAIT_ABOVE 0 it does nothing, so the
# exact setting stays bit-identical.
FIDELITY: bool = bool(_env("PAGED_MOE_FIDELITY", 1, int))
FIDELITY_SCALE: float = _env("PAGED_MOE_FIDELITY_SCALE", 0.6, float)
# Scale inside tool-call spans only. A pruned token there is not a rough word,
# it is a wrong oldString or a misspelled `...existing code...` marker, and the
# tool then edits the file wrong. 0 runs those tokens on every routed expert,
# which roughly halves tok/s for the call.
# Unset follows FIDELITY_SCALE.
FIDELITY_TOOL_SCALE: float = _env("PAGED_MOE_FIDELITY_TOOL_SCALE", FIDELITY_SCALE, float)
#
# TURN_START is 64, not 12: a thinking model reads the task back to itself in
# its first sentence of reasoning, and everything after builds on that. Flash
# at PRUNE 0.8 with the prune head opened one turn with "The user's message is
# empty" on a full prompt. 12 tokens covered the think marker and a few words.
FIDELITY_TURN_START: int = _env("PAGED_MOE_FIDELITY_TURN_START", 64, int)
FIDELITY_REPLY: bool = bool(_env("PAGED_MOE_FIDELITY_REPLY", 1, int))
# Degenerate-decode guard (fidelity.py). Approximate configs only; a trip
# holds the next GUARD_TOKENS decode exact, GUARD_STICKY trips hold the rest
# of the turn. Trips on a run of GUARD_RUN identical tokens, or a full
# GUARD_WINDOW with at most GUARD_DISTINCT distinct tokens. Qwen3-235B at
# PRUNE 0.88 + prune head wrote ";\n\n" lines into its reasoning (2 distinct).
FIDELITY_GUARD: bool = bool(_env("PAGED_MOE_FIDELITY_GUARD", 1, int))
FIDELITY_GUARD_RUN: int = _env("PAGED_MOE_FIDELITY_GUARD_RUN", 12, int)
FIDELITY_GUARD_WINDOW: int = _env("PAGED_MOE_FIDELITY_GUARD_WINDOW", 32, int)
FIDELITY_GUARD_DISTINCT: int = _env("PAGED_MOE_FIDELITY_GUARD_DISTINCT", 3, int)
FIDELITY_GUARD_TOKENS: int = _env("PAGED_MOE_FIDELITY_GUARD_TOKENS", 256, int)
FIDELITY_GUARD_STICKY: int = _env("PAGED_MOE_FIDELITY_GUARD_STICKY", 2, int)
# Loop rescue (rescue.py). Speed slider RESCUE_UP_TO and under, read off the
# model's kept-mass table (prune_curve.position). Until it has one: PRUNE >=
# RESCUE_MIN_PRUNE or WAIT_ABOVE >= RESCUE_MIN_WAIT, about 15 on a 0-20 speed
# slider (0.88 / 0.43 at 15 and 0.87 / 0.42 at 16). Output that starts
# copying itself (RESCUE_GRAM tokens seen again within RESCUE_MAX_PERIOD) is
# held back; a copy that runs a full period and RESCUE_CONFIRM tokens is
# dropped and decoded again exact from the last shown token, for
# RESCUE_TOKENS (doubling per trip, the rest of the turn from RESCUE_STICKY
# trips). RESCUE_NUDGE keeps the first redone prose token off the loop's.
# Calibrated on GLM-5.2's tokenizer: 0 trips in 140k tokens of agent prose,
# holds of ~1 token 5.7 times per 1k; repeated lines in code and JSON trip
# 0.02-0.16 per 1k and cost an exact redo.
RESCUE: bool = bool(_env("PAGED_MOE_RESCUE", 1, int))
RESCUE_UP_TO: float = _env("PAGED_MOE_RESCUE_UP_TO", 0.15, float)
RESCUE_MIN_PRUNE: float = _env("PAGED_MOE_RESCUE_MIN_PRUNE", 0.875, float)
RESCUE_MIN_WAIT: float = _env("PAGED_MOE_RESCUE_MIN_WAIT", 0.425, float)
RESCUE_GRAM: int = _env("PAGED_MOE_RESCUE_GRAM", 4, int)
RESCUE_MAX_PERIOD: int = _env("PAGED_MOE_RESCUE_MAX_PERIOD", 48, int)
RESCUE_CONFIRM: int = _env("PAGED_MOE_RESCUE_CONFIRM", 12, int)
RESCUE_TOKENS: int = _env("PAGED_MOE_RESCUE_TOKENS", 32, int)
RESCUE_STICKY: int = _env("PAGED_MOE_RESCUE_STICKY", 5, int)
RESCUE_MAX_TRIPS: int = _env("PAGED_MOE_RESCUE_MAX_TRIPS", 6, int)
RESCUE_NUDGE: bool = bool(_env("PAGED_MOE_RESCUE_NUDGE", 1, int))
# Garbled text does not repeat. Tokens go out RESCUE_LAG behind the decode
# (tool calls, think markers and EOS flush), so the start of a collapse is
# still unshown when a watcher sees it. RESCUE_SHAPES also watches prose for
# copies with every bare number one symbol (counting). RESCUE_NUM_MIN bare
# numbers in RESCUE_NUM_WINDOW prose tokens is how both GLM-5.2 collapses
# started (7-8 tokens in); agent prose sets it off ~1.2 times per 1k, number
# heavy notes ~5. It asks the exact model (RESCUE_AUDIT): one pass over up to
# RESCUE_AUDIT_MAX unshown tokens from the first of those numbers, cut at the
# first it gives less than RESCUE_AUDIT_P, exact for RESCUE_AUDIT_TOKENS from
# there. A clean audit keeps the watch quiet for RESCUE_AUDIT_COOL tokens,
# doubling. An audited token costs about what an exact one does (GLM-5.2 at
# 0.9 / 0.45: 13 audited tokens in 7 s).
RESCUE_LAG: int = _env("PAGED_MOE_RESCUE_LAG", 12, int)
RESCUE_SHAPES: bool = bool(_env("PAGED_MOE_RESCUE_SHAPES", 1, int))
RESCUE_NUM_WINDOW: int = _env("PAGED_MOE_RESCUE_NUM_WINDOW", 12, int)
RESCUE_NUM_MIN: int = _env("PAGED_MOE_RESCUE_NUM_MIN", 4, int)
RESCUE_AUDIT: bool = bool(_env("PAGED_MOE_RESCUE_AUDIT", 1, int))
RESCUE_AUDIT_P: float = _env("PAGED_MOE_RESCUE_AUDIT_P", 0.02, float)
RESCUE_AUDIT_MAX: int = _env("PAGED_MOE_RESCUE_AUDIT_MAX", 8, int)
RESCUE_AUDIT_TOKENS: int = _env("PAGED_MOE_RESCUE_AUDIT_TOKENS", 16, int)
RESCUE_AUDIT_COOL: int = _env("PAGED_MOE_RESCUE_AUDIT_COOL", 48, int)

# One-graph decode. The per-layer sync exists to learn which experts to read.
# Once the experts this token will actually compute are already in the slab,
# that sync only splits the GPU graph.
#
# STITCH (on by default, 0 forces the per-layer path): the whole routed set
# fits in the RAM budget, so load it into fixed slots. KEEP_FREE means a
# speed-slider prune still computes every expert, so this stays exact
# relative to the slider. Coder-Next-REAP on 48 GB: 38 -> 67 tok/s. A prefill
# that gives the slab back clears the set; the first decode token reads it
# back in (26 GB in about 3 s on REAP) before it runs.
#
# STITCH_PARTIAL (off by default): the set does not fit. After four
# per-layer decode tokens in a row fetch nothing, try one graph over the live
# residency table. The keep set is the slider's (prune, wait, top-k, a strict
# fidelity span). A kept expert that is not resident rewinds that token, runs
# the per-layer path, and waits 32 tokens before trying again. Flow tokens do
# not count as warm, and stitch leaves them alone. Flash on 48 GB, warm,
# prune 0, flow off: 22 -> 26 tok/s in a bench. In real chats it took 0 to 20
# tokens out of about 300 and replayed one or two, so it stays off.
STITCH: bool = bool(_env("PAGED_MOE_STITCH", 1, int))
STITCH_PARTIAL: bool = bool(_env("PAGED_MOE_STITCH_PARTIAL", 0, int))

# Flow decode: one sync per token instead of one per MoE layer.
#
# The per-layer sync exists because expert->slot lives in a Python dict, so
# `_run_slab` cannot build its gather indices until the router's choices have
# reached the CPU. Flow decode keeps that mapping in a GPU-resident table
# instead, and does selection with mx ops, so a whole token is one graph. On
# GLM-4.7 the per-layer round trip is 89 blocking evals: syncfree_probe puts
# the removable part at 15.5 ms/token, and on top of that every blocking miss
# is on the critical path (~41 ms/token measured).
#
# What it costs: the CPU no longer learns the routing in time to read for it,
# so a layer computes with the experts that are already resident and zeroes the
# rest - the reads still get issued, one token late. That is the same trade
# WAIT_ABOVE already makes, taken to its limit, so this is a *fidelity* knob and
# stays off by default. Selection is "the FLOW_TOPM highest-weight resident
# experts of the router's top-k": every expert computed is one the router
# picked, so this never substitutes.
#
# Needs the slab (uniform experts) and a recognized router; anything else falls
# back to the per-layer path automatically. Also needs PRUNE > 0: PRUNE 0 is
# the exact setting and flow is not exact. Reads follow PRUNE (only experts
# PRUNE keeps are read), while compute takes every resident expert of the
# top-k.
#
# It is a speed tier, not a free win. Flash, bench/quality_probe.py, steady
# state, PRUNE 0.7, WAIT_ABOVE 0.24, same env for both:
#   per-layer  18.7 tok/s  agree 0.86  nll +0.05
#   flow       23.0 tok/s  agree 0.83  nll +0.09
# See FLOW_MAX_CRIT for what keeps it from running where it costs too much.
FLOW: bool = bool(_env("PAGED_MOE_FLOW", 0, int))
# Experts computed per layer per token. Fixed, because a static shape is what
# lets the gather be built without a sync. 0 = the router's own top-k (no
# reduction, so the only saving is the sync itself). Below top-k it also cuts
# GPU work: on GLM-4.7 each expert is ~13 MB of the ~14 GB a token reads.
FLOW_TOPM: int = _env("PAGED_MOE_FLOW_TOPM", 0, int)
# Decode tokens after each prefill that run the ordinary per-layer path before
# flow takes over. Prefill installs loose arrays, not slots, so the slab is
# empty when decode starts: a flow token there would find nothing resident and
# compute almost nothing. These tokens block on their reads the usual way and
# leave the slab holding exactly the working set the next token wants.
FLOW_WARMUP: int = _env("PAGED_MOE_FLOW_WARMUP", 2, int)
# After FLOW_WARMUP, keep the per-layer path until its recent hit rate (EMA
# over tokens) reaches this. Two tokens leave the slab far from warm after a
# big prefill, and a cold flow token skips what the per-layer path would have
# waited for: on Flash the first ~200 tokens after a 48-token prefill scored
# +0.34 nll against +0.06 for the per-layer path. There is no token cap on
# purpose: a model whose cache never gets this warm would skip too much in
# flow, so it stays on the per-layer path. 0 = count only.
FLOW_WARM_HIT: float = _env("PAGED_MOE_FLOW_WARM_HIT", 0.9, float)
# Hit rate alone is not enough. A critical miss is an expert PRUNE keeps, worth
# at least WAIT_ABOVE of the mixture, and not resident: the per-layer path
# blocks on it, flow computes without it. Flow only runs while the EMA of these
# per token (summed over MoE layers) stays under this, and hands back to the
# per-layer path when it goes over (re-entry needs half of it). 0 = no check.
#
# Qwen3-235B passed the hit gate at 0.90 in agent use and produced garbled
# text. quality_probe (tok/s, nll vs PRUNE 0):
#   Flash      PRUNE 0.7   per-layer 18.7 +0.05 | no check 23.0 +0.09 | 5/token 23.3 +0.11
#   235B       PRUNE 0.7   per-layer 10.1 +0.06 | no check 15.8 +0.21 | 5/token 12.2 +0.10
#   Moonlight  PRUNE 0.85  per-layer 46.2 +0.01 | no check 86.9 +0.42 | 5/token 59.4 +0.08
#   Coder-Next PRUNE 0.7   per-layer 28.6 +0.01 | no check 48.5 +0.04 | 5/token 39.7 -0.02
# Moonlight is the DeepSeek-V3 / Kimi K2 router, run with a 1.5 GB cache.
# Both Qwen models run ~0.07 critical misses per layer in flow, but 235B has
# 94 layers to Flash's 48 and pays about 4x the nll for it, hence per token.
FLOW_MAX_CRIT: float = _env("PAGED_MOE_FLOW_MAX_CRIT", 5.0, float)

# N-gram self-draft decoding (pagedmoe/lookup_decode.py).
#
# Off by default because it loses on agent traffic. Keep it that way unless a
# measurement on the workload in question says otherwise.
#
# The idea: decode is disk-bandwidth-bound, so buy tokens per byte instead of
# bytes per second by verifying k drafted tokens in one pass. A pass runs k+1
# positions and pays all of their expert reads, so it wins only when most of
# those positions turn into kept tokens.
#
# Accept rate is the whole ballgame, and it depends on the text, not the model.
# An n-gram drafter can only copy something the context already contains:
#   short self-repeating prompt   accept 0.55   5.44 -> 8.05 tok/s
#   12k varied agent context      accept 0.09   2.77 -> 2.03 tok/s
# The first is what this was originally tuned on, and it is not the workload.
#
# Not quality-neutral alongside PRUNE, either. Bit-identity holds at PRUNE=0
# (tests/test_lookup.py), but prune drops experts per pass, and a k-token pass
# prunes on different mass than k single-token steps, so the two paths diverge.
# Every model in the catalog ships PRUNE > 0. VERIFY (below) replaces this
# loop on streamed MoE: its passes run exact routing, so they emit exact tokens
# under any PRUNE, and it only drafts where drafting measured faster.
#
# The drafter itself is cheap and correct: an n-gram scan, no weights, no RAM
# from the slab, ~3.6 ms/step at 13k context, and rejected drafts are trimmed
# out of the KV cache. It is the read amplification that sinks it.
LOOKUP: bool = bool(_env("PAGED_MOE_LOOKUP", 0, int))
# Draft tokens per verify pass. The pass runs k+1 positions, which must stay
# under streaming._DECODE_MAX_TOKENS to keep the streamed decode path. Higher k
# shares more reads but wastes more work when the draft is wrong.
LOOKUP_TOKENS: int = _env("PAGED_MOE_LOOKUP_TOKENS", 4, int)
# Suffix lengths the drafter will match on, longest first. Longer is a more
# specific context and a better bet; too short and it copies noise.
LOOKUP_NGRAM_MAX: int = _env("PAGED_MOE_LOOKUP_NGRAM_MAX", 8, int)
LOOKUP_NGRAM_MIN: int = _env("PAGED_MOE_LOOKUP_NGRAM_MIN", 3, int)
# Time each pass and draft only where drafting has been faster per emitted
# token than plain decode (per matched suffix length, with a periodic probe).
# Flash k=2, varied 12k context: 0.86x ungated, 0.96x gated. Flash does not
# gain from lookup at all once the baseline is warm (repetitive code: plain
# 25.1, lookup 24.8 tok/s), since a 3-token pass costs 1.5x a sync step and
# plain decode runs on the cheaper flow graph. See lookup_decode._Gate.
LOOKUP_ADAPTIVE: bool = bool(_env("PAGED_MOE_LOOKUP_ADAPTIVE", 1, int))

# Exact-verify decode (verify_decode.py). Drafts are verified under
# streaming.exact_step(), so every token a verify pass emits comes from the
# exact model whatever PRUNE / WAIT_ABOVE / flow are set to. Plain steps are
# mlx-lm's own pipelined step, and a per-span timing governor only lets a
# drafter run where it has measured faster per token than that plain step.
# Streamed MoE only. LOOKUP=1 still selects the older loop. 0 turns it off.
VERIFY: bool = bool(_env("PAGED_MOE_VERIFY", 1, int))
# Where the n-gram drafter may engage on an approximate config: "strict" is
# fidelity spans (tool calls, turn start, the visible reply), "all" is every
# token. On an exact config (PRUNE 0) it is always every token.
VERIFY_SPANS: str = _env("PAGED_MOE_VERIFY_SPANS", "strict", str)
# Resident models draft from n-gram matches too (every span: resident decode
# is exact), next to the MTP head when there is one. Copies of paths and file
# text in tool calls check many tokens in one pass. 0 = MTP head only.
VERIFY_RESIDENT_NGRAM: bool = bool(_env("PAGED_MOE_VERIFY_RESIDENT_NGRAM", 1, int))
# Draft tokens per n-gram verify pass (pass width k+1, kept under
# streaming._DECODE_MAX_TOKENS). Tool arguments copy long runs of file text.
VERIFY_TOKENS: int = _env("PAGED_MOE_VERIFY_TOKENS", 8, int)
# How far the drafter extends a match when choosing between candidates, and
# the top of the governor's match-length buckets. Past 8 tokens it can tell
# the true source of a copy from a look-alike, and a 32-token match is a far
# better bet than a 5-token one.
VERIFY_MATCH_CAP: int = _env("PAGED_MOE_VERIFY_MATCH_CAP", 48, int)
# Size each draft by how long its match is: shorter than VERIFY_TOKENS for weak
# matches, up to 3 more for long ones. Timing only; every drafted token is
# still verified. 0 drafts VERIFY_TOKENS every time.
VERIFY_WIDE: bool = bool(_env("PAGED_MOE_VERIFY_WIDE", 1, int))
# Draft length once a match is 32 tokens or longer (a verbatim copy of a file
# already in context or in a copy source). The pass is this many tokens plus
# one, and it has to stay under streaming._DECODE_MAX_TOKENS. Shorter matches
# stay at VERIFY_TOKENS. Every drafted token is still verified.
VERIFY_LONG_TOKENS: int = _env("PAGED_MOE_VERIFY_LONG_TOKENS", 16, int)
# What a verify pass computes. "plain": the same per-row routing the plain step
# it replaces would use (PRUNE / WAIT_ABOVE / fidelity span scale), so the
# tokens it emits come from the same model plain decode samples from. "exact":
# the router's full top-k (streaming.exact_step). On the app's speed settings
# an exact pass has to block on every expert pruned decode skipped: a 9-token
# pass on Flash at PRUNE 0.9 cost 546 ms and read 4.3 GB against 140 ms and
# 0.17 GB plain (bench/verify_live_cost.py), so exact passes never beat plain
# steps and the governor kept them off. On an exact config both are the same.
VERIFY_FIDELITY: str = _env("PAGED_MOE_VERIFY_FIDELITY", "plain", str)
# Also draft from the JSON-escaped text of recent tool results. Calls whose
# arguments are JSON (Hermes-style templates, or any non-string argument a
# template writes with tojson) copy file text as "\n" and "\"" escapes, which
# never match the raw text the tool result put in context.
VERIFY_COPY_SOURCES: bool = bool(_env("PAGED_MOE_VERIFY_COPY_SOURCES", 1, int))
# Size each pass by what its rows cost in reads (verify_decode._RowCost). A
# row is worth adding while the chance the drafts before it are all accepted,
# times a plain step's time, beats the time to read what a row reads. On a
# model whose cache holds the working set rows are nearly free and drafts keep
# their full width (Flash copy: 46 tok/s either way); where every row brings
# its own cold experts (MiniMax-M3: ~1.5 GB a row) only rows likely to be kept
# survive, for the same speed on ~25% fewer bytes per pass. Timing only. 0
# restores the match-length widths alone.
VERIFY_TRUNCATE: bool = bool(_env("PAGED_MOE_VERIFY_TRUNCATE", 1, int))
# Route replay (route_log.py): keep each position's router ids so a pass that
# copies context can read the copied positions' experts before it starts.
# Prefetch only. Ring of VERIFY_REPLAY_TOKENS positions (2 bytes per layer per
# routed expert per position: 235B at 32k is ~50 MB). Every layer of every
# plan is scored against the pass's real demand without reading, and a layer
# reads ahead only at VERIFY_REPLAY_MIN_PREC or better. A copy routes like its
# source in early layers and much less in late ones (Flash 0.70 vs 0.24 top-k
# overlap, MiniMax-M3 0.71 vs 0.28), so mostly early layers qualify.
VERIFY_REPLAY: bool = bool(_env("PAGED_MOE_VERIFY_REPLAY", 1, int))
VERIFY_REPLAY_TOKENS: int = _env("PAGED_MOE_VERIFY_REPLAY_TOKENS", 32768, int)
VERIFY_REPLAY_MIN_PREC: float = _env("PAGED_MOE_VERIFY_REPLAY_MIN_PREC", 0.5, float)
# Self-draft: k cheap approximate steps, then one pass over them at the same
# routing the plain step uses. "auto" = everywhere when the plain step is
# already exact, and inside tool calls otherwise (file and edit bodies). The
# tool path drafts SELF_DRAFT_TOOL_TOKENS (2), not SELF_DRAFT_TOKENS, because
# a wide pass on divergent text reads more experts than it saves. With
# DRAFT_FLOW live a draft step is cheap, so "auto" drafts SELF_DRAFT_TOKENS in
# every span the drafter engages in and the governor decides. 1 / 0 force
# the full self-draft on or off. SELF_DRAFT_TOOL=0 leaves the tool path off
# while "auto" still self-drafts an exact config.
SELF_DRAFT: str = _env("PAGED_MOE_SELF_DRAFT", "auto", str)
SELF_DRAFT_TOKENS: int = _env("PAGED_MOE_SELF_DRAFT_TOKENS", 4, int)
SELF_DRAFT_TOOL: bool = bool(_env("PAGED_MOE_SELF_DRAFT_TOOL", 1, int))
SELF_DRAFT_TOOL_TOKENS: int = _env("PAGED_MOE_SELF_DRAFT_TOOL_TOKENS", 2, int)
# Routing the self-drafter uses (streaming.draft_step). Drafts never reach the
# output, so these trade acceptance against draft cost only.
DRAFT_PRUNE: float = _env("PAGED_MOE_DRAFT_PRUNE", 0.7, float)
DRAFT_WAIT_ABOVE: float = _env("PAGED_MOE_DRAFT_WAIT_ABOVE", 0.2, float)
# Self-draft steps on the flow graph: each MoE layer computes the resident part
# of the router's top-k from the GPU slot table, with no per-layer sync and no
# reads. On a model whose cache holds the working set the per-layer syncs are
# most of a step, so a draft costs a fraction of a plain step. Drafts never
# reach the output (the verify pass decides every token), so this changes
# acceptance and timing only. The slot tables it needs are kept whenever this
# is on, FLOW or not; plain decode does not read them.
DRAFT_FLOW: bool = bool(_env("PAGED_MOE_DRAFT_FLOW", 1, int))
# Experts computed per layer in a flow draft. 0 = the router's top-k.
DRAFT_FLOW_TOPM: int = _env("PAGED_MOE_DRAFT_FLOW_TOPM", 0, int)
# Multi-token-prediction head (mtp.py) as a drafter, where the checkpoint has
# one (bundled, or fetched by mtp_fetch into MTP_DIR). Drafts are verified like
# any other, so this only changes speed. 0 never loads a head.
MTP: bool = bool(_env("PAGED_MOE_MTP", 1, int))
MTP_DIR: str = _env("PAGED_MOE_MTP_DIR", "~/.paged_moe/mtp", str)
MTP_TOKENS: int = _env("PAGED_MOE_MTP_TOKENS", 3, int)
# The head comes out of the expert cache. Past this share of it the head costs
# more misses than its drafts save, so the model loads without it.
MTP_MAX_FRAC: float = _env("PAGED_MOE_MTP_MAX_FRAC", 0.12, float)
# Head KV is dropped past this many positions (it only sees decoded tokens).
MTP_CTX: int = _env("PAGED_MOE_MTP_CTX", 16384, int)
# Positions queued for the head while it is not drafting; older ones are dropped.
MTP_CATCHUP: int = _env("PAGED_MOE_MTP_CATCHUP", 64, int)
# Draft the next pass's MTP tokens inside the current pass, assuming it is
# accepted in full. Used only when it is; otherwise the head cache is trimmed
# back and drafting goes on as before.
MTP_SPECULATE: bool = bool(_env("PAGED_MOE_MTP_SPECULATE", 1, int))
# Partial accepts on hybrid caches rebuild the recurrent state from the
# attention and MoE outputs the pass already computed (fast_replay.py) instead
# of running the model again. The first FAST_REPLAY_CHECKS of every load also
# run the full replay and compare; a gap past FAST_REPLAY_TOL of how far the
# kept tokens moved the state turns it off for that model. Kernel rounding
# measured up to 0.25 of the move, a replay missing a token 1.3 and up.
FAST_REPLAY: bool = bool(_env("PAGED_MOE_FAST_REPLAY", 1, int))
FAST_REPLAY_CHECKS: int = _env("PAGED_MOE_FAST_REPLAY_CHECKS", 6, int)
FAST_REPLAY_TOL: float = _env("PAGED_MOE_FAST_REPLAY_TOL", 0.5, float)
# Prefetch experts that tool-call bodies actually routed. 1 (default) learns
# during the span and prefetches once the body is underway, plus at the start
# of the next tool span. Held when the plan's precision EMA drops. 0 = off.
TOOL_WARM: bool = bool(_env("PAGED_MOE_TOOL_WARM", 1, int))
TOOL_WARM_GB: float = _env("PAGED_MOE_TOOL_WARM_GB", 2.0, float)
# Tokens into a tool span before its demand counts (the name and JSON keys
# route differently from the file body).
TOOL_WARM_SKIP: int = _env("PAGED_MOE_TOOL_WARM_SKIP", 24, int)
TOOL_WARM_MIN_TOKENS: int = _env("PAGED_MOE_TOOL_WARM_MIN_TOKENS", 32, int)
TOOL_WARM_DECAY: float = _env("PAGED_MOE_TOOL_WARM_DECAY", 0.85, float)
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
ROUTE_RENORM: bool = bool(_env("PAGED_MOE_ROUTE_RENORM", 0, int))

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
SLAB: str = _env("PAGED_MOE_SLAB", "auto", str)
# Debug: check at compute time that no slot a layer is about to read is being
# written, and that slot bookkeeping agrees. Slot reuse overwrites bytes in
# place, so a double-owned slot corrupts output instead of raising.
SLAB_VERIFY: bool = bool(_env("PAGED_MOE_SLAB_VERIFY", 0, int))
# Debug: log Metal memory at the prefill/decode transitions. A prefill OOM is
# the one failure that kills the process outright (the command buffer fails
# asynchronously, so there is nothing to catch), which makes knowing what was
# actually committed at that moment the difference between a fix and a guess.
MEM_DEBUG: bool = bool(_env("PAGED_MOE_MEM_DEBUG", 0, int))
_slab_raw = _env("PAGED_MOE_SLAB_GB", None, float)
SLAB_BYTES: int | None = None if _slab_raw is None else int(_slab_raw * (1 << 30))

# Frequency-protected eviction (LFRU-style second chance).
LFRU: bool = bool(_env("PAGED_MOE_LFRU", 1, int))
LFRU_SCAN: int = _env("PAGED_MOE_LFRU_SCAN", 8, int)

# Ceiling on prefetched-but-not-yet-used numpy staging. None = auto-scale
# with the model's expert size (see staging_bytes_auto).
_staging_raw = _env("PAGED_MOE_STAGING_GB", None, float)
STAGING_BYTES: int | None = (
    None if _staging_raw is None else int(_staging_raw * (1 << 30))
)


# Flat KV+scratch reserve used when RESERVE_CTX is 0. Machine provider scales
# this down on 16 GB Macs (6 GB would eat the whole budget). 48 GB default.
HEADROOM_GB: float = _env("PAGED_MOE_HEADROOM_GB", 6.0, float)
# Decode activations + Metal recycled-buffer pool. Prefill yields the slab.
ACTIVATION_GB: float = _env("PAGED_MOE_ACTIVATION_GB", 3.0, float)
# Cap on staging_bytes_auto. Scaled with RAM by the machine provider.
STAGING_CAP_GB: float = _env("PAGED_MOE_STAGING_CAP_GB", 6.0, float)


def staging_bytes_auto(expert_nbytes: int) -> int:
    """Staging must hold the prefetch horizon in *experts*, not bytes.

    A fixed 1 GB holds ~400 qwen3-next experts (plenty) but <100 of a 235B's
    10.6 MB experts - measured on Qwen3-235B-A22B, that dropped ~100 staged
    experts per decoded token, many of them correct predictions that were then
    re-read from disk as blocking misses. ~256 experts of headroom covers the
    deepest prefetch horizon (PREFETCH_DEPTH + PREDICT_DEPTH layers of top-k
    plus slack) with room for timing skew.
    """
    cap = int(float(STAGING_CAP_GB) * (1 << 30))
    return max(1 << 30, min(cap, 256 * max(1, expert_nbytes)))


# Bytes evicted between mx.clear_cache() calls. None = auto-scale with the
# cache budget (see clear_bytes_auto).
_clear_raw = _env("PAGED_MOE_CLEAR_GB", None, float)
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
GROUP_BYTES: int = _env("PAGED_MOE_GROUP_MB", 1024, int) * (1 << 20)

# Prefill read shape. A prefill group is a *sorted, near-contiguous* run of
# expert ids, so with a large enough run cap each component of the group is one
# contiguous byte range on disk and the whole group costs a handful of big
# sequential reads. The old 32 MB run cap was smaller than a single DeepSeek
# expert (~25 MB), which defeated coalescing entirely and turned every prefill
# pass into ~15k independent per-expert reads.
#
# RUN_MB bounds one sequential span (and hence one reader thread's share of the
# group); SLICE_MB is the pread granularity inside a span - large, because
# prefill already has abundant parallelism and what it needs is *bytes per
# syscall*, not more concurrent small reads.
PREFILL_RUN_BYTES: int = _env("PAGED_MOE_PREFILL_RUN_MB", 512, int) * (1 << 20)
PREFILL_SLICE_BYTES: int = _env("PAGED_MOE_PREFILL_SLICE_MB", 8, int) * (1 << 20)

# Query rows per attention call during fused prefill. This is a *performance*
# target, not a safety bound - the Metal score-matrix bound still applies on top
# (adaptive_prefill.attention_sub_chunk takes the min).
#
# Smaller is strictly less work. Prefilling T tokens in sub-chunks of C computes
# ~T^2/2 + T*C/2 score pairs against a causal need of T^2/2, because a
# materialized mask gives the kernel no way to skip the sub-chunk's own upper
# triangle. The overhead fraction is therefore C/T: on a 14k prompt, C=9398 (the
# old Metal-max sizing) computes 56% more score pairs than necessary, C=2048
# computes 15%, C=1024 computes 7%. The floor is GEMM efficiency - below ~1k
# query rows the score matmuls get too skinny to fill the GPU.
#
# Safe on DSA: the indexer applies the causal mask *before* argpartition, so a
# query attends to the top-`index_topk` of its causally valid keys no matter
# which sub-chunk it lands in. C changes cost, not the attended key set.
ATTN_SUB_CHUNK: int = _env("PAGED_MOE_ATTN_SUB_CHUNK", 2048, int)

# Residency budget while a large prefill chunk runs (see ExpertCache.enter_prefill).
PREFILL_CACHE_GB: float = _env("PAGED_MOE_PREFILL_CACHE_GB", 4.0, float)

# A prefill that keeps the slab (chunks below PREFILL_SHRINK_TOKENS: every
# agent turn after the first) reads its misses into free slab slots instead of
# fresh numpy buffers that are then copied into fresh Metal buffers. Same
# bytes, same per-expert kernel on the same slot views hits already use, so
# output is unchanged. Never evicts a resident expert. 0 = old run-read path.
PREFILL_SLAB: bool = bool(_env("PAGED_MOE_PREFILL_SLAB", 1, int))

# A prefill layer can't name its experts until its router has run, so the
# drive idles at every layer boundary while the last group computes and the
# next layer's attention and router run (10-22% of a small prefill on the
# 235B, bench/prefill_gap_probe.py). With this on, each layer also runs the
# next layer's router on its own hidden state and starts those reads behind
# its own, most-predicted first. Prefetch only, never an eviction: the next
# layer still computes exactly what its router picks, and guesses it didn't
# route are handed back. Stands down by itself if the next layer routes under
# half of what it guessed.
PREFILL_AHEAD: bool = bool(_env("PAGED_MOE_PREFILL_AHEAD", 1, int))
# Past the free slots, PREFILL_AHEAD may start up to this many MB of the next
# layer's predicted reads as plain run reads (host buffers, copied in when
# claimed). Bounded because it is host RAM on top of the layer's own groups:
# never more than half of what the kernel can spare past the slab margin, and
# none at memory pressure WARN or worse. 0 = free slots only.
PREFILL_AHEAD_LOOSE_MB: int = _env("PAGED_MOE_PREFILL_AHEAD_LOOSE_MB", 512, int)
# Prefill chunks the server runs for their KV only (the last prompt token goes
# through the decode step, so no chunk's logits are read). The final decoder
# layer's MoE output then reaches nothing but those logits, so that layer
# skips its expert reads and compute: 1/48 of each pass on Flash, 1/57 on
# MiniMax-M3. The KV written is unchanged. Only when the last streamed MoE
# layer is the model's last decoder layer.
PREFILL_SKIP_FINAL: bool = bool(_env("PAGED_MOE_PREFILL_SKIP_FINAL", 1, int))
# Prefill misses that get no free slab slot (and every miss once the slab is
# released) are read by the native pool into a stage: a small Metal buffer
# laid out like the slab, held only while a prefill runs. No host buffer, no
# numpy -> Metal copy, no Python read threads, and the group computes as one
# sorted gather_qmm per projection. Without it those reads take the run-read
# path at 5.1-6.1 GB/s on Flash against a 12 GB/s drive. 0 = run reads.
PREFILL_STAGE: bool = bool(_env("PAGED_MOE_PREFILL_STAGE", 1, int))
# Group size while the stage is up. The stage is 4 groups: the one computing,
# the one reading behind it, and room for the next layer's lookahead plus that
# layer's first group. Halved until 4 fit the memory left next to the chunk
# (small Macs), never under 8 experts a group; no room at all = run reads.
PREFILL_STAGE_GROUP_MB: int = _env("PAGED_MOE_PREFILL_STAGE_GROUP_MB", 384, int)

# Prompt tokens per prefill pass (server/CLI default). With adaptive prefill
# this is the per-step *ceiling*; DSA models shrink below it as KV grows.
PREFILL_CHUNK: int = _env("PAGED_MOE_PREFILL_CHUNK", 32768, int)

# Adaptive prefill: recompute chunk size each step from current KV length.
# Quality-neutral vs a fixed schedule of the same sizes; unlocks large early
# chunks on DSA models without blowing Metal's max single buffer.
ADAPTIVE_PREFILL: bool = bool(_env("PAGED_MOE_ADAPTIVE_PREFILL", 0, int))
ADAPTIVE_PREFILL_MODE: str = _env("PAGED_MOE_ADAPTIVE_PREFILL_MODE", "auto", str)
ADAPTIVE_PREFILL_MAX: int = _env(
    "PAGED_MOE_ADAPTIVE_PREFILL_MAX", PREFILL_CHUNK, int
)
ADAPTIVE_PREFILL_SAFETY: float = _env(
    "PAGED_MOE_ADAPTIVE_PREFILL_SAFETY", 0.75, float
)
ADAPTIVE_PREFILL_DEBUG: bool = bool(
    _env("PAGED_MOE_ADAPTIVE_PREFILL_DEBUG", 0, int)
)

# Layer-fused prefill: sub-chunk attention *inside* each decoder layer so the
# model-level chunk (and therefore the number of expert-mass passes) is bounded
# by activation memory instead of by the attention score matrix. See
# prefill_fused. Off makes every architecture behave as it did before.
FUSED_PREFILL: bool = bool(_env("PAGED_MOE_FUSED_PREFILL", 1, int))

# Contexts below this many tokens keep an fp16 KV cache; past it, the cache is
# quantized to 8-bit (kv_bits=8) as before. fp16 KV measured +8% decode
# throughput on Qwen3-235B at 2k context - quantized-KV attention adds
# dequant/quant ops to every layer's graph, and with 94 per-layer syncs the
# encode cost of those ops is paid on the critical path 94 times per token.
# fp16 KV costs ~192 KB/token on Qwen3-235B, so 8k tokens is ~1.6 GB - cheap.
# Past the threshold the memory argument wins. 0 = always quantize (old
# behavior).
KV_FP16_CTX: int = _env("PAGED_MOE_KV_FP16_CTX", 8192, int)

# Precision of the decode-time KV cache. 8-bit at group 64 costs 1.0625 B per
# element against fp16's 2, and quantized-KV attention is only used for decode
# (one query row), where its lack of a fused kernel does not matter.
KV_BITS: int = _env("PAGED_MOE_KV_BITS", 8, int)
KV_GROUP_SIZE: int = _env("PAGED_MOE_KV_GROUP_SIZE", 64, int)

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
RESERVE_CTX: int = _env("PAGED_MOE_RESERVE_CTX", 0, int)

# Multiplier on the reserved KV bytes, covering the prompt-cache store's steady
# state next to the live cache. 1.25 is one full conversation plus a partial
# user-segment snapshot (see server._snapshot_user_segment).
KV_STORE_SLACK: float = _env("PAGED_MOE_KV_STORE_SLACK", 1.25, float)

# Hand a reused prompt cache to the request instead of deep-copying it
# (see kvmem.take_nearest_cache). 0 restores mlx-lm's copying.
PROMPT_CACHE_MOVE: bool = bool(_env("PAGED_MOE_PROMPT_CACHE_MOVE", 1, int))

# While macOS has reported memory pressure in the last SLAB_REGROW_CALM_S, the
# prompt-cache store keeps this fraction of its budget. Older snapshots are
# the cheapest memory the engine holds: losing one costs a re-prefill only if
# a request branches from that point, while the compressor taking it costs
# every decode step. The live conversation is always kept. 1 = off.
PROMPT_CACHE_PRESSURE_FRAC: float = _env("PAGED_MOE_PROMPT_CACHE_PRESSURE_FRAC", 0.35, float)

# Resume a reused 8-bit KV cache without building an fp16 copy of it first
# (kvmem.ResumedKVCache). Same logits. 0 restores the full dequantize, which
# on a long conversation makes the expert slab yield every turn.
KV_LAZY: bool = bool(_env("PAGED_MOE_KV_LAZY", 1, int))

# Recurrent-state checkpoints for hybrid models (see hybrid_ckpt.py). A hybrid
# prompt cache cannot trim, so a request that branches one message back from
# the last stored entry used to prefill cold from token 0. Holding the
# recurrent state at each turn's resume point and mid-snap boundary lets a
# longer stored entry be rewound there instead. Capture holds references (no
# copy); the bytes count against the prompt-store budget in kvmem.bound_store.
# Inert on models whose cache trims. 0 restores the old cold path.
HYBRID_CKPT: bool = bool(_env("PAGED_MOE_HYBRID_CKPT", 1, int))
# Checkpoints kept per server, newest wins. Two or three per turn.
HYBRID_CKPT_MAX: int = _env("PAGED_MOE_HYBRID_CKPT_MAX", 8, int)
# Byte ceiling. Flash is about 110 MB per checkpoint, Coder-Next about 75 MB.
# Also capped at a quarter of the prompt-store budget.
HYBRID_CKPT_MB: float = _env("PAGED_MOE_HYBRID_CKPT_MB", 1024.0, float)
# Boundaries shorter than this re-prefill fast enough on their own.
HYBRID_CKPT_MIN_TOKENS: int = _env("PAGED_MOE_HYBRID_CKPT_MIN_TOKENS", 1024, int)
# Take a hybrid turn's mid-snap inside the one prefill pass (inpass_snap.py)
# instead of stopping at it and running the generation prompt as a second
# chunk, which re-read ~2 GB of experts per Flash turn.
HYBRID_INPASS_SNAP: bool = bool(_env("PAGED_MOE_HYBRID_INPASS_SNAP", 1, int))
# Spell a replayed assistant turn with the ids decode produced (replay_splice.py).
# Encoding the turn's text again gives other ids, so the stored prompt + turn
# entry was never a prefix of the next prompt and every turn prefilled its own
# output again (79 of 80 turns in one Flash agent log). 0 = plain encode.
REPLAY_SPLICE: bool = bool(_env("PAGED_MOE_REPLAY_SPLICE", 1, int))
# Turns remembered per server. Each later prompt has to spell every turn it
# replays the same way, so an evicted turn costs one prefill from that point.
REPLAY_SPLICE_MAX: int = _env("PAGED_MOE_REPLAY_SPLICE_MAX", 512, int)

# The stable pin (host front matter) on disk, see prefix_disk.py. A restart
# used to prefill the front matter again: 14k tokens, 40 s on Flash.
PREFIX_DISK: bool = bool(_env("PAGED_MOE_PREFIX_DISK", 1, int))
# Empty = `prefix_cache` next to SIDECAR_DIR, so a host app keeps it under its own data dir.
PREFIX_DISK_DIR: str = _env("PAGED_MOE_PREFIX_DISK_DIR", "", str)
# An inherited fd the host writes a 32-byte hex key to, then closes: every
# prefix file is then sealed with it (seal.py). -1 = plain files.
PREFIX_DISK_KEY_FD: int = _env("PAGED_MOE_PREFIX_DISK_KEY_FD", -1, int)
# Models kept, most recently used first, and files per model and kind (one per
# workspace; a new save replaces its own workspace's older file first). A
# model is 1 to 2.6 GB on disk, so 6 fits the 12 GB cap. A slot whose model
# folder is gone (moved or deleted) is dropped on the next save.
PREFIX_DISK_MODELS: int = _env("PAGED_MOE_PREFIX_DISK_MODELS", 6, int)
PREFIX_DISK_PER_MODEL: int = _env("PAGED_MOE_PREFIX_DISK_PER_MODEL", 2, int)
PREFIX_DISK_MAX_GB: float = _env("PAGED_MOE_PREFIX_DISK_MAX_GB", 12.0, float)
# Whole conversations a host asked to keep (stable_prefix_chat_messages), per
# model. Inside PREFIX_DISK_MAX_GB with everything else.
PREFIX_DISK_CHATS: int = _env("PAGED_MOE_PREFIX_DISK_CHATS", 6, int)
# Recurrent state checkpoints inside a hybrid file, every this many tokens, so
# a prompt that changes part way in (an update to the system prompt) resumes
# from the last one before the change instead of prefilling the whole file.
# Taken inside passes that already run; 0 = none.
PREFIX_DISK_CKPT_EVERY: int = _env("PAGED_MOE_PREFIX_DISK_CKPT_EVERY", 2048, int)
# Checkpoints per file. Each is held in RAM until the file is written: about
# 110 MB on Flash, 75 MB on Coder-Next.
PREFIX_DISK_CKPT_MAX: int = _env("PAGED_MOE_PREFIX_DISK_CKPT_MAX", 4, int)

# Gentle host stages (stage_gentle.py): disk class for the engine's reads
# while a background prompt stage runs ("utility", "throttle", or "off"), and
# the pause after each MoE layer's router sync. A request waiting on the stage
# boosts it back to full speed. M5 Pro 48 GB, stage of the loaded model with
# nothing else busy: 235B 46 s plain, 58 s at 12 ms, no measurable CPU, GPU
# or disk contention either way; the pause is for busier and smaller Macs.
STAGE_IOPOL: str = _env("PAGED_MOE_STAGE_IOPOL", "utility", str)
STAGE_PACE_MS: float = _env("PAGED_MOE_STAGE_PACE_MS", 6.0, float)
# Input (keyboard, mouse, trackpad) within this many seconds holds the next
# layer of a host stage, so stages run in the user's pauses. Users still saw
# lag at app launch with pacing alone. 0 = never wait for the user.
STAGE_YIELD_S: float = _env("PAGED_MOE_STAGE_YIELD_S", 2.0, float)

# Only chunks *larger* than this switch the cache to the shrunk budget.
#
# Reading the expert mass costs ~the same for a 500-token chunk as for a
# 32k-token one (either way, essentially every expert of every layer gets
# routed to). So shrinking the cache pays off only when it buys *fewer passes*:
# for a big cold prompt, yes; for the few-hundred-token delta of an agent turn
# that hits the prompt cache, shrinking would just evict the warm experts that
# make the delta cheap. 8192 is the largest chunk that still fits alongside a
# full cache; at or above it the slab is released (streaming.py).
PREFILL_SHRINK_TOKENS: int = _env("PAGED_MOE_PREFILL_SHRINK_TOKENS", 8192, int)


def total_ram_bytes() -> int:
    """Physical RAM (macOS + Linux).

    ``PAGED_MOE_SIMULATE_RAM_GB`` (device sandbox) overrides the sysctl so
    auto cache math can pretend this is a smaller Mac. Metal still uses the
    real GPU; only the budget envelope changes.
    """
    from . import release

    fake = release.env("PAGED_MOE_SIMULATE_RAM_GB").strip()
    if fake:
        try:
            gb = float(fake)
            if gb > 0:
                return int(gb * (1 << 30))
        except ValueError:
            pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        import subprocess

        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
        )
        return int(out.stdout.strip())
