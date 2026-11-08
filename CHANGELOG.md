# Changelog

Notable changes to PagedMoE.

Format loosely follows Keep a Changelog. Versioning aims for SemVer.

## [Unreleased]

<!-- Nothing yet. -->

## [0.3.0] - 2026-11-07

The full engine, same build as the one I run every day. 0.2.7 held a few
pieces back from the public package. Those are back, plus everything since.

### Breaking

- Import package renamed `expert_stream` -> `pagedmoe`.
- Env knobs renamed `EXPERT_STREAM_*` -> `PAGED_MOE_*`. Old names are not
  read. Rename them in `~/paged-moe-config.yaml`. See "Upgrading from 0.2.x"
  in the README.

### Back in the package

- Sidecar net-time governor (the sidecar only acts while decode is measurably
  faster with it on).
- Wrap head defaults on again once the sidecar is enabled.
- Early-decode warm pool and the `[ttw]` time-to-warm lines.

### Prefill

- One pass over the expert mass per prompt where it fits. The expert cache
  shrinks out of the way while a big chunk runs. Qwen3-Coder-Next, 31.5k-token
  cold prompt: 541 -> 937 tok/s, 85 -> 44 GB read, lower peak.
- Prefill lookahead (`PREFILL_AHEAD`): each layer starts the next layer's
  reads. Qwen3-235B, 512 tokens on seeded KV: 9.1 s -> 6.0 s.
- Prefill stage (`PREFILL_STAGE`): misses read through the native pool into a
  temporary slab, no host copy.
- Short chunks read misses straight into free slab slots.
- The last decoder layer is skipped during prefill when nothing reads its
  output. KV is bit-identical.
- Prefill no longer reorders the decode working set.

### Agent turns

- Prompt-cache snapshots at the last divergence point, the think tail and
  message boundaries. Agent turns reuse the conversation and prefill only the
  new tokens (0.3 to 2 s on a 13.5k session).
- `stable_prefix_messages`, `warm_prefix` and cache breakpoints.
- Front matter KV on disk (`PREFIX_DISK`), with a fingerprint over weights,
  config, engine numerics and settings. Head and shared files so a memory edit
  or a new workspace doesn't start cold. Optional sealing at rest (AES-256-CTR
  + HMAC-SHA256, key passed on a pipe).
- Recurrent-state checkpoints (`HYBRID_CKPT`) so hybrid models can branch back
  to an earlier message. Bit-identical to a fresh prefill, about 1 ms.
- Replay splice: a returned assistant turn keeps the ids decode produced.

### Decode

- Verify decode (`VERIFY`, on): n-gram, self-draft and MTP drafters, one
  pass per draft, per-span timing governor. 2.26x on a file edit with prune
  off. Passes run at the same fidelity as plain decode by default,
  `VERIFY_FIDELITY=exact` for exact-model tokens.
- Row truncation and route replay keep verify passes cheap on big models.
- Native reads split each expert across idle readers, demand misses first.
  MiniMax-M3: 5.65 -> 9.31 tok/s. Qwen3.8-Flash-Next: 5.92 -> 8.66.
- Hits early: slab-resident experts start on the GPU before the misses land.
- Fidelity spans: tool calls and turn starts decode stricter under prune.
  Critical-miss gate and loop rescue for aggressive settings.
- Live `PRUNE` / `WAIT_ABOVE` changes without a restart.

### Memory and stability

- Every pread capped at 1 MB (past that macOS ignores `F_NOCACHE`) plus a page
  cache trimmer. A long prefill used to fill the file cache and stall the
  whole machine.
- Slab follows macOS memory pressure down and regrows when it clears.
- Resumed turns dequantize stored 8-bit KV one layer at a time (`KV_LAZY`).
- Watchdog: a stalled request or sustained critical pressure exits with code
  70 instead of hanging.
- Load checks for GPU memory still held by the driver after a crash.
- KV priced per architecture, including DSA caches.

### Models

- GLM-5.2 and DeepSeek-V3.2 through shared DSA classes, sparse prefill
  attention (~110 s off a cold 32k prefill on GLM-5.2).
- MiniMax-M2 / M3: sigmoid-bias routers, XML tool parser, `<mm:think>` tags,
  checkpoint-shipped architecture files registered per process.
- Qwen3.8-Flash-Next (`qwen4_exp`).
- MTP heads as verify drafters, fetched when the conversion dropped them.
- Image input for resident Qwen VL chat.
- Kimi-K2 tokenizer needs `tiktoken`, now a dependency.

### Repo

- `tests/`, `bench/`, `docs/`, the Node client and a pdb walkthrough ship
  again.
- README rewritten around current numbers.

## [0.2.7] - 2026-09-18

### Changed

- Sidecar wrap head defaults **off** (opt-in). Public samples match.
- Leaner default surface: route prediction, fused prefill, prune/wait recipes
  remain the documented speed path; optional wrap is available when enabled.
- Drop early-decode warm-pool path and related `[ttw]` helpers from the public
  package so cold-start behavior stays simple and documented.

### Removed

- Public packaging of the early-decode warm pool modules.
- Sidecar net-time actuation governor from the public package (ByteLedger
  accounting remains for optional wrap).

## [0.2.6] - 2026-09-18

### Added

- Native C expert-read pool (`EXPERT_STREAM_NATIVE_READ`, default on): one
  job per expert with the GIL released for the latch. Same bytes as the
  Python pool; falls back automatically if the extension cannot load or
  compile. Optional `setup.py` build; runtime JIT if needed.
- Early-decode warm pool (`EXPERT_STREAM_EARLY_DECODE`): frequency table over
  the first decode tokens after prefill, budgeted prefetch into the slab.
  Prefetch-only (quality-safe). Not a sidecar head.
- Cold bootstrap when early-decode is on and train is off: train until a
  high cover/prec bar, then freeze to prefetch-only. Explicit
  `EARLY_DECODE_TRAIN=1` keeps training. Cover/prec hold + count decay.
- Greppable `[ttw]` time-to-warm lines (`EXPERT_STREAM_TTW_LOG`, default on)
  for A/B'ing early-decode: first decode through the early window.
- Sample yaml turns early-decode on (train off) for the sidecar models.
- `expert_stream/machine/` (detect / ladder / envelope) shipped in the package
  so `paged-moe machine` / `ladder` match the README.

### Changed

- Install path documents the optional native extension build.
- Public tree no longer ships `tests/` (runtime does not need them).

## [0.2.5] - 2026-09-17

### Added

- Kimi-K2-Instruct support: sample yaml entry, server `trust_remote_code` by
  default, transformers-5 tokenizer shim, and the `kimi_k2` tool parser when
  the chat template uses `tool_declare`
- Plain MLA (`kimi_k2` / `deepseek_v3`) keeps fp16 KV - mlx-lm's 8-bit KV
  cache breaks that attention path after the first decode token

### Changed

- Install sample config lists Kimi alongside the other ladder checkpoints

## [0.2.4] - 2026-09-17

### Added

- Flow decode (`EXPERT_STREAM_FLOW`, off by default): one sync per token via
  a GPU expert->slot table instead of one sync per MoE layer. Fidelity knob;
  falls back when the slab or router is unavailable
- `EXPERT_STREAM_FLOW_TOPM` / `EXPERT_STREAM_FLOW_WARMUP` for resident-only
  selection and post-prefill warm-up before flow takes over
- N-gram self-draft decode (`EXPERT_STREAM_LOOKUP`, off by default): more
  tokens per expert-byte pass when the context repeats. Bit-identical under
  greedy at `PRUNE=0`; leave off for prune recipes / varied agent traffic
- DeepSeek-V3.2 chat-template shim so mlx-lm's `enable_thinking` maps to
  `thinking_mode`

### Changed

- MoE layers reuse the block's own router output when available (no second
  router pass on the hot path)

## [0.2.3] - 2026-09-16

### Added

- Layer-fused prefill: attention sub-chunks inside each decoder layer; MoE
  streams once per prompt chunk (`EXPERT_STREAM_FUSED_PREFILL`, default on)
- `EXPERT_STREAM_ATTN_SUB_CHUNK` (default 2048): attention size target under
  the Metal score-matrix bound
- Adaptive prefill helpers / modes (`EXPERT_STREAM_ADAPTIVE_PREFILL*`) for
  DSA-aware step sizing
- Single-pass prefix snapshots on trimmable KV: one prefill, then deepcopy+trim
  forks (no mid-prefill expert-mass stop)
- `[prefill]` cost lines: tokens, expert GB, passes over the mass, disk-blocked
  vs compute

### Changed

- Prefill read shape sized for bandwidth (larger runs/slices vs decode)
- Default model-level prefill chunk ceiling raised so a long agent prompt can
  be one expert pass when fused is on

### Fixed

- Adaptive sizing broken on DeepSeek `CacheList` (offset always looked like 0)
- Prefill coalescing defeated when the run cap was smaller than one expert

## [0.2.2] - 2026-09-16

### Added

- Net-time sidecar governor: A/B actuation on real decode tok/s; ties settle off
- Wrap issuance gated on calibrated probability, byte budget, and read slack
- Residency head redesigned (evidence-gated); stays off by default
- `read_slack()` on the expert cache so speculation only uses idle reader capacity
- Decode miss path: precomputed read plans, slot `memoryview`s, `_SlotLatch` +
  `_ReadPool` (fewer Futures / less GIL on the hot path).
  `EXPERT_STREAM_READ_POOL_THREADS` (`0` = auto, `-1` = old executor path)

### Changed

- Sidecar docs: governor + route prediction called out honestly (no oversell)
- Prefill head only actuates when it beats the free copy-warm baseline

### Fixed

- Sidecar could previously look like a win on recall/`gain` while slightly
  slowing decode; governor and byte accounting close that gap
- Decode miss path Python overhead that left the SSD under-fed on big MoEs

## [0.2.0] - 2026-09-15

### Added

- Initial public engine snapshot (import name `expert_stream`)
- Sample `~/paged-moe-config.yaml` seeding; sidecar data under `~/.paged_moe/`

[Unreleased]: https://github.com/noahclark556/paged-moe/compare/v0.2.4...HEAD
[0.2.4]: https://github.com/noahclark556/paged-moe/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/noahclark556/paged-moe/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/noahclark556/paged-moe/compare/v0.2.0...v0.2.2
[0.2.0]: https://github.com/noahclark556/paged-moe/releases/tag/v0.2.0
