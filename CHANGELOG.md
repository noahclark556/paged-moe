# Changelog

Notable changes to PagedMoE.

Format loosely follows Keep a Changelog. Versioning aims for SemVer.

## [Unreleased]

<!-- Nothing yet. -->

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
