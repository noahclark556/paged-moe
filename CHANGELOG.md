# Changelog

Notable changes to PagedMoE.

Format loosely follows Keep a Changelog. Versioning aims for SemVer.

## [Unreleased]

<!-- Nothing yet. -->

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

[Unreleased]: https://github.com/noahclark556/paged-moe/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/noahclark556/paged-moe/compare/v0.2.0...v0.2.2
[0.2.0]: https://github.com/noahclark556/paged-moe/releases/tag/v0.2.0
