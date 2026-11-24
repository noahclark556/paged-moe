# Changelog

Notable changes to PagedMoE.

Format loosely follows Keep a Changelog. Versioning aims for SemVer.

## [0.4.0] - 2026-11-23

GLM-5.3-Flash (`glm5_next`): the engine now runs and trains the hybrid
architecture (KDA linear attention + sparse MLA, mHC, SwitchGLU MoE) through
a bundled port, so MLX checkpoints like `mlx-community/GLM-5.3-Flash-4bit`
stream. Matches the transformers reference, and LoRA exports to PEFT. Text
only for now, the vision tower is dropped at load. GGUF builds won't load,
use an MLX build or convert the original with `paged-moe convert`.

- The MTP layer is a verify drafter now. It's a plain residual sparse MLA +
  MoE block with no mHC, so the model builds it itself instead of cloning a
  backbone layer. On a 48 GB Mac decode goes from 0.54 to 0.93 tok/s with
  the same tokens. The head is 4.6 GB and comes out of the expert cache
  (12.8 to 8.7 GB there), which barely matters with 105 GB of experts, so
  this model lets the head take up to 40% of the cache instead of the usual
  12%. `PAGED_MOE_MTP_MAX_FRAC` still overrides it.
- A head that can't fit is skipped before it's read, so it doesn't add to
  load time.
- Fixed the sparse attention indexer reusing its pooled keys after a cache
  trim. A single token step right after a trim past `index_topk` could read
  pools that covered dropped tokens or crash on an empty pool.

Conversion. `paged-moe convert` (and `pagedmoe.convert`) turns a Hugging Face
safetensors checkpoint into an MLX one with a per-part recipe.

- Presets (`balanced`, `standard`, `quality`, `max`, `compact`, `float`,
  `fit`) and per-part settings for routed experts (gate/up and down),
  shared experts, attention, dense MLP, router, embeddings and output head:
  affine bits and group size, `mxfp4` / `mxfp8` / `nvfp4`, or float. Edge
  layers and regex rules on top.
- `plan` reads only the safetensors headers: output size, bits per weight,
  resident or streamed on this Mac, disk and peak memory, and `--compare`
  for every preset side by side.
- Hub sources stream a few layers at a time (download, convert, delete), so
  the disk only needs the output plus a little staging. Shards are written
  in place with uncached writes, and a stopped run resumes from the last
  finished layer.
- `--adapter` merges a LoRA / DoRA adapter before quantizing, same math as
  `mlx_lm.fuse`. Quantized sources are dequantized and requantized.
- Every output is loaded by mlx-lm in strict mode before it's done. The
  folder gets a model card and `pagedmoe_conversion.json` with the recipe and
  per-part weight SNR. `check` compares against a reference (top-1, KL,
  perplexity), `upload` pushes to the Hub.
- `standard` gives the same weights as `mlx_lm.convert -q`. Tests cover
  that, resume, streamed against whole-checkpoint output, adapter merge and
  requantizing.

Training. `pagedmoe.train` fine-tunes MoE checkpoints with LoRA / DoRA,
including ones much bigger than RAM: the backbone stays resident and the
experts page from the SSD every step, through the same read path as the
server. Methods are SFT, CPT, ORPO, DPO (sigmoid / IPO / hinge), SimPO, CPO,
KTO and GRPO, plus custom objectives, custom rows and trainable input
encoders. Adapters are mlx-lm format and load with `pagedmoe.load` or
`mlx_lm`.

- `paged-moe-train` (same as `python -m pagedmoe.train`): train, `plan`,
  `resume`, `chat` and `eval` against any checkpoint, and `bench` for a
  training benchmark that compares across Macs.
- `paged-moe-train export <run>` writes the adapter in Hugging Face PEFT
  format for GPU users (transformers, vLLM, SGLang, llama.cpp's
  `convert_lora_to_gguf.py`). Names and shapes are checked against the
  original model's safetensors headers, no weights downloaded. Works for
  any run or checkpoint, or at the end of a run with `export_peft: true`.
  Routed-expert LoRA is skipped.
- The run is planned from the Mac: resident or streamed, rows per step,
  micro-batches and the expert cache, from a short probe of real steps. The
  first steps check the plan, and a memory guard gives memory back under
  macOS pressure without changing the gradients.
- Streamed steps run layer by layer for the whole batch, so the expert mass
  is read about twice per step. Frozen quantized weights get a cheaper
  backward (1.0 s to ~0.6 s per 235B layer at 8k tokens), and hybrid linear
  attention trains in chunks instead of a per-token loop.
- Resume restores the optimizer and data position exactly. Checkpoints keep
  the last N and save on SIGTERM.
- Tests: every method on a tiny MoE, streamed against resident gradients,
  the fast backwards against MLX's own, the chunked gated delta against the
  per-token loop.
- Optional extra `pip install -e ".[train]"` for parquet and Hugging Face
  datasets.
- Docs: [docs/paged-training.md](./docs/paged-training.md) and
  [docs/why-it-trains.md](./docs/why-it-trains.md).
  [Paged Trainer](https://pagedmoe.com/train) is a Mac app that runs the
  same trainer from a UI. Runs started from the CLI or
  Python print one line about it; `PAGED_MOE_TRAIN_HINT=0` hides it.

`paged-moe serve <model>` starts the server from just a model path, no config
needed. It reads the checkpoint (size, active params, bit width) and runs exact
by default. `--speed` goes from 1 (exact) to 0 (fastest), or `auto` to pick a
prune for that model, and works on `paged-moe-server` and `run.py` too.
`paged-moe detect <model>` prints what it found and what `--speed` would set.
The server also prints a line on startup saying whether output is exact.

Dependencies are now pinned to the exact versions the engine was built and
benchmarked on (mlx 0.32.2, mlx-lm 0.31.3, and the tokenizer stack). A fresh
install was pulling mlx-lm 0.32, which changed the internals the engine hooks
and broke generation. `install_mlx.sh` also makes `~/.mlx-env` when it's
missing, checks for Python 3.10+ on arm64 (3.12 preferred), and installs the
hook into that venv even when another one is active.

`constraints.txt` pins everything the pinned packages pull in (regex, hf-xet,
httpx, and so on) to the bench env versions, and `install_mlx.sh` installs with
it. pyyaml is pinned directly now since the engine reads its config with it.

`paged-moe bench <model>` benches any checkpoint on your own Mac. It replays a
fixed coding agent task (8 requests, about 13k tokens of context) against the
server at `--speed 0.25` and exact, then prints decode tok/s, time to first
token, prefill, memory and the loss of the speed run against the exact model.
`--quick` halves the budgets for a smoke test. Also callable from Python as
`pagedmoe.benchmark.run()`.

Macs with more than 48 GB no longer drop to fewer SSD readers than a 48 GB
one. The reader count came from the performance core count, and macOS only
reports 6 of those on M5 chips, so a 96 GB M5 ran with 6 readers instead of
16. At the 1 MB read size the engine uses, that left about a fifth of the
drive's bandwidth unused. From 48 GB up it's now never below 16, and more cores
still raise it up to 32. Smaller Macs are unchanged, and `read_threads` in
your config or `PAGED_MOE_READ_THREADS` still overrides it. Pull and restart
the server; `paged-moe machine` shows the new count.

The server now fits the usual chat template kwargs to what each model's
template actually reads, so the thinking and effort switches work without
knowing each family's quirks. It checks the template once at load by
rendering it, and only steps in where that shows it matters:

- MiniMax-M3 ignored `enable_thinking` and always ran adaptive. It now maps to
  the template's `thinking_mode`. DeepSeek-V3.1 gets the same for its
  `thinking` switch.
- Templates that keep every earlier turn's reasoning when `preserve_thinking`
  isn't sent (qwen38-coder-flash does) now drop it unless you ask. Long multi
  turn agent sessions were losing the thread on that old reasoning.
- An effort the template would reject (`high` on that same Flash template,
  which only takes low, medium and xhigh) moves to the nearest one it takes
  instead of failing the request. `reasoning_effort` at the top of the request, the way OpenAI
  clients send it, is passed to the template too.

Anything you send explicitly is kept as is, apart from an effort the template
can't take. Templates that already handle these render exactly as before. The
startup log says what it changed for that model, and
`PAGED_MOE_TEMPLATE_KWARGS=0` turns it all off.

Tool calls the model writes in a shape the server's parser doesn't expect are
no longer lost as plain text. mlx-lm only catches a call when it starts with
the exact tokens of the one parser the template picked, so anything else ended
up in the reply and the client never ran it. The server now also picks up:

- DeepSeek V3 and V3.1 native calls. mlx-lm has no parser for these at all, so
  tool calling on those models didn't work through the server before.
- Calls in another family's format, like Hermes JSON from a Qwen XML model or
  `<invoke name=...>` markup, both on their own and inside the template's tags.
- A reply that is nothing but a `{"name": ..., "arguments": ...}` object.
- A call left in the reasoning, but only when the turn ends with no call and
  no reply text.

It only runs on requests that send tools, and a block only counts as a call
once its name matches one of those tools, so normal text and markup you are
just talking about come through untouched. Recovered calls stream the same as
native ones. DeepSeek-V2 and V2.5, whose templates never render tools, now get
the tool list in the system prompt and their earlier tool turns written out in
the model's own format, so they keep calling past the first round.

A client that already does its own repair can send `"tool_call_repair": false`
to get the old output back, and `PAGED_MOE_TOOL_REPAIR=0` turns it off for the
whole server.

Requests that don't send sampling settings now get the model's own instead of
greedy decoding. mlx-lm falls back to temperature 0 with no top-k, which isn't
what any of these models were tuned for, and thinking models tend to loop on
it. Plenty of OpenAI clients only send the messages, so that's what they got.

Anything left out now comes from the checkpoint's `generation_config.json`,
and whatever that file doesn't set comes from the family's serving recipe:
top-k 40 for DeepSeek and GLM, and a light 1.05 repetition penalty for Qwen,
which stops the paste loops it falls into without hurting code or tool calls.
A checkpoint whose config says `do_sample: false` stays greedy. The startup log
shows what it picked and where each value came from.

Anything the request sends still wins, and so do `--temp`, `--top-p`, `--top-k`
and `--min-p` on the server. `PAGED_MOE_SAMPLING_DEFAULTS=0` goes back to
mlx-lm's defaults.

A model path that doesn't exist (`/...`, `~/...` or `./...`) now fails with
"model folder not found" instead of being sent to Hugging Face as a repo id,
which used to end in a confusing `HFValidationError`.

## [0.3.2] - 2026-11-14

`paged-moe-server` and `run.py` now read `~/paged-moe-config.yaml`. Before this
only the mlx_lm.load hook did, so the server ran on bare defaults even with
per-model `env:` set. Anything you export still wins over the file, and
`--config PATH` points at a different one. `PAGED_MOE_HOOK=0` turns it off.

## [0.3.1] - 2026-11-14

Fixed loading checkpoints that ship their own model file (Qwen Flash with
qwen4_exp.py, MiniMax-M3, etc.) on mlx-lm 0.32. Newer mlx-lm won't build those
without trust_remote_code, so the loader now passes it through the same way the
tokenizer already did. Older mlx-lm still works.

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
