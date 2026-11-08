# PagedMoE

Run Mixture-of-Experts models that are way bigger than your Mac's RAM.
Not as a demo, as a real coding agent: tools, multi-turn, long prompts.

On a 48 GB MacBook Pro this runs a **578 GB Kimi-K2 at ~9 tok/s** with a
32 GB memory peak (with its prune recipe, details below). It prefills a **31,500-token prompt in 34 seconds**. And
after the first turn, an agent's next turn usually starts in **under 2
seconds**, not a minute.

I haven't benched it, but the largest model I've had running reliably as an
agent on a 48 GB MacBook is **GLM-5.2, a 743B parameter model**, through the
Global Agent app.

The trick is that MoE models are sparse. A token only uses a few percent of
the weights. PagedMoE keeps the always-on backbone in memory and pages the
routed experts in from the SSD when the router actually asks for them. Same
checkpoints you already have (plain mlx-community safetensors), same mlx-lm
underneath, and the output is bit-identical to a fully resident run unless
you turn on pruning.

> **Status:** `v0.3.0` pre-release. Installs from a clone, not on PyPI yet.
> APIs can still change before `v1.0`.

**Fastest way to see it run:** download [Global Agent](https://globalagent.dev/)
and pick a model. Global Agent is my local coding IDE (VSCodium-based, MLX under
the hood) and this engine is built into it. It detects your Mac, picks the
settings for that model and runs it right there in the app. No clone, no env
vars. The download is free.

Another app built on the same engine is the
[PagedMoE Assistant](https://pagedmoe.com/assistant), a private AI assistant
that runs on your Mac.

**Jump:** [Numbers](#numbers) | [What's new in 0.3](#whats-new-in-03) |
[Quick start](#quick-start) | [Agents](#built-for-agent-loops) |
[Exact tokens](#exact-tokens-where-they-matter) |
[Your Mac](#results-by-mac) | [The hard parts](#the-hard-parts) |
[Fidelity](#what-is-exact-and-what-isnt) | [Models](#compatible-models) |
[Config](#config) | [Upgrading](#upgrading-from-02x) | [License](#license-dual)

---

## Numbers

Baseline is an M5 Pro with 48 GB, checkpoints on the internal SSD. None of
these load with stock mlx-lm on this machine. The RAM budget is about 35 GB,
so the smallest one is already 1.8x over and Kimi is 16x over.

| Model | Checkpoint | Peak RAM | Prefill | Decode | Setting |
| --- | ---: | ---: | ---: | ---: | --- |
| **Qwen3-Coder-Next** 6-bit | 65 GB | 30.7 GB | 235 tok/s | **20.6 tok/s** | exact (no prune) |
| **Qwen3-235B-A22B** 4-bit | 132 GB | 30.6 GB | 99 tok/s | **7.9 tok/s** | prune 0.7, wait 0.2 |
| **GLM-4.7** 4-bit | 199 GB | 31.8 GB | 73 tok/s | **5.9 tok/s** | prune 0.8, wait 0.2 |
| **DeepSeek-V3.2** 4-bit | 378 GB | 33.2 GB | 45 tok/s | **5.7 tok/s** | prune 0.8, wait 0.2 |
| **Kimi-K2-Instruct** 4-bit | 578 GB | 31.6 GB | 27 tok/s | **8.8 tok/s** | prune 0.8, wait 0.2 |

How these were measured: cold ~2k-token prompt, 128 warmup tokens on the same
KV (not counted), then 512 timed greedy tokens with EOS ignored. Decode is
wall clock. Kimi's last quarter of the run held 10.2 tok/s. The "Setting"
column matters: prune trades a bit of fidelity for fewer disk reads, more on
that [here](#what-is-exact-and-what-isnt). With prune off everything is
bit-identical and the big ones decode slower.

This ladder was last run in September. A lot landed since then, mostly on
the prefill and agent-turn side, and each of those was measured on its own
(below). `bench/memory_benchmark.sh <tier>` reruns it on your machine. It runs
exact unless you say otherwise, so export the Setting column first to match a
row, e.g. `PAGED_MOE_PRUNE=0.8 PAGED_MOE_WAIT_ABOVE=0.2 ./bench/memory_benchmark.sh 3`
for GLM-4.7.

The headline isn't "fastest tokens on earth." It's that a 199 GB or 578 GB
model on a laptop goes from a hard no to something you can actually work with.

---

## What's new in 0.3

This release is the whole engine, the same build I run every day. 0.2.7 held a
few things back to keep the public surface small. They're all in now, along
with about seven weeks of work since. Highlights:

**Prefill got a lot faster.** A prompt chunk routes to basically every expert,
so prefill cost is set by how many times you read the expert mass, not by
FLOPs. The engine now does one pass per prompt where it can: attention is
sub-chunked inside each layer and the MoE runs once, and the expert cache
shrinks out of the way while a big chunk runs.

| Qwen3-Coder-Next 6-bit, 31,510-token cold prompt | Prefill | Read | Peak |
| --- | ---: | ---: | ---: |
| 2048-token chunks (mlx-lm's default step) | 420 tok/s | | |
| four 8192-token passes (0.2.x) | 541 tok/s | 85 GB | 31.6 GB |
| **one pass (0.3)** | **937 tok/s** (34 s) | **44 GB** | **28.7 GB** |

Faster and a lower memory peak at the same time. On top of that, each layer now
starts the next layer's reads before it's done (Qwen3-235B, 512 tokens on
seeded KV: 9.1 s to 6.0 s), and prefill misses read straight into a staging
slab with the native reader, no host copy (tool rounds on a 16k context:
31.0 s to 26.7 s).

**Agent turns don't re-prefill the conversation.** On a real coding-agent
session (~13.5k tokens) turn 1 prefills once, then every later turn reuses
12.4k to 13.3k tokens and prefills only the 340 to 1,115 new ones, in **0.3 to
2 seconds**. Hybrid-attention models (Qwen3-Next style) can branch back to an
earlier message now too, which used to mean prefilling from token 0. And the
system prompt plus tool schemas survive a restart on disk, so a fresh engine
doesn't pay ~40 s for them again.

**Exact multi-token decode.** When the model is copying something from context
(editing a file, repeating a path in a tool call) the engine drafts what comes
next and checks it in one pass. Every emitted token is still the model's own
pick. With prune off that's **2.26x** on a file edit. Details
[below](#exact-tokens-where-they-matter).

**Decode misses use the whole drive.** Each expert miss used to be read by one
worker in 1 MB pieces, so with about one miss per layer the SSD sat at queue
depth 1. Misses are now split across every idle reader. MiniMax-M3 4-bit went
from 5.65 to 9.31 tok/s and Qwen3.8-Flash-Next from 5.92 to 8.66 on the same
prompts.

**More models.** GLM-5.2 and DeepSeek-V3.2 (DSA sparse attention, with a sparse
prefill path that takes ~110 s off a cold 32k prefill on GLM-5.2), MiniMax-M3
(sigmoid routers, its XML tool format and think tags), Qwen3.8-Flash-Next, and
image input for resident Qwen VL chat.

**It doesn't freeze your Mac.** Expert reads bypass the page cache with every
read capped where macOS still honors that, a trimmer drops whatever leaks
anyway, the slab shrinks when other apps need memory and grows back when they
don't, and a watchdog kills a hung engine instead of letting it sit there
holding 40 GB.

Full list in the [CHANGELOG](./CHANGELOG.md).

---

## Quick start

Three steps. After that you use mlx-lm / mlx-openai-server the way you
already do.

### 1. Install

```bash
git clone https://github.com/noahclark556/paged-moe.git
cd paged-moe
./install_mlx.sh
# or: ./install_mlx.sh --python /path/to/venv/bin/python
# or: ./install_mlx.sh --models-dir ~/mlx-models
```

That installs the package into your mlx venv, turns on the `mlx_lm.load` hook,
and writes `~/paged-moe-config.yaml`. By hand:

```bash
source ~/.mlx-env/bin/activate
pip install -e .
paged-moe install --models-dir ~/mlx-models
```

### 2. List the models to stream

`~/paged-moe-config.yaml` decides what streams. Put the real paths in:

```yaml
enable_all: false
machine:
  auto: true
models:
  - path: /Users/you/mlx-models/qwen3-next
  - path: /Users/you/mlx-models/qwen3-235-4bit
    env:
      PAGED_MOE_PRUNE: "0.7"
      PAGED_MOE_WAIT_ABOVE: "0.2"
```

Only listed paths stream. Everything else stays stock mlx-lm. Edit it any
time, no reinstall. You can also drop a `.paged_moe` file in a checkpoint
folder, or `export PAGED_MOE=1` to stream every load. Kill switch:
`PAGED_MOE_HOOK=0`.

### 3. Use mlx like normal

```python
from mlx_lm import load, stream_generate

model, tokenizer = load("~/mlx-models/qwen3-235-4bit")   # streams if listed
for chunk in stream_generate(model, tokenizer, prompt, max_tokens=200):
    print(chunk.text, end="", flush=True)
```

Or run the OpenAI-compatible server and point your agent at it:

```bash
paged-moe-server --model ~/mlx-models/qwen3-235-4bit --port 8080
# http://127.0.0.1:8080/v1, any API key
```

Quick checks:

```bash
paged-moe status                 # hook, config, detected Mac
paged-moe which /path/to/model   # will this stream?
paged-moe machine                # what this Mac resolves to
paged-moe ladder                 # what fits on this Mac
python run.py --model <path> --chat
```

There's also a direct Python API (`from pagedmoe import load, get_stats`) and
a zero-dependency Node wrapper in `node/` that spawns the server, waits for it
and kills it when you're done.

### Who it's for

- Qwen / GLM / DeepSeek / Kimi / MiniMax class MoEs on a Mac that can't hold
  them
- Agent loops, not just one-shot chat
- Plain MLX checkpoints. No converted format, no second copy on disk

### Who it's not for

- Models that already fit in RAM. Stock mlx-lm is enough (PagedMoE just
  loads those resident anyway)
- The smallest possible memory footprint. This spends RAM on a big expert
  cache on purpose, so agent turns stay warm
- NVIDIA / CUDA

---

## How it works

mlx-lm sends all MoE expert compute through one module, `SwitchGLU`. PagedMoE
loads the model lazily, swaps every `SwitchGLU` for a disk-backed version, and
only materializes the backbone. Attention, KV cache, chat templates and
sampling are mlx-lm's own. Experts are read from the unmodified checkpoint,
kept in a byte-budgeted cache, and prefetched ahead of time. Since the math is
the same, the output is the same.

The parts that make it fast enough to use:

- **Route prediction.** Each layer's router is a small resident matrix, so
  the engine runs the next few layers' routers on the current hidden state and
  starts those reads early. On a disk-bound test (64-expert MoE with an
  external-SSD read delay) that was 10.6 to 19.0 tok/s. On a model the cache
  already holds it costs a few percent, so a governor measures both ways and
  keeps whichever wins.
- **Slab storage.** Cached experts live in fixed GPU slots and compute takes
  views into them. +11% decode on Qwen3-235B, +20% on Qwen3-Next.
- **Native reads.** A small C pool does the reads with the GIL released,
  split across readers, with demand misses ahead of speculation. Falls back to
  Python if it can't build.
- **Frequency-aware eviction.** Hot experts get a second chance, so one
  prefill sweep can't flush the working set.
- **Memory that behaves.** KV is priced up front, 8-bit for long contexts,
  and dequantized one layer at a time when a stored conversation is resumed.
- **An optional sidecar.** A tiny online model that learns which experts come
  next and prefetches them. It only acts while decode is measurably faster
  with it on.

None of the speed work changes which experts run. A wrong guess wastes a read,
that's it. The full walkthrough is in [docs/DESIGN.md](./docs/DESIGN.md), and
[docs/why-it-fits.md](./docs/why-it-fits.md) is the short version.

---

## Built for agent loops

Most of the pain with big local models in an agent isn't decode speed, it's
the wait between turns. On a streamed MoE every prefill is a pass over the
expert weights, so re-prefilling a 30k conversation costs 60 to 80 seconds.
The server works hard to never do that.

- **Prefix reuse that survives real clients.** Agent clients rewrite things:
  thinking text that never gets replayed, reminders parked at the tail,
  blocks edited near the front. The server snapshots the KV at the last point
  a prompt diverged from the one before it, so it learns each client's stable
  and volatile parts from real traffic. `PAGED_MOE_PREFIX_DEBUG=1` logs how
  much each request reused and where it diverged.
- **Stable front matter.** Send `stable_prefix_messages: N` and the first N
  messages are pinned. History compaction after that only prefills the new
  history.
- **Front matter on disk.** The pinned prefix (system prompt, tools, codebase
  map, memory) is written to disk and loaded after a restart. A file only
  loads if the weights, config, engine numerics, settings and token ids all
  match and its CRC checks out. Otherwise it's a normal prefill. Hosts that
  encrypt their data can hand the engine a key over a pipe and the files are
  sealed at rest.
- **Branching on hybrid caches.** Linear-attention layers can't be trimmed,
  so a retry or an edited message used to prefill from scratch. The engine
  keeps the small recurrent state at each turn's resume point and restores it
  in about a millisecond, bit-identical to a fresh prefill.
- **Replay with the real ids.** When a turn comes back as text and
  re-tokenizes differently than what decode produced, the engine splices the
  original ids back in so the stored cache still matches.
- **`warm_prefix`.** POST a message list with `max_tokens: 0` and
  `warm_prefix: true` and it gets prefilled and cached without decoding.
  Handy after a task finishes and you collapse history to a summary.
- **Cache breakpoints.** `cache_control` on a message, `cache_breaks: [...]`
  on the request, or `<|cache_break|>` in the text.

All of this is plain OpenAI-compatible JSON. Fields the server doesn't know
are ignored by other servers, so they're safe to send. The host side of the
contract (what not to rewrite, how to compact) is in
[docs/CLIENT_CONTEXT.md](./docs/CLIENT_CONTEXT.md).

---

## Exact tokens where they matter

More tokens per byte read is the one decode lever that doesn't fight the SSD.
Decode on a streamed MoE runs through a verify loop: when the last few tokens
show up earlier in the context, it drafts what followed and checks the draft
in a single pass that shares its expert reads across up to nine positions. With
prune off the model can also draft itself with a cheap step, and checkpoints
with a multi-token-prediction head get a third drafter. Drafts never reach the
output unchecked, and a timing governor only drafts where drafting has
actually measured faster, per span.

Qwen3.8-Flash-Next, greedy, 256 tokens, same warm cache:

| Config | Edit a file | Explain a file |
| --- | --- | --- |
| prune 0, plain | 5.9 tok/s | 7.1 tok/s |
| **prune 0, verify + self-draft** | **13.4 tok/s (2.26x)** | 8.3 tok/s (1.16x) |
| prune 0.7 + flow, plain | 9.8 tok/s | 12.1 tok/s |
| prune 0.7 + flow, exact verify | 15.6 tok/s (1.60x), 99% exact tokens | 12.0 tok/s (governor kept it off) |

Copying a method back verbatim on the same model, at a speed-leaning prune
setting, went from about 15 to **48.8 tok/s**. `bench/verify_ab.py` and `bench/copy_decode_ab.py`
reproduce these. Each reply logs a `[verify] done` line with passes, accept
rate and what the governor is going by.

---

## Results by Mac

Measured in a sandbox that holds the engine to each Mac's RAM, cache budget,
reader count and SSD speed. The GPU is still the M5 Pro, so on a real chip of
that class tok/s can be a bit off. The 64 GB and 128 GB rows are IO-scaled
estimates, not runs. Prune / wait match the table above.

| Mac | Model | Checkpoint | Peak RAM | Prefill | Decode |
| --- | --- | ---: | ---: | ---: | ---: |
| 16 GB Air (M2 / M3 / M4) | Qwen3-Coder-Next 6-bit | 65 GB | 8.4 GB | 172 tok/s | 10.8 tok/s |
| 16 GB Air (M2 / M3 / M4) | KAT-Coder-V2.5 6-bit | 30 GB | 10.7 GB | 450 tok/s | 20.1 tok/s |
| 18 GB M3 Pro | Qwen3-235B-A22B 4-bit | 132 GB | 8.3 GB | 119 tok/s | 8.6 tok/s |
| 24 GB M4 Pro | Qwen3-235B-A22B 4-bit | 132 GB | 12.6 GB | 108 tok/s | 9.1 tok/s |
| 32 GB M2 Pro | GLM-4.7 4-bit | 199 GB | 18.3 GB | 77 tok/s | 6.5 tok/s |
| 36 GB M3 Pro | GLM-4.7 4-bit | 199 GB | 21.0 GB | 75 tok/s | 7.3 tok/s |
| 48 GB M4 Pro / M4 Max | Qwen3-Coder-Next 6-bit | 65 GB | 30.7 GB | 220 tok/s | 23.4 tok/s |
| 48 GB M4 Pro / M4 Max | Qwen3-235B-A22B 4-bit | 132 GB | 28.6 GB | 103 tok/s | 9.0 tok/s |
| 64 GB M5 Pro (est.) | GLM-4.7 4-bit | 199 GB | ~45 GB | ~73 tok/s | ~7.9 tok/s |
| 128 GB M5 Max (est.) | Qwen3-235B-A22B 4-bit | 132 GB | ~85 GB | ~115 tok/s | ~9.5 tok/s |

Yes, that's a 132 GB model on an 18 GB envelope. The expert cache there is 1.6
GB, so it leans entirely on the SSD, and it still decodes. 16 / 18 / 24 GB
can't fit GLM-4.7's backbone plus headroom, so it starts at 32 GB.

`paged-moe ladder` prints the fit table for your Mac, and `paged-moe machine`
shows the cache, threads and headroom it picked.

---

## The hard parts

This didn't come out of the box working. Some of what it took, and what's
still rough.

- **The 235B started at about 2 tok/s.** Profiling got it to 2.6 at identical
  output. Slab storage, route prediction and the read path got it to 6 and up.
  It's still disk-bound: a token needs ~8 GB of expert weights and misses
  ~1.5 GB of it, so the SSD sets the pace.
- **It froze the machine.** First it was mlx-lm wiring nearly all of RAM. Then
  mmap reads double-counting every expert in the page cache. Then `F_NOCACHE`,
  which macOS quietly ignores for reads over 1 MB, so a long prefill filled the
  file cache and the kernel started compressing the slab (a 16 second system
  stall). Each one is fixed and has a guard now.
- **Long prompts crashed.** A 67,586-token prompt used to take the server
  down. It prefills in three chunks now (91 s) with the server alive.
- **A quantized KV reuse bug produced confident, fluent, off-topic text.** That
  one was nasty because nothing errored. There's a regression bench for it.
- **Speculative decoding with a draft model is a 2 to 3x regression on the
  235B.** A verify batch needs each layer's expert union, which thrashes the
  cache. N-gram drafting won on a synthetic prompt and lost on agent traffic.
  What finally worked is the governed verify loop above, which only drafts
  where it measures faster.
- **Pruning can break text.** Aggressive settings on the 235B once produced
  garbage, and GLM-5.2 at its fastest setting can fall into a loop. The engine
  now runs tool calls and turn starts at stricter settings, gates critical
  misses, and has a loop rescue. It's still a fidelity knob, not a free
  speedup.
- **The GPU hung.** A Metal wait that never returned held 40 GB with no error,
  and once a killed process left 30 GB held by the driver until a reboot. The
  watchdog exits on stalls, and load checks for memory the driver is still
  holding before it plans a budget.
- **Small prefills are still expensive.** A few hundred tokens of tool result
  routes to ~60% of the expert mass, so on Qwen3.8-Flash-Next even a short
  result reads 46 to 82 GB. It's at the drive's speed now, but it's the main
  thing between turns that's left.
- **Some ideas just didn't pay.** The sidecar's prefill head, lookup decode on
  agent traffic, batched prefill gathers, aligned reads. Measured, left off.
- **The very biggest are slow at full fidelity.** Qwen3-Coder-480B with prune
  off ran at about 1 tok/s in September. It runs. I wouldn't call it usable.

---

## What is exact and what isn't

- **Default (prune off): bit-identical** to a fully resident load. Prefetch,
  prediction, the sidecar, slabs, staging and verify drafts only change timing,
  never which experts a layer computes with. The tests check this against
  resident runs, including a starved cache.
- **`PAGED_MOE_PRUNE` / `PAGED_MOE_WAIT_ABOVE`** skip low-weight experts when
  they'd cost a disk read. Faster, and an approximation. Tool calls and the
  first 64 tokens of a turn run stricter on their own.
- **Verify** samples from the same model plain decode does at whatever prune
  you set. `PAGED_MOE_VERIFY_FIDELITY=exact` makes every emitted token the
  exact model's pick even with prune on, at the cost of heavier passes.
- **The sidecar's prune head** is the only head that changes logits, and it's
  off by default.

---

## Compatible models

Anything mlx-lm loads whose MoE layers use `SwitchGLU`, which is nearly all of
them:

| Family | Notes |
| --- | --- |
| Qwen3-MoE / Coder / 235B / 480B | 235B-A22B is the everyday stress test |
| Qwen3-Next, Qwen3.8-Flash-Next | Hybrid attention, branching and replay handled |
| GLM-4.5 / 4.6 / 4.7 | GLM-4.5-Air is a great fit for 48 GB |
| GLM-5.2, DeepSeek-V3.2 | DSA sparse attention, sparse prefill path |
| DeepSeek-V2 / V3, Kimi-K2 | MLA. Kimi needs `tiktoken` (installed by default) |
| MiniMax-M2 / M3 | Sigmoid routers, XML tool calls, `<mm:think>` tags |
| Mixtral, OLMoE, others | If it's SwitchGLU it should stream |

When a checkpoint ships an architecture file mlx-lm doesn't have yet (MiniMax-M3
does), the engine registers it for that process instead of asking you to copy
it into mlx-lm.

No custom format. Keep the checkpoints on the internal SSD when you can.
Decode is mostly disk bandwidth, and Thunderbolt drives are several times
slower.

---

## Config

Per-model knobs go under `env:` in `~/paged-moe-config.yaml`, or in the
process environment. Every knob is documented at the top of its block in
`pagedmoe/config.py`. The ones worth knowing:

| Knob | Default | What it does |
| --- | --- | --- |
| `PAGED_MOE_PRUNE` | `0` | skip weak experts that would need a read (fidelity knob) |
| `PAGED_MOE_WAIT_ABOVE` | `0` | only stall on misses above this mixture weight |
| `PAGED_MOE_CACHE_GB` / `_MAX_CACHE_GB` | auto / `26` | expert cache size and its ceiling |
| `PAGED_MOE_PREDICT` | `auto` | route-prediction prefetch, governor picks on or off |
| `PAGED_MOE_VERIFY` | `1` | verify decode on streamed models |
| `PAGED_MOE_VERIFY_FIDELITY` | `plain` | `exact` = every token is the exact model's pick |
| `PAGED_MOE_SELF_DRAFT` | `auto` | self-drafting, only when prune is 0 |
| `PAGED_MOE_MTP` | `1` | MTP head as a drafter when the checkpoint has one |
| `PAGED_MOE_FUSED_PREFILL` | `1` | attention sub-chunks, one MoE pass per chunk |
| `PAGED_MOE_PREFILL_CHUNK` | `32768` | max tokens per prefill pass |
| `PAGED_MOE_PREFILL_AHEAD` | `1` | start the next layer's prefill reads early |
| `PAGED_MOE_PREFIX_DISK` | `1` | pinned front matter KV on disk |
| `PAGED_MOE_HYBRID_CKPT` | `1` | recurrent-state checkpoints for branching |
| `PAGED_MOE_NATIVE_READ` | `1` | C read pool, falls back to Python |
| `PAGED_MOE_NOCACHE` | `1` | bypass the page cache. Leave it on |
| `PAGED_MOE_MEM_PRESSURE` | `1` | shrink and regrow the slab with macOS memory pressure |
| `PAGED_MOE_WATCHDOG` | `1` | exit on a stalled request or sustained critical pressure |
| `PAGED_MOE_SIDECAR` | `0` | the online prefetch sidecar (wrap head on once enabled) |
| `PAGED_MOE_EARLY_DECODE` | `0` | prefetch experts common right after prefill |
| `PAGED_MOE_DOWNLOAD_DIR` | `~/models/paged-moe` | where HF repo ids download to |

Recipes used for the table up top:

```yaml
# Qwen3-235B / Qwen3-Coder-480B
env:
  PAGED_MOE_PRUNE: "0.7"
  PAGED_MOE_WAIT_ABOVE: "0.2"
  PAGED_MOE_SIDECAR: "1"

# GLM-4.7 / DeepSeek-V3.2 / Kimi-K2
env:
  PAGED_MOE_PRUNE: "0.8"
  PAGED_MOE_WAIT_ABOVE: "0.2"
  PAGED_MOE_SIDECAR: "1"
```

Leave prune unset for bit-identical decode.

### Sidecar warm-start

The sidecar learns online during normal decode. To warm it from a prompt
corpus first:

```bash
paged-moe-pretrain --quick --model ~/mlx-models/qwen3-235-4bit --corpus prompts.jsonl
paged-moe-pretrain --fit-only --model ~/mlx-models/qwen3-235-4bit   # refit from what live use recorded
paged-moe-pretrain --unlock-all
```

Weights live under `~/.paged_moe/sidecar/` (`PAGED_MOE_SIDECAR_DIR` to move
them). More in [docs/tinymodel-improvements.md](./docs/tinymodel-improvements.md).

### CLI

| Command | Does |
| --- | --- |
| `paged-moe status` | hook, config paths, detected Mac |
| `paged-moe which <model>` | will this path stream |
| `paged-moe machine` | hardware plus derived cache / threads / headroom |
| `paged-moe ladder` | per-Mac fit table |
| `paged-moe install` / `uninstall` | the `mlx_lm.load` hook |
| `paged-moe-server ...` | OpenAI-compatible server |
| `paged-moe-pretrain ...` | sidecar warm-start / fit / unlock |
| `python run.py --model ... --chat` | quick REPL from a clone |

---

## Upgrading from 0.2.x

The import package is `pagedmoe` now (was `expert_stream`) and every env knob is
`PAGED_MOE_*` (was `EXPERT_STREAM_*`). Old names are not read, so rename them
in `~/paged-moe-config.yaml` or they're silently ignored.

```bash
paged-moe uninstall            # old hook, before the package goes away
pip uninstall -y paged-moe
git pull
./install_mlx.sh
sed -i '' 's/EXPERT_STREAM_/PAGED_MOE_/g' ~/paged-moe-config.yaml
```

If you skip the first line, Python prints an error about
`zz_paged_moe_hook.pth` at startup until the reinstall rewrites it. Harmless,
just noisy.

---

## Tests and benches

```bash
pip install -e ".[dev]"
python -m pytest                   # most run in seconds on a tiny synthetic MoE
python tests/test_real_model.py    # real MoE, ~4 GB download the first time
```

Tests that need a specific checkpoint skip when it isn't there. Run them with
no `PAGED_MOE_*` exported, since a pruning setting makes decode approximate on
purpose. `bench/` has the probes behind the numbers in this README.

`walkthrough/tour.py` runs one real request and drops you into pdb at each
major engine stage, if you want to see it work from the inside.

---

## License (dual)

| | |
| --- | --- |
| Open source / research / AGPL use | [GNU Affero GPL v3](./LICENSE) |
| Closed product / SaaS / OEM | Commercial license, email `noah@qwertycode.org` |
| Contributors | [CLA](./CLA.md) + [DCO](./DCO.md), see [CONTRIBUTING.md](./CONTRIBUTING.md) |

Free under AGPL-3.0 for people and open projects. If you're shipping a closed
product, a hosted service or an appliance and don't want to publish your source
under the AGPL, email me for a commercial license. More in
[LICENSING.md](./LICENSING.md) and [COMMERCIAL_LICENSE.md](./COMMERCIAL_LICENSE.md).

Copyright (c) 2026 Noah Clark.

## Citation

```bibtex
@software{clark_paged_moe,
  author = {Clark, Noah},
  title = {PagedMoE: Page MoE experts from disk on Apple Silicon},
  year = {2026},
  url = {https://github.com/noahclark556/paged-moe}
}
```

## Contact

- Commercial / dual license: `noah@qwertycode.org`
- Bugs and features: GitHub issues
