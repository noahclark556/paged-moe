# How paged-moe works

A walkthrough for reading the code. Files are listed in the order data flows
through them. This covers the core paging path. The newer pieces (verify
decode, front matter on disk, hybrid checkpoints, the prefill stage, the
watchdog) are summarized in the README and documented in their own module
docstrings.

## The one-paragraph version

MoE models spend ~95% of their bytes on routed experts but touch only a few
per token. mlx-lm funnels *all* expert compute through one module class
(`SwitchGLU`), so we load the model with lazy (unmaterialized) weights, swap
every `SwitchGLU` for our disk-backed `StreamedSwitchGLU`, and evaluate only
what's left - the backbone. Experts are read from the *unmodified* checkpoint
at inference time, cached in unified memory under a byte budget, and
prefetched speculatively. Output is bit-identical to a fully-resident load
because the math is identical - only the storage location of the weights
changed.

## Why not a from-scratch engine?

The first prototype here was a hand-rolled numpy/MLX MoE with a custom
`weights.bin` format. That approach dead-ends fast: a real model needs
attention (and new models keep shipping new kinds of it), KV caches, RoPE variants, tokenizers, chat templates, quantized
kernels... reimplementing all of it means months of work and a permanent
porting treadmill. The insight that collapsed the problem: **every mlx-lm MoE
model isolates expert weights behind a single seam**, and streaming is purely
a storage concern. Hook the seam, inherit the ecosystem.

## Data flow

```
 checkpoint dir (plain MLX safetensors, on any disk)
   |
   |-- safetensors_index.py   parse headers -> byte range of every tensor,
   |                          and of every individual expert within them
   |
   |-- loader.py  load()      lazy-build model -> measure -> pick mode
   |      |                   resident: eval everything, done (plain mlx-lm)
   |      |                   streamed: v
   |      |-- streaming.py    patch_model(): SwitchGLU -> StreamedSwitchGLU
   |      `-- mx.eval(...)    materializes ONLY the backbone (experts were
   |                          dropped before ever being read)
   |
   `-- at inference, per MoE layer:
         router picks top-k experts -> StreamedSwitchGLU.__call__
           -> cache.fetch(layer, ids)     hits: free; misses: parallel pread
           -> expert matmuls (two paths, see below)
           -> prefetch ring warms next layers' likely experts
```

## File by file

### `safetensors_index.py` - stream from standard checkpoints, no conversion

Safetensors = 8-byte header length + JSON header + raw contiguous tensors.
`read_headers()` gives every tensor's `(file, offset, nbytes, dtype, shape)`
without touching weight data.

Two checkpoint layouts exist in the wild, both handled with zero conversion:

- **stacked** (most mlx-community repos): one tensor per projection per layer,
  shape `[n_experts, ...]`. Row-major => expert `e` is one contiguous slice - `TensorLoc.expert_slice()` computes its byte range. `StackedLocator`.
- **per-expert** (HF-style, e.g. the OLMoE repo): separate small tensors
  `...mlp.experts.<e>.gate_proj.weight`. Already expert-granular.
  `DirectLocator`.

`StackedLocator.span(first, count)` addresses a *run* of consecutive experts
as one contiguous byte range - the basis for coalesced prefill reads (below).

Reads go through one shared read-only fd per file (`FilePool`) using
`pread()` with **`F_NOCACHE`** set: expert reads bypass the macOS page cache
entirely. This is deliberate - an earlier mmap-based version let every expert
byte land in the kernel file cache *in addition to* our own cache, and a
prefill pass over a long prompt (which reads nearly the whole expert mass)
ballooned the page cache by tens of GB and drove the machine into swap. With
`F_NOCACHE`, expert bytes exist in exactly one place in RAM: the
`ExpertCache`. RAM usage is deterministic. (`PAGED_MOE_NOCACHE=0`
restores page-cached reads if you ever want them.)

bfloat16 detail: numpy has no bf16, so BF16 tensors are read as uint16 and
reinterpreted with `mx.array(...).view(mx.bfloat16)`.

### `cache.py` - the residency policy

One `ExpertCache` shared by all layers. Key = `(layer_path, expert_id)`,
value = dict of ready-to-use mx arrays (a hit costs nothing). Evicted by
**byte budget**, so it adapts to any model/quantization.

Eviction is LRU **with a second chance for frequently reused experts**. Expert
use is heavily skewed: a few experts per layer serve almost every token while
the long tail serves one. Under pure LRU, a single sweep of tail traffic evicts
the hot set and every later token pays disk for experts it immediately wants
again. So each entry carries a small use counter (incremented on reuse, capped
at 8); an entry at the LRU tail with a non-zero counter halves it and returns to
MRU instead of being evicted, with the scan bounded so a uniformly hot cache
still makes progress. Protection is earned by *reuse*, not by installation,
which is what makes it resistant to a one-pass prefill sweep. Worth ~10% decode
throughput and ~7% fewer bytes read (`PAGED_MOE_LFRU=0` to compare).

Entry points:
- `fetch(layer, ids, install=)` - blocking, called by the layer that needs
  experts *now*. All misses dispatch to a thread pool at once; SSDs need
  queue depth (16 readers by default) to reach their rated bandwidth.
- `fetch_groups(layer, ids, group_bytes)` - the prefill path: yields experts
  in bounded-size groups while the reads for the *next* group are already in
  flight (GPU and SSD stay busy simultaneously).
- `prefetch(layer, ids)` - fire-and-forget, for speculation. In-flight
  dedup ensures a prefetch and a fetch of the same expert share one read, and
  surviving ids are run-coalesced like the fetch path. (Done-callbacks are
  attached *outside* the cache lock: an already-complete future runs its
  callback inline on the calling thread, and the callback takes the lock - attaching under the lock self-deadlocks.) Speculation yields to demand:
  prefetch is a no-op while more than `2 x read_threads` reads are in flight,
  since adding queue depth would delay the read a layer is blocked on. That
  guard is what keeps a cache-starved model at parity instead of regressing.

**Coalesced run reads**: when `_begin` classifies misses, consecutive expert
ids of a stacked layer are grouped into runs and each run is read with one
`pread` per component tensor (`_read_expert_run` via `StackedLocator.span`),
capped at ~32 MB per read so the reader threads still get parallel work.
Prefill needs nearly every expert of a layer, and `np.unique` hands us sorted
ids - so prefill's read pattern becomes a handful of large sequential reads
per layer instead of hundreds of ~1 MB scattered ones. Each per-expert future
is resolved from the shared run read, so dedup with `prefetch` keeps working.

Install modes: `"lru"` (normal insert + eviction - decode) and `"soft"`
(insert only while there's free room, never evict - prefill). Soft install is
what stops a long prompt, which touches nearly every expert, from churning
the entire cache once per layer and destroying the decode working set.

**Prefill runs on a shrunk budget** (`enter_prefill` / `leave_prefill`). A
prefill chunk routes to ~every expert of every layer, so it reads the whole
uncached expert mass *per chunk*: the cost is set by the number of chunks, not
their size. Since nothing is revisited within one pass, a large cache buys
nothing there - but the memory it occupies is exactly what prevents processing
the prompt in one big chunk. So chunks >= `PREFILL_SHRINK_TOKENS` (8192) drop
the budget to `PREFILL_CACHE_GB` (4), which makes 32k-token chunks fit. Decode
restores the full budget and re-warms as it reads. Smaller chunks (the cached
agent turn's delta) keep the full cache, because there the resident experts are
precisely what makes the delta cheap.

Eviction drops Python refs *and* (throttled, every ~2 GB evicted) calls
`mx.clear_cache()` so Metal actually returns pages to the OS.

The cache hierarchy is intentionally flat: `ExpertCache` (unified memory),
then the SSD. No page cache in between (see `F_NOCACHE` above).

### `streaming.py` - the seam

`StreamedSwitchGLU` matches `SwitchGLU`'s contract exactly:
input `x[..., D]` + router `indices[..., K]` -> output `[..., K, D]`.

Per call:
1. `np.asarray(indices)` - **the one unavoidable sync point**: the router's
   choice must reach the CPU before we know what to read from disk. (One tiny
   GPU->CPU sync per MoE layer per token, irrelevant next to disk time.)
2. Fetch and compute, one code path, two fetch regimes:
   - **decode** (<= 4 tokens): `cache.fetch(..., install="lru")`, then a
     per-expert loop of `mx.quantized_matmul` running *directly on the cached
     arrays* - zero weight copies, zero stacking. (An earlier version kept
     "sticky" stacked buffers per layer; on GLM-Air-sized experts those
     duplicated ~3.6 GB of cache memory and were removed.)
   - **prefill** (long prompts light up most experts): experts stream
     through in <= `PAGED_MOE_GROUP_MB` groups via `fetch_groups`
     (install="soft"), each group's matmuls forced with `mx.eval` so its
     weights are freed before the next group lands. Tokens are grouped
     per expert once per call (single argsort), so each expert does one
     batched matmul over all rows routed to it.
3. During decode, `PrefetchRing.speculate()` warms the layers about to run, so
   their experts arrive while the GPU is still busy with this one. Two sources:

   **Route prediction (primary).** Every layer's router is a small matrix in the
   resident backbone, so instead of guessing what layer `i+1` will want we simply
   *ask it*, feeding it layer `i`'s hidden state - a good estimate of its own
   input, since the two differ by one attention block and one FFN on a residual
   stream. This is the same routing decision that layer makes a millisecond
   later, which is why it holds up exactly where token-similarity heuristics
   break: the first token of a reply, a topic change, a long wandering
   generation. Measured coverage of real demand is 0.80-0.89, against ~0.45 for
   the previous-token heuristic, worth **+79% decode throughput when reads are
   the bottleneck** (a 64-expert MoE with simulated read delay; on real
   checkpoints the gain depends on how disk-bound the model is).

   Predictions are issued on a stride of `depth` rather than at every layer, so
   each layer is predicted exactly once with a lead of 1-3 layers. Predicting at
   every layer predicts each layer `depth` times, and the repeats cost a GPU sync
   and a pass of host work to re-derive ids already queued (3x the overhead for
   less throughput). All depths of one stride are resolved under a single
   CPU sync, because syncs are the expensive part of this path.

   **This win is conditional, so it is measured rather than assumed.** Hiding a
   read behind compute only helps if the read is the bottleneck. On unified
   memory the reader threads and the GPU share bandwidth, so when experts already
   arrive at near-RAM speed - a small model whose files the drive caches - or the
   drive is already saturated, speculative reads contend with the kernels instead
   of hiding behind them: measured 0.73 -> 0.90 hit rate, disk wait halved, and
   still 6% *slower* overall. `PAGED_MOE_PREDICT=auto` (the default)
   therefore alternates prediction on and off for 8-token windows, keeps whichever
   measured faster, and re-probes every 32 windows in case conditions drift. It
   picked correctly in every regime tested; `PREDICT=1` forces it on.

   Router shapes differ by family and both are handled generically: `nn.Linear`
   routers (Qwen3-MoE, Qwen3-Next) return logits, and since softmax is monotonic
   the top-k of the logits is the same set the block would pick - plus `slack`
   extra, because the experts just outside the cut are the ones a one-block-old
   hidden state most often gets wrong. `MoEGate` routers (DeepSeek-V3, GLM-4-MoE)
   already return selected indices, correction bias and group top-k included, so
   they are used as-is. An unfamiliar layout simply disables prediction for that
   layer. All depths are predicted under **one** CPU sync, because syncs are what
   cost on this path, not the matvecs.

   **Previous-token routing (fallback).** For layers beyond prediction depth, and
   for architectures where no router was found, the ring replays what each layer
   used for the previous token. It also wraps past the last layer to warm the
   first layers of the *next* token. Prefill seeds this with the last prompt
   token's routing.

   Both sources only call `prefetch()`, and the layer always computes with what
   its own router actually selected - so prediction changes timing, never values.
   Every A/B in `bench/` checks a greedy token checksum for exactly this reason.

`patch_model()` walks the module tree, and for each `SwitchGLU`:
captures its quantization spec (`group_size/bits/mode`) and activation,
registers the layer's disk layout with the cache, finds the layer's router for
prediction, and swaps the module via `setattr` on its parent. The router
reference is held in a plain object, not as a module attribute, so it does not
enter the model's parameter tree a second time. Because the model was loaded `lazy=True`, the big
expert arrays are discarded **without ever being read**.

### `loader.py` - policy and public API

`load()` is a drop-in for `mlx_lm.load()` (same signature + streaming
kwargs). Sizing comes from headers only. Mode decision:

    resident  if total_bytes + 6 GB headroom <= RAM_FRACTION x RAM
    streamed  otherwise (requires MoE layers; a too-big dense model errors)

Auto cache budget = RAM budget - backbone - 6 GB headroom (KV cache,
activations), hard-capped at `PAGED_MOE_MAX_CACHE_GB` (26).

That cap dominates how an agent session feels. Every turn appends a few
hundred fresh tokens, and a short prefill chunk routes across nearly every
expert, so the turn re-reads the whole non-resident expert mass: the pause
before each edit is `(expert mass - cache) / bandwidth` and is almost
independent of how many tokens arrived. On qwen3-next (40.5 GiB experts,
1.26 GiB backbone) over 4 steady-state turns of ~750 fresh tokens:

| cache | read/turn | prefill/turn | hit rate | decode |
|---|---|---|---|---|
| 18 GB | 16.7 GB | 2.87 s | 0.52 | 14.3 tok/s |
| 26 GB | 9.9 GB | 2.29 s | 0.65 | 18.6 tok/s |
| 32 GB | 4.8 GB | 1.87 s | 0.75 | 23.0 tok/s |

Measure this with `bench/turn_probe.py`, not a single long generation - one
generation amortizes the sweep that an agent session pays every turn.

Memory discipline (all in `_set_memory_limits` / `_neutralize_wired_limit`):
- `mx.set_cache_limit(max(2 GB, 384 experts' worth, cap 6 GB))` - MLX's
  recycled-buffer pool stays bounded, but *large enough to recycle one decode
  token's expert churn*: expert tensors are uniform sizes, so a big-expert
  model that misses gigabytes per token re-uses last token's buffers instead
  of paying zero-fill page faults on every miss. Same logic scales the
  `mx.clear_cache()` cadence (cache budget / 3) and the prefetch staging cap
  (256 experts' worth) - all three were fixed constants tuned on ~2 MB-expert
  models and quietly throttled 10 MB-expert models like Qwen3-235B.
- `mx.set_memory_limit(backbone + cache + headroom)` - soft ceiling.
- **The Metal wired limit is forced to 0 and mlx-lm's `wired_limit` context
  manager is no-op'd.** mlx-lm pins RAM up to Apple's "max recommended
  working set" around every generation; on a machine where
  `iogpu.wired_limit_mb` has been raised, that is ~all of physical RAM.
  Wired memory can't be swapped - one heavy generation could freeze macOS
  outright. A streaming engine cycles GB of weights per token; it must pin
  nothing.

`get_stats(model)` -> mode, sizes, cache hit rate, GB read, disk wait, Metal
memory. Wired to `run.py --stats` and importable anywhere.

### `server.py` - API surface

mlx-lm ships a production OpenAI-compatible server; its `ModelProvider`
calls a module-level `load` symbol. We substitute ours. Chat templates, SSE
streaming, prompt caching, all mlx-lm server flags: inherited. On top:

- streamed models generate with an **8-bit quantized KV cache** (a 32k-token
  fp16 KV cache for GLM-4.5-Air is ~10 GB; 8-bit halves it, quality impact
  is negligible)
- `relieve_pressure()` after every response (drops prefetch staging +
  returns Metal pages, keeps the expert LRU warm)
- memory-safe flag defaults (only when you don't pass the flag yourself):
  `--prompt-cache-size 4 --prompt-cache-bytes 6G --prefill-step-size 8192
  --prompt-concurrency 1 --decode-concurrency 4`
- streamed models are forced onto the single-request path - the batch engine
  would bypass all of the above and prefill several prompts at once
- **prefix-snapshot prompt caching**: mlx-lm's prompt cache only reuses a
  stored sequence that is a byte-exact prefix of the new prompt, and
  hybrid-attention models (qwen3-next) can't trim their cache backwards - so
  any divergence means re-prefilling the whole conversation. Thinking
  templates diverge at the tail (cached reasoning vs replayed content), and
  agent clients re-park volatile blocks there too. `_snapshot_user_segment`
  prefills to each boundary from `_snapshot_boundaries` (<=2 per request),
  stores the cache under that prefix, and hands the remainder to
  `stream_generate`. The boundaries are the **last divergence point** against
  the previous prompt - which learns the client's stable/volatile split from
  real traffic and keeps advancing every turn - plus the think tail, with a
  message-boundary heuristic covering the first turn. `PAGED_MOE_PREFIX_DEBUG=1`
  logs reuse and decoded divergence context; that is the tool for diagnosing a
  client that rewrites its own history.

### `node/` - Node as the front door

`client.mjs` is a dependency-free client + supervisor: `ExpertStream.launch()`
spawns `python -m pagedmoe.server`, polls `/health`, returns a client
with `chat()` / `chatStream()` / `close()`. `cli.mjs` is a thin terminal
wrapper. Anything that speaks OpenAI (openai npm, LangChain, etc.)
can also just target `http://host:port/v1` directly.

## Correctness guarantees

`tests/test_stream.py` builds a tiny random quantized MoE and asserts
resident, streamed, and streamed-with-starved-cache (constant eviction)
produce identical greedy tokens. `tests/test_real_model.py` does the same on
a real 64-expert model with the cache capped below the working set. Identical
tokens => the streaming layer is invisible to the math.

## Performance model

Decode cost ~ backbone compute + (expert bytes needed x miss rate) / SSD
bandwidth. Levers, in order of impact:

1. **Cache size** (`PAGED_MOE_CACHE_GB` / `RAM_FRACTION`) - drives hit
   rate. Routing is heavy-tailed, so hit rate is much better than
   cache_fraction alone would suggest.
2. **Quantization bits** - 3-bit shrinks both disk traffic and cache
   footprint by 25% vs 4-bit.
3. **SSD bandwidth** - internal NVMe (~6-7 GB/s) > TB4/USB4 external
   (~2.5-3.5 GB/s). Queue depth matters: keep `READ_THREADS` >= 8.
4. **Prediction depth** (`PAGED_MOE_PREDICT_DEPTH`) - how many layers ahead
   the routers are consulted, and therefore also the stride. Measured decode
   tok/s at a 41% cache ratio and 300 us read latency: 13.7 (d1), 16.4 (d2),
   18.5 (d3), 18.7 (d4), 19.1 (d6). Coverage *falls* with depth (0.87 -> 0.71)
   while throughput rises, because a wider stride cuts the cost of predicting
   faster than accuracy declines. 3-6 are within 3%; the default is 3.
5. **Approximate routing** - the quality/speed dial for models whose per-token
   demand dwarfs the cache. Three selectors decide which of the router's K
   experts a decode token actually computes, and they compose (intersection):
   `PAGED_MOE_PRUNE` drops experts weighted below `t x` the winner,
   `PAGED_MOE_ROUTE_TOP_K` caps the count outright, `PAGED_MOE_ROUTE_TOP_P` keeps
   the fewest experts covering `p` of the mixture mass. Prefill always computes
   the full mixture, so prompt states stay exact.

   Works for both router families: `nn.Linear` logits (Qwen3-MoE / Qwen3-Next)
   and `MoEGate` `(inds, scores)` (GLM / DeepSeek). Weights are always reduced
   to unit mass before the selectors run, so a `WAIT_ABOVE=0.2` gate means
   "20% of the mixture" on either architecture - MoEGate's
   `routed_scaling_factor` is divided out, not treated as absolute weight.

   The thing that makes any of them cheap is `PAGED_MOE_KEEP_FREE` (default
   on): an expert that is resident, staged, or already in flight needs no disk
   read, so dropping it buys nothing and is never done. At a decode hit rate of
   ~0.9 that confines the entire quality cost to the ~10% of demand that would
   have blocked on the SSD - and it means a *harder* threshold with rescue beats
   a soft one without. Measured on Qwen3-235B (2k ctx, vs the full mixture at
   3.17 tok/s / ppl 4.90): prune 0.5 without rescue 5.74 tok/s at ppl 5.17;
   prune 0.7 with rescue 6.96 tok/s at ppl 5.13. Faster and closer to exact.

   `PAGED_MOE_WAIT_ABOVE` closes the loop from the other side. Even with the
   selectors set, a surviving expert that misses the cache stalls the GPU on a
   ~10 MB read, and that wait is the largest single cost in a streamed decode.
   The gate says "only wait when the missing expert carries at least this much
   of the mixture"; anything below it is skipped and prefetched for later
   tokens. On Qwen3-235B at prune 0.7, `0.2` measures 7.09 -> 8.22 tok/s at
   *identical* token agreement (91.1%). Never waiting reaches 8.34 but drops to
   85.9% - the weight gate, not the skipping, is what makes this cheap.

   The corollary of both is that the mixture now depends on cache state, so the
   same prompt can decode slightly differently depending on session history. Set
   `PAGED_MOE_KEEP_FREE=0` when reproducibility matters more than either.
6. **Speculative decoding** (`PAGED_MOE_DRAFT_MODEL`,
   `PAGED_MOE_DRAFT_TOKENS`) - supported, greedy-identical, and off by
   default because on a disk-streamed MoE its payoff is decided by content,
   not by acceptance. A verify batch of k+1 positions needs each layer's
   expert *union*: when consecutive positions route alike (repetitive text)
   the union stays small and batching wins (+43% measured); when they diverge
   (reasoning, agent traffic) the union's working set exceeds the cache and
   throughput falls 2-3x. Acceptance (~70%) and tokens-per-round (~3.0) are
   the same in both regimes, so measure bytes and working set, not
   acceptance, before enabling it. Verify batches ride the decode path via
   `_DECODE_MAX_TOKENS = 12`, and prefetch state is kept per-position (see
   `streaming.py`) or speculative reads multiply by k+1.

**Measuring any of this requires a disk-bound run.** A checkpoint small enough
for the drive to cache is not disk-bound, and every streaming lever then measures
as a small loss. `bench/predict_probe.py --read-delay-us` adds latency per read
to stand in for the SSD the deployment actually reads from; without it, or a
model too large to cache, throughput comparisons are measuring the wrong machine.

## Slab storage (slot-addressed residency)

`slab.py` holds resident experts as slots in one contiguous tensor per
component (`[slots, *expert_shape]`) instead of one mx array per expert. Two
costs vanish at once, which is why they are one change and not two:

* **The copy.** Reads pread straight into the slot's bytes through a writeable
  numpy view of Metal memory, so there is no `mx.array(np_buf)` host->device
  copy and no materialization step (`_materialize` returns the slot number).
* **Per-expert dispatch.** A layer becomes one `mx.gather_qmm` per projection
  with `rhs_indices` = each row's slot, replacing 3 calls per expert. Every
  output row is its own batch element, so a variable number of rows per expert
  is free and the sort/unsort `_run` needs disappears.

Measured: +11% on Qwen3-235B at equal memory (5.5 -> 6.1 tok/s; it is
disk-bound, so most of the compute win hides behind the SSD) and +20% on
qwen3-next (16.4 -> 19.7 tok/s, which is *less* disk-bound and so had more
dispatch overhead to give back). Isolated, the compute itself is 5.3x faster
(1.72 -> 0.32 ms/layer at 235B shapes).

Writing into MLX-owned memory is outside MLX's contract, so three properties
carry it, all asserted in `bench/slab_probe.py` (bit-identical gather output):
unified memory makes CPU writes visible to the GPU; slabs are allocated and
evaluated once, so later ops read the buffer at eval time; and a slot is only
written while it holds no live expert.

That last one is the whole correctness argument, and it rests on **slot reuse
being destructive** - unlike an mx array, whose buffer MLX keeps alive as long
as a graph references it. Two rules enforce it:

* Every MoE layer syncs on its own router *before* it fetches (it cannot know
  what to read otherwise), and that sync forces all earlier layers' expert
  matmuls - transitively across tokens, since a token's input depends on the
  previous token's last layer. So by the time a fetch can evict, everything
  previously fetched has been consumed.
* Slots the in-progress fetch depends on are **pinned**, and pinning also
  protects a slot whose read a fetch is already waiting on. That second case
  is subtle and was a real 20%-flaky corruption: prefetch's completion
  callback runs on a worker thread and can hit the staging-overflow path
  first, freeing the slot of a read that `_finish` is about to install.
  `PAGED_MOE_SLAB_VERIFY=1` checks both invariants at compute time and is
  how that was found - reach for it before reasoning, since the failure is
  silently wrong output, not an exception.

Slabs are committed at load and cannot be *lent* to prefill activations the
way a soft byte budget could - a 32k chunk wants >10 GB, which no 48 GB
machine has spare next to a 24 GB slab (it OOMs Metal outright). So a big
prefill hands the whole allocation back and decode rebuilds it
(`enter_prefill`/`leave_prefill`). Residency is lost either way: the
per-expert path shrinks to `PREFILL_CACHE_GB` at the same point and for the
same reason.

Three things make that handover safe, each of which failed in a real session:

* **Drain before freeing.** Slot reads `pread` from worker threads straight
  into slab memory, and decode starts them speculatively, so a prompt
  arriving right after a decode turn has reads in flight. Freeing underneath
  them is a use-after-free - Metal hands the pages to the prefill and the
  worker finishes its read into someone else's memory. `_release_slab`
  freezes new slot reads, waits out the pending ones, and only then frees.
* **Size the rebuild, not the load.** The KV cache grows with the
  conversation while the slab's size was fixed at load, so a 28k-token
  session had the rebuild asking for 40.3 GB against a 35.05 GB budget and
  the server died. `_slots_that_fit` gives the KV cache right of way and
  rebuilds into whatever is left (28 GB at 13k context, 22.3 GB at 28k). A
  smaller slab costs hit rate; a failed allocation costs the process, because
  Metal reports it asynchronously and there is nothing to catch.
* **Both paths must read slot entries.** Decode installs slot-backed entries
  into the LRU, and the *next* request's prefill takes them as cache hits - so the per-expert path sees entries holding a slot number instead of
  arrays. `_run` converts them (`_loose`); before it did, every turn after
  the first died with `KeyError: 'up_proj.weight'`. The tests cover it by
  generating twice against one model, since a single generation always
  prefills against a cold cache and never hits this.

## Known limits & future work

- Decode-path overhead: streamed-but-fully-cached runs at ~20-40% of resident
  speed on tiny models (per-layer sync + per-expert loop). Vanishes into the
  noise for big disk-bound models.
- Smarter speculation is **done** - the next layers' own routers are run on the
  current hidden state (see `streaming.py`), which roughly doubles prefetch
  coverage over the "same as last token" heuristic. What remains open is the
  *cold* case: prediction only helps once decode is underway, so the first
  tokens of a fresh session still pay full price. The early-decode warm pool
  (`PAGED_MOE_EARLY_DECODE`) is the first step at that.
- Prefill reads ~the whole expert mass once per chunk, and the default 32k
  chunk makes typical agent prompts a single pass. Prompts beyond 32k tokens
  still take multiple passes; going wider needs less activation memory per
  chunk, which is where the per-expert compute loop comes in (below).
- **Decode now batches per layer, via slab storage** (see "Slab storage"
  below). The note that follows is kept because it is still true of the
  *per-expert* path - batching only pays once the cache stops handing out
  separate arrays, which is exactly what the slab changed:
    - In isolation, one `gather_qmm` per projection beats the per-expert loop by
      3.4x at decode shapes, and stacking the cache's per-expert arrays to feed
      it costs only ~0.2 ms/layer, so the arithmetic predicted ~9 ms/token.
    - End to end on qwen3-next-think it measured 12.0 tok/s batched vs 12.2
      tok/s per-expert (160 tokens, temp 0, identical 85.9% hit rate). The
      batched path was confirmed live (152 calls, zero fallbacks) and
      bit-identical, so this is a real null result: decode time sits in disk
      misses and the backbone, not in expert-matmul dispatch.
    - At prefill row counts (~640 rows/expert) the per-expert loop is *faster*,
      and batching there would need 512 experts stacked at once.
  The measurement that unlocked this was re-running the probe at the *other*
  model's shapes: Coder-Next's experts are 1.6 MB, Qwen3-235B's are 10.6 MB
  across 94 layers, where the per-expert loop costs 1.19 ms/layer (112
  ms/token). Always pass `--hidden/--inter/--topk/--layers`; the answer is
  shape-dependent and the default shapes hide it.
- GLM-4.5-Air decode is bounded by physics: layers x experts x bytes x
  miss rate. The 3-bit DWQ build cuts reads
  and cache pressure 25% vs 4-bit: measured 3.7 -> ~4.8 tok/s.
- `SwitchMLP`-based models (a few non-GLU MoEs) aren't streamed yet - they
  load resident with a warning-by-omission. Same treatment would work.
