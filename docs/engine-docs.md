# Engine usage

How to run and tune the open-source PagedMoE engine. This is the source of
truth for day-to-day use: install paths, config, CLI, server, fidelity, and
the env knobs that matter. Architecture lives in [DESIGN.md](./DESIGN.md).
Install steps live in [INSTALL.md](./INSTALL.md). Agent-host prompt layout
lives in [CLIENT_CONTEXT.md](./CLIENT_CONTEXT.md).

Defaults, full names, and comments for every knob also sit at the top of
`pagedmoe/config.py`. When this doc and that file disagree, trust the code.

---

## What you are running

PagedMoE keeps the always-on backbone of an MoE checkpoint in RAM and pages
routed experts in from the SSD through a fixed-size cache. Prefetch, slabs,
the sidecar, and verify-decode only change *when* bytes arrive. They do not
change *which* experts the router picked, so output stays bit-identical to a
fully resident mlx-lm run unless you turn on a fidelity knob (prune, wait,
flow, or the sidecar prune head).

The public package is `paged-moe`. The import name is `pagedmoe`. Entry
points:

| Command | Role |
| --- | --- |
| `paged-moe` | status, hook install, which, detect, serve, bench, machine, ladder |
| `paged-moe-server` | OpenAI-compatible HTTP server |
| `paged-moe-pretrain` | sidecar warm-start / fit / unlock |
| `python run.py` | one-shot or chat REPL from a clone |
| `python -m pagedmoe.server` | same server as `paged-moe-server` |

There is no separate weight conversion. Point it at a normal mlx-community
safetensors directory (or an HF repo id) and it streams from those files.

---

## Ways to use it

Three common setups. Pick one.

### 1. `paged-moe serve` (fastest)

No config required. It reads the checkpoint, sizes the cache for this Mac,
and runs exact by default:

```bash
./install_mlx.sh
source ~/.mlx-env/bin/activate
paged-moe serve ~/mlx-models/qwen3-235-4bit --port 8080
```

Trade some fidelity for speed with `--speed`:

```bash
paged-moe serve ~/mlx-models/qwen3-235-4bit --speed auto --port 8080
paged-moe serve ~/mlx-models/qwen3-235-4bit --speed 0.5 --port 8080
```

`--speed` takes `exact` / `1` (bit-identical), a number from 0 to 1, or
`auto` (picks from model size, active params, and bit width). See
[Speed vs exact](#speed-vs-exact).

### 2. mlx_lm drop-in (hook + yaml)

Install the hook once, list checkpoints in `~/paged-moe-config.yaml`, then
use mlx-lm or mlx-openai-server the way you already do:

```bash
./install_mlx.sh
paged-moe install
# edit ~/paged-moe-config.yaml so paths match your checkpoints
python -c 'from mlx_lm import load; load("~/mlx-models/qwen3-235-4bit")'
```

Only listed paths stream (unless you opt everything in). Everything else
stays stock mlx-lm. Details in [INSTALL.md](./INSTALL.md).

### 3. Direct Python / `run.py`

```bash
python run.py --model ~/mlx-models/qwen3-235-4bit --chat
python run.py --model ~/mlx-models/qwen3-235-4bit \
  --mode streamed --stats --max-tokens 200 \
  --prompt "Merge two sorted lists in Python."
```

Or from your own code:

```python
from pagedmoe import load, get_stats

model, tokenizer = load("~/mlx-models/qwen3-235-4bit", mode="streamed", verbose=True)
print(get_stats(model))
```

`load` kwargs such as `mode=` and `cache_gb=` win over env for that call.
`run.py` and the server also pick up per-model `env:` from the yaml (see
[Precedence](#precedence)).

---

## Config file

Path: `~/paged-moe-config.yaml` (override with `PAGED_MOE_CONFIG`).

Data (sidecar weights, prefix-cache files, etc.) lives under `~/.paged_moe/`
(override with `PAGED_MOE_HOME`). The config file and the data root are
separate on purpose.

Minimal shape:

```yaml
enable_all: false
debug: false
machine:
  auto: true
# defaults:
#   PAGED_MOE_SIDECAR: "1"
models:
  - path: /Users/you/mlx-models/qwen3-235-4bit
    env:
      PAGED_MOE_PRUNE: "0.7"
      PAGED_MOE_WAIT_ABOVE: "0.2"
      PAGED_MOE_SIDECAR: "1"
```

| Field | Meaning |
| --- | --- |
| `enable_all` | stream every load (tiny models that fit still stay resident) |
| `debug` | print `[paged-moe] stream / passthrough` lines |
| `machine` | hardware overrides; leave `auto: true` unless you need to pin |
| `defaults` | env applied to every matched model first |
| `models[].path` | checkpoint directory |
| `models[].env` | per-model `PAGED_MOE_*` overrides |

`paged-moe install` and `paged-moe ensure-config` seed a sample file with
the tested checkpoints under `~/mlx-models/` (or `--models-dir`). Edit the
paths; you do not need to reinstall after that.

### Machine block

Capacity knobs (RAM budget, expert cache ceiling, reader threads, headroom)
scale from this Mac's RAM, cores, and GPU working set. Auto-detect is the
default.

```yaml
machine:
  auto: true
  # ram_gb: 16
  # perf_cores: 8
  # disk_gb_s: 6.0
  # storage_gb: 512
  # profile: mba-m2-2023-16gb
```

```bash
paged-moe machine                 # what this Mac resolved to
paged-moe machine probe           # measure SSD + GPU cores
paged-moe machine catalog         # named profiles
paged-moe ladder                  # model fit table for this Mac
```

Fidelity knobs (prune, wait, sidecar) stay per-model. Do not put them under
`machine:`.

---

## Opting a model in

The mlx_lm hook streams a load when any of these are true:

1. The path is listed under `models:` in the yaml
2. The checkpoint dir has a marker: `.paged_moe` or `paged_moe.yaml`
3. `enable_all: true` in the yaml, or `PAGED_MOE=1` / `PAGED_MOE_ENABLE_ALL=1`

Kill switch: `PAGED_MOE_HOOK=0` leaves every load alone, even if the hook is
installed.

Marker with optional env:

```bash
# empty marker
touch ~/mlx-models/qwen3-235-4bit/.paged_moe

# or yaml next to the weights
cat > ~/mlx-models/qwen3-235-4bit/paged_moe.yaml <<'EOF'
env:
  PAGED_MOE_SIDECAR: "1"
EOF
```

`paged-moe which <path>` prints `STREAM` or `passthrough`, and the merged
env when it would stream.

`paged-moe serve` / `paged-moe-server` / `run.py` always go through the
engine loader for the model you pass. They do not need the path listed, but
they still read matching `env:` from the yaml when the hook is enabled.

---

## Precedence

How settings layer, lowest to highest:

**`paged-moe-server` / `paged-moe serve` / `run.py`** (via `launch_env.py`):

1. Engine defaults for this Mac
2. yaml `defaults:`
3. matched model's `env:` (or marker `paged_moe.yaml`)
4. anything already exported in the process environment
5. `--speed` on the command line

Already-exported env wins over the yaml. `--speed` wins over both.

**Hooked `mlx_lm.load`:**

1. Engine defaults
2. yaml `defaults:` then the matched model's `env:` (applied into the
   process for that load)
3. `load(..., mode=, cache_gb=, ...)` kwargs for that call

Pass `--config /path/to/file.yaml` on the server / `run.py` to use a
different yaml (`PAGED_MOE_CONFIG` under the hood).

Most knobs are read once when the engine module loads. Change them, then
start a new process. A few prune-related knobs can change live on a running
server; see [Live /config](#live-config).

---

## CLI

```text
paged-moe [status|install|uninstall|ensure-config|
           which <model>|detect <model> [--speed S]|
           serve <model> [--speed S] [server flags]|
           bench <model> [--speed S|--exact] [--quick]|
           machine|ladder] [--models-dir PATH]
```

| Command | Does |
| --- | --- |
| `paged-moe` / `status` | hook, config path, detected Mac (JSON) |
| `paged-moe install` | sitecustomize hook + sample yaml |
| `paged-moe uninstall` | remove the hook (package stays) |
| `paged-moe ensure-config` | write/keep the sample yaml |
| `paged-moe which <path>` | STREAM or passthrough + merged env |
| `paged-moe detect <path> [--speed S]` | checkpoint summary and what `--speed` would set, no load |
| `paged-moe serve <path> ...` | server with launch_env applied |
| `paged-moe bench <path> ...` | agent-task bench on this Mac |
| `paged-moe machine ...` | show / detect / probe / catalog / apply |
| `paged-moe ladder` | per-Mac fit table (`--profile`, `--json`) |

Related binaries:

```bash
paged-moe-server --model ~/mlx-models/qwen3-235-4bit --port 8080
paged-moe-pretrain --quick --model ~/mlx-models/qwen3-235-4bit --corpus prompts.jsonl
paged-moe-pretrain --fit-only --model ~/mlx-models/qwen3-235-4bit
paged-moe-pretrain --unlock-all
```

Debug helpers:

```bash
paged-moe status
paged-moe which ~/mlx-models/qwen3-235-4bit
PAGED_MOE_DEBUG=1          # stream / passthrough lines
PAGED_MOE_SIDECAR_DEBUG=1  # [sidecar] tick lines
```

---

## Server

```bash
paged-moe-server --model ~/mlx-models/qwen3-235-4bit --port 8080
# or
paged-moe serve ~/mlx-models/qwen3-235-4bit --port 8080
python -m pagedmoe.server --model ~/mlx-models/qwen3-235-4bit --port 8080
```

OpenAI-compatible surface at `http://127.0.0.1:<port>/v1`. Any API key is
accepted unless you put something in front of it.

Useful extras on top of stock mlx-lm server behavior:

- Streamed models use an 8-bit KV cache by default so long contexts fit
- Bigger default prefill chunks so a long prompt makes fewer passes over
  the expert mass
- `relieve_pressure()` after each generation so Metal gives pages back
- Speculative decoding via `PAGED_MOE_DRAFT_MODEL` + `PAGED_MOE_DRAFT_TOKENS`
  (small same-tokenizer draft; greedy accept is exact)
- `warm_prefix: true` on `/v1/chat/completions` prefills and stores that
  exact prompt, then skips decode (useful right after a turn ends)

Example warm:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "default_model",
    "messages": [{"role":"user","content":"...pinned front matter..."}],
    "max_tokens": 0,
    "warm_prefix": true
  }'
```

One model per process. Do not run two large MLX / PagedMoE loads against the
same GPU. Keep big checkpoints on the internal SSD when you can; decode is
mostly disk bandwidth.

---

## Speed vs exact

Exact is the default. Every routed expert is computed. Prefetch mistakes only
waste a read.

Approximate decode is opt-in. The main knobs:

| Knob | Default | Effect |
| --- | --- | --- |
| `PAGED_MOE_PRUNE` | `0` | drop a routed expert whose weight is below this fraction of the strongest one, if it would need a disk read |
| `PAGED_MOE_WAIT_ABOVE` | `0` | only stall on a miss when that expert carries at least this much of the mixture |
| `PAGED_MOE_FLOW` | `0` | one-pass decode; fidelity knob, needs prune > 0 |
| `PAGED_MOE_SIDECAR_HEAD_PRUNE` | `0` | sidecar adaptive prune/wait (also changes logits) |

`--speed` is a convenient front end for prune / wait (and at 0, flow). Rough
fixed map before a per-model kept-mass curve exists:

| `--speed` | Rough mapping |
| --- | --- |
| `1` / `exact` | prune 0, wait 0 |
| `0.5` | prune ~0.7, wait ~0.25 |
| `0` | prune ~0.95, wait ~0.5, flow on |
| `auto` | pick from total size, active size, bit width |

Once the engine has recorded a prune curve for that model under the sidecar
dir, in-between values follow kept mixture mass instead of the fixed map.
`paged-moe detect <model> --speed auto` prints the pick without loading.

When prune is on, a few helpers keep the damage off the worst tokens:

- `PAGED_MOE_FIDELITY` (default on) softens prune/wait on tool-call spans,
  the start of a turn, and the visible reply
- `PAGED_MOE_FIDELITY_GUARD` trips exact decode on degenerate loops
- `PAGED_MOE_RESCUE` holds and redoes looping / garbled spans exact

Leave prune unset for bit-identical output. Measure on your own workload
before you turn it on.

Recipes used for the large models in the README:

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

---

## Env knobs worth knowing

Set them in the process environment, under yaml `env:`, or (for the server /
`run.py`) via `--speed`. Names always use the `PAGED_MOE_` prefix.

### Paths and switches

| Knob | Default | What it does |
| --- | --- | --- |
| `PAGED_MOE_CONFIG` | `~/paged-moe-config.yaml` | config file path |
| `PAGED_MOE_HOME` | `~/.paged_moe` | data root (sidecar, caches) |
| `PAGED_MOE_DOWNLOAD_DIR` | `~/models/paged-moe` | where HF repo ids download |
| `PAGED_MOE_MODELS_DIR` | `~/mlx-models` | sample-config path root |
| `PAGED_MOE_HOOK` | `1` | `0` disables the mlx_lm hook |
| `PAGED_MOE` / `PAGED_MOE_ENABLE_ALL` | off | stream every load |
| `PAGED_MOE_DEBUG` | off | stream / passthrough / verbose |

### Capacity and I/O

| Knob | Default | What it does |
| --- | --- | --- |
| `PAGED_MOE_MODE` | `auto` | `auto` / `resident` / `streamed` |
| `PAGED_MOE_RAM_FRACTION` | `0.68` up to 48 GB, auto above | share of RAM the engine may use. Above 48 GB it rises a little but stays 4.8 GB under the GPU working set (about 0.70 on 96 GB). The extra only goes to the expert cache. Setting it yourself turns that off |
| `PAGED_MOE_WS_MARGIN_LEARN` | `1` | on Macs above 48 GB, measure how close each model really gets to the GPU working set and let its next load use a smaller gap (2 to 4.8 GB). `0` keeps the fixed 4.8 |
| `PAGED_MOE_CACHE_GB` | auto | fixed expert-cache size |
| `PAGED_MOE_MAX_CACHE_GB` | `26` on 48 GB | ceiling on the auto cache. Grows with the RAM budget on other Macs |
| `PAGED_MOE_RESERVE_CTX` | `0` | tokens of KV to hold back when sizing the cache (set this on heavy-KV models) |
| `PAGED_MOE_HEADROOM_GB` | `6` on 48 GB | flat headroom when `RESERVE_CTX` is 0, scaled on other Macs |
| `PAGED_MOE_READ_THREADS` | `16` | parallel disk readers |
| `PAGED_MOE_NATIVE_READ` | `1` | C pread pool; falls back to Python |
| `PAGED_MOE_NOCACHE` | `1` | `F_NOCACHE` expert reads. Leave on |
| `PAGED_MOE_PAGECACHE_TRIM` | `1` | drop leaked file-cache pages |
| `PAGED_MOE_MEM_PRESSURE` | `1` | shrink/regrow the slab with macOS pressure |
| `PAGED_MOE_WATCHDOG` | `1` | exit on a stalled request or sustained critical pressure |
| `PAGED_MOE_SLAB` | `auto` | slot-addressed resident experts (see DESIGN.md) |
| `PAGED_MOE_LFRU` | `1` | frequency-protected eviction |

Don't raise Metal's wired limit (`sudo sysctl iogpu.wired_limit_mb=...`)
for this. The engine keeps its own wired limit at 0 and sizes its memory from
the Mac on its own, so leave it at the default and let macOS keep its room.
Don't turn `NOCACHE` off on a 48 GB machine with a big checkpoint;
the page cache will double-count RAM and can freeze the box during prefill.

### Prefill

| Knob | Default | What it does |
| --- | --- | --- |
| `PAGED_MOE_PREFILL_CHUNK` | `32768` | max tokens per prefill pass |
| `PAGED_MOE_FUSED_PREFILL` | `1` | attention sub-chunks, one MoE pass per chunk |
| `PAGED_MOE_ADAPTIVE_PREFILL` | `0` | resize the step each pass (`1` on DSA / huge models) |
| `PAGED_MOE_ADAPTIVE_PREFILL_MODE` | `auto` | `auto` / `dsa` / `fused` / `off` |
| `PAGED_MOE_PREFILL_CACHE_GB` | `4` | expert-cache budget during a big prefill chunk |
| `PAGED_MOE_PREFILL_AHEAD` | `1` | start the next layer's predicted reads early |
| `PAGED_MOE_ATTN_SUB_CHUNK` | `2048` | attention-only sub-chunk inside fused prefill (lowered if Metal needs it) |

Prefill is priced in expert bytes, not FLOPs. A prompt chunk routes across
essentially every expert, so chunk count multiplies the checkpoint. Fused
prefill keeps attention Metal-safe without re-reading the expert mass.

### Decode timing (quality-safe)

| Knob | Default | What it does |
| --- | --- | --- |
| `PAGED_MOE_PREDICT` | `auto` | route-predict prefetch; governor A/Bs on vs off |
| `PAGED_MOE_PREDICT_DEPTH` | `3` | layers ahead to predict |
| `PAGED_MOE_PREFETCH_DEPTH` | `8` | speculative prefetch lookahead |
| `PAGED_MOE_VERIFY` | `1` | exact-verify decode (drafts never reach output unchecked) |
| `PAGED_MOE_VERIFY_FIDELITY` | `plain` | `exact` = every accepted token is the exact model's pick |
| `PAGED_MOE_SELF_DRAFT` | `auto` | self-draft when prune is 0 |
| `PAGED_MOE_MTP` | `1` | use a checkpoint MTP head as a drafter when present |
| `PAGED_MOE_STITCH` | `1` | when every routed expert fits in RAM, hold them all and run each token as one graph (exact relative to the speed setting) |
| `PAGED_MOE_EARLY_DECODE` | `0` | prefetch experts common right after prefill |
| `PAGED_MOE_TOOL_WARM` | `1` | prefetch experts seen in tool-call bodies |
| `PAGED_MOE_DRAFT_MODEL` | unset | path to a small same-tokenizer draft model |
| `PAGED_MOE_DRAFT_TOKENS` | `3` | draft tokens per speculation round |

### KV, prefix, hybrid

| Knob | Default | What it does |
| --- | --- | --- |
| `PAGED_MOE_KV_BITS` | `8` | decode KV precision |
| `PAGED_MOE_KV_FP16_CTX` | `8192` (`4096` up to 32 GB, `0` up to 18 GB) | keep KV in fp16 below this length; `0` = always 8-bit |
| `PAGED_MOE_KV_LAZY` | `1` | resume reused 8-bit KV without a full fp16 copy |
| `PAGED_MOE_PROMPT_CACHE_MOVE` | `1` | hand a reused prompt cache over instead of deep-copying |
| `PAGED_MOE_HYBRID_CKPT` | `1` | recurrent-state checkpoints for branching |
| `PAGED_MOE_PREFIX_DISK` | `1` | pin front-matter KV to disk |
| `PAGED_MOE_PREFIX_DISK_DIR` | under `~/.paged_moe` | where those files go |
| `PAGED_MOE_PREFIX_DISK_MAX_GB` | `12` | total size cap |
| `PAGED_MOE_PREFIX_DISK_MODELS` | `6` | how many models to keep files for |
| `PAGED_MOE_PREFIX_DISK_PER_MODEL` | `2` | files kept per model |

More on the on-disk prefix cache in [PREFIX_CACHE.md](./PREFIX_CACHE.md).

### Fidelity (changes logits)

| Knob | Default | What it does |
| --- | --- | --- |
| `PAGED_MOE_PRUNE` | `0` | weight-threshold expert drop on miss |
| `PAGED_MOE_WAIT_ABOVE` | `0` | miss-stall gate |
| `PAGED_MOE_ROUTE_TOP_K` | `0` | keep only the strongest K routed experts |
| `PAGED_MOE_ROUTE_TOP_P` | `0` | nucleus-style mass cap on routed experts |
| `PAGED_MOE_KEEP_FREE` | `1` | never drop an expert that is already resident |
| `PAGED_MOE_FLOW` | `0` | one-graph decode over resident experts |
| `PAGED_MOE_FIDELITY` | `1` | soften prune on tools / turn start / reply |
| `PAGED_MOE_FIDELITY_SCALE` | `0.6` | how much prune/wait apply in those spans |
| `PAGED_MOE_FIDELITY_TOOL_SCALE` | follows scale | tool-call spans only (`0` = exact there) |
| `PAGED_MOE_FIDELITY_TURN_START` | `64` | first decode tokens of a turn that run softened |
| `PAGED_MOE_RESCUE` | `1` | detect loops / garble and redo exact |

### Sidecar

| Knob | Default | What it does |
| --- | --- | --- |
| `PAGED_MOE_SIDECAR` | `0` | master switch. Off = zero cost during decode |
| `PAGED_MOE_SIDECAR_DIR` | `~/.paged_moe/sidecar` | adapter + wrap-bank storage |
| `PAGED_MOE_SIDECAR_DEBUG` | `0` | print `[sidecar]` ticks |
| `PAGED_MOE_SIDECAR_HEAD_WRAP` | on when sidecar on | next-token early-layer prefetch |
| `PAGED_MOE_SIDECAR_HEAD_PREFILL` | `0` | chunk hot-set prefetch |
| `PAGED_MOE_SIDECAR_HEAD_RESIDENCY` | `0` | eviction protect scores |
| `PAGED_MOE_SIDECAR_HEAD_PRUNE` | `0` | adaptive prune/wait (affects logits) |
| `PAGED_MOE_SIDECAR_MIN_PRECISION` | `0.22` | confidence floor before acting |
| `PAGED_MOE_SIDECAR_MIN_GAIN` | `0.02` | marginal coverage over last-token replay |
| `PAGED_MOE_SIDECAR_LOCK_HITS` | `0` | auto-lock after N good hits (`0` = keep learning) |
| `PAGED_MOE_SIDECAR_GOVERNOR` | `1` | A/B actuation on measured tok/s |

Wrap is quality-safe (prefetch only). Prefill and residency heads are also
quality-safe. The prune head is not. A head only starts acting once it beats
what the engine already prefetches from the last token, and the wrap bank is
saved when the server is idle, not in the middle of decode.

Warm-start:

```bash
paged-moe-pretrain --quick --model ~/mlx-models/qwen3-235-4bit --corpus prompts.jsonl
paged-moe-pretrain --fit-only --model ~/mlx-models/qwen3-235-4bit
paged-moe-pretrain --unlock-all
```

`--quick` uses short prompts, more epochs, and sliding-window expand.
`--fit-only` densifies from the on-disk wrap bank (filled by prior pretrain
or live decode) with no MoE load. More design notes in
[tinymodel-improvements.md](./tinymodel-improvements.md).

### Everything else

There are many more knobs (governor windows, rescue audit, MTP, pagecache
trim thresholds, staging caps, and so on). They are all named
`PAGED_MOE_*`, documented next to their bindings in `pagedmoe/config.py`,
and safe to leave alone unless you are measuring a specific path.

---

## Live /config

A running `paged-moe-server` can take a few settings without a restart:

```bash
curl -s http://127.0.0.1:8080/config
curl -s -X POST http://127.0.0.1:8080/config \
  -H 'content-type: application/json' \
  -d '{"env":{"PAGED_MOE_PRUNE":"0.5","PAGED_MOE_WAIT_ABOVE":"0.2"}}'
```

Live keys today:

- `PAGED_MOE_PRUNE`
- `PAGED_MOE_WAIT_ABOVE`
- `PAGED_MOE_FIDELITY_TOOL_SCALE`
- `PAGED_MOE_FIDELITY_TURN_START`

Anything else is "read at load" and rejected. The apply is all-or-nothing:
if any key in the request needs a restart, nothing is applied. A successful
change bumps an epoch so the next request drops KV that was built under the
old values.

---

## Python API (public package)

```python
from pagedmoe import load, get_stats, relieve_pressure, resolve_model_path

model, tokenizer = load(
    "~/mlx-models/qwen3-235-4bit",
    mode="auto",          # auto | resident | streamed
    cache_gb=20,          # optional
    verbose=True,
)
stats = get_stats(model)
relieve_pressure()        # give Metal recycled pages back between heavy turns
```

That is the supported in-process surface for this repo. Host apps that want
an HTTP boundary should use `paged-moe-server`. There is also a small Node
wrapper under `node/` that spawns the server, waits for readiness, and kills
it on exit.

---

## Sidecar pretrain CLI

```text
paged-moe-pretrain
  --model PATH
  --corpus FILE_OR_DIR
  --quick | --fit-only | --unlock-all | --unlock-only
  --max-tokens N --max-prompt-tokens N --max-prompts N
  --chunk-prompts N --epochs N --cache-gb GB
  --bank PATH --fresh|--force --no-resume --start-at N
```

| Flag | Meaning |
| --- | --- |
| `--quick` | short decode, few prompts, many epochs, sliding-window expand |
| `--fit-only` | no MoE generate; dense-fit wrap from the wrap bank |
| `--unlock-all` | clear lock bits on every slot under `SIDECAR_DIR` |
| `--fresh` / `--force` | wipe slot weights and corpus progress |
| `--chunk-prompts N` | process N prompts from the resume point, then stop |

Labels always come from the real MoE router. Pretrain and fit only reuse
what decode recorded; they never make up routing or mix one prompt's history
with another's.

---

## Agent loops and long prompts

Short version for hosts:

- Treat the live prompt as append-only during a tool loop
- Pin stable front matter; put the ask at the tail
- Use prefix-disk / `warm_prefix` so the next turn does not re-prefill the
  pinned bytes
- Collapse finished tool bodies out of the hot prefix when the task is done
- Keep one Metal owner; do not load a second mlx or PagedMoE server beside it

Full host checklist: [CLIENT_CONTEXT.md](./CLIENT_CONTEXT.md).
Prefix-disk details: [PREFIX_CACHE.md](./PREFIX_CACHE.md).

---

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Model loads resident and OOMs | path not opted in; `paged-moe which <path>` should say STREAM |
| Hook does nothing | `paged-moe status` -> `pth_present`; kill switch `PAGED_MOE_HOOK=0` |
| Config changes ignored | restart the process; most knobs are import-time |
| Decode much slower than expected | checkpoint on external disk; cache too small; `PAGED_MOE_DEBUG=1` and watch miss lines |
| Machine freezes on long prefill | leave `PAGED_MOE_NOCACHE=1` and pagecache trim on |
| Output not bit-identical | prune / wait / flow / prune-head are on; use `--speed 1` or unset them |
| Sidecar silent | `PAGED_MOE_SIDECAR=1` and optionally `PAGED_MOE_SIDECAR_DEBUG=1` |
| Two loads fight over Metal | one model per process; stop the other server |
| mlx-lm version break | this tree pins mlx / mlx-lm; reinstall with `constraints.txt` |

Useful commands when filing a bug:

```bash
paged-moe status
paged-moe machine
paged-moe which /path/to/model
paged-moe detect /path/to/model --speed auto
```

---

## Where to read next

| Doc | Topic |
| --- | --- |
| [INSTALL.md](./INSTALL.md) | install, hook, first curl |
| [DESIGN.md](./DESIGN.md) | data flow, cache, slab, correctness |
| [why-it-fits.md](./why-it-fits.md) | short capacity story |
| [PREFIX_CACHE.md](./PREFIX_CACHE.md) | on-disk front-matter KV |
| [CLIENT_CONTEXT.md](./CLIENT_CONTEXT.md) | how a host should shape prompts |
| [tinymodel-improvements.md](./tinymodel-improvements.md) | sidecar heads |
| `pagedmoe/config.py` | every env binding and its default |
| README | benches, model list, quick start |
