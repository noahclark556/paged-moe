# PagedMoE

Run large Mixture-of-Experts (MoE) LLMs locally on Apple Silicon
even when the model is larger than your Mac's unified memory.

Page experts from the SSD so 500 GB+ models actually load on 48 GB unified
memory. Route prediction runs the next layers' real routers early and
prefetches what they ask for. Decode miss reads go through a lean latch +
read pool (less Python overhead on the hot path). An optional tiny **sidecar**
can learn from your traffic too; a net-time governor only lets it spend disk
when decode actually gets faster. Same bit-identical output when pruning is
off.

> **MLX / Apple Silicon.** Offloads MoE expert weights to the internal SSD
> and pages them into Metal on demand - no CUDA, no NVIDIA, no separate
> server or quantization pass required.

**Jump:** [Quick start](#quick-start) | [Requirements](#requirements) |
[Why it fits](#why-it-fits) | [Compatible models](#compatible-models) |
[Extras](#extras) | [License](#license-dual) |
[Commercial](#license-dual) | [Citation](#citation) |
[Contact](#contact)

Measured on an Apple M5 Pro, **48 GB**. Without this, none of these load.
With it, they run full agent loops - tools, multi-turn, long prompts - at
usable speed; experts stay on disk until the router asks.

| Model | Full resident | Peak RAM | Prefill | Decode |
| --- | ---: | ---: | ---: | ---: |
| **Qwen3-Coder-Next** 6-bit | ~65 GB (OOM) | ~30 GB | ~250 tok/s | ~22 tok/s |
| **Qwen3-235B-A22B** 4-bit | ~132 GB (OOM) | ~33 GB | ~115 tok/s | ~7 tok/s |
| **GLM-4.7** 4-bit | ~199 GB (OOM) | ~34 GB | ~73 tok/s | ~6 tok/s |
| **Qwen3-Coder-480B** 4-bit | ~270 GB (OOM) | ~34 GB | ~80 tok/s | ~5 tok/s |
| **DeepSeek-V3.2** 4-bit | ~378 GB (OOM) | ~33 GB | ~45 tok/s | ~6 tok/s |
| **Kimi-K2-Instruct** 4-bit | ~578 GB (OOM) | ~32 GB | ~27 tok/s | ~9 tok/s |

Prefill is a cold ~2k-token prompt; decode is wall-clock after warmup on the
same KV. The listed models are full ladder runs on this machine. 
Kimi is the top of the ladder: a 578 GB
MoE that will not load resident, streamed under the same prune / wait /
sidecar recipe as DeepSeek and GLM.

- **235B** - ladder run at prune 0.5 measured **6.67 tok/s** (steady, 128 tok
  warmup); shipped recipe (prune 0.7 + wait 0.2 + sidecar) expected ~7 tok/s
- **GLM-4.7** - shipped recipe measured **6.4 tok/s** vs 1.8 tok/s full-mixture
  on the same A/B (96% token agree)
- **480B** - 480B at 270 GB with prune 0.7 lands ~5 tok/s. 
- **Kimi-K2** - steady ladder at prune 0.8 measured **8.75 tok/s** decode
  (peak **31.59 GB**; prefill ~27 tok/s). Largest SwitchGLU in the chart.

Defaults leave prune off and are slower on the disk-bound models. Every knob
is in `expert_stream/config.py`.

The RAM gap looks wrong until you remember MoEs are sparse: most experts are
idle on any given token, so you do not need the whole library in RAM.

**PagedMoE** keeps the always-on backbone in memory and pages experts when
the router picks them. LRU cache + route prediction (and an optional
governed sidecar) keep agent turns warm.

> **Status:** `v0.2.5` pre-release. The engine is in this repo and installs
> from a clone. Not on PyPI yet. APIs may change before `v1.0`.
>
> **Naming:** product / repo / PyPI / CLI = **PagedMoE** (`paged-moe`).
> Python import stays `expert_stream`; env knobs stay `EXPERT_STREAM_*`.

---

## Quick start

Three steps. After that, use mlx / mlx-lm / mlx-openai-server the way you
already do - no new load API required.

1. **Install** PagedMoE into your mlx venv (hooks `mlx_lm.load`)
2. **Edit** `~/paged-moe-config.yaml` and list the MoE checkpoints to stream
3. **Use mlx as normal** - loads of those paths go through PagedMoE automatically

### 1. Install

```bash
git clone https://github.com/noahclark556/paged-moe.git
cd paged-moe
chmod +x install_mlx.sh
./install_mlx.sh
# or: ./install_mlx.sh --python /path/to/venv/bin/python
# or: ./install_mlx.sh --models-dir ~/mlx-models
```

That installs the package, enables the auto-hook, and writes
`~/paged-moe-config.yaml` (asks which directory holds your MLX checkpoints).

Manual equivalent:

```bash
source ~/.mlx-env/bin/activate
pip install -e .
paged-moe install --models-dir ~/mlx-models
```

### 2. Edit `~/paged-moe-config.yaml`

This file is the control plane. Put real paths to the MoEs you want to
stream. Install seeds the tested checkpoint names under the models dir you
chose (default `~/mlx-models/`) - fix the paths if yours differ.

```yaml
enable_all: false
debug: false
models:
  - path: /Users/you/mlx-models/qwen3-next
    env:
      EXPERT_STREAM_SIDECAR: "1"
  - path: /Users/you/mlx-models/qwen3-235-4bit
    env:
      EXPERT_STREAM_SIDECAR: "1"
```

Only paths listed under `models:` stream (unless you set `enable_all: true`).
Everything else stays stock mlx-lm. Edit this file anytime; no reinstall.

Optional instead of yaml: drop a `.paged_moe` marker in the checkpoint folder,
or `export PAGED_MOE=1` for every load. Kill switch: `PAGED_MOE_HOOK=0`.

If you use mlx-openai-server, the `path` here should match `model_path` in
`~/mlx-config.yaml`.

### 3. Use mlx as normal

```python
from mlx_lm import load, stream_generate

# Same call as always. If the path is in ~/paged-moe-config.yaml, it streams.
model, tokenizer = load("~/mlx-models/qwen3-235-4bit")
for chunk in stream_generate(model, tokenizer, prompt, max_tokens=200):
 print(chunk.text, end="", flush=True)
```

```bash
# Same idea for a local OpenAI-compatible server:
python -m expert_stream.server --model ~/mlx-models/qwen3-235-4bit --port 8080
# Point your agent at http://127.0.0.1:8080/v1
```

Check a path:

```bash
paged-moe status
paged-moe which /path/to/model
PAGED_MOE_DEBUG=1 # stderr: [paged-moe] stream via ... / passthrough ...
```

| Path | Role |
| --- | --- |
| `~/paged-moe-config.yaml` | Which models stream (+ optional env knobs) |
| `./install_mlx.sh` / `paged-moe install` | Install + auto-hook |
| `~/.paged_moe/sidecar/` | Sidecar weights (created automatically) |
| `paged-moe` CLI | `status` / `which` / `install` / `uninstall` |

### Who this is for

- Qwen / GLM / DeepSeek-class MoEs on Apple Silicon without a 128-512 GB box
 (tunable outside that range too)
- Agent loops (tools, multi-turn, long prompts), not just one-shot chat
- Plain mlx-community (or converted) checkpoints, not a special "streaming
 edition" of two curated models

### Who this is not for

- Absolute minimum RAM footprint (~3 GB active). Other stacks optimize for
 tiny resident sets; this one spends unified memory on a large expert cache
 so agent turns stay warm
- NVIDIA / CUDA servers. This is an MLX / Mac tool
- Models that already fit in RAM. Stock `mlx_lm.load` is enough

### What you get

1. Drop-in for `mlx_lm.load` on SwitchGLU MoEs - checkpoint can be 2x or 5x RAM
2. An OpenAI-compatible server your agent already knows how to talk to
3. Bit-identical output to a fully resident run when expert pruning is off
4. Agent-session hardening: prompt-prefix reuse (one prefill, then fork
 snapshots on trimmable KV), memory that yields to long prefills instead of
 OOMing Metal

Under the hood, mlx-lm routes MoE expert compute through `SwitchGLU`. This
replaces that with a disk-backed version and keeps attention, KV cache,
templates, and sampling. LRU + route prediction + optional slabs hide as much
disk latency as the SSD allows.

Built to drive coding agents on a MacBook Pro (M5 Pro,
48 GB) against models the platform normally will not start.

---

## Requirements

| | |
| --- | --- |
| OS | macOS on Apple Silicon |
| Python | **3.10+** (built and tested on 3.10 / 3.11) |
| MLX | `mlx>=0.32` |
| mlx-lm | `mlx-lm>=0.31.3` |

Any OpenAI-compatible client (including mlx-openai-server) can talk to
`python -m expert_stream.server`. That stack is optional; it is not a hard
install dependency.

Pins live in `pyproject.toml` (source of truth). `./install_mlx.sh`
regenerates `requirements.txt` from those deps before `pip install`.

---

## Why it fits

On a normal load these MoEs want roughly their full checkpoint in RAM.
PagedMoE keeps a working set resident (peaks below measured on that 48 GB
M5 Pro).

| Model | Normally (full resident) | With PagedMoE |
| --- | ---: | ---: |
| **Qwen3-Coder-Next** (6-bit) | ~65 GB (will not load) | ~30 GB peak |
| **Qwen3-235B-A22B** (4-bit) | ~132 GB (will not load) | ~33 GB peak |
| **GLM-4.7** (4-bit) | ~199 GB (will not load) | ~34 GB peak |
| **Qwen3-Coder-480B** (4-bit) | ~270 GB (will not load) | ~34 GB peak |
| **DeepSeek-V3.2** (4-bit) | ~378 GB (will not load) | ~36 GB peak |
| **Kimi-K2-Instruct** (4-bit) | ~578 GB (will not load) | ~32 GB peak |

Only the always-on backbone stays in RAM. Routed experts live on disk until
needed. MoE size is mostly experts that almost never fire on a given token.

Experts are fetched from the SSD on demand into a working set that can be
dropped later. The model never has to be fully materialized.

A large in-memory expert cache keeps recently used experts warm under your
RAM budget.

Expert reads bypass the OS page cache so you are not double-spending memory
on an invisible kernel copy of the same bytes.

Route prediction asks upcoming layers what they want and starts those reads
before the GPU stalls. That is the big disk-bound decode win. The optional
sidecar can add next-token wrap prefetch on top; by default a governor A/Bs
it and settles off unless decode tok/s actually improves.

Optional decode pruning skips weak experts so fewer SSD reads happen per
token (you trade a bit of fidelity for speed).

Memory is reserved for context (KV) so a long conversation does not steal the
expert cache mid-session.

Long prefills can temporarily shrink the expert footprint so Metal does not
OOM, then warm back up for decode.

Prefill is priced in expert bytes, not FLOPs. A chunk that routes widely
streams essentially the whole expert mass once. Layer-fused prefill (on by
default) sub-chunks attention inside each decoder layer and runs the MoE once
per prompt chunk, so a long agent prompt can be one pass over the experts
instead of one pass per attention-sized step. Attention sub-chunk size is
`EXPERT_STREAM_ATTN_SUB_CHUNK` (default 2048); the Metal score-matrix bound
still caps it. Watch stderr for `[prefill] ... passes over the mass ...
blocked on disk ... compute`.

Agent turns reuse the prompt prefix instead of redoing the whole history each
tool round. On trimmable KV caches (DeepSeek / Qwen3-MoE / GLM and friends)
the cold path prefills once, then forks prefix snapshots with deepcopy+trim.
Hybrid / non-trimmable stacks still stop mid-prefill at snapshot boundaries.

If the model already fits, it can run fully resident. Streaming is the escape
hatch for oversized MoEs. Same API either way.

---

## Compatible models

Anything mlx-lm loads whose MoE layers use `SwitchGLU`:

| Family | Notes |
| --- | --- |
| Qwen3-MoE / Coder / 235B / 480B | Workhorse; 235B-A22B is the 48 GB stress test |
| Qwen3-Next (hybrid attention) | Coder-Next, 30B-A3B |
| GLM-4.x MoE | GLM-4.5-Air, GLM-4.7, etc. |
| DeepSeek V2/V3/V3.2-style | Large checkpoints; decode is disk-bound on TB SSDs |
| Kimi-K2 Instruct | ~578 GB class; listed in the sample yaml. Needs `tiktoken` |
| Mixtral, OLMoE, others | If it is SwitchGLU, it should stream |

No custom checkpoint format. Plain MLX safetensors.

---

## Extras

Things beyond install -> yaml -> chat. Optional; the Quick start path works
without them.

### Sidecar offline pretrain

Online learning already runs during normal decode when
`EXPERT_STREAM_SIDECAR=1`. For a faster warm-start from a prompt corpus
(before or instead of a long agent session):

```bash
# Fast warm-start (short decode, few prompts, many fit epochs)
paged-moe-pretrain --quick \
 --model ~/mlx-models/qwen3-235-4bit \
 --corpus /path/to/prompts.jsonl

# Same module form:
python -m expert_stream.pretrain_sidecar --quick \
 --model ~/mlx-models/qwen3-235-4bit \
 --corpus /path/to/prompts.jsonl
```

Corpus can be a directory of `.txt` / `.md` / `.jsonl`, or one `.jsonl` with
`prompt` / `text` / chat `messages` fields. Weights land under
`~/.paged_moe/sidecar/` (override with `EXPERT_STREAM_SIDECAR_DIR`).

Useful variants:

```bash
# Dense-fit only from a wrap bank already filled by live use or a prior run
paged-moe-pretrain --fit-only --epochs 12 --model ~/mlx-models/qwen3-235-4bit

# Clear lock bits so online training resumes (no MoE load)
paged-moe-pretrain --unlock-all

# Ctrl+C saves progress; re-run the same command to continue.
# --fresh / --force wipes slot weights + progress and starts over.
```

Shipped builds may also seed a small pretrained bundle under
`expert_stream/pretrained_sidecar/` on install when that allowlist is filled
for a release.

### OpenAI-compatible server

```bash
paged-moe-server --model ~/mlx-models/qwen3-235-4bit --port 8080
# or: python -m expert_stream.server --model <path> --port 8080
```

Same streaming load path as `mlx_lm.load`. Point any OpenAI-compatible client
at `http://127.0.0.1:8080/v1`.

### Tuning via yaml `env`

Per-model knobs go under each entry in `~/paged-moe-config.yaml` (or the
process environment). Common ones:

| Knob | Role |
| --- | --- |
| `EXPERT_STREAM_SIDECAR` | `1` = online sidecar; `0` = paging only |
| `EXPERT_STREAM_SIDECAR_GOVERNOR` | `1` (default) = only actuate while decode is faster |
| `EXPERT_STREAM_PREDICT` | `auto` / `1` / `0` - route-prediction prefetch |
| `EXPERT_STREAM_FUSED_PREFILL` | `1` (default) = attention sub-chunks; MoE once per prompt chunk |
| `EXPERT_STREAM_ATTN_SUB_CHUNK` | Query rows per attention call in fused prefill (default 2048) |
| `EXPERT_STREAM_ADAPTIVE_PREFILL` | `1` = resize the model-level step each pass |
| `EXPERT_STREAM_ADAPTIVE_PREFILL_MODE` | `auto` / `dsa` / `fused` / `off` |
| `EXPERT_STREAM_PREFILL_CHUNK` | Model-level step ceiling (default 32768) |
| `EXPERT_STREAM_READ_POOL_THREADS` | Decode slot-read queue depth (`0` = auto, `-1` = old executor path) |
| `EXPERT_STREAM_PRUNE` | Drop weak routed experts (speed vs fidelity) |
| `EXPERT_STREAM_WAIT_ABOVE` | Only stall on high-weight disk misses |
| `EXPERT_STREAM_FLOW` | `1` = one sync per token (GPU slot table); fidelity knob, off by default |
| `EXPERT_STREAM_LOOKUP` | `1` = n-gram self-draft (more tokens per disk pass); off by default |
| `EXPERT_STREAM_CACHE_GB` | Pin expert-cache size (else auto under RAM budget) |
| `EXPERT_STREAM_SIDECAR_DIR` | Where sidecar weights live |

Full catalog lives in comments at the top of `expert_stream/config.py`.
Wrap is on when the sidecar is; prefill / residency / prune heads stay off
unless you turn them on.

Chart recipes (optional - not the package defaults). Drop under a model's
`env:` in `~/paged-moe-config.yaml`:

```yaml
# DeepSeek-V3.2 / GLM-4.7 class (MoEGate) - matches the ~4 tok/s DeepSeek row
env:
  EXPERT_STREAM_PRUNE: "0.8"
  EXPERT_STREAM_WAIT_ABOVE: "0.2"
  EXPERT_STREAM_SIDECAR: "1"
  EXPERT_STREAM_ADAPTIVE_PREFILL: "1"

# Qwen3-235B / Qwen3-Coder-480B - same idea, slightly softer prune
env:
  EXPERT_STREAM_PRUNE: "0.7"
  EXPERT_STREAM_WAIT_ABOVE: "0.2"
  EXPERT_STREAM_SIDECAR: "1"
  EXPERT_STREAM_ADAPTIVE_PREFILL: "1"
```

`PRUNE` / `WAIT_ABOVE` trade a bit of fidelity for disk reads skipped. Leave
them unset for bit-identical decode (slower on the big MoEs).

### CLI cheat sheet

| Command | Purpose |
| --- | --- |
| `paged-moe status` | Hook + config paths |
| `paged-moe which <model>` | Will this path stream? |
| `paged-moe install` / `uninstall` | Auto-hook for `mlx_lm.load` |
| `paged-moe-pretrain ...` | Sidecar corpus warm-start / fit / unlock |
| `paged-moe-server ...` | Local OpenAI-compatible HTTP |
| `python run.py --model ... --chat` | Quick interactive REPL (clone only) |

---

## License (dual)

| | |
| --- | --- |
| Open / research / AGPL use | [GNU Affero GPL v3](./LICENSE) |
| Proprietary / closed product / SaaS / OEM | Commercial license - email `noah@qwertycode.org` |
| Outside contributors | [CLA](./CLA.md) + [DCO](./DCO.md) (see [CONTRIBUTING.md](./CONTRIBUTING.md)) |

Free under AGPL-3.0 for people and open projects. Companies shipping closed
products, hosted services, or appliances without AGPL source disclosure should
email for a commercial license:

**Noah Clark** - `noah@qwertycode.org`

In-repo summaries: [LICENSING.md](./LICENSING.md),
[COMMERCIAL_LICENSE.md](./COMMERCIAL_LICENSE.md).

Copyright (c) 2026 Noah Clark.

---

## Citation

If you use this in research or write about it, please cite the repository
(see [CITATION.cff](./CITATION.cff)):

```bibtex
@software{clark_paged_moe,
 author = {Clark, Noah},
 title = {PagedMoE: Page MoE experts from disk on Apple Silicon},
 year = {2026},
 url = {https://github.com/noahclark556/paged-moe}
}
```

## Contact

- Commercial license / dual-license: `noah@qwertycode.org`
- Bugs / features: GitHub issues
