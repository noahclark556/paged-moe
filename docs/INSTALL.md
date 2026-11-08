# Install PagedMoE for MLX

Set up PagedMoE so **mlx_lm** and **mlx-openai-server** stream opted-in MoE
checkpoints instead of trying to hold them fully in RAM.

## What you get

| Path | Role |
| --- | --- |
| `./install_mlx.sh` | Install into your mlx venv + enable auto-hook |
| Auto-hook | Makes `mlx_lm.load` use PagedMoE when a model is opted in |
| `~/paged-moe-config.yaml` | Which models stream + optional tuning + `machine:` |
| `~/.paged_moe/sidecar/` | Sidecar weights (created automatically) |
| `paged-moe` CLI | `status` / `which` / `machine` / `ladder` / `install` / `uninstall` |

If you use mlx-openai-server, list the same checkpoint path here that you use
as `model_path` in `~/mlx-config.yaml` (or drop a marker file in the checkpoint
folder).

## Requirements

- macOS + Apple Silicon (Metal)
- Python 3.10+ venv with **mlx** / **mlx-lm** (same one as openai-server if you use it), typically `~/.mlx-env`
- Matching versions: see `requirements.txt` / `pyproject.toml` (`mlx-lm>=0.31.3`)

## Install

```bash
cd paged-moe               # cloned repo root
chmod +x install_mlx.sh
./install_mlx.sh
# or: ./install_mlx.sh --python /path/to/venv/bin/python
```

Manual equivalent:

```bash
source ~/.mlx-env/bin/activate
pip install -e .
paged-moe install                 # auto-hook + sample ~/paged-moe-config.yaml
```

Install without the auto-hook:

```bash
./install_mlx.sh --no-hook
# or enable later with: paged-moe install
```

Remove the auto-hook (package stays):

```bash
./install_mlx.sh --uninstall
# or: paged-moe uninstall
```

## Opt a model in

**Option A - config** (`~/paged-moe-config.yaml`):

Install seeds the tested checkpoints under `~/mlx-models/` with good defaults
(Coder-Next, Qwen3-235B, GLM-4.7, Qwen3-Coder-480B, DeepSeek-V3.2, Kimi-K2).
Edit paths if yours differ.

```yaml
enable_all: false
debug: false
models:
  - path: /Users/you/mlx-models/qwen3-next
    env:
      PAGED_MOE_SIDECAR: "1"
  # ... plus qwen3-235-4bit, glm-47-4bit, qwen3-coder-480-4bit, deepseek-v32-4bit
```

**Option B - marker** in the checkpoint directory:

```bash
touch /Users/you/mlx-models/qwen3-coder-480-4bit/.paged_moe
# or write paged_moe.yaml with an optional env: block
```

**Option C - everything** (small models that fit in RAM still stay resident):

```bash
export PAGED_MOE=1
```

Kill switch: `PAGED_MOE_HOOK=0`.

Tuning keys use the `PAGED_MOE_*` names documented in
`~/paged-moe-config.yaml`. Omit a key to keep the shipped default.

Capacity knobs (expert cache ceiling, reader threads, headroom) scale from
this Mac's RAM. Auto-detect is on by default. Pin or override in yaml:

```yaml
machine:
  auto: true
  # ram_gb: 16
  # profile: mba-m2-2023-16gb
```

```bash
paged-moe machine          # RAM, cores, SSD, derived knobs
paged-moe machine probe    # sequential SSD + GPU cores
paged-moe ladder           # what fits on this Mac
```

## Debug

```bash
paged-moe status
paged-moe which /path/to/model
PAGED_MOE_DEBUG=1   # stderr: [paged-moe] stream via ... / passthrough ...
```

Restart openai-server (or your Python process) after changing the config or
reinstalling the hook.

---

## Quick checks

Assume you ran `./install_mlx.sh` and have a checkpoint (example:
`~/mlx-models/qwen3-next`).

### 1) Direct mlx_lm (Python)

```bash
source ~/.mlx-env/bin/activate

paged-moe which ~/mlx-models/qwen3-next
# -> STREAM once the path is in config (or has a marker)

PAGED_MOE_DEBUG=1 python - <<'PY'
from mlx_lm import load, generate
model, tok = load("/Users/you/mlx-models/qwen3-next")
print(generate(model, tok, prompt="Say hi in one word.", max_tokens=8))
PY
```

Expect: a `[paged-moe] stream via ...` debug line and a short completion.
Sidecar weights (if enabled) land under `~/.paged_moe/sidecar/`.

### 2) mlx-openai-server + curl

1. Put the same path in `~/paged-moe-config.yaml` that you use as `model_path`
   in `~/mlx-config.yaml` (or add a marker file).
2. Restart the server:

```bash
source ~/.mlx-env/bin/activate
mlx-openai-server launch --config ~/mlx-config.yaml
```

3. Curl the served name:

```bash
curl -s http://127.0.0.1:11434/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "YOUR_SERVED_MODEL_NAME",
    "messages": [{"role":"user","content":"Say hi in one word."}],
    "max_tokens": 16
  }'
```

Expect: 200 + completion. With `PAGED_MOE_DEBUG=1` in the server env, logs show
`[paged-moe] stream via ...` on first load of that path.

### Quick negatives

| Check | Expect |
| --- | --- |
| Model not in config, no marker | `which` -> `passthrough` |
| `PAGED_MOE_HOOK=0` | Hook inert even if installed |
| `./install_mlx.sh --uninstall` | `status` -> `pth_present: false` |
