#!/usr/bin/env bash
# Install PagedMoE as an mlx_lm drop-in for this machine.
#
# What it does:
#   1. Regenerate requirements.txt from pyproject.toml (source of truth)
#   2. pip install -e this tree into ~/.mlx-env (or $MLX_ENV / $EXPERT_STREAM_PYTHON)
#   3. Write a site-packages .pth so mlx_lm.load is wrapped automatically
#   4. Create ~/paged-moe-config.yaml + ~/.paged_moe/sidecar (opt-in model list)
#
# Does NOT modify mlx-lm or mlx-openai-server source (license-safe).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLX_ENV="${AGENT_MLX_ENV:-${MLX_ENV:-$HOME/.mlx-env}}"
ES_PYTHON="${EXPERT_STREAM_PYTHON:-$MLX_ENV/bin/python}"
UNINSTALL=0
NO_HOOK=0
MODELS_DIR="${PAGED_MOE_MODELS_DIR:-}"

usage() {
  cat <<EOF
Usage: ./install_mlx.sh [--uninstall] [--no-hook] [--python PATH] [--models-dir PATH]

  --uninstall     Remove the mlx auto-hook .pth (package stays installed)
  --no-hook       pip install only; do not auto-wrap mlx_lm.load
  --python PATH   Python with mlx-lm (default: $ES_PYTHON)
  --models-dir D  Checkpoint root used to seed ~/paged-moe-config.yaml
                  (default: prompt interactively, else ~/mlx-models)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --uninstall) UNINSTALL=1; shift ;;
    --no-hook) NO_HOOK=1; shift ;;
    --python) ES_PYTHON="$2"; shift 2 ;;
    --models-dir) MODELS_DIR="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ ! -x "$ES_PYTHON" ]]; then
  echo "error: python not found: $ES_PYTHON" >&2
  echo "  Create ~/.mlx-env and install mlx-lm / mlx-openai-server first." >&2
  exit 1
fi

if [[ "$UNINSTALL" -eq 1 ]]; then
  echo "-> remove PagedMoE mlx auto-hook"
  # Prefer the console script so we avoid ``python -m`` re-importing a module
  # the still-present .pth already loaded (runpy RuntimeWarning).
  if command -v paged-moe >/dev/null 2>&1; then
    paged-moe uninstall || true
  else
    "$ES_PYTHON" -c "from expert_stream.mlx_hook import main; raise SystemExit(main(['uninstall']))" || true
  fi
  echo "Done. Package left in place (pip uninstall paged-moe to remove)."
  exit 0
fi

# pyproject.toml is the dependency source of truth - keep requirements.txt aligned.
echo "-> sync requirements.txt from pyproject.toml"
"$ES_PYTHON" - "$ROOT" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
text = (root / "pyproject.toml").read_text()
m = re.search(r"dependencies\s*=\s*\[(.*?)\]", text, re.S)
if not m:
    raise SystemExit("could not find [project] dependencies in pyproject.toml")
deps = re.findall(r'"([^"]+)"', m.group(1))
(root / "requirements.txt").write_text("\n".join(deps) + "\n")
print(f"  wrote {len(deps)} deps -> requirements.txt")
PY

echo "-> pip install -e PagedMoE  ($ES_PYTHON)"
"$ES_PYTHON" -m pip install -U pip setuptools wheel >/dev/null
if [[ -f "$ROOT/requirements.txt" ]]; then
  "$ES_PYTHON" -m pip install -r "$ROOT/requirements.txt"
fi
# Native expert-read extension builds when a compiler is present; install
# still succeeds without it (runtime JIT or Python pool fallback).
"$ES_PYTHON" -m pip install -e "$ROOT"

echo "-> import check"
"$ES_PYTHON" -c "import expert_stream; print('  ok', expert_stream.__file__)"

echo "-> seed pretrained sidecar weights (if shipped; skip existing files)"
"$ES_PYTHON" -c "from expert_stream.user_config import install_pretrained_sidecars; n=install_pretrained_sidecars(); print(f'  wrote {len(n)} files' if n else '  none / already present')"

# Ask where MLX checkpoints live so seeded config paths match this machine.
if [[ -z "$MODELS_DIR" ]]; then
  DEFAULT_MODELS="$HOME/mlx-models"
  echo ""
  echo "PagedMoE seeds ~/paged-moe-config.yaml with example model paths under a"
  echo "models directory. You can edit those paths anytime in the config file."
  if [[ -t 0 ]]; then
    read -r -p "Models directory [$DEFAULT_MODELS]: " MODELS_DIR || true
  fi
  MODELS_DIR="${MODELS_DIR:-$DEFAULT_MODELS}"
fi
MODELS_DIR="${MODELS_DIR/#\~/$HOME}"
echo "-> models directory for sample config: $MODELS_DIR"

HOOK_ARGS=(--models-dir "$MODELS_DIR")

if [[ "$NO_HOOK" -eq 0 ]]; then
  echo "-> install PagedMoE mlx auto-hook + ~/paged-moe-config.yaml"
  if command -v paged-moe >/dev/null 2>&1; then
    paged-moe install "${HOOK_ARGS[@]}"
  else
    "$ES_PYTHON" -c "from expert_stream.mlx_hook import main; raise SystemExit(main(['install', '--models-dir', r'''$MODELS_DIR''']))"
  fi
else
  echo "-> skip auto-hook (--no-hook); ensuring ~/paged-moe-config.yaml"
  if command -v paged-moe >/dev/null 2>&1; then
    paged-moe ensure-config "${HOOK_ARGS[@]}"
  else
    "$ES_PYTHON" -c "from expert_stream.mlx_hook import main; raise SystemExit(main(['ensure-config', '--models-dir', r'''$MODELS_DIR''']))"
  fi
fi

echo ""
echo "Done."
echo "  package     editable from $ROOT"
echo "  user config \$HOME/paged-moe-config.yaml  (list models to stream)"
echo "  data dir    \$HOME/.paged_moe/            (sidecar weights, etc.)"
echo "  pretrained  seeded into sidecar/ when shipped with this build"
echo "  models dir  $MODELS_DIR  (edit paths in config anytime)"
echo "  status      paged-moe status"
echo ""
echo "Next: confirm paths in ~/paged-moe-config.yaml, then:"
echo "  paged-moe which /path/to/model"
echo "See README.md (Install) for mlx test steps."
