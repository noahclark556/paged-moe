#!/usr/bin/env bash
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
# Memory / throughput ladder for paged-moe - one model per invocation.
#
#   ./bench/memory_benchmark.sh list
#   ./bench/memory_benchmark.sh status
#   ./bench/memory_benchmark.sh 1          # Coder-Next thinking 6-bit
#   ./bench/memory_benchmark.sh 2          # Qwen3-235B 4-bit
# hf download mlx-community/Qwen3-235B-A22B-4bit --local-dir ~/mlx-models/qwen3-235-4bit
#   ./bench/memory_benchmark.sh 3          # GLM-4.7 4-bit
# hf download mlx-community/GLM-4.7-4bit --local-dir ~/mlx-models/glm-47-4bit
#   ./bench/memory_benchmark.sh 4          # Qwen3-Coder-480B 4-bit
# hf download mlx-community/Qwen3-Coder-480B-A35B-Instruct-4bit --local-dir ~/mlx-models/qwen3-coder-480-4bit
#   ./bench/memory_benchmark.sh 5          # DeepSeek-V3.2 4-bit
# hf download mlx-community/DeepSeek-V3.2-4bit --local-dir ~/mlx-models/deepseek-v32-4bit
#   ./bench/memory_benchmark.sh 6          # Kimi-K2-Instruct 4-bit
# hf download mlx-community/Kimi-K2-Instruct-4bit --local-dir ~/mlx-models/Kimi-K2-Instruct-4bit
#
# Streamed knobs come from a model catalog (ga_id on each tier) when one sits
# next to this repo. Without it every tier runs exact unless you export
# PAGED_MOE_PRUNE / PAGED_MOE_WAIT_ABOVE first (README "Numbers" has the recipes).
#
# Results append/replace sections in docs/MEMORY_BENCHMARK.md and land as
# JSON under bench/memory_results/tier_N.json.
#
# Env:
#   PAGED_MOE_BENCH_DIR   download/local root (default: ~/mlx-models)
#   Pass -y to skip interactive pauses; --force to re-run a finished tier.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -x "$HOME/.mlx-env/bin/python" ]]; then
    PY="$HOME/.mlx-env/bin/python"
  else
    PY="$(command -v python3)"
  fi
fi

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  ./bench/memory_benchmark.sh list
  ./bench/memory_benchmark.sh status
  ./bench/memory_benchmark.sh <1-6> [-y] [--force] [--skip-resident] [--steady]

  --steady   longer warmup (128) + decode (512), long-form prompt, ignore EOS.
             Use for GLM / 480B / DeepSeek / Kimi (and fine for every tier).

See docs/MEMORY_BENCHMARK.md for the ladder and protocol.
EOF
  exit 1
fi

cmd="$1"
shift || true

case "$cmd" in
  list|status)
    exec "$PY" bench/memory_benchmark.py "$cmd"
    ;;
  [1-6])
    exec "$PY" bench/memory_benchmark.py run "$cmd" "$@"
    ;;
  *)
    echo "unknown command: $cmd" >&2
    echo "expected: list | status | 1..6" >&2
    exit 1
    ;;
esac
