#!/usr/bin/env bash
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
# Matched A/B: decode tok/s without sidecar vs with a trained sidecar.
#
# Each arm is a fresh process so Metal / expert-cache state does not leak.
# Use warm sidecar weights already on disk (~/.paged_moe/sidecar by default,
# or wherever PAGED_MOE_SIDECAR_DIR points).
#
#   ./bench/sidecar_ab.sh ~/mlx-models/qwen3-235-4bit 28 0.5
#   ./bench/sidecar_ab.sh ~/mlx-models/qwen3-235-4bit 28 0.5 256 3 0.2
#
# Args:
#   model-path  cache-gb  prune  [max-tokens=256] [reps=1] [wait-above=] [sidecar-dir=]
#
# Prints one JSON line per arm, then a summary with baseline / trained / gain.
set -euo pipefail

MODEL="${1:?model path}"
CACHE="${2:?cache gb}"
PRUNE="${3:?prune}"
TOK="${4:-256}"
REPS="${5:-1}"
WAIT="${6:-}"
SIDECAR_DIR="${7:-}"

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

# Prefer the Global Agent sidecar dir when it exists.
if [[ -z "$SIDECAR_DIR" ]]; then
  if [[ -d "$HOME/.global_agent/expert_sidecar" ]]; then
    SIDECAR_DIR="$HOME/.global_agent/expert_sidecar"
  else
    SIDECAR_DIR="$HOME/.paged_moe/sidecar"
  fi
fi

EXTRA=(--model "$MODEL" --cache-gb "$CACHE" --prune "$PRUNE" --max-tokens "$TOK" --sidecar-dir "$SIDECAR_DIR")
if [[ -n "$WAIT" ]]; then
  EXTRA+=(--wait-above "$WAIT")
fi

run_arm() {
  local sidecar="$1" label="$2"
  # Production-like heads when sidecar is on; lock hits left at 0 so existing
  # warm weights keep learning lightly (short timed window; effect is tiny).
  env PAGED_MOE_HOOK=0 \
    PAGED_MOE_SIDECAR="$sidecar" \
    PAGED_MOE_SIDECAR_DIR="$SIDECAR_DIR" \
    PAGED_MOE_SIDECAR_HEAD_WRAP=1 \
    PAGED_MOE_SIDECAR_HEAD_PREFILL=1 \
    PAGED_MOE_SIDECAR_HEAD_RESIDENCY=1 \
    PAGED_MOE_SIDECAR_HEAD_PRUNE=1 \
    PAGED_MOE_SIDECAR_MIN_PRECISION=0.22 \
    PAGED_MOE_SIDECAR_LOCK_HITS=0 \
    PAGED_MOE_SIDECAR_DEBUG=0 \
    "$PY" bench/sidecar_probe.py "${EXTRA[@]}" --label "$label"
}

echo "sidecar_ab model=$MODEL cache=$CACHE prune=$PRUNE tokens=$TOK reps=$REPS dir=$SIDECAR_DIR" >&2

LINES=()
for i in $(seq 1 "$REPS"); do
  # Alternate so thermals / disk drift hit both arms similarly.
  line0="$(run_arm 0 "no-sidecar-$i")"
  echo "$line0"
  LINES+=("$line0")
  line1="$(run_arm 1 "trained-sidecar-$i")"
  echo "$line1"
  LINES+=("$line1")
done

printf '%s\n' "${LINES[@]}" | PAGED_MOE_HOOK=0 "$PY" -c '
import json, sys
rows = [json.loads(line) for line in sys.stdin if line.strip()]
off = [r for r in rows if not r.get("sidecar")]
on = [r for r in rows if r.get("sidecar")]

def mean(xs):
    return sum(xs) / len(xs) if xs else None

base = mean([r["decode_tok_s"] for r in off])
trained = mean([r["decode_tok_s"] for r in on])
summary = {
    "reps_no_sidecar": len(off),
    "reps_trained_sidecar": len(on),
    "decode_tok_s_no_sidecar": None if base is None else round(base, 2),
    "decode_tok_s_trained_sidecar": None if trained is None else round(trained, 2),
    "gain_tok_s": None if base is None or trained is None else round(trained - base, 2),
    "gain_pct": None
    if base is None or trained is None or base <= 0
    else round(100.0 * (trained / base - 1.0), 1),
    "hit_rate_no_sidecar": None if not off else round(mean([(r.get("hit_rate") or 0) for r in off]), 4),
    "hit_rate_trained_sidecar": None if not on else round(mean([(r.get("hit_rate") or 0) for r in on]), 4),
}
print(json.dumps({"summary": summary}), flush=True)
print(
    "summary: no-sidecar=%s tok/s  trained=%s tok/s  gain=%s tok/s (%s%%)"
    % (
        summary["decode_tok_s_no_sidecar"],
        summary["decode_tok_s_trained_sidecar"],
        summary["gain_tok_s"],
        summary["gain_pct"],
    ),
    file=sys.stderr,
)
'
