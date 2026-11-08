#!/bin/bash
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
# A/B route prediction against the previous-token heuristic, one clean process
# per variant. Default model is the small real MoE (OLMoE, 64 experts, top-8);
# --cache-gb should be set so cache/expert-mass matches the deployment ratio,
# otherwise the run is bandwidth-bound and measures nothing about prefetch.
#
#   bench/predict_sweep.sh <model-path> <cache-gb> [max-tokens]
set -u
MODEL="${1:?model path}"
CACHE="${2:?cache gb}"
TOK="${3:-128}"
cd "$(dirname "$0")/.."

run() {
  local label="$1"; shift
  env "$@" python bench/predict_probe.py \
    --model "$MODEL" --cache-gb "$CACHE" --max-tokens "$TOK" --label "$label" \
    2>/dev/null | tail -1
}

run heuristic-only PAGED_MOE_PREDICT=0
run predict-d1      PAGED_MOE_PREDICT=1 PAGED_MOE_PREDICT_DEPTH=1
run predict-d2      PAGED_MOE_PREDICT=1 PAGED_MOE_PREDICT_DEPTH=2
run predict-d3      PAGED_MOE_PREDICT=1 PAGED_MOE_PREDICT_DEPTH=3
run predict-d2-raw  PAGED_MOE_PREDICT=1 PAGED_MOE_PREDICT_DEPTH=2 PAGED_MOE_PREDICT_RENORM=0
run predict-d2-noslack PAGED_MOE_PREDICT=1 PAGED_MOE_PREDICT_DEPTH=2 PAGED_MOE_PREDICT_SLACK=0
run predict-only    PAGED_MOE_PREDICT=1 PAGED_MOE_PREDICT_DEPTH=2 PAGED_MOE_PREFETCH_DEPTH=0
run no-lfru         PAGED_MOE_PREDICT=1 PAGED_MOE_PREDICT_DEPTH=2 PAGED_MOE_LFRU=0
