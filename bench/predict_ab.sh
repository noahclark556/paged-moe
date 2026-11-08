#!/bin/bash
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
# Interleaved, repeated A/B of route prediction.
#
# This machine's read throughput drifts between sessions (thermals, memory
# pressure from previous runs), enough to swamp the effect if OFF and ON are
# measured in separate batches. So alternate them and read the distribution,
# never a single pair.
#
# Both settings are forced, so this measures the mechanism rather than the
# "auto" governor's decision.
#
# On a machine whose drive caches the whole checkpoint there is no read latency
# to hide, and prefetching cannot help by construction - pass a --read-delay-us
# through to stand in for the SSD the deployment actually reads from:
#
#   bench/predict_ab.sh <model-path> <cache-gb> [tokens] [reps] [read-delay-us]
set -u
MODEL="${1:?model path}"
CACHE="${2:?cache gb}"
TOK="${3:-160}"
REPS="${4:-4}"
DELAY="${5:-0}"
cd "$(dirname "$0")/.."

for i in $(seq 1 "$REPS"); do
  for mode in 0 1; do
    lbl=$([ "$mode" = 0 ] && echo "off-$i" || echo "on-$i")
    PAGED_MOE_PREDICT=$mode python bench/predict_probe.py --model "$MODEL" \
      --cache-gb "$CACHE" --max-tokens "$TOK" --read-delay-us "$DELAY" \
      --label "$lbl" 2>/dev/null | tail -1
  done
done
