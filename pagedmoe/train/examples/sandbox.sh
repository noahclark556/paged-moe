#!/usr/bin/env bash
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
# Paged training checks, in a throwaway venv or the env you are in.
#
#   pagedmoe/train/examples/sandbox.sh                 temp venv under /tmp, pinned to tested versions
#   pagedmoe/train/examples/sandbox.sh --here          the python on PATH (activate your mlx env first)
#   MODEL=~/mlx-models/Qwen3.6-35B-A3B-4bit pagedmoe/train/examples/sandbox.sh --here
#                                                      + short SFT / ORPO / GRPO runs on a real model
#   SIM_RAM_GB=16 MODEL=... sandbox.sh --here           same, planned as if this were a 16 GB Mac
#
# Runs from source (PYTHONPATH), so nothing gets installed into your env.
# Run outputs go to /tmp/paged-train-check.
set -euo pipefail

CORE="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="${OUT:-/tmp/paged-train-check}"
mkdir -p "$OUT"

if [[ "${1:-}" == "--here" ]]; then
  PY="$(command -v python)"
else
  VENV="${VENV:-$OUT/venv}"
  if [[ ! -x "$VENV/bin/python" ]]; then
    echo "creating $VENV"
    uv venv -q --python 3.12 "$VENV"
    # the versions the trainer is tested with
    uv pip install -q --python "$VENV/bin/python" \
      mlx==0.32.2 mlx-metal==0.32.2 mlx-lm==0.31.3 numpy==2.3.5 \
      transformers==5.14.1 tokenizers==0.22.2 safetensors==0.8.0 \
      huggingface_hub==1.16.1 tiktoken==0.13.0 pyyaml==6.0.3 jinja2==3.1.6
  fi
  PY="$VENV/bin/python"
fi
export PYTHONPATH="$CORE${PYTHONPATH:+:$PYTHONPATH}"
cd "$CORE"
echo "python: $PY"
"$PY" -c "import mlx.core as mx, mlx_lm; print('mlx', mx.__version__, '| mlx-lm', mlx_lm.__version__)"

echo; echo "== chunked gated delta vs mlx-lm's loop"
"$PY" tests/test_train_linear_attn.py
echo; echo "== gradient parity: fused / checkpoint / layered / streamed"
"$PY" tests/test_train_parity.py
echo; echo "== every method end to end on a tiny model"
"$PY" tests/test_train_methods.py

if [[ -z "${MODEL:-}" ]]; then
  echo; echo "unit checks passed. Set MODEL=<checkpoint> for short real runs."
  exit 0
fi

if [[ -n "${SIM_RAM_GB:-}" ]]; then
  export PAGED_MOE_SIMULATE_RAM_GB="$SIM_RAM_GB"
  echo; echo "planning as a ${SIM_RAM_GB} GB Mac"
fi

"$PY" - "$OUT" <<'PY'
import json, random, sys
out = sys.argv[1]
random.seed(0)
sft, pref, grpo = [], [], []
for _ in range(64):
    a, b = random.randint(10, 99), random.randint(10, 99)
    q = f"What is {a} + {b}? Answer in one short sentence."
    good, bad = f"{a} + {b} = {a + b}.", f"I think {a} + {b} is about {a + b + 10}."
    sft.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": good}]})
    pref.append({"prompt": q, "chosen": good, "rejected": bad})
    c, d = random.randint(100, 999), random.randint(11, 99)
    grpo.append({"prompt": f"What is {c} * {d}? Reply with only <answer>N</answer>.", "answer": str(c * d)})
for name, rows in (("sft", sft), ("orpo", pref), ("grpo", grpo)):
    with open(f"{out}/{name}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
PY

common=(--model "$MODEL" --template-kwargs '{"enable_thinking": false}' --save-steps 0)

echo; echo "== SFT (4 steps)"
"$PY" -m pagedmoe.train sft "${common[@]}" --data "$OUT/sft.jsonl" --output-dir "$OUT/sft" --max-steps 4
echo; echo "== SFT, experts streamed with a 4 GB cache (4 steps)"
"$PY" -m pagedmoe.train sft "${common[@]}" --data "$OUT/sft.jsonl" --output-dir "$OUT/sft-streamed" \
  --max-steps 4 --mode streamed --cache-gb 4
echo; echo "== ORPO (4 steps)"
"$PY" -m pagedmoe.train orpo "${common[@]}" --data "$OUT/orpo.jsonl" --output-dir "$OUT/orpo" --max-steps 4
echo; echo "== GRPO (3 steps, 4 prompts x 4 generations)"
"$PY" -m pagedmoe.train grpo "${common[@]}" --data "$OUT/grpo.jsonl" --output-dir "$OUT/grpo" \
  --max-steps 3 --batch-size 4 --num-generations 4 --max-completion-tokens 32 \
  --rewards 'exact_answer,regex:<answer>\d+</answer>'

echo; echo "real runs passed. adapters under $OUT"
