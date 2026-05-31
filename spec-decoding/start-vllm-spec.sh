#!/bin/bash
# Launch vLLM with speculative decoding configured at K = $1 spec tokens.
# K=0 disables spec (baseline). K>=1 enables with Llama-3.2-1B as draft.
#
# Tmux this so it survives SSH disconnect.
set -euo pipefail

K="${1:-0}"

ARGS=(
  meta-llama/Meta-Llama-3.1-8B-Instruct
  --max-model-len 4096
  --gpu-memory-utilization 0.85
)

if (( K > 0 )); then
  ARGS+=(
    --speculative-config "{\"model\": \"meta-llama/Llama-3.2-1B-Instruct\", \"num_speculative_tokens\": $K}"
  )
fi

echo "Launching vLLM with K=$K"
printf '  arg: %s\n' "${ARGS[@]}"

sudo docker run --gpus all -it --rm \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN="$(cat ~/.cache/huggingface/token)" \
  vllm/vllm-openai:v0.19.1-cu130 \
  "${ARGS[@]}"
