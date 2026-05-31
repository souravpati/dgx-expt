#!/bin/bash
# vLLM server with chunked prefill explicitly DISABLED.
# Tmux it (`tmux new -s vllm`) before running so it survives SSH disconnect.
set -euo pipefail

sudo docker run --gpus all -it --rm \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN="$(cat ~/.cache/huggingface/token)" \
  vllm/vllm-openai:v0.19.1-cu130 \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --no-enable-chunked-prefill
