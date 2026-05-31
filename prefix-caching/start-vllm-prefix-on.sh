#!/bin/bash
# vLLM server with prefix caching ENABLED.
# Tmux this so it survives SSH disconnect.
set -euo pipefail

sudo docker run --gpus all -it --rm \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN="$(cat ~/.cache/huggingface/token)" \
  vllm/vllm-openai:v0.19.1-cu130 \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching
