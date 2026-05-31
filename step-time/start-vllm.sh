#!/bin/bash
# Launch vLLM server for Llama 3.1 8B Instruct on :8000.
# Uses the same image and HF cache as ../vllm-vs-ollama/.
# Stop with Ctrl+C in the terminal where this is running.
#
# We bump --max-model-len to 40k so we can test S=32k decode contexts
# (need headroom for the prompt itself).

set -euo pipefail

sudo docker run --gpus all -it --rm \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN="$(cat ~/.cache/huggingface/token)" \
  vllm/vllm-openai:v0.19.1-cu130 \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.85
