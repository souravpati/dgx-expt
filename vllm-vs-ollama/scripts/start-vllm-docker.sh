#!/bin/bash
# Launch vLLM server in NVIDIA-compatible container.
# First run pulls ~8 GB image; subsequent runs are instant.
# Model weights cached at ~/.cache/huggingface (first pull ~16 GB for Llama 3.1 8B FP16).
#
# Listens on http://localhost:8000. Stop with Ctrl+C.

sudo docker run --gpus all -it --rm \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=$(cat ~/.cache/huggingface/token) \
  vllm/vllm-openai:v0.19.1-cu130 \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct
