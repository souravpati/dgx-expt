#!/bin/bash
# Run probe_marlin.py in the vLLM container and dump output.
set -euo pipefail
cd "$(dirname "$0")"

sudo docker run --gpus all --rm \
  -v "$PWD":/work \
  -w /work \
  --entrypoint python3 \
  vllm/vllm-openai:v0.19.1-cu130 \
  probe_marlin.py
