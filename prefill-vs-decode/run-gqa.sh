#!/bin/bash
# Just the GQA sweep + plot. Skips the heavier attention/decode-kernels
# benches that have already been recorded. Takes <1 min.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p results

IMAGE="vllm/vllm-openai:v0.19.1-cu130"
REPO_ROOT="$(cd .. && pwd)"

echo "[1/2] GQA ratio sweep..."
sudo docker run --gpus all --rm \
  -v "$REPO_ROOT":/work \
  -w /work/prefill-vs-decode \
  --entrypoint python3 \
  "$IMAGE" \
  measure_gqa_sweep.py

echo "[2/2] plot..."
sudo docker run --gpus all --rm \
  -v "$REPO_ROOT":/work \
  -w /work/prefill-vs-decode \
  --entrypoint bash \
  "$IMAGE" \
  -c "pip install --quiet --root-user-action=ignore matplotlib && python3 plot_gqa.py"

echo
echo "done."
ls -la results/gqa_sweep.*
