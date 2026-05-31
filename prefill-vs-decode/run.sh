#!/bin/bash
# Run attention-intensity measurement + plot inside the vLLM container.
# Mounts ../roofline so the plot can read its roofline.json. Writes
# results back to ./results via bind mount.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p results

IMAGE="vllm/vllm-openai:v0.19.1-cu130"

# Mount this dir + the sibling roofline dir so the plot can read both.
REPO_ROOT="$(cd .. && pwd)"

run_in_container() {
  sudo docker run --gpus all --rm \
    -v "$REPO_ROOT":/work \
    -w /work/prefill-vs-decode \
    --entrypoint "$1" \
    "$IMAGE" \
    "${@:2}"
}

echo "[1/4] measuring attention sweeps..."
run_in_container python3 measure_attention.py

echo "[2/4] comparing decode kernels..."
run_in_container python3 compare_decode_kernels.py

echo "[3/4] GQA ratio sweep..."
run_in_container python3 measure_gqa_sweep.py

echo "[4/4] plotting..."
sudo docker run --gpus all --rm \
  -v "$REPO_ROOT":/work \
  -w /work/prefill-vs-decode \
  --entrypoint bash \
  "$IMAGE" \
  -c "pip install --quiet --root-user-action=ignore matplotlib && python3 plot_attention.py && python3 plot_gqa.py"

echo
echo "done. results in: $PWD/results/"
ls -la results/
