#!/bin/bash
# Run roofline measurement + plot inside the vLLM container, which has
# PyTorch with Blackwell (sm_121) tensor-core kernels. Writes results
# back to ./results on the host via bind mount.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p results

IMAGE="vllm/vllm-openai:v0.19.1-cu130"

run_in_container() {
  sudo docker run --gpus all --rm \
    -v "$PWD":/work \
    -w /work \
    --entrypoint python3 \
    "$IMAGE" \
    "$@"
}

echo "[1/2] measuring..."
run_in_container measure_roofline.py

# matplotlib isn't in the vLLM image -- install it once into a writable
# location inside the container, then plot.
echo "[2/2] plotting..."
sudo docker run --gpus all --rm \
  -v "$PWD":/work \
  -w /work \
  --entrypoint bash \
  "$IMAGE" \
  -c "pip install --quiet --root-user-action=ignore matplotlib && python3 plot_roofline.py"

echo
echo "done. results in: $PWD/results/"
ls -la results/
