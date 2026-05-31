#!/bin/bash
# Launch vLLM under Nsight Systems profiler.
# Trace output written to ./nsys-traces/vllm_trace.nsys-rep on the host.
#
# Usage:
#   1. Run this script (it will block, server stays in foreground).
#   2. Wait for "Application startup complete" log line.
#   3. In a second terminal, run a benchmark (e.g. bench-single-vllm.sh).
#   4. Ctrl+C this process when done — nsys flushes the trace on shutdown.
#   5. Open the trace: scp nsys-traces/*.nsys-rep to your laptop,
#      open in Nsight Systems GUI. Or run `nsys stats nsys-traces/vllm_trace.nsys-rep`
#      for a CLI summary.
#
# Note: traces grow ~100 MB/min. Keep runs short (< 2 min of load).

set -u

TRACE_DIR="$(cd "$(dirname "$0")/.." && pwd)/nsys-traces"
mkdir -p "$TRACE_DIR"

echo "Trace will be written to: $TRACE_DIR/vllm_trace.nsys-rep"
echo ""

sudo docker run --gpus all -it --rm \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$TRACE_DIR":/traces \
  -e HF_TOKEN=$(cat ~/.cache/huggingface/token) \
  --entrypoint bash \
  vllm/vllm-openai:v0.19.1-cu130 \
  -c '
set -e
echo ">>> Installing nsight-systems-cli inside container..."
apt-get update -qq
apt-get install -y --no-install-recommends nsight-systems-cli 2>/dev/null \
  || apt-get install -y --no-install-recommends nsight-systems
which nsys && nsys --version

echo ">>> Launching vLLM under nsys..."
exec nsys profile \
  --trace=cuda,nvtx,osrt \
  --gpu-metrics-device=all \
  --output=/traces/vllm_trace \
  --force-overwrite=true \
  --stats=false \
  python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct
'
