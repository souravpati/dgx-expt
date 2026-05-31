#!/bin/bash
# Run benchmark + plot. Assumes vLLM server is already up on :8000
# (start it in another terminal with ./start-vllm.sh).
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p results

echo "[1/3] checking vLLM is reachable..."
curl -fsS http://localhost:8000/v1/models > /dev/null || {
  echo "ERROR: vLLM not responding on :8000. Run ./start-vllm.sh first."
  exit 1
}

echo "[2/3] regenerating prediction..."
python3 predict_step_time.py

echo "[3/3] running streaming benchmark..."
python3 bench_step_time.py

echo "[4/4] plotting..."
# matplotlib on host; install only if missing.
python3 -c "import matplotlib" 2>/dev/null || pip install --user --quiet matplotlib
python3 plot_step_time.py

echo
echo "done."
ls -la results/
