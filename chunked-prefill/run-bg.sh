#!/bin/bash
# Background launcher for one mode of the chunked-prefill experiment.
# Pass the mode as $1: "off" or "on". Survives SSH disconnect.
#
#   ./run-bg.sh off   # records results/chunked_off.json
#   ./run-bg.sh on    # records results/chunked_on.json
#
# Assumes the matching vLLM server is already running.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p results

MODE="${1:-off}"
if [[ "$MODE" != "off" && "$MODE" != "on" ]]; then
  echo "usage: $0 {off|on}"
  exit 1
fi

OUT="results/chunked_${MODE}.json"
LOG="results/bench_${MODE}.log"
PID_FILE="results/bench_${MODE}.pid"

setsid nohup bash -c "
  set -euo pipefail
  echo '===== started '\"\$(date -Is)\"' (mode=$MODE) ====='
  curl -fsS http://localhost:8000/v1/models > /dev/null || {
    echo 'ERROR: vLLM not reachable'
    exit 1
  }
  python3 -u bench_chunked.py --out '$OUT' --label '$MODE'
  echo '===== finished '\"\$(date -Is)\"' ====='
" > "$LOG" 2>&1 < /dev/null &

echo $! > "$PID_FILE"

echo "Launched $MODE bench in background."
echo "  PID:   $(cat $PID_FILE)"
echo "  Log:   $PWD/$LOG"
echo "  Out:   $PWD/$OUT"
echo
echo "Tail:   tail -f $PWD/$LOG"
