#!/bin/bash
# Launch the benchmark + plot in the background so it survives SSH
# disconnect. Logs to results/bench.log, PID in results/bench.pid.
#
# Usage:
#   ./run-bg.sh                      # default sweep
#   ./run-bg.sh --max-tokens 200 --prefill-skip 120 --timeout 300
#
# Check progress later:
#   tail -f results/bench.log
#   cat results/bench.pid     # PID
#   ps -p $(cat results/bench.pid)   # is it still alive?
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p results

LOG=results/bench.log
PID_FILE=results/bench.pid

# Forward all args to bench_step_time.py
setsid nohup bash -c "
  set -euo pipefail
  echo '===== started '\"\$(date -Is)\"' ====='
  echo 'args: $*'

  echo '[1/3] checking vLLM...'
  curl -fsS http://localhost:8000/v1/models > /dev/null || {
    echo 'ERROR: vLLM not reachable on :8000'
    exit 1
  }

  echo '[2/3] regenerating prediction...'
  python3 predict_step_time.py | tail -5

  echo '[3/3] benchmark...'
  python3 -u bench_step_time.py $*

  echo '[plot]'
  python3 plot_step_time.py

  echo '===== finished '\"\$(date -Is)\"' ====='
" > "$LOG" 2>&1 < /dev/null &

echo $! > "$PID_FILE"

echo "Launched bench in background."
echo "  PID:   $(cat $PID_FILE)"
echo "  Log:   $PWD/$LOG"
echo
echo "Reconnect later and run:"
echo "  tail -f $PWD/$LOG"
echo "  ps -p \$(cat $PWD/$PID_FILE)   # check if still running"
