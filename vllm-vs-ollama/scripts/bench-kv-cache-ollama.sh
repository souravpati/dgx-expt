#!/bin/bash
# KV cache behavior test for Ollama.
# Demonstrates how memory scales with output length and concurrent requests.
#
# DGX Spark has unified CPU+GPU memory, so we use `free -m` to capture
# total memory pressure. Ollama doesn't expose KV cache metrics, so this
# is coarser than the vLLM version but still shows the growth pattern.
#
# Prereq: Ollama server running on :11434, llama3.1:8b pulled.
# Output: /tmp/ollama_kv_trace.csv  (timestamp, mem_used_mb, scenario)

set -u

ENDPOINT="http://localhost:11434/api/chat"
MODEL="llama3.1:8b"
TRACE="/tmp/ollama_kv_trace.csv"
SCENARIO_FILE="/tmp/ollama_kv_scenario"

send_req() {
  local max_tokens=$1
  curl -s --max-time 180 "$ENDPOINT" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a detailed essay about the history of computing. Be thorough.\"}],\"stream\":false,\"options\":{\"num_predict\":$max_tokens}}" \
    > /dev/null
}

run_concurrent() {
  local n=$1
  local max_tokens=$2
  local pids=()
  for i in $(seq 1 $n); do
    send_req "$max_tokens" &
    pids+=($!)
  done
  wait "${pids[@]}"
}

mem_used_mb() {
  free -m | awk '/^Mem:/ {print $3}'
}

echo "timestamp,mem_used_mb,scenario" > "$TRACE"
echo "warmup" > "$SCENARIO_FILE"

(
  while [ -f "$SCENARIO_FILE" ]; do
    ts=$(date +%s)
    mem=$(mem_used_mb)
    scen=$(cat "$SCENARIO_FILE" 2>/dev/null)
    echo "$ts,$mem,$scen" >> "$TRACE"
    sleep 1
  done
) &
SAMPLER_PID=$!

mark() {
  echo "$1" > "$SCENARIO_FILE"
}

cleanup() {
  rm -f "$SCENARIO_FILE"
  kill $SAMPLER_PID 2>/dev/null
  wait $SAMPLER_PID 2>/dev/null
}
trap cleanup EXIT

echo "=== Ollama KV Cache Behavior Test ==="
echo ""

echo "[warmup] ensuring model is loaded..."
send_req 5
sleep 2

mark "idle"
echo "[1/6] idle baseline (5s)..."
sleep 5

mark "1req_short"
echo "[2/6] 1 request, max_tokens=50..."
send_req 50
sleep 2

mark "1req_long"
echo "[3/6] 1 request, max_tokens=500..."
send_req 500
sleep 2

mark "5conc_short"
echo "[4/6] 5 concurrent, max_tokens=50..."
run_concurrent 5 50
sleep 2

mark "5conc_long"
echo "[5/6] 5 concurrent, max_tokens=500..."
run_concurrent 5 500
sleep 2

mark "10conc_long"
echo "[6/6] 10 concurrent, max_tokens=500..."
run_concurrent 10 500
sleep 3

cleanup

echo ""
echo "=== Summary (peak system memory per scenario) ==="
python3 <<EOF
import csv
rows = list(csv.DictReader(open("$TRACE")))
scenarios = {}
for r in rows:
    s = r['scenario']
    if not s: continue
    if s not in scenarios:
        scenarios[s] = {'mem_peak': 0, 'mem_min': 10**9, 'samples': 0}
    try:
        mb = int(r['mem_used_mb'])
        scenarios[s]['mem_peak'] = max(scenarios[s]['mem_peak'], mb)
        scenarios[s]['mem_min'] = min(scenarios[s]['mem_min'], mb)
    except: pass
    scenarios[s]['samples'] += 1

# Use idle as baseline for delta
base = scenarios.get("idle", {}).get("mem_peak", 0)

order = ["idle", "1req_short", "1req_long", "5conc_short", "5conc_long", "10conc_long"]
print(f"{'Scenario':<20} {'Mem peak (MB)':>14} {'Delta from idle':>18} {'Samples':>10}")
print("-" * 66)
for s in order:
    if s in scenarios:
        d = scenarios[s]
        delta = d['mem_peak'] - base if base else 0
        sign = '+' if delta >= 0 else ''
        print(f"{s:<20} {d['mem_peak']:>14} {sign}{delta:>17} {d['samples']:>10}")

print("")
print("Note: Ollama allocates KV cache dynamically per request,")
print("so memory grows with concurrent load. This is different from")
print("vLLM which pre-reserves the pool up-front.")
EOF

echo ""
echo "Full trace: $TRACE"
