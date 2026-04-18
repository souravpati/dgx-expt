#!/bin/bash
# KV cache behavior test for vLLM.
# Demonstrates how KV cache utilization scales with output length and
# concurrent requests.
#
# Measures:
#   - system memory used (DGX Spark has unified CPU+GPU memory, so
#     nvidia-smi memory.used returns "Not Supported"; we use `free -m`)
#   - vLLM's own kv_cache_usage_perc metric (1.0 = 100% of KV pool)
#
# Prereq: vLLM container running on :8000 (see start-vllm-docker.sh).
# Output: /tmp/vllm_kv_trace.csv  (timestamp, mem_used_mb, kv_pct, scenario)

set -u

ENDPOINT="http://localhost:8000/v1/chat/completions"
METRICS="http://localhost:8000/metrics"
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
TRACE="/tmp/vllm_kv_trace.csv"
SCENARIO_FILE="/tmp/vllm_kv_scenario"

send_req() {
  local max_tokens=$1
  curl -s --max-time 120 "$ENDPOINT" -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a detailed essay about the history of computing. Be thorough.\"}],\"max_tokens\":$max_tokens}" \
    > /dev/null
}

# Run a list of concurrent requests and wait ONLY for them (not for sampler)
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

echo "timestamp,mem_used_mb,kv_pct,scenario" > "$TRACE"
echo "warmup" > "$SCENARIO_FILE"

# Single persistent sampler — 1s interval
(
  while [ -f "$SCENARIO_FILE" ]; do
    ts=$(date +%s)
    mem=$(mem_used_mb)
    kv=$(curl -s --max-time 2 "$METRICS" 2>/dev/null | grep -E '^vllm:kv_cache_usage_perc' | head -1 | awk '{print $2}')
    scen=$(cat "$SCENARIO_FILE" 2>/dev/null)
    echo "$ts,$mem,$kv,$scen" >> "$TRACE"
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

echo "=== vLLM KV Cache Behavior Test ==="
echo ""

echo "[warmup] loading CUDA graphs..."
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
echo "=== Summary (peak values per scenario) ==="
python3 <<EOF
import csv
rows = list(csv.DictReader(open("$TRACE")))
scenarios = {}
for r in rows:
    s = r['scenario']
    if not s: continue
    if s not in scenarios:
        scenarios[s] = {'mem_peak': 0, 'mem_min': 10**9, 'kv_peak': 0.0, 'samples': 0}
    try:
        mb = int(r['mem_used_mb'])
        scenarios[s]['mem_peak'] = max(scenarios[s]['mem_peak'], mb)
        scenarios[s]['mem_min'] = min(scenarios[s]['mem_min'], mb)
    except: pass
    try:
        if r['kv_pct']:
            scenarios[s]['kv_peak'] = max(scenarios[s]['kv_peak'], float(r['kv_pct']))
    except: pass
    scenarios[s]['samples'] += 1

order = ["idle", "1req_short", "1req_long", "5conc_short", "5conc_long", "10conc_long"]
print(f"{'Scenario':<20} {'Sys mem peak (MB)':>18} {'KV cache peak':>15} {'Samples':>10}")
print("-" * 66)
for s in order:
    if s in scenarios:
        d = scenarios[s]
        print(f"{s:<20} {d['mem_peak']:>18} {d['kv_peak']*100:>14.2f}% {d['samples']:>10}")

print("")
print("Note: system memory is ~constant (~118 GB) because vLLM reserves")
print("its KV cache pool up front. What changes is the 'kv cache peak' —")
print("the percentage of that pool actually in use.")
EOF

echo ""
echo "Full trace: $TRACE"
