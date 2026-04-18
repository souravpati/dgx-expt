#!/bin/bash
# 10-parallel-request throughput benchmark for vLLM Llama 3.1 8B (FP16).
# Prereq: vLLM container running on :8000 (see start-vllm-docker.sh).

set -u

ENDPOINT="http://localhost:8000/v1/chat/completions"
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
PROMPT="Write a haiku about CPUs"
MAX_TOKENS=50
PARALLEL=10
GPU_LOG="/tmp/gpu_usage_vllm.log"

gpu_mem() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
}

echo "=== vLLM concurrent benchmark: $PARALLEL parallel requests ==="
echo ""
echo "GPU memory before warmup: $(gpu_mem) MB"

# Warmup
curl -s "$ENDPOINT" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5}" > /dev/null
echo "GPU memory after warmup:  $(gpu_mem) MB"
echo ""

# Start background GPU memory logger (1-sec samples)
nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 1 > "$GPU_LOG" 2>/dev/null &
LOGGER_PID=$!

start_time=$(date +%s.%N)

for i in $(seq 1 $PARALLEL); do
  (curl -s "$ENDPOINT" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAX_TOKENS}" \
    > /tmp/vllm_resp_$i.json) &
done
wait

end_time=$(date +%s.%N)
total_time=$(python3 -c "print(f'{$end_time - $start_time:.2f}')")

# Stop logger and compute peak memory
kill $LOGGER_PID 2>/dev/null
wait $LOGGER_PID 2>/dev/null
peak_mem=$(awk -F',' 'NR>1 {gsub(/[^0-9]/,"",$2); if ($2>m) m=$2} END {print m}' "$GPU_LOG")

total_tokens=$(python3 -c "
import json, glob
total = 0
for f in sorted(glob.glob('/tmp/vllm_resp_*.json')):
    try:
        total += json.load(open(f))['usage']['completion_tokens']
    except: pass
print(total)
")

aggregate_tps=$(python3 -c "print(f'{$total_tokens / $total_time:.1f}')")

echo ""
echo "  Wall-clock time:   ${total_time}s"
echo "  Total tokens:      $total_tokens"
echo "  Aggregate tok/s:   $aggregate_tps"
echo "  Peak GPU mem:      ${peak_mem} MB"
echo "  Full GPU trace:    $GPU_LOG"

rm -f /tmp/vllm_resp_*.json
