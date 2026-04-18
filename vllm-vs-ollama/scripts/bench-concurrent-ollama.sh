#!/bin/bash
# 10-parallel-request throughput benchmark for Ollama llama3.1:8b.
# Prereq: Ollama server running, model loaded (run once before this script).

set -u

ENDPOINT="http://localhost:11434/api/chat"
MODEL="llama3.1:8b"
PROMPT="Write a haiku about CPUs"
MAX_TOKENS=50
PARALLEL=10
GPU_LOG="/tmp/gpu_usage_ollama.log"

gpu_mem() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
}

echo "=== Ollama concurrent benchmark: $PARALLEL parallel requests ==="
echo ""
echo "GPU memory before warmup: $(gpu_mem) MB"

# Warmup to make sure model is loaded
curl -s "$ENDPOINT" -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false}" > /dev/null
echo "GPU memory after warmup:  $(gpu_mem) MB"
echo ""

# Start background GPU memory logger (1-sec samples)
nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 1 > "$GPU_LOG" 2>/dev/null &
LOGGER_PID=$!

start_time=$(date +%s.%N)

for i in $(seq 1 $PARALLEL); do
  (curl -s "$ENDPOINT" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"stream\":false,\"options\":{\"num_predict\":$MAX_TOKENS}}" \
    > /tmp/ollama_resp_$i.json) &
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
for f in sorted(glob.glob('/tmp/ollama_resp_*.json')):
    try:
        total += json.load(open(f)).get('eval_count', 0)
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

rm -f /tmp/ollama_resp_*.json
