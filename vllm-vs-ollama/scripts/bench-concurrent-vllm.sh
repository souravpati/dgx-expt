#!/bin/bash
# 10-parallel-request throughput benchmark for vLLM Llama 3.1 8B (FP16).
# Prereq: vLLM container running on :8000 (see start-vllm-docker.sh).

set -u

ENDPOINT="http://localhost:8000/v1/chat/completions"
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
PROMPT="Write a haiku about CPUs"
MAX_TOKENS=50
PARALLEL=10

echo "=== vLLM concurrent benchmark: $PARALLEL parallel requests ==="
echo ""

# Warmup
curl -s "$ENDPOINT" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5}" > /dev/null

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
echo "  Wall-clock time:  ${total_time}s"
echo "  Total tokens:     $total_tokens"
echo "  Aggregate tok/s:  $aggregate_tps"

rm -f /tmp/vllm_resp_*.json
