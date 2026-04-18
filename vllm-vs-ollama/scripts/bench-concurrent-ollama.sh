#!/bin/bash
# 10-parallel-request throughput benchmark for Ollama llama3.1:8b.
# Prereq: Ollama server running, model loaded (run once before this script).

set -u

ENDPOINT="http://localhost:11434/api/chat"
MODEL="llama3.1:8b"
PROMPT="Write a haiku about CPUs"
MAX_TOKENS=50
PARALLEL=10

echo "=== Ollama concurrent benchmark: $PARALLEL parallel requests ==="
echo ""

# Warmup to make sure model is loaded
curl -s "$ENDPOINT" -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false}" > /dev/null

start_time=$(date +%s.%N)

for i in $(seq 1 $PARALLEL); do
  (curl -s "$ENDPOINT" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"stream\":false,\"options\":{\"num_predict\":$MAX_TOKENS}}" \
    > /tmp/ollama_resp_$i.json) &
done
wait

end_time=$(date +%s.%N)
total_time=$(python3 -c "print(f'{$end_time - $start_time:.2f}')")

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
echo "  Wall-clock time:  ${total_time}s"
echo "  Total tokens:     $total_tokens"
echo "  Aggregate tok/s:  $aggregate_tps"

rm -f /tmp/ollama_resp_*.json
