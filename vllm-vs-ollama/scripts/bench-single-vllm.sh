#!/bin/bash
# Sequential latency benchmark for vLLM serving Llama 3.1 8B (FP16).
# Prereq: vLLM container running on :8000 (see start-vllm-docker.sh).

set -u

ENDPOINT="http://localhost:8000/v1/chat/completions"
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
PROMPT="Explain gravity in 3 sentences"
MAX_TOKENS=100
RUNS=4

echo "=== vLLM single-request benchmark: $MODEL ==="
echo ""

# Warmup — let CUDA graphs compile
echo "Warming up..."
curl -s "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5}" > /dev/null
echo "Ready."
echo ""

for i in $(seq 1 $RUNS); do
  echo "--- Run $i ---"
  result=$(curl -s -w "\nTTFB:%{time_starttransfer}\nTOTAL:%{time_total}" \
    "$ENDPOINT" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAX_TOKENS}")

  ttfb=$(echo "$result" | grep "TTFB:" | cut -d: -f2)
  total=$(echo "$result" | grep "TOTAL:" | cut -d: -f2)
  json=$(echo "$result" | grep -v "TTFB:" | grep -v "TOTAL:")

  tokens=$(echo "$json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])" 2>/dev/null)
  tps=$(python3 -c "print(f'{$tokens / $total:.1f}' if '$tokens' and '$tokens' != '0' else 'N/A')")

  echo "  TTFB:        ${ttfb}s"
  echo "  Total:       ${total}s"
  echo "  Tokens:      $tokens"
  echo "  Tokens/sec:  $tps"
  echo ""
done
