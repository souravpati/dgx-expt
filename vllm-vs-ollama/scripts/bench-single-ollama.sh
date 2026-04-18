#!/bin/bash
# Sequential latency benchmark for Ollama serving llama3.1:8b.
# Prereq: Ollama server running on :11434, model pulled.

set -u

ENDPOINT="http://localhost:11434/api/generate"
MODEL="llama3.1:8b"
PROMPT="Explain gravity in 3 sentences"
RUNS=4

gpu_mem() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
}

echo "=== Ollama single-request benchmark: $MODEL ==="
echo ""
echo "GPU memory before warmup: $(gpu_mem) MB"

# Warmup — loads model to GPU if unloaded
curl -s "$ENDPOINT" -d "{\"model\":\"$MODEL\",\"prompt\":\"hi\",\"stream\":false}" > /dev/null
echo "GPU memory after warmup:  $(gpu_mem) MB"
echo ""

for i in $(seq 1 $RUNS); do
  echo "--- Run $i ---"
  result=$(curl -s -w "\nTTFB:%{time_starttransfer}\nTOTAL:%{time_total}" \
    "$ENDPOINT" \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"$PROMPT\",\"stream\":false}")

  json=$(echo "$result" | head -1)
  ttfb=$(echo "$result" | grep "TTFB:" | cut -d: -f2)
  total=$(echo "$result" | grep "TOTAL:" | cut -d: -f2)

  eval_count=$(echo "$json" | python3 -c "import sys,json; print(json.load(sys.stdin).get('eval_count',0))")
  eval_duration=$(echo "$json" | python3 -c "import sys,json; print(json.load(sys.stdin).get('eval_duration',0))")
  prompt_eval_duration=$(echo "$json" | python3 -c "import sys,json; print(json.load(sys.stdin).get('prompt_eval_duration',0))")

  tps=$(python3 -c "print(f'{$eval_count / ($eval_duration / 1e9):.1f}' if $eval_duration else 'N/A')")
  prefill_ms=$(python3 -c "print(f'{$prompt_eval_duration / 1e6:.1f}')")

  echo "  TTFB:        ${ttfb}s"
  echo "  Total:       ${total}s"
  echo "  Tokens:      $eval_count"
  echo "  Tokens/sec:  $tps"
  echo "  Prefill:     ${prefill_ms}ms"
  echo "  GPU mem:     $(gpu_mem) MB"
  echo ""
done
