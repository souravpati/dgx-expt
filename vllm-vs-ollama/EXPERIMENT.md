# Experiment

Two tests are run against each framework.

## Test 1 — Single-request latency

Four sequential requests. For each request we capture:

- **TTFB** — time-to-first-byte (prefill + first token)
- **Total time** — full response duration
- **Tokens generated** — from framework's own usage field
- **Tokens/sec** — `tokens_generated / total_time`

**Prompt:** `"Explain gravity in 3 sentences"`
**`max_tokens`:** 100

Scripts: [`bench-single-ollama.sh`](scripts/bench-single-ollama.sh), [`bench-single-vllm.sh`](scripts/bench-single-vllm.sh)

## Test 2 — Concurrent throughput

Ten parallel requests fired simultaneously via backgrounded `curl` processes. We wait for all to finish, then compute:

- **Wall-clock time** — from firing the first request to the last response
- **Total tokens** — summed across all responses
- **Aggregate tok/s** — `total_tokens / wall_clock_time`

**Prompt:** `"Write a haiku about CPUs"`
**`max_tokens`:** 50

Scripts: [`bench-concurrent-ollama.sh`](scripts/bench-concurrent-ollama.sh), [`bench-concurrent-vllm.sh`](scripts/bench-concurrent-vllm.sh)

## Test preconditions

Before each framework's concurrent test, the **other framework's GPU memory must be freed**. vLLM's ~110 GB reservation plus Ollama's ~21 GB exceeds the 128 GB unified memory on the Spark.

```bash
# Before running Ollama tests
# Stop the vLLM container (Ctrl+C in its terminal)

# Before running vLLM tests
ollama stop llama3.1:8b

# Verify:
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader
```

## Warmup

Both frameworks benefit from a warmup request before measurement:
- Ollama: loads model into GPU memory (~2–3 s disk-to-GPU transfer)
- vLLM: compiles CUDA graphs and fills caches (first few requests are slower)

The single-request scripts include a warmup call before measurement. Runs 2–4 are what we report.

## What's not measured

- **Quality of generated text** — noted qualitatively, not scored
- **Prefill scaling with prompt length** — all prompts are short
- **Sustained load over long periods** — 10 concurrent requests is a snapshot, not a stress test
- **Memory pressure effects** — KV cache eviction isn't triggered by these tiny prompts
