# KV Cache Behavior — vLLM vs Ollama

Goal: see how the two frameworks use memory as load scales.

## Why KV cache matters

When an LLM generates a token, it needs all previous tokens' attention key/value vectors from every layer. Rather than recomputing them each step, these are kept in a **KV cache** — the thing that actually dominates GPU memory during inference.

KV cache grows with:
- **Sequence length** — longer output → more cached entries
- **Concurrent requests** — more users → more parallel sequences to cache
- **Model layers × heads × head-dim** — architectural constant per model

**Therefore:** how a framework manages its KV cache determines how it scales with load.

## How we measured

Six scenarios with increasing load:

| # | Scenario | Output tokens | Concurrency |
|---|----------|---------------|-------------|
| 1 | `idle` | — | — |
| 2 | `1req_short` | 50 | 1 |
| 3 | `1req_long` | 500 | 1 |
| 4 | `5conc_short` | 50 | 5 |
| 5 | `5conc_long` | 500 | 5 |
| 6 | `10conc_long` | 500 | 10 |

**Sampling:** 1 Hz background sampler captures system memory (`free -m`) and, for vLLM, `vllm:kv_cache_usage_perc` from `/metrics`. DGX Spark's unified memory means `nvidia-smi memory.used` reports "Not Supported", so `free -m` is the right metric on this hardware.

Scripts: [`scripts/bench-kv-cache-vllm.sh`](scripts/bench-kv-cache-vllm.sh), [`scripts/bench-kv-cache-ollama.sh`](scripts/bench-kv-cache-ollama.sh)

## Results

### vLLM (FP16, pool pre-allocated)

| Scenario | Sys mem peak (MB) | KV cache peak | Samples |
|----------|------------------:|--------------:|--------:|
| idle | 118,563 | 0.00% | 5 |
| 1req_short | 118,551 | 0.01% | 5 |
| 1req_long | 118,548 | 0.07% | 39 |
| 5conc_short | 118,540 | 0.05% | 5 |
| 5conc_long | 118,553 | 0.36% | 38 |
| 10conc_long | 118,637 | 0.69% | 39 |

**System memory stays ~constant (~118 GB).** vLLM reserved its KV cache pool up front. What changes is how much of that pool is in use.

**KV cache scales predictably with the work:**
- `1req_long` (0.07%) vs `1req_short` (0.01%) → 10× output = 7× cache ✓
- `5conc_long` (0.36%) vs `1req_long` (0.07%) → 5× concurrency = 5× cache ✓
- `10conc_long` (0.69%) vs `5conc_long` (0.36%) → 2× concurrency = 2× cache ✓

**Headroom:** At 0.69% utilization for 10 concurrent 500-token requests, the reserved pool can handle ~1,400 simultaneous requests before filling. This is why vLLM shines on multi-tenant serving infra.

### Ollama (Q4_K_M, dynamic allocation)

| Scenario | Mem peak (MB) | Δ from idle (MB) | Samples |
|----------|--------------:|-----------------:|--------:|
| idle | 27,030 | 0 | 5 |
| 1req_short | 27,015 | -15 | 3 |
| 1req_long | 27,000 | -30 | 15 |
| 5conc_short | 27,096 | **+66** | 9 |
| 5conc_long | 27,107 | **+77** | 67 |
| 10conc_long | 27,246 | **+216** | 134 |

**Idle footprint is ~4× smaller than vLLM** (~27 GB — mostly the Q4 quantized weights).

**Memory grows dynamically with concurrent load.** Single-request scenarios even show *lower* memory than idle (cache cleanup / buffer release after prior scenario). Only at 5+ concurrent does memory climb past idle by 66-216 MB.

**Ollama doesn't expose KV cache metrics.** The growth we see via `free -m` is the KV cache plus any runner-process allocations — but there's no way to get a framework-level usage percentage.

## Head-to-head

| Aspect | vLLM | Ollama |
|--------|------|--------|
| Idle memory | 118 GB (pool reserved) | 27 GB (weights only) |
| Memory under 10-conc load | +74 MB | +216 MB |
| Allocation strategy | Pre-reserve pool, page in/out | Allocate per-request |
| KV cache visibility | Exposed via `/metrics` | Hidden |
| Scales with concurrency? | Yes, near-linearly | Yes, but no batching wins |
| Time for 10conc_long scenario | ~39 s | ~134 s |

The most striking number is the last row: **Ollama took 3.4× longer** to process the 10-concurrent scenario. That's continuous batching showing up again, just from a different angle — Ollama needs more memory per request *and* takes longer because it doesn't batch.

## When each design wins

**vLLM's pre-allocated pool is the right design when:**
- You serve an API with unpredictable concurrency
- You need consistent latency regardless of load
- You have GPU memory to dedicate to the serving workload
- You want to saturate the GPU efficiently

**Ollama's dynamic allocation is the right design when:**
- You want small idle footprint (shares memory with other apps)
- Single-user or low-concurrency workloads
- You prioritize simplicity over throughput
- You're developing/prototyping on a laptop

## Caveats

- Both tests ran on the same hardware but with different model quantization (FP16 vs Q4). Ollama's smaller footprint is partly quantization, not just its allocation strategy.
- vLLM's memory reservation (~110 GB on Spark) is a configurable default. It can be tuned down via `--gpu-memory-utilization`. The "scales with load" behavior would look similar but with a smaller pool.
- DGX Spark's unified memory (CPU+GPU share the same physical memory) makes memory measurement different from typical discrete-GPU setups. On a traditional system with dedicated VRAM, you'd measure via `nvidia-smi` and see a cleaner separation.
