# Results

All numbers from runs on 2026-04-18, llama3.1:8b-instruct, DGX Spark (GB10).

## Single-request latency

### Ollama (Q4_K_M, 4-bit)

| Run | TTFB | Total | Tokens | Tokens/sec | Prefill |
|-----|------|-------|--------|------------|---------|
| 1 | 2.26 s | 2.26 s | 89 | 42.8 | 23 ms |
| 2 | 2.78 s | 2.78 s | 112 | 42.9 | 25 ms |
| 3 | 2.52 s | 2.52 s | 99 | 42.9 | 26 ms |

**Average: ~43 tok/s.** Extremely consistent.

### vLLM (FP16, 16-bit)

| Run | TTFB | Total | Tokens | Tokens/sec |
|-----|------|-------|--------|------------|
| 1 | 5.82 s | 5.82 s | 81 | 13.9 |
| 2 | 6.19 s | 6.19 s | 87 | 14.1 |
| 3 | 6.69 s | 6.69 s | 94 | 14.0 |
| 4 | 7.11 s | 7.11 s | 100 | 14.1 |

**Average: ~14 tok/s.** Also consistent.

### Observation

Ollama is **~3× faster per-token** on single requests. The gap is purely a quantization effect: 4-bit weights move ~4× less data through GPU memory per decode step. Quality differs — vLLM's FP16 output should be slightly better but both are coherent llama3.1 responses.

## Concurrent throughput (10 parallel requests, `max_tokens=50`)

| Metric | Ollama (Q4_K_M) | vLLM (FP16) |
|--------|-----------------|-------------|
| Wall-clock for 10 requests | 4.26 s | **2.52 s** |
| Total tokens generated | 158 | 156 |
| **Aggregate tokens/sec** | 37.1 | **61.9** |

## Concurrency scaling

For each framework, comparing single-request tok/s to concurrent-aggregate tok/s:

| | Single-request | Concurrent (10) | Speedup |
|---|---|---|---|
| **Ollama** | 43 | 37 | **0.86×** (gets worse) |
| **vLLM**   | 14 | 62 | **4.4×** (scales well) |

### Key observation

**Ollama concurrent throughput is *lower* than single-request.** Requests queue and contend for GPU compute with minimal batching — adding parallel load doesn't help and marginally hurts.

**vLLM concurrent throughput is 4.4× single-request.** Continuous batching + PagedAttention merge sequences into a single forward pass, extracting real GPU utilization.

vLLM wins aggregate throughput **even while running 4× heavier (FP16 vs Q4) weights**. A vLLM deployment with quantized weights would likely reach ~150-200 aggregate tok/s on this hardware.

## Bottom-line comparison

| Use case | Winner | Why |
|----------|--------|-----|
| Desktop chat, one user | **Ollama** | 3× lower latency per token, simpler setup, quantization by default |
| API serving, many users | **vLLM** | Continuous batching scales GPU utilization with load |
| Model experimentation | Either | Both work; Ollama easier, vLLM more configurable |
| Production at scale | **vLLM** | Designed for it; PagedAttention, prefix caching, tensor parallelism |

## Caveats

- Not apples-to-apples quantization. A fair comparison would pit both on the same precision. The "Ollama wins single-request" result reflects defaults, not the frameworks themselves.
- Only 10 concurrent requests tested. Real throughput curves require sweeps to see where each framework saturates.
- Prompt is short (<20 tokens). Prefill cost scales linearly with prompt length, which would widen the TTFB gap between frameworks.
- vLLM's 110 GB memory reservation is largely KV cache capacity, not model weights (which are ~16 GB FP16). That reservation is what lets it batch efficiently.
