# Empirical roofline on DGX Spark (GB10)

A microbenchmark that measures the actual compute and memory ceilings
of this machine, then sweeps matmul shapes across five precisions to
see where each sits on the roofline. Sister experiment to
[`../vllm-vs-ollama/`](../vllm-vs-ollama/) — the roofline gives the
theoretical frame for the serving numbers in that folder.

## Hardware

- **NVIDIA DGX Spark** (GB10 Grace Blackwell, sm_121)
- 20-core ARM CPU, Blackwell GPU, ~128 GB **unified** LPDDR5X
- Memory bandwidth (spec): 273 GB/s, **shared with the CPU**
- BF16 dense compute (theoretical): ~125 TF/s
- FP8 / INT8: 2× / 1.5× of BF16 on Blackwell tensor cores
- FP4 native: yes, but PyTorch path still settling — not measured here

## Precisions covered

| Name | Activation | Weight | Output | Kernel |
|---|---|---|---|---|
| `bf16` | bfloat16 | bfloat16 | bfloat16 | `torch.matmul` |
| `fp16` | float16 | float16 | float16 | `torch.matmul` |
| `fp8_e4m3` | FP8 e4m3 | FP8 e4m3 | bfloat16 | `torch._scaled_mm` |
| `int8` | int8 | int8 | int32 | `torch._int_mm` |
| `w4a16` | bfloat16 | int4 (group 128) | bfloat16 | `aten._weight_int4pack_mm` |
| `w4a16_marlin` | bfloat16 | int4 (group 128) | bfloat16 | `vllm._custom_ops.marlin_gemm` |

The two W4A16 entries are the same precision but different kernels:
the PyTorch builtin is the generic CUDA path, Marlin is the
production tensor-core kernel that vLLM uses for `--quantization gptq`
and `--quantization awq`. Comparing them is half the point of this
experiment.

## What it measures

1. **Peak DRAM bandwidth** — 1 GiB device-to-device byte copy. Sets
   the diagonal (bandwidth-bound) roof. Same for every precision —
   bandwidth is a property of the hardware, not the dtype.
2. **Peak compute, per precision** — one square `8192³` matmul. Sets
   the horizontal compute roof for that precision.
3. **B-sweep, per (shape, precision)** — `(B, D) @ (D, F)` for
   `B ∈ {1 … 8192}`, `D = F ∈ {4096, 1024}`. Each point gets achieved
   FLOPs/s and arithmetic intensity (FLOPs ÷ bytes-touched) so it
   lands at a specific spot under the appropriate roof.

## Run

```bash
./run.sh                       # measure + PNG, runs in vLLM container
python3 ascii_roofline.py      # log-log ASCII plot, stdlib only on host
```

`run.sh` mounts this directory into the `vllm/vllm-openai:v0.19.1-cu130`
image (PyTorch 2.10 with Blackwell tensor-core kernels and the Marlin
binding) and writes results back to `results/`. Takes ~3–5 minutes.

If a precision's kernel isn't available, that sweep is skipped with an
error in the JSON and the rest of the run continues.

## Headline results

Peak compute and bandwidth measured on this machine:

| Quantity | Measured | Theoretical | Notes |
|---|---|---|---|
| BF16 / FP16 peak | 90–93 TF/s | 125 | ~74% — typical cuBLAS efficiency |
| **FP8 e4m3 peak** | **189–199 TF/s** | 250 | **2.1× over BF16** as expected |
| INT8 peak | 142–145 TF/s | 250 | underperforms; FP8 path is better-tuned |
| W4A16 (PyTorch builtin) peak | **8 TF/s** | (n/a) | **broken — not using tensor cores** |
| **W4A16 (Marlin) peak** | **~92 TF/s** | (n/a) | **matches BF16 roof** (dequantizes to bf16 internally) |
| DRAM bandwidth | 213 GB/s | 273 | ~78% of LPDDR5X spec |
| BF16 knee (intensity) | 409 F/B | — | `peak_flops / peak_bw` |
| BF16 knee (batch B for 4096²) | ~512 | — | tokens-per-step before compute-bound |

## Reading the plot

- **Diagonal** = bandwidth roof, slope = peak bytes/s. Same line for
  every precision because it's a hardware property.
- **Horizontal lines** = compute roof per precision. Higher = better.
- **Each sweep** climbs the diagonal at small `B` (bandwidth-bound),
  bends at the knee, then plateaus on the precision's compute roof.
- **The knee in intensity** is `peak_flops / peak_bw`. The knee in `B`
  for matmul `(B, D) @ (D, F)` falls out of solving
  `B·D·F / (B·D + D·F + B·F) = I_crit` — for D=F=4096 BF16 it's ~512.

## Per-precision findings

### FP8 — the headline win

FP8 hits **199 TF/s** at 8192³, exactly the 2× over BF16 that the spec
predicts. The B-sweep at D=F=4096 climbs cleanly:

```
B=128   →  91 TF/s  (where BF16 plateaus)
B=512   → 160 TF/s
B=8192  → 194 TF/s  (at peak)
```

For any batch size ≥ 64, FP8 is the best-throughput option on this
hardware. `vllm --quantization fp8` is the under-appreciated sweet
spot for serving: higher compute roof, smaller bytes-moved, much less
precision loss than INT4.

### Marlin (W4A16 production kernel) — matches BF16's roof

Marlin's compute ceiling is **~92 TF/s**, the same as BF16. This
surprised me at first but it's by design: Marlin dequantizes 4-bit
weights to bf16 *at the tensor-core boundary* and runs the actual
matmul in bf16. So the peak FLOPs/s is the bf16 ceiling.

What Marlin gives you is **4× less bytes-moved on weights**, which
shifts the entire bandwidth-bound region up:

```
B=1, D=F=4096    BF16     FP8      Marlin
                 0.23     0.43     1.54  TF/s    ← Marlin wins by ~7× over BF16
                 1.0      2.0      3.9   intensity (F/B)
```

The Marlin knee in intensity is `92 / 213 = 432 F/B`, which for a
4096² matmul corresponds to **B ≈ 373** — almost 30% lower than
BF16's `B ≈ 512` knee. So Marlin reaches its (lower) ceiling at
smaller batch sizes.

### The Marlin → FP8 crossover

At what batch size does FP8 catch up and pass Marlin?

| `B` | Marlin | FP8 | Winner |
|---|---|---|---|
| 1 | 1.54 | 0.43 | Marlin (3.6×) |
| 8 | 12.2 | 9.3 | Marlin |
| 32 | 35.6 | 28.3 | Marlin |
| 64 | 49.7 | 61.1 | **FP8 takes lead** |
| 128 | 66.1 | 90.7 | FP8 (1.4×) |
| 512 | 83.8 | 159.6 | FP8 (1.9×) |
| 8192 | 92.5 | 193.5 | FP8 (2.1×) |

Crossover at **B ≈ 32–64**. Below that, Marlin's bandwidth advantage
dominates. Above that, FP8's higher compute roof wins.

### INT8 — strictly worse than FP8 on this machine

143 TF/s peak, vs FP8's 199. cuBLAS INT8 GEMM on Blackwell isn't as
well-tuned as FP8 — INT8 was the Hopper headline, FP8/FP4 is what
NVIDIA pushes on Blackwell. Don't bother using INT8 here when FP8 is
available with the same bytes-moved.

### W4A16 PyTorch builtin — broken

`aten._weight_int4pack_mm` saturates at **8 TF/s regardless of B**.
It isn't using the tensor cores — it's a generic CUDA kernel that
predates the modern quant push. The graph shows this dramatically:
the W4A16-builtin line is a flat shelf 12× below BF16.

The takeaway is **not** "Q4 quant is slow." It's "Q4 quant in default
PyTorch is slow." Production paths (Marlin, llama.cpp's GGUF kernel,
TensorRT-LLM) bypass this and hit ~7× the throughput. Don't use this
path for anything real.

## Mapping to LLM serving (vLLM and Ollama)

For a transformer forward pass, **`B` (matmul batch dim) = N (tokens
processed in this step)**. Llama 3.1 8B is mostly matmuls of shape
`(N, 4096) @ (4096, *)`, so the sweep at D=F=4096 maps directly to
serving scenarios:

| Scenario | `N` per forward | Precision | Where on roofline | Achieved |
|---|---|---|---|---|
| Ollama, 1 user decoding (Q4_K_M ≈ Marlin) | 1 | W4A16 | far left, intensity ≈ 4 | ~1.5 TF/s |
| Ollama, 1 user prefilling 500 tokens | 500 | W4A16 | climbing past knee | ~85 TF/s |
| vLLM BF16, 1 user decoding | 1 | BF16 | far left, intensity ≈ 1 | ~0.25 TF/s |
| vLLM BF16, 10 concurrent decodes | 10 | BF16 | climbing | ~2.5 TF/s |
| vLLM BF16, 100 concurrent decodes | 100 | BF16 | climbing past midpoint | ~25 TF/s |
| vLLM BF16, mixed prefill+decode ≥ 512 | ≥512 | BF16 | at knee | ~85 TF/s |
| vLLM `--quantization gptq` (Marlin), 100 conc. | 100 | Marlin | climbing | ~60 TF/s |
| vLLM `--quantization fp8`, 100 concurrent | 100 | FP8 | climbing | ~80 TF/s |

To convert measured tokens/sec from `vllm-vs-ollama/RESULTS.md` to
TF/s for direct comparison: multiply by `2 × num_parameters` per
token. For Llama 3.1 8B that's `~16 GFLOPs/token`, so `30 tok/s ≈
0.48 TF/s`. The gap between this end-to-end TF/s and the
microbenchmark line at the same `B` is the **framework overhead**
(Python dispatch, KV-cache management, attention compute, sampling).

## Recommendations for serving on DGX Spark

| Workload | Best choice | Why |
|---|---|---|
| Single-user, latency-critical | Q4 (Ollama or vLLM `--quantization gptq`) | ~7× speedup at B=1 from bandwidth amortization |
| Small-batch (B ≤ 32) serving | Q4 (Marlin) | bandwidth-bound regime, intensity is king |
| Mid-batch (B = 64–512) | FP8 (`--quantization fp8`) | crossover already passed; FP8 has 2× the compute roof |
| Large-batch / max-throughput | FP8 | clean 2× over BF16 at peak |
| Don't bother | INT8 alone, PyTorch builtin W4A16 | strictly worse than FP8 / Marlin |

For your existing `vllm-vs-ollama` comparison, the roofline explains
the result: Ollama's single-user advantage is bandwidth amortization
via Q4, vLLM's concurrent advantage is climbing the diagonal via
continuous batching. **A vLLM run with `--quantization gptq` would
get the best of both** — production Q4 at low concurrency, BF16-level
peak at high concurrency.

## How DGX Spark compares to H100

The roofline shape is the same; the scale shifts.

| Quantity | DGX Spark (measured) | H100 SXM (spec) | Ratio |
|---|---|---|---|
| BF16 peak | 92 TF/s | ~750 TF/s | 8× |
| FP8 peak | 199 TF/s | ~1500 TF/s | 7.5× |
| Memory bandwidth | 213 GB/s | 3350 GB/s | 15× |
| BF16 knee in intensity | 409 F/B | ~295 F/B | 0.7× |
| BF16 knee in B (4096²) | 512 | ~370 | 0.7× |

H100's bandwidth advantage (15×) is bigger than its compute
advantage (8×), so the knee shifts slightly leftward there: H100
needs less batching to be compute-bound. For decode-heavy workloads
(intensity ≈ 1), H100 is ~13× faster per token than Spark — it's the
bandwidth gap doing the work, not the compute gap.

## What this experiment doesn't measure

- **Attention itself** — has its own roofline; reads the KV cache
  with a different intensity profile. We measured matmul only.
- **End-to-end serving overhead** — Python dispatch, kernel launch,
  scheduler, sampling. Real frameworks get 60–80% of these
  microbenchmark numbers.
- **Q4_K_M's exact format** — Ollama's actual GGUF kernel isn't on
  the plot. Marlin is the closest production proxy; the roofline
  position is similar (same bytes/FLOP ratio).
- **FP4** — Blackwell's headline precision (~370 TF/s theoretical).
  PyTorch path on this image isn't ready; would add ~2× over FP8 if
  the kernel ecosystem catches up.
- **Multi-GPU / NVLink** — DGX Spark is single-GPU; the inter-GPU
  bandwidth roof doesn't apply.

## Files

- `measure_roofline.py` — main benchmark; runs in container
- `marlin_kernel.py` — Marlin W4A16 setup (vLLM bindings)
- `plot_roofline.py` — generates `results/roofline.png`
- `ascii_roofline.py` — terminal-friendly log-log plot
- `probe_marlin.py` / `probe.sh` — diagnostics for Marlin API
- `run.sh` — one-shot wrapper that runs measurement + plotting
- `results/roofline.json` — raw numbers
- `results/roofline.png` — annotated roofline plot
