# Empirical roofline on DGX Spark (GB10)

Measures the actual compute and memory ceilings of this machine and
plots matmul shapes against them across five precisions, so we can see
where each precision's roof sits and how quantization moves a workload
relative to the bandwidth wall.

## Precisions covered

| Name | Activation | Weight | Output | Kernel |
|---|---|---|---|---|
| `bf16` | bfloat16 | bfloat16 | bfloat16 | `torch.matmul` |
| `fp16` | float16 | float16 | float16 | `torch.matmul` |
| `fp8_e4m3` | FP8 e4m3 | FP8 e4m3 | bfloat16 | `torch._scaled_mm` |
| `int8` | int8 | int8 | int32 | `torch._int_mm` |
| `w4a16` | bfloat16 | int4 (group 128) | bfloat16 | `aten._weight_int4pack_mm` |

The `w4a16` path matches what serving frameworks call "Q4 weight-only"
quantization — what Ollama (Q4_K_M) and vLLM (`--quantization awq`/`gptq`)
ultimately do for batch=1 decode.

FP4 (Blackwell-native) is not included — the PyTorch path is still
settling and the integration cost outweighs the marginal insight.

## What it measures

1. **Peak DRAM bandwidth** — 1 GiB device-to-device byte copy. Sets
   the diagonal (BW-bound) roof. Same for every precision because
   bandwidth is a property of the hardware, not the dtype.
2. **Peak compute, per precision** — one square matmul (`8192³`).
   Sets the horizontal roof for that precision.
3. **B-sweep, per (shape, precision)** — `(B, D) @ (D, F)` for
   `B ∈ {1 … 8192}`, `D = F ∈ {4096, 1024}`. Each point gets achieved
   FLOPs/s and arithmetic intensity (FLOPs ÷ bytes-touched), placing
   it under the appropriate roof.

## Run

```bash
./run.sh                                 # measure + PNG, in vLLM container
python3 ascii_roofline.py                # ASCII view, host stdlib only
```

`run.sh` mounts this directory into the `vllm/vllm-openai:v0.19.1-cu130`
image (which has PyTorch 2.10 with Blackwell tensor-core kernels) and
writes results back to `results/`. Takes ~2–3 minutes.

If a precision's kernel isn't available (e.g. layout-incompatible op
on this PyTorch build), that sweep is skipped with an error message in
the JSON; everything else still runs.

## Reading the plot

- **Diagonal** = bandwidth roof. Same for all precisions.
- **Horizontal lines** = compute roof per precision. Lower precision
  → higher roof (more ops per cycle on the tensor cores).
- **Each sweep** climbs the diagonal until it hits its precision's
  roof. The intensity at that knee is `peak_flops / peak_bw` for that
  precision.
- **Lower precision shifts a given `(B, D, F)` point rightward** on
  the F/B axis, because the bytes-moved denominator shrinks. So at a
  fixed batch size `B`, a quantized matmul has higher intensity AND
  more available compute — both effects help.

## Why this matters for your serving experiment

- **Decode at batch=1** sits at intensity ≈ 1 in BF16. Quantizing to
  W4A16 multiplies intensity by ~3× (weights drop from 2 to 0.5 bytes)
  *and* doesn't lower the compute roof, so the achievable throughput
  goes up roughly linearly with the bytes-moved reduction. That's the
  Ollama Q4 advantage.
- **The crossover** between "Q4 single-user" and "BF16 batched" lands
  where the BF16 sweep climbing rightward overtakes the W4A16 sweep at
  its small-B point. The two sweeps on the same plot make this visible.
- **Past the knee**, lower precision still wins on absolute throughput
  (higher roof), but the ratio is bounded by the tensor-core ratio
  (~2× per precision step), not the bytes-moved ratio.
