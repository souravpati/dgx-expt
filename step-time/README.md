# Step-time formula validation (Experiment #1)

Does Chapter 7's decode-step-time formula predict actual vLLM
inter-token latency on DGX Spark (GB10)? This is the close-the-loop
experiment: microbench numbers from `../roofline/` and
`../prefill-vs-decode/` feed an analytic prediction, which we then
compare against the real serving stack.

## The formula

For one decode step on a batch of B requests, each with KV cache
depth S, model with parameters P, bandwidth W_HBM, compute W_FLOPs:

```
t_step = t_attn + t_mlp
t_attn = B · KV_per_token · S / W_HBM            (always BW-bound)
t_mlp  = max( P / W_HBM,  2·B·P / W_FLOPs )      (BW-bound below B_crit, compute-bound above)
```

Per-token latency (ms/token from the *user's* perspective):
```
ms_per_token = t_step / B
```

This is the latency a single user sees between consecutive streamed
tokens — independent of TTFT and aggregated across all concurrent
users that share the batch.

## Numbers we plug in (Llama 3.1 8B on GB10)

| Quantity | Value | Source |
|---|---|---|
| P (params, bf16) | 16 GB | model card |
| KV per token | 128 KiB | 4 KiB/layer × 32 layers (engine reads all layers' KV every step) |
| W_HBM | 213 GB/s | clean 1-GiB d2d copy from `../roofline/`; SDPA attention reaches 225–232 GB/s in practice (with some L2 contribution) |
| W_FLOPs (bf16) | 90 TF/s | `../roofline/` peak |
| B_crit | 367 | W_FLOPs / W_HBM in bf16 |

## What we sweep

| Axis | Values |
|---|---|
| Batch (concurrency) | 1, 4, 16, 64, 128, 256 |
| Context length S | 2048, 8192, 32768 |

For each (B, S) we record median inter-token latency (ITL), measured
from streaming chunks, skipping the first 5 tokens to avoid prefill
artifacts.

## Files

- `predict_step_time.py` — analytic curves, GPU-free.
- `bench_step_time.py` — async streaming benchmark client.
- `plot_step_time.py` — overlays measured vs predicted.
- `start-vllm.sh` — convenience wrapper for the vLLM server.
- `run.sh` — assumes server is up; runs bench + plot.

## Expected outcome

- **Small B (< B_crit)**: step time grows roughly linearly with B
  because attention dominates and is BW-bound on the KV cache. Per-token
  latency falls — batching amortizes the param-load.
- **B near B_crit (~370)**: knee in the curve; MLP starts hitting the
  compute roof.
- **Above B_crit**: step time grows faster (compute-bound); per-token
  latency flattens.

Discrepancy between measured and predicted reveals overhead the
formula doesn't model: Python dispatch, scheduler, sampling, NCCL,
kernel-launch latency. If we're within 10–20%, the chapter is honest
on this hardware.
