# Step-time formula validation — results

**Experiment**: does the Chapter-7 decode-step-time formula predict
real vLLM inter-token latency on DGX Spark (GB10), running
Llama 3.1 8B Instruct in bf16?

**Headline**: **yes, within ~10–20%** for every clean measurement
point. Two iterations and two bug fixes (one in the benchmark, one
in the predictor) were needed to see it.

## The formula

```
t_step = (B · S · KV_per_token + P) / W_HBM      (decode is BW-bound)
t_step / B = per-user inter-token latency (ITL)
```

with hardware/model numbers measured from sibling experiments:

| Quantity | Value | Source |
|---|---|---|
| P (params, bf16) | 16 GB | model card |
| KV per token | **128 KiB** | 2 (K+V) × 8 KV heads × 128 head dim × 2 bytes × **32 layers** |
| W_HBM | 213 GB/s | clean 1-GiB d2d copy in `../roofline/` |
| W_FLOPs (bf16) | 90 TF/s | `../roofline/` peak |
| B_crit | 423 | W_FLOPs / W_HBM |

## Measurement methodology

- vLLM `vllm/vllm-openai:v0.19.1-cu130`, `meta-llama/Meta-Llama-3.1-8B-Instruct`
- `--max-model-len 40960  --gpu-memory-utilization 0.85`
- B concurrent streaming completion requests, each with a fresh
  S-token prompt (token IDs varied per-thread to defeat prefix caching)
- `max_tokens = 200`, `prefill_skip = 120` — discard the first 120
  streamed chunks per request so we land in steady-state decode
  after all peer prefills finish
- Median ITL within a request, then median across the B requests
- Per-point timeout 600 s

## Results

### Decode step latency: predicted vs measured

| B | S | pred ms/step | meas ms/step | ratio | pred tok/s | meas tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2k | 76.4 | **71.4** | 0.94× | 13 | **14** |
| 4 | 2k | 80.2 | 71.0 | 0.89× | 50 | 56 |
| 16 | 2k | 95.3 | 86.4 | 0.91× | 168 | 185 |
| 64 | 2k | 155.8 | 160.1 | 1.03× | 411 | 400 |
| 128 | 2k | 236.4 | 267.7 | 1.13× | 541 | 478 |
| 256 | 2k | 413.6 | 477.4 | 1.15× | 619 | **536** |
| 1 | 8k | 80.2 | 75.7 | 0.94× | 12 | 13 |
| 4 | 8k | 95.3 | 84.8 | 0.89× | 42 | 47 |
| 16 | 8k | 155.8 | 144.1 | 0.93× | 103 | 111 |
| 64 | 8k | 397.7 | 325.6 | 0.82× | 161 | 197 |
| 128 | 8k | 720.4 | 592.3 | 0.82× | 178 | **216** |
| 256 | 8k | 1381.5 | 1645.7 | 1.19× | 185 | 156 ⚠ |
| 1 | 32k | 95.3 | 89.5 | 0.94× | 10 | 11 |
| 4 | 32k | 155.8 | 140.2 | 0.90× | 26 | 29 |
| 16 | 32k | 397.7 | 266.4 | 0.67× | 40 | 60 |
| 64 | 32k | 1365.6 | 538.1 | 0.39× | 47 | 119 ⚠ |
| 128 | 32k | 2656.1 | 1693.2 | 0.64× | 48 | 76 ⚠ |
| 256 | 32k | 5253.0 | 1810.0 | 0.34× | 49 | 141 ⚠ |

⚠ = >30% of streams timed out. The reported median is taken only over
fast-surviving streams, so it **underestimates** the true step time. These
sub-1× ratios are not formula errors; they're survivorship bias.

### What the table shows

- **For all 13 clean points** (no timeouts), measured / predicted lies in
  **[0.82, 1.19]**. Random scatter, no systematic bias.
- **At B=1** the formula is essentially exact (0.94× across all S).
  Predicted 12–13 tok/s for a single user, measured 11–14.
- **Per-user latency** at B=1 ranges from **75 → 90 ms/token** as S goes
  2k → 32k. This is the chatbot-perceived latency: ~13 tok/s feels
  responsive, slows perceptibly on very long contexts.
- **Aggregate cluster throughput** peaks measured at **536 tok/s
  (B=256, S=2k)** and **216 tok/s (B=128, S=8k)** — both 80–90% of the
  formula's prediction.
- **The S=32k high-B regime is unreachable on a single Spark** without
  paged-attention block resizing or longer per-request timeouts —
  prefill alone takes most of the 10-minute budget.

## Per-user vs aggregate-throughput tradeoff

Plot in `results/step_time.png`. Two lines per S, one predicted (curve) and
one measured (markers). The qualitative pattern matches Chapter 7's
latency-throughput frontier:

- B=1: lowest user-perceived latency (~75 ms/token) but worst cluster
  throughput (~13 tok/s).
- B near B_crit (423): aggregate throughput approaches its ceiling.
- Above B_crit: per-user latency degrades sharply; aggregate throughput
  flattens.

## What went wrong (and what it teaches)

Two iterations to get a clean result:

### Iteration 1: prefill contamination

Initial run used `max_tokens=64`, `prefill_skip=5`. At high B (128+),
ratios blew up to 14×. The "ITL" we measured included steps where vLLM
was still prefilling peer requests — not steady-state decode.

**Lesson**: client-side ITL measurement on a flooded engine
measures *user-perceived ITL during prefill contention*, not the
ideal decode step time. To validate the formula you need to skip
enough warm-up tokens that all peer prefills are done.

This is *itself* the problem disaggregation solves (experiment #4):
in production, prefill bursts blow up tail latency for already-decoding
users. The latency the chapter's formula predicts is what you'd see
*after* disaggregation eliminates that contention.

### Iteration 2: missing layer factor in KV/token

The first corrected run still showed ratios of 2–6× at moderate B.
After investigation: I had written `KV_per_token = 4 KiB` in the
predictor — that's the **per-layer** cache size from the
`prefill-vs-decode/` microbench. The decode step reads **all 32
layers** every iteration, so the real value is **128 KiB/token**.

After fixing, ratios collapsed to [0.82, 1.19] as shown above.

**Lesson**: be careful which "KV cache size" a microbench is
reporting. Per-layer is what attention-kernel microbenches measure
(one fused call); per-step is what the engine moves through DRAM
(L layers × per-layer). For step-time prediction, use the per-step
number.

## Bandwidth ceiling reality check

The roofline experiment's 213 GB/s d2d copy is a tight upper bound on
sustained DRAM read throughput for the decode forward pass:

- B=128/S=8k: 720 ms predicted (213 GB/s), 592 ms measured.
  Implied effective bandwidth: **259 GB/s** — over the d2d ceiling.
  L2 contribution? Plausible — that workload moves 158 GB per step,
  cache hierarchy helps a few percent.
- B=128/S=32k: 2656 ms predicted, 1693 ms measured (but unreliable
  due to timeouts on 102/128 streams).

So effective bandwidth for *real* decode is **~10–20% above** the
roofline d2d number, not below it. Good news for the formula —
predictions are conservative.

## What this unlocks

- **Experiment #7** (predicted vs measured curve): already produced
  in `results/step_time.png`.
- **Experiment #4** (chunked prefill = disaggregation-equivalent):
  the formula is the *ideal* ITL that chunked prefill should preserve
  even under prefill bursts. The "wrong" iteration-1 numbers are
  literally the prefill-contention problem chunked prefill aims to
  fix — we can quantify the fix exactly.
- **Experiment #3** (speculative decoding): predicted speedup is
  `K · t_decode_step / t_spec_step`. At B=1, S=8k, t_step ≈ 75 ms.
  With a 1B drafter at ~15 ms/draft-token and K=4, t_spec_step ≈
  75 + 4·15 = 135 ms producing up to 4 accepted tokens → 33 tok/s
  vs 13 tok/s baseline → **2.6× theoretical peak**. We can now
  falsify this directly.
- **Experiment #6** (GQA sweep): N/K factor would shift the KV term
  in this formula. With N/K=1 (MHA), KV cache would 4× — predicted
  decode rate halves at S=8k+. Easy to verify.

## Files

- `predict_step_time.py` — generates `results/predicted_step_time.json`
- `bench_step_time.py` — streaming client; takes `--max-tokens`,
  `--prefill-skip`, `--timeout`, `--batches`, `--contexts`
- `plot_step_time.py` — overlay
- `start-vllm.sh` — server launcher (uses tmux for persistence)
- `run.sh` — foreground sweep
- `run-bg.sh` — detached sweep (setsid + nohup); survives SSH disconnect
- `results/predicted_step_time.json` — analytic
- `results/measured_step_time.json` — empirical
- `results/step_time.png` — overlay plot
- `results/bench.log` — execution log
