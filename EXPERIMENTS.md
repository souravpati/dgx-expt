# Experiment backlog — DGX Spark (GB10)

Planned experiments grounded in Chapters 4 (transformers) and 7
(inference) of the JAX scaling book. Each item lists the question
it answers, the method, the expected result, and rough effort.

## Done

- [`vllm-vs-ollama/`](vllm-vs-ollama/) — single vs concurrent serving comparison.
- [`roofline/`](roofline/) — matmul compute + memory ceilings across precisions. BF16 peak ~90 TF/s, DRAM peak 213 GB/s (245 GB/s effective for streaming reads, measured in #7-ish derivation).
- [`prefill-vs-decode/`](prefill-vs-decode/) — attention arithmetic intensity. Prefill climbs to 90 TF/s by T=16k; decode pinned at I = N/K = 4 F/B regardless of S or B.

## Backlog

| # | Question | Method | Expected | Effort |
|---|---|---|---|---|
| 1 | Does the Chapter-7 step-time formula predict vLLM decode latency? | Sweep B ∈ {1, 4, 16, 64, 128, 256} at S ∈ {2k, 8k, 32k}. Measure ITL. Compare to `(B·KV + params)/W_HBM`. | Within 10–20% across the matrix. Discrepancy reveals scheduler / Python / sampling overhead. | 1 evening |
| 2a | Does prefix caching give near-free TTFT on shared prompts? | 4k-token shared system prompt, 10 requests with different suffixes, `--enable-prefix-caching` on/off. Measure TTFT for request #2+. | Cache-hit TTFT ≈ TTFT of suffix only. Speedup = (prefix + suffix) / suffix. | 1 hour |
| 2b | How much does prefix caching help in realistic multi-turn? | Simulate a 5-turn conversation; growing prefix each turn. Compare aggregate latency w/ vs w/o caching. | Turn N+1 prefill ~free for prior turns' tokens. | 2 hours |
| 3 | Does speculative decoding give 2–3× single-user tokens/sec? | vLLM target=Llama-3.1-8B, draft=Llama-3.2-1B. Sweep `num_speculative_tokens` ∈ {1..6}. Measure tokens/sec, acceptance rate, GPU mem. Test on prose (XSum-like) and code (HumanEval-like). | Prose: peak at K=3–4, ~2× speedup, ~70% acceptance. Code: peak at K=5–6, ~2.5×, ~85% acceptance. | 1 day |
| 4 | What problem does disaggregation solve, measured on one GPU? | vLLM with `--enable-chunked-prefill` on vs off. One steady decode-only client; periodic prefill-heavy client (8k-token prompts). Measure ITL distribution for the decode client when a prefill lands. | Chunked off: ITL spikes O(seconds), p99 blows up. Chunked on (chunk=512): ITL flat ~50 ms throughout. The gap is the latency win disaggregation gives across machines, here as a within-GPU scheduler knob. | 1 day |
| 5 | Validate B_crit ≈ 240–400 rule on real model | vLLM at B ∈ {1, 4, 16, 64, 240, 512, 1024}. Measure aggregate decode tokens/sec. Find the knee. | Knee 200–500 (matmul knee was 409 in `roofline/`; vLLM overhead and KV reads lower it). | 0.5 day |
| 6 | Does decode throughput scale linearly with GQA ratio N/K? | Extend `prefill-vs-decode/measure_attention.py`: sweep N_KV_HEADS ∈ {1, 2, 4, 8, 16, 32} at fixed N=32, decode at S=4096, B=8. Plot achieved TF/s vs N/K. | Linear scaling: AI = N/K so TF/s = 245 GB/s × N/K. MHA (N/K=1) → 0.25 TF/s; MQA (N/K=32) → 7.8 TF/s. Cap = DRAM bandwidth. | 1 hour |
| 7 | Step latency vs batch size — predicted vs measured | Standalone artifact: compute `t_step(B, S)` from Chapter-7 formula using our measured numbers (P=16 GB, W_HBM=245 GB/s, W_FLOPs=90 TF/s, KV/tok=4 KiB). Generate predicted curve. Overlay vLLM measurements from #1 once done. | Two curves per S ∈ {2k, 8k, 32k}: ms/step rises with B; ms/token *falls* until B_crit ≈ 367 then flattens. Knee location is the key finding. | 0.5 day (prediction now; measurement is part of #1) |
| 8 | Sharding strategies on Spark | **Out of scope.** Needs ≥2 GPUs. MIG unavailable on GB10; MPS/stream-partitioning measures contention not communication. Read Ch. 9–10 and work problems on paper; revisit with multi-GPU access. | n/a (theoretical only) | 0 |

## Suggested order

1. **#1** (step-time validation) — generates the measured half of #7 for free; closes the microbench → analytic-model → real-serving loop.
2. **#7** prediction first, then drop #1 measurements on top.
3. **#6** (GQA sweep) — small enough to slot in anywhere.
4. **#4** (chunked prefill = disaggregation-equivalent) — most novel single-GPU result.
5. **#3** (speculative decoding) — biggest potential single-user win.
6. **#2a / #2b** (prefix caching) — quick wins.
7. **#5** (B_crit confirmation) — optional, mostly a confirmation.
8. **#8** (sharding) — skip unless multi-GPU appears.
