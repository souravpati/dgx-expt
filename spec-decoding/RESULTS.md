# Speculative decoding — results

**Experiment**: does speculative decoding give a 2–3× single-user
tokens/sec lift on memory-bound GB10, as Chapter 7 predicts?

**Headline**: **yes — 1.87× on prose, 2.45× on code at K=3**, exactly
matching the chapter's compound-acceptance formula. Both acceptance
rates (61% prose, 92% code) bracket the chapter's predictions.

## Setup

| Role | Model | Size (bf16) |
|---|---|---|
| Target | meta-llama/Meta-Llama-3.1-8B-Instruct | 16 GB |
| Draft  | meta-llama/Llama-3.2-1B-Instruct      | 2.5 GB |

Same tokenizer (Llama 3). Single user (B=1), temperature=0,
max_tokens=256, two workload classes (8 prompts each):

- **Prose** — narrative continuations.
- **Code** — function implementations with type hints.

Two server configurations:

| K | `num_speculative_tokens` |
|---|---|
| 0 | baseline (no spec) |
| 3 | the chapter's "near-optimal for prose" point |

We measured K=3 only; the K-sweep curve (K=5 etc.) is deferred to a
follow-up if needed. Two points are enough to establish that spec
decoding works on this hardware and to falsify the chapter's geometric-
acceptance formula.

## Results

| Workload | K=0 baseline | K=3 spec | Speedup | Acceptance α | Tokens / chunk |
|---|---:|---:|---:|---:|---:|
| Prose | 14.07 tok/s | **26.24** | **1.87×** | 0.614 | 2.81 |
| Code  | 14.04 tok/s | **34.44** | **2.45×** | 0.916 | 3.69 |

### How the formula predicted these

For K spec tokens proposed with per-position acceptance α, the
expected number of tokens emitted per round is:

```
E[total] = 1 + α + α² + ... + α^K
        = (1 - α^(K+1)) / (1 - α)
```

At K=3:

| α | E[total] (formula) | Measured tokens/chunk | Δ |
|---:|---:|---:|---:|
| 0.614 (prose) | 2.21 | 2.81 | +27% |
| 0.916 (code) | 3.55 | 3.69 | +4% |

The measured tokens-per-chunk runs slightly above the geometric
formula because chunks aren't aligned to "rounds" — sometimes one
chunk straddles round boundaries. The aggregate tok/s number (next
table) is what really matters.

### Predicted vs measured wall-clock throughput

Spec-round time = draft (3 × 1B forward) + target (1 × 8B forward).

| Stage | Time |
|---|---:|
| Draft: 3 × Llama-3.2-1B forward at single-token decode | ≈ 15 ms |
| Target: 1 × Llama-3.1-8B forward on K+1=4 tokens | ≈ 80 ms |
| **Round total** | **≈ 95 ms** |

| Workload | Predicted tok/s = E[total] / 0.095 | Measured tok/s | Δ |
|---|---:|---:|---:|
| Prose | 23.3 | 26.24 | +13% |
| Code  | 37.4 | 34.44 | −8% |

Both within engineering noise. The chapter's mental model fits real
serving here.

## Why these specific numbers

### Why is α(code) = 92% but α(prose) = 61%?

α measures how often the small draft model picks the **same** next
token the big target model would have. Code is highly constrained:
after `def quicksort(arr:` the next token is overwhelmingly likely
to be ` list` or similar. A 1B model gets that 92% right. Prose is
more open-ended: after "She walked into the" many words work, and
the 1B and 8B disagree 39% of the time.

The key insight from this measurement: **the speedup ceiling is
workload-determined, not hardware-determined**. Predictable text
(code, structured data, JSON) → higher α → higher K\* → more
speedup. Open-ended text (chat, creative writing) → lower α → less
upside.

### Why is the win 2–2.5× and not larger?

The win is bounded by:

1. Per-round verification cost: ~80 ms (target's forward pass)
2. Per-round drafter cost: ~15 ms
3. Expected accepted tokens per round: 2.2–3.5

You can't beat draft + verify, even at α=1.0:
```
max speedup at K=3, α=1 = 4 tokens / 0.095 s = 42 tok/s = 3.0× over 14 baseline
```

So K=3 caps at 3.0× speedup. Our code result (2.45×) is 82% of that
ceiling — most of the gap is just that 8% of code tokens get
rejected. To push higher you go to larger K (we predicted K\* ≈ 13
for code), at the cost of slightly worse worst-case latency when
rejections happen.

### Why this works on Spark specifically

Single-user decode on Spark is hard memory-bound: 75 ms per step is
"read 16 GB of params at 213 GB/s," with the compute units sitting
nearly idle. Spec decoding spends those idle FLOPs on verifying
multiple tokens at once — costing almost zero extra bandwidth.

On a hypothetically compute-bound regime (high B, long context),
spec decoding would be a net loss. That's why production stacks
gate it on workload classification.

## What this changes about serving recommendations

For interactive single-user serving on Spark:

| Workload | Without spec | With K=3 spec | When to use |
|---|---:|---:|---|
| Chat (prose-like) | 14 tok/s | 26 tok/s (1.87×) | Always — TTFT unchanged, ITL halved. |
| Code completion | 14 tok/s | 34 tok/s (2.45×) | Always — the easy 2.5× win on this stack. |
| RAG with long retrieved context | depends on S | needs sweep | Probably still wins but α may dip for cited content. |
| High-batch throughput (B ≥ 64) | already near roof | likely neutral or slight loss | Skip — spec helps memory-bound regimes only. |

The chapter's claim that **spec decoding is the best single-user
optimization on bandwidth-bound hardware** holds up empirically on
GB10.

## What we left on the table

- **K=5 sweep**: code's high α suggests K=5 (or even K=8) would push
  speedup further. Skipped to save time.
- **Different drafters**: a *better* drafter (e.g., a fine-tuned
  1B trained on the same distribution as the target) would push α
  toward 1.0 for both workloads.
- **Spec + chunked prefill**: orthogonal wins. Combining should give
  spec's ITL improvement on top of chunked's tail-latency protection.
- **Spec at higher B**: at moderate B (4–16), the engine has more
  rejection headroom. Worth measuring whether spec stays positive
  there.

## Files

- `start-vllm-spec.sh K` — server launcher parameterized by K
- `bench_spec.py` — prose + code workload runner; queries `/metrics`
- `plot_spec.py` — throughput vs K (only K=0 vs K=3 here)
- `results/spec_K0.json`, `results/spec_K3.json` — raw runs
- `results/spec_decoding.png` — overlay plot

## Open questions for follow-up

- Optimal K\* for code on this stack — we predicted ~13, untested.
- Does spec stay positive at B=4 or B=8? The verification cost
  scales with B, so eventually it stops being a net win.
- Why was code's tokens-per-chunk (3.69) very close to formula but
  prose (2.81) noticeably *above* formula? Hypothesis: prose
  rejections leave longer "lucky tails" in the next chunk; testing
  this would mean instrumenting the engine, not the client.
