# Speculative decoding

**Question**: does speculative decoding give the 2–3× single-user
tokens/sec lift Chapter 7 predicts for bandwidth-bound decode on GB10?

**Theory**: decode on Spark is hard memory-bound (we measured 13 tok/s
at B=1 in `../step-time/`). The kernel waits on DRAM for KV+params;
its compute units are idle. Spec decoding fills those idle FLOPs with
draft-token *verification* — a small drafter proposes K tokens, the
big model scores all K in one batched forward pass at the cost of
one normal step. Accepted prefix-length determines the win.

Predicted theoretical max speedup (perfect acceptance, K spec tokens):
```
speedup_max = K
```
Realistic speedup with acceptance rate α per position (compounding):
```
E[accepted] = (1 - α^(K+1)) / (1 - α)
speedup = E[accepted] / (1 + overhead)
```
For α = 0.7 (prose, K=4): E[accepted] ≈ 2.7 → ~2.5× after overhead.
For α = 0.85 (code, K=5): E[accepted] ≈ 4.0 → ~3.5×.

## Setup

| Role | Model | Size |
|---|---|---|
| Target | meta-llama/Meta-Llama-3.1-8B-Instruct | 16 GB bf16 |
| Draft | meta-llama/Llama-3.2-1B-Instruct | ~2.5 GB bf16 |

Same family (Llama 3) → tokenizers compatible (required by vLLM).
Draft is 8× smaller, so it's ~8× cheaper to evaluate per token — and
the bandwidth amortization win works because all K spec-tokens share
the same KV-cache read from the target.

## Sweep

| K (num_speculative_tokens) | Why this value |
|---:|---|
| 0 | baseline (no spec) |
| 3 | conservative — chapter predicts this is near-optimal for prose |
| 5 | aggressive — chapter predicts near-optimal for code |

Three vLLM server restarts (1.5–2 min each). Bench is ~30 s per run.

## Workload

Two prompt populations × 8 runs each at temperature=0 (deterministic):

1. **Prose** — narrative continuation prompts. Predicted acceptance
   rate ~70%.
2. **Code** — function completion. Predicted acceptance ~85% (code
   has more local structure → drafter agrees more often).

For each (K, workload) we measure:
- Wall-clock tokens/sec
- Average tokens accepted per spec step (from vLLM `/metrics`)
- Acceptance rate = accepted / draft proposals

## Files

- `start-vllm-spec.sh K` — parameterized server launcher
- `bench_spec.py` — workload runner; queries `/metrics`
- `plot_spec.py` — throughput vs K, prose vs code
- `run-bg.sh` — backgrounded runner
- `results/spec_K{0,3,5}.json` — raw per-config results

## Protocol

1. Start baseline (no spec):
   ```
   ./start-vllm-spec.sh 0
   ```
   Wait for `Uvicorn running`, then `python3 bench_spec.py --out results/spec_K0.json --K 0`.
2. Ctrl-C, `./start-vllm-spec.sh 3`. Wait. Run `... --out results/spec_K3.json --K 3`.
3. Ctrl-C, `./start-vllm-spec.sh 5`. Wait. Run `... --out results/spec_K5.json --K 5`.
4. `python3 plot_spec.py` and analyze.
