# Prefix caching

**Question**: how much does `--enable-prefix-caching` reduce TTFT
for requests that share a long prompt prefix?

**Theory**: vLLM stores KV cache in fixed-size paged blocks. When a
new request arrives whose prompt starts with a sequence that's
already in cache (e.g., same system prompt as a recent request), the
scheduler **reuses those KV blocks** instead of recomputing prefill.
The cost of prefilling the shared prefix is paid **once**, by the
first request. All subsequent requests skip it.

For multi-turn chat or RAG with a fixed system prompt + few-shot
examples, this is a free win: 4 KB of shared prefix → cache the
4 KB → never prefill it again until eviction.

## Workload

A fixed **4096-token system prompt** + a per-request short **64-token
query**. We fire 10 requests sequentially with the same system prompt
and different queries. Each request generates 32 output tokens.

We measure TTFT (time from request submission to first streamed token)
for each request. With caching:

- Request 1: cold (must prefill all 4096 system tokens + 64 query)
- Requests 2–10: warm (system prompt KV is in cache; only 64 query
  tokens need prefilling)

The cold/warm gap is the cost of prefilling the shared prefix.

## Predicted numbers

A 4096-token prefill at ~90 TF/s compute, with each Llama 3.1 8B
forward pass costing ~16 GFLOPs/token:

```
prefill cost = 4096 tokens × 16 GFLOPs/tok / 90 TFLOPs = ~0.73 s
```

So TTFT for cold should be ~750 ms + small constants (queue,
sampling). TTFT for warm requests should be the just-the-query
prefill: 64 × 16 GFLOPs / 90 TF/s ≈ 11 ms + constants, so probably
~80 ms in practice.

Expected speedup ratio for warm requests: **~10×**.

## Two configurations

| Mode | Flag | Default in v0.19.1 |
|---|---|---|
| OFF | `--no-enable-prefix-caching` | n/a (must disable explicitly) |
| ON | `--enable-prefix-caching` | yes |

With OFF, every request pays full prefill cost — TTFT is roughly
constant across requests 1–10 at ~750 ms.

With ON, request 1 pays full cost; requests 2–10 should drop by
~10×.

## Files

- `start-vllm-prefix-off.sh` — server with prefix caching disabled
- `start-vllm-prefix-on.sh` — server with prefix caching enabled
- `bench_prefix.py` — sends N sequential requests with shared prefix,
  records TTFT per request
- `plot_prefix.py` — bar chart of TTFT for the 10 requests, OFF vs ON
- `results/prefix_off.json`, `results/prefix_on.json` — raw runs

## Protocol

1. In tmux: `./start-vllm-prefix-off.sh`. Wait for ready.
2. `python3 bench_prefix.py --out results/prefix_off.json --label off`
3. Ctrl-C server. `./start-vllm-prefix-on.sh`. Wait.
4. `python3 bench_prefix.py --out results/prefix_on.json --label on`
5. `python3 plot_prefix.py`
