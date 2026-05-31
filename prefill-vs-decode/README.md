# Attention arithmetic intensity: prefill vs decode

Microbenchmark for the **attention block** on DGX Spark (GB10),
verifying the Chapter-4 claims from the JAX scaling book
([transformers chapter](https://jax-ml.github.io/scaling-book/transformers/)).
Companion to [`../roofline/`](../roofline/) — which measured matmuls
only and explicitly left attention as future work.

## What the chapter claims

For one self-attention layer with N query heads, K KV heads (GQA),
head dim H, batch B:

- **Prefill** (T_q = T_kv = T, causal): FLOPs grow as T², bytes as T.
  Arithmetic intensity is **O(T)** — climbs the roofline diagonal
  with sequence length, eventually saturating compute.
- **Decode** (T_q = 1, T_kv = S, KV cache read from DRAM): FLOPs and
  bytes both grow as B·S. Intensity is **N/K** (the GQA group size),
  a constant **independent of S and B** — deep in the bandwidth-bound
  region.

## Llama 3.1 8B shape (per layer)

| | Value |
|---|---|
| Query heads (N) | 32 |
| KV heads (K) | 8 |
| Head dim (H) | 128 |
| Model dim (D = N·H) | 4096 |
| Predicted decode intensity (N/K) | **4 F/B** |
| BF16 roofline knee (from `../roofline/`) | **409 F/B** |

The decode prediction sits 100× below the knee → memory-bound by a
large margin, batch size won't change this.

## What we measure

`torch.nn.functional.scaled_dot_product_attention` in bf16 on a single
attention layer, dispatched to flash-attn on Blackwell.

| Sweep | Knob | Held fixed | Why |
|---|---|---|---|
| Prefill T | T ∈ {64 … 16384} | B=1, causal, T_q=T_kv=T | Show I = O(T) |
| Decode S | S ∈ {128 … 32768} | B=1, T_q=1 | Show I is flat in S |
| Decode B | B ∈ {1 … 256} | S=4096, T_q=1 | Show I is flat in B |

For each point we record achieved TFLOPs/s and arithmetic intensity
(FLOPs ÷ bytes-moved). Bytes-moved counts Q + K + V + O traffic where
K/V use the **GQA-shaped** cache (K=8 heads, not N=32) — that's the
true DRAM footprint.

## Run

```bash
./run.sh
```

Mounts this directory into the vLLM container (same image as
`../roofline/`) and writes `results/attention.json`.

## Expected shape of results

If the chapter is right, on the same roofline as `../roofline/`:

- **Prefill points** should climb diagonally at small T, then bend
  and approach the BF16 compute roof (~92 TF/s) around T ≥ 4096.
  Intensity should roughly double when T doubles.
- **Decode-over-S points** should form a flat horizontal line at
  I ≈ 4 F/B and ≈ 2 TF/s (4 F/B × 213 GB/s ÷ 1e12 × 2 = 1.7 TF/s),
  regardless of how big S gets.
- **Decode-over-B points** should also form a flat horizontal line
  at the same I and TF/s — batching does not help intensity when
  each request owns its KV cache.

The decode-vs-batch result is the most surprising one to many people
and the one most worth plotting: in single-stream serving on this
hardware, attention's per-token cost is bandwidth times N/K. The
only ways to move that point right on the roofline are GQA-ratio
changes (architecture), shared KV (e.g., prefix caching, beam
search), or paged batching at the same key-set.

## Files

- `measure_attention.py` — runs the three sweeps; writes JSON
- `run.sh` — docker wrapper (vllm/vllm-openai:v0.19.1-cu130)
- `results/attention.json` — raw numbers (after first run)

## Next steps (not yet built)

- Plot points on top of `../roofline/results/roofline.png` for the
  full picture.
- Verify the chapter's attention-vs-MLP crossover at T > 8D by
  combining these numbers with the matmul sweep from `../roofline/`.
- Add a "shared-prefix" decode sweep (same K, V across batch) to
  show intensity scaling with B in the way the chapter describes
  for that special case.
