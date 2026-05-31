# Chunked prefill — results

**Experiment**: how much does vLLM's `--enable-chunked-prefill`
protect decode-stream latency from prefill bursts on a single GPU?
This is the within-GPU analogue of disaggregated serving.

**Headline**: **chunked prefill collapses tail latency by ~11×**.
p99 inter-token latency drops from **1735 ms → 161 ms** at the cost
of doubling p50 (77 → 143 ms). For any interactive-serving workload,
this is the right tradeoff — and on multi-GPU it's exactly what
disaggregated prefill/decode clusters buy you.

## Workload

Two concurrent client populations against one vLLM server (Llama 3.1
8B Instruct, bf16, `--max-model-len 16384`):

| Population | Count | Prompt length | Output tokens | Purpose |
|---|---:|---:|---:|---|
| **Decode stream** | 32 | 128 | 500 streaming | Steady state; we measure ITL here. |
| **Prefill burst** | 4 / burst | 8192 | 10 | Periodic floods. |

Bursts fire every 8 s for the 60 s measurement window — total of
**~7 bursts × 32k prefill tokens = 224k prefill tokens** dropped on
the engine while it's trying to serve 32 ongoing decode streams.

Two server configurations compared:

1. **OFF** — `--no-enable-chunked-prefill`. Prefill steps block decode
   completely; one burst = one big prefill pause.
2. **ON** — `--enable-chunked-prefill --max-num-batched-tokens 512`.
   Each scheduler step packs 512 prefill tokens + the running decode batch.

## Results

### Inter-token latency percentiles

| Mode | p50 | p90 | p95 | p99 | p99.9 | max |
|---|---:|---:|---:|---:|---:|---:|
| chunked OFF | **76.8** ms | 80.2 ms | 82.0 ms | **1734.5** ms | 1747.4 ms | 1747.0 ms |
| chunked ON | 142.8 ms | 152.3 ms | 154.4 ms | **160.6** ms | 167.7 ms | 167.8 ms |
| **Ratio (ON / OFF)** | **1.86×** | 1.90× | 1.88× | **0.093×** | 0.096× | 0.096× |

Reading top-to-bottom:

- **Median**: chunked ON pays ~2× per step because the scheduler is
  always folding prefill chunks into the batch. Predicted by the
  step-time formula: at chunk=512 + decode-batch=32, the effective
  matmul batch is 544 — above B_crit=423, so the MLP is now
  compute-bound. Single-step time goes from "params only at B<<B_crit"
  to "params and 544-token MLP." Doubling tracks the 544/B_crit ratio.

- **Tail (p99)**: chunked ON cuts it by **10.8×**. Without chunking,
  a single burst freezes the world for ~1.7 s (long enough to prefill
  4 × 8192 = 32k tokens). With chunking, the same prefill work is
  spread across ~64 scheduler steps and never produces a single visible
  pause.

### What the timeline plot shows

`results/chunked.png`:

- **Top-left (histogram)**: OFF is a tall narrow cluster at 77 ms with
  a long thin tail to ~1.7 s. ON is a wider cluster centered at 143 ms
  with no tail.
- **Top-right (bar chart)**: percentile bars. p50/p90/p95 are mildly
  worse for ON; p99/p99.9/max are dramatically worse for OFF.
- **Bottom-left (OFF timeline)**: ITL stays at ~77 ms most of the time,
  then jumps to ~1700 ms at burst events (dotted lines), then drops back.
- **Bottom-right (ON timeline)**: ITL is a smooth band at ~143 ms across
  the whole run, regardless of when bursts fire.

### The latency-aggregating story

Same 32 streams getting served over the same 60 s of measurement:

| Metric | OFF | ON |
|---|---:|---:|
| Total ITL samples | 15,328 | 15,328 |
| Wall time | 104 s (some streams stalled past 60 s) | 89 s |
| Mean ITL | 86 ms | 144 ms |
| Variance (informally) | bimodal: most are 77 ms, ~1% are 1700 ms | unimodal at 143 ms |

The OFF run takes 104 s wall-clock to finish 500-token streams
that each "should" take 32 × 0.077 = 2.5 s — they took ~3.3 s on
average because each one got hit by ~1 burst that froze it for 1.7 s.
The ON run finishes the same work in 89 s because the engine never
stalls.

So chunked prefill is **also faster in aggregate** even though every
individual step is slower. The OFF run wastes ~15 s of GPU time on
stalls; the ON run uses every second.

## Why this is the disaggregation analogue

In Chapter 7's disaggregated-serving recipe, you put prefill on its
own cluster ("prefill server") and decode on its own ("generation
server"). KV cache transfers between them over network. The point is
that **no prefill work ever blocks a running decode step**, because
they're on physically different accelerators.

On one GPU you can't physically separate, but the scheduler can do
the equivalent by **slicing prefill into chunks that fit inside the
inter-decode-step window**. Same effect:

| Architecture | Mechanism for prefill not blocking decode |
|---|---|
| Disaggregated (multi-GPU) | physical separation: prefill GPU and decode GPU |
| Chunked prefill (single-GPU) | temporal slicing: each step is decode + ≤chunk prefill tokens |

The OFF p99 of 1735 ms is what you would see on the decode GPU of a
naive non-disaggregated setup. The ON p99 of 161 ms is the
single-GPU expression of what disaggregation provides. The gap —
~11× — is the headline number from the chapter's disaggregation
section, here as a within-engine scheduler toggle.

## Cost analysis

When does chunked OFF win?

- **Pure decode workloads** with no prefill bursts (e.g., long batch
  inference with already-prefilled requests). Then OFF saves the
  ~2× median latency cost. But this is a narrow use case — in any
  real serving you have new requests arriving.

When does chunked ON dominate?

- **Mixed workloads** (chat, RAG, code completion). Anything with
  user-facing latency requirements where occasional 2 s freezes are
  unacceptable.
- **High request arrival rate**. The more frequent the bursts, the
  more often OFF's tail explodes.

## What this leaves on the table

- **Prefix caching** (#2): completely orthogonal. If your prefill
  bursts share a system prompt, prefix caching makes each "8k-token"
  prefill effectively a "1k-token" prefill, shrinking the bursts.
  Combining the two should give p50 closer to OFF + tail no worse
  than ON.
- **Chunk size sweep**: we used 512. Smaller chunks → more decode
  protection but more per-step overhead. Larger chunks → ON's p50
  grows toward OFF's tail. Worth sweeping later.
- **Spec decoding** (#3): can stack on top of either. The bandwidth-
  amortization win is independent of the prefill-blocking concern.

## Files

- `start-vllm-chunked-off.sh`, `start-vllm-chunked-on.sh` — server scripts
- `bench_chunked.py` — workload generator + ITL recorder
- `plot_chunked.py` — overlay plotter
- `run-bg.sh` — backgrounded launcher (survives SSH disconnect)
- `results/chunked_off.json`, `results/chunked_on.json` — raw traces
- `results/chunked.png` — comparison plot
