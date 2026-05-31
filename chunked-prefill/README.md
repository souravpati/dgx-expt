# Chunked prefill — the single-GPU disaggregation analogue

**Question**: how much does `--enable-chunked-prefill` protect a
running decode stream from latency spikes when prefill bursts hit
the engine?

True disaggregation (separate prefill and decode clusters) is what
the chapter recommends at scale. On one GPU you can't physically
separate, but vLLM's chunked-prefill scheduler slices prefill work
into small chunks and **interleaves them with decode steps**.
That's mathematically equivalent to running prefill on a "virtual
secondary worker" inside the same GPU — same problem, same fix.

The gap between chunked-OFF and chunked-ON ITL distributions is
literally the latency win that disaggregation provides across machines.

## Workload

Two concurrent client populations against one vLLM server:

| Population | Count | Prompt size | Generates | Purpose |
|---|---:|---:|---:|---|
| **Decode stream** | 32 | ~128 tokens | 500 tokens streaming | Steady state. We measure ITL on these. |
| **Prefill burst** | 4 every 8 s | 8192 tokens | 10 tokens | Floods the engine periodically. |

We pin a `model=meta-llama/Meta-Llama-3.1-8B-Instruct` instance and
run the bench against it under two configurations:

1. **chunked OFF** — `--no-enable-chunked-prefill`
2. **chunked ON**  — `--enable-chunked-prefill --max-num-batched-tokens 512`

The bench records per-token timestamps for every decode stream and
the start time of each prefill burst, so we can correlate ITL
spikes to burst events.

## What we expect

- **OFF**: when a burst lands, vLLM runs all 4×8192 = 32k prefill
  tokens before the decode batch gets another step. At ~80 ms per
  prefill step, that's a multi-second pause. Decode-stream ITL p99
  should spike to seconds.
- **ON**: each prefill request is sliced into 512-token chunks. Each
  scheduler step packs one chunk + the running decode batch. ITL
  thickens slightly (~30%) but no big spikes.

## Files

- `start-vllm-chunked-off.sh` — server with chunked prefill disabled
- `start-vllm-chunked-on.sh` — server with chunked prefill enabled
- `bench_chunked.py` — workload generator + ITL recorder
- `plot_chunked.py` — overlays both runs' ITL distributions
- `run-bg.sh` — backgrounded wrapper for safe SSH disconnect
- `results/chunked_off.json`, `results/chunked_on.json` — raw traces

## Protocol

1. Start the OFF server (Terminal A):
   ```
   ./start-vllm-chunked-off.sh
   ```
2. Run the bench (Terminal B, or via Claude):
   ```
   python3 bench_chunked.py --out results/chunked_off.json
   ```
3. Ctrl-C the OFF server, start the ON server (Terminal A):
   ```
   ./start-vllm-chunked-on.sh
   ```
4. Run the bench:
   ```
   python3 bench_chunked.py --out results/chunked_on.json
   ```
5. Plot:
   ```
   python3 plot_chunked.py
   ```
