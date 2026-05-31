"""
Streaming benchmark client for vLLM Llama 3.1 8B decode step time.

For each (B, S):
  - Fire B concurrent streaming requests, each with a prompt of S
    tokens (different start per thread so vLLM can't dedupe via
    prefix caching).
  - Record per-chunk timestamps for every stream.
  - Compute inter-token latency (ITL) per request, skip the first
    PREFILL_SKIP tokens (covers TTFT and warmup), take the median.
  - Median across requests = decode step time estimate.

vLLM's OpenAI-compatible endpoint accepts `prompt` as a list of
integer token IDs, so we don't need a local tokenizer. We use IDs in
[100, 30099] which are safe for the Llama 3 vocab.

Talks to http://localhost:8000 by default. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

RESULTS = Path(__file__).parent / "results" / "measured_step_time.json"

ENDPOINT = "http://localhost:8000/v1/completions"
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def make_prompt_ids(S: int, thread_id: int) -> list[int]:
    # Unique-per-thread pseudo-random token IDs to defeat prefix caching.
    offset = thread_id * 7919
    return [((i + offset) % 30000) + 100 for i in range(S)]


def stream_request(prompt_ids: list[int], out_timestamps: list[float],
                   timeout: float, max_tokens: int) -> str | None:
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0,
        "ignore_eos": True,  # don't stop early
    })
    req = urllib.request.Request(
        ENDPOINT,
        data=body.encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                # Each SSE event = one streaming chunk = ~1 token in vLLM.
                out_timestamps.append(time.perf_counter())
        return None
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:200]}"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def run_point(B: int, S: int, timeout: float,
              max_tokens: int, prefill_skip: int) -> dict:
    print(f"  B={B:4d} S={S:5d}  launching {B} streams...", flush=True)
    streams: list[list[float]] = [[] for _ in range(B)]
    errors: list[str | None] = [None] * B
    threads = []

    def worker(idx: int):
        prompt_ids = make_prompt_ids(S, idx)
        errors[idx] = stream_request(prompt_ids, streams[idx], timeout, max_tokens)

    t_start = time.perf_counter()
    for i in range(B):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join(timeout=timeout + 30)
    t_end = time.perf_counter()

    wall = t_end - t_start
    err_count = sum(1 for e in errors if e is not None)
    if err_count:
        sample_err = next((e for e in errors if e is not None), None)
        print(f"    WARNING: {err_count}/{B} streams errored. sample: {sample_err}")

    # Per-request: ITL = diffs between consecutive chunks, drop first few.
    per_request_median_itl = []
    per_request_chunk_count = []
    for ts in streams:
        per_request_chunk_count.append(len(ts))
        if len(ts) <= prefill_skip + 2:
            continue
        ts_eff = ts[prefill_skip:]
        diffs = [ts_eff[j + 1] - ts_eff[j] for j in range(len(ts_eff) - 1)]
        if diffs:
            per_request_median_itl.append(statistics.median(diffs))

    if not per_request_median_itl:
        return {
            "B": B, "S": S,
            "error": "no usable streams",
            "wall_s": wall,
            "errors_count": err_count,
            "chunk_counts": per_request_chunk_count,
        }

    # Step time = median across requests of (median ITL within a request).
    step_s = statistics.median(per_request_median_itl)
    return {
        "B": B, "S": S,
        "wall_s": wall,
        "errors_count": err_count,
        "n_used_streams": len(per_request_median_itl),
        "ms_per_step": step_s * 1000,
        "ms_per_token_user": step_s * 1000,         # what one user sees
        "ms_per_token_aggregate": step_s * 1000 / B, # cluster perspective
        "itl_min_ms": min(per_request_median_itl) * 1000,
        "itl_max_ms": max(per_request_median_itl) * 1000,
        "chunk_counts": per_request_chunk_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(RESULTS))
    parser.add_argument("--batches", default="1,4,16,64,128,256",
                        help="Comma-separated batch sizes to sweep.")
    parser.add_argument("--contexts", default="2048,8192,32768",
                        help="Comma-separated context lengths (S).")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Decode tokens per request. Bump to 200+ for "
                             "steady-state ITL measurement with large prefill_skip.")
    parser.add_argument("--prefill-skip", type=int, default=5,
                        help="Drop first N chunks before computing ITL. "
                             "Use ~120 for steady-state mode.")
    args = parser.parse_args()

    batches = [int(x) for x in args.batches.split(",")]
    contexts = [int(x) for x in args.contexts.split(",")]

    # Warmup: small request so the engine is hot before we time anything.
    print(f"warmup... (max_tokens={args.max_tokens}, prefill_skip={args.prefill_skip})")
    warm_ts: list[float] = []
    err = stream_request(make_prompt_ids(128, 0), warm_ts, timeout=60.0,
                         max_tokens=8)
    if err:
        print(f"warmup error: {err}")
        return
    print(f"  warmup got {len(warm_ts)} chunks; first ITL ok.")

    results = {
        "endpoint": ENDPOINT,
        "model": MODEL,
        "max_tokens": args.max_tokens,
        "prefill_skip": args.prefill_skip,
        "timestamp": time.time(),
        "rows": [],
    }

    for S in contexts:
        for B in batches:
            try:
                row = run_point(B, S, timeout=args.timeout,
                                max_tokens=args.max_tokens,
                                prefill_skip=args.prefill_skip)
            except Exception:
                traceback.print_exc()
                row = {"B": B, "S": S, "error": "exception during run_point"}
            results["rows"].append(row)
            if "ms_per_step" in row:
                print(
                    f"    -> ms/step (user-visible ITL) = {row['ms_per_step']:7.2f}  "
                    f"ms/token aggregate = {row['ms_per_token_aggregate']:6.2f}  "
                    f"wall {row['wall_s']:6.1f}s",
                    flush=True,
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
