"""
Chunked-prefill workload generator + ITL recorder.

Spawns two client populations against vLLM:
  - "decode" streams: N_DECODE long-running, small-prompt requests
    that stream tokens slowly. We record per-token timestamps for
    each so we can compute ITL distributions and timelines.
  - "burst" requests: every BURST_PERIOD_S, fire N_BURST requests
    with FAT_PROMPT_LEN tokens that generate just a few output
    tokens. These hammer the engine with prefill work.

The decode streams stay alive for RUN_DURATION_S total. After they
finish, we dump per-stream timestamp series and burst event times
to a JSON file so the plot can correlate ITL spikes with bursts.

Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ENDPOINT = "http://localhost:8000/v1/completions"
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def make_prompt_ids(length: int, salt: int) -> list[int]:
    """Pseudo-random token IDs unique per (salt) to defeat prefix caching."""
    offset = salt * 7919
    return [((i + offset) % 30000) + 100 for i in range(length)]


def stream_request(prompt_ids: list[int], max_tokens: int,
                   timeout: float, ts_out: list[float],
                   t0: float) -> str | None:
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0,
        "ignore_eos": True,
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
                ts_out.append(time.perf_counter() - t0)
        return None
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def decode_worker(idx: int, t0: float, run_until: float,
                  results: list, decode_prompt_len: int):
    """Repeatedly run streaming completions until run_until."""
    salt = 1_000 + idx
    while time.perf_counter() < run_until:
        ts_local: list[float] = []
        err = stream_request(
            make_prompt_ids(decode_prompt_len, salt),
            max_tokens=500,
            timeout=180.0,
            ts_out=ts_local, t0=t0,
        )
        results.append({
            "stream_id": idx,
            "salt": salt,
            "error": err,
            "timestamps_rel": ts_local,
        })
        salt += 10_000  # new prompt next iteration
        # If the request finished too fast (rejected), back off briefly.
        if not ts_local and err:
            time.sleep(1.0)


def burst_worker(t0: float, run_until: float,
                 burst_period_s: float, n_burst: int,
                 fat_prompt_len: int,
                 burst_events: list[dict]):
    """Fire periodic bursts of fat-prompt requests."""
    salt_base = 100_000
    next_at = time.perf_counter() + burst_period_s
    while time.perf_counter() < run_until:
        if time.perf_counter() < next_at:
            time.sleep(0.05)
            continue
        burst_t = time.perf_counter() - t0
        # Fire n_burst concurrent fat requests; don't wait — let them
        # finish on their own threads.
        burst_threads = []
        for k in range(n_burst):
            salt = salt_base + k
            t = threading.Thread(target=stream_request, args=(
                make_prompt_ids(fat_prompt_len, salt),
                10, 180.0, [], t0,
            ))
            t.start()
            burst_threads.append(t)
        burst_events.append({
            "t_rel": burst_t,
            "n_requests": n_burst,
            "prompt_len": fat_prompt_len,
        })
        salt_base += n_burst
        next_at += burst_period_s
    # Don't wait for outstanding burst threads — main joins via timeout.


def compute_itl_stats(stream_results: list[dict]) -> dict:
    """Pull ITLs from all streams, skip first PREFILL_SKIP per stream."""
    PREFILL_SKIP = 20
    all_itls: list[float] = []
    per_stream_medians = []
    for s in stream_results:
        ts = s.get("timestamps_rel", [])
        if len(ts) <= PREFILL_SKIP + 2:
            continue
        ts_eff = ts[PREFILL_SKIP:]
        diffs = [ts_eff[j+1] - ts_eff[j] for j in range(len(ts_eff)-1)]
        if not diffs:
            continue
        all_itls.extend(diffs)
        per_stream_medians.append(statistics.median(diffs))
    if not all_itls:
        return {"n_itls": 0}
    sorted_itls = sorted(all_itls)
    n = len(sorted_itls)
    def pct(p): return sorted_itls[int(p * (n - 1))]
    return {
        "n_itls": n,
        "n_streams_used": len(per_stream_medians),
        "ms_mean": 1000 * sum(all_itls) / n,
        "ms_p50": 1000 * pct(0.50),
        "ms_p90": 1000 * pct(0.90),
        "ms_p95": 1000 * pct(0.95),
        "ms_p99": 1000 * pct(0.99),
        "ms_p99_9": 1000 * pct(0.999),
        "ms_max": 1000 * max(all_itls),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Total run duration in seconds.")
    parser.add_argument("--n-decode", type=int, default=32,
                        help="Number of concurrent decode streams.")
    parser.add_argument("--decode-prompt-len", type=int, default=128)
    parser.add_argument("--burst-period", type=float, default=8.0,
                        help="Seconds between prefill bursts.")
    parser.add_argument("--n-burst", type=int, default=4,
                        help="Requests per burst.")
    parser.add_argument("--fat-prompt-len", type=int, default=8192)
    parser.add_argument("--label", default="chunked",
                        help="Free-form label saved into output JSON.")
    args = parser.parse_args()

    # Sanity-check vLLM is reachable
    try:
        urllib.request.urlopen("http://localhost:8000/v1/models", timeout=10).read()
    except Exception as e:
        print(f"ERROR: vLLM not reachable: {e}")
        return

    print(f"label={args.label}, duration={args.duration}s, "
          f"decode_streams={args.n_decode}, "
          f"burst every {args.burst_period}s ({args.n_burst}x{args.fat_prompt_len})")

    t0 = time.perf_counter()
    run_until = t0 + args.duration

    stream_results: list[dict] = []
    burst_events: list[dict] = []

    workers = []
    for idx in range(args.n_decode):
        t = threading.Thread(target=decode_worker, args=(
            idx, t0, run_until, stream_results, args.decode_prompt_len,
        ))
        t.start()
        workers.append(t)

    burst_thread = threading.Thread(target=burst_worker, args=(
        t0, run_until, args.burst_period, args.n_burst,
        args.fat_prompt_len, burst_events,
    ))
    burst_thread.start()

    # Wait for all decode threads
    for t in workers:
        t.join(timeout=args.duration + 60)
    burst_thread.join(timeout=10)

    wall = time.perf_counter() - t0
    stats = compute_itl_stats(stream_results)

    print(f"\nRun done in {wall:.1f}s. {stats.get('n_itls', 0)} ITL samples "
          f"from {stats.get('n_streams_used', 0)} streams.")
    if "ms_p50" in stats:
        print(f"  p50 = {stats['ms_p50']:7.1f} ms   p95 = {stats['ms_p95']:8.1f} ms")
        print(f"  p99 = {stats['ms_p99']:7.1f} ms   max = {stats['ms_max']:8.1f} ms")

    out = {
        "label": args.label,
        "endpoint": ENDPOINT,
        "model": MODEL,
        "duration_s": wall,
        "config": vars(args),
        "stats": stats,
        "burst_events": burst_events,
        "streams": stream_results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}  ({len(stream_results)} stream records)")


if __name__ == "__main__":
    main()
