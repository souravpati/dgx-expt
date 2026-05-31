"""
Prefix-caching benchmark: send N sequential requests sharing the same
~4096-token system-prompt prefix, record TTFT for each.

With caching ON: request 1 pays full prefill, requests 2..N pay only
the small per-query prefill.
With caching OFF: every request pays the full prefill.

Uses /v1/completions with `prompt` as a list of token IDs to skip
tokenization and guarantee exact prefix lengths.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ENDPOINT = "http://localhost:8000/v1/completions"
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

SHARED_PREFIX_LEN = 4096
QUERY_LEN = 64
OUTPUT_TOKENS = 32
N_REQUESTS = 10


def make_shared_prefix() -> list[int]:
    """Deterministic 4096-token prefix shared across all requests."""
    return [((i * 7919) % 30000) + 100 for i in range(SHARED_PREFIX_LEN)]


def make_query(idx: int) -> list[int]:
    """Per-request query; different content per idx so we're not just
    re-asking the same generation, but tokens are valid IDs."""
    offset = idx * 1009 + 50000
    return [((i + offset) % 25000) + 200 for i in range(QUERY_LEN)]


def stream_ttft(prompt_ids: list[int], timeout: float) -> dict:
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt_ids,
        "max_tokens": OUTPUT_TOKENS,
        "stream": True,
        "temperature": 0,
        "ignore_eos": True,
    })
    req = urllib.request.Request(
        ENDPOINT,
        data=body.encode(),
        headers={"Content-Type": "application/json"},
    )
    t_send = time.perf_counter()
    t_first = None
    n_chunks = 0
    err = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                if t_first is None:
                    t_first = time.perf_counter()
                n_chunks += 1
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    t_end = time.perf_counter()
    return {
        "ttft_s": (t_first - t_send) if t_first is not None else None,
        "wall_s": t_end - t_send,
        "n_chunks": n_chunks,
        "error": err,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--n", type=int, default=N_REQUESTS)
    args = parser.parse_args()

    try:
        urllib.request.urlopen("http://localhost:8000/v1/models", timeout=10).read()
    except Exception as e:
        print(f"ERROR: vLLM not reachable: {e}")
        return

    print(f"label={args.label}, shared_prefix={SHARED_PREFIX_LEN}, "
          f"query={QUERY_LEN}, n_requests={args.n}")

    shared = make_shared_prefix()

    # Tiny warmup so first measured request isn't paying CUDA-graph init.
    print("warmup...")
    stream_ttft([100, 101, 102, 103], timeout=60.0)

    rows = []
    for i in range(args.n):
        prompt = shared + make_query(i)
        r = stream_ttft(prompt, timeout=60.0)
        r["i"] = i
        r["prompt_len"] = len(prompt)
        rows.append(r)
        ttft_ms = (r["ttft_s"] * 1000) if r["ttft_s"] else float("nan")
        flag = "  ERR: " + r["error"] if r["error"] else ""
        print(f"  [{i+1:>2}/{args.n}] TTFT = {ttft_ms:7.1f} ms  "
              f"chunks={r['n_chunks']:>3}  wall={r['wall_s']:5.2f}s{flag}")

    ttfts = [r["ttft_s"] * 1000 for r in rows if r["ttft_s"]]
    if ttfts:
        print(f"\nTTFT  first={ttfts[0]:.1f} ms  "
              f"rest_min={min(ttfts[1:]):.1f}  "
              f"rest_median={sorted(ttfts[1:])[len(ttfts)//2-1]:.1f}  "
              f"rest_max={max(ttfts[1:]):.1f}")

    out = {
        "label": args.label,
        "model": MODEL,
        "config": {
            "shared_prefix_len": SHARED_PREFIX_LEN,
            "query_len": QUERY_LEN,
            "output_tokens": OUTPUT_TOKENS,
            "n_requests": args.n,
        },
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
