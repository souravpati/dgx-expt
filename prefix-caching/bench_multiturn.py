"""
Multi-turn prefix caching: simulate a growing conversation where each
turn extends the prior context. Measure TTFT per turn.

We use a deterministic token-ID sequence and "grow the prompt by
STEP tokens" each turn. This is equivalent in cache terms to a real
multi-turn dialog where each turn appends (user message + assistant
response) — because vLLM's prefix-caching key is the leading token-ID
sequence, regardless of how it was generated.

With caching ON: TTFT(turn N) ≈ time to prefill STEP new tokens,
roughly constant across N regardless of total prompt length.

Without caching: TTFT(turn N) scales linearly with N (full prefill).

Server must be running with --enable-prefix-caching.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

ENDPOINT = "http://localhost:8000/v1/completions"
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

STEP_TOKENS = 1024
N_TURNS = 5
OUTPUT_TOKENS = 16


def make_growing_prompt(turn_idx: int) -> list[int]:
    """Turn N's prompt is the first (N+1)*STEP_TOKENS of a deterministic
    sequence. Shared prefix grows by STEP_TOKENS per turn."""
    total = (turn_idx + 1) * STEP_TOKENS
    return [((i * 7919) % 30000) + 100 for i in range(total)]


def stream_ttft(prompt_ids: list[int]) -> dict:
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt_ids,
        "max_tokens": OUTPUT_TOKENS,
        "stream": True,
        "temperature": 0,
        "ignore_eos": True,
    })
    req = urllib.request.Request(
        ENDPOINT, data=body.encode(),
        headers={"Content-Type": "application/json"},
    )
    t_send = time.perf_counter()
    t_first = None
    n_chunks = 0
    err = None
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
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
    return {
        "ttft_s": (t_first - t_send) if t_first is not None else None,
        "wall_s": time.perf_counter() - t_send,
        "n_chunks": n_chunks,
        "error": err,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="multiturn_on")
    parser.add_argument("--n-turns", type=int, default=N_TURNS)
    args = parser.parse_args()

    try:
        urllib.request.urlopen("http://localhost:8000/v1/models", timeout=10).read()
    except Exception as e:
        print(f"ERROR: vLLM not reachable: {e}")
        return

    print(f"label={args.label}, step={STEP_TOKENS} tokens, n_turns={args.n_turns}")

    # Warmup so first turn doesn't pay graph-init costs.
    print("warmup...")
    stream_ttft([100, 101, 102, 103, 104])

    rows = []
    for n in range(args.n_turns):
        prompt = make_growing_prompt(n)
        r = stream_ttft(prompt)
        r["turn"] = n + 1
        r["prompt_len"] = len(prompt)
        r["new_tokens_this_turn"] = STEP_TOKENS  # turn 1 adds STEP fresh, turns 2+ add STEP new
        rows.append(r)
        ttft_ms = (r["ttft_s"] * 1000) if r["ttft_s"] else float("nan")
        flag = "  ERR: " + r["error"] if r["error"] else ""
        print(f"  Turn {n+1}: prompt_len={r['prompt_len']:>5}  "
              f"TTFT={ttft_ms:7.1f} ms  wall={r['wall_s']:5.2f}s{flag}")

    out = {
        "label": args.label,
        "model": MODEL,
        "step_tokens": STEP_TOKENS,
        "n_turns": args.n_turns,
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
