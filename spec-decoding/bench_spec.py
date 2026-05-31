"""
Speculative decoding bench: prose + code workloads, single-stream (B=1).

For each prompt we measure wall-clock tokens/sec from streamed
completions and snapshot vLLM's /metrics endpoint before/after to
compute the acceptance rate.

Run pattern:
    python3 bench_spec.py --K 0 --out results/spec_K0.json
    # then restart vLLM with K=3 and:
    python3 bench_spec.py --K 3 --out results/spec_K3.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

CHAT_ENDPOINT = "http://localhost:8000/v1/chat/completions"
METRICS_ENDPOINT = "http://localhost:8000/metrics"
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# --- Prose prompts: free-form narrative ---
PROSE_PROMPTS = [
    "Write a four-paragraph short story about a lighthouse keeper who finds a message in a bottle. Use vivid sensory detail.",
    "Explain to a curious 10-year-old why the sky is blue, using simple analogies and everyday examples.",
    "Describe a typical winter morning in a small mountain town from the perspective of someone who has lived there their whole life.",
    "Write a thoughtful 200-word essay on why people enjoy reading fiction.",
    "Compose a letter from a retired ship captain to his grandchild, recounting his most memorable storm at sea.",
    "Write a calm, descriptive passage about a quiet afternoon spent in a public library.",
    "Imagine you are a botanist; describe in elegant prose the first time you encountered a rare orchid in the wild.",
    "Write a reflective passage about the joys and frustrations of learning to play a musical instrument as an adult.",
]

# --- Code prompts: focused completion ---
CODE_PROMPTS = [
    "Write a Python function `quicksort(arr: list[int]) -> list[int]` that returns a sorted copy of the list. Include a clear docstring and one inline comment.",
    "Implement a Python class `LRUCache` with `get(key) -> Optional[Any]` and `put(key, value) -> None`, using `collections.OrderedDict`. Include type hints.",
    "Write a Python function `is_palindrome(s: str) -> bool` that ignores case and non-alphanumeric characters. Include three example assertions.",
    "Write a Python function `fibonacci(n: int) -> list[int]` returning the first n Fibonacci numbers. Use iteration, not recursion. Include type hints.",
    "Implement Python `binary_search(arr: list[int], target: int) -> int` returning the index or -1 if not found. Add a docstring with O(log n) note.",
    "Write a Python function `merge_intervals(intervals: list[tuple[int,int]]) -> list[tuple[int,int]]` that merges overlapping intervals. Sort first.",
    "Write a Python `Trie` class supporting `insert(word: str)` and `search(word: str) -> bool` and `starts_with(prefix: str) -> bool`. Use dict children.",
    "Write a Python function `flatten(nested: list) -> list` that flattens an arbitrarily nested list of integers. Handle empty input.",
]

OUTPUT_TOKENS = 256


def stream_chat(prompt: str, t0: float) -> tuple[int, float, str | None]:
    """Stream a chat completion. Returns (n_tokens_received, wall_seconds, err)."""
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": OUTPUT_TOKENS,
        "stream": True,
        "temperature": 0,
        "ignore_eos": False,
    })
    req = urllib.request.Request(
        CHAT_ENDPOINT,
        data=body.encode(),
        headers={"Content-Type": "application/json"},
    )
    n = 0
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=180.0) as r:
            for raw in r:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                # Count tokens by counting delta-content chunks; each
                # streaming event in vLLM is one decode step's emit.
                # With spec decoding, multiple tokens may show up in a
                # single chunk's content -- which is fine: we'll also
                # report total tokens from the final usage block via
                # /metrics polling.
                for ch in chunk.get("choices", []):
                    delta = ch.get("delta", {})
                    if "content" in delta and delta["content"]:
                        # one chunk -> one decode step regardless of token count
                        n += 1
        wall = time.perf_counter() - start
        return n, wall, None
    except Exception as e:
        return n, time.perf_counter() - start, f"{type(e).__name__}: {e}"


def get_metrics() -> dict[str, float]:
    """Read /metrics, parse the spec_decode counters we care about."""
    try:
        raw = urllib.request.urlopen(METRICS_ENDPOINT, timeout=10).read().decode()
    except Exception as e:
        return {"error": str(e)}
    metrics = {}
    interesting = (
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_emitted_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:spec_decode_draft_acceptance_rate",
        "vllm:spec_decode_efficiency",
        "vllm:generation_tokens_total",
        "vllm:prompt_tokens_total",
        "vllm:time_per_output_token_seconds",  # might be a histogram
    )
    for line in raw.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r"^([^\s{]+)(\{[^}]*\})?\s+([\d.eE+-]+)", line)
        if not m:
            continue
        name = m.group(1)
        if name not in interesting:
            continue
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        # If the same metric appears multiple times (labels), keep the
        # sum so we don't lose info.
        metrics[name] = metrics.get(name, 0.0) + v
    return metrics


def metric_delta(after: dict, before: dict, key: str) -> float | None:
    if key in after and key in before:
        return after[key] - before[key]
    return None


def run_workload(name: str, prompts: list[str]) -> dict:
    print(f"\n--- {name} ({len(prompts)} prompts) ---")
    m_before = get_metrics()
    rows = []
    for i, p in enumerate(prompts):
        t0 = time.perf_counter()
        n, wall, err = stream_chat(p, t0)
        tps = n / wall if wall > 0 else 0
        rows.append({
            "i": i, "n_streamed_chunks": n,
            "wall_s": wall, "tps_chunks": tps,
            "error": err,
        })
        flag = "  ERR: " + err if err else ""
        print(f"  [{i+1:>2}/{len(prompts)}] {n:>4} chunks  {wall:6.2f}s  "
              f"{tps:6.1f} chunks/s{flag}")
    m_after = get_metrics()
    drafted = metric_delta(m_after, m_before, "vllm:spec_decode_num_draft_tokens_total")
    accepted = metric_delta(m_after, m_before, "vllm:spec_decode_num_accepted_tokens_total")
    emitted = metric_delta(m_after, m_before, "vllm:spec_decode_num_emitted_tokens_total")
    generated = metric_delta(m_after, m_before, "vllm:generation_tokens_total")
    accept_rate = (accepted / drafted) if (drafted and accepted is not None) else None
    return {
        "name": name,
        "n_prompts": len(prompts),
        "runs": rows,
        "mean_wall_s": statistics.mean(r["wall_s"] for r in rows if not r["error"]) if any(not r["error"] for r in rows) else None,
        "mean_chunks_per_s": statistics.mean(r["tps_chunks"] for r in rows if not r["error"]) if any(not r["error"] for r in rows) else None,
        "spec_drafted": drafted,
        "spec_accepted": accepted,
        "spec_emitted": emitted,
        "spec_acceptance_rate": accept_rate,
        "generation_tokens_total_delta": generated,
        "metrics_before": m_before,
        "metrics_after": m_after,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, required=True,
                        help="num_speculative_tokens the server is configured with.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    try:
        urllib.request.urlopen("http://localhost:8000/v1/models", timeout=10).read()
    except Exception as e:
        print(f"ERROR: vLLM not reachable: {e}")
        return

    print(f"K = {args.K}")
    print(f"Model = {MODEL}")

    out = {
        "K": args.K,
        "model": MODEL,
        "endpoint": CHAT_ENDPOINT,
        "timestamp": time.time(),
        "max_output_tokens": OUTPUT_TOKENS,
    }
    # Warmup so first prompt's compile/CUDA-graph cost doesn't skew prose.
    print("warmup...")
    stream_chat("Say hi briefly.", time.perf_counter())

    out["prose"] = run_workload("prose", PROSE_PROMPTS)
    out["code"]  = run_workload("code",  CODE_PROMPTS)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")

    def summarize(label, rs):
        if rs.get("mean_chunks_per_s") is None:
            print(f"  {label}: no data")
            return
        print(f"  {label}:  {rs['mean_chunks_per_s']:6.1f} chunks/s  "
              f"acc rate = {rs['spec_acceptance_rate']}  "
              f"emitted = {rs['spec_emitted']}")
    print("\nSummary:")
    summarize("prose", out["prose"])
    summarize("code",  out["code"])


if __name__ == "__main__":
    main()
