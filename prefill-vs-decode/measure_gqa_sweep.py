"""
GQA ratio sweep: does decode throughput scale linearly with N/K?

Chapter 4's prediction: decode arithmetic intensity = N/K (the GQA
group size), and decode is bandwidth-bound. So achieved TF/s should
be exactly W_HBM x N/K -- a clean linear function of the architecture
choice, independent of every other knob.

We sweep N_KV_HEADS in {1, 2, 4, 8, 16, 32} at fixed N_HEADS=32,
T_q=1, T_kv=4096 (footprint big enough to escape L2). Each ratio
maps to a recognizable architecture:

  N/K = 1  -> MHA              (Llama 2 7B-shaped)
  N/K = 2  -> GQA-16           (e.g. Llama 2 70B)
  N/K = 4  -> GQA-8            (Llama 3.1 8B)
  N/K = 8  -> GQA-4            (Llama 3 70B)
  N/K = 16 -> GQA-2
  N/K = 32 -> MQA              (single shared KV head)

We use SDPA with enable_gqa=True so the kernel reads K/V at K heads
(not N) -- matches what we found in compare_decode_kernels.py.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

DEVICE = "cuda"
DTYPE = torch.bfloat16
BYTES_PER_ELEM = 2
RESULTS = Path(__file__).parent / "results" / "gqa_sweep.json"

N_HEADS = 32
HEAD_DIM = 128
S_KV = 4096          # KV depth: 8 KV heads * 4096 * 128 * 2 = 8 MB; with B=8 -> 64 MB (out of L2)
WARMUP = 5
ITERS = 30


def _sync():
    torch.cuda.synchronize()


def time_fn(fn) -> float:
    for _ in range(WARMUP):
        fn()
    _sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(ITERS):
        start.record()
        fn()
        end.record()
        _sync()
        times.append(start.elapsed_time(end) / 1e3)
    times.sort()
    return times[len(times) // 2]


def kv_cache_bytes(B: int, S: int, n_kv: int) -> int:
    """Bytes the kernel reads from DRAM: K + V + Q + O (Q/O small)."""
    bytes_q = B * N_HEADS * HEAD_DIM * BYTES_PER_ELEM
    bytes_kv = 2 * B * n_kv * S * HEAD_DIM * BYTES_PER_ELEM
    bytes_o = B * N_HEADS * HEAD_DIM * BYTES_PER_ELEM
    return bytes_q + bytes_kv + bytes_o


def attention_flops(B: int, S: int) -> int:
    """Compute is always at full N expansion (Q@K + Attn@V)."""
    return 4 * B * N_HEADS * S * HEAD_DIM


def make_step(B: int, S: int, n_kv: int):
    q = torch.randn(B, N_HEADS, 1, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, n_kv, S, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, n_kv, S, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    # enable_gqa=True only valid when N_HEADS % n_kv == 0 (always true here).
    if n_kv == N_HEADS:
        return lambda: F.scaled_dot_product_attention(q, k, v, is_causal=False)
    return lambda: F.scaled_dot_product_attention(q, k, v, is_causal=False,
                                                   enable_gqa=True)


def measure(B: int, n_kv: int) -> dict:
    nk_ratio = N_HEADS / n_kv
    try:
        step = make_step(B, S_KV, n_kv)
        sec = time_fn(step)
    except Exception as e:
        return {
            "B": B, "n_kv_heads": n_kv, "n_over_k": nk_ratio,
            "error": f"{type(e).__name__}: {e}",
        }
    flops = attention_flops(B, S_KV)
    bm = kv_cache_bytes(B, S_KV, n_kv)
    point = {
        "B": B,
        "n_kv_heads": n_kv,
        "n_over_k": nk_ratio,
        "seconds": sec,
        "flops": flops,
        "bytes_moved": bm,
        "intensity_flops_per_byte": flops / bm,
        "tflops_per_s": flops / sec / 1e12,
        "gb_per_s": bm / sec / 1e9,
    }
    print(
        f"  B={B:2d}  K={n_kv:2d} heads (N/K={nk_ratio:5.1f})  "
        f"I={point['intensity_flops_per_byte']:6.2f} F/B  "
        f"{point['tflops_per_s']:6.3f} TF/s  "
        f"{point['gb_per_s']:6.1f} GB/s"
    )
    return point


def main():
    assert torch.cuda.is_available()
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Fixed: N_HEADS={N_HEADS}, HEAD_DIM={HEAD_DIM}, T_q=1, T_kv={S_KV}")

    kv_choices = [1, 2, 4, 8, 16, 32]   # all clean divisors of 32
    batches = [1, 8]

    results = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "timestamp": time.time(),
        "fixed": {
            "N_HEADS": N_HEADS,
            "HEAD_DIM": HEAD_DIM,
            "T_q": 1,
            "T_kv": S_KV,
            "dtype": "bfloat16",
        },
        "peak_bw_gb_per_s": 213.0,
        "rows": [],
    }

    for B in batches:
        print(f"\n--- B={B} ---")
        for n_kv in kv_choices:
            results["rows"].append(measure(B, n_kv))

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
