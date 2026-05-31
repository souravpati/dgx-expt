"""
Why does decode hit only 28% of peak bandwidth?

Hypothesis A: the `repeat_interleave` in measure_attention.py materializes
a 4x-expanded K/V tensor in DRAM. SDPA then reads the expanded tensor, so
the kernel is actually moving 4x the bytes we charge it for. The "kernel
inefficiency" is just measurement bookkeeping.

Hypothesis B: PyTorch SDPA's decode kernel doesn't use a split-K /
flash-decode parallelization, leaving SMs idle.

This script runs the same decode shape through three paths to find out:
  1. SDPA + manual repeat_interleave (the current path -- baseline)
  2. SDPA + enable_gqa=True (no manual expand; lets the kernel handle GQA)
  3. flash_attn_with_kvcache (production decode kernel from the flash-attn
     package, has split-K)

For each we report achieved bandwidth as bytes-moved / wall time, against
the GQA-shaped cache footprint (the bytes any *correct* kernel must touch).
If (2) and (3) hit ~213 GB/s, hypothesis A wins. If they're also ~60 GB/s,
hypothesis B wins.
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
RESULTS = Path(__file__).parent / "results" / "decode_kernels.json"

N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
GQA_GROUP = N_HEADS // N_KV_HEADS

WARMUP = 5
ITERS = 50  # decode steps are fast; more iters for stable timing


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


def kv_cache_bytes(B: int, S: int) -> int:
    """Bytes a correct kernel must read from DRAM (GQA-shaped cache)."""
    bytes_q = B * N_HEADS * HEAD_DIM * BYTES_PER_ELEM
    bytes_kv = 2 * B * N_KV_HEADS * S * HEAD_DIM * BYTES_PER_ELEM
    bytes_o = B * N_HEADS * HEAD_DIM * BYTES_PER_ELEM
    return bytes_q + bytes_kv + bytes_o


# --- Variant 1: manual repeat_interleave + SDPA (current baseline) ---
def make_sdpa_manual(B: int, S: int):
    q = torch.randn(B, N_HEADS, 1, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, N_KV_HEADS, S, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, N_KV_HEADS, S, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    # IMPORTANT: pre-expand once, hold the expanded tensors so we time only
    # the SDPA kernel itself, not the repeat. The DRAM cost of *holding*
    # the expanded tensor still affects the kernel's reads.
    k_full = k.repeat_interleave(GQA_GROUP, dim=1).contiguous()
    v_full = v.repeat_interleave(GQA_GROUP, dim=1).contiguous()
    return lambda: F.scaled_dot_product_attention(q, k_full, v_full)


# --- Variant 2: native GQA via SDPA enable_gqa=True ---
def make_sdpa_native_gqa(B: int, S: int):
    q = torch.randn(B, N_HEADS, 1, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, N_KV_HEADS, S, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, N_KV_HEADS, S, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    return lambda: F.scaled_dot_product_attention(q, k, v, enable_gqa=True)


# --- Variant 3: flash_attn_with_kvcache (production decode kernel) ---
def make_flash_attn(B: int, S: int):
    from flash_attn import flash_attn_with_kvcache
    # flash-attn uses (B, T, N, H) layout, not (B, N, T, H)
    q = torch.randn(B, 1, N_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k_cache = torch.randn(B, S, N_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v_cache = torch.randn(B, S, N_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    return lambda: flash_attn_with_kvcache(q, k_cache, v_cache)


VARIANTS = [
    ("sdpa_manual_expand",  make_sdpa_manual),
    ("sdpa_native_gqa",     make_sdpa_native_gqa),
    ("flash_attn_kvcache",  make_flash_attn),
]


def measure(variant_name, factory, B: int, S: int) -> dict:
    try:
        step = factory(B, S)
    except Exception as e:
        return {"variant": variant_name, "B": B, "S": S,
                "error": f"setup: {type(e).__name__}: {e}"}
    try:
        sec = time_fn(step)
    except Exception as e:
        return {"variant": variant_name, "B": B, "S": S,
                "error": f"run: {type(e).__name__}: {e}"}
    bm = kv_cache_bytes(B, S)
    # FLOPs of one decode step: 4*B*N*S*H (Q@K + Attn@V at full N expansion)
    flops = 4 * B * N_HEADS * S * HEAD_DIM
    return {
        "variant": variant_name,
        "B": B, "S": S,
        "seconds": sec,
        "bytes_moved_gqa": bm,
        "flops": flops,
        "gb_per_s": bm / sec / 1e9,
        "tflops_per_s": flops / sec / 1e12,
        "intensity_flops_per_byte": flops / bm,
    }


def main():
    assert torch.cuda.is_available()
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # Single fixed shape that's representative of vLLM-style decode mix.
    test_points = [
        (1,   4096),
        (8,   4096),
        (32,  4096),
        (128, 4096),
        (1,   16384),
        (8,   16384),
    ]

    results = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "timestamp": time.time(),
        "peak_bw_gb_per_s": 213.0,  # measured in ../roofline/
        "variants": {},
    }

    for variant_name, factory in VARIANTS:
        print(f"\n=== {variant_name} ===")
        rows = []
        for B, S in test_points:
            r = measure(variant_name, factory, B, S)
            rows.append(r)
            if "error" in r:
                print(f"  B={B:4d} S={S:5d}  ERROR: {r['error']}")
            else:
                print(
                    f"  B={B:4d} S={S:5d}  "
                    f"{r['gb_per_s']:6.1f} GB/s  "
                    f"({100*r['gb_per_s']/213:5.1f}% of peak)  "
                    f"{r['tflops_per_s']:6.3f} TF/s  "
                    f"t={r['seconds']*1e3:7.3f} ms"
                )
        results["variants"][variant_name] = rows

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
