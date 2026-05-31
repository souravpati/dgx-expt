"""
Attention arithmetic intensity on DGX Spark (GB10).

Verifies the Chapter-4 claims from the JAX scaling book:
  - Prefill self-attention intensity is O(T) -- climbs the roofline
    diagonal as sequence length grows, eventually saturating compute.
  - Decode self-attention intensity is N/K (the GQA ratio) -- a
    constant independent of S and B, sitting deep in the
    bandwidth-bound region.

Shapes match Llama 3.1 8B's attention block (one layer):
  N=32 query heads, K=8 KV heads (GQA), H=128 head dim, D=N*H=4096.

We time torch SDPA (flash-attn under the hood on Blackwell). FLOPs are
counted analytically using the standard 4*B*N*T_q*T_kv*H formula
(QK^T + Attn@V, no softmax). Bytes-moved is the minimum traffic any
kernel must touch through DRAM: Q + K + V + O, where K/V use K (not N)
heads because the KV cache is GQA-shaped on disk.

Run:
    python measure_attention.py
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
BYTES_PER_ELEM = 2  # bf16
RESULTS = Path(__file__).parent / "results" / "attention.json"

# Llama 3.1 8B attention shape (per layer)
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
GQA_GROUP = N_HEADS // N_KV_HEADS  # 4
D_MODEL = N_HEADS * HEAD_DIM       # 4096

WARMUP = 5
ITERS = 20


def _sync():
    torch.cuda.synchronize()


def time_fn(fn, warmup=WARMUP, iters=ITERS) -> float:
    """Median seconds per call, timed with CUDA events."""
    for _ in range(warmup):
        fn()
    _sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        _sync()
        times.append(start.elapsed_time(end) / 1e3)
    times.sort()
    return times[len(times) // 2]


def attention_step(B: int, T_q: int, T_kv: int, causal: bool):
    """Allocate Q/K/V and return a callable that runs SDPA once.

    Uses SDPA's native GQA support (`enable_gqa=True`) so the kernel
    only reads the GQA-shaped cache from DRAM. Manual repeat_interleave
    would materialize a N/K-expanded tensor and inflate bytes-moved
    by N/K (verified in compare_decode_kernels.py).
    """
    q = torch.randn(B, N_HEADS, T_q, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, N_KV_HEADS, T_kv, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, N_KV_HEADS, T_kv, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    return lambda: F.scaled_dot_product_attention(
        q, k, v, is_causal=causal, enable_gqa=True
    )


def flops_bytes(B: int, T_q: int, T_kv: int, causal: bool) -> tuple[int, int]:
    """
    FLOPs counts Q@K^T + Attn@V at the full N-head expansion (this is
    what the kernel actually computes after GQA repeat). For causal
    masking, flash-attention skips the upper triangle, so useful work
    is half. Bytes-moved counts the *cache footprint*: K and V are
    stored at K_HEADS, not N_HEADS, so that's what DRAM has to deliver.
    """
    flops = 4 * B * N_HEADS * T_q * T_kv * HEAD_DIM
    if causal:
        flops //= 2
    bytes_q = B * N_HEADS * T_q * HEAD_DIM * BYTES_PER_ELEM
    bytes_kv = 2 * B * N_KV_HEADS * T_kv * HEAD_DIM * BYTES_PER_ELEM
    bytes_o = B * N_HEADS * T_q * HEAD_DIM * BYTES_PER_ELEM
    return flops, bytes_q + bytes_kv + bytes_o


def measure_point(B: int, T_q: int, T_kv: int, causal: bool, label: str) -> dict:
    step = attention_step(B, T_q, T_kv, causal)
    sec = time_fn(step)
    flops, bm = flops_bytes(B, T_q, T_kv, causal)
    point = {
        "label": label,
        "B": B,
        "T_q": T_q,
        "T_kv": T_kv,
        "causal": causal,
        "seconds": sec,
        "flops": flops,
        "bytes_moved": bm,
        "flops_per_s": flops / sec,
        "tflops_per_s": flops / sec / 1e12,
        "intensity_flops_per_byte": flops / bm,
    }
    print(
        f"  {label:>14s}  B={B:4d} T_q={T_q:5d} T_kv={T_kv:5d}  "
        f"{point['tflops_per_s']:6.2f} TF/s  "
        f"I={point['intensity_flops_per_byte']:8.2f} F/B  "
        f"t={sec*1e3:7.2f} ms"
    )
    return point


def sweep_prefill(B: int = 1) -> list[dict]:
    """Self-attention on a fresh prompt: T_q = T_kv = T, causal."""
    Ts = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    out = []
    for T in Ts:
        try:
            out.append(measure_point(B, T, T, causal=True, label="prefill"))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"    T={T} prefill: OOM, stopping sweep")
            break
        except Exception as e:
            print(f"    T={T} prefill: {type(e).__name__}: {e}")
            break
    return out


def sweep_decode_S(B: int = 1) -> list[dict]:
    """Decode at varying KV-cache depth: T_q=1, T_kv=S."""
    Ss = [128, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    out = []
    for S in Ss:
        try:
            out.append(measure_point(B, 1, S, causal=False, label="decode_S"))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"    S={S} decode: OOM, stopping sweep")
            break
        except Exception as e:
            print(f"    S={S} decode: {type(e).__name__}: {e}")
            break
    return out


def sweep_decode_B(S: int = 4096) -> list[dict]:
    """Decode at varying batch size, fixed KV depth: T_q=1, T_kv=S."""
    Bs = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    out = []
    for B in Bs:
        try:
            out.append(measure_point(B, 1, S, causal=False, label="decode_B"))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"    B={B} decode: OOM, stopping sweep")
            break
        except Exception as e:
            print(f"    B={B} decode: {type(e).__name__}: {e}")
            break
    return out


def main():
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Attention shape: N={N_HEADS} K={N_KV_HEADS} H={HEAD_DIM} D={D_MODEL}")
    print(f"Predicted decode intensity (N/K): {N_HEADS / N_KV_HEADS:.1f} F/B")

    results = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "timestamp": time.time(),
        "shape": {
            "N_HEADS": N_HEADS,
            "N_KV_HEADS": N_KV_HEADS,
            "HEAD_DIM": HEAD_DIM,
            "D_MODEL": D_MODEL,
            "dtype": "bfloat16",
        },
    }

    print("\n[1] Prefill sweep over T (B=1, causal, T_q=T_kv=T)")
    results["prefill_T"] = sweep_prefill(B=1)

    print("\n[2] Decode sweep over S (B=1, T_q=1, T_kv=S)")
    results["decode_S"] = sweep_decode_S(B=1)

    print("\n[3] Decode sweep over B (S=4096, T_q=1)")
    results["decode_B"] = sweep_decode_B(S=4096)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
