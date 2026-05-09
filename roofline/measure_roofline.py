"""
Empirical roofline for DGX Spark (GB10).

Measures, for each precision (BF16 / FP16 / FP8 / INT8 / W4A16):
  1. Peak compute ceiling   -- one big square matmul (8192^3)
  2. B-sweep                -- (B, D) @ (D, F) over many B
And once globally:
  3. Peak DRAM bandwidth    -- large device-to-device copy

For each sweep point we record achieved FLOPs/s and arithmetic
intensity, where intensity uses the *actual* per-element byte cost
of the inputs/weights/output for that precision. Bytes-moved is
the data the kernel must touch through DRAM, so this captures the
real benefit of low-precision weights.

Run:
    python measure_roofline.py
    python plot_roofline.py
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

DEVICE = "cuda"
RESULTS = Path(__file__).parent / "results" / "roofline.json"
WARMUP_ITERS = 5
TIMED_ITERS = 20


def _sync():
    torch.cuda.synchronize()


def time_kernel(fn, warmup=WARMUP_ITERS, iters=TIMED_ITERS) -> float:
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


def measure_peak_bandwidth(nbytes: int = 1 << 30) -> dict:
    """1 GiB d2d copy. Reads + writes = 2*nbytes traffic."""
    src = torch.empty(nbytes, device=DEVICE, dtype=torch.uint8)
    dst = torch.empty(nbytes, device=DEVICE, dtype=torch.uint8)
    src.fill_(1)
    sec = time_kernel(lambda: dst.copy_(src))
    bytes_moved = 2 * nbytes
    return {
        "nbytes": nbytes,
        "seconds": sec,
        "bytes_per_s": bytes_moved / sec,
        "gb_per_s": bytes_moved / sec / 1e9,
    }


# ---------------------------------------------------------------------------
# Kernel configs: one entry per precision.
#
# Each KernelSpec describes how to build a one-shot matmul step for given
# (B, D, F), the per-element byte cost on inputs/weights/outputs, and a
# label.  Sweeping then becomes uniform across precisions.
# ---------------------------------------------------------------------------


@dataclass
class KernelSpec:
    name: str
    bytes_in: float       # bytes per activation element
    bytes_w: float        # bytes per weight element
    bytes_out: float      # bytes per output element
    make: Callable[[int, int, int], Callable[[], object]]
    extra_bytes: Callable[[int, int, int], int] = field(
        default_factory=lambda: lambda B, D, F: 0
    )
    min_B: int = 1        # kernels with size constraints can bump this


def bytes_moved(spec: KernelSpec, B: int, D: int, F: int) -> float:
    return (
        spec.bytes_in * B * D
        + spec.bytes_w * D * F
        + spec.bytes_out * B * F
        + spec.extra_bytes(B, D, F)
    )


# --- BF16 / FP16 (vanilla torch.matmul) ---


def _make_torch_matmul(dtype):
    def factory(B, D, F):
        x = torch.randn(B, D, device=DEVICE, dtype=dtype)
        w = torch.randn(D, F, device=DEVICE, dtype=dtype)
        out = torch.empty(B, F, device=DEVICE, dtype=dtype)
        return lambda: torch.matmul(x, w, out=out)
    return factory


# --- FP8 e4m3 (torch._scaled_mm) ---


def _make_fp8():
    """
    _scaled_mm wants:
      - a: (M, K) FP8 row-major
      - b: (K, N) FP8 column-major  (so we transpose-contiguous-transpose)
      - scale_a, scale_b: scalar fp32 tensors
      - out_dtype: bf16/fp16
    """
    def factory(B, D, F):
        x = torch.randn(B, D, device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn(D, F, device=DEVICE, dtype=torch.bfloat16)
        x_fp8 = x.to(torch.float8_e4m3fn)
        # column-major (K, N) layout for b
        w_fp8 = w.to(torch.float8_e4m3fn).t().contiguous().t()
        scale_a = torch.tensor(1.0, device=DEVICE, dtype=torch.float32)
        scale_b = torch.tensor(1.0, device=DEVICE, dtype=torch.float32)
        return lambda: torch._scaled_mm(
            x_fp8, w_fp8,
            scale_a=scale_a, scale_b=scale_b,
            out_dtype=torch.bfloat16,
        )
    return factory


# --- INT8 (torch._int_mm) ---


def _make_int8():
    """
    _int_mm wants:
      - a: (M, K) int8 row-major
      - b: (K, N) int8, K must be a multiple of 8 and b column-major
      - returns int32
    """
    def factory(B, D, F):
        x = torch.randint(-127, 128, (B, D), device=DEVICE, dtype=torch.int8)
        w = torch.randint(-127, 128, (D, F), device=DEVICE, dtype=torch.int8)
        w = w.t().contiguous().t()  # column-major
        return lambda: torch._int_mm(x, w)
    return factory


# --- W4A16 (4-bit weight, 16-bit activation) via aten._weight_int4pack_mm ---


def _make_w4a16(group_size: int = 128, inner_k_tiles: int = 8):
    """
    Llama-style weight-only 4-bit quant. Activation is bf16, weights
    pack two int4 values per byte, plus per-group bf16 scales+zeros.

    On CUDA, aten._convert_weight_to_int4pack expects a pre-packed
    uint8 tensor of shape (N, K/2) where each byte holds two int4
    values. Then aten._weight_int4pack_mm runs the actual matmul.
    """
    def factory(B, D, F):
        x = torch.randn(B, D, device=DEVICE, dtype=torch.bfloat16)
        # weight (N=F, K=D) int values 0..15
        w_int = torch.randint(0, 16, (F, D), device=DEVICE, dtype=torch.uint8)
        # Pack two 4-bit values into each byte → (F, D/2) uint8
        lo = w_int[:, 0::2] & 0xF
        hi = (w_int[:, 1::2] & 0xF) << 4
        w_packed_byte = (lo | hi).contiguous()
        w_pack = torch.ops.aten._convert_weight_to_int4pack(
            w_packed_byte, inner_k_tiles
        )
        n_groups = D // group_size
        scales_and_zeros = torch.randn(
            n_groups, F, 2, device=DEVICE, dtype=torch.bfloat16
        )
        return lambda: torch.ops.aten._weight_int4pack_mm(
            x, w_pack, group_size, scales_and_zeros
        )

    def extra(B, D, F):
        # scales+zeros: (D/group_size) * F * 2 elems * 2 bytes (bf16)
        return (D // group_size) * F * 2 * 2

    return factory, extra


# Build the registry of precisions to sweep.
_w4_factory, _w4_extra = _make_w4a16()

KERNEL_SPECS: list[KernelSpec] = [
    KernelSpec("bf16",     2, 2,   2, _make_torch_matmul(torch.bfloat16)),
    KernelSpec("fp16",     2, 2,   2, _make_torch_matmul(torch.float16)),
    KernelSpec("fp8_e4m3", 1, 1,   2, _make_fp8()),                       # out bf16
    KernelSpec("int8",     1, 1,   4, _make_int8(), min_B=32),            # _int_mm needs M>16
    KernelSpec("w4a16",    2, 0.5, 2, _w4_factory, _w4_extra),
]

# Optionally append Marlin (vLLM's tensor-core W4A16 kernel) -- only if the
# vLLM bindings are importable.
try:
    import marlin_kernel
    available, reason = marlin_kernel.is_available()
    if available:
        KERNEL_SPECS.append(KernelSpec(
            "w4a16_marlin", 2, 0.5, 2,
            marlin_kernel.make_marlin(),
            lambda B, D, F: marlin_kernel.extra_bytes(B, D, F),
        ))
        print("Marlin kernel available -- will sweep w4a16_marlin")
    else:
        print(f"Marlin not available ({reason}) -- skipping")
except Exception as e:
    print(f"Marlin import failed ({type(e).__name__}: {e}) -- skipping")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def measure_peak(spec: KernelSpec, n: int = 8192) -> dict | None:
    try:
        step = spec.make(n, n, n)
        sec = time_kernel(step)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    flops = 2 * n * n * n
    return {
        "kernel": spec.name,
        "shape": [n, n, n],
        "seconds": sec,
        "flops_per_s": flops / sec,
        "tflops_per_s": flops / sec / 1e12,
    }


def matmul_point(spec: KernelSpec, B: int, D: int, F: int) -> dict:
    step = spec.make(B, D, F)
    sec = time_kernel(step)
    flops = 2 * B * D * F
    bm = bytes_moved(spec, B, D, F)
    return {
        "B": B, "D": D, "F": F,
        "kernel": spec.name,
        "seconds": sec,
        "flops_per_s": flops / sec,
        "bytes_moved": bm,
        "intensity_flops_per_byte": flops / bm,
    }


def sweep_b(spec: KernelSpec, D: int, F: int) -> list[dict]:
    Bs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768,
          1024, 1536, 2048, 3072, 4096, 6144, 8192]
    out = []
    for B in Bs:
        if B < spec.min_B:
            continue
        try:
            point = matmul_point(spec, B, D, F)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"    B={B} {spec.name}: {type(e).__name__}: {e}")
            break
        out.append(point)
        print(
            f"  B={B:5d} D={D} F={F} {spec.name:>9s}  "
            f"{point['flops_per_s']/1e12:7.2f} TF/s  "
            f"I={point['intensity_flops_per_byte']:8.2f} F/B"
        )
    return out


def main():
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"Device: {torch.cuda.get_device_name(0)}")

    results = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "timestamp": time.time(),
    }

    print("\n[1] Peak DRAM bandwidth (1 GiB d2d copy)")
    results["peak_bandwidth"] = measure_peak_bandwidth()
    print(f"  {results['peak_bandwidth']['gb_per_s']:.1f} GB/s")

    print("\n[2] Peak compute per precision (8192^3 matmul)")
    peaks = {}
    for spec in KERNEL_SPECS:
        peak = measure_peak(spec)
        peaks[spec.name] = peak
        if peak and "tflops_per_s" in peak:
            print(f"  {spec.name:>9s}: {peak['tflops_per_s']:7.1f} TF/s")
        else:
            print(f"  {spec.name:>9s}: SKIPPED -- {peak.get('error') if peak else 'n/a'}")
    results["peak_compute"] = peaks

    # Keep legacy keys so the existing plotter/ASCII scripts still work.
    if "tflops_per_s" in (peaks.get("bf16") or {}):
        results["peak_compute_bf16"] = peaks["bf16"]
    if "tflops_per_s" in (peaks.get("fp16") or {}):
        results["peak_compute_fp16"] = peaks["fp16"]

    print("\n[3] Matmul B-sweeps")
    shapes = [
        ("D4096_F4096", 4096, 4096),
        ("D1024_F1024", 1024, 1024),
    ]
    sweeps = {}
    for shape_name, D, F in shapes:
        for spec in KERNEL_SPECS:
            label = f"{shape_name}_{spec.name}"
            print(f" sweep {label}")
            try:
                sweeps[label] = sweep_b(spec, D, F)
            except Exception as e:
                print(f"    {label} crashed: {e}")
                traceback.print_exc()
                sweeps[label] = []
    results["sweeps"] = sweeps

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    main()
