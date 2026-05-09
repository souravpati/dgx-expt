"""
Probe what W4A16 / Marlin kernels exist in this container.

Goal: figure out which import path + call signature actually works on
Blackwell sm_121 in vllm/vllm-openai:v0.19.1-cu130, so we can write the
real integration with confidence.

Run inside the container (same way as measure_roofline.py).
"""

from __future__ import annotations

import importlib
import inspect
import traceback

import torch

print(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device={torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability(0)}")
print()


def probe_module(name: str) -> object | None:
    try:
        m = importlib.import_module(name)
        print(f"  [OK] {name}  ({getattr(m, '__file__', '?')})")
        return m
    except Exception as e:
        print(f"  [--] {name}: {type(e).__name__}: {e}")
        return None


print("=== Top-level imports ===")
vllm = probe_module("vllm")
vllm_ops = probe_module("vllm._custom_ops")
vllm_marlin_utils = probe_module(
    "vllm.model_executor.layers.quantization.utils.marlin_utils"
)
torchao = probe_module("torchao")
torchao_marlin = probe_module("torchao.quantization.marlin_qqq")
print()

print("=== Marlin-related ops on vllm._custom_ops ===")
if vllm_ops is not None:
    matches = sorted(
        a for a in dir(vllm_ops)
        if any(k in a.lower() for k in ("marlin", "int4", "w4", "gptq", "awq"))
    )
    for name in matches:
        obj = getattr(vllm_ops, name)
        sig = ""
        try:
            sig = str(inspect.signature(obj))
        except (TypeError, ValueError):
            pass
        print(f"  {name}{sig}")
    if not matches:
        print("  (none found)")
print()

print("=== marlin_utils helpers ===")
if vllm_marlin_utils is not None:
    interesting = sorted(
        a for a in dir(vllm_marlin_utils)
        if not a.startswith("_") and (
            "quantize" in a.lower()
            or "workspace" in a.lower()
            or "permute" in a.lower()
            or "pack" in a.lower()
        )
    )
    for name in interesting:
        obj = getattr(vllm_marlin_utils, name)
        sig = ""
        try:
            sig = str(inspect.signature(obj))
        except (TypeError, ValueError):
            pass
        print(f"  {name}{sig}")
print()

print("=== aten ops with marlin/int4 in name ===")
try:
    aten_ops = [
        op for op in dir(torch.ops.aten)
        if any(k in op.lower() for k in ("marlin", "int4", "w4"))
    ]
    for name in aten_ops:
        print(f"  aten.{name}")
except Exception as e:
    print(f"  failed: {e}")
print()

print("=== Try a tiny gptq_marlin_gemm call ===")
if vllm_ops is not None and hasattr(vllm_ops, "gptq_marlin_gemm") and vllm_marlin_utils is not None:
    M, N, K = 256, 4096, 4096
    try:
        # Build a quantized weight using vLLM's helpers
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_quantize, marlin_make_workspace,
        )
        from vllm.scalar_type import scalar_types  # type for b_q_type

        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        w = torch.randn(K, N, device="cuda", dtype=torch.float16)

        # marlin_quantize signature varies by version; print and try
        print(f"  marlin_quantize sig: {inspect.signature(marlin_quantize)}")
        print(f"  marlin_make_workspace sig: {inspect.signature(marlin_make_workspace)}")
    except Exception:
        traceback.print_exc()
else:
    print("  prerequisites missing -- skipping live call")
print()

print("=== Done ===")
