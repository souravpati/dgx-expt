"""
Marlin W4A16 kernel factory for the roofline experiment.

Marlin is the production W4A16 GEMM used by vLLM for `--quantization gptq`
and `--quantization awq`. It runs on tensor cores, unlike PyTorch's
built-in `aten._weight_int4pack_mm`, so it's the right kernel to put on
the roofline if we want a realistic Q4 inference comparison.

Wire-up reference: vllm/tests/kernels/quantization/test_marlin_gemm.py
and vllm/benchmarks/kernels/benchmark_marlin.py at v0.19.1.
"""

from __future__ import annotations

from typing import Callable

import torch

DEVICE = "cuda"


def is_available() -> tuple[bool, str]:
    try:
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_make_workspace_new,  # noqa: F401
            marlin_permute_scales,      # noqa: F401
        )
        from vllm.scalar_type import scalar_types  # noqa: F401
        # gptq_marlin_repack is a C++ op on _custom_ops, not in marlin_utils.
        if not hasattr(ops, "gptq_marlin_repack"):
            return False, "vllm._custom_ops.gptq_marlin_repack missing"
        if not hasattr(ops, "marlin_gemm"):
            return False, "vllm._custom_ops.marlin_gemm missing"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    return True, "ok"


def _pack_along_k(q: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
    """Pack int values along axis 0 (K axis).

    Input:  (K, N) integer tensor with values in [0, 2**num_bits).
    Output: (K // pack_factor, N) int32, where pack_factor = 32 // num_bits.
    """
    pack_factor = 32 // num_bits
    K, N = q.shape
    assert K % pack_factor == 0, f"K={K} must be divisible by {pack_factor}"
    mask = (1 << num_bits) - 1
    q = (q.to(torch.int32) & mask).reshape(K // pack_factor, pack_factor, N)
    shifts = (
        torch.arange(pack_factor, device=q.device, dtype=torch.int32) * num_bits
    ).view(1, pack_factor, 1)
    return (q << shifts).sum(dim=1).to(torch.int32)


def make_marlin(group_size: int = 128) -> Callable[[int, int, int], Callable[[], object]]:
    """Returns a factory((B, D, F) -> step()) for W4A16 Marlin GEMM.

    Maps to a matmul of shape (M=B, K=D) @ (K=D, N=F).
    """
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
        marlin_permute_scales,
    )
    from vllm.scalar_type import scalar_types
    # C++ op on _custom_ops:
    gptq_marlin_repack = ops.gptq_marlin_repack

    def factory(B: int, D: int, F: int) -> Callable[[], object]:
        M, K, N = B, D, F
        # Activation -- bf16 for parity with our other sweeps.
        a = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16)

        # Random "quantized" weight of shape (K, N), values in [0, 15].
        q = torch.randint(0, 16, (K, N), device=DEVICE, dtype=torch.int32)

        # Per-group scales: shape (K/group_size, N).
        scales = (
            torch.rand(K // group_size, N, device=DEVICE, dtype=torch.bfloat16) * 0.1
            + 0.01
        )

        # Pack 4-bit values along the K axis -> (K/8, N) int32. The
        # vllm helper `pack_cols` packs along N, which is the wrong axis
        # for marlin's repack -- so we do it ourselves.
        packed_q = _pack_along_k(q, num_bits=4)

        # Marlin's GPTQ-format repack expects a permutation tensor (empty
        # means "no permutation"). Repack into Marlin's tile layout.
        perm = torch.empty(0, device=DEVICE, dtype=torch.int32)
        b_q_weight = gptq_marlin_repack(packed_q, perm, K, N, 4)

        # Permute scales into Marlin's expected memory layout.
        b_scales = marlin_permute_scales(scales, K, N, group_size)

        # Workspace -- allocated once, reused across steps.
        workspace = marlin_make_workspace_new(torch.device(DEVICE))

        # Type tag for the weight format. uint4b8 = "4-bit unsigned, with
        # an additive bias of 8" -- the standard GPTQ format.
        b_q_type = scalar_types.uint4b8

        def step():
            return ops.marlin_gemm(
                a=a,
                c=None,
                b_q_weight=b_q_weight,
                b_bias=None,
                b_scales=b_scales,
                a_scales=None,
                global_scale=None,
                b_zeros=None,
                g_idx=None,
                perm=perm,
                workspace=workspace,
                b_q_type=b_q_type,
                size_m=M,
                size_n=N,
                size_k=K,
                is_k_full=True,
                use_atomic_add=False,
                use_fp32_reduce=False,
                is_zp_float=False,
            )

        return step

    return factory


def extra_bytes(B: int, D: int, F: int, group_size: int = 128) -> int:
    """Scales -- (K/group_size) * N elements of bf16 (2 bytes each)."""
    return (D // group_size) * F * 2
