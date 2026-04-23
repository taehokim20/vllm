# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE BGMV CUDA kernel Python wrappers.

Thin wrappers around the C++ ops in csrc/lora/torch_bindings.cpp (_lora_C).
Weight pointer population is handled by the caller (punica_gpu.py) to
enable caching across calls.

The kernel accepts a lora_stride parameter so it can work directly with
the original vLLM weight layout [max_loras, num_experts, rank, feat]
without needing a transposed copy.
"""

import torch

import vllm._lora_C  # noqa: F401 — triggers torch op registration
from vllm.utils.torch_utils import direct_register_custom_op


# ---------------------------------------------------------------------------
# Vectorized w_ptr population (works with original layout)
# ---------------------------------------------------------------------------
_expert_arange_cache: dict[tuple[int, torch.device], torch.Tensor] = {}


def _fill_w_ptr_vectorized(
    w_ptr_row: torch.Tensor,
    w: torch.Tensor,
    num_experts: int,
) -> None:
    """Fill w_ptr_row[0:num_experts] with data_ptr for each expert.

    Works with the original weight layout [max_loras, num_experts, rank, feat].
    Each w_ptr entry points to the start of that expert's data within lora_id=0,
    i.e. w[0, expert_id, 0, 0]. The kernel uses lora_stride to jump between
    lora slots.
    """
    # w shape: [max_loras, num_experts, rank, feat]
    # Expert stride: distance between consecutive experts in elements
    # = rank * feat (stride along dim 1)
    base_ptr = w.data_ptr()
    expert_stride_bytes = w.stride(1) * w.element_size()
    cache_key = (num_experts, w.device)
    arange = _expert_arange_cache.get(cache_key)
    if arange is None or arange.size(0) < num_experts:
        arange = torch.arange(num_experts, dtype=torch.int64,
                              device=w.device)
        _expert_arange_cache[cache_key] = arange
    torch.mul(arange[:num_experts], expert_stride_bytes,
              out=w_ptr_row[:num_experts])
    w_ptr_row[:num_experts].add_(base_ptr)


def _get_lora_stride(w: torch.Tensor) -> int:
    """Return the lora stride in elements for the original weight layout.

    w shape: [max_loras, num_experts, rank, feat]
    lora_stride = stride along dim 0 = num_experts * rank * feat
    """
    return w.stride(0)


# ---------------------------------------------------------------------------
# Thin kernel wrappers — w_ptr is pre-populated by the caller
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _moe_cuda_shrink(
    y: torch.Tensor,
    x: torch.Tensor,
    w_ptr: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    lora_indices: torch.Tensor,
    lora_stride: int,
) -> None:
    """MoE LoRA shrink. w_ptr must already be populated."""
    torch.ops._lora_C.dispatch_moe_shrink(
        y, x, w_ptr, sorted_token_ids, expert_ids, lora_indices, lora_stride)


def _moe_cuda_shrink_fake(
    y: torch.Tensor, x: torch.Tensor, w_ptr: torch.Tensor,
    sorted_token_ids: torch.Tensor, expert_ids: torch.Tensor,
    lora_indices: torch.Tensor, lora_stride: int,
) -> None:
    return


@torch.inference_mode()
def _moe_cuda_expand(
    y: torch.Tensor,
    x: torch.Tensor,
    w_ptr: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    lora_indices: torch.Tensor,
    slice_start_loc: torch.Tensor,
    output_slices: list[int],
    lora_stride: int,
) -> None:
    """MoE LoRA expand. w_ptr and slice_start_loc must already be populated."""
    torch.ops._lora_C.dispatch_moe_expand(
        y, x, w_ptr, sorted_token_ids, expert_ids, topk_weights,
        lora_indices, slice_start_loc, output_slices, lora_stride)


def _moe_cuda_expand_fake(
    y: torch.Tensor, x: torch.Tensor, w_ptr: torch.Tensor,
    sorted_token_ids: torch.Tensor, expert_ids: torch.Tensor,
    topk_weights: torch.Tensor, lora_indices: torch.Tensor,
    slice_start_loc: torch.Tensor, output_slices: list[int],
    lora_stride: int,
) -> None:
    return


# ---------------------------------------------------------------------------
# Register custom ops
# ---------------------------------------------------------------------------
try:
    direct_register_custom_op(
        op_name="moe_cuda_shrink", op_func=_moe_cuda_shrink,
        mutates_args=["y"], fake_impl=_moe_cuda_shrink_fake)
    moe_cuda_shrink = torch.ops.vllm.moe_cuda_shrink
except AttributeError:
    moe_cuda_shrink = _moe_cuda_shrink

try:
    direct_register_custom_op(
        op_name="moe_cuda_expand", op_func=_moe_cuda_expand,
        mutates_args=["y"], fake_impl=_moe_cuda_expand_fake)
    moe_cuda_expand = torch.ops.vllm.moe_cuda_expand
except AttributeError:
    moe_cuda_expand = _moe_cuda_expand
