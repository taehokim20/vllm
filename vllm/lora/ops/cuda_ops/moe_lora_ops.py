# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE BGMV CUDA kernel Python wrappers.

Thin wrappers around the C++ ops in csrc/lora/torch_bindings.cpp (_lora_C).
Weight permutation and w_ptr population are handled by the caller
(punica_gpu.py) to enable caching across calls.
"""

import torch

import vllm._lora_C  # noqa: F401 — triggers torch op registration
from vllm.utils.torch_utils import direct_register_custom_op


# ---------------------------------------------------------------------------
# Weight permutation with pre-allocated buffers
# ---------------------------------------------------------------------------
_permuted_weight_buffers: dict[int, tuple[int, torch.Tensor]] = {}


def _get_permuted_weight(w: torch.Tensor) -> torch.Tensor:
    """Return w permuted to [num_experts, max_loras, rank, feat_in]."""
    key = w.data_ptr()
    ver = w._version
    entry = _permuted_weight_buffers.get(key)
    if entry is not None and entry[0] == ver:
        return entry[1]
    if entry is not None:
        buf = entry[1]
    else:
        buf = torch.empty(
            (w.shape[1], w.shape[0], w.shape[2], w.shape[3]),
            dtype=w.dtype, device=w.device)
    buf.copy_(w.permute(1, 0, 2, 3))
    _permuted_weight_buffers[key] = (ver, buf)
    return buf


# ---------------------------------------------------------------------------
# Vectorized w_ptr population
# ---------------------------------------------------------------------------
_expert_arange_cache: dict[tuple[int, torch.device], torch.Tensor] = {}


def _fill_w_ptr_vectorized(
    w_ptr_row: torch.Tensor,
    w_perm: torch.Tensor,
    num_experts: int,
) -> None:
    """Fill w_ptr_row[0:num_experts] with data_ptr for each expert."""
    base_ptr = w_perm.data_ptr()
    expert_stride_bytes = w_perm.stride(0) * w_perm.element_size()
    cache_key = (num_experts, w_perm.device)
    arange = _expert_arange_cache.get(cache_key)
    if arange is None or arange.size(0) < num_experts:
        arange = torch.arange(num_experts, dtype=torch.int64,
                              device=w_perm.device)
        _expert_arange_cache[cache_key] = arange
    torch.mul(arange[:num_experts], expert_stride_bytes,
              out=w_ptr_row[:num_experts])
    w_ptr_row[:num_experts].add_(base_ptr)


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
) -> None:
    """MoE LoRA shrink. w_ptr must already be populated."""
    torch.ops._lora_C.dispatch_moe_shrink(
        y, x, w_ptr, sorted_token_ids, expert_ids, lora_indices)


def _moe_cuda_shrink_fake(
    y: torch.Tensor, x: torch.Tensor, w_ptr: torch.Tensor,
    sorted_token_ids: torch.Tensor, expert_ids: torch.Tensor,
    lora_indices: torch.Tensor,
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
) -> None:
    """MoE LoRA expand. w_ptr and slice_start_loc must already be populated."""
    torch.ops._lora_C.dispatch_moe_expand(
        y, x, w_ptr, sorted_token_ids, expert_ids, topk_weights,
        lora_indices, slice_start_loc, output_slices)


def _moe_cuda_expand_fake(
    y: torch.Tensor, x: torch.Tensor, w_ptr: torch.Tensor,
    sorted_token_ids: torch.Tensor, expert_ids: torch.Tensor,
    topk_weights: torch.Tensor, lora_indices: torch.Tensor,
    slice_start_loc: torch.Tensor, output_slices: list[int],
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
