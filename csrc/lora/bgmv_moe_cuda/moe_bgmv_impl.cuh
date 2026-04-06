#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>

#include "vec_dtypes.cuh"
#include "kernel_config.h"

namespace cg = cooperative_groups;

// Dimension macros and forward declarations are in moe_bgmv_config.h.
// This file provides kernel implementations and instantiation macros.
//
// Only two kernels are needed: SHRINK (sliced) and EXPAND (sliced).
// W2 (single-slice) is just the sliced kernel with num_slices=1.


// ============================================================
// MoE BGMV Shrink Sliced Kernel
// Grid: (num_pairs, feat_out, num_slices)
// w_ptr is a flat [num_slices * num_experts] array of weight pointers.
// Index as w_ptr[slice_id * num_experts + expert_id].
//
// For W2 (single-slice): num_slices=1, num_experts=num_experts.
// For W13 (gate+up): num_slices=2 (or more), blockIdx.z = slice_id.
//
// num_pairs is in gridDim.x (limit ~2B) to avoid the 65535 gridDim.y limit.
// ============================================================
template <int feat_in, int feat_out, size_t vec_size, size_t X_copy_size,
          size_t W_copy_size, int tx, int ty,
          typename in_T, typename out_T, typename W_T>
__global__ void
moe_bgmv_shrink_sliced_kernel(out_T *__restrict__ Y,
                               const in_T *__restrict__ X,
                               W_T **__restrict__ w_ptr,
                               const int64_t *__restrict__ sorted_token_ids,
                               const int64_t *__restrict__ expert_ids,
                               const int64_t *__restrict__ lora_indices,
                               int64_t num_pairs,
                               int64_t num_experts,
                               int64_t num_tokens,
                               float scale) {
  int    slice_id = blockIdx.z;
  size_t pair_idx = blockIdx.x;
  size_t j        = blockIdx.y;

  int64_t token_idx = sorted_token_ids[pair_idx];
  if (token_idx < 0 || token_idx >= num_tokens) return;
  int64_t expert_id = expert_ids[pair_idx];
  int64_t lora_id   = lora_indices[token_idx];
  if (lora_id < 0) return;

  const W_T  *W     = w_ptr[slice_id * num_experts + expert_id] + (lora_id * feat_out + j) * feat_in;
  const in_T *X_tok = X + token_idx * feat_in;

  auto block = cg::this_thread_block();
  constexpr size_t num_pipeline_stages = 2;
  constexpr size_t tile_size = tx * ty * vec_size;
  __shared__ W_T  W_shared[num_pipeline_stages * tile_size];
  __shared__ in_T X_shared[num_pipeline_stages * tile_size];
  __shared__ float y_warpwise[ty];

  size_t W_shared_offset[num_pipeline_stages] = {0U, 1U * tile_size};
  size_t X_shared_offset[num_pipeline_stages] = {0U, 1U * tile_size};
  auto pipe = cuda::make_pipeline();

  pipe.producer_acquire();
  if (threadIdx.y * tx * vec_size < feat_in) {
    cuda::memcpy_async(W_shared + (threadIdx.y * tx + threadIdx.x) * vec_size,
                       W + (threadIdx.y * tx + threadIdx.x) * vec_size,
                       cuda::aligned_size_t<W_copy_size>(W_copy_size), pipe);
    cuda::memcpy_async(X_shared + (threadIdx.y * tx + threadIdx.x) * vec_size,
                       X_tok + (threadIdx.y * tx + threadIdx.x) * vec_size,
                       cuda::aligned_size_t<X_copy_size>(X_copy_size), pipe);
  }
  pipe.producer_commit();

  float y = 0.f;
  vec_t<in_T, vec_size> x_vec;
  vec_t<W_T, vec_size>  w_vec;
  size_t tile_idx, copy_idx, compute_idx;

#pragma unroll
  for (tile_idx = 1; tile_idx < (feat_in + tile_size - 1) / tile_size; ++tile_idx) {
    copy_idx = tile_idx % num_pipeline_stages;
    pipe.producer_acquire();
    if (tile_idx * tile_size + threadIdx.y * tx * vec_size < feat_in) {
      cuda::memcpy_async(W_shared + W_shared_offset[copy_idx] +
                             (threadIdx.y * tx + threadIdx.x) * vec_size,
                         W + tile_idx * tile_size +
                             (threadIdx.y * tx + threadIdx.x) * vec_size,
                         cuda::aligned_size_t<W_copy_size>(W_copy_size), pipe);
      cuda::memcpy_async(X_shared + X_shared_offset[copy_idx] +
                             (threadIdx.y * tx + threadIdx.x) * vec_size,
                         X_tok + tile_idx * tile_size +
                             (threadIdx.y * tx + threadIdx.x) * vec_size,
                         cuda::aligned_size_t<X_copy_size>(X_copy_size), pipe);
    }
    pipe.producer_commit();
    compute_idx = (tile_idx - 1) % num_pipeline_stages;
    pipe.consumer_wait();
    block.sync();
    x_vec.load(X_shared + X_shared_offset[compute_idx] +
               (threadIdx.y * tx + threadIdx.x) * vec_size);
    w_vec.load(W_shared + W_shared_offset[compute_idx] +
               (threadIdx.y * tx + threadIdx.x) * vec_size);
    float sum = 0.f;
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i)
      sum += float(w_vec[i]) * float(x_vec[i]) * scale;
#pragma unroll
    for (size_t offset = tx / 2; offset > 0; offset /= 2)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (threadIdx.x == 0) y_warpwise[threadIdx.y] = sum;
    block.sync();
#pragma unroll
    for (size_t i = 0; i < ty; ++i) y += y_warpwise[i];
    block.sync();
    pipe.consumer_release();
  }

  compute_idx = (tile_idx - 1) % num_pipeline_stages;
  pipe.consumer_wait();
  block.sync();
  x_vec.load(X_shared + X_shared_offset[compute_idx] +
             (threadIdx.y * tx + threadIdx.x) * vec_size);
  w_vec.load(W_shared + W_shared_offset[compute_idx] +
             (threadIdx.y * tx + threadIdx.x) * vec_size);
  {
    float sum = 0.f;
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i)
      sum += float(w_vec[i]) * float(x_vec[i]) * scale;
#pragma unroll
    for (size_t offset = tx / 2; offset > 0; offset /= 2)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (threadIdx.x == 0)
      y_warpwise[threadIdx.y] =
          ((tile_idx - 1) * tile_size + threadIdx.y * tx * vec_size < feat_in) ? sum : 0.f;
  }
  block.sync();
#pragma unroll
  for (size_t i = 0; i < ty; ++i) y += y_warpwise[i];
  block.sync();
  pipe.consumer_release();

  if (block.thread_rank() == 0)
    Y[slice_id * num_pairs * feat_out + pair_idx * feat_out + j] += static_cast<out_T>(y);
}


// ============================================================
// MoE BGMV Expand Sliced Kernel
// Grid: (num_pairs, feat_out / (ty*tz), num_slices)
// Y is float32 accumulation buffer; atomicAdd is safe.
// w_ptr is a flat [num_slices * num_experts] array of weight pointers.
// Index as w_ptr[slice_id * num_experts + expert_id].
//
// For W2 (single-slice): num_slices=1, slice_start_loc=[0],
//   total_feat_out=feat_out, current_feat_out=feat_out.
// For W13 (gate+up): num_slices=2+, blockIdx.z = slice_id.
//
// num_pairs is in gridDim.x (limit ~2B) to avoid the 65535 gridDim.y limit.
// ============================================================
template <int feat_in, int feat_out, size_t vec_size, int tx, int ty, int tz,
          typename in_T, typename W_T>
__global__ void
moe_bgmv_expand_sliced_kernel(float *__restrict__ Y,
                               const in_T *__restrict__ X,
                               W_T **__restrict__ w_ptr,
                               const int64_t *__restrict__ sorted_token_ids,
                               const int64_t *__restrict__ expert_ids,
                               const int64_t *__restrict__ lora_indices,
                               const float *__restrict__ topk_weights,
                               const int64_t *__restrict__ slice_start_loc,
                               int64_t num_pairs,
                               int64_t num_experts,
                               int64_t total_feat_out,
                               int32_t current_feat_out,
                               int64_t num_tokens,
                               float scale) {
  int    slice_id = blockIdx.z;
  size_t pair_idx = blockIdx.x;
  size_t tile_idx = blockIdx.y;

  int64_t token_idx  = sorted_token_ids[pair_idx];
  if (token_idx < 0 || token_idx >= num_tokens) return;
  int64_t expert_id  = expert_ids[pair_idx];
  int64_t lora_id    = lora_indices[token_idx];
  if (lora_id < 0) return;

  float   topk_w     = topk_weights[pair_idx];
  int64_t col_offset = slice_start_loc[slice_id];
  const W_T *W = w_ptr[slice_id * num_experts + expert_id] + lora_id * current_feat_out * feat_in;

  auto block = cg::this_thread_block();

  vec_t<in_T, vec_size> x_vec;
  x_vec.load(X + slice_id * num_pairs * feat_in + pair_idx * feat_in +
             threadIdx.x * vec_size);

  vec_t<W_T, vec_size> w_vec;
  w_vec.load(W + (tile_idx * tz * ty) * feat_in + block.thread_rank() * vec_size);

  float sum = 0.f;
#pragma unroll
  for (size_t i = 0; i < vec_size; ++i)
    sum += float(w_vec[i]) * float(x_vec[i]) * scale;

  cg::thread_block_tile<tx> g = cg::tiled_partition<tx>(block);
#pragma unroll
  for (size_t offset = tx / 2; offset > 0; offset /= 2)
    sum += g.shfl_down(sum, offset);
  sum = g.shfl(sum, 0);

  if (threadIdx.x == 0) {
    int out_col = col_offset + tile_idx * (tz * ty) + threadIdx.z * ty + threadIdx.y;
    atomicAdd(Y + token_idx * total_feat_out + out_col, sum * topk_w);
  }
}


// ============================================================
// Host-side dispatch wrappers
// ============================================================

// Shrink sliced: handles both W13 (num_slices>=2) and W2 (num_slices=1).
template <int feat_in, int feat_out, typename in_T, typename out_T, typename W_T>
void moe_bgmv_shrink_sliced(out_T *__restrict__ Y,
                             const in_T *__restrict__ X,
                             W_T **__restrict__ w_ptr,
                             const int64_t *__restrict__ sorted_token_ids,
                             const int64_t *__restrict__ expert_ids,
                             const int64_t *__restrict__ lora_indices,
                             int64_t num_pairs,
                             int64_t num_slices,
                             int64_t num_experts,
                             int64_t num_tokens,
                             float scale) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr size_t vec_size = MoeShrinkKernelConfig::vec_size;
  constexpr int cfg_tx = MoeShrinkKernelConfig::tx;
  constexpr int cfg_ty = MoeShrinkKernelConfig::ty;

  if constexpr (feat_in % (vec_size * cfg_tx) == 0) {
    dim3 nblks(num_pairs, feat_out, num_slices);
    dim3 nthrs(cfg_tx, cfg_ty);
    moe_bgmv_shrink_sliced_kernel<feat_in, feat_out, vec_size,
                                  vec_size * sizeof(in_T), vec_size * sizeof(W_T),
                                  cfg_tx, cfg_ty, in_T, out_T, W_T>
        <<<nblks, nthrs, 0, stream>>>(Y, X, w_ptr, sorted_token_ids,
                                      expert_ids, lora_indices, num_pairs, num_experts, num_tokens, scale);
  } else if constexpr (feat_in % (vec_size / 2 * cfg_tx) == 0) {
    constexpr size_t hv = vec_size / 2;
    dim3 nblks(num_pairs, feat_out, num_slices);
    dim3 nthrs(cfg_tx, cfg_ty);
    moe_bgmv_shrink_sliced_kernel<feat_in, feat_out, hv,
                                  hv * sizeof(in_T), hv * sizeof(W_T),
                                  cfg_tx, cfg_ty, in_T, out_T, W_T>
        <<<nblks, nthrs, 0, stream>>>(Y, X, w_ptr, sorted_token_ids,
                                      expert_ids, lora_indices, num_pairs, num_experts, num_tokens, scale);
  } else if constexpr (feat_in % (vec_size / 4 * cfg_tx) == 0) {
    constexpr size_t qv = vec_size / 4;
    dim3 nblks(num_pairs, feat_out, num_slices);
    dim3 nthrs(cfg_tx, cfg_ty);
    moe_bgmv_shrink_sliced_kernel<feat_in, feat_out, qv,
                                  qv * sizeof(in_T), qv * sizeof(W_T),
                                  cfg_tx, cfg_ty, in_T, out_T, W_T>
        <<<nblks, nthrs, 0, stream>>>(Y, X, w_ptr, sorted_token_ids,
                                      expert_ids, lora_indices, num_pairs, num_experts, num_tokens, scale);
  }
}

// Expand sliced: handles both W13 (num_slices>=2) and W2 (num_slices=1).
template <int feat_in, int feat_out, typename in_T, typename W_T>
void moe_bgmv_expand_sliced(float *__restrict__ Y,
                             const in_T *__restrict__ X,
                             W_T **__restrict__ w_ptr,
                             const int64_t *__restrict__ sorted_token_ids,
                             const int64_t *__restrict__ expert_ids,
                             const int64_t *__restrict__ lora_indices,
                             const float *__restrict__ topk_weights,
                             const int64_t *__restrict__ slice_start_loc,
                             int64_t num_pairs,
                             int64_t num_slices,
                             int64_t num_experts,
                             int64_t total_feat_out,
                             int32_t current_feat_out,
                             int64_t num_tokens,
                             float scale) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr size_t vec_size = MoeExpandKernelConfig::vec_size;
  constexpr int tz = MoeExpandKernelConfig::tz;
  static_assert(feat_in % vec_size == 0, "feat_in must be divisible by vec_size");
  constexpr int tx = feat_in / vec_size;

  if constexpr (32 % tx == 0 && feat_out % (32 / tx * tz) == 0) {
    constexpr int ty = 32 / tx;
    dim3 nblks(num_pairs, feat_out / (ty * tz), num_slices);
    dim3 nthrs(tx, ty, tz);
    moe_bgmv_expand_sliced_kernel<feat_in, feat_out, vec_size, tx, ty, tz, in_T, W_T>
        <<<nblks, nthrs, 0, stream>>>(Y, X, w_ptr, sorted_token_ids,
                                      expert_ids, lora_indices, topk_weights,
                                      slice_start_loc, num_pairs, num_experts,
                                      total_feat_out, current_feat_out, num_tokens, scale);
  } else if constexpr (16 % tx == 0 && feat_out % (16 / tx * tz) == 0) {
    constexpr int ty = 16 / tx;
    dim3 nblks(num_pairs, feat_out / (ty * tz), num_slices);
    dim3 nthrs(tx, ty, tz);
    moe_bgmv_expand_sliced_kernel<feat_in, feat_out, vec_size, tx, ty, tz, in_T, W_T>
        <<<nblks, nthrs, 0, stream>>>(Y, X, w_ptr, sorted_token_ids,
                                      expert_ids, lora_indices, topk_weights,
                                      slice_start_loc, num_pairs, num_experts,
                                      total_feat_out, current_feat_out, num_tokens, scale);
  } else if constexpr (8 % tx == 0 && feat_out % (8 / tx * tz) == 0) {
    constexpr int ty = 8 / tx;
    dim3 nblks(num_pairs, feat_out / (ty * tz), num_slices);
    dim3 nthrs(tx, ty, tz);
    moe_bgmv_expand_sliced_kernel<feat_in, feat_out, vec_size, tx, ty, tz, in_T, W_T>
        <<<nblks, nthrs, 0, stream>>>(Y, X, w_ptr, sorted_token_ids,
                                      expert_ids, lora_indices, topk_weights,
                                      slice_start_loc, num_pairs, num_experts,
                                      total_feat_out, current_feat_out, num_tokens, scale);
  }
}


// ============================================================
// Instantiation macros
// ============================================================

// Only two instantiation macros needed — both use the sliced kernel.
#define INST_MOE_BGMV_SHRINK_SLICED(feat_in, feat_out, in_T, out_T, W_T)       \
  template void moe_bgmv_shrink_sliced<feat_in, feat_out, in_T, out_T, W_T>(   \
      out_T*, const in_T*, W_T**, const int64_t*, const int64_t*,              \
      const int64_t*, int64_t, int64_t, int64_t, int64_t, float);

#define INST_MOE_BGMV_EXPAND_SLICED(feat_in, feat_out, in_T, W_T)              \
  template void moe_bgmv_expand_sliced<feat_in, feat_out, in_T, W_T>(          \
      float*, const in_T*, W_T**, const int64_t*, const int64_t*,              \
      const int64_t*, const float*, const int64_t*,                            \
      int64_t, int64_t, int64_t, int64_t, int32_t, int64_t, float);

// Convenience: instantiate both shrink and expand for a (narrow, wide) pair.
#define INST_MOE_BGMV_TWOSIDE(in_T, out_T, W_T, narrow, wide)                  \
  INST_MOE_BGMV_SHRINK_SLICED(wide, narrow, in_T, out_T, W_T)                 \
  INST_MOE_BGMV_EXPAND_SLICED(narrow, wide, in_T, W_T)
