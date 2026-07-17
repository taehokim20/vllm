
#pragma once
#ifndef MOE_SCALE_INPUTS_CU
  #define MOE_SCALE_INPUTS_CU

  #ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
    #error Do not include this file directly.
  #endif

  #include <cstdint>
  #include <cstring>

  #include <cuda_bf16.h>

  #include "moe_interface.h"
  #include "moe_internal.h"
  #include "moe_grid_barrier.h"
  #include "ptx_utils.h"

namespace moe_monokernel {

/**
 * @brief Sets NaNs (any sign/payload) to 0.0; other values pass through.
 */
__device__ static __forceinline__ __nv_bfloat162
mask_NaNs_to_zero(__nv_bfloat162 xs) {
  return type_pun<__nv_bfloat162>(type_pun<uint32_t>(xs) & __heq2_mask(xs, xs));
}

/**
 * @brief Per-128-K-block bf16 → fp8 quantization for one token.
 *
 * Called by exactly one warp per (token, k_block) pair: converts the
 * token's 128 bf16 K-values into fp8 with one per-(token, block) scale.
 * Each of the 32 lanes owns 4 consecutive K-values; a warp reduce finds
 * the block max.
 *
 * Output layout (canonical WGMMA K-major): fp8_act[kc][tok][ki] with
 * kc*16 + ki = K within the tile; only the caller's `tok` row is written
 * (other tokens are covered by other warps).  The token dim may be padded
 * to 9 rows (bank-conflict avoidance — see fp8_act_full in
 * moe_internal.h); the 9th row is never written.
 *
 * For `tok >= batch_size` the slot is zero-filled and the scale set to a
 * neutral 1.0f (never consumed — the scale-apply zeroes padded tokens).
 *
 * @param bf16_row            SHM bf16 row for this (token, k_block).
 * @param fp8_act             SHM fp8 output atom [kc][tok][ki].
 * @param tok                 Token slot to write.
 * @param batch_size          Real token count.
 * @param act_scale_for_step  fp32 destination for this pair's scale
 *                            (`&shmem->act_scale[k_block][tok]`).
 */
template <typename Dims, std::size_t BF16InCols, std::size_t Fp8NumChunks,
          std::size_t Fp8Tok, std::size_t Fp8KInner>
__device__ __forceinline__ void moe_streaming_quantize_k128(
    const A_element (&bf16_row)[BF16InCols],
    AQ_element (&fp8_act)[Fp8NumChunks][Fp8Tok][Fp8KInner], std::uint32_t tok,
    std::uint32_t batch_size, float* __restrict__ act_scale_for_step) {
  static_assert(Dims::BS <= 8, "Streaming quantize is for BS<=8");
  static_assert(BF16InCols == 128, "bf16_row must have 128 K cols");
  static_assert(Fp8NumChunks == 8, "fp8_act must have 8 K-chunks of 16");
  static_assert(Fp8Tok == 8 || Fp8Tok == 9,
                "fp8_act must have 8 or 9 (kc-padded) token rows; only the "
                "first 8 are written");
  static_assert(Fp8KInner == 16, "fp8_act inner dim must be 16");

  const std::uint32_t thread = get_thread<Dims>();  // 0..31
  constexpr float FP8_MAX = 448.f;
  constexpr float FP8_MAX_INV = 1.0f / 448.f;

  if (tok >= batch_size) {
    // Zero-fill so the WGMMA B operand sees no stray fp8 bit patterns.
    const uint32_t col = thread * 4;
    const uint32_t kc = col / 16;
    const uint32_t ki = col % 16;
    fp8_act[kc][tok][ki + 0] = (AQ_element)0;
    fp8_act[kc][tok][ki + 1] = (AQ_element)0;
    fp8_act[kc][tok][ki + 2] = (AQ_element)0;
    fp8_act[kc][tok][ki + 3] = (AQ_element)0;
    if (thread == 0) *act_scale_for_step = 1.0f;
    return;
  }

  const uint32_t col = thread * 4;
  __nv_bfloat162 bf_01 =
      *reinterpret_cast<const __nv_bfloat162*>(&bf16_row[col + 0]);
  __nv_bfloat162 bf_23 =
      *reinterpret_cast<const __nv_bfloat162*>(&bf16_row[col + 2]);
  bf_01 = mask_NaNs_to_zero(bf_01);
  bf_23 = mask_NaNs_to_zero(bf_23);
  float2 f01 = __bfloat1622float2(bf_01);
  float2 f23 = __bfloat1622float2(bf_23);
  float r0 = f01.x, r1 = f01.y, r2 = f23.x, r3 = f23.y;

  float local_max =
      fmaxf(fmaxf(fabsf(r0), fabsf(r1)), fmaxf(fabsf(r2), fabsf(r3)));
  float blk_max = warp_reduce_max_float(local_max);
  if (blk_max < __FLT_MIN__) blk_max = 1.f;

  const float blk_act_scale = blk_max * FP8_MAX_INV;
  const float blk_inv_scale = FP8_MAX / blk_max;

  AQ_element q0 = (AQ_element)(r0 * blk_inv_scale);
  AQ_element q1 = (AQ_element)(r1 * blk_inv_scale);
  AQ_element q2 = (AQ_element)(r2 * blk_inv_scale);
  AQ_element q3 = (AQ_element)(r3 * blk_inv_scale);

  const uint32_t kc = col / 16;
  const uint32_t ki = col % 16;
  fp8_act[kc][tok][ki + 0] = q0;
  fp8_act[kc][tok][ki + 1] = q1;
  fp8_act[kc][tok][ki + 2] = q2;
  fp8_act[kc][tok][ki + 3] = q3;

  if (thread == 0) *act_scale_for_step = blk_act_scale;
}

/**
 * @brief Phase-2 routing-window BF16 → FP8 quantization (warps 1..11).
 *
 * Distributes the BS * K_BLOCKS_TOTAL (token, k_block) quantization tasks
 * across the 11 participating warps (stride-11 over the linear pair
 * index; warp 0 runs prepare_moe_topk_BS8 instead).  Each pair invokes
 * moe_streaming_quantize_k128 once with a single-row view onto the
 * prefetched tile-major `bf16_in_full[k_block][token]`.
 *
 * Caller contract (enforced at the call site in moe.cu):
 *   1. Gate warp 0 OUT of this call.
 *   2. Wait on bar_rwin on every warp 1..11 thread first — reading
 *      bf16_in_full earlier races the in-flight Phase-1 TMA.
 *   3. Emit a block-wide __syncthreads() AFTER this returns to publish
 *      the FP8 atoms + scales before Phase 3 reads them.
 */
template <typename Dims, std::size_t KBlocks, std::size_t Bs, std::size_t KStep,
          std::size_t Fp8KBlocks, std::size_t Fp8NumChunks, std::size_t Fp8Tok,
          std::size_t Fp8KInner, std::size_t ScaleKBlocks, std::size_t ScaleBs>
__device__ inline void routing_phase_quantize(
    A_element (&bf16_in_full)[KBlocks][Bs][KStep],
    AQ_element (&fp8_act_full)[Fp8KBlocks][Fp8NumChunks][Fp8Tok][Fp8KInner],
    float (&act_scale)[ScaleKBlocks][ScaleBs], std::uint32_t batch_size,
    // EP in-kernel dispatch OVERLAP (Stage B): when `peer_activations` is
    // non-null, tokens OUTSIDE this rank's owned range [own_lo, own_hi) are
    // REMOTE — their bf16 row is peer-read here (folded into the quantize,
    // overlapping warp-0 routing) instead of being staged into
    // `activations_in` by a separate prologue copy + grid_barrier. Each
    // (token, kblk) pair is owned by exactly one warp, so a `__syncwarp()`
    // after the per-warp overwrite is sufficient (no cross-warp sync).
    // Cross-RANK ordering (peer must have staged before we read it) is
    // provided by the graph-safe ep_wait_ready handshake gated at the kernel
    // entry (see moe.cu), not here. Defaults (nullptr/0/0) preserve the
    // original non-EP behavior byte-identically.
    const A_element* __restrict__ peer_activations = nullptr,
    std::uint32_t own_lo = 0u, std::uint32_t own_hi = 0u) {
  using CoreDims = MoECoreDims<Dims>;

  static_assert(Dims::BS <= 8, "routing_phase_quantize is BS8-only");
  constexpr std::uint32_t K_BLOCKS_TOTAL =
      Dims::HIDDEN_STATES / CoreDims::K_STEP_WGMMA;
  static_assert(KBlocks == K_BLOCKS_TOTAL,
                "bf16_in_full outer extent must be K_BLOCKS_TOTAL");
  static_assert(Bs == Dims::BS,
                "bf16_in_full middle extent must be Dims::BS");
  static_assert(KStep == CoreDims::K_STEP_WGMMA,
                "bf16_in_full inner extent must be K_STEP_WGMMA (128)");
  static_assert(Fp8NumChunks == 8, "fp8_act_full middle dim must be 8");
  static_assert(Fp8Tok == CoreDims::T_TILE || Fp8Tok == CoreDims::T_TILE + 1,
                "fp8_act_full token dim must be T_TILE or T_TILE+1 (padded)");
  static_assert(Fp8KInner == 16, "fp8_act_full inner dim must be 16");
  static_assert(ScaleBs == Dims::BS, "act_scale inner extent must be Dims::BS");
  static_assert(Fp8KBlocks == K_BLOCKS_TOTAL,
                "fp8_act_full outer extent must be K_BLOCKS_TOTAL");
  static_assert(ScaleKBlocks == K_BLOCKS_TOTAL,
                "act_scale outer extent must be K_BLOCKS_TOTAL");

  constexpr std::uint32_t PAIRS_TOTAL = Dims::BS * K_BLOCKS_TOTAL;
  constexpr std::uint32_t NUM_QUANT_WARPS = 11u;

  const std::uint32_t warp = get_any_warp<Dims>();  // in [1, 12)
  assert(warp >= 1u && warp < 12u);
  const std::uint32_t w_idx = warp - 1u;

  #pragma unroll 1
  for (std::uint32_t i = w_idx; i < PAIRS_TOTAL; i += NUM_QUANT_WARPS) {
    const std::uint32_t token = i / K_BLOCKS_TOTAL;
    const std::uint32_t kblk = i % K_BLOCKS_TOTAL;

    // EP dispatch overlap: for REMOTE tokens, peer-read this (token, kblk)
    // 128-bf16 row directly into bf16_in_full before quantizing it. The
    // owning warp's 32 lanes cover the 128 elements (4 each); __syncwarp()
    // orders the overwrite before moe_streaming_quantize_k128 reads the row.
    // bf16_in_full is dead after quantize (Phase 3 reads only fp8_act_full),
    // so overwriting it here is safe. peer layout: [BS, HIDDEN] row-major,
    // HIDDEN = K_BLOCKS_TOTAL * KStep, so row (token, kblk) starts at
    // token*HIDDEN + kblk*KStep.
    if (peer_activations != nullptr && (token < own_lo || token >= own_hi)) {
      const std::uint32_t lane = threadIdx.x & 31u;
      A_element* dst = &bf16_in_full[kblk][token][0];
      const A_element* src =
          peer_activations +
          static_cast<std::size_t>(token) * (K_BLOCKS_TOTAL * KStep) +
          static_cast<std::size_t>(kblk) * KStep;
      for (std::uint32_t e = lane; e < KStep; e += 32u) dst[e] = src[e];
      __syncwarp();
    }

    const auto& bf_row = bf16_in_full[kblk][token];
    auto& fp8_atom = fp8_act_full[kblk];

    moe_streaming_quantize_k128<Dims>(bf_row, fp8_atom, /*tok=*/token,
                                      batch_size, &act_scale[kblk][token]);
  }
}

}  // namespace moe_monokernel

#endif
