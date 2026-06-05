/**
 * LL (Low Latency) communication primitives for the MoE monokernel.
 *
 * Implements the SwiftSpec NCCL-LL protocol (arXiv:2506.11309 §3.3 Algorithm 2)
 * for fusing all-reduce into the monokernel's Phase 5.
 *
 * The LL protocol uses 128-bit atomic stores/loads with embedded flags for
 * synchronization — no explicit barriers or NCCL calls needed.
 */

#pragma once
#ifndef MOE_COMM_LL_H
#define MOE_COMM_LL_H

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace moe_monokernel {

// ============================================================================
// LL Packet: 128-bit structure matching NCCL LL format
// ============================================================================

struct LLPacket {
  uint32_t data_low;   // lower 32 bits of payload (2 bf16 values)
  uint32_t flag1;      // flag value
  uint32_t data_high;  // upper 32 bits of payload (2 bf16 values)
  uint32_t flag2;      // flag value (same as flag1, for atomicity)
};

static_assert(sizeof(LLPacket) == 16, "LLPacket must be 128 bits");

// Each packet carries 4 bf16 values (8 bytes of payload)
static constexpr int LL_ELEMS_PER_PACKET = 4;

// ============================================================================
// storeLL: write payload + flag to a peer's LL buffer via NVLink
// ============================================================================

/**
 * @brief Store a 64-bit payload (4 bf16 values) + flag into an LL packet.
 *
 * Uses st.volatile.global.v4.u32 for NVLink visibility without fences.
 * The 128-bit store is atomic at the 64-bit granularity on NVLink.
 */
__device__ __forceinline__ void storeLL(LLPacket* dst, uint64_t val,
                                        uint32_t flag) {
  uint32_t val_low = static_cast<uint32_t>(val);
  uint32_t val_high = static_cast<uint32_t>(val >> 32);
  // volatile store for cross-GPU visibility (single-process, peer access)
  asm volatile(
      "st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};\n"
      :
      : "l"(dst), "r"(val_low), "r"(flag), "r"(val_high), "r"(flag)
      : "memory");
}

// ============================================================================
// readLL: poll until expected flag appears, then return payload
// ============================================================================

/**
 * @brief Read a 64-bit payload from an LL packet, spinning until the
 *        expected flag appears in both flag fields.
 *
 * Uses ld.volatile.global.v4.u32 for polling.
 */
__device__ __forceinline__ uint64_t readLL(const LLPacket* src,
                                           uint32_t expected_flag) {
  uint32_t data1, flag1, data2, flag2;
  do {
    // volatile load for cross-GPU visibility (single-process, peer access)
    asm volatile(
        "ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(data1), "=r"(flag1), "=r"(data2), "=r"(flag2)
        : "l"(src)
        : "memory");
  } while (flag1 != expected_flag || flag2 != expected_flag);

  return static_cast<uint64_t>(data1) |
         (static_cast<uint64_t>(data2) << 32);
}

// ============================================================================
// Helper: warp-reduce sum for RMSNorm variance computation
// ============================================================================

__device__ __forceinline__ float ll_warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = 16; offset >= 1; offset /= 2) {
    val += __shfl_xor_sync(0xFFFFFFFF, val, offset, 32);
  }
  return val;
}

// ============================================================================
// AR-only Phase 5: storeLL + readLL + reduce (no residual, no RMSNorm)
//
// Minimal version that replaces NCCL all-reduce. Output is the sum of all
// ranks' down_partial_out, cast to bf16. Residual and RMSNorm remain as
// separate ops outside the kernel.
// ============================================================================

/**
 * @brief All-reduce only via LL protocol for one block's column stripe.
 *
 * Each block:
 *   1. Casts its fp32 partial to bf16 and sends via storeLL to all peers
 *   2. Reads peer contributions via readLL
 *   3. Sums local + all peers and writes bf16 to activations_out
 */
template <typename Dims>
__device__ void phase5_ar_only_ll(
    MoEGemmSpec<Dims>* __restrict__ spec,
    R_element* __restrict__ activations_out,
    void** __restrict__ peer_ll_buffers, uint32_t ll_flag,
    uint32_t tp_rank, uint32_t tp_size, uint32_t batch_size,
    uint32_t base_col, uint32_t col_tile) {
  const uint32_t packets_per_stripe = col_tile / LL_ELEMS_PER_PACKET;
  const uint32_t packets_per_row = Dims::HIDDEN_STATES / LL_ELEMS_PER_PACKET;
  const uint32_t pkt_offset = base_col / LL_ELEMS_PER_PACKET;

  // ── Step 1: cast fp32 → bf16 and storeLL to all peers ──
  for (uint32_t flat = threadIdx.x; flat < batch_size * packets_per_stripe;
       flat += blockDim.x) {
    const uint32_t tok = flat / packets_per_stripe;
    const uint32_t pkt_in_stripe = flat % packets_per_stripe;
    const uint32_t col = base_col + pkt_in_stripe * LL_ELEMS_PER_PACKET;

    __nv_bfloat16 bf16_vals[LL_ELEMS_PER_PACKET];
#pragma unroll
    for (int i = 0; i < LL_ELEMS_PER_PACKET; ++i) {
      float v = spec->down_partial_out[tok * Dims::HIDDEN_STATES + col + i];
      bf16_vals[i] = (__nv_bfloat16)v;
    }

    uint64_t payload;
    memcpy(&payload, bf16_vals, sizeof(uint64_t));

    const uint32_t ll_idx =
        tp_rank * batch_size * packets_per_row +
        tok * packets_per_row + pkt_offset + pkt_in_stripe;

    for (uint32_t peer = 0; peer < tp_size; ++peer) {
      if (peer == tp_rank) continue;
      LLPacket* peer_buf = reinterpret_cast<LLPacket*>(peer_ll_buffers[peer]);
      storeLL(&peer_buf[ll_idx], payload, ll_flag);
    }
  }

  // Ensure stores are visible before reads
  __threadfence_system();

  // ── Step 2: readLL from peers + reduce + write ──
  for (uint32_t flat = threadIdx.x; flat < batch_size * packets_per_stripe;
       flat += blockDim.x) {
    const uint32_t tok = flat / packets_per_stripe;
    const uint32_t pkt_in_stripe = flat % packets_per_stripe;
    const uint32_t col = base_col + pkt_in_stripe * LL_ELEMS_PER_PACKET;

    // Start with local partial (fp32 for precision)
    float vals[LL_ELEMS_PER_PACKET];
#pragma unroll
    for (int i = 0; i < LL_ELEMS_PER_PACKET; ++i) {
      vals[i] = spec->down_partial_out[tok * Dims::HIDDEN_STATES + col + i];
    }

    // Add peer contributions
    for (uint32_t peer = 0; peer < tp_size; ++peer) {
      if (peer == tp_rank) continue;
      LLPacket* my_buf = reinterpret_cast<LLPacket*>(peer_ll_buffers[tp_rank]);
      const uint32_t peer_ll_idx =
          peer * batch_size * packets_per_row +
          tok * packets_per_row + pkt_offset + pkt_in_stripe;
      uint64_t peer_data = readLL(&my_buf[peer_ll_idx], ll_flag);

      __nv_bfloat16 peer_bf16[LL_ELEMS_PER_PACKET];
      memcpy(peer_bf16, &peer_data, sizeof(uint64_t));
#pragma unroll
      for (int i = 0; i < LL_ELEMS_PER_PACKET; ++i) {
        vals[i] += (float)peer_bf16[i];
      }
    }

    // Write reduced result as bf16
    const uint32_t elem_offset = tok * Dims::HIDDEN_STATES + col;
#pragma unroll
    for (int i = 0; i < LL_ELEMS_PER_PACKET; ++i) {
      activations_out[elem_offset + i] = (R_element)vals[i];
    }
  }

  // Zero padding rows [batch_size, Dims::BS)
  for (uint32_t flat = threadIdx.x;
       flat < (Dims::BS - batch_size) * packets_per_stripe;
       flat += blockDim.x) {
    const uint32_t tok = batch_size + flat / packets_per_stripe;
    const uint32_t pkt_in_stripe = flat % packets_per_stripe;
    const uint32_t col = base_col + pkt_in_stripe * LL_ELEMS_PER_PACKET;
    const uint32_t elem_offset = tok * Dims::HIDDEN_STATES + col;
#pragma unroll
    for (int i = 0; i < LL_ELEMS_PER_PACKET; ++i) {
      activations_out[elem_offset + i] = (R_element)0.0f;
    }
  }
}

// ============================================================================
// Fused Phase 5: storeLL + readLL + reduce + residual + RMSNorm
//
// Replaces the simple fp32→bf16 cast in Phase 5 when tp_size > 1.
// Called by blocks with down_group_r == 0 (the Phase-5 writers).
// ============================================================================

/**
 * @brief Fused all-reduce (LL) + residual + RMSNorm for one block's
 *        column stripe.
 *
 * @param spec            Scratchpad (reads down_partial_out)
 * @param activations_out Output buffer [BS, HIDDEN_STATES] bf16 — final result
 * @param residual_in     Residual from previous layer [BS, HIDDEN_STATES] bf16
 * @param rms_gamma       RMSNorm weight [HIDDEN_STATES] bf16
 * @param rms_eps         RMSNorm epsilon
 * @param peer_ll_buffers Array of tp_size pointers to peer LL buffers
 * @param ll_flag         Current flag value for this invocation
 * @param tp_rank         This rank's index
 * @param tp_size         Total TP ranks
 * @param batch_size      Number of active tokens
 * @param base_col        First column this block owns
 * @param col_tile        Number of columns this block owns (DOWN_COL_TILE)
 */
template <typename Dims>
__device__ void phase5_fused_ar_ll(
    MoEGemmSpec<Dims>* __restrict__ spec,
    R_element* __restrict__ activations_out,
    const R_element* __restrict__ residual_in,
    const R_element* __restrict__ rms_gamma, float rms_eps,
    void** __restrict__ peer_ll_buffers, uint32_t ll_flag,
    uint32_t tp_rank, uint32_t tp_size, uint32_t batch_size,
    uint32_t base_col, uint32_t col_tile) {
  // Number of LL packets per token for this block's stripe
  const uint32_t packets_per_stripe = col_tile / LL_ELEMS_PER_PACKET;
  // Total packets per token across the full hidden dim (for buffer indexing)
  const uint32_t packets_per_row = Dims::HIDDEN_STATES / LL_ELEMS_PER_PACKET;
  // Packet offset for this block's stripe within a token row
  const uint32_t pkt_offset = base_col / LL_ELEMS_PER_PACKET;

  // Reset the B2 arrival counter at the start of Phase 5.
  // The counter lives at the tail of down_partial_out and must be zero
  // before any block increments it in B2.
  constexpr uint32_t DOWN_GRID_P5 = Dims::HIDDEN_STATES / MoEGemmSpec<Dims>::DOWN_COL_TILE;
  uint32_t* arrival_counter = reinterpret_cast<uint32_t*>(
      &spec->down_partial_out[Dims::BS * Dims::HIDDEN_STATES - 1]);
  if (threadIdx.x == 0 && base_col == 0) {
    // Only one block (the first) resets the counter
    *arrival_counter = 0u;
  }
  // All blocks must see the reset before proceeding
  __threadfence();
  __syncthreads();

  // ── Step A: cast fp32 → bf16 and storeLL to all peers ──────────────
  for (uint32_t flat = threadIdx.x; flat < batch_size * packets_per_stripe;
       flat += blockDim.x) {
    const uint32_t tok = flat / packets_per_stripe;
    const uint32_t pkt_in_stripe = flat % packets_per_stripe;
    const uint32_t col = base_col + pkt_in_stripe * LL_ELEMS_PER_PACKET;

    // Read 4 fp32 values from down_partial_out, cast to bf16, pack
    __nv_bfloat16 bf16_vals[LL_ELEMS_PER_PACKET];
#pragma unroll
    for (int i = 0; i < LL_ELEMS_PER_PACKET; ++i) {
      float v = spec->down_partial_out[tok * Dims::HIDDEN_STATES + col + i];
      bf16_vals[i] = (__nv_bfloat16)v;
    }

    // Pack 4 bf16 into a 64-bit payload
    uint64_t payload;
    memcpy(&payload, bf16_vals, sizeof(uint64_t));

    // Store to every peer's LL buffer
    const uint32_t ll_idx =
        tp_rank * batch_size * packets_per_row +
        tok * packets_per_row + pkt_offset + pkt_in_stripe;

    for (uint32_t peer = 0; peer < tp_size; ++peer) {
      if (peer == tp_rank) continue;
      LLPacket* peer_buf = reinterpret_cast<LLPacket*>(peer_ll_buffers[peer]);
      storeLL(&peer_buf[ll_idx], payload, ll_flag);
    }
  }

  // Ensure all storeLL writes are visible to other GPUs (system-scope fence)
  __threadfence_system();

  // ── Step B: readLL + reduce + residual + RMSNorm ───────────────────
  //
  // Strategy for RMSNorm across blocks:
  //   Each block owns DOWN_COL_TILE columns. RMSNorm needs variance over
  //   the full HIDDEN_STATES dimension. We use a 3-sub-step approach:
  //
  //   B1: AR reduce + residual → write pre-norm fp32 to activations_out
  //       (temporarily as fp32 via reinterpret, or to a scratchpad region)
  //       + compute partial sum-of-squares for this block's stripe.
  //       Write partial_sq_sum[block_idx][tok] to scratchpad.
  //
  //   B2: grid_barrier — all Phase-5 blocks sync so partial_sq_sums are
  //       visible. (Reuse the existing grid_barrier infrastructure.)
  //       NOTE: We use spec->down_partial_out as scratch for the partial
  //       sums since it's no longer needed after Step A consumed it.
  //
  //   B3: Each block reads all DOWN_GRID partial sums for its tokens,
  //       computes the full variance, applies RMSNorm scale * gamma,
  //       writes final bf16 to activations_out.
  //
  // Memory layout for partial sums (reusing down_partial_out):
  //   spec->down_partial_out[block_idx * BS + tok] = partial_sq_sum
  //   where block_idx ∈ [0, DOWN_GRID) and tok ∈ [0, batch_size)

  // ── B1: AR reduce + residual + compute partial sum-of-squares ──────
  for (uint32_t flat = threadIdx.x; flat < batch_size * packets_per_stripe;
       flat += blockDim.x) {
    const uint32_t tok = flat / packets_per_stripe;
    const uint32_t pkt_in_stripe = flat % packets_per_stripe;
    const uint32_t col = base_col + pkt_in_stripe * LL_ELEMS_PER_PACKET;

    // Start with local value
    float vals[LL_ELEMS_PER_PACKET];
#pragma unroll
    for (int i = 0; i < LL_ELEMS_PER_PACKET; ++i) {
      vals[i] = spec->down_partial_out[tok * Dims::HIDDEN_STATES + col + i];
    }

    // Add peer contributions
    for (uint32_t peer = 0; peer < tp_size; ++peer) {
      if (peer == tp_rank) continue;
      LLPacket* my_buf = reinterpret_cast<LLPacket*>(peer_ll_buffers[tp_rank]);
      const uint32_t peer_ll_idx =
          peer * batch_size * packets_per_row +
          tok * packets_per_row + pkt_offset + pkt_in_stripe;
      uint64_t peer_data = readLL(&my_buf[peer_ll_idx], ll_flag);

      __nv_bfloat16 peer_bf16[LL_ELEMS_PER_PACKET];
      memcpy(peer_bf16, &peer_data, sizeof(uint64_t));
#pragma unroll
      for (int i = 0; i < LL_ELEMS_PER_PACKET; ++i) {
        vals[i] += (float)peer_bf16[i];
      }
    }

    // Add residual
    const uint32_t elem_offset = tok * Dims::HIDDEN_STATES + col;
#pragma unroll
    for (int i = 0; i < LL_ELEMS_PER_PACKET; ++i) {
      vals[i] += (float)residual_in[elem_offset + i];
    }

    // Write pre-norm result to activations_out (will be overwritten in B3
    // with the normed value). Also write to residual_out if provided
    // (the pre-norm value IS the updated residual for the next layer).
#pragma unroll
    for (int i = 0; i < LL_ELEMS_PER_PACKET; ++i) {
      activations_out[elem_offset + i] = (R_element)vals[i];
    }
  }

  // Compute partial sum-of-squares for this block's stripe.
  // Each thread accumulates over its assigned packets, then we
  // block-reduce to get one value per (block, token).
  __syncthreads();

  // Use shared memory for per-token partial sums within this block
  // Reuse a small region — we only need BS floats.
  extern __shared__ char phase5_smem[];
  float* tok_partial_sq = reinterpret_cast<float*>(phase5_smem);
  // Initialize
  for (uint32_t t = threadIdx.x; t < batch_size; t += blockDim.x) {
    tok_partial_sq[t] = 0.f;
  }
  __syncthreads();

  for (uint32_t flat = threadIdx.x; flat < batch_size * packets_per_stripe;
       flat += blockDim.x) {
    const uint32_t tok = flat / packets_per_stripe;
    const uint32_t pkt_in_stripe = flat % packets_per_stripe;
    const uint32_t col = base_col + pkt_in_stripe * LL_ELEMS_PER_PACKET;
    const uint32_t elem_offset = tok * Dims::HIDDEN_STATES + col;

    float local_sq = 0.f;
#pragma unroll
    for (int i = 0; i < LL_ELEMS_PER_PACKET; ++i) {
      float v = (float)activations_out[elem_offset + i];
      local_sq += v * v;
    }
    atomicAdd(&tok_partial_sq[tok], local_sq);
  }
  __syncthreads();

  // Write this block's partial sum-of-squares to scratchpad.
  // Layout: down_partial_out[block_idx * BS + tok]
  // (Reusing down_partial_out which is no longer needed.)
  const uint32_t block_idx_in_grid = base_col / col_tile;
  for (uint32_t t = threadIdx.x; t < batch_size; t += blockDim.x) {
    spec->down_partial_out[block_idx_in_grid * Dims::BS + t] = tok_partial_sq[t];
  }

  // ── B2: barrier — wait for all Phase-5 blocks to write their partials ─
  // We need a lightweight sync across the DOWN_GRID Phase-5 blocks.
  // Use __threadfence() + a simple flag-based spin on the scratchpad.
  // For correctness we need all DOWN_GRID blocks to have written before
  // any block reads. We'll use a simple atomic counter approach.
  __threadfence();  // Ensure our write to down_partial_out is visible

  // Use the last element of down_partial_out as an atomic arrival counter
  // (safe because we only use [0, DOWN_GRID * BS) elements above, and
  // down_partial_out has BS * HIDDEN_STATES elements total — plenty of room)

  if (threadIdx.x == 0) {
    atomicAdd(arrival_counter, 1u);
    // Spin until all DOWN_GRID blocks have arrived
    while (atomicAdd(arrival_counter, 0u) < DOWN_GRID_P5) {
      // spin
    }
  }
  __syncthreads();

  // ── B3: compute full variance + apply RMSNorm ─────────────────────
  for (uint32_t flat = threadIdx.x; flat < batch_size * packets_per_stripe;
       flat += blockDim.x) {
    const uint32_t tok = flat / packets_per_stripe;
    const uint32_t pkt_in_stripe = flat % packets_per_stripe;
    const uint32_t col = base_col + pkt_in_stripe * LL_ELEMS_PER_PACKET;
    const uint32_t elem_offset = tok * Dims::HIDDEN_STATES + col;

    // Sum partial sq sums across all blocks for this token
    float total_sq = 0.f;
    for (uint32_t b = 0; b < DOWN_GRID_P5; ++b) {
      total_sq += spec->down_partial_out[b * Dims::BS + tok];
    }

    // RMSNorm scale
    float rms_scale = rsqrtf(
        total_sq / static_cast<float>(Dims::HIDDEN_STATES) + rms_eps);

    // Apply gamma * scale and write final bf16
#pragma unroll
    for (int i = 0; i < LL_ELEMS_PER_PACKET; ++i) {
      float pre_norm = (float)activations_out[elem_offset + i];
      float gamma_val = (float)rms_gamma[col + i];
      activations_out[elem_offset + i] =
          (R_element)(pre_norm * rms_scale * gamma_val);
    }
  }

  // Zero padding rows [batch_size, Dims::BS)
  for (uint32_t flat = threadIdx.x;
       flat < (Dims::BS - batch_size) * packets_per_stripe;
       flat += blockDim.x) {
    const uint32_t tok = batch_size + flat / packets_per_stripe;
    const uint32_t pkt_in_stripe = flat % packets_per_stripe;
    const uint32_t col = base_col + pkt_in_stripe * LL_ELEMS_PER_PACKET;
    const uint32_t elem_offset = tok * Dims::HIDDEN_STATES + col;
#pragma unroll
    for (int i = 0; i < LL_ELEMS_PER_PACKET; ++i) {
      activations_out[elem_offset + i] = (R_element)0.0f;
    }
  }
}

}  // namespace moe_monokernel

#endif  // MOE_COMM_LL_H
