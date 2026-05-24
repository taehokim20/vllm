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

  // ── Step B: readLL + reduce + residual + RMSNorm ───────────────────
  // We need the full token row for RMSNorm (variance over hidden_dim).
  // But this block only owns col_tile columns. Two approaches:
  //   (a) Each block computes partial variance over its stripe, then
  //       a cross-block reduction gives the full variance.
  //   (b) Each block does AR + residual for its stripe, writes to
  //       activations_out, then a separate RMSNorm pass reads the
  //       full row.
  //
  // For simplicity in v1, we do (b): Phase 5 does AR + residual and
  // writes the pre-norm result. RMSNorm is left as a trivial follow-up
  // (either a second pass within this kernel or a tiny downstream kernel).
  //
  // TODO: Fuse RMSNorm into Phase 5 using cross-block shared-memory
  // or a second grid-barrier + norm pass within the same kernel.

  for (uint32_t flat = threadIdx.x; flat < batch_size * packets_per_stripe;
       flat += blockDim.x) {
    const uint32_t tok = flat / packets_per_stripe;
    const uint32_t pkt_in_stripe = flat % packets_per_stripe;
    const uint32_t col = base_col + pkt_in_stripe * LL_ELEMS_PER_PACKET;

    // Start with local value (already cast to bf16 above; re-read from
    // down_partial_out to avoid storing intermediate)
    float vals[LL_ELEMS_PER_PACKET];
#pragma unroll
    for (int i = 0; i < LL_ELEMS_PER_PACKET; ++i) {
      vals[i] = spec->down_partial_out[tok * Dims::HIDDEN_STATES + col + i];
    }

    // Add peer contributions
    for (uint32_t peer = 0; peer < tp_size; ++peer) {
      if (peer == tp_rank) continue;
      // Read from MY LL buffer, at the slot where this peer wrote
      LLPacket* my_buf = reinterpret_cast<LLPacket*>(peer_ll_buffers[tp_rank]);
      const uint32_t peer_ll_idx =
          peer * batch_size * packets_per_row +
          tok * packets_per_row + pkt_offset + pkt_in_stripe;
      uint64_t peer_data = readLL(&my_buf[peer_ll_idx], ll_flag);

      // Unpack 4 bf16 values and accumulate as fp32
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

    // Write pre-norm result (AR'd + residual) to activations_out
    // RMSNorm will be applied in a follow-up pass (see TODO above)
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

}  // namespace moe_monokernel

#endif  // MOE_COMM_LL_H
