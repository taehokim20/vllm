#ifndef MOE_INTERFACE_H
#define MOE_INTERFACE_H

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace moe_monokernel {

// Weight quantization granularity
enum class QuantGranularity : uint32_t {
  PER_CHANNEL = 0,  // one scale per row
  BLOCK_WISE = 1,   // one scale per (block_row, block_col) tile
};

template <uint32_t m, uint32_t n, uint32_t k, uint32_t num_experts>
struct MoEDimensions {
  static constexpr uint32_t HIDDEN_STATES = k;
  static constexpr uint32_t K = k;
  static constexpr uint32_t N = n;
  static constexpr uint32_t BS = m;
  static constexpr uint32_t M = m;
  static constexpr uint32_t NUM_EXPERTS = num_experts;

  static constexpr QuantGranularity QUANT_GRAN = QuantGranularity::PER_CHANNEL;
  static constexpr uint32_t BLOCK_SCALE_ROW = 0;
  static constexpr uint32_t BLOCK_SCALE_COL = 0;

  struct KernelConfig {
    static constexpr std::uint32_t GRID_SIZE = (2 * N) / 16;
    static constexpr std::uint32_t BLOCK_SIZE = 384;
  };
};

// ── Per-shape Dims structs (GENERATED) ───────────────────────────────────
// Emitted from csrc/moe/moe_monokernel/shapes.json by tools/gen_shapes.py.
// Each shape's KernelConfig knobs come from its config[0] (the shipped
// default), so the base Dims is byte-identical to the config-0 tunable
// instantiation.  To add/edit a shape: edit shapes.json, run gen_shapes.py,
// rebuild.  DO NOT hand-edit the generated file.
#include "../generated/dims_generated.inc"

// ── EP (expert-parallel) variant of the DeepSeek-V4-Flash full-N shape ──
// Hand-written (NOT generated): the EP dispatch path is maintained outside
// gen_shapes.py so re-onboarding never clobbers it. Cloned from the generated
// `Dims_BS8_E256_N2048_K4096_BlockFP8_WGMMA_TMA` (keep the two in sync if the
// base shape's KernelConfig is retuned).
//
// EP shards EXPERTS, not N: NUM_EXPERTS stays the GLOBAL 256 (routing /
// router_logits width / software-barrier sizing are unchanged and run over all
// 256), N stays full (2048). NUM_LOCAL_EXPERTS=128 is the EP=2 per-rank slice;
// each rank is given a [256, ...] weight buffer with only its local experts
// filled, and the `is_ep` filter in prepare_moe_topk_BS8 keeps only routed ids
// in [expert_base, expert_base + 128) (others -> 0xFFFF sentinel, skipped).
// Structurally identical to the base Dims otherwise, so the whole TMA / WGMMA
// pipeline is unchanged.
struct Dims_BS8_E256_N2048_K4096_BlockFP8_WGMMA_TMA_EP {
  static constexpr uint32_t HIDDEN_STATES = 4096;
  static constexpr uint32_t K = 4096;
  static constexpr uint32_t N = 2048;
  static constexpr uint32_t BS = 8;
  static constexpr uint32_t M = 8;
  static constexpr uint32_t NUM_EXPERTS = 256;        // GLOBAL expert count
  static constexpr uint32_t NUM_LOCAL_EXPERTS = 128;  // per-rank slice (EP=2)
  static constexpr QuantGranularity QUANT_GRAN = QuantGranularity::BLOCK_WISE;
  static constexpr uint32_t BLOCK_SCALE_ROW = 128;
  static constexpr uint32_t BLOCK_SCALE_COL = 128;
  static constexpr uint32_t UP_SCALE_ROWS =
      (2 * N + BLOCK_SCALE_ROW - 1) / BLOCK_SCALE_ROW;
  static constexpr uint32_t UP_SCALE_COLS =
      (K + BLOCK_SCALE_COL - 1) / BLOCK_SCALE_COL;
  static constexpr uint32_t DOWN_SCALE_ROWS =
      (K + BLOCK_SCALE_ROW - 1) / BLOCK_SCALE_ROW;
  static constexpr uint32_t DOWN_SCALE_COLS =
      (N + BLOCK_SCALE_COL - 1) / BLOCK_SCALE_COL;
  struct KernelConfig {
    static constexpr std::uint32_t GRID_SIZE = 128;
    static constexpr std::uint32_t BLOCK_SIZE = 384;
    static constexpr bool USE_WGMMA = true;
    static constexpr bool USE_TMA = true;
    static constexpr std::uint32_t K_STEP_DOWN = 128;
    static constexpr std::uint32_t K_STEP_UP = 256;
    static constexpr std::uint32_t UP_W_SLOTS = 2;
    static constexpr bool USE_PAIR_LAYOUT = true;
    static constexpr bool IS_EP = true;  // enables local-expert filter + peer-read
  };
};

// Extracts a shape's explicit `KernelConfig::UP_COL_HALVES` if it pins one
// (decoupled shapes), else 0 to signal "derive from DOWN_COL_TILE" (coupled
// shapes).  Self-contained here (no dependency on the detectors in
// moe_internal.h, which is included after this file).
template <typename Base>
struct base_explicit_uch_or_zero {
  template <typename D>
  static constexpr auto test(int)
      -> decltype((std::uint32_t)D::KernelConfig::UP_COL_HALVES) {
    return (std::uint32_t)D::KernelConfig::UP_COL_HALVES;
  }
  template <typename>
  static constexpr std::uint32_t test(...) {
    return 0u;
  }
  static constexpr std::uint32_t value = test<Base>(0);
};

// ── Tunable-config Dims wrapper ──────────────────────────────────────────
// Clones the SHAPE of `Base` (dims, quant granularity, scale-tensor extents)
// but overrides the tunable KernelConfig knobs.  One
// `moe_kernel_topk<DimsTunable<...>>` is instantiated per entry in the
// per-shape config table; the runtime dispatcher (moe_wrapper.cu) picks one
// by config_id.  config_id 0 always equals the bare `Base` Dims.
template <typename Base, std::uint32_t GRID, std::uint32_t DCT,
          std::uint32_t KUP, std::uint32_t KDN, std::uint32_t SLOTS>
struct DimsTunable {
  static constexpr uint32_t HIDDEN_STATES = Base::HIDDEN_STATES;
  static constexpr uint32_t K = Base::K;
  static constexpr uint32_t N = Base::N;
  static constexpr uint32_t BS = Base::BS;
  static constexpr uint32_t M = Base::M;
  static constexpr uint32_t NUM_EXPERTS = Base::NUM_EXPERTS;
  static constexpr QuantGranularity QUANT_GRAN = Base::QUANT_GRAN;
  static constexpr uint32_t BLOCK_SCALE_ROW = Base::BLOCK_SCALE_ROW;
  static constexpr uint32_t BLOCK_SCALE_COL = Base::BLOCK_SCALE_COL;
  static constexpr uint32_t UP_SCALE_ROWS = Base::UP_SCALE_ROWS;
  static constexpr uint32_t UP_SCALE_COLS = Base::UP_SCALE_COLS;
  static constexpr uint32_t DOWN_SCALE_ROWS = Base::DOWN_SCALE_ROWS;
  static constexpr uint32_t DOWN_SCALE_COLS = Base::DOWN_SCALE_COLS;
  struct KernelConfig {
    static constexpr std::uint32_t GRID_SIZE = GRID;
    static constexpr std::uint32_t BLOCK_SIZE = Base::KernelConfig::BLOCK_SIZE;
    static constexpr bool USE_WGMMA = Base::KernelConfig::USE_WGMMA;
    static constexpr bool USE_TMA = Base::KernelConfig::USE_TMA;
    static constexpr std::uint32_t K_STEP_DOWN = KDN;
    static constexpr std::uint32_t K_STEP_UP = KUP;
    static constexpr std::uint32_t DOWN_COL_TILE = DCT;
    static constexpr std::uint32_t UP_W_SLOTS = SLOTS;
    // Decoupled shapes pin UP_COL_HALVES on the Base — use it verbatim.
    // Coupled shapes derive it from THIS config's DCT (not Base's config-0
    // DCT), since one shape can mix DCT values across configs.
    static constexpr std::uint32_t UP_COL_HALVES =
        base_explicit_uch_or_zero<Base>::value != 0u
            ? base_explicit_uch_or_zero<Base>::value
            : (((2u * N * DCT) / (128u * HIDDEN_STATES) > 0u)
                   ? (2u * N * DCT) / (128u * HIDDEN_STATES)
                   : 1u);
    static constexpr bool USE_PAIR_LAYOUT = Base::KernelConfig::USE_PAIR_LAYOUT;
  };
};

// Scoring function enum for routing
enum class ScoringFunc : uint32_t {
  SIGMOID = 0,
  SOFTMAX = 1,
};

using W_element = __nv_fp8_e4m3;   // expert weights
using A_element = __nv_bfloat16;   // activations as they go into the GEMM
using AQ_element = __nv_fp8_e4m3;  // activations after quantization
using S_element = float;           // scaling factors
using R_element = __nv_bfloat16;   // MoE output

/**
 * @brief Returns the maximum amount of shared memory necessary to run
 * moe_kernel_topk()
 */
constexpr size_t get_moe_max_shmem_size();

/**
 * @brief Returns the maximum amount of global scratchpad memory to run
 * moe_kernel_topk()
 */
constexpr size_t get_moe_max_scratchpad_size();

/**
 * @brief W8A8 MoE kernel with configurable top-K routing, scoring function,
 *        and renormalization.
 *
 * Supports block-wise (128×128) FP8 quantization for the shapes declared in
 * shapes.json (e.g. Qwen3.5-35B/122B: softmax scoring, top_k=8, 256
 * experts).  See DESIGN.md for the architecture.
 *
 * @param [in] activations_in Input activations. Shape: [M, K]
 * @param [in] token_count Number of active tokens
 * @param [in] router_logits Router logits. Shape: [M, E]
 * @param [in] expert_weights_up Up-projection weights. Shape: [E, 2*N, K]
 * @param [in] expert_scales_up Up-projection scales.
 *             Per-channel: Shape [E, 2*N]
 *             Block-wise:  Shape [E, ceil(2*N/128), ceil(K/128)]
 * @param [in] expert_weights_down Down-projection weights. Shape: [E, K, N]
 * @param [in] expert_scales_down Down-projection scales.
 *             Per-channel: Shape [E, K]
 *             Block-wise:  Shape [E, ceil(K/128), ceil(N/128)]
 * @param [out] activations_out Output buffer. Shape: [M, K]
 * @param [out] scratchpad Global memory for temporary data
 * @param [in] scratchpad_size Size of the scratchpad
 * @param [in] shmem_size Size of the shared memory
 * @param [in] top_k Number of experts to select per token
 * @param [in] scoring_func Scoring function (SIGMOID or SOFTMAX)
 * @param [in] renormalize Whether to renormalize top-K weights to sum to 1
 * @param [in] expert_bias Optional per-expert selection bias [E]
 * @param [in] routed_scaling_factor Scalar folded into every routing weight
 */
template <typename Dims>
__global__ extern void moe_kernel_topk(
    const A_element* __restrict__ activations_in, std::uint32_t token_count,
    const __nv_bfloat16* __restrict__ router_logits,
    const W_element* __restrict expert_weights_up,
    const S_element* __restrict expert_scales_up,
    const W_element* __restrict expert_weights_down,
    const S_element* __restrict expert_scales_down,
    R_element* __restrict activations_out, void* __restrict__ scratchpad,
    size_t scratchpad_size, size_t shmem_size, std::uint32_t top_k,
    ScoringFunc scoring_func, bool renormalize,
    const float* __restrict__ expert_bias, float routed_scaling_factor,
    __grid_constant__ CUtensorMap const up_weights_desc,
    __grid_constant__ CUtensorMap const activations_desc,
    __grid_constant__ CUtensorMap const down_weights_desc,
    __grid_constant__ CUtensorMap const down_activations_desc,
    // EP dispatch args (inert when peer_activations==nullptr / n_local==0).
    std::uint32_t expert_base,
    const A_element* __restrict__ peer_activations,
    std::uint32_t local_token_start, std::uint32_t n_local_tokens);

}  // namespace moe_monokernel

#endif
