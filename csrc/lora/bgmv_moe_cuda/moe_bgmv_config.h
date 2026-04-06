#pragma once

// clang-format off

// ===== Unified dimension set (union of all model wide values, deduplicated) =====
//
// Models covered:
//   GPT-OSS-120B:             gate_up=(3072,5888), down=(2944,3072)
//   GPT-OSS-20B:              gate_up=(3072,5888), down=(2944,3072)  [same dims as 120B]
//   Qwen3-30B-A3B:            gate_up=(2048,768),  down=(768,2048)
//   Nemotron-Nano-3-30B-A3B:  gate_up=(2688,1856), down=(1856,2688)
//   Nemotron-3-Super-120B-A12B: gate_up=(4096,2688), down=(2688,4096)
//
// Legacy dims kept for backward compat: 1024, 2880, 5120, 7168, 8192, 10240, 14336, 16384, 28672
//
// Union (sorted, deduplicated):
#define FOR_MOE_ALL_WIDE(f, in_T, out_T, W_T, narrow) \
    f(in_T, out_T, W_T, narrow, 768)   \
    f(in_T, out_T, W_T, narrow, 1024)  \
    f(in_T, out_T, W_T, narrow, 1856)  \
    f(in_T, out_T, W_T, narrow, 2048)  \
    f(in_T, out_T, W_T, narrow, 2688)  \
    f(in_T, out_T, W_T, narrow, 2880)  \
    f(in_T, out_T, W_T, narrow, 2944)  \
    f(in_T, out_T, W_T, narrow, 3072)  \
    f(in_T, out_T, W_T, narrow, 4096)  \
    f(in_T, out_T, W_T, narrow, 5120)  \
    f(in_T, out_T, W_T, narrow, 5888)  \
    f(in_T, out_T, W_T, narrow, 7168)  \
    f(in_T, out_T, W_T, narrow, 8192)  \
    f(in_T, out_T, W_T, narrow, 10240) \
    f(in_T, out_T, W_T, narrow, 14336) \
    f(in_T, out_T, W_T, narrow, 16384) \
    f(in_T, out_T, W_T, narrow, 28672)

#define FOR_MOE_ALL_WIDE_NARROW(f, in_T, out_T, W_T) \
    FOR_MOE_ALL_WIDE(f, in_T, out_T, W_T, 8)  \
    FOR_MOE_ALL_WIDE(f, in_T, out_T, W_T, 16) \
    FOR_MOE_ALL_WIDE(f, in_T, out_T, W_T, 32) \
    FOR_MOE_ALL_WIDE(f, in_T, out_T, W_T, 64)

// clang-format on

// ===== Forward declarations =====

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
                             float scale);

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
                             float scale);
