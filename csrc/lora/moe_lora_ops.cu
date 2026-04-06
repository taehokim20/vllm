#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <vector>

#include "bgmv_moe_cuda/moe_bgmv_config.h"

//====== Utils ======

inline constexpr uint64_t pack_u32(uint32_t a, uint32_t b) {
  return (uint64_t(a) << 32) | uint64_t(b);
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_DIM(d, x) \
  TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")
#define CHECK_EQ(a, b) \
  TORCH_CHECK(a == b, "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

//====== MoE BGMV Shrink Launcher ======

template <typename T>
inline bool launch_moe_shrink_sliced_kernel(
    T *Y, const T *X, T **w_ptr,
    const int64_t *sorted_token_ids, const int64_t *expert_ids,
    const int64_t *lora_indices,
    uint32_t feat_in, uint32_t feat_out,
    int64_t num_pairs, int64_t num_slices, int64_t num_experts,
    int64_t num_tokens) {

  switch (pack_u32(feat_in, feat_out)) {
#define CASE_MOE_SHRINK(in_T, out_T, W_T, narrow, wide)                    \
  case pack_u32(wide, narrow):                                              \
    moe_bgmv_shrink_sliced<wide, narrow, in_T, out_T, W_T>(                \
        Y, X, w_ptr, sorted_token_ids, expert_ids, lora_indices,            \
        num_pairs, num_slices, num_experts, num_tokens, 1.0f);              \
    return true;
    FOR_MOE_ALL_WIDE_NARROW(CASE_MOE_SHRINK, T, T, T)
#undef CASE_MOE_SHRINK
  default:
    return false;
  }
}

//====== MoE BGMV Expand Launcher ======

template <typename T>
inline bool launch_moe_expand_sliced_kernel(
    float *Y, const T *X, T **w_ptr,
    const int64_t *sorted_token_ids, const int64_t *expert_ids,
    const int64_t *lora_indices, const float *topk_weights,
    const int64_t *slice_start_loc,
    uint32_t feat_in, uint32_t feat_out,
    int64_t num_pairs, int64_t num_slices, int64_t num_experts,
    int64_t total_feat_out, int64_t num_tokens) {

  switch (pack_u32(feat_in, feat_out)) {
#define CASE_MOE_EXPAND(in_T, out_T, W_T, narrow, wide)                    \
  case pack_u32(narrow, wide):                                              \
    moe_bgmv_expand_sliced<narrow, wide, in_T, W_T>(                       \
        Y, X, w_ptr, sorted_token_ids, expert_ids, lora_indices,            \
        topk_weights, slice_start_loc, num_pairs, num_slices,               \
        num_experts, total_feat_out, wide, num_tokens, 1.0f);              \
    return true;
    FOR_MOE_ALL_WIDE_NARROW(CASE_MOE_EXPAND, T, T, T)
#undef CASE_MOE_EXPAND
  default:
    return false;
  }
}

//====== Public dispatch: MoE Shrink ======

void dispatch_moe_shrink(torch::Tensor y, torch::Tensor x,
                         torch::Tensor w_ptr,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor expert_ids,
                         torch::Tensor lora_indices) {
  CHECK_INPUT(y); CHECK_INPUT(x); CHECK_INPUT(w_ptr);
  CHECK_INPUT(sorted_token_ids); CHECK_INPUT(expert_ids); CHECK_INPUT(lora_indices);
  CHECK_DIM(3, y); CHECK_DIM(2, x); CHECK_DIM(2, w_ptr);
  CHECK_DIM(1, sorted_token_ids); CHECK_DIM(1, expert_ids); CHECK_DIM(1, lora_indices);

  int64_t num_slices  = y.size(0);
  int64_t num_pairs   = sorted_token_ids.size(0);
  int64_t num_tokens  = lora_indices.size(0);
  int64_t feat_in     = x.size(1);
  int64_t feat_out    = y.size(2);
  int64_t num_experts = w_ptr.size(1);

  CHECK_EQ(w_ptr.size(0), num_slices);
  TORCH_CHECK(w_ptr.scalar_type() == at::ScalarType::Long);
  TORCH_CHECK(lora_indices.scalar_type() == at::ScalarType::Long);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  bool ok = false;

  switch (x.scalar_type()) {
  case at::ScalarType::Half:
    ok = launch_moe_shrink_sliced_kernel(
        static_cast<nv_half *>(y.data_ptr()),
        static_cast<nv_half *>(x.data_ptr()),
        reinterpret_cast<nv_half **>(w_ptr.data_ptr<int64_t>()),
        sorted_token_ids.data_ptr<int64_t>(),
        expert_ids.data_ptr<int64_t>(),
        lora_indices.data_ptr<int64_t>(),
        feat_in, feat_out, num_pairs, num_slices, num_experts, num_tokens);
    break;
  case at::ScalarType::BFloat16:
    ok = launch_moe_shrink_sliced_kernel(
        static_cast<nv_bfloat16 *>(y.data_ptr()),
        static_cast<nv_bfloat16 *>(x.data_ptr()),
        reinterpret_cast<nv_bfloat16 **>(w_ptr.data_ptr<int64_t>()),
        sorted_token_ids.data_ptr<int64_t>(),
        expert_ids.data_ptr<int64_t>(),
        lora_indices.data_ptr<int64_t>(),
        feat_in, feat_out, num_pairs, num_slices, num_experts, num_tokens);
    break;
  default:
    TORCH_CHECK(false, "MoE shrink: unsupported dtype: ", x.scalar_type());
  }

  TORCH_CHECK(ok, "MoE shrink failed. feat_in=", feat_in, " feat_out=", feat_out);
}

//====== Public dispatch: MoE Expand ======

void dispatch_moe_expand(torch::Tensor y, torch::Tensor x,
                         torch::Tensor w_ptr,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor expert_ids,
                         torch::Tensor topk_weights,
                         torch::Tensor lora_indices,
                         torch::Tensor slice_start_loc,
                         std::vector<int64_t> output_slices) {
  CHECK_INPUT(y); CHECK_INPUT(x); CHECK_INPUT(w_ptr);
  CHECK_INPUT(sorted_token_ids); CHECK_INPUT(expert_ids);
  CHECK_INPUT(topk_weights); CHECK_INPUT(lora_indices); CHECK_INPUT(slice_start_loc);
  CHECK_DIM(2, y); CHECK_DIM(3, x); CHECK_DIM(2, w_ptr);

  int64_t num_slices     = x.size(0);
  int64_t num_pairs      = sorted_token_ids.size(0);
  int64_t num_tokens     = lora_indices.size(0);
  int64_t feat_in        = x.size(2);
  int64_t total_feat_out = y.size(1);
  int64_t num_experts    = w_ptr.size(1);

  CHECK_EQ(w_ptr.size(0), num_slices);
  TORCH_CHECK(w_ptr.scalar_type() == at::ScalarType::Long);
  TORCH_CHECK(lora_indices.scalar_type() == at::ScalarType::Long);
  TORCH_CHECK(slice_start_loc.scalar_type() == at::ScalarType::Long);
  TORCH_CHECK(topk_weights.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(y.scalar_type() == at::ScalarType::Float,
              "MoE expand: y must be float32 accumulation buffer");

  int32_t first_feat_out = static_cast<int32_t>(output_slices[0]);
  for (size_t i = 1; i < output_slices.size(); ++i) {
    TORCH_CHECK(output_slices[i] == first_feat_out,
                "MoE expand: all output_slices must be equal");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  bool ok = false;

  switch (x.scalar_type()) {
  case at::ScalarType::Half:
    ok = launch_moe_expand_sliced_kernel(
        static_cast<float *>(y.data_ptr()),
        static_cast<nv_half *>(x.data_ptr()),
        reinterpret_cast<nv_half **>(w_ptr.data_ptr<int64_t>()),
        sorted_token_ids.data_ptr<int64_t>(),
        expert_ids.data_ptr<int64_t>(),
        lora_indices.data_ptr<int64_t>(),
        topk_weights.data_ptr<float>(),
        slice_start_loc.data_ptr<int64_t>(),
        feat_in, first_feat_out, num_pairs, num_slices, num_experts,
        total_feat_out, num_tokens);
    break;
  case at::ScalarType::BFloat16:
    ok = launch_moe_expand_sliced_kernel(
        static_cast<float *>(y.data_ptr()),
        static_cast<nv_bfloat16 *>(x.data_ptr()),
        reinterpret_cast<nv_bfloat16 **>(w_ptr.data_ptr<int64_t>()),
        sorted_token_ids.data_ptr<int64_t>(),
        expert_ids.data_ptr<int64_t>(),
        lora_indices.data_ptr<int64_t>(),
        topk_weights.data_ptr<float>(),
        slice_start_loc.data_ptr<int64_t>(),
        feat_in, first_feat_out, num_pairs, num_slices, num_experts,
        total_feat_out, num_tokens);
    break;
  default:
    TORCH_CHECK(false, "MoE expand: unsupported dtype: ", x.scalar_type());
  }

  TORCH_CHECK(ok, "MoE expand failed. feat_in=", feat_in, " feat_out=", first_feat_out);
}
