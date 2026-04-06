#include "../core/registration.h"
#include "moe_lora_ops.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def(
      "dispatch_moe_shrink(Tensor! y, Tensor x, Tensor w_ptr, "
      "Tensor sorted_token_ids, Tensor expert_ids, "
      "Tensor lora_indices) -> ()");
  m.impl("dispatch_moe_shrink", torch::kCUDA, &dispatch_moe_shrink);

  m.def(
      "dispatch_moe_expand(Tensor! y, Tensor x, Tensor w_ptr, "
      "Tensor sorted_token_ids, Tensor expert_ids, "
      "Tensor topk_weights, Tensor lora_indices, "
      "Tensor slice_start_loc, int[] output_slices) -> ()");
  m.impl("dispatch_moe_expand", torch::kCUDA, &dispatch_moe_expand);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
