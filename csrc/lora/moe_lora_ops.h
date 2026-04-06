#pragma once
#include <torch/all.h>

void dispatch_moe_shrink(torch::Tensor y, torch::Tensor x,
                         torch::Tensor w_ptr,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor expert_ids,
                         torch::Tensor lora_indices);

void dispatch_moe_expand(torch::Tensor y, torch::Tensor x,
                         torch::Tensor w_ptr,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor expert_ids,
                         torch::Tensor topk_weights,
                         torch::Tensor lora_indices,
                         torch::Tensor slice_start_loc,
                         std::vector<int64_t> output_slices);
