// cuBLAS bf16 x bf16 -> fp32 router GEMM fallback
// Computes output = input @ weight.T in fp32 output precision.

#include <torch/all.h>

torch::Tensor router_gemm_bf16_fp32(torch::Tensor const& input,
                                    torch::Tensor const& weight) {
  // input:  [num_tokens, hidden_dim] bf16
  // weight: [num_experts, hidden_dim] bf16
  // output: [num_tokens, num_experts] fp32
  TORCH_CHECK(input.is_cuda() && weight.is_cuda(),
              "router_gemm_bf16_fp32: inputs must be CUDA tensors");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 &&
                  weight.scalar_type() == at::kBFloat16,
              "router_gemm_bf16_fp32: inputs must be bf16");

  // Use torch::matmul with fp32 output via cuBLAS
  auto input_fp32 = input.to(at::kFloat);
  auto weight_fp32 = weight.to(at::kFloat);
  return torch::mm(input_fp32, weight_fp32.t());
}
