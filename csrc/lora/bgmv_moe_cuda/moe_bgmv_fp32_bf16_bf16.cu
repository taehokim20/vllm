#include "moe_bgmv_config.h"
#include "moe_bgmv_impl.cuh"

// Shrink only (in_T=float, out_T=nv_bfloat16, W_T=nv_bfloat16).
// Expand is covered by nv_bfloat16_fp32_nv_bfloat16.cu.

#define INST_MOE_BGMV_SHRINK_ONLY(in_T, out_T, W_T, narrow, wide) \
  INST_MOE_BGMV_SHRINK_SLICED(wide, narrow, in_T, out_T, W_T)

FOR_MOE_ALL_WIDE_NARROW(INST_MOE_BGMV_SHRINK_ONLY, float, nv_bfloat16, nv_bfloat16)
