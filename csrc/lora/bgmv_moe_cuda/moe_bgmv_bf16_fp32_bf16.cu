#include "moe_bgmv_config.h"
#include "moe_bgmv_impl.cuh"

// Expand only (in_T=nv_bfloat16, Y=float32, W_T=nv_bfloat16).
// Shrink is covered by nv_bfloat16_nv_bfloat16_nv_bfloat16.cu.

#define INST_MOE_BGMV_EXPAND_ONLY(in_T, out_T, W_T, narrow, wide) \
  INST_MOE_BGMV_EXPAND_SLICED(narrow, wide, in_T, W_T)

FOR_MOE_ALL_WIDE_NARROW(INST_MOE_BGMV_EXPAND_ONLY, nv_bfloat16, float, nv_bfloat16)
