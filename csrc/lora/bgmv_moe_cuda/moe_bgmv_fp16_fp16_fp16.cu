#include "moe_bgmv_config.h"
#include "moe_bgmv_impl.cuh"

// Shrink + expand (in_T=out_T=W_T=nv_half).
// Uses FOR_MOE_ALL_WIDE_NARROW to cover the union of W13+W2 dims without duplicates.
FOR_MOE_ALL_WIDE_NARROW(INST_MOE_BGMV_TWOSIDE, nv_half, nv_half, nv_half)
