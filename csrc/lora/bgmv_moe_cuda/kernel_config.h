#pragma once

// ── Base configuration (all architectures) ──
// Improving point 1: RANK_TILE tiling.
struct MoeShrinkKernelConfig {
    static constexpr int tx = 32;       // threads per warp (x-dimension)
    static constexpr int ty = 4;        // number of warps (y-dimension)
    static constexpr int vec_size = 8;  // elements per vectorized load
    static constexpr int rank_tile = 4; // rank elements per block

    // ── Improving point 2: multi-pair decode path ──
    // PPB=1 for prefill (grid already saturates GPU).
    // PPB=4 for decode on sm_80+ (uses dynamic shared memory).
    // PPB=1 for decode on sm_70/75 (48 KB static shmem limit).
    static constexpr int pairs_per_block_prefill = 1;
    static constexpr int pairs_per_block_decode = 4;
    static constexpr int decode_threshold = 32;

    // ── Improving point 3: deeper pipeline on sm_80+ ──
    // 3 stages on sm_80+ (more shmem available via opt-in).
    // 2 stages on sm_70/75 (48 KB static limit).
    static constexpr int num_stages_default = 2;
    static constexpr int num_stages_extended = 3;
};

struct MoeExpandKernelConfig {
    static constexpr int tz = 4;
    static constexpr int vec_size = 8;
};
