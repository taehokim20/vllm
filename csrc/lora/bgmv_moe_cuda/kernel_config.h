#pragma once

struct MoeShrinkKernelConfig {
    static constexpr int tx = 32;
    static constexpr int ty = 4;
    static constexpr int vec_size = 8;
};

struct MoeExpandKernelConfig {
    static constexpr int tz = 4;
    static constexpr int vec_size = 8;
};
