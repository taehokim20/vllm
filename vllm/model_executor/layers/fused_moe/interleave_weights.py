"""
Weight interleaving for the MoE monokernel's TMA+WGMMA path.

The TMA descriptor expects up-projection weights in a specific interleaved
layout so that a single 128x128 TMA fetch retrieves a full WGMMA A-tile
containing both gate and up rows.

Original layout (per expert): [gate[0..N_half), up[0..N_half)] along dim=1
  i.e., rows 0..N_half-1 are gate, rows N_half..N-1 are up.

Interleaved layout (per expert): for every 64-gate-row block k in [0, N_half/64):
  new_rows[128k +  0 .. 128k + 32) = gate[64k     .. 64k + 32)
  new_rows[128k + 32 .. 128k + 64) =   up[64k     .. 64k + 32)
  new_rows[128k + 64 .. 128k + 96) = gate[64k + 32 .. 64k + 64)
  new_rows[128k + 96 .. 128k +128) =   up[64k + 32 .. 64k + 64)

Total rows remain 2*N_half = N (same footprint).
"""

import torch


def interleave_for_tma_wgmma_up(weights: torch.Tensor) -> torch.Tensor:
    """
    Repack fused gate+up weights for the TMA+WGMMA monokernel path.

    Args:
        weights: [E, N, K] fp8 tensor where N = 2 * N_half.
                 Rows [0, N_half) are gate weights.
                 Rows [N_half, N) are up weights.

    Returns:
        [E, N, K] tensor with rows interleaved per the TMA layout spec.
    """
    E, N, K = weights.shape
    assert N % 2 == 0, f"N must be even (gate+up), got N={N}"
    N_half = N // 2
    assert N_half % 64 == 0, (
        f"N_half must be divisible by 64 for 128-row TMA tiles, got N_half={N_half}"
    )

    gate = weights[:, :N_half, :]       # [E, N_half, K]
    up = weights[:, N_half:, :]         # [E, N_half, K]

    num_blocks = N_half // 64
    # Reshape into 64-row blocks
    gate_blocks = gate.reshape(E, num_blocks, 64, K)  # [E, num_blocks, 64, K]
    up_blocks = up.reshape(E, num_blocks, 64, K)      # [E, num_blocks, 64, K]

    # Split each 64-row block into two 32-row halves
    gate_lo = gate_blocks[:, :, :32, :]   # [E, num_blocks, 32, K]
    gate_hi = gate_blocks[:, :, 32:, :]   # [E, num_blocks, 32, K]
    up_lo = up_blocks[:, :, :32, :]       # [E, num_blocks, 32, K]
    up_hi = up_blocks[:, :, 32:, :]       # [E, num_blocks, 32, K]

    # Interleave: [gate_lo, up_lo, gate_hi, up_hi] per block
    # Each block becomes 128 rows
    interleaved = torch.cat([gate_lo, up_lo, gate_hi, up_hi], dim=2)
    # interleaved shape: [E, num_blocks, 128, K]

    # Reshape back to [E, N, K]
    return interleaved.reshape(E, N, K)
