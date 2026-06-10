"""
Side-channel for passing residual + RMSNorm weight into the monokernel's
fused AR+residual+RMSNorm Phase 5.

The decoder layer sets the residual and next-layer norm weight before
calling the MoE layer. The Fp8MoEMethod.apply_monolithic reads them
and passes them to the kernel. After the kernel returns, the decoder
layer reads back the updated residual.

This avoids changing the FusedMoE/runner/SparseMoeBlock interfaces.
"""

import torch
from typing import Optional


# Module-level state (single-threaded model forward in each process)
_fused_norm_residual_in: Optional[torch.Tensor] = None
_fused_norm_residual_out: Optional[torch.Tensor] = None
_fused_norm_gamma: Optional[torch.Tensor] = None
_fused_norm_eps: float = 0.0
_fused_norm_active: bool = False
_fused_norm_did_fuse: bool = False  # Set by apply_monolithic after kernel runs


def set_fused_norm_inputs(
    residual: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
) -> None:
    """Called by the decoder layer before MoE forward."""
    global _fused_norm_residual_in, _fused_norm_gamma, _fused_norm_eps
    global _fused_norm_active, _fused_norm_residual_out, _fused_norm_did_fuse
    _fused_norm_residual_in = residual
    _fused_norm_gamma = gamma
    _fused_norm_eps = eps
    _fused_norm_active = True
    _fused_norm_did_fuse = False
    # Pre-allocate residual_out with same shape as residual
    _fused_norm_residual_out = torch.empty_like(residual)


def get_fused_norm_inputs() -> tuple[
    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], float
]:
    """Called by apply_monolithic to get (residual_in, residual_out, gamma, eps)."""
    global _fused_norm_active
    if not _fused_norm_active:
        return None, None, None, 0.0
    return (
        _fused_norm_residual_in,
        _fused_norm_residual_out,
        _fused_norm_gamma,
        _fused_norm_eps,
    )


def mark_fused() -> None:
    """Called by apply_monolithic after the kernel ran the fused path."""
    global _fused_norm_did_fuse
    _fused_norm_did_fuse = True


def did_fuse() -> bool:
    """Called by the decoder layer to check if fusion actually happened."""
    return _fused_norm_did_fuse


def get_fused_norm_residual_out() -> Optional[torch.Tensor]:
    """Called by the decoder layer after MoE forward to get the new residual."""
    return _fused_norm_residual_out


def clear_fused_norm() -> None:
    """Called after MoE forward to reset state."""
    global _fused_norm_residual_in, _fused_norm_residual_out
    global _fused_norm_gamma, _fused_norm_eps, _fused_norm_active
    global _fused_norm_did_fuse
    _fused_norm_residual_in = None
    _fused_norm_residual_out = None
    _fused_norm_gamma = None
    _fused_norm_eps = 0.0
    _fused_norm_active = False
    _fused_norm_did_fuse = False
