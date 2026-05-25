"""
MoE Monokernel Communication Integration

Provides the glue between the MoE model layer and the monokernel's
fused all-reduce. The model layer imports `MonokernelCommState` at init
time and passes its fields to `moe_monokernel_topk()` at each forward.

Usage in the model's MoE quantization layer:

    from vllm.model_executor.layers.fused_moe.monokernel_comm import (
        MonokernelCommState,
    )

    class MyMoELayer:
        def __init__(self, ...):
            # At model init, after TP group is established:
            self.comm_state = MonokernelCommState.create_if_tp(
                max_num_tokens=8,
                hidden_dim=2048,
            )

        def forward(self, x, router_logits, residual, norm_weight, ...):
            # Get comm kwargs (empty dict if tp_size=1)
            comm_kwargs = self.comm_state.get_kernel_kwargs(
                residual_in=residual,
                rms_gamma=norm_weight,
                rms_eps=1e-5,
            )

            output = moe_monokernel_topk(
                activations_in=x,
                router_logits=router_logits,
                ...,
                **comm_kwargs,
            )
            # When tp_size > 1: output is already all-reduced + residual + RMSNorm'd
            # When tp_size = 1: output is the per-rank partial (needs downstream AR)
            return output
"""

from typing import Optional

import torch

from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)


class MonokernelCommState:
    """
    Manages the LL workspace and flag counter for the monokernel's
    fused all-reduce.

    Lightweight wrapper around MoELLWorkspace that provides a clean
    interface for the model layer.
    """

    def __init__(
        self,
        max_num_tokens: int,
        hidden_dim: int,
        tp_size: int,
        tp_rank: int,
    ):
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.max_num_tokens = max_num_tokens
        self.hidden_dim = hidden_dim
        self._workspace = None

        if tp_size > 1:
            from vllm.distributed.moe_ll_workspace import MoELLWorkspace

            self._workspace = MoELLWorkspace(
                max_num_tokens=max_num_tokens,
                hidden_dim=hidden_dim,
                tp_group=get_tensor_model_parallel_group(),
            )

    @staticmethod
    def create_if_tp(
        max_num_tokens: int = 8,
        hidden_dim: int = 2048,
    ) -> "MonokernelCommState":
        """
        Factory that creates the comm state based on the current TP config.

        If tp_size == 1, the workspace is not allocated (no communication
        needed). The returned state's get_kernel_kwargs() will return
        defaults that make Phase 5 use the original bf16-cast path.
        """
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        return MonokernelCommState(
            max_num_tokens=max_num_tokens,
            hidden_dim=hidden_dim,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )

    def get_kernel_kwargs(
        self,
        residual_in: Optional[torch.Tensor] = None,
        rms_gamma: Optional[torch.Tensor] = None,
        rms_eps: float = 1e-5,
    ) -> dict:
        """
        Returns the keyword arguments to pass to moe_monokernel_topk().

        When tp_size == 1: returns defaults (None/0/1) that make the kernel
        use the original Phase 5 path (plain bf16 cast, no communication).

        When tp_size > 1: returns the workspace pointers, residual, gamma,
        and an incremented flag value for the LL exchange.
        """
        if self.tp_size <= 1 or self._workspace is None:
            # No communication — return defaults that trigger the
            # original Phase 5 path in the kernel
            return {
                "peer_ll_buffers": None,
                "residual_in": None,
                "rms_gamma": None,
                "rms_eps": 0.0,
                "ll_flag": 0,
                "tp_rank": 0,
                "tp_size": 1,
            }

        return {
            "peer_ll_buffers": self._workspace.peer_ll_buffers,
            "residual_in": residual_in,
            "rms_gamma": rms_gamma,
            "rms_eps": rms_eps,
            "ll_flag": self._workspace.next_flag(),
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
        }

    def reset_for_new_forward(self) -> None:
        """
        Call at the start of each forward pass to reset the flag counter
        and clear stale LL buffer data.
        """
        if self._workspace is not None:
            self._workspace.reset_flags()

    @property
    def is_fused_ar_enabled(self) -> bool:
        """Whether the fused AR path is active (tp_size > 1)."""
        return self.tp_size > 1 and self._workspace is not None

    def destroy(self) -> None:
        """Free resources."""
        if self._workspace is not None:
            self._workspace.destroy()
            self._workspace = None
