# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Swappable one-sided signaling backend for the monokernel EP all-to-all hiding.

The monokernel's in-kernel dispatch peer-read and one-sided combine peer-reduce
need three things from the communication substrate:

  * symmetric staging views  — where a rank writes its owned rows and where it
                               peer-reads a remote rank's rows;
  * a readiness flag region  — the one-sided store/poll handshake that orders a
                               peer-read after the peer's write;
  * a CUDA-graph-safe epoch  — a device-resident, per-step monotonic counter the
                               in-kernel signaling reads as its flag base.

This module hides the substrate behind ``EpSignalBackend`` so the current
single-node NVLink path (raw PTX over CUDA-IPC peer memory, ``LLIpcSignalBackend``)
can be swapped for an NVSHMEM backend later without touching the kernel wiring.
Backend selection is via ``VLLM_MOE_EP_SIGNAL_BACKEND`` (default: ``ll_ipc``).

Scope: single node, EP=2 (single peer, ``rank ^ 1``). The interface is written
for general ep_size; the current kernel path targets the single-peer case.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

import torch


class EpSignalBackend(ABC):
    """Substrate-agnostic handle for the monokernel EP signaling."""

    ep_size: int
    ep_rank: int

    @abstractmethod
    def dispatch_views(self, num_tokens: int):
        """Return (local_view, peer_views): bf16 [num_tokens, hidden] staging
        for the dispatch peer-read. peer_views[ep_rank] is local_view."""

    @abstractmethod
    def combine_views(self, num_tokens: int):
        """Return (local_view, peer_views) for the combine peer-reduce, at a
        region distinct from dispatch so a peer-read cannot be clobbered."""

    @abstractmethod
    def flag_buffers(self) -> torch.Tensor:
        """int64 [ep_size]: peer-mapped pointer to each rank's readiness-flag
        slot, for the in-kernel ep_set_ready / ep_wait_ready handshake."""

    @abstractmethod
    def epoch_ptr(self) -> int:
        """Device pointer to the int32 CUDA-graph-safe epoch counter."""

    @abstractmethod
    def advance_epoch(self) -> None:
        """Advance the epoch by one. MUST be issued once per forward step on
        the capture path (as a captured op) so every replay bumps it."""

    def peer_of(self, rank: Optional[int] = None) -> int:
        """The single EP peer of ``rank`` (EP=2). Generalizes to a peer set
        for ep_size > 2 via peer_views, but the kernel path assumes one peer."""
        r = self.ep_rank if rank is None else rank
        return r ^ 1

    @abstractmethod
    def destroy(self) -> None:
        ...


class LLIpcSignalBackend(EpSignalBackend):
    """NVLink CUDA-IPC substrate: raw PTX put/poll over symmetric IPC memory,
    with a device-memory epoch for CUDA-graph safety. Backed by
    ``MoELLWorkspace``."""

    def __init__(self, max_num_tokens: int, hidden_dim: int, ep_group=None):
        from vllm.distributed.moe_ll_workspace import MoELLWorkspace

        if ep_group is None:
            from vllm.distributed.parallel_state import get_ep_group
            ep_group = get_ep_group().device_group

        self._ws = MoELLWorkspace(
            max_num_tokens=max_num_tokens,
            hidden_dim=hidden_dim,
            tp_group=ep_group,
        )
        self.ep_size = self._ws.tp_size
        self.ep_rank = self._ws.tp_rank

    def dispatch_views(self, num_tokens: int):
        return self._ws.ep_activation_views(num_tokens)

    def combine_views(self, num_tokens: int):
        return self._ws.ep_combine_views(num_tokens)

    def flag_buffers(self) -> torch.Tensor:
        return self._ws.ep_flag_buffers()

    def epoch_ptr(self) -> int:
        return self._ws.epoch_ptr()

    def advance_epoch(self) -> None:
        self._ws.advance_epoch()

    def destroy(self) -> None:
        self._ws.destroy()


def create_ep_signal_backend(
    max_num_tokens: int,
    hidden_dim: int,
    ep_group=None,
) -> Optional[EpSignalBackend]:
    """Create the EP signaling backend, or None when EP is inactive (ep_size==1).

    Backend chosen by ``VLLM_MOE_EP_SIGNAL_BACKEND``:
      * ``ll_ipc`` (default) — single-node NVLink CUDA-IPC + PTX (this file).
      * ``nvshmem``          — reserved; not implemented (single-node target
                               makes it a follow-on; see the design notes).
    """
    from vllm.distributed.parallel_state import (
        get_ep_group,
        get_tensor_model_parallel_world_size,
    )

    # EP is only meaningful when there is more than one rank in the EP group.
    try:
        ep_world = get_ep_group().world_size
    except Exception:
        ep_world = get_tensor_model_parallel_world_size()
    if ep_world <= 1:
        return None

    backend = os.environ.get("VLLM_MOE_EP_SIGNAL_BACKEND", "ll_ipc").lower()
    if backend == "ll_ipc":
        return LLIpcSignalBackend(max_num_tokens, hidden_dim, ep_group)
    if backend == "nvshmem":
        raise NotImplementedError(
            "NVSHMEM EP signaling backend is not implemented. The single-node "
            "target makes raw NVLink CUDA-IPC (ll_ipc) the lower-risk substrate; "
            "NVSHMEM is the documented follow-on for multi-node/scale-out."
        )
    raise ValueError(
        f"Unknown VLLM_MOE_EP_SIGNAL_BACKEND={backend!r} "
        f"(expected 'll_ipc' or 'nvshmem')."
    )
