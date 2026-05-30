"""
MoE Monokernel LL (Low Latency) Workspace

Allocates symmetric IPC-registered buffers for the fused all-reduce
in the monokernel's Phase 5. Each rank allocates one LL buffer and
exchanges IPC handles with all peers so every rank can write to every
other rank's buffer via NVLink-mapped memory.

Usage (at model init, once per process):

    from vllm.distributed.moe_ll_workspace import MoELLWorkspace

    workspace = MoELLWorkspace(
        max_num_tokens=8,
        hidden_dim=2048,
        tp_group=get_tp_group().device_group,
    )

    # Pass to the monokernel:
    output = moe_monokernel_topk(
        ...,
        peer_ll_buffers=workspace.peer_ll_buffers,
        residual_in=residual,
        rms_gamma=norm_weight,
        rms_eps=1e-5,
        ll_flag=workspace.next_flag(),
        tp_rank=workspace.tp_rank,
        tp_size=workspace.tp_size,
    )
"""

import torch
import torch.distributed as dist
from typing import Optional


class MoELLWorkspace:
    """
    Symmetric LL buffer workspace for the monokernel's fused all-reduce.

    Each rank allocates a local LL buffer large enough to receive data from
    all peers. The buffer is registered for IPC access so peers can write
    to it directly via NVLink using storeLL.

    Buffer layout per rank:
        [tp_size, max_num_tokens, hidden_dim / LL_ELEMS_PER_PACKET] LLPackets
        where each LLPacket is 16 bytes (4 bf16 payload + flags).

    The `peer_ll_buffers` tensor is a [tp_size] int64 tensor where entry[i]
    is the device pointer to rank i's LL buffer, as seen from this rank.
    """

    # Each LL packet carries 4 bf16 values = 8 bytes of payload
    LL_ELEMS_PER_PACKET = 4
    LL_PACKET_BYTES = 16  # sizeof(LLPacket)

    def __init__(
        self,
        max_num_tokens: int,
        hidden_dim: int,
        tp_group: Optional[dist.ProcessGroup] = None,
    ):
        """
        Args:
            max_num_tokens: Maximum batch size the monokernel will process.
            hidden_dim: Hidden dimension (K) of the model.
            tp_group: The tensor-parallel process group. If None, uses the
                      default process group.
        """
        if tp_group is None:
            tp_group = dist.group.WORLD

        self.tp_size = dist.get_world_size(group=tp_group)
        self.tp_rank = dist.get_rank(group=tp_group)
        self.tp_group = tp_group
        self.max_num_tokens = max_num_tokens
        self.hidden_dim = hidden_dim

        # Number of LL packets per token row
        assert hidden_dim % self.LL_ELEMS_PER_PACKET == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by "
            f"LL_ELEMS_PER_PACKET ({self.LL_ELEMS_PER_PACKET})"
        )
        self.packets_per_row = hidden_dim // self.LL_ELEMS_PER_PACKET

        # Total buffer size per rank:
        # Each rank's buffer holds data FROM all tp_size peers
        # (each peer writes to a different slot indexed by its rank).
        # Shape: [tp_size, max_num_tokens, packets_per_row] in LLPacket units
        self.buffer_num_packets = self.tp_size * max_num_tokens * self.packets_per_row
        self.buffer_size_bytes = self.buffer_num_packets * self.LL_PACKET_BYTES

        # Allocate local buffer (uint8 to avoid dtype interpretation)
        self._local_buffer = torch.zeros(
            self.buffer_size_bytes, dtype=torch.uint8, device="cuda"
        )

        # Flag counter — monotonically increasing per layer invocation
        self._flag_counter: int = 0

        # Exchange IPC handles to get peer pointers
        self._peer_buffers: list[int] = []  # device pointers as ints
        self._setup_ipc()

        # Build the peer_ll_buffers tensor that gets passed to the kernel
        self.peer_ll_buffers = torch.tensor(
            self._peer_buffers, dtype=torch.int64, device="cuda"
        )

    def _setup_ipc(self) -> None:
        """Exchange IPC memory handles with all peers in the TP group."""
        if self.tp_size == 1:
            # No peers — just store our own pointer
            self._peer_buffers = [self._local_buffer.data_ptr()]
            return

        import ctypes

        # Load CUDA runtime
        cudart = ctypes.CDLL("libcudart.so")

        # CUDA IPC handle is 64 bytes (cudaIpcMemHandle_t)
        IPC_HANDLE_SIZE = 64

        class CudaIpcMemHandle(ctypes.Structure):
            _fields_ = [("reserved", ctypes.c_byte * IPC_HANDLE_SIZE)]

        # Set proper argument/return types
        cudart.cudaIpcGetMemHandle.restype = ctypes.c_int
        cudart.cudaIpcGetMemHandle.argtypes = [
            ctypes.POINTER(CudaIpcMemHandle),
            ctypes.c_void_p,
        ]
        cudart.cudaIpcOpenMemHandle.restype = ctypes.c_int
        cudart.cudaIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            CudaIpcMemHandle,
            ctypes.c_uint,
        ]

        # Get IPC handle for our local buffer
        local_handle = CudaIpcMemHandle()
        err = cudart.cudaIpcGetMemHandle(
            ctypes.byref(local_handle),
            ctypes.c_void_p(self._local_buffer.data_ptr()),
        )
        if err != 0:
            raise RuntimeError(f"cudaIpcGetMemHandle failed with error {err}")

        # Exchange IPC handles via all_gather
        handle_bytes = bytes(local_handle.reserved)
        local_handle_tensor = torch.tensor(
            list(handle_bytes), dtype=torch.uint8, device="cuda"
        )
        all_handles = [
            torch.zeros(IPC_HANDLE_SIZE, dtype=torch.uint8, device="cuda")
            for _ in range(self.tp_size)
        ]
        dist.all_gather(all_handles, local_handle_tensor, group=self.tp_group)

        # Open each peer's IPC handle to get a local pointer to their memory
        self._peer_buffers = []
        self._ipc_ptrs = []  # Keep references to prevent GC
        for i in range(self.tp_size):
            if i == self.tp_rank:
                # Use our own local pointer directly
                self._peer_buffers.append(self._local_buffer.data_ptr())
            else:
                # Open peer's IPC handle
                peer_handle_bytes = all_handles[i].cpu().numpy().tobytes()
                peer_handle = CudaIpcMemHandle()
                ctypes.memmove(
                    ctypes.byref(peer_handle.reserved),
                    peer_handle_bytes,
                    IPC_HANDLE_SIZE,
                )
                peer_ptr = ctypes.c_void_p()
                err = cudart.cudaIpcOpenMemHandle(
                    ctypes.byref(peer_ptr),
                    peer_handle,
                    1,  # cudaIpcMemLazyEnablePeerAccess
                )
                if err != 0:
                    raise RuntimeError(
                        f"cudaIpcOpenMemHandle for rank {i} failed with error {err}"
                    )
                self._peer_buffers.append(peer_ptr.value)
                self._ipc_ptrs.append(peer_ptr)

    def next_flag(self) -> int:
        """
        Get the next flag value and increment the counter.

        Called once per layer invocation. The flag value is passed to the
        kernel so storeLL/readLL can distinguish between invocations.
        Must be called in the same order on all ranks.
        """
        self._flag_counter += 1
        return self._flag_counter

    def reset_flags(self) -> None:
        """
        Reset the flag counter (e.g., at the start of each forward pass).

        Also zeros the LL buffer to clear stale flags from the previous
        forward pass. This ensures readLL won't accidentally match an old
        flag value.
        """
        self._flag_counter = 0
        self._local_buffer.zero_()

    def destroy(self) -> None:
        """Free resources."""
        del self._local_buffer
        del self.peer_ll_buffers
        self._peer_buffers = []

    @property
    def local_buffer(self) -> torch.Tensor:
        """The raw local LL buffer (for debugging/testing)."""
        return self._local_buffer
