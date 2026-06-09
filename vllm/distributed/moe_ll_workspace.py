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
        """Setup cross-process GPU memory access for the LL protocol.
        
        For multi-process (torchrun): uses cuMemCreate + cuMemMap +
        cuMemSetAccess (CUDA VMM) for kernel-level cross-process access.
        
        For single-process: uses raw pointer exchange with peer access.
        """
        if self.tp_size == 1:
            self._peer_buffers = [self._local_buffer.data_ptr()]
            return

        import ctypes
        import os

        # Detect multi-process TP: if ranks are in different PIDs, we need
        # VMM for cross-process kernel access. Check by gathering PIDs.
        local_pid = torch.tensor([os.getpid()], dtype=torch.int64, device="cuda")
        all_pids = [torch.zeros(1, dtype=torch.int64, device="cuda")
                    for _ in range(self.tp_size)]
        dist.all_gather(all_pids, local_pid, group=self.tp_group)
        pids = [t.item() for t in all_pids]
        is_multiprocess = len(set(pids)) > 1  # Different PIDs = different processes

        if not is_multiprocess:
            # Single-process multi-GPU: raw pointers with peer access
            cudart = ctypes.CDLL("libcudart.so")
            cudart.cudaDeviceEnablePeerAccess.restype = ctypes.c_int
            cudart.cudaDeviceEnablePeerAccess.argtypes = [ctypes.c_int, ctypes.c_uint]
            cudart.cudaGetLastError.restype = ctypes.c_int
            cudart.cudaGetLastError.argtypes = []
            local_device = torch.cuda.current_device()
            for peer in range(self.tp_size):
                if peer != local_device:
                    err = cudart.cudaDeviceEnablePeerAccess(peer, 0)
                    if err == 704:  # cudaErrorPeerAccessAlreadyEnabled
                        cudart.cudaGetLastError()  # Clear the error state

            local_ptr_tensor = torch.tensor(
                [self._local_buffer.data_ptr()], dtype=torch.int64, device="cuda"
            )
            all_ptrs = [
                torch.zeros(1, dtype=torch.int64, device="cuda")
                for _ in range(self.tp_size)
            ]
            dist.all_gather(all_ptrs, local_ptr_tensor, group=self.tp_group)
            self._peer_buffers = [t.item() for t in all_ptrs]
        else:
            # Multi-process: use CUDA VMM for kernel-level cross-process access
            self._setup_vmm()

    def _setup_vmm(self) -> None:
        """Setup cross-process access using CUDA VMM + Unix socket fd passing.

        Each rank:
        1. Allocates physical memory with cuMemCreate
        2. Exports a shareable POSIX fd with cuMemExportToShareableHandle
        3. Exchanges fds via Unix domain sockets (SCM_RIGHTS)
        4. Imports peer handles and maps them into local VA space
        5. Grants read/write access to all peer GPUs
        """
        import ctypes
        import struct
        import socket
        import array
        import os

        cuda = ctypes.CDLL("libcuda.so")
        cudart = ctypes.CDLL("libcudart.so")

        # Type aliases
        CUdeviceptr = ctypes.c_uint64
        CUmemGenericAllocationHandle = ctypes.c_uint64

        # Constants
        CU_MEM_ALLOCATION_TYPE_PINNED = 1
        CU_MEM_LOCATION_TYPE_DEVICE = 1
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
        CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3

        class CUmemAllocationProp(ctypes.Structure):
            _fields_ = [
                ("type", ctypes.c_int),
                ("requestedHandleTypes", ctypes.c_int),
                ("location_type", ctypes.c_int),
                ("location_id", ctypes.c_int),
                ("win32_security", ctypes.c_void_p),
                ("reserved", ctypes.c_uint64 * 4),
            ]

        class CUmemAccessDesc(ctypes.Structure):
            _fields_ = [
                ("location_type", ctypes.c_int),
                ("location_id", ctypes.c_int),
                ("flags", ctypes.c_int),
            ]

        # Setup function signatures
        cuda.cuMemGetAllocationGranularity.restype = ctypes.c_int
        cuda.cuMemGetAllocationGranularity.argtypes = [
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(CUmemAllocationProp), ctypes.c_int
        ]
        cuda.cuMemCreate.restype = ctypes.c_int
        cuda.cuMemCreate.argtypes = [
            ctypes.POINTER(CUmemGenericAllocationHandle), ctypes.c_size_t,
            ctypes.POINTER(CUmemAllocationProp), ctypes.c_uint64
        ]
        cuda.cuMemAddressReserve.restype = ctypes.c_int
        cuda.cuMemAddressReserve.argtypes = [
            ctypes.POINTER(CUdeviceptr), ctypes.c_size_t, ctypes.c_size_t,
            CUdeviceptr, ctypes.c_uint64
        ]
        cuda.cuMemMap.restype = ctypes.c_int
        cuda.cuMemMap.argtypes = [
            CUdeviceptr, ctypes.c_size_t, ctypes.c_size_t,
            CUmemGenericAllocationHandle, ctypes.c_uint64
        ]
        cuda.cuMemSetAccess.restype = ctypes.c_int
        cuda.cuMemSetAccess.argtypes = [
            CUdeviceptr, ctypes.c_size_t, ctypes.POINTER(CUmemAccessDesc), ctypes.c_size_t
        ]
        cuda.cuMemExportToShareableHandle.restype = ctypes.c_int
        cuda.cuMemExportToShareableHandle.argtypes = [
            ctypes.POINTER(ctypes.c_int), CUmemGenericAllocationHandle,
            ctypes.c_int, ctypes.c_uint64
        ]
        cuda.cuMemImportFromShareableHandle.restype = ctypes.c_int
        cuda.cuMemImportFromShareableHandle.argtypes = [
            ctypes.POINTER(CUmemGenericAllocationHandle), ctypes.c_int, ctypes.c_int
        ]

        local_device = torch.cuda.current_device()

        # Get granularity
        prop = CUmemAllocationProp()
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        prop.location_type = CU_MEM_LOCATION_TYPE_DEVICE
        prop.location_id = local_device

        granularity = ctypes.c_size_t()
        err = cuda.cuMemGetAllocationGranularity(ctypes.byref(granularity), ctypes.byref(prop), 0)
        if err != 0:
            raise RuntimeError(f"cuMemGetAllocationGranularity failed: {err}")

        # Round up buffer size
        alloc_size = ((self.buffer_size_bytes + granularity.value - 1)
                      // granularity.value * granularity.value)

        # Create physical memory
        local_handle = CUmemGenericAllocationHandle()
        err = cuda.cuMemCreate(ctypes.byref(local_handle), alloc_size, ctypes.byref(prop), 0)
        if err != 0:
            raise RuntimeError(f"cuMemCreate failed: {err}")

        # Reserve VA and map
        local_va = CUdeviceptr()
        err = cuda.cuMemAddressReserve(ctypes.byref(local_va), alloc_size, granularity.value, 0, 0)
        if err != 0:
            raise RuntimeError(f"cuMemAddressReserve failed: {err}")
        err = cuda.cuMemMap(local_va, alloc_size, 0, local_handle, 0)
        if err != 0:
            raise RuntimeError(f"cuMemMap failed: {err}")

        # Set access for all GPUs
        access_descs = (CUmemAccessDesc * self.tp_size)()
        for i in range(self.tp_size):
            access_descs[i].location_type = CU_MEM_LOCATION_TYPE_DEVICE
            access_descs[i].location_id = i
            access_descs[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        err = cuda.cuMemSetAccess(local_va, alloc_size, access_descs, self.tp_size)
        if err != 0:
            raise RuntimeError(f"cuMemSetAccess failed: {err}")

        # Export shareable fd
        local_fd = ctypes.c_int()
        err = cuda.cuMemExportToShareableHandle(
            ctypes.byref(local_fd), local_handle,
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0
        )
        if err != 0:
            raise RuntimeError(f"cuMemExportToShareableHandle failed: {err}")

        # Zero the buffer
        cudart.cudaMemset.restype = ctypes.c_int
        cudart.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        cudart.cudaMemset(ctypes.c_void_p(local_va.value), 0, alloc_size)
        torch.cuda.synchronize()

        # ── Exchange fds via Unix domain sockets (SCM_RIGHTS) ──
        # For each pair (i, j) where i < j, rank i is server, rank j is client.
        # Each pair exchanges fds bidirectionally.
        peer_fds = {}  # peer_rank -> received fd

        for peer in range(self.tp_size):
            if peer == self.tp_rank:
                continue
            sock_path = f"/dev/shm/moe_ll_sock_{min(self.tp_rank, peer)}_{max(self.tp_rank, peer)}"

            if self.tp_rank < peer:
                # Server
                if os.path.exists(sock_path):
                    os.remove(sock_path)
                server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                server.bind(sock_path)
                server.listen(1)
                dist.barrier(group=self.tp_group)
                conn, _ = server.accept()
                # Send our fd
                fds = array.array("i", [local_fd.value])
                conn.sendmsg(
                    [struct.pack("q", alloc_size)],
                    [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds)]
                )
                # Receive peer's fd
                msg, ancdata, _, _ = conn.recvmsg(8, socket.CMSG_SPACE(4))
                for cmsg_level, cmsg_type, cmsg_data in ancdata:
                    if cmsg_type == socket.SCM_RIGHTS:
                        peer_fds[peer] = array.array("i", cmsg_data)[0]
                conn.close()
                server.close()
                os.remove(sock_path)
            else:
                # Client
                dist.barrier(group=self.tp_group)
                client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client.connect(sock_path)
                # Receive peer's fd
                msg, ancdata, _, _ = client.recvmsg(8, socket.CMSG_SPACE(4))
                for cmsg_level, cmsg_type, cmsg_data in ancdata:
                    if cmsg_type == socket.SCM_RIGHTS:
                        peer_fds[peer] = array.array("i", cmsg_data)[0]
                # Send our fd
                fds = array.array("i", [local_fd.value])
                client.sendmsg(
                    [struct.pack("q", alloc_size)],
                    [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds)]
                )
                client.close()

        dist.barrier(group=self.tp_group)

        # ── Import peer handles and map into local VA ──
        self._peer_buffers = []
        self._vmm_handles = [local_handle]
        self._vmm_vas = [local_va.value]

        for i in range(self.tp_size):
            if i == self.tp_rank:
                self._peer_buffers.append(local_va.value)
            else:
                peer_handle = CUmemGenericAllocationHandle()
                err = cuda.cuMemImportFromShareableHandle(
                    ctypes.byref(peer_handle), peer_fds[i],
                    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
                )
                if err != 0:
                    raise RuntimeError(f"cuMemImportFromShareableHandle for rank {i} failed: {err}")

                peer_va = CUdeviceptr()
                err = cuda.cuMemAddressReserve(
                    ctypes.byref(peer_va), alloc_size, granularity.value, 0, 0
                )
                if err != 0:
                    raise RuntimeError(f"cuMemAddressReserve for peer {i} failed: {err}")

                err = cuda.cuMemMap(peer_va, alloc_size, 0, peer_handle, 0)
                if err != 0:
                    raise RuntimeError(f"cuMemMap for peer {i} failed: {err}")

                err = cuda.cuMemSetAccess(peer_va, alloc_size, access_descs, self.tp_size)
                if err != 0:
                    raise RuntimeError(f"cuMemSetAccess for peer {i} failed: {err}")

                self._peer_buffers.append(peer_va.value)
                self._vmm_handles.append(peer_handle)
                self._vmm_vas.append(peer_va.value)

        # Override local buffer pointer to use VMM-allocated memory
        self._vmm_alloc_size = alloc_size

        dist.barrier(group=self.tp_group)

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
