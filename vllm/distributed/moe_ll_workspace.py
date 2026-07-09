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

import ctypes
import torch
import torch.distributed as dist
from typing import Optional


class _cudaIpcMemHandle_t(ctypes.Structure):
    # CUDA IPC handle is an opaque 64-byte blob; the struct is declared as
    # CUDA_IPC_HANDLE_SIZE (64) bytes. We use 128 to be safe against ABI drift;
    # only the first CUDA_IPC_HANDLE_SIZE bytes are meaningful and the same size
    # is used symmetrically on every rank, so the extra padding is harmless.
    _fields_ = [("internal", ctypes.c_byte * 128)]


def _cudart():
    return ctypes.CDLL("libcudart.so")


# ── DLPack plumbing to alias a raw CUDA device pointer as a zero-copy torch
# tensor. torch.as_tensor does NOT consume __cuda_array_interface__, but
# torch.utils.dlpack.from_dlpack reliably imports a DLManagedTensor capsule. ──
class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8),
                ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DL_DELETER = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DL_DELETER),
]

_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

# Keep the ctypes managed-tensor / shape objects alive until torch calls the
# deleter (i.e. until the aliasing tensor is freed).
_DLPACK_KEEPALIVE: dict = {}


def _dl_deleter(mt_ptr):
    key = ctypes.cast(mt_ptr, ctypes.c_void_p).value
    _DLPACK_KEEPALIVE.pop(key, None)


_DL_DELETER_CB = _DL_DELETER(_dl_deleter)

# DLPack device_type kDLCUDA = 2; dtype code kDLUInt = 1.
_DLCUDA = 2
_DLUINT = 1


def _wrap_cuda_ptr(ptr: int, nbytes: int, device_index: int) -> torch.Tensor:
    """Alias a raw CUDA device pointer as a zero-copy uint8 torch tensor.

    The returned tensor does NOT own the memory; the caller must keep the
    allocation alive and free it (cudaFree) separately.
    """
    import torch.utils.dlpack as _dlpack

    shape = (ctypes.c_int64 * 1)(nbytes)
    mt = _DLManagedTensor()
    mt.dl_tensor.data = ptr
    mt.dl_tensor.device = _DLDevice(_DLCUDA, device_index)
    mt.dl_tensor.ndim = 1
    mt.dl_tensor.dtype = _DLDataType(_DLUINT, 8, 1)
    mt.dl_tensor.shape = shape
    mt.dl_tensor.strides = ctypes.cast(None, ctypes.POINTER(ctypes.c_int64))
    mt.dl_tensor.byte_offset = 0
    mt.manager_ctx = None
    mt.deleter = _DL_DELETER_CB
    _DLPACK_KEEPALIVE[ctypes.addressof(mt)] = (mt, shape)
    capsule = _PyCapsule_New(ctypes.addressof(mt), b"dltensor", None)
    return _dlpack.from_dlpack(capsule)


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
        # Multi-process CUDA-IPC bookkeeping (set by _setup_ipc_multiprocess).
        self._ipc_multiprocess: bool = False
        self._ipc_local_ptr: Optional[int] = None  # our cudaMalloc'd base
        self._ipc_opened_ptrs: list[int] = []       # peer ptrs we opened
        self._setup_ipc()

        # Build the peer_ll_buffers tensor that gets passed to the kernel
        self.peer_ll_buffers = torch.tensor(
            self._peer_buffers, dtype=torch.int64, device="cuda"
        )

    def _setup_ipc(self) -> None:
        """Setup cross-process GPU memory access for the LL protocol.

        Single-process multi-GPU (test harness): raw pointer exchange with
        cudaDeviceEnablePeerAccess.

        Multi-process (real DP / torchrun): CUDA IPC mem handles
        (cudaMalloc + cudaIpcGetMemHandle + cudaIpcOpenMemHandle), exchanged
        as plain bytes over the process group. This is backend-agnostic, needs
        no shared filesystem path, and works for any group size and any number
        of concurrent EP groups.
        """
        if self.tp_size == 1:
            self._peer_buffers = [self._local_buffer.data_ptr()]
            return

        import os

        # Detect multi-process TP by gathering PIDs. Use all_gather_object so
        # this works on any collective backend (gloo/nccl) without a CUDA
        # collective.
        pids = [None] * self.tp_size
        dist.all_gather_object(pids, os.getpid(), group=self.tp_group)
        is_multiprocess = len(set(pids)) > 1

        if is_multiprocess:
            self._setup_ipc_multiprocess()
        else:
            self._setup_ipc_singleprocess()

    def _setup_ipc_singleprocess(self) -> None:
        """Single-process multi-GPU: raw pointers with peer access enabled."""
        cudart = _cudart()
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

    def _setup_ipc_multiprocess(self) -> None:
        """Cross-process GPU sharing via classic CUDA IPC memory handles.

        We deliberately DO NOT reuse torch's caching-allocator buffer here: when
        torch is configured with expandable_segments (cuMem/VMM), its storage
        IPC export (_share_cuda_) returns a torch-wire-format handle (~66 bytes)
        that a raw cudaIpcOpenMemHandle rejects (cudaErrorInvalidValue). Instead
        we make our own classic cudaMalloc allocation (immune to torch's
        allocator config), which yields a clean 64-byte cudaIpcMemHandle, then
        alias that pointer back into a torch tensor via DLPack so local writes /
        zeroing are unchanged.

        Handles are exchanged as plain bytes via all_gather_object: backend-
        agnostic, no shared filesystem path, correct for any group size and any
        number of concurrent EP groups. Replaces the earlier cuMem VMM +
        Unix-socket fd-exchange path (deadlocked for group size > 2; collided
        across EP groups on a shared /dev/shm socket path).
        """
        cudart = _cudart()
        cudart.cudaMalloc.restype = ctypes.c_int
        cudart.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        cudart.cudaMemset.restype = ctypes.c_int
        cudart.cudaMemset.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        cudart.cudaIpcGetMemHandle.restype = ctypes.c_int
        cudart.cudaIpcGetMemHandle.argtypes = [
            ctypes.POINTER(_cudaIpcMemHandle_t), ctypes.c_void_p]
        cudart.cudaIpcOpenMemHandle.restype = ctypes.c_int
        cudart.cudaIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), _cudaIpcMemHandle_t, ctypes.c_uint]

        size = self.buffer_size_bytes
        local_device = torch.cuda.current_device()
        hsize = ctypes.sizeof(_cudaIpcMemHandle_t)

        # 1. Our own classic allocation (offset 0, clean IPC handle).
        local_ptr = ctypes.c_void_p()
        err = cudart.cudaMalloc(ctypes.byref(local_ptr), size)
        if err != 0:
            raise RuntimeError(f"cudaMalloc({size}) failed: {err}")
        err = cudart.cudaMemset(local_ptr, 0, size)
        if err != 0:
            raise RuntimeError(f"cudaMemset failed: {err}")

        # 2. Adopt as the local buffer (zero-copy DLPack alias of the IPC mem).
        self._ipc_multiprocess = True
        self._ipc_local_ptr = local_ptr.value
        self._local_buffer = _wrap_cuda_ptr(local_ptr.value, size, local_device)

        # 3. Export our clean handle + all-gather everyone's (raw bytes).
        handle = _cudaIpcMemHandle_t()
        err = cudart.cudaIpcGetMemHandle(ctypes.byref(handle), local_ptr)
        if err != 0:
            raise RuntimeError(f"cudaIpcGetMemHandle failed: {err}")
        my_handle_bytes = ctypes.string_at(ctypes.byref(handle), hsize)

        all_handles: list = [None] * self.tp_size
        dist.all_gather_object(all_handles, my_handle_bytes, group=self.tp_group)

        import os as _os
        if _os.environ.get("VLLM_MOE_LL_DEBUG") == "1":
            print(f"[MoELLWorkspace rank {self.tp_rank}] "
                  f"local_ptr={local_ptr.value:#x} size={size} "
                  f"handle_len={len(my_handle_bytes)}", flush=True)

        # 4. Open peer handles -> peer device pointers.
        CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1
        self._peer_buffers = []
        self._ipc_opened_ptrs = []
        for i in range(self.tp_size):
            if i == self.tp_rank:
                self._peer_buffers.append(self._ipc_local_ptr)
                continue
            h = _cudaIpcMemHandle_t()
            ctypes.memmove(ctypes.byref(h), all_handles[i],
                           min(len(all_handles[i]), hsize))
            peer_ptr = ctypes.c_void_p()
            err = cudart.cudaIpcOpenMemHandle(
                ctypes.byref(peer_ptr), h, CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS)
            if err != 0:
                raise RuntimeError(
                    f"cudaIpcOpenMemHandle for rank {i} failed: {err}")
            self._peer_buffers.append(peer_ptr.value)
            self._ipc_opened_ptrs.append(peer_ptr.value)

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
        if self._ipc_multiprocess:
            cudart = _cudart()
            cudart.cudaIpcCloseMemHandle.restype = ctypes.c_int
            cudart.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
            cudart.cudaFree.restype = ctypes.c_int
            cudart.cudaFree.argtypes = [ctypes.c_void_p]
            for peer_ptr in self._ipc_opened_ptrs:
                cudart.cudaIpcCloseMemHandle(ctypes.c_void_p(peer_ptr))
            self._ipc_opened_ptrs = []
            # Drop the DLPack alias before freeing the underlying allocation.
            self._local_buffer = None
            if self._ipc_local_ptr is not None:
                cudart.cudaFree(ctypes.c_void_p(self._ipc_local_ptr))
                self._ipc_local_ptr = None
        else:
            del self._local_buffer
        del self.peer_ll_buffers
        self._peer_buffers = []

    @property
    def local_buffer(self) -> torch.Tensor:
        """The raw local LL buffer (for debugging/testing)."""
        return self._local_buffer

    # ──────────────────────────────────────────────────────────────────────
    # EP in-kernel dispatch views.
    #
    # For the EP dispatch peer-read (Stage 2), the monokernel needs, per rank:
    #   - a PLAIN bf16 activation staging region it writes its owned tokens
    #     into (indexed by GLOBAL token id), and
    #   - the PEER's staging region, peer-mapped, to read the remote tokens;
    #   - a tiny per-rank readiness-FLAG slot (the storeLL/readLL handshake),
    #     peer-mapped, so a rank does not read before the peer has written.
    #
    # We carve the symmetric IPC buffer as:
    #   [ bf16 activation region: max_num_tokens * hidden_dim * 2 bytes ]
    #   [ ... unused ... ]
    #   [ flag region: last _EP_FLAG_REGION_BYTES bytes ]
    # The buffer is symmetric across ranks, so a peer's region is at the same
    # offset in the peer pointer. Only valid on a workspace created for this
    # purpose (its LL-all-reduce role must not be used simultaneously).
    # ──────────────────────────────────────────────────────────────────────
    _EP_FLAG_REGION_BYTES = 256   # >> tp_size * 16; kept off the activations
    _EP_FLAG_STRIDE = 16          # one LLPacket-sized slot per rank

    def _ep_flag_base_offset(self) -> int:
        return self.buffer_size_bytes - self._EP_FLAG_REGION_BYTES

    def ep_activation_views(self, num_tokens: int):
        """Return (local_view, peer_views) as bf16 [num_tokens, hidden_dim].

        local_view aliases THIS rank's staging region (write owned rows here);
        peer_views[i] aliases rank i's staging region (peer-mapped) for the
        kernel's `peer_activations`. peer_views[tp_rank] is local_view.
        """
        K = self.hidden_dim
        nbytes = num_tokens * K * 2  # bf16
        assert nbytes <= self._ep_flag_base_offset(), (
            f"activation region {nbytes} B overruns flag region at "
            f"{self._ep_flag_base_offset()} B (raise max_num_tokens/buffer)"
        )
        dev_index = torch.cuda.current_device()
        local_view = (
            self._local_buffer[:nbytes].view(torch.bfloat16).view(num_tokens, K)
        )
        peer_views = []
        for i in range(self.tp_size):
            if i == self.tp_rank:
                peer_views.append(local_view)
            else:
                u8 = _wrap_cuda_ptr(self._peer_buffers[i], nbytes, dev_index)
                peer_views.append(u8.view(torch.bfloat16).view(num_tokens, K))
        return local_view, peer_views

    def ep_combine_views(self, num_tokens: int):
        """Return (local_view, peer_views) for the in-kernel EP COMBINE.

        Symmetric to ep_activation_views but at a DISTINCT offset (the second
        half of the buffer), so the combine can stage this rank's full per-rank
        partial [num_tokens, hidden_dim] without clobbering the dispatch
        activation region (which a peer may still be reading). Each rank writes
        its partial into local_view; after a cross-rank ordering barrier, rank r
        peer-reads peer_views[p][own_rows] and sums to reduce-scatter WITHOUT a
        NCCL collective moving the activation data.
        """
        K = self.hidden_dim
        nbytes = num_tokens * K * 2  # bf16
        off = self.buffer_size_bytes // 2
        assert num_tokens * K * 2 <= off, (
            f"combine region base {off} overlaps dispatch region "
            f"({num_tokens * K * 2} B)"
        )
        assert off + nbytes <= self._ep_flag_base_offset(), (
            f"combine region {nbytes} B at {off} overruns flag region at "
            f"{self._ep_flag_base_offset()} B (raise max_num_tokens/buffer)"
        )
        dev_index = torch.cuda.current_device()
        local_view = (
            self._local_buffer[off:off + nbytes]
            .view(torch.bfloat16).view(num_tokens, K)
        )
        peer_views = []
        for i in range(self.tp_size):
            if i == self.tp_rank:
                peer_views.append(local_view)
            else:
                u8 = _wrap_cuda_ptr(self._peer_buffers[i] + off, nbytes, dev_index)
                peer_views.append(u8.view(torch.bfloat16).view(num_tokens, K))
        return local_view, peer_views

    def ep_flag_buffers(self) -> torch.Tensor:
        """int64 [tp_size]: peer-mapped pointer to each rank's readiness-flag
        slot (same layout the M2c handshake expects for `peer_ll_buffers`)."""
        off = self._ep_flag_base_offset()
        return torch.tensor(
            [p + off for p in self._peer_buffers],
            dtype=torch.int64, device="cuda",
        )

    def ep_zero_flags(self) -> None:
        """Zero this rank's readiness-flag slot (call once per forward, before
        the first layer, so stale flags from the previous forward do not match)."""
        off = self._ep_flag_base_offset()
        self._local_buffer[off:off + self._EP_FLAG_REGION_BYTES].zero_()
        self._flag_counter = 0
