# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MoE Monokernel EP symmetric-memory workspace (single-node, NVLink).

Allocates symmetric CUDA-IPC buffers so each rank in an EP group can read
another rank's staging regions directly over NVLink from inside the
monokernel (in-kernel dispatch peer-read + one-sided combine peer-reduce).

This is the Stage-A foundation for the EP all-to-all *hiding* work on
DeepSeek-V4-Flash (EP=2, single peer). It is intentionally scoped to the
EP use-case (dispatch/combine staging + a small readiness-flag region) and
adds a **CUDA-graph-safe** device-memory epoch counter that the in-kernel
one-sided signaling reads at runtime.

Why the device-memory epoch matters
------------------------------------
The earlier fused-AR handshake advanced a *host* flag counter and passed the
resulting integer to the kernel each layer. Under CUDA-graph capture (decode)
that integer is baked into the captured graph, so every replay re-uses the
same constant and a reader cannot distinguish this step's write from the
previous step's occupant of the same slot -> stale-read hazard. That is why
the reference fell back to a collective barrier (router all-gather) for
ordering. Here the epoch lives in *device* memory and is advanced by a graph-
captured op, so each replay observes a fresh, monotonically-increasing value.
The reader's expected flag is therefore unique-per-step and a stale value is
always strictly smaller -> no false match, no reset needed.

The signaling substrate is deliberately kept behind a narrow interface
(see ``vllm/model_executor/layers/fused_moe/ep_signal.py``) so an NVSHMEM
backend can replace this NVLink CUDA-IPC path later if multi-node/scale-out
becomes a target.
"""

import ctypes
from typing import Optional

import torch
import torch.distributed as dist


class _cudaIpcMemHandle_t(ctypes.Structure):
    # CUDA IPC handle is an opaque 64-byte blob (CUDA_IPC_HANDLE_SIZE = 64).
    # 128 bytes of storage is used to be safe against ABI drift; only the
    # first CUDA_IPC_HANDLE_SIZE bytes are meaningful and the same size is
    # used symmetrically on every rank, so the extra padding is harmless.
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
    Symmetric CUDA-IPC buffer workspace for the monokernel's EP all-to-all.

    Each rank allocates one local buffer and exchanges CUDA-IPC handles with
    the other rank(s) in its EP group, so every rank can read another rank's
    staging regions directly over NVLink. The buffer is carved into three
    regions (dispatch staging, combine staging, readiness flags); see the
    ``ep_*`` view helpers below.

    ``peer_ll_buffers`` is an int64 [ep_size] tensor where entry[i] is the
    device pointer to rank i's buffer, as seen from THIS rank.

    Scope: EP dispatch/combine hiding on a single node (NVLink). For EP=2 the
    peer of rank r is ``r ^ 1``. The layout generalizes to ep_size > 2, but
    the current monokernel EP path targets the single-peer (EP=2) case.
    """

    # Each LL packet carries 4 bf16 values = 8 bytes of payload.
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
            max_num_tokens: Maximum gathered token count the monokernel will
                process in one EP step (BS per chunk, or the whole gathered
                tile if chunked).
            hidden_dim: Hidden dimension (K) of the model.
            tp_group: The EP process group. If None, uses the default group.
        """
        if tp_group is None:
            tp_group = dist.group.WORLD

        self.tp_size = dist.get_world_size(group=tp_group)
        self.tp_rank = dist.get_rank(group=tp_group)
        self.tp_group = tp_group
        self.max_num_tokens = max_num_tokens
        self.hidden_dim = hidden_dim

        assert hidden_dim % self.LL_ELEMS_PER_PACKET == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by "
            f"LL_ELEMS_PER_PACKET ({self.LL_ELEMS_PER_PACKET})"
        )
        self.packets_per_row = hidden_dim // self.LL_ELEMS_PER_PACKET

        # Buffer must hold: dispatch staging (first half) + combine staging
        # (second half) + a small readiness-flag region at the tail. Size the
        # halves off the max activation footprint (bf16 [max_num_tokens, K]).
        act_bytes = max_num_tokens * hidden_dim * 2  # bf16
        half = max(act_bytes, self.tp_size * max_num_tokens
                   * self.packets_per_row * self.LL_PACKET_BYTES)
        self.buffer_size_bytes = 2 * half + self._EP_FLAG_REGION_BYTES

        # Allocate the local buffer (see _setup_ipc for the multi-process path,
        # which replaces this with a classic cudaMalloc allocation).
        self._local_buffer = torch.zeros(
            self.buffer_size_bytes, dtype=torch.uint8, device="cuda"
        )

        # ── CUDA-graph-safe epoch counter (Stage A) ──────────────────────
        # Device-resident, advanced by a graph-captured op (advance_epoch)
        # once per forward step. The in-kernel one-sided signaling reads
        # *epoch_ptr at runtime as its flag base, so each graph replay sees a
        # fresh monotonically-increasing value (no baked constant, no reset).
        self._d_epoch = torch.zeros(1, dtype=torch.int32, device="cuda")

        # Legacy host flag counter (eager path / back-compat only; NOT graph
        # safe — see module docstring).
        self._flag_counter: int = 0

        self._peer_buffers: list[int] = []  # device pointers as ints
        self._ipc_multiprocess: bool = False
        self._ipc_local_ptr: Optional[int] = None
        self._ipc_opened_ptrs: list[int] = []
        self._setup_ipc()

        self.peer_ll_buffers = torch.tensor(
            self._peer_buffers, dtype=torch.int64, device="cuda"
        )

    # ──────────────────────────────────────────────────────────────────────
    # CUDA-graph-safe epoch API (Stage A)
    # ──────────────────────────────────────────────────────────────────────
    def epoch_ptr(self) -> int:
        """Device pointer to the int32 epoch counter.

        Pass this to the kernel; the in-kernel signaling reads ``*epoch_ptr``
        as the flag base for THIS step. Value is identical across ranks
        because all ranks advance in lock-step (EP requires it).
        """
        return self._d_epoch.data_ptr()

    def advance_epoch(self) -> None:
        """Advance the device epoch by one.

        MUST be issued once per forward step INSIDE the captured region (or as
        a captured graph node) so every replay bumps it. Implemented as an
        in-place device add, which is a single capturable kernel. Do NOT read
        the value back to the host on the capture path (that would break
        capture); it is only consumed on-device by the signaling primitives.
        """
        self._d_epoch.add_(1)

    def reset_epoch(self) -> None:
        """Zero the epoch (eager teardown / test setup only; not on the
        capture path)."""
        self._d_epoch.zero_()

    def _setup_ipc(self) -> None:
        """Set up cross-process GPU memory access.

        Single-process multi-GPU (test harness): raw pointer exchange with
        cudaDeviceEnablePeerAccess.

        Multi-process (real DP/EP, torchrun): CUDA-IPC mem handles
        (cudaMalloc + cudaIpcGetMemHandle + cudaIpcOpenMemHandle) exchanged as
        plain bytes over the process group — backend-agnostic, no shared
        filesystem path, correct for any group size and concurrent EP groups.
        """
        if self.tp_size == 1:
            self._peer_buffers = [self._local_buffer.data_ptr()]
            return

        import os

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
        """Cross-process GPU sharing via classic CUDA-IPC memory handles.

        We deliberately DO NOT reuse torch's caching-allocator buffer: with
        expandable_segments (cuMem/VMM) torch's storage IPC export returns a
        torch-wire-format handle that raw cudaIpcOpenMemHandle rejects. Instead
        we make our own classic cudaMalloc allocation (immune to torch's
        allocator config), which yields a clean 64-byte cudaIpcMemHandle, then
        alias that pointer back into a torch tensor via DLPack so local writes/
        zeroing are unchanged. Handles are exchanged as plain bytes via
        all_gather_object (backend-agnostic).
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

    # ──────────────────────────────────────────────────────────────────────
    # Legacy host flag counter (eager path only — NOT CUDA-graph safe).
    # Prefer the device-memory epoch API above on the capture path.
    # ──────────────────────────────────────────────────────────────────────
    def next_flag(self) -> int:
        self._flag_counter += 1
        return self._flag_counter

    def reset_flags(self) -> None:
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
        """The raw local buffer (for debugging/testing)."""
        return self._local_buffer

    # ──────────────────────────────────────────────────────────────────────
    # EP staging views (dispatch = first half, combine = second half,
    # readiness flags = tail). The buffer is symmetric across ranks, so a
    # peer's region is at the same offset in the peer pointer.
    # ──────────────────────────────────────────────────────────────────────
    _EP_FLAG_REGION_BYTES = 256   # >> ep_size * LL_PACKET_BYTES
    _EP_FLAG_STRIDE = 16          # one LLPacket-sized slot per rank

    def _ep_flag_base_offset(self) -> int:
        return self.buffer_size_bytes - self._EP_FLAG_REGION_BYTES

    def _combine_base_offset(self) -> int:
        # Second half of the (buffer minus flag region).
        return (self.buffer_size_bytes - self._EP_FLAG_REGION_BYTES) // 2

    def ep_activation_views(self, num_tokens: int):
        """(local_view, peer_views) as bf16 [num_tokens, hidden_dim].

        Write this rank's owned rows into local_view; peer_views[i] aliases
        rank i's dispatch staging region (peer-mapped) for the kernel's
        ``peer_activations``. peer_views[tp_rank] is local_view.
        """
        K = self.hidden_dim
        nbytes = num_tokens * K * 2  # bf16
        assert nbytes <= self._combine_base_offset(), (
            f"dispatch region {nbytes} B overruns combine region at "
            f"{self._combine_base_offset()} B (raise max_num_tokens)"
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
        """(local_view, peer_views) for the in-kernel EP combine.

        At a DISTINCT offset from the dispatch region so the combine can stage
        this rank's full per-rank partial [num_tokens, hidden_dim] without
        clobbering the dispatch activations a peer may still be reading.
        """
        K = self.hidden_dim
        nbytes = num_tokens * K * 2  # bf16
        off = self._combine_base_offset()
        assert off + nbytes <= self._ep_flag_base_offset(), (
            f"combine region {nbytes} B at {off} overruns flag region at "
            f"{self._ep_flag_base_offset()} B (raise max_num_tokens)"
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
        """int64 [ep_size]: peer-mapped pointer to each rank's readiness-flag
        slot. Used by the in-kernel one-sided store/poll handshake."""
        off = self._ep_flag_base_offset()
        return torch.tensor(
            [p + off for p in self._peer_buffers],
            dtype=torch.int64, device="cuda",
        )
