# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Stage-A standalone proof: CUDA-graph-safe one-sided EP signaling.

Runs 2 ranks (1 GPU each) that exchange a staged row over NVLink using the
one-sided handshake in ``csrc/moe/moe_monokernel/src/moe_comm_ll.h``
(``ep_set_ready`` / ``ep_wait_ready``) with the flag read from a device-memory
epoch counter. The producer writes a payload that DEPENDS on the current epoch
(``epoch*10 + rank``); the consumer waits on the peer's flag and reads the
peer's row. The whole [advance_epoch -> write -> read] sequence is captured
into a CUDA graph and replayed many times.

What it proves
--------------
Every replay must observe the peer's CURRENT-epoch payload
(``epoch*10 + peer``). A design that baked the flag as a capture-time constant
(the reference's abandoned handshake) would either deadlock or read a stale
payload here — this test would then fail. Passing demonstrates the device-
epoch flag is graph-safe, which is the whole premise of Stage A.

No vLLM rebuild required: the probe kernel is JIT-compiled via load_inline and
includes the header directly.

Run on the 2-GPU box (e.g. p5en):

    torchrun --nproc_per_node=2 tests/moe/test_ep_graphsafe_signal.py

Optional: --steps N (default 200), --hidden H (default 4096, V4-Flash K).
"""

import argparse
import os

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline

_HERE = os.path.dirname(os.path.abspath(__file__))
_MONO_SRC = os.path.abspath(
    os.path.join(_HERE, "..", "..", "csrc", "moe", "moe_monokernel", "src")
)

# Probe kernels: include the real signaling header so we test the shipped code.
_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <stdint.h>
#include "moe_comm_ll.h"

using moe_monokernel::ep_set_ready;
using moe_monokernel::ep_wait_ready;
using moe_monokernel::load_epoch;

// The staging region is raw bytes; we reinterpret its first word as uint32 so
// the payload is delivered EXACTLY (a bf16 buffer cannot represent odd/large
// integers, which would make the proof's equality check spuriously fail on
// rounding rather than on any signaling error).
//
// Producer: write (epoch*1000 + rank) into this rank's staging word, then
// publish readiness into this rank's own flag slot.
__global__ void probe_write_kernel(uint32_t* stage, uint32_t* my_flag,
                                    const uint32_t* epoch_ptr, int rank) {
  if (threadIdx.x == 0) {
    uint32_t epoch = load_epoch(epoch_ptr);
    stage[0] = epoch * 1000u + (uint32_t)rank;
    // ep_set_ready issues __threadfence_system() before the flag store, so the
    // stage[0] write above is guaranteed visible to the peer once the flag is.
    ep_set_ready(my_flag, epoch_ptr);
  }
}

// Consumer: wait on the peer's flag for this epoch, then read the peer's word.
__global__ void probe_read_kernel(const uint32_t* peer_stage,
                                   const uint32_t* peer_flag,
                                   const uint32_t* epoch_ptr, uint32_t* out) {
  if (threadIdx.x == 0) {
    ep_wait_ready(peer_flag, epoch_ptr);
    *out = peer_stage[0];
  }
}

void probe_write(int64_t stage_ptr, int64_t my_flag_ptr, int64_t epoch_ptr,
                 int64_t rank) {
  // Launch on the CURRENT stream so that under CUDA-graph capture the kernel
  // is (a) captured and (b) ordered AFTER the epoch increment on the same
  // stream. A bare <<<...>>> would use the default stream, which is not the
  // capture stream -> the kernel would read a stale (never-advancing) epoch.
  cudaStream_t s = c10::cuda::getCurrentCUDAStream();
  probe_write_kernel<<<1, 32, 0, s>>>(
      reinterpret_cast<uint32_t*>(stage_ptr),
      reinterpret_cast<uint32_t*>(my_flag_ptr),
      reinterpret_cast<const uint32_t*>(epoch_ptr), (int)rank);
}

void probe_read(int64_t peer_stage_ptr, int64_t peer_flag_ptr,
                int64_t epoch_ptr, int64_t out_ptr) {
  cudaStream_t s = c10::cuda::getCurrentCUDAStream();
  probe_read_kernel<<<1, 32, 0, s>>>(
      reinterpret_cast<const uint32_t*>(peer_stage_ptr),
      reinterpret_cast<const uint32_t*>(peer_flag_ptr),
      reinterpret_cast<const uint32_t*>(epoch_ptr),
      reinterpret_cast<uint32_t*>(out_ptr));
}
"""

_CPP_DECL = (
    "void probe_write(int64_t, int64_t, int64_t, int64_t);\n"
    "void probe_read(int64_t, int64_t, int64_t, int64_t);\n"
)


def _build():
    return load_inline(
        name="ep_graphsafe_probe",
        cpp_sources=_CPP_DECL,
        cuda_sources=_CUDA_SRC,
        functions=["probe_write", "probe_read"],
        extra_include_paths=[_MONO_SRC],
        extra_cuda_cflags=["-O2"],
        verbose=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=4096)  # V4-Flash K
    args = ap.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, "This proof targets EP=2 (2 ranks)."
    torch.cuda.set_device(rank)
    peer = rank ^ 1

    from vllm.distributed.moe_ll_workspace import MoELLWorkspace

    ws = MoELLWorkspace(max_num_tokens=1, hidden_dim=args.hidden,
                        tp_group=dist.group.WORLD)
    probe = _build()

    # Views: this rank's dispatch staging row + the peer's (peer-mapped) row.
    local_view, peer_views = ws.ep_activation_views(num_tokens=1)
    peer_view = peer_views[peer]
    flags = ws.ep_flag_buffers()          # int64[2] peer-mapped flag slots
    my_flag_ptr = int(flags[rank].item())
    peer_flag_ptr = int(flags[peer].item())
    epoch_ptr = ws.epoch_ptr()
    out = torch.zeros(1, dtype=torch.int32, device="cuda")

    stage_ptr = int(local_view.data_ptr())
    peer_stage_ptr = int(peer_view.data_ptr())

    # Make sure both ranks have their IPC mappings ready before capture.
    dist.barrier()

    # ── Capture: advance_epoch -> write my row -> read peer row ──────────
    # advance_epoch is an in-place device add (captured), so every replay
    # bumps the epoch and the payload changes with it.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # warmup (also lets JIT/autotune settle) — not captured
        ws.advance_epoch()
        probe.probe_write(stage_ptr, my_flag_ptr, epoch_ptr, rank)
        probe.probe_read(peer_stage_ptr, peer_flag_ptr, epoch_ptr,
                         int(out.data_ptr()))
    torch.cuda.current_stream().wait_stream(stream)
    dist.barrier()
    # Reset epoch to 0 and clear the buffer so warmup's stale flags cannot be
    # matched: the first replay must genuinely wait on the producer's write.
    ws.reset_epoch()
    ws.local_buffer.zero_()
    dist.barrier()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ws.advance_epoch()
        probe.probe_write(stage_ptr, my_flag_ptr, epoch_ptr, rank)
        probe.probe_read(peer_stage_ptr, peer_flag_ptr, epoch_ptr,
                         int(out.data_ptr()))

    # ── Replay: each step epoch increments; expect peer's CURRENT payload ──
    failures = 0
    for step in range(1, args.steps + 1):
        g.replay()
        torch.cuda.synchronize()
        expected = step * 1000 + peer  # peer wrote epoch*1000 + peer (exact)
        got = int(out.item())
        if got != expected:
            failures += 1
            if failures <= 5:
                print(f"[rank {rank}] step {step}: got {got}, "
                      f"expected {expected}", flush=True)
        dist.barrier()  # keep ranks in lock-step (epochs aligned)

    result = torch.tensor([failures], device="cuda")
    dist.all_reduce(result)
    if rank == 0:
        total = int(result.item())
        if total == 0:
            print(f"PASS: {args.steps} graph replays x2 ranks, all reads saw "
                  f"the peer's current-epoch payload (graph-safe).", flush=True)
        else:
            print(f"FAIL: {total} mismatches across replays "
                  f"(stale/baked-flag behavior).", flush=True)

    ws.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
