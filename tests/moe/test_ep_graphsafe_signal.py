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
#include <cuda_bf16.h>
#include <torch/extension.h>
#include "moe_comm_ll.h"

using moe_monokernel::ep_set_ready;
using moe_monokernel::ep_wait_ready;
using moe_monokernel::load_epoch;

// Producer: fill this rank's staging row with (epoch*10 + rank), then publish
// readiness into this rank's own flag slot.
__global__ void probe_write_kernel(__nv_bfloat16* stage, uint32_t* my_flag,
                                    const uint32_t* epoch_ptr, int rank,
                                    int hidden) {
  uint32_t epoch = load_epoch(epoch_ptr);
  float payload = (float)(epoch * 10u + (uint32_t)rank);
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    stage[i] = (__nv_bfloat16)payload;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    ep_set_ready(my_flag, epoch_ptr);
  }
}

// Consumer: wait on the peer's flag for this epoch, then read the peer's row.
__global__ void probe_read_kernel(const __nv_bfloat16* peer_stage,
                                   const uint32_t* peer_flag,
                                   const uint32_t* epoch_ptr, float* out,
                                   int hidden) {
  if (threadIdx.x == 0) {
    ep_wait_ready(peer_flag, epoch_ptr);
  }
  __syncthreads();
  // Read element 0 (all elements are identical) into out.
  if (threadIdx.x == 0) {
    *out = (float)peer_stage[0];
  }
}

void probe_write(int64_t stage_ptr, int64_t my_flag_ptr, int64_t epoch_ptr,
                 int64_t rank, int64_t hidden) {
  probe_write_kernel<<<1, 256>>>(
      reinterpret_cast<__nv_bfloat16*>(stage_ptr),
      reinterpret_cast<uint32_t*>(my_flag_ptr),
      reinterpret_cast<const uint32_t*>(epoch_ptr), (int)rank, (int)hidden);
}

void probe_read(int64_t peer_stage_ptr, int64_t peer_flag_ptr,
                int64_t epoch_ptr, int64_t out_ptr, int64_t hidden) {
  probe_read_kernel<<<1, 256>>>(
      reinterpret_cast<const __nv_bfloat16*>(peer_stage_ptr),
      reinterpret_cast<const uint32_t*>(peer_flag_ptr),
      reinterpret_cast<const uint32_t*>(epoch_ptr),
      reinterpret_cast<float*>(out_ptr), (int)hidden);
}
"""

_CPP_DECL = (
    "void probe_write(int64_t, int64_t, int64_t, int64_t, int64_t);\n"
    "void probe_read(int64_t, int64_t, int64_t, int64_t, int64_t);\n"
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
    out = torch.zeros(1, dtype=torch.float32, device="cuda")

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
        probe.probe_write(stage_ptr, my_flag_ptr, epoch_ptr, rank, args.hidden)
        probe.probe_read(peer_stage_ptr, peer_flag_ptr, epoch_ptr,
                         int(out.data_ptr()), args.hidden)
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
        probe.probe_write(stage_ptr, my_flag_ptr, epoch_ptr, rank, args.hidden)
        probe.probe_read(peer_stage_ptr, peer_flag_ptr, epoch_ptr,
                         int(out.data_ptr()), args.hidden)

    # ── Replay: each step epoch increments; expect peer's CURRENT payload ──
    failures = 0
    for step in range(1, args.steps + 1):
        g.replay()
        torch.cuda.synchronize()
        expected = float(step * 10 + peer)  # peer wrote epoch*10 + peer
        got = float(out.item())
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
