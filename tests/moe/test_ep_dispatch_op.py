#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone 2-GPU correctness test for the in-kernel EP dispatch (Stage B).

Validates the in-kernel peer-read dispatch + the ``expert_base`` local-expert
filter end-to-end at the op level: each of the two EP=2 ranks computes only its
128 LOCAL experts over the FULL token set — its own token rows are staged
locally, the REMOTE rows are peer-read over NVLink from the peer's symmetric
IPC staging buffer inside the kernel — producing a per-rank PARTIAL. The two
partials SUM to the single-rank non-EP reference over all 256 experts.

    reference(all 256 experts)  ==  partial_rank0(experts 0..127)
                                   + partial_rank1(experts 128..255)

Ordering (a rank must not peer-read the peer's rows before the peer has staged
them) is provided here by ``dist.barrier()``. The CUDA-graph-safe one-sided
handshake (ep_set_ready / ep_wait_ready) is validated separately in
``test_ep_graphsafe_signal.py``; this test isolates the KERNEL numerics.

Target shape: DeepSeek-V4-Flash full-N (E=256, N_half=2048, K=4096), which is
the EP=2 per-rank shape (EP shards experts, not N). Requires the shape to be
onboarded + built into the extension.

Run on 2 GPUs:
    torchrun --nproc_per_node=2 tests/moe/test_ep_dispatch_op.py
Options: --shape e256_n2048_k4096  --tokens 8  --tol 2e-2
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

# test_monokernel_accuracy lives at the repo root; when run via
# `torchrun tests/moe/...` the script's own dir is on sys.path[0], not the
# repo root, so add it explicitly.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import test_monokernel_accuracy as H  # noqa: E402
from vllm import _custom_ops as ops
from vllm.distributed.moe_ll_workspace import MoELLWorkspace


def build_scratchpad(model_cfg, M: int) -> torch.Tensor:
    """Allocate the monokernel scratchpad for this shape.

    Mirrors the sizing in test_monokernel_accuracy.accuracy_test (which the
    working bench uses), plus a large safety margin — over-allocation is safe
    because the kernel addresses the scratchpad via compile-time MoEGemmSpec
    offsets and only requires size >= the footprint.
    """
    N_HALF = model_cfg["N_HALF"]
    K = model_cfg["K"]
    is_bs8 = M <= 8
    BS = 8 if is_bs8 else 64
    TEMP_ROWS = BS * 8 + 8
    UP_PROJ_BLOCKS = (N_HALF + 7) // 8
    ACT_SCALE_BLOCKS = (K + 127) // 128
    GRID_SIZE = 128
    if is_bs8 and model_cfg.get("up_col_halves") is not None:
        UP_COL_HALVES = model_cfg["up_col_halves"]
    elif is_bs8:
        DOWN_GROUPS = 16
        DOWN_GRID = GRID_SIZE // DOWN_GROUPS
        DOWN_COL_TILE = K // DOWN_GRID
        UP_COL_HALVES = (2 * N_HALF * DOWN_COL_TILE) // (128 * K)
    else:
        UP_COL_HALVES = 1
    DOWN_ACT_BLOCK_SIZE = max(UP_COL_HALVES * 64, 64)
    TEMP_ACT_SCALE_COLS = N_HALF // DOWN_ACT_BLOCK_SIZE
    spec_size = (
        BS * K
        + TEMP_ROWS * N_HALF * 2
        + TEMP_ROWS * UP_PROJ_BLOCKS * 4
        + TEMP_ROWS * N_HALF * 1
        + TEMP_ROWS * TEMP_ACT_SCALE_COLS * 4
        + BS * K * 4
        + BS * ACT_SCALE_BLOCKS * 4
    )
    scratch_floats = (spec_size + 3) // 4 + 4096 + (1 << 20)  # + 4 MB margin
    return torch.zeros(scratch_floats, dtype=torch.float32, device="cuda")


def resolve_model_cfg(shape: str):
    if shape in H.MODELS:
        return H.MODELS[shape]
    # accept an alias (e.g. deepseek_v4_flash) via the registry mapping.
    from vllm.model_executor.layers.fused_moe import monokernel_shapes as REG

    row = REG.BY_NAME.get(shape)
    if row is not None:
        for key in (row["key"], *row.get("aliases", [])):
            if key in H.MODELS:
                return H.MODELS[key]
    raise SystemExit(f"shape {shape!r} not in H.MODELS ({sorted(H.MODELS)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="e256_n2048_k4096")
    ap.add_argument("--tokens", type=int, default=8, help="gathered tile size M (<=8)")
    ap.add_argument("--tol", type=float, default=2e-2, help="L2-relative tolerance")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, "EP=2 test expects 2 ranks."
    torch.cuda.set_device(rank)
    peer = rank ^ 1

    model_cfg = resolve_model_cfg(args.shape)
    E, N_HALF, K = model_cfg["E"], model_cfg["N_HALF"], model_cfg["K"]
    top_k = model_cfg.get("default_top_k", 6)
    scoring = model_cfg.get("scoring_func", "softmax")
    M = args.tokens
    assert M <= 8 and M % world == 0, "M must be <=8 and divisible by world size."
    assert E % world == 0
    n_local = E // world  # 128 experts/rank at EP=2

    # Identical inputs on every rank (same seed) so each rank holds the full
    # weight set and the same tokens/router logits; EP is emulated purely by
    # the expert_base filter + the peer-read staging.
    torch.manual_seed(args.seed)
    w13_f = torch.randn(E, 2 * N_HALF, K, device="cuda") * 0.1
    w2_f = torch.randn(E, K, N_HALF, device="cuda") * 0.1
    w13_fp8, s13 = H.quant_fp8_block_wise(w13_f)
    w2_fp8, s2 = H.quant_fp8_block_wise(w2_f)
    w13_fp8, s13 = w13_fp8.contiguous(), s13.contiguous()
    w2_fp8, s2 = w2_fp8.contiguous(), s2.contiguous()
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    logits = torch.randn(M, E, device="cuda", dtype=torch.bfloat16)

    # ── Reference: full non-EP output over all 256 experts. ──────────────
    ref = ops.moe_monokernel_topk(
        x, logits, w13_fp8, s13, w2_fp8, s2, build_scratchpad(model_cfg, M),
        top_k=top_k, scoring_func=scoring, renormalize=True,
    )

    # ── EP path: stage owned rows, peer-read remote rows, local experts. ──
    ws = MoELLWorkspace(max_num_tokens=M, hidden_dim=K, tp_group=dist.group.WORLD)
    local_view, peer_views = ws.ep_activation_views(M)
    own = M // world
    own_lo, own_hi = rank * own, rank * own + own

    # Own rows := x[own]; remote rows := 0 (so a correct result REQUIRES the
    # in-kernel peer-read to overwrite them from the peer's staging buffer).
    local_view.zero_()
    local_view[own_lo:own_hi].copy_(x[own_lo:own_hi])
    dist.barrier()  # both ranks have staged -> safe to peer-read

    partial = ops.moe_monokernel_topk_ep(
        local_view, logits, w13_fp8, s13, w2_fp8, s2,
        build_scratchpad(model_cfg, M),
        peer_activations=peer_views[peer],
        expert_base=rank * n_local,
        local_token_start=own_lo,
        n_local_tokens=own,
        top_k=top_k, scoring_func=scoring, renormalize=True,
    )
    dist.barrier()

    # Sum the per-rank partials in fp32 across the EP group.
    summed = partial.float()
    dist.all_reduce(summed)  # partial_rank0 + partial_rank1

    diff = (summed - ref.float())
    l2_rel = (diff.norm() / ref.float().norm().clamp(min=1e-9)).item()
    ok = l2_rel < args.tol

    if rank == 0:
        print(
            f"[EP dispatch] shape={args.shape} M={M} top_k={top_k} "
            f"scoring={scoring} EP=2 (n_local={n_local})\n"
            f"  l2_rel(partial0+partial1 vs full-256 reference) = {l2_rel:.4e}\n"
            f"  {'PASS' if ok else 'FAIL'} (tolerance {args.tol:g})",
            flush=True,
        )

    ws.destroy()
    dist.destroy_process_group()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
