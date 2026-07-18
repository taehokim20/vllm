#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-GPU accuracy test for the monokernel's sqrt-softplus routing.

Validates the new ScoringFunc::SQRT_SOFTPLUS path (added for DeepSeek-V4-Flash)
by comparing the monokernel output — with in-kernel sqrt(softplus) routing —
against a Python reference that routes the SAME way (sqrt(softplus(logit)),
top-k, renormalize, * routed_scaling_factor) and runs the block-wise MoE via
test_monokernel_accuracy.python_reference.

    score = sqrt(softplus(logit)),  softplus(x) = log1p(exp(x))  (beta=1, thr 20)
    weight_k = score_k / sum_topk(score) * routed_scaling_factor   (norm_topk_prob)

Run (1 GPU), from the repo root:
    python tests/moe/test_sqrtsoftplus_routing.py --shape e256_n2048_k4096
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import test_monokernel_accuracy as H  # noqa: E402
from vllm import _custom_ops as ops  # noqa: E402


def build_scratchpad(model_cfg, M):
    N_HALF, K = model_cfg["N_HALF"], model_cfg["K"]
    BS = 8 if M <= 8 else 64
    TEMP_ROWS = BS * 8 + 8
    UP_PROJ_BLOCKS = (N_HALF + 7) // 8
    ACT_SCALE_BLOCKS = (K + 127) // 128
    if model_cfg.get("up_col_halves") is not None:
        UP_COL_HALVES = model_cfg["up_col_halves"]
    else:
        DOWN_COL_TILE = K // (128 // 16)
        UP_COL_HALVES = (2 * N_HALF * DOWN_COL_TILE) // (128 * K)
    DOWN_ACT_BLOCK_SIZE = max(UP_COL_HALVES * 64, 64)
    TEMP_ACT_SCALE_COLS = N_HALF // DOWN_ACT_BLOCK_SIZE
    spec_size = (
        BS * K + TEMP_ROWS * N_HALF * 2 + TEMP_ROWS * UP_PROJ_BLOCKS * 4
        + TEMP_ROWS * N_HALF * 1 + TEMP_ROWS * TEMP_ACT_SCALE_COLS * 4
        + BS * K * 4 + BS * ACT_SCALE_BLOCKS * 4
    )
    return torch.zeros((spec_size + 3) // 4 + 4096 + (1 << 20),
                       dtype=torch.float32, device=H.DEV)


def sqrtsoftplus_routing(logits, top_k, routed_scaling_factor, renormalize):
    """Reference: score = sqrt(softplus(logit)); top-k; renorm; * scaling."""
    scores = torch.sqrt(F.softplus(logits.float(), beta=1.0, threshold=20.0))
    topk_vals, topk_ids = torch.topk(scores, top_k, dim=-1)
    if renormalize:
        topk_w = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp(min=1e-20)
    else:
        topk_w = topk_vals
    topk_w = topk_w * routed_scaling_factor
    return topk_w, topk_ids.to(torch.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="e256_n2048_k4096")
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--routed-scaling-factor", type=float, default=1.5)
    ap.add_argument("--tol", type=float, default=2e-2)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    model_cfg = H.MODELS.get(args.shape)
    if model_cfg is None:
        raise SystemExit(f"{args.shape!r} not in H.MODELS ({sorted(H.MODELS)})")
    E, N_HALF, K = model_cfg["E"], model_cfg["N_HALF"], model_cfg["K"]
    M, top_k = args.tokens, args.top_k
    rsf = args.routed_scaling_factor

    torch.manual_seed(args.seed)
    w13_f = torch.randn(E, 2 * N_HALF, K, device=H.DEV) * 0.1
    w2_f = torch.randn(E, K, N_HALF, device=H.DEV) * 0.1
    w13_fp8, s13 = H.quant_fp8_block_wise(w13_f)
    w2_fp8, s2 = H.quant_fp8_block_wise(w2_f)
    x = torch.randn(M, K, device=H.DEV, dtype=torch.bfloat16)
    logits = torch.randn(M, E, device=H.DEV, dtype=torch.bfloat16)

    # Python reference: identical sqrt-softplus routing + block-wise MoE.
    topk_w, topk_ids = sqrtsoftplus_routing(logits, top_k, rsf, renormalize=True)
    ref = H.python_reference(
        x, w13_fp8, s13, w2_fp8, s2, topk_w, topk_ids, N_HALF, K
    )[0]

    # Monokernel with in-kernel sqrt-softplus routing.
    cuda = ops.moe_monokernel_topk(
        x, logits, w13_fp8.contiguous(), s13.contiguous(),
        w2_fp8.contiguous(), s2.contiguous(), build_scratchpad(model_cfg, M),
        top_k=top_k, scoring_func="sqrtsoftplus", renormalize=True,
        routed_scaling_factor=rsf,
    )

    diff = (cuda.float() - ref.float())
    l2_rel = (diff.norm() / ref.float().norm().clamp(min=1e-9)).item()
    cs = H.cos_sim(cuda, ref)
    ok = l2_rel < args.tol
    print(
        f"[sqrtsoftplus routing] shape={args.shape} M={M} top_k={top_k} "
        f"rsf={rsf}\n  l2_rel={l2_rel:.4e}  cos_sim={cs:.6f}  "
        f"{'PASS' if ok else 'FAIL'} (tol {args.tol:g})",
        flush=True,
    )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
