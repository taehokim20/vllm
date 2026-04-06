#!/usr/bin/env python3
"""Compare MoE LoRA benchmark results: Triton vs CUDA vs Hybrid."""
import json
import sys
from pathlib import Path


def load_result(path):
    with open(path) as f:
        return json.load(f)


def fmt(val):
    if val is None:
        return "N/A"
    return f"{val:.2f}"


def print_comparison(results: dict[str, dict], concurrency: str):
    backends = sorted(results.keys())
    base = results.get("triton") or next(iter(results.values()))

    metrics = [
        ("Output Throughput (tok/s)", "output_throughput", True),
        ("TTFT mean (ms)", "mean_ttft_ms", False),
        ("TTFT p50 (ms)", "median_ttft_ms", False),
        ("TTFT p99 (ms)", "p99_ttft_ms", False),
        ("TPOT mean (ms)", "mean_tpot_ms", False),
        ("TPOT p50 (ms)", "median_tpot_ms", False),
        ("TPOT p99 (ms)", "p99_tpot_ms", False),
        ("E2E Latency p50 (ms)", "median_e2el_ms", False),
        ("E2E Latency p99 (ms)", "p99_e2el_ms", False),
        ("Completed", "completed", None),
        ("Failed", "failed", None),
    ]

    col_w = 12
    hdr = f"{'Metric':<28}"
    for b in backends:
        hdr += f" {b:>{col_w}}"
    if "triton" in results and len(backends) > 1:
        for b in backends:
            if b != "triton":
                hdr += f" {'vs triton':>{col_w}}"

    print(f"\n{'='*(len(hdr)+2)}")
    print(f"  MoE LoRA E2E  (concurrency={concurrency})")
    print(f"{'='*(len(hdr)+2)}")
    print(hdr)
    print("-" * (len(hdr) + 2))

    for label, key, higher_better in metrics:
        row = f"{label:<28}"
        vals = {}
        for b in backends:
            v = results[b].get(key)
            vals[b] = v
            row += f" {fmt(v):>{col_w}}"

        if "triton" in vals and higher_better is not None:
            t = vals["triton"]
            for b in backends:
                if b != "triton":
                    c = vals[b]
                    if c and t and c != 0 and t != 0:
                        ratio = c / t if higher_better else t / c
                        row += f" {ratio:>{col_w}.3f}x"
                    else:
                        row += f" {'':>{col_w}}"
        print(row)

    print(f"{'='*(len(hdr)+2)}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_moe_lora_results.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])

    # Group by concurrency
    from collections import defaultdict
    groups = defaultdict(dict)
    for f in sorted(results_dir.glob("*.json")):
        # e.g. "triton_concurrency4.json" -> backend="triton", conc="4"
        parts = f.stem.split("_concurrency")
        if len(parts) == 2:
            backend, conc = parts
            groups[conc][backend] = load_result(f)

    if not groups:
        print(f"No result files found in {results_dir}")
        sys.exit(1)

    for conc in sorted(groups.keys(), key=int):
        print_comparison(groups[conc], conc)


if __name__ == "__main__":
    main()
