#!/bin/bash
# MoE LoRA E2E Benchmark: CUDA BGMV vs Triton
# Usage: bash benchmarks/bench_moe_lora_e2e.sh [concurrency]
#
# Prerequisites:
#   1. Download LoRA adapter: huggingface-cli download jeeejeee/qwen3-moe-text2sql-spider
#   2. Note the local path (printed by the download command)
#
# This script:
#   1. Starts vLLM server with CUDA MoE kernel, runs benchmark, kills server
#   2. Starts vLLM server with Triton MoE kernel, runs benchmark, kills server
#   3. Prints comparison

set -euo pipefail

MODEL="Qwen/Qwen3-30B-A3B"
LORA_NAME="qwen3-moe-lora"
LORA_PATH="jeeejeee/qwen3-moe-text2sql-spider"
TP=1
PORT=8000
MAX_LORAS=2
MAX_LORA_RANK=32
INPUT_LEN=512
OUTPUT_LEN=128
CONCURRENCY=${1:-4}
NUM_PROMPTS=100
RESULTS_DIR="bench_results_moe_lora"

mkdir -p "$RESULTS_DIR"

run_benchmark() {
    local backend=$1  # "cuda" or "triton" or "hybrid"
    local tag="${backend}_concurrency${CONCURRENCY}"
    local result_file="${RESULTS_DIR}/${tag}.json"

    echo "============================================"
    echo "  Backend: ${backend} | Concurrency: ${CONCURRENCY}"
    echo "============================================"

    # Set the MoE LoRA backend
    unset VLLM_MOE_LORA_BACKEND 2>/dev/null || true
    unset VLLM_MOE_LORA_PREFILL_BACKEND 2>/dev/null || true
    unset VLLM_MOE_LORA_DECODE_BACKEND 2>/dev/null || true
    unset VLLM_MOE_LORA_DECODE_THRESHOLD 2>/dev/null || true

    if [ "$backend" = "triton" ]; then
        export VLLM_MOE_LORA_BACKEND=triton
    elif [ "$backend" = "cuda" ]; then
        export VLLM_MOE_LORA_BACKEND=cuda
    elif [ "$backend" = "hybrid" ]; then
        # Default hybrid: Triton for prefill, CUDA for decode
        # Threshold=32: use CUDA when num_tokens <= 32 (decode),
        # Triton when num_tokens > 32 (prefill)
        export VLLM_MOE_LORA_PREFILL_BACKEND=triton
        export VLLM_MOE_LORA_DECODE_BACKEND=cuda
        export VLLM_MOE_LORA_DECODE_THRESHOLD=32
    fi

    # Start vLLM server
    echo "Starting vLLM server (${backend})..."
    vllm serve "$MODEL" \
        --port "$PORT" \
        --tensor-parallel-size "$TP" \
        --enable-lora \
        --max-loras "$MAX_LORAS" \
        --max-lora-rank "$MAX_LORA_RANK" \
        --lora-modules "${LORA_NAME}=${LORA_PATH}" \
        --enforce-eager \
        --max-model-len 1024 \
        --gpu-memory-utilization 0.95 \
        --trust-remote-code \
        &
    SERVER_PID=$!

    # Wait for server to be ready
    echo "Waiting for server to start..."
    for i in $(seq 1 120); do
        if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "Server ready after ${i}s"
            break
        fi
        sleep 1
    done

    if ! curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "ERROR: Server failed to start"
        kill $SERVER_PID 2>/dev/null || true
        return 1
    fi

    # Run benchmark
    echo "Running benchmark..."
    vllm bench serve \
        --model "$MODEL" \
        --backend openai-chat \
        --endpoint /v1/chat/completions \
        --base-url "http://localhost:${PORT}" \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency "$CONCURRENCY" \
        --lora-modules "$LORA_NAME" \
        --percentile-metrics ttft,tpot,e2el \
        --save-result \
        --result-dir "$RESULTS_DIR" \
        --result-filename "${tag}.json" \
        2>&1 | tee "${RESULTS_DIR}/${tag}.log"

    # Kill server
    echo "Stopping server..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 5

    echo "Results saved to ${result_file}"
    echo ""
}

# Run all three backends
run_benchmark "cuda"
run_benchmark "hybrid"
run_benchmark "triton"

echo ""
echo "============================================"
echo "  COMPARISON COMPLETE"
echo "============================================"
echo "Results in: ${RESULTS_DIR}/"
echo ""
echo "To compare, run:"
echo "  python benchmarks/compare_moe_lora_results.py ${RESULTS_DIR}"
