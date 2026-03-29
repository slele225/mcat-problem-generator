#!/bin/bash
# Launch vLLM server for MCAT question generation
#
# Prerequisites:
#   pip install vllm
#
# This starts vLLM with the Qwen2.5-32B-Instruct model on your A100.
# Adjust settings below as needed.

set -e

MODEL="Qwen/Qwen2.5-32B-Instruct"
PORT=8000
HOST="0.0.0.0"

# GPU memory utilization — 0.90 is safe for 80GB A100 with 32B model
GPU_MEMORY_UTILIZATION=0.90

# Max model length — 4096 is enough for our prompts
MAX_MODEL_LEN=4096

# Tensor parallel — set to number of GPUs if using multiple
TENSOR_PARALLEL_SIZE=1

echo "Starting vLLM server..."
echo "  Model: ${MODEL}"
echo "  Port: ${PORT}"
echo "  GPU Memory: ${GPU_MEMORY_UTILIZATION}"
echo "  Max Length: ${MAX_MODEL_LEN}"
echo ""
echo "Server will be available at http://localhost:${PORT}/v1"
echo "Health check: curl http://localhost:${PORT}/v1/models"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --dtype auto \
    --trust-remote-code
