#!/bin/bash
# =============================================================================
# MCAT Question Generator — Full Setup & Run
# =============================================================================
# Run this after SSH-ing into your GPU server:
#   chmod +x run.sh && ./run.sh
#
# What it does:
#   1. Installs uv (if not present)
#   2. Creates venv and installs all dependencies
#   3. Starts vLLM server in the background
#   4. Waits for server to be ready
#   5. Runs the question generation pipeline
#   6. Shows stats when done (or on Ctrl+C)
#
# Flags (pass after ./run.sh):
#   --discrete-only    Only generate discrete questions
#   --cars-only        Only generate CARS passages
#   --stats            Just show progress stats and exit
#   --reset            Clear all checkpoints
#   -v                 Verbose logging
#
# Examples:
#   ./run.sh                     # Run everything
#   ./run.sh --discrete-only     # Discrete questions only
#   ./run.sh --stats             # Check progress
#   ./run.sh --discrete-only -v  # Discrete with debug logs
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="Qwen/Qwen2.5-32B-Instruct"
VLLM_PORT=8000
VLLM_PID_FILE=".vllm.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[mcat-gen]${NC} $1"; }
warn() { echo -e "${YELLOW}[mcat-gen]${NC} $1"; }
err()  { echo -e "${RED}[mcat-gen]${NC} $1"; }

# ---------------------------------------------------------------------------
# Cleanup: kill vLLM server on exit
# ---------------------------------------------------------------------------
cleanup() {
    if [[ -f "$VLLM_PID_FILE" ]]; then
        local pid
        pid=$(cat "$VLLM_PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            log "Shutting down vLLM server (PID $pid)..."
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
        rm -f "$VLLM_PID_FILE"
    fi
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Step 1: Install uv if needed
# ---------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"

    if ! command -v uv &>/dev/null; then
        # Try sourcing the env file uv creates
        [[ -f "$HOME/.cargo/env" ]] && source "$HOME/.cargo/env"
    fi

    if ! command -v uv &>/dev/null; then
        err "Failed to install uv. Install manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
    log "uv installed: $(uv --version)"
else
    log "uv found: $(uv --version)"
fi

# ---------------------------------------------------------------------------
# Step 2: Create venv and install dependencies
# ---------------------------------------------------------------------------
if [[ ! -d ".venv" ]]; then
    log "Creating virtual environment and installing dependencies..."
    uv sync
    log "Dependencies installed."
else
    log "Virtual environment exists. Running uv sync to ensure deps are current..."
    uv sync
fi

# ---------------------------------------------------------------------------
# Step 3: Handle --stats and --reset (no server needed)
# ---------------------------------------------------------------------------
for arg in "$@"; do
    if [[ "$arg" == "--stats" || "$arg" == "--reset" ]]; then
        uv run python -m src.main "$@"
        exit 0
    fi
done

# ---------------------------------------------------------------------------
# Step 4: Start vLLM server (if not already running)
# ---------------------------------------------------------------------------
vllm_running() {
    curl -s "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1
}

if vllm_running; then
    log "vLLM server already running on port ${VLLM_PORT}"
else
    log "Starting vLLM server with ${MODEL}..."
    log "This will download the model on first run (~20GB). Be patient."
    echo ""

    uv run python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --host "0.0.0.0" \
        --port "${VLLM_PORT}" \
        --gpu-memory-utilization 0.90 \
        --max-model-len 4096 \
        --tensor-parallel-size 1 \
        --dtype auto \
        --trust-remote-code \
        > logs/vllm.log 2>&1 &

    VLLM_PID=$!
    echo "$VLLM_PID" > "$VLLM_PID_FILE"
    log "vLLM server starting (PID ${VLLM_PID}), logs at logs/vllm.log"

    # Wait for server to be ready
    log "Waiting for vLLM server to be ready..."
    MAX_WAIT=600  # 10 minutes (model download can take a while)
    WAITED=0
    while ! vllm_running; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            err "vLLM server died. Check logs/vllm.log for errors."
            tail -20 logs/vllm.log 2>/dev/null || true
            exit 1
        fi
        sleep 5
        WAITED=$((WAITED + 5))
        if (( WAITED % 30 == 0 )); then
            log "Still waiting... (${WAITED}s elapsed)"
        fi
        if (( WAITED >= MAX_WAIT )); then
            err "Timed out waiting for vLLM server after ${MAX_WAIT}s"
            exit 1
        fi
    done

    log "vLLM server is ready! (took ${WAITED}s)"
fi

echo ""
log "============================================"
log "  Starting MCAT question generation"
log "============================================"
echo ""

# ---------------------------------------------------------------------------
# Step 5: Run the pipeline
# ---------------------------------------------------------------------------
mkdir -p logs
uv run python -m src.main "$@" 2>&1 | tee logs/generation.log

echo ""
log "Done! Output files are in ./output/"
log "  Discrete: output/discrete_questions.jsonl"
log "  CARS:     output/cars_passages.jsonl"
