#!/bin/bash
# =============================================================================
# MCAT Question Generator — Setup & Run (Anthropic API)
# =============================================================================
# Usage:
#   export ANTHROPIC_API_KEY=sk-ant-...
#   chmod +x run.sh && ./run.sh
#
# Flags (pass after ./run.sh):
#   --discrete-only    Only generate discrete questions
#   --cars-only        Only generate CARS passages
#   --stats            Just show progress stats and exit
#   --reset            Clear all checkpoints
#   -v                 Verbose logging
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[mcat-gen]${NC} $1"; }
warn() { echo -e "${YELLOW}[mcat-gen]${NC} $1"; }
err()  { echo -e "${RED}[mcat-gen]${NC} $1"; }

# ---------------------------------------------------------------------------
# Check API key
# ---------------------------------------------------------------------------
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    err "ANTHROPIC_API_KEY is not set."
    err "Export it before running:  export ANTHROPIC_API_KEY=sk-ant-..."
    exit 1
fi

# ---------------------------------------------------------------------------
# Install uv if needed
# ---------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    [[ -f "$HOME/.cargo/env" ]] && source "$HOME/.cargo/env"

    if ! command -v uv &>/dev/null; then
        err "Failed to install uv. Install manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
    log "uv installed: $(uv --version)"
else
    log "uv found: $(uv --version)"
fi

# ---------------------------------------------------------------------------
# Install dependencies
# ---------------------------------------------------------------------------
log "Syncing dependencies..."
uv sync

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
mkdir -p logs

echo ""
log "============================================"
log "  Starting MCAT question generation"
log "============================================"
echo ""

uv run python -m src.main "$@" 2>&1 | tee logs/generation.log

echo ""
log "Done! Output files are in ./output/"
log "  Discrete: output/discrete_questions.jsonl"
log "  CARS:     output/cars_passages.jsonl"
