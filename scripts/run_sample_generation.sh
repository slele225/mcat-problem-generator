#!/usr/bin/env bash
#
# run_sample_generation.sh — unattended ~$100 MCAT sample generation.
#
# Runs, in order:
#   1. CARS         (full, 30 passages)            -> runs/prod_cars/
#   2. Science      (partial, 18 topics)           -> runs/prod_science/
#   3. Science figs (render only, no API calls)    -> runs/prod_science/
#   4. Discrete     (partial, 100 topics)          -> runs/prod_discrete/
#
# Run names are stable (prod_cars / prod_science / prod_discrete) so Friday's
# FULL run can point at the same --run-name folders and resume via checkpoints
# instead of regenerating today's work.
#
# Usage (from repo root):
#   bash scripts/run_sample_generation.sh
#
# Requires ANTHROPIC_API_KEY in the environment. Nothing is deleted; src/ and
# configs are never touched.

set -u -o pipefail

# --- Resolve repo root so the script works no matter where it's invoked from ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || { echo "ERROR: cannot cd to repo root ${REPO_ROOT}"; exit 1; }

CONFIG="configs/opus.yaml"

# --- Preflight checks -------------------------------------------------------
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ERROR: ANTHROPIC_API_KEY is not set. Export it before running, e.g.:" >&2
  echo "  export ANTHROPIC_API_KEY=sk-ant-...   (Git Bash / WSL)" >&2
  echo "Aborting before spending anything." >&2
  exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config not found at ${CONFIG} (run from the repo, configs/ must exist)." >&2
  exit 1
fi

mkdir -p runs

# --- Helpers ----------------------------------------------------------------
ts() { date '+%Y-%m-%d %H:%M:%S'; }

# run_step "Step name" "runs/<name>_log.txt" <python args...>
# Returns the command's exit code; never aborts the whole script itself.
run_step() {
  local name="$1"; shift
  local logfile="$1"; shift
  local start end elapsed rc

  echo ""
  echo "============================================================"
  echo ">>> [$(ts)] START: ${name}"
  echo ">>> log: ${logfile}"
  echo "============================================================"

  start="$(date +%s)"
  # tee so progress is visible live AND saved. pipefail makes $? reflect python.
  uv run python -m src.main "$@" 2>&1 | tee -a "${logfile}"
  rc="${PIPESTATUS[0]}"
  end="$(date +%s)"
  elapsed=$(( end - start ))

  if [[ "${rc}" -eq 0 ]]; then
    echo "<<< [$(ts)] OK: ${name}  (exit 0, ${elapsed}s)"
  else
    echo "<<< [$(ts)] FAILED: ${name}  (exit ${rc}, ${elapsed}s)"
  fi
  return "${rc}"
}

# count_lines <path> -> non-empty line count, or "0 (missing)" if absent
count_lines() {
  local p="$1"
  if [[ -f "${p}" ]]; then
    grep -c '[^[:space:]]' "${p}" 2>/dev/null || echo 0
  else
    echo "0 (missing)"
  fi
}

# --- Run --------------------------------------------------------------------
OVERALL_START="$(date +%s)"
FAILED_STEP=""

# A hard failure stops the run and reports which step died. We stop on the
# FIRST hard error rather than charging ahead and racking up spend on a broken
# pipeline. (Within a single pipeline, non-fatal per-item hiccups are handled
# by the pipeline's own retry/skip logic and do not exit non-zero.)
run_pipeline() {
  local name="$1"; shift
  local logfile="$1"; shift
  if ! run_step "${name}" "${logfile}" "$@"; then
    FAILED_STEP="${name}"
    return 1
  fi
  return 0
}

{
  run_pipeline "1/4 CARS (full, 30 passages)" "runs/prod_cars_log.txt" \
      --config "${CONFIG}" --cars-only --run-name prod_cars \
  && run_pipeline "2/4 Science passages (partial, 18 topics)" "runs/prod_science_log.txt" \
      --config "${CONFIG}" --science-passage-only --run-name prod_science --max-topics 18 \
  && run_pipeline "3/4 Science figure render (no API)" "runs/prod_science_log.txt" \
      --config "${CONFIG}" --run-name prod_science --render-figures \
  && run_pipeline "4/4 Discrete (partial, 100 topics)" "runs/prod_discrete_log.txt" \
      --config "${CONFIG}" --discrete-only --run-name prod_discrete --max-topics 100
}
RUN_RC=$?

OVERALL_END="$(date +%s)"
TOTAL_ELAPSED=$(( OVERALL_END - OVERALL_START ))

# --- Summary ----------------------------------------------------------------
echo ""
echo "############################################################"
echo "# SUMMARY  [$(ts)]   total elapsed: ${TOTAL_ELAPSED}s"
echo "############################################################"
echo ""
printf "  %-12s %-42s %s\n" "RUN" "OUTPUT FILE" "COUNT"
printf "  %-12s %-42s %s\n" "prod_cars"     "runs/prod_cars/cars_passages.jsonl"        "$(count_lines runs/prod_cars/cars_passages.jsonl) passages"
printf "  %-12s %-42s %s\n" "prod_science"  "runs/prod_science/science_passages.jsonl"  "$(count_lines runs/prod_science/science_passages.jsonl) passages"
printf "  %-12s %-42s %s\n" "prod_discrete" "runs/prod_discrete/discrete_questions.jsonl" "$(count_lines runs/prod_discrete/discrete_questions.jsonl) questions"
echo ""
echo "  Logs: runs/prod_cars_log.txt, runs/prod_science_log.txt, runs/prod_discrete_log.txt"
echo ""

if [[ "${RUN_RC}" -ne 0 ]]; then
  echo "  RESULT: STOPPED — step FAILED: ${FAILED_STEP}"
  echo "  Check its log above, fix the cause, and re-run (checkpointing will resume)."
  echo ""
  echo "  >>> DO NOT kick off Friday's full run until this is resolved. <<<"
  exit 1
fi

echo "  RESULT: ALL STEPS OK."
echo ""
echo "  >>> Before Friday's FULL run: eyeball the counts above and spot-check the"
echo "  >>> jsonl output (e.g. scripts/show_cars.py / show_passage.py) to confirm"
echo "  >>> quality. Friday's run reuses these prod_* folders and will RESUME,"
echo "  >>> not regenerate, today's work. <<<"
echo ""
