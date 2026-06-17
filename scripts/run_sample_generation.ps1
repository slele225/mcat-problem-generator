<#
  run_sample_generation.ps1 - unattended ~$100 MCAT sample generation (PowerShell).

  Runs, in order:
    1. CARS         (full, 30 passages)            -> runs/prod_cars/
    2. Science      (partial, 18 topics)           -> runs/prod_science/
    3. Science figs (render only, no API calls)    -> runs/prod_science/
    4. Discrete     (partial, 100 topics)          -> runs/prod_discrete/

  Run names are stable (prod_cars / prod_science / prod_discrete) so Friday's
  FULL run can point at the same -run-name folders and resume via checkpoints
  instead of regenerating today's work.

  Usage (from PowerShell, repo root):
    powershell -ExecutionPolicy Bypass -File scripts\run_sample_generation.ps1

  Requires ANTHROPIC_API_KEY in the environment. Nothing is deleted; src\ and
  configs are never touched.
#>

$ErrorActionPreference = 'Stop'

# --- Resolve repo root so the script works no matter where it's invoked from ---
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

$Config = 'configs/opus.yaml'

# --- Preflight checks -------------------------------------------------------
if ([string]::IsNullOrEmpty($env:ANTHROPIC_API_KEY)) {
    Write-Host "ERROR: ANTHROPIC_API_KEY is not set. Set it before running, e.g.:" -ForegroundColor Red
    Write-Host "  `$env:ANTHROPIC_API_KEY = 'sk-ant-...'   (current PowerShell session)"
    Write-Host "Aborting before spending anything."
    exit 1
}

if (-not (Test-Path $Config)) {
    Write-Host "ERROR: config not found at $Config (run from the repo, configs\ must exist)." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path 'runs')) { New-Item -ItemType Directory -Path 'runs' | Out-Null }

# --- Helpers ----------------------------------------------------------------
function Get-Timestamp { Get-Date -Format 'yyyy-MM-dd HH:mm:ss' }

# Invoke-Step -Name <step> -LogFile <runs\<name>_log.txt> -PyArgs @(...)
# Returns the command's exit code; never aborts the whole script itself.
function Invoke-Step {
    param(
        [string]   $Name,
        [string]   $LogFile,
        [string[]] $PyArgs
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host ">> [$(Get-Timestamp)] START: $Name"
    Write-Host ">> log: $LogFile"
    Write-Host "============================================================"

    $start = Get-Date

    # IMPORTANT: python logs to stderr (normal INFO/WARNING output). With
    # `2>&1`, Windows PowerShell wraps each stderr line in a NativeCommandError
    # ErrorRecord; under the script's global $ErrorActionPreference='Stop' the
    # FIRST such line would terminate the script even though the process exits 0.
    # So we drop to 'Continue' for the duration of the native call, merge stderr
    # into stdout for the log, and decide success/failure SOLELY from
    # $LASTEXITCODE (the python/uv exit code) -- never from stderr presence.
    $rc = $null
    $prevEAP = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        # Out-Host sends the live/merged output to the console only. Without it,
        # Tee-Object's pass-through would land on this function's OUTPUT stream
        # and `return $rc` would hand the caller all the log lines + the code as
        # one array (making `$rc -ne 0` misfire). With Out-Host, the only thing
        # this function returns is the integer $rc below.
        uv run python -m src.main @PyArgs 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Host
        $rc = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prevEAP
    }
    # Tee-Object is a cmdlet and does not touch $LASTEXITCODE, so $rc is the
    # exit code of `uv`. Guard against a null (e.g. uv never launched).
    if ($null -eq $rc) { $rc = 1 }
    $elapsed = [int]((Get-Date) - $start).TotalSeconds

    if ($rc -eq 0) {
        Write-Host "<< [$(Get-Timestamp)] OK: $Name  (exit 0, ${elapsed}s)"
    } else {
        Write-Host "<< [$(Get-Timestamp)] FAILED: $Name  (exit $rc, ${elapsed}s)" -ForegroundColor Red
    }
    return $rc
}

# Get-LineCount <path> -> non-empty line count, or "0 (missing)" if absent
function Get-LineCount {
    param([string] $Path)
    if (Test-Path $Path) {
        $n = (Get-Content $Path | Where-Object { $_.Trim() -ne '' } | Measure-Object).Count
        return "$n"
    } else {
        return "0 (missing)"
    }
}

# --- Run --------------------------------------------------------------------
$OverallStart = Get-Date
$FailedStep   = ''

# A hard failure stops the run and reports which step died. We stop on the
# FIRST hard error rather than charging ahead and racking up spend on a broken
# pipeline. (Within a single pipeline, non-fatal per-item hiccups are handled
# by the pipeline's own retry/skip logic and do not exit non-zero.)
$steps = @(
    @{ Name = '1/4 CARS (full, 30 passages)';                 Log = 'runs/prod_cars_log.txt';    Args = @('--config', $Config, '--cars-only', '--run-name', 'prod_cars') },
    @{ Name = '2/4 Science passages (partial, 18 topics)';    Log = 'runs/prod_science_log.txt'; Args = @('--config', $Config, '--science-passage-only', '--run-name', 'prod_science', '--max-topics', '18') },
    @{ Name = '3/4 Science figure render (no API)';           Log = 'runs/prod_science_log.txt'; Args = @('--config', $Config, '--run-name', 'prod_science', '--render-figures') },
    @{ Name = '4/4 Discrete (partial, 100 topics)';           Log = 'runs/prod_discrete_log.txt'; Args = @('--config', $Config, '--discrete-only', '--run-name', 'prod_discrete', '--max-topics', '100') }
)

$RunFailed = $false
foreach ($s in $steps) {
    $rc = Invoke-Step -Name $s.Name -LogFile $s.Log -PyArgs $s.Args
    if ($rc -ne 0) {
        $FailedStep = $s.Name
        $RunFailed  = $true
        break
    }
}

$TotalElapsed = [int]((Get-Date) - $OverallStart).TotalSeconds

# --- Summary ----------------------------------------------------------------
Write-Host ""
Write-Host "############################################################"
Write-Host "# SUMMARY  [$(Get-Timestamp)]   total elapsed: ${TotalElapsed}s"
Write-Host "############################################################"
Write-Host ""
"{0,-13} {1,-44} {2}" -f 'RUN', 'OUTPUT FILE', 'COUNT' | Write-Host
"{0,-13} {1,-44} {2}" -f 'prod_cars',     'runs/prod_cars/cars_passages.jsonl',         "$(Get-LineCount 'runs/prod_cars/cars_passages.jsonl') passages"   | Write-Host
"{0,-13} {1,-44} {2}" -f 'prod_science',  'runs/prod_science/science_passages.jsonl',   "$(Get-LineCount 'runs/prod_science/science_passages.jsonl') passages" | Write-Host
"{0,-13} {1,-44} {2}" -f 'prod_discrete', 'runs/prod_discrete/discrete_questions.jsonl', "$(Get-LineCount 'runs/prod_discrete/discrete_questions.jsonl') questions" | Write-Host
Write-Host ""
Write-Host "  Logs: runs/prod_cars_log.txt, runs/prod_science_log.txt, runs/prod_discrete_log.txt"
Write-Host ""

if ($RunFailed) {
    Write-Host "  RESULT: STOPPED - step FAILED: $FailedStep" -ForegroundColor Red
    Write-Host "  Check its log above, fix the cause, and re-run (checkpointing will resume)."
    Write-Host ""
    Write-Host "  NOTE: DO NOT kick off Friday's full run until this is resolved."
    exit 1
}

Write-Host "  RESULT: ALL STEPS OK." -ForegroundColor Green
Write-Host ""
Write-Host "  NOTE: Before Friday's FULL run, eyeball the counts above and spot-check the"
Write-Host "  NOTE: jsonl output (e.g. scripts/show_cars.py / show_passage.py) to confirm"
Write-Host "  NOTE: quality. Friday's run reuses these prod_* folders and will RESUME,"
Write-Host "  NOTE: not regenerate, today's work."
Write-Host ""
