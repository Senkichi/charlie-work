# dashboard.ps1 -- launches the charlie-work fleet dashboard (read-only, loopback).
#
# Restart semantics mirror the fleet supervise-loop wrapper: `dashboard serve`
# exits with supervise_loop.EXIT_RESTART_REQUESTED when the orchestrator HEAD
# drifted (self-deploy), and this launcher then starts it again to load the new
# code. The number is a cross-version wire contract (ADR-0004) and is declared
# ONCE below; tests/test_dashboard_launcher.py pins it to the Python constant.
# Any other exit code ends the loop and is returned to Task Scheduler, whose
# restart-on-failure policy (3 retries, 1 minute apart) takes over.
# The fleet pause flag is deliberately ignored: the dashboard is read-only.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Mirrors the default runtime.state_dir, same as fleet-pass.ps1.
$logDir = Join-Path $root '.var\charlie-work\logs'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Force -Path $logDir | Out-Null }
$log = Join-Path $logDir 'dashboard.log'

$ExitRestartRequested = 3
# Bound the relaunch loop so a stale launcher cannot spin forever.
$maxRelaunches = 20

# Native stderr is redirected inside cmd, never by PowerShell (PS 5.1 wraps it in
# ErrorRecords and writes UTF-16LE); see the long note in fleet-pass.ps1.
$ErrorActionPreference = 'Continue'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8:surrogateescape'

$cmdLine = "uv run --no-sync --project `"$root`" --directory `"$root`" python -m charlie_work dashboard serve >> `"$log`" 2>&1"
$exitCode = 0
for ($attempt = 0; $attempt -le $maxRelaunches; $attempt++) {
    "--- dashboard serve start $(Get-Date -Format o) ---" | Out-File -FilePath $log -Append -Encoding utf8
    & cmd /c $cmdLine
    $exitCode = $LASTEXITCODE
    "--- dashboard serve exit=$exitCode $(Get-Date -Format o) ---" | Out-File -FilePath $log -Append -Encoding utf8
    if ($exitCode -ne $ExitRestartRequested) { break }
}
exit $exitCode
