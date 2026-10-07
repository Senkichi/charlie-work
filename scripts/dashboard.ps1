# dashboard.ps1 -- launches the charlie-work fleet dashboard (read-only, loopback).
#
# Restart semantics mirror the fleet supervise-loop wrapper: `dashboard serve`
# exits with supervise_loop.EXIT_RESTART_REQUESTED when the orchestrator HEAD
# drifted (self-deploy or a manual pull), and this launcher syncs the venv to
# uv.lock and starts it again to load the new code. The number is a cross-version wire contract (ADR-0004) and is declared
# ONCE below; tests/test_dashboard_launcher.py pins it to the Python constant.
# Any other exit code ends the loop and is returned to Task Scheduler, whose
# restart-on-failure policy (3 retries, 1 minute apart) takes over.
# The relaunch budget is a RATE, not a lifetime count: a normal fleet self-deploys
# dozens of times a day, so only a burst (more than $maxRelaunches within
# $windowMinutes minutes, i.e. a crash-looping build) stops the launcher. It then logs
# the reason and exits 1 so Task Scheduler's failure policy and the repeating trigger
# in charlie-dashboard-task.xml take over. A short delay separates relaunches.
# The fleet pause flag is deliberately ignored: the dashboard is read-only.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Mirrors the default runtime.state_dir, same as fleet-pass.ps1.
$logDir = Join-Path $root '.var\charlie-work\logs'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Force -Path $logDir | Out-Null }
$log = Join-Path $logDir 'dashboard.log'

$ExitRestartRequested = 3
$maxRelaunches = 5
$windowMinutes = 10
$relaunchDelaySeconds = 5

# Native stderr is redirected inside cmd, never by PowerShell (PS 5.1 wraps it in
# ErrorRecords and writes UTF-16LE); see the long note in fleet-pass.ps1.
$ErrorActionPreference = 'Continue'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8:surrogateescape'

# Venv sync before EVERY launch. A HEAD drift is usually a pull of new code, and new
# code can need a new lock (e.g. config rules for fields only a newer ci-fleet has).
# `serve` itself runs with --no-sync, so without this step the relaunch loads the new
# tree over the old venv and fails at config load until someone syncs by hand.
#   --locked  : never rewrite uv.lock; a stale lock fails loudly instead.
#   --inexact : never uninstall extras or dev tools someone added to this venv.
# Best effort: a failed sync (e.g. a venv file held open) is logged and the
# launch goes ahead -- serving on the old venv beats not serving at all. When
# the venv already matches the lock this is a no-op of about a second.
$syncLine = "uv sync --locked --inexact --project `"$root`" --directory `"$root`" >> `"$log`" 2>&1"
$cmdLine = "uv run --no-sync --project `"$root`" --directory `"$root`" python -m charlie_work dashboard serve >> `"$log`" 2>&1"
$exitCode = 0
$relaunches = New-Object System.Collections.Generic.List[datetime]
while ($true) {
    & cmd /c $syncLine
    if ($LASTEXITCODE -ne 0) {
        "--- dashboard venv sync failed exit=$LASTEXITCODE; launching on the current venv $(Get-Date -Format o) ---" | Out-File -FilePath $log -Append -Encoding utf8
    }
    "--- dashboard serve start $(Get-Date -Format o) ---" | Out-File -FilePath $log -Append -Encoding utf8
    & cmd /c $cmdLine
    $exitCode = $LASTEXITCODE
    "--- dashboard serve exit=$exitCode $(Get-Date -Format o) ---" | Out-File -FilePath $log -Append -Encoding utf8
    if ($exitCode -ne $ExitRestartRequested) { break }
    $now = Get-Date
    $cutoff = $now.AddMinutes(-$windowMinutes)
    [void]$relaunches.RemoveAll([Predicate[datetime]]{ param($t) $t -lt $cutoff })
    if ($relaunches.Count -ge $maxRelaunches) {
        "--- dashboard launcher giving up: $($relaunches.Count) relaunches within $windowMinutes minutes (crash loop?) $(Get-Date -Format o) ---" | Out-File -FilePath $log -Append -Encoding utf8
        $exitCode = 1
        break
    }
    $relaunches.Add($now)
    Start-Sleep -Seconds $relaunchDelaySeconds
}
exit $exitCode
