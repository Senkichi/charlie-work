# fleet-pass.ps1 — launches the charlie-work fleet supervisor (long-running daemon).
#
# `charlie fleet supervise` runs a continuous delta-aware loop: it polls local
# sidecar/verdict mtimes every ~20 s and only runs a full fleet_loop pass when
# something actionable changed or full_pass_interval_seconds (300 s) expires.
# This replaces the former one-shot `charlie fleet bash-rats` that the scheduled
# task re-invoked every 5 minutes. The scheduled task's 5-minute trigger now
# acts as a watchdog: if the daemon crashes, the next tick restarts it
# (MultipleInstancesPolicy=IgnoreNew prevents double-launch while it's alive).
#
# cwd MUST be a registered repo so the fleet command resolves the global config
# layer (%LOCALAPPDATA%\charlie-work\config.yaml) for the notify digest.
$ErrorActionPreference = 'Stop'   # fail fast on setup (bad root / unwritable log dir)
# Derived from this script's own location ($PSScriptRoot = .../scripts) rather
# than hardcoded, so the scheduled task keeps working if the repo is ever
# relocated or checked out under a different user profile.
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# NOTE: '.var\charlie-work' mirrors the default `runtime.state_dir`
# (config.RuntimeConfig, default ".var/charlie-work"). This path is
# deliberately NOT derived from config here (no Python/charlie invocation
# has happened yet at this point in the script) -- if a repo's config ever
# overrides runtime.state_dir, this literal must be updated to match.
$logDir = Join-Path $root '.var\charlie-work\logs'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Force -Path $logDir | Out-Null }
$log = Join-Path $logDir 'fleet-pass.log'

"--- fleet supervise start $(Get-Date -Format o) ---" | Out-File -FilePath $log -Append -Encoding utf8

# Native supervisor call. Three settings are REQUIRED for this to run at all under
# Windows PowerShell 5.1 (observed 2026-07-17 — old launcher wrote the marker then
# died before charlie ran, dispatching nothing):
#   --no-sync : the venv is synced out-of-band. An implicit `uv sync` emits progress
#     on STDERR; under PS 5.1 `& native *>> file` wraps the first stderr line as a
#     NativeCommandError, and with ErrorActionPreference=Stop that TERMINATES the
#     launcher (and kills the child) before the pass begins. --no-sync also avoids
#     the Windows venv sync-lock hang.
#   ErrorActionPreference=Continue : charlie logs to stderr on every pass; those
#     wrapped native-stderr records must stay non-terminating.
#   PYTHONUNBUFFERED=1 : flush charlie's stdout to the log live (else block-buffered
#     when redirected, so the log looks empty while the pass is actually running).
$ErrorActionPreference = 'Continue'
$env:PYTHONUNBUFFERED = '1'
# Force UTF-8 on charlie's own streams. Source comments and event payloads contain
# em-dashes; without this the daemon's output arrives mojibaked under the console's
# legacy codepage (see charlie-work notes on PS5.1 em-dash mojibake).
$env:PYTHONIOENCODING = 'utf-8'

# Redirect INSIDE cmd, not in PowerShell.
#
# The former `& uv run ... *>> $log` lost most of the daemon's logging (diagnosed
# 2026-07-25 while investigating issue #590). Under PS 5.1, redirecting a native
# command's stderr in PowerShell wraps each stderr write in a NativeCommandError
# ErrorRecord and emits it through the error stream, which:
#   - reorders stderr relative to stdout (a 09:48 log line appeared ABOVE a 09:45
#     one, so timestamps in the file were not monotonic),
#   - drops most records entirely (a 16-minute pass that emitted many INFO lines
#     left exactly one in the file), and
#   - writes UTF-16LE while the Out-File markers above write UTF-8, leaving a file
#     no single decoder can read.
# The practical consequence was that `charlie_work.fleet_dispatch` never appeared in
# 34,688 log lines even though it logs on every pass — so a WARNING that would have
# explained #590 immediately was invisible, and log silence could not be used as
# evidence about anything.
#
# Letting cmd own the handles means the child writes its own bytes straight to the
# file: correct ordering, nothing dropped, one consistent encoding, and no
# ErrorRecord wrapping to interact with ErrorActionPreference.
#
# Self-deploy happens IN-PROCESS: run_fleet_supervise calls self_deploy()
# (FF-pull origin/main + uv sync on dep changes) before each fleet_loop pass.
# The one-pass lag applies: a pass pulls new code but runs the already-imported
# module; the pulled code takes effect on the NEXT pass.
$cmdLine = "uv run --no-sync --project `"$root`" --directory `"$root`" charlie fleet supervise --max-runtime 0 >> `"$log`" 2>&1"
& cmd /c $cmdLine
"--- fleet supervise exit=$LASTEXITCODE $(Get-Date -Format o) ---" | Out-File -FilePath $log -Append -Encoding utf8
