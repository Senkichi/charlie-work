# fleet-pass.ps1 — launches the charlie-work fleet supervisor (long-running daemon).
#
# `charlie fleet supervise` runs a continuous delta-aware loop: it polls local
# sidecar/verdict mtimes every ~20 s and only runs a full fleet_loop pass when
# something actionable changed or full_pass_interval_seconds (300 s) expires.
# This replaces the former one-shot `charlie fleet bash-rats` that the scheduled
# task re-invoked every 5 minutes. The scheduled task's 5-minute trigger now
# acts as a crash-recovery watchdog: if the daemon process dies, the next
# tick restarts it (MultipleInstancesPolicy=IgnoreNew prevents double-launch
# while it's alive). This covers a CRASHED supervisor only — a WEDGED
# supervisor (alive but not making progress) is still suppressed by
# IgnoreNew, so the trigger never fires. Wedge detection is handled in-
# process by the supervise-loop wrapper's WedgeWatchdog (issue #728), which
# monitors the supervisor heartbeat and terminates a wedged child so the
# next tick relaunches it.
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

# Must name the same command as the exit marker below. These two lines bracket one
# run in the log, and #862 changed the command under the exit marker only, leaving a
# pass that started as "supervise" and ended as "supervise-loop" -- which reads like
# two interleaved runs precisely when someone is untangling a restart.
"--- fleet supervise-loop start $(Get-Date -Format o) ---" | Out-File -FilePath $log -Append -Encoding utf8

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
#
# Set BOTH, and keep the ':surrogateescape' suffix.
#   - PYTHONUTF8=1 also fixes the open() default, which PYTHONIOENCODING does not.
#   - A BARE 'utf-8' is actively harmful: it overrides the surrogateescape handlers
#     that UTF-8 mode supplies and resets stdin/stdout to 'strict', turning graceful
#     degradation into a hard crash on any not-quite-clean byte off a pipe. This line
#     assigns unconditionally, so it overwrites any machine-wide or User-scope
#     hardening on the highest-frequency Python process on this box (the
#     charlie-fleet-pass scheduled task). Do not shorten.
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8:surrogateescape'

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
#
# INVOKED AS `python -m charlie_work`, NOT AS THE `charlie` CONSOLE SCRIPT.
# This is load-bearing and must not be "simplified" back (issue #854).
#
# `uv sync` reinstalls the editable project on every run, and its uninstall half
# must delete `.venv/Scripts/charlie.exe`. Windows locks running executables, so
# launching via the console script means the supervisor holds an exclusive handle
# on the exact file its own in-process self-deploy has to replace:
#
#   error: failed to remove file `...\.venv\...\../../Scripts/charlie.exe`:
#          Access is denied. (os error 5)
#
# That is structural, not a race — the process invoking the sync IS the lock
# holder, so no retry or backoff can ever succeed. Entering through
# `python -m charlie_work` (src/charlie_work/__main__.py, same `cli:main`) means
# the locked image is python.exe, which `uv sync` never replaces, leaving
# charlie.exe free to be rewritten.
# Entered through `fleet supervise-loop`, which runs `fleet supervise` as a child
# and relaunches it immediately when it exits to pick up new code (issue #862).
#
# Before this, the only thing that relaunched a self-deployed supervisor was this
# script's own 5-minute scheduled trigger, so every self-deploy left the fleet
# with no supervisor for up to a full interval -- silently, because the exit code
# was 0 either way.
#
# The relaunch decision stays inside Python: `supervise-loop` compares the child's
# exit code against its own EXIT_RESTART_REQUESTED constant, so this script never
# hardcodes that number. The bound (--max-relaunches) matters as much as the
# relaunch: on hitting it the wrapper EXITS, handing restart authority back to the
# 5-minute trigger below rather than pinning a stale wrapper process forever.
#
# Everything after `--` is forwarded to `fleet supervise` verbatim.
$cmdLine = "uv run --no-sync --project `"$root`" --directory `"$root`" python -m charlie_work fleet supervise-loop -- --max-runtime 0 >> `"$log`" 2>&1"
& cmd /c $cmdLine
"--- fleet supervise-loop exit=$LASTEXITCODE $(Get-Date -Format o) ---" | Out-File -FilePath $log -Append -Encoding utf8
