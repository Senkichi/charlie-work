#!/usr/bin/env bash
# fleet-pass.sh -- POSIX port of fleet-pass.ps1: launches the charlie-work fleet
# supervisor (`fleet supervise-loop`, a long-running daemon) on macOS and Linux.
#
# Same contract as fleet-pass.ps1; read that file for the full history. In short:
# - The caller (a launchd job with a 300 s StartInterval on the Mac, the
#   Task Scheduler trigger on Windows) is the crash-recovery restart. launchd
#   never starts a second instance of a job that is still running, which is
#   what MultipleInstancesPolicy=IgnoreNew gives the Windows task.
# - The operator pause flag (`charlie fleet pause`) stops a launch: exit 0,
#   one log line, no supervisor.
# - Self-deploy and relaunch stay inside Python (`supervise-loop`); this script
#   never hardcodes EXIT_RESTART_REQUESTED.
#
# bash 3.2 and BSD userland compatible: no arrays, no `date -I`, no GNU flags.
set -u

# Derived from this script's own location, like $PSScriptRoot in the .ps1.
root=$(cd "$(dirname "$0")/.." && pwd) || exit 1
cd "$root" || exit 1

# Mirrors the default `runtime.state_dir` (.var/charlie-work), as the .ps1 does.
log_dir="$root/.var/charlie-work/logs"
mkdir -p "$log_dir" || exit 1
log="$log_dir/fleet-pass.log"

now() { date +%Y-%m-%dT%H:%M:%S%z; }

# Issue #1776 pause flag. Mirrors layout.fleet_pause_path / fleet_paths.fleet_dir
# on POSIX: CHARLIE_WORK_FLEET_DIR, else ${XDG_STATE_HOME:-~/.local/state}/charlie-work.
# The filename is a literal pinned to layout.FLEET_PAUSE_FILENAME by
# tests/test_fleet_pass_sh.py. Exit 0: a honored pause is not a launcher failure.
fleet_dir="${CHARLIE_WORK_FLEET_DIR:-}"
if [ -z "$fleet_dir" ]; then
    fleet_dir="${XDG_STATE_HOME:-$HOME/.local/state}/charlie-work"
fi
pause_flag="$fleet_dir/fleet-pause.json"
if [ -e "$pause_flag" ]; then
    echo "--- fleet supervise-loop NOT started: fleet paused ($pause_flag present); run 'charlie fleet resume' $(now) ---" >> "$log"
    exit 0
fi

echo "--- fleet supervise-loop start $(now) ---" >> "$log"

# Same environment as the .ps1: live-flushed output, UTF-8 everywhere, and the
# ':surrogateescape' suffix kept (a bare 'utf-8' resets the stdio error handler
# to 'strict').
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export PYTHONIOENCODING='utf-8:surrogateescape'

# --no-sync: the venv is synced out-of-band (and by self-deploy). `python -m
# charlie_work`, not the `charlie` console script, for parity with the .ps1
# (issue #854). Everything after `--` is forwarded to `fleet supervise`.
uv run --no-sync --project "$root" --directory "$root" \
    python -m charlie_work fleet supervise-loop -- --max-runtime 0 >> "$log" 2>&1
rc=$?
echo "--- fleet supervise-loop exit=$rc $(now) ---" >> "$log"
exit 0
