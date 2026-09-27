#!/usr/bin/env bash
# Launch the project venv's interpreter for the hooks in .claude/settings.json.
#
# The venv layout is platform-specific: .venv/Scripts/python.exe on the Windows
# host (hooks run under Git Bash there), .venv/bin/python on Linux/macOS (Claude
# Code on the web). Resolved from this script's own location, never the cwd, so a
# session that has cd'd into a worktree still gets the project root's venv.
#
# No venv at all -> exit 0 with a note on stderr: the hooks' documented
# missing-interpreter contract is fail-OPEN (see worker_stop_gate.py).
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for py in "$root/.venv/Scripts/python.exe" "$root/.venv/bin/python"; do
  if [ -x "$py" ]; then
    exec "$py" "$@"
  fi
done
echo "hook-python: no interpreter under $root/.venv; hook skipped (fail-open)" >&2
exit 0
