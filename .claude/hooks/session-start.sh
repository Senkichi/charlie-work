#!/bin/bash
# SessionStart hook: build the venv in Claude Code on the web sessions so tests,
# ruff, and the .claude/settings.json gates (scripts/hook-python.sh) all work.
#
# Cloud-only on purpose: on the operator's host the main checkout's .venv is the
# live orchestrator's interpreter, and a uv sync there could swap packages out
# from under a running process.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
uv sync --all-extras
