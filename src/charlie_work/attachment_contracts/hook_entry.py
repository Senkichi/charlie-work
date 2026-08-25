"""PreToolUse stdin protocol.

stdin JSON: `{"tool_name": "Write|Edit|MultiEdit", "tool_input": {"file_path": ...}}`.

- No `.attachment-budgets.json` found walking up from the target file -> exit 0
  silently (fast no-op outside piloted repos).
- Unattended (`CHARLIE_FLEET_WORKER=1` or `CLAUDE_CODE_UNATTENDED=1`) -> ALWAYS
  advisory, never exit 2: print `{"hookSpecificOutput": {"additionalContext": ...}}`
  to stdout and best-effort append a marker line to
  `.var/attachment-contracts/advisories.jsonl`.
- Interactive + mode=enforce -> exit 2 with the redirect message on stderr.
- Mode source: `ATTACHMENT_CONTRACTS_MODE` env, else the baseline file's `mode`
  key, else `"advise"`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import IO, Mapping

from charlie_work.attachment_contracts.baseline import BASELINE_FILENAME, TamperError
from charlie_work.attachment_contracts.baseline import load as load_baseline
from charlie_work.attachment_contracts.check import check_file
from charlie_work.attachment_contracts.model import Finding

_ADVISORY_LOG_REL = Path(".var/attachment-contracts/advisories.jsonl")
_ACTIONABLE_SEVERITIES = frozenset({"block", "error"})


def _find_baseline_root(target: Path) -> Path | None:
    """Walk upward from `target` looking for `.attachment-budgets.json`."""
    start = target if target.is_dir() else target.parent
    current = start.resolve()
    while True:
        if (current / BASELINE_FILENAME).is_file():
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _is_unattended(env: Mapping[str, str]) -> bool:
    return env.get("CHARLIE_FLEET_WORKER") == "1" or env.get("CLAUDE_CODE_UNATTENDED") == "1"


def _resolve_mode(root: Path, env: Mapping[str, str]) -> str:
    override = env.get("ATTACHMENT_CONTRACTS_MODE")
    if override:
        return override
    try:
        document = load_baseline(root / BASELINE_FILENAME)
    except (TamperError, OSError, ValueError):
        return "advise"
    mode = document.get("mode", "advise")
    return mode if isinstance(mode, str) and mode else "advise"


def _format_message(findings: list[Finding]) -> str:
    lines = ["Attachment-Point Contracts:"]
    for f in findings:
        line = f"[{f.severity}] {f.identity} ({f.file}): {f.message}"
        if f.redirect:
            line += f" -> redirect: {f.redirect}"
        lines.append(line)
    return "\n".join(lines)


def _append_advisory_log(root: Path, findings: list[Finding]) -> None:
    log_path = root / _ADVISORY_LOG_REL
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            for f in findings:
                fh.write(
                    json.dumps(
                        {
                            "severity": f.severity,
                            "file": f.file,
                            "identity": f.identity,
                            "message": f.message,
                        }
                    )
                    + "\n"
                )
    except OSError:
        pass  # best-effort only; never block on log-write failure


def _extract_file_path(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    file_path = tool_input.get("file_path")
    return file_path if isinstance(file_path, str) and file_path else None


def main(
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    in_stream = stdin if stdin is not None else sys.stdin
    out_stream = stdout if stdout is not None else sys.stdout
    err_stream = stderr if stderr is not None else sys.stderr
    environ = env if env is not None else os.environ

    try:
        payload = json.load(in_stream)
    except (json.JSONDecodeError, ValueError):
        return 0  # malformed input -> fast no-op, never block on our own bug

    file_path = _extract_file_path(payload)
    if file_path is None:
        return 0

    target = Path(file_path)
    root = _find_baseline_root(target)
    if root is None:
        return 0  # no piloted repo above this file -> fast no-op

    try:
        rel_path = target.resolve().relative_to(root).as_posix()
    except ValueError:
        return 0  # target somehow outside root

    findings = check_file(rel_path, root)
    actionable = [f for f in findings if f.severity in _ACTIONABLE_SEVERITIES]
    if not actionable:
        return 0

    unattended = _is_unattended(environ)
    mode = _resolve_mode(root, environ)
    message = _format_message(actionable)

    if unattended or mode != "enforce":
        out_stream.write(json.dumps({"hookSpecificOutput": {"additionalContext": message}}) + "\n")
        _append_advisory_log(root, actionable)
        return 0

    err_stream.write(message + "\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
