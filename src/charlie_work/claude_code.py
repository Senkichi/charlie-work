"""Claude Code worker adapter.

Codifies the emergent "empericus" Claude Code worker pattern described in
``docs/design/extraction-dossier.md`` (search "Claude Code worker loop"): in
production practice, no adapter code ever spawned a worker process — a human
created a git worktree, copied a junctioned venv in by hand, and pasted a
rendered prompt into an interactive ``claude`` session running in that
worktree. The worktree checkout alone gives the session the repo's tracked
``.claude/settings.json`` permissions and hooks for free.

This module promotes that into real code: create the worktree, hand the
rendered prompt to a headless ``claude -p`` process, and record a sidecar
JSON per worker so the orchestrator can reconcile state without parsing logs.
Field names intentionally mirror the sibling ``devin_shell`` adapter's
sidecar so ``doctor``/reconcile code can treat both worker kinds uniformly.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .state import utc_now
from .subprocess_runner import RunResult, run_captured
from .worktree import WorktreeInfo, create_worktree, remove_worktree

PROMPT_FILENAME = ".orchestrator-prompt.md"

# Windows-only flag: isolates the worker's process group so a Ctrl+C to the
# orchestrator doesn't propagate into an in-flight `claude` session. Absent
# on non-Windows platforms, where Popen simply ignores creationflags=0.
_CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


@dataclass(frozen=True)
class ClaudeWorkerRecord:
    issue_number: int
    branch: str
    worktree_path: str
    prompt_path: str
    command: tuple[str, ...]
    pid: int | None
    started_at: str
    log_path: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["command"] = list(self.command)
        return payload


def _sidecar_path(sessions_dir: Path, issue_number: int) -> Path:
    return sessions_dir / f"issue-{issue_number}.claude.json"


def _log_path(sessions_dir: Path, issue_number: int) -> Path:
    return sessions_dir / f"issue-{issue_number}.claude.log"


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def _write_record(sessions_dir: Path, record: ClaudeWorkerRecord) -> ClaudeWorkerRecord:
    _write_json_atomic(_sidecar_path(sessions_dir, record.issue_number), record.to_dict())
    return record


def _error_record(
    *,
    issue_number: int,
    branch: str,
    worktree_path: str,
    prompt_path: str,
    command: tuple[str, ...],
    log_path: str,
    error: str,
) -> ClaudeWorkerRecord:
    return ClaudeWorkerRecord(
        issue_number=issue_number,
        branch=branch,
        worktree_path=worktree_path,
        prompt_path=prompt_path,
        command=command,
        pid=None,
        started_at=utc_now(),
        log_path=log_path,
        error=error,
    )


def _render_command(
    command_template: tuple[str, ...],
    prompt_path: Path,
    *,
    issue_number: int,
    branch: str,
) -> tuple[str, ...]:
    # Same placeholder set as devin_shell so the two adapters' command
    # templates are drop-in compatible: {prompt_path} {issue_number} {branch}.
    values = {
        "prompt_path": str(prompt_path),
        "issue_number": str(issue_number),
        "branch": branch,
    }
    return tuple(part.format(**values) for part in command_template)


def launch_claude_worker(
    issue_number: int,
    branch: str,
    prompt_text: str,
    *,
    repo_root: Path,
    sessions_dir: Path,
    worktrees_dir: Path | None = None,
    venv_source: Path | None = None,
    command_template: tuple[str, ...] = ("claude", "-p", "--permission-mode", "acceptEdits"),
    env: dict[str, str] | None = None,
    rework: bool = False,
) -> ClaudeWorkerRecord:
    """Create an isolated worktree and launch a headless Claude Code worker in it.

    Never raises: worktree-creation failures and process-launch (``OSError``)
    failures both come back as an error record. If the worktree was created
    but the process failed to launch, the worktree is removed best-effort so
    a failed launch doesn't leak a half-made worktree.

    If ``rework`` is True, the worktree is created in rework mode (reuse existing
    worktree or attach to existing branch instead of creating a new branch).
    """
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = _log_path(sessions_dir, issue_number)

    try:
        worktree: WorktreeInfo = create_worktree(
            repo_root,
            branch,
            worktrees_dir=worktrees_dir,
            venv_source=venv_source,
            rework=rework,
        )
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as exc:
        record = _error_record(
            issue_number=issue_number,
            branch=branch,
            worktree_path="",
            prompt_path="",
            command=command_template,
            log_path=str(log_path),
            error=f"worktree creation failed: {exc}",
        )
        return _write_record(sessions_dir, record)

    prompt_path = worktree.path / PROMPT_FILENAME
    try:
        prompt_path.write_text(prompt_text, encoding="utf-8")
    except OSError as exc:
        remove_worktree(repo_root, worktree.path, force=True)
        record = _error_record(
            issue_number=issue_number,
            branch=branch,
            worktree_path=str(worktree.path),
            prompt_path=str(prompt_path),
            command=command_template,
            log_path=str(log_path),
            error=f"failed to write prompt file: {exc}",
        )
        return _write_record(sessions_dir, record)

    command = _render_command(
        command_template, prompt_path, issue_number=issue_number, branch=branch
    )
    feed_stdin = "{prompt_path}" not in "".join(command_template)
    # Workers inherit the orchestrator's environment, with config-provided
    # overrides merged on top — e.g. PYTEST_XDIST_AUTO_NUM_WORKERS to bound a
    # worker's local `pytest -n auto` so a fleet of them doesn't oversubscribe
    # the shared host (see docs/RUNBOOK.md "Local host saturation ceiling
    # (claude-code adapter)"). `env` is a validated mapping (see config.py).
    worker_env = {**os.environ, **{str(k): str(v) for k, v in (env or {}).items()}}

    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
            if feed_stdin:
                with prompt_path.open("r", encoding="utf-8") as prompt_handle:
                    process = subprocess.Popen(
                        command,
                        cwd=str(worktree.path),
                        stdin=prompt_handle,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        env=worker_env,
                        creationflags=_CREATE_NEW_PROCESS_GROUP,
                    )
            else:
                process = subprocess.Popen(
                    command,
                    cwd=str(worktree.path),
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=worker_env,
                    creationflags=_CREATE_NEW_PROCESS_GROUP,
                )
    except OSError as exc:
        remove_worktree(repo_root, worktree.path, force=True)
        record = _error_record(
            issue_number=issue_number,
            branch=branch,
            worktree_path=str(worktree.path),
            prompt_path=str(prompt_path),
            command=command,
            log_path=str(log_path),
            error=f"failed to launch claude: {exc}",
        )
        return _write_record(sessions_dir, record)

    record = ClaudeWorkerRecord(
        issue_number=issue_number,
        branch=branch,
        worktree_path=str(worktree.path),
        prompt_path=str(prompt_path),
        command=command,
        pid=process.pid,
        started_at=utc_now(),
        log_path=str(log_path),
        error=None,
    )
    return _write_record(sessions_dir, record)


def read_worker_records(sessions_dir: Path) -> list[ClaudeWorkerRecord]:
    """Load every ``issue-*.claude.json`` sidecar in ``sessions_dir``.

    Malformed sidecars are skipped rather than raising — a corrupt file from
    a crashed write must not take down reconciliation for every other worker.
    """
    if not sessions_dir.is_dir():
        return []
    records: list[ClaudeWorkerRecord] = []
    for path in sorted(sessions_dir.glob("issue-*.claude.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        try:
            records.append(
                ClaudeWorkerRecord(
                    issue_number=int(data["issue_number"]),
                    branch=str(data["branch"]),
                    worktree_path=str(data["worktree_path"]),
                    prompt_path=str(data["prompt_path"]),
                    command=tuple(data.get("command") or ()),
                    pid=data.get("pid"),
                    started_at=str(data.get("started_at", "")),
                    log_path=str(data.get("log_path", "")),
                    error=data.get("error"),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return records


def probe_claude(
    repo_root: Path, *, command: tuple[str, ...] = ("claude", "--version")
) -> RunResult:
    """Check the ``claude`` CLI is on PATH and runnable, for ``doctor``.

    ``command`` defaults to the package-default binary so callers that do not
    configure a custom ``claude_code.command`` get the standard probe.  Pass a
    custom tuple to exercise a configured wrapper binary.
    """
    return run_captured(list(command), cwd=repo_root, timeout_seconds=15)


__all__ = [
    "PROMPT_FILENAME",
    "ClaudeWorkerRecord",
    "launch_claude_worker",
    "read_worker_records",
    "probe_claude",
]
