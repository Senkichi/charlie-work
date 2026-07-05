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
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .state import utc_now
from .subprocess_runner import RunResult, run_captured
from .worktree import WorktreeInfo, create_worktree, remove_worktree

PROMPT_FILENAME = ".orchestrator-prompt.md"

# Provider throttle signatures — matched against session log tails to classify
# failure kinds. Keep these in one adapter-owned constant, not scattered.
# Pattern for rate-limit errors (e.g., "Reached overall message rate limit")
_RATE_LIMIT_PATTERN = re.compile(
    r"Reached overall message rate limit|rate limit|too many requests",
    re.IGNORECASE,
)
# Pattern for quota-exhaustion errors (e.g., "daily usage quota has been exhausted")
_QUOTA_EXHAUSTED_PATTERN = re.compile(
    r"daily usage quota has been exhausted|quota exceeded|usage limit",
    re.IGNORECASE,
)
# Pattern for "resets in N minutes" to extract cooldown duration
_RESETS_IN_PATTERN = re.compile(r"resets? in (\d+) minutes?", re.IGNORECASE)

# Default cooldown durations when we can't parse a specific reset time
_DEFAULT_RATE_LIMIT_COOLDOWN_MINUTES = 15
_DEFAULT_QUOTA_COOLDOWN_HOURS = 24

# Windows-only flag: isolates the worker's process group so a Ctrl+C to the
# orchestrator doesn't propagate into an in-flight `claude` session. Absent
# on non-Windows platforms, where Popen simply ignores creationflags=0.
_CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

_WIN_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_WIN_STILL_ACTIVE = 259


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
    failure_kind: str | None = None  # "rate_limited" | "quota_exhausted" | ...
    reclaimed: str | None = None  # "fetch-fallback" | "pruned" | "salvaged" | None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["command"] = list(self.command)
        return payload


def _sidecar_path(sessions_dir: Path, issue_number: int) -> Path:
    return sessions_dir / f"issue-{issue_number}.claude.json"


def _log_path(sessions_dir: Path, issue_number: int, *, rework: bool = False) -> Path:
    suffix = "-rework.claude.log" if rework else ".claude.log"
    return sessions_dir / f"issue-{issue_number}{suffix}"


def _classify_session_failure(log_path: Path) -> tuple[str | None, str | None]:
    """Classify a session failure by matching the log tail against provider throttle signatures.

    Returns a tuple of (failure_kind, throttled_until_iso):
    - failure_kind: "rate_limited" | "quota_exhausted" | None
    - throttled_until_iso: ISO timestamp when the cooldown ends, or None if not applicable

    This is called after a session exits to detect provider throttling and set a cool-down window.
    """
    if not log_path.exists():
        return None, None

    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None

    # Check the last 2KB of the log (where error messages appear)
    tail = log_text[-2048:] if len(log_text) > 2048 else log_text

    # Check for quota exhaustion first (more severe)
    if _QUOTA_EXHAUSTED_PATTERN.search(tail):
        # Quota exhaustion uses a fixed 24-hour cooldown regardless of reset time
        cooldown = timedelta(hours=_DEFAULT_QUOTA_COOLDOWN_HOURS)
        throttled_until = datetime.now(UTC) + cooldown
        return "quota_exhausted", throttled_until.replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        )

    # Check for rate limiting
    if _RATE_LIMIT_PATTERN.search(tail):
        # Try to parse "resets in N minutes"
        match = _RESETS_IN_PATTERN.search(tail)
        if match:
            minutes = int(match.group(1))
            cooldown = timedelta(minutes=minutes)
        else:
            cooldown = timedelta(minutes=_DEFAULT_RATE_LIMIT_COOLDOWN_MINUTES)
        throttled_until = datetime.now(UTC) + cooldown
        return "rate_limited", throttled_until.replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        )

    return None, None


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
    recovery: dict[str, Any] | None = None,
) -> ClaudeWorkerRecord:
    """Create an isolated worktree and launch a headless Claude Code worker in it.

    Never raises: worktree-creation failures and process-launch (``OSError``)
    failures both come back as an error record. If the worktree was created
    but the process failed to launch, the worktree is removed best-effort so
    a failed launch doesn't leak a half-made worktree.

    If ``rework`` is True, the worktree is created in rework mode (reuse existing
    worktree or attach to existing branch instead of creating a new branch).

    If ``recovery`` is provided (a dict with state file dispatch record), this is
    a dead-worker recovery re-dispatch. The worktree layer will inspect the
    leftover worktree/branch and either clean it (no commits) or reuse it (has
    commits/dirty work).
    """
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = _log_path(sessions_dir, issue_number, rework=rework)

    try:
        worktree: WorktreeInfo = create_worktree(
            repo_root,
            branch,
            worktrees_dir=worktrees_dir,
            venv_source=venv_source,
            rework=rework,
            recovery=recovery,
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
        remove_worktree(repo_root, worktree.path, force=True, branch=None if rework else branch)
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

    try:
        command = _render_command(
            command_template, prompt_path, issue_number=issue_number, branch=branch
        )
    except (KeyError, IndexError, ValueError) as exc:
        remove_worktree(repo_root, worktree.path, force=True, branch=None if rework else branch)
        record = _error_record(
            issue_number=issue_number,
            branch=branch,
            worktree_path=str(worktree.path),
            prompt_path=str(prompt_path),
            command=command_template,
            log_path=str(log_path),
            error=f"command template rendering failed: {exc}",
        )
        return _write_record(sessions_dir, record)

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
        remove_worktree(repo_root, worktree.path, force=True, branch=None if rework else branch)
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
        reclaimed=worktree.reclaimed,
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
                    failure_kind=data.get("failure_kind"),
                    reclaimed=data.get("reclaimed"),
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


def _win_is_alive(pid: int) -> bool:
    """Windows PID liveness via ``OpenProcess`` + ``GetExitCodeProcess``.

    ``os.kill(pid, 0)`` is NOT usable for this on Windows: CPython's Windows
    implementation of signal 0 does not probe the process at all (there is no
    real "probe" signal in the Win32 API) — it was empirically verified
    (see ``test_is_session_alive_reflects_real_process``) to keep reporting a
    long-dead, already-``wait()``-ed PID as alive indefinitely. Instead, open
    a limited-info handle and ask the kernel for the actual exit code:
    ``STILL_ACTIVE`` (259) means running, anything else (or a failed
    ``OpenProcess``, e.g. the PID never existed or was already reused) means
    not running.
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(_WIN_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == _WIN_STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _posix_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def is_worker_alive(record: ClaudeWorkerRecord) -> bool:
    """Check whether the process behind ``record`` is still running.

    Dispatches to a platform-appropriate liveness probe: ``OpenProcess`` +
    ``GetExitCodeProcess`` on Windows, ``os.kill(pid, 0)`` on POSIX (where it
    is the standard, reliable idiom). This avoids a hard `psutil` dependency
    and the slow `tasklist` subprocess round trip.
    """
    if record.pid is None or record.pid <= 0:
        return False
    if sys.platform == "win32":
        return _win_is_alive(record.pid)
    return _posix_is_alive(record.pid)


def update_worker_record_with_failure_classification(
    sessions_dir: Path, issue_number: int
) -> tuple[str | None, str | None]:
    """Update a worker record with failure classification after the session exits.

    This reads the existing sidecar, classifies the failure from the log tail,
    and writes back an updated record with failure_kind set.

    Returns a tuple of (failure_kind, throttled_until_iso) for the caller to
    update runtime state if needed.
    """
    sidecar_path = _sidecar_path(sessions_dir, issue_number)
    if not sidecar_path.exists():
        return None, None

    try:
        with sidecar_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None, None

    if not isinstance(payload, dict):
        return None, None

    # Skip if already classified
    if payload.get("failure_kind") is not None:
        return payload.get("failure_kind"), None

    log_path_str = payload.get("log_path")
    if not log_path_str:
        return None, None

    log_path = Path(log_path_str)
    failure_kind, throttled_until = _classify_session_failure(log_path)

    if failure_kind:
        payload["failure_kind"] = failure_kind
        _write_json_atomic(sidecar_path, payload)

    return failure_kind, throttled_until


__all__ = [
    "PROMPT_FILENAME",
    "ClaudeWorkerRecord",
    "launch_claude_worker",
    "read_worker_records",
    "probe_claude",
    "is_worker_alive",
    "update_worker_record_with_failure_classification",
]
