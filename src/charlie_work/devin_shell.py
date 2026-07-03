"""Headless Devin CLI dispatch — non-blocking session launch with a durable
sidecar so the orchestrator and ``doctor`` can see what is in flight.

There is no Devin session-creation API (see docs/design/extraction-dossier.md,
"headless"/"--prompt-file"). Production reality is spawning the ``devin`` CLI
in print mode: ``devin --prompt-file <path> --print --permission-mode
dangerous``. Sessions run for many minutes, so dispatch must return immediately
after ``Popen`` — callers must never block on the worker finishing. Each launch
writes a JSON sidecar file (``sessions_dir/issue-<n>.json``) atomically (tmp +
replace, matching ``adapters._write_json``) *before* returning, so a crash of
the orchestrator process itself never loses track of a session that was actually
spawned.

Each worker is launched in an isolated per-issue git worktree (created via
``worktree.create_worktree()``, mirroring the claude-code adapter) so
concurrent sessions do not contend over the shared checkout.
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

_WIN_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_WIN_STILL_ACTIVE = 259

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

# ``--permission-mode dangerous`` is required for headless workers: without it
# the Devin CLI defaults to ``auto`` (read-only tools), stalls on any
# git/uv/gh call, and exits asking the operator to restart with this flag.
# {model_args} is a placeholder for config-driven model selection (e.g.
# "--model claude-sonnet-4-5"). When devin.worker_model is empty, this renders
# to an empty string, preserving CLI default behavior.
DEFAULT_COMMAND_TEMPLATE: tuple[str, ...] = (
    "devin",
    "{model_args}",
    "--prompt-file",
    "{prompt_path}",
    "--print",
    "--permission-mode",
    "dangerous",
)


@dataclass(frozen=True)
class SessionRecord:
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

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["command"] = list(self.command)
        return payload

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> SessionRecord:
        command = payload.get("command") or []
        return SessionRecord(
            issue_number=int(payload["issue_number"]),
            branch=str(payload.get("branch", "")),
            worktree_path=str(payload.get("worktree_path", "")),
            prompt_path=str(payload.get("prompt_path", "")),
            command=tuple(str(part) for part in command),
            pid=int(payload["pid"]) if payload.get("pid") is not None else None,
            started_at=str(payload.get("started_at", "")),
            log_path=str(payload.get("log_path", "")),
            error=payload.get("error"),
            failure_kind=payload.get("failure_kind"),
        )


def _sidecar_path(sessions_dir: Path, issue_number: int) -> Path:
    return sessions_dir / f"issue-{issue_number}.json"


def _log_path(sessions_dir: Path, issue_number: int, *, rework: bool = False) -> Path:
    suffix = "-rework.log" if rework else ".log"
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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def _render_command(
    command_template: tuple[str, ...],
    *,
    issue_number: int,
    branch: str,
    prompt_path: Path,
    worker_model: str = "",
) -> tuple[str, ...]:
    model_args = f"--model {worker_model}" if worker_model else ""
    values = {
        "prompt_path": str(prompt_path),
        "issue_number": str(issue_number),
        "branch": branch,
        "model_args": model_args,
    }
    rendered = tuple(part.format(**values) for part in command_template)
    # Filter out empty-string placeholders to avoid spurious empty argv tokens.
    # Also split model_args into separate tokens if it contains --model.
    result: list[str] = []
    for part in rendered:
        if not part:
            continue
        if part.startswith("--model "):
            # Split "--model <value>" into two separate tokens
            result.extend(part.split())
        else:
            result.append(part)
    return tuple(result)


def launch_devin_session(
    issue_number: int,
    branch: str,
    prompt_path: Path,
    *,
    repo_root: Path,
    sessions_dir: Path,
    worktrees_dir: Path | None = None,
    command_template: tuple[str, ...] = DEFAULT_COMMAND_TEMPLATE,
    worker_model: str = "",
    rework: bool = False,
) -> SessionRecord:
    """Launch a headless Devin CLI session for one issue and return immediately.

    Creates an isolated per-issue git worktree (via ``worktree.create_worktree``)
    and launches the Devin CLI inside it, so concurrent workers do not contend
    over a shared checkout. Mirrors the claude-code adapter's worktree lifecycle:
    creation before launch; ``remove_worktree`` (junction-safe) on failure.

    Non-blocking: uses ``Popen`` (never waits for the process). stdout/stderr
    are redirected to a per-session log file since the worker can run for many
    minutes. The sidecar JSON is written atomically before this function
    returns, so any crash after that point still leaves a durable record for
    ``read_session_records``/``doctor`` to find. Never raises — worktree-
    creation failures, a missing ``devin`` binary, or any other ``OSError``
    comes back as a record with ``pid=None`` and ``error`` set.

    If ``rework`` is True, the worktree is created in rework mode (reuse existing
    worktree or attach to existing branch instead of creating a new branch).
    """
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = _log_path(sessions_dir, issue_number, rework=rework)

    # --- worktree creation ---------------------------------------------------
    try:
        worktree: WorktreeInfo = create_worktree(
            repo_root,
            branch,
            worktrees_dir=worktrees_dir,
            rework=rework,
        )
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as exc:
        record = SessionRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path="",
            prompt_path=str(prompt_path),
            command=command_template,
            pid=None,
            started_at=utc_now(),
            log_path=str(log_path),
            error=f"worktree creation failed: {exc}",
        )
        _write_json(_sidecar_path(sessions_dir, issue_number), record.to_dict())
        return record

    # --- command rendering (prompt_path is caller-supplied, lives outside wt) -
    command = _render_command(
        command_template,
        issue_number=issue_number,
        branch=branch,
        prompt_path=prompt_path,
        worker_model=worker_model,
    )

    kwargs: dict[str, Any] = {}
    if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    pid: int | None = None
    error: str | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                list(command),
                cwd=str(worktree.path),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                **kwargs,
            )
        pid = process.pid
    except OSError as exc:
        remove_worktree(repo_root, worktree.path, force=True)
        error = f"failed to launch devin: {exc}"

    record = SessionRecord(
        issue_number=issue_number,
        branch=branch,
        worktree_path=str(worktree.path),
        prompt_path=str(prompt_path),
        command=command,
        pid=pid,
        started_at=utc_now(),
        log_path=str(log_path),
        error=error,
    )
    _write_json(_sidecar_path(sessions_dir, issue_number), record.to_dict())
    return record


def read_session_records(sessions_dir: Path) -> list[SessionRecord]:
    """Read every sidecar JSON in ``sessions_dir`` back into ``SessionRecord``s.

    Unreadable or malformed sidecars are skipped rather than raising — a
    corrupt file must not take down doctor/status reporting for every other
    in-flight session.
    """
    if not sessions_dir.is_dir():
        return []
    records: list[SessionRecord] = []
    for path in sorted(sessions_dir.glob("issue-*.json")):
        # `issue-*.json` also matches the claude-code adapter's
        # `issue-N.claude.json` sidecars (both adapters share one sessions_dir).
        # Skip them so doctor doesn't read every Claude worker twice.
        if path.name.endswith(".claude.json"):
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        try:
            records.append(SessionRecord.from_dict(payload))
        except (KeyError, TypeError, ValueError):
            continue
    return records


def probe_devin(
    repo_root: Path, *, command: tuple[str, ...] = ("devin", "--version")
) -> RunResult:
    """Run a cheap Devin CLI probe (e.g. ``devin --version``) for
    ``doctor --adapter-probe``. Delegates to ``run_captured``, so a missing
    binary or non-zero exit comes back as a not-ok result, never an exception.
    """
    return run_captured(list(command), cwd=repo_root, timeout_seconds=30)


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


def is_session_alive(record: SessionRecord) -> bool:
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


def update_session_record_with_failure_classification(
    sessions_dir: Path, issue_number: int
) -> tuple[str | None, str | None]:
    """Update a session record with failure classification after the session exits.

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
        _write_json(sidecar_path, payload)

    return failure_kind, throttled_until


__all__ = [
    "DEFAULT_COMMAND_TEMPLATE",
    "SessionRecord",
    "launch_devin_session",
    "read_session_records",
    "probe_devin",
    "is_session_alive",
    "update_session_record_with_failure_classification",
]
