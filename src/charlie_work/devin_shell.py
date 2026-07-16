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
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from charlie_work.process_utils import is_pid_alive, parse_proc_stat_starttime
from .config import OrchestratorConfig
from .env_sanitize import sanitize_env
from .post_mortem import merge_attempt_snapshot
from .state import _canonical_started_at, utc_now
from .subprocess_runner import RunResult, no_console_window_kwargs, run_captured
from .throttle_signatures import match_throttle_tail
from .worktree import (
    LiveWorkerRedispatchError,
    WorktreeInfo,
    WorktreeProbeFailedError,
    WorktreeUnsafeError,
    create_worktree,
    remove_worktree,
)

_WIN_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

# Provider throttle signatures — matched against session log tails to classify
# failure kinds. The defaults are sourced from RuntimeConfig so there is a single
# default list; callers can override via config for new provider phrasings.
# Matching itself (substring + "resets in N minutes" extraction) is unified in
# throttle_signatures.match_throttle_tail — used here and by
# get_rate_limit_defer_until below (PR #262 review findings F1/F5).
_DEFAULT_THROTTLE_ERROR_MARKERS = OrchestratorConfig().runtime.throttle_error_markers
# Pattern for quota-exhaustion errors (e.g., "daily usage quota has been exhausted")
_QUOTA_EXHAUSTED_PATTERN = re.compile(
    r"daily usage quota has been exhausted|quota exceeded|usage limit",
    re.IGNORECASE,
)

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
    process_start_time: float | None = None  # Unix timestamp in seconds (process creation time)
    reclaimed: str | None = None  # "fetch-fallback" | "pruned" | "salvaged" | None
    last_activity_at: str | None = None  # ISO timestamp from log_path.stat().st_mtime
    log_bytes: int | None = None  # log_path.stat().st_size
    attempt_ref: str | None = None  # refs/charlie/attempts/issue-<n>/attempt-<k> (issue #261)
    attempt_ahead_of_main: int | None = None  # commit count ahead of base_ref at snapshot time
    rate_limit_defer_until: str | None = (
        None  # ISO timestamp when the stall kill is deferred (issue #247)
    )
    inconclusive_probe_deferred_count: int = 0  # Signal-1 deferral counter (issue #338)

    def __post_init__(self) -> None:
        """Enforce a canonical ISO-8601 UTC ``started_at`` at construction time."""
        canonical = _canonical_started_at(self.started_at, self.process_start_time)
        if canonical != self.started_at:
            object.__setattr__(self, "started_at", canonical)

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
            process_start_time=payload.get("process_start_time"),
            reclaimed=payload.get("reclaimed"),
            last_activity_at=payload.get("last_activity_at"),
            log_bytes=payload.get("log_bytes"),
            attempt_ref=payload.get("attempt_ref"),
            attempt_ahead_of_main=payload.get("attempt_ahead_of_main"),
            rate_limit_defer_until=payload.get("rate_limit_defer_until"),
            inconclusive_probe_deferred_count=int(
                payload.get("inconclusive_probe_deferred_count") or 0
            ),
        )


def _sidecar_path(sessions_dir: Path, issue_number: int) -> Path:
    return sessions_dir / f"issue-{issue_number}.json"


def _log_path(sessions_dir: Path, issue_number: int, *, rework: bool = False) -> Path:
    suffix = "-rework.log" if rework else ".log"
    return sessions_dir / f"issue-{issue_number}{suffix}"


def _classify_session_failure(
    log_path: Path,
    throttle_error_markers: Sequence[str] | None = None,
) -> tuple[str | None, str | None]:
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

    # Check for rate limiting / provider throttling using configurable substrings.
    # Single point of enforcement (throttle_signatures.match_throttle_tail) shared
    # with get_rate_limit_defer_until below — see that function's docstring.
    markers = (
        throttle_error_markers
        if throttle_error_markers is not None
        else _DEFAULT_THROTTLE_ERROR_MARKERS
    )
    matched, reset_minutes = match_throttle_tail(tail, markers)
    if matched:
        cooldown = timedelta(
            minutes=reset_minutes
            if reset_minutes is not None
            else _DEFAULT_RATE_LIMIT_COOLDOWN_MINUTES
        )
        throttled_until = datetime.now(UTC) + cooldown
        return "rate_limited", throttled_until.replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        )

    return None, None


def get_rate_limit_defer_until(
    log_path: Path,
    slack_minutes: int,
    now: datetime | None = None,
    throttle_error_markers: Sequence[str] | None = None,
) -> str | None:
    """Return a defer-until ISO timestamp for a log tail containing a rate-limit signature.

    Reads the same 2KB tail as ``_classify_session_failure`` and matches
    against the same config-driven markers via ``throttle_signatures.
    match_throttle_tail`` (issue #247; unified with ``_classify_session_
    failure`` per PR #262 review findings F1/F5 — previously each function
    carried its own copy of this matching logic). If the tail matches and a
    ``"resets in N minutes"`` value is found, the defer deadline is
    ``now + N minutes + slack``. Otherwise the fallback
    ``_DEFAULT_RATE_LIMIT_COOLDOWN_MINUTES`` is used.

    ``throttle_error_markers`` defaults to ``RuntimeConfig``'s default list
    when not provided (backward compatible with pre-#260 callers).

    Returns None when the log is missing, unreadable, or does not contain a
    rate-limit signature. Quota exhaustion is intentionally not deferred here.
    """
    if now is None:
        now = datetime.now(UTC)

    if not log_path.exists():
        return None

    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    tail = log_text[-2048:] if len(log_text) > 2048 else log_text

    markers = (
        throttle_error_markers
        if throttle_error_markers is not None
        else _DEFAULT_THROTTLE_ERROR_MARKERS
    )
    matched, reset_minutes = match_throttle_tail(tail, markers)
    if not matched:
        return None

    minutes = reset_minutes if reset_minutes is not None else _DEFAULT_RATE_LIMIT_COOLDOWN_MINUTES

    defer_until = now + timedelta(minutes=minutes + slack_minutes)
    return defer_until.replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
    venv_source: Path | None = None,
    worker_env: dict[str, str] | None = None,
    materialize_dirs: tuple[str, ...] = (),
    rework: bool = False,
    recovery: dict[str, Any] | None = None,
    base_ref: str = "",
    config: OrchestratorConfig | None = None,
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

    If ``recovery`` is provided (a dict with state file dispatch record), this is
    a dead-worker recovery re-dispatch. The worktree layer will inspect the
    leftover worktree/branch and either clean it (no commits) or reuse it (has
    commits/dirty work).
    """
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = _log_path(sessions_dir, issue_number, rework=rework)

    # --- worktree creation ---------------------------------------------------
    try:
        worktree: WorktreeInfo = create_worktree(
            repo_root,
            branch,
            worktrees_dir=worktrees_dir,
            venv_source=venv_source,
            materialize_dirs=materialize_dirs,
            rework=rework,
            recovery=recovery,
            base_ref=base_ref,
            issue_number=issue_number,
            config=config,
        )
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as exc:
        if isinstance(exc, WorktreeProbeFailedError):
            # Transient probe contention (e.g. index lock), not a confirmed-dirty
            # worktree. Must stay off DETERMINISTIC_ESCALATION_FAILURE_KINDS so it
            # takes the ordinary redispatch-cap path (issue #288 follow-up, PR #314).
            failure_kind = "worktree_probe_failed"
        elif isinstance(exc, WorktreeUnsafeError):
            failure_kind = "worktree_unsafe"
        elif isinstance(exc, LiveWorkerRedispatchError):
            failure_kind = "live_worker_redispatch_averted"
        else:
            failure_kind = None
        record = SessionRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path="",
            prompt_path=str(prompt_path),
            command=command_template,
            pid=exc.pid if isinstance(exc, LiveWorkerRedispatchError) else None,
            started_at=utc_now(),
            log_path=str(log_path),
            error=str(exc)
            if isinstance(exc, LiveWorkerRedispatchError)
            else f"worktree creation failed: {exc}",
            failure_kind=failure_kind,
            process_start_time=exc.process_start_time
            if isinstance(exc, LiveWorkerRedispatchError)
            else None,
        )
        _write_json(_sidecar_path(sessions_dir, issue_number), record.to_dict())
        return record

    # A redispatch may have just preserved the prior attempt's branch tip
    # (issue #261) — fold that into whatever post-mortem sidecar already
    # exists for this issue so the ref is discoverable alongside the block
    # diagnosis it belongs to. Best-effort: never blocks or fails dispatch.
    if worktree.attempt_snapshot is not None and worktree.attempt_snapshot.ref_name is not None:
        merge_attempt_snapshot(sessions_dir, issue_number, worktree.attempt_snapshot)

    # --- command rendering (prompt_path is caller-supplied, lives outside wt) -
    try:
        command = _render_command(
            command_template,
            issue_number=issue_number,
            branch=branch,
            prompt_path=prompt_path,
            worker_model=worker_model,
        )
    except (KeyError, IndexError, ValueError) as exc:
        remove_worktree(repo_root, worktree.path, force=True, branch=None if rework else branch)
        record = SessionRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path=str(worktree.path),
            prompt_path=str(prompt_path),
            command=command_template,
            pid=None,
            started_at=utc_now(),
            log_path=str(log_path),
            error=f"command template rendering failed: {exc}",
        )
        _write_json(_sidecar_path(sessions_dir, issue_number), record.to_dict())
        return record

    kwargs: dict[str, Any] = {}
    kwargs.update(no_console_window_kwargs(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)))
    # POSIX: detach worker into its own session to prevent killpg from killing the orchestrator
    if os.name != "nt":
        kwargs["start_new_session"] = True

    # Sanitize environment to prevent VIRTUAL_ENV leaks from the orchestrator,
    # then merge user-provided worker_env overrides on top (e.g. PYTEST_XDIST_AUTO_NUM_WORKERS)
    worker_env_dict = {
        **sanitize_env(worktree.path),
        **{str(k): str(v) for k, v in (worker_env or {}).items()},
    }
    kwargs["env"] = worker_env_dict

    pid: int | None = None
    error: str | None = None
    process_start_time: float | None = None
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
        # Capture process creation time immediately after spawn to verify identity later
        process_start_time = _get_process_start_time(pid)
    except OSError as exc:
        remove_worktree(repo_root, worktree.path, force=True, branch=None if rework else branch)
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
        process_start_time=process_start_time,
        reclaimed=worktree.reclaimed,
        attempt_ref=worktree.attempt_snapshot.ref_name if worktree.attempt_snapshot else None,
        attempt_ahead_of_main=(
            worktree.attempt_snapshot.ahead_of_main_count if worktree.attempt_snapshot else None
        ),
    )
    _write_json(_sidecar_path(sessions_dir, issue_number), record.to_dict())
    return record


_DEVIN_SIDECAR_STEM_RE = re.compile(r"^issue-\d+$")


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
        # `issue-*.json` also matches every other adapter's/subsystem's
        # sidecar that happens to share the `issue-<n>.<extra>.json` naming
        # scheme in the same sessions_dir — e.g. the claude-code adapter's
        # `issue-N.claude.json` and post_mortem's `issue-N.post-mortem.json`.
        # A devin session sidecar's stem (path name minus the final `.json`)
        # is exactly `issue-<digits>`; anything with additional dotted
        # segments belongs to a different writer and must be skipped, or its
        # foreign schema gets misread as a bogus SessionRecord (pid=None)
        # that bypasses corroboration downstream (issue #343).
        if not _DEVIN_SIDECAR_STEM_RE.match(path.stem):
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


def _get_process_start_time(pid: int) -> float | None:
    """Get the process creation time as a Unix timestamp in seconds.

    Returns None if the process does not exist or the start time cannot be retrieved.
    This is used to verify that a PID has not been recycled by the OS.

    On Windows: Uses GetProcessTimes via ctypes to retrieve process creation time.
    On POSIX: Reads /proc/<pid>/stat field 22 (starttime in clock ticks).
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(_WIN_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return None
        try:
            creation_time = wintypes.FILETIME()
            exit_time = wintypes.FILETIME()
            kernel_time = wintypes.FILETIME()
            user_time = wintypes.FILETIME()
            if not kernel32.GetProcessTimes(
                handle,
                ctypes.byref(creation_time),
                ctypes.byref(exit_time),
                ctypes.byref(kernel_time),
                ctypes.byref(user_time),
            ):
                return None
            # Convert FILETIME to Unix timestamp
            # FILETIME is 100-nanosecond intervals since 1601-01-01
            # Unix timestamp is seconds since 1970-01-01
            # Difference between 1601-01-01 and 1970-01-01 is 11644473600 seconds
            filetime = (creation_time.dwHighDateTime << 32) | creation_time.dwLowDateTime
            unix_time = filetime / 10_000_000 - 11644473600
            return unix_time
        finally:
            kernel32.CloseHandle(handle)
    else:
        # POSIX: read /proc/<pid>/stat
        try:
            with open(f"/proc/{pid}/stat", "r") as f:
                stat = f.read()
            starttime_ticks = parse_proc_stat_starttime(stat)
            if starttime_ticks is None:
                return None
            # Convert to seconds: need system clock tick frequency
            tick_hz = os.sysconf("SC_CLK_TCK")
            if tick_hz <= 0:
                tick_hz = 100  # Default fallback
            # Get system uptime to convert to absolute time
            try:
                with open("/proc/uptime", "r") as f:
                    uptime_seconds = float(f.read().split()[0])
            except (OSError, ValueError, IndexError):
                return None
            # Process start time = current time - uptime + process starttime
            boot_time = time.time() - uptime_seconds
            return boot_time + (starttime_ticks / tick_hz)
        except (OSError, ValueError, IndexError):
            return None


def is_session_alive(record: SessionRecord) -> bool:
    """Check whether the process behind ``record`` is still running.

    Delegates to ``charlie_work.process_utils.is_pid_alive`` so liveness
    semantics are enforced in a single place.  This avoids a hard `psutil`
    dependency and the slow `tasklist` subprocess round trip.

    Process identity is verified by checking that the current process start time
    matches the recorded start time (captured at spawn).  A start-time probe
    that returns ``None`` is treated as indeterminate and returns ``True`` rather
    than reaping a potentially-live worker on a transient probe failure
    (issue #360 criterion #1 / issue #343).

    Legacy records without ``process_start_time`` fall back to pid-only liveness
    (vulnerable to recycling but preserves backward compatibility).
    """
    if record.pid is None or record.pid <= 0:
        return False
    return is_pid_alive(record.pid, record.process_start_time)


def update_session_record_with_failure_classification(
    sessions_dir: Path,
    issue_number: int,
    *,
    fallback_kind: str | None = None,
    config: OrchestratorConfig | None = None,
) -> tuple[str | None, str | None]:
    """Update a session record with failure classification after the session exits.

    This reads the existing sidecar, classifies the failure from the log tail,
    and writes back an updated record with failure_kind set.

    Log-tail classification (``_classify_session_failure``) always runs first.
    If it detects a provider throttle signature (``rate_limited`` /
    ``quota_exhausted``), that classification wins — including its computed
    ``throttled_until`` cooldown — regardless of ``fallback_kind``. Only when
    the log shows no throttle signature does ``fallback_kind`` apply (e.g. the
    stall watchdog's "stalled" default, or the launch-stall watchdog's
    "launch_stalled" default). This ordering matters: a worker that dies
    because it hit a provider rate limit must be classified as such even when
    the caller only knows "this looked stalled" — otherwise ``throttled_until``
    never gets set and dispatch keeps relaunching workers into the same limit.

    ``config`` is optional for backward compatibility; when provided, its
    ``runtime.throttle_error_markers`` are used instead of the defaults.

    Returns a tuple of (failure_kind, throttled_until_iso) for the caller to
    update runtime state if needed. ``throttled_until_iso`` is only non-None
    when log-tail classification actually matched a throttle signature.
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

    classified_kind: str | None = None
    throttled_until: str | None = None
    log_path_str = payload.get("log_path")
    if log_path_str:
        throttle_markers = config.runtime.throttle_error_markers if config is not None else None
        classified_kind, throttled_until = _classify_session_failure(
            Path(log_path_str), throttle_markers
        )

    resolved_kind = classified_kind or fallback_kind
    if resolved_kind is None:
        return None, None

    payload["failure_kind"] = resolved_kind
    _write_json(sidecar_path, payload)
    return resolved_kind, throttled_until


__all__ = [
    "DEFAULT_COMMAND_TEMPLATE",
    "SessionRecord",
    "launch_devin_session",
    "read_session_records",
    "probe_devin",
    "is_session_alive",
    "update_session_record_with_failure_classification",
    "get_rate_limit_defer_until",
    "_get_process_start_time",
    "_sidecar_path",
]
