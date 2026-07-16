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
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from charlie_work.process_utils import is_pid_alive, parse_proc_stat_starttime
from .config import CLAUDE_CODE_PROMPT_FILENAME, OrchestratorConfig
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
    create_review_checkout,
    create_worktree,
    remove_review_checkout,
    remove_worktree,
)

PROMPT_FILENAME = CLAUDE_CODE_PROMPT_FILENAME

# Provider throttle signatures — matched against session log tails to classify
# failure kinds. The defaults are sourced from RuntimeConfig so there is a single
# default list; callers can override via config for new provider phrasings.
# Matching itself (substring + "resets in N minutes" extraction) is unified in
# throttle_signatures.match_throttle_tail, shared with the devin_shell sibling
# adapter (PR #262 review findings F1/F5).
_DEFAULT_THROTTLE_ERROR_MARKERS = OrchestratorConfig().runtime.throttle_error_markers
# Pattern for quota-exhaustion errors (e.g., "daily usage quota has been exhausted")
_QUOTA_EXHAUSTED_PATTERN = re.compile(
    r"daily usage quota has been exhausted|quota exceeded|usage limit",
    re.IGNORECASE,
)

# Default cooldown durations when we can't parse a specific reset time
_DEFAULT_RATE_LIMIT_COOLDOWN_MINUTES = 15
_DEFAULT_QUOTA_COOLDOWN_HOURS = 24

# Windows-only flag: isolates the worker's process group so a Ctrl+C to the
# orchestrator doesn't propagate into an in-flight `claude` session. Absent
# on non-Windows platforms, where Popen simply ignores creationflags=0.
_CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

_WIN_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

# Default command templates. Workers get write access (acceptEdits) since
# they are expected to commit/push changes. Reviewers get a read-only plan
# mode by default — a reviewer's judgment stays in its verdict, never in a
# commit to the PR's real branch (see create_review_checkout's isolation
# guarantee, which this pairs with: no branch checkout AND no write mode).
_WORKER_COMMAND_TEMPLATE: tuple[str, ...] = ("claude", "-p", "--permission-mode", "acceptEdits")
_REVIEW_COMMAND_TEMPLATE: tuple[str, ...] = ("claude", "-p", "--permission-mode", "plan")


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

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["command"] = list(self.command)
        return payload

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> ClaudeWorkerRecord:
        command = payload.get("command") or []
        return ClaudeWorkerRecord(
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
    return sessions_dir / f"issue-{issue_number}.claude.json"


def _log_path(
    sessions_dir: Path, issue_number: int, *, rework: bool = False, review: bool = False
) -> Path:
    if review:
        suffix = "-review.claude.log"
    elif rework:
        suffix = "-rework.claude.log"
    else:
        suffix = ".claude.log"
    return sessions_dir / f"issue-{issue_number}{suffix}"


def _events_path(
    sessions_dir: Path, issue_number: int, *, rework: bool = False, review: bool = False
) -> Path:
    """Path to the structured events.jsonl file for Claude Code stream-json output.

    This file is created only when tee_stream_json is enabled. It contains
    structured JSONL events from Claude Code's --output-format stream-json mode,
    enabling downstream parsing of tool_call_count, turn_count, tokens, and cost_usd.
    """
    if review:
        suffix = "-review.events.jsonl"
    elif rework:
        suffix = "-rework.events.jsonl"
    else:
        suffix = ".events.jsonl"
    return sessions_dir / f"issue-{issue_number}{suffix}"


@dataclass(frozen=True)
class ClaudeProgress:
    """Progress metrics parsed from Claude Code's stream-json events.

    This dataclass holds cumulative counts and usage metrics extracted from
    the events.jsonl file produced when tee_stream_json is enabled.
    """

    tool_call_count: int = 0
    turn_count: int = 0
    tokens: int | None = None
    cost_usd: float | None = None


def parse_claude_events(events_path: Path) -> ClaudeProgress | None:
    """Parse Claude Code's stream-json events file and extract progress metrics.

    Reads the events.jsonl file line-by-line, accumulating tool_call_count and
    turn_count, and taking the last-seen cumulative usage fields (tokens, cost_usd).

    Tolerates partial/incomplete final lines (file is being appended to live).
    Malformed/unparseable lines are skipped, not raised.

    Returns None if the file doesn't exist (devin workers, or claude workers
    launched without the tee_stream_json flag) — absence is a valid, non-error state.

    Args:
        events_path: Path to the events.jsonl file

    Returns:
        ClaudeProgress with accumulated metrics, or None if file doesn't exist
    """
    if not events_path.exists():
        return None

    tool_call_count = 0
    turn_count = 0
    tokens = None
    cost_usd = None

    try:
        with events_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue

                if not isinstance(event, dict):
                    continue

                # Count tool calls
                if event.get("type") == "tool_call":
                    tool_call_count += 1

                # Count turns (user/assistant exchanges)
                if event.get("type") in ("user_message", "assistant_message"):
                    turn_count += 1

                # Extract cumulative usage fields (take last-seen value)
                if "tokens" in event and isinstance(event["tokens"], int):
                    tokens = event["tokens"]
                if "cost_usd" in event and isinstance(event["cost_usd"], (int, float)):
                    cost_usd = float(event["cost_usd"])

    except OSError:
        # File read error - treat as no progress data
        return None

    # If we parsed no valid events, return None
    if tool_call_count == 0 and turn_count == 0 and tokens is None and cost_usd is None:
        return None

    return ClaudeProgress(
        tool_call_count=tool_call_count,
        turn_count=turn_count,
        tokens=tokens,
        cost_usd=cost_usd,
    )


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
    # with devin_shell._classify_session_failure / get_rate_limit_defer_until.
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
    failure_kind: str | None = None,
    pid: int | None = None,
    process_start_time: float | None = None,
) -> ClaudeWorkerRecord:
    return ClaudeWorkerRecord(
        issue_number=issue_number,
        branch=branch,
        worktree_path=worktree_path,
        prompt_path=prompt_path,
        command=command,
        pid=pid,
        started_at=utc_now(),
        log_path=log_path,
        error=error,
        failure_kind=failure_kind,
        process_start_time=process_start_time,
    )


def _sanitize_review_command_template(
    command_template: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Hard-pin the read-only reviewer posture onto ``command_template``.

    A review launch must always carry ``--permission-mode plan`` — this is
    an invariant, not a default a caller-supplied (or config-forwarded)
    ``command_template`` can defeat. ``ClaudeCodeConfig.command`` is a single
    field shared with worker dispatch (workers want ``acceptEdits``); if an
    operator's worker-tuning override were honored verbatim for reviewers
    too, it would silently grant write access to reviewer sessions,
    defeating ``create_review_checkout``'s paired guarantee (no branch
    checkout AND no write mode). See PR #397 round-2 review.

    Every occurrence of ``--permission-mode`` is stripped from the template
    first — the bare flag plus its following value token (if any), and any
    token of the form ``--permission-mode=<value>`` — and a single
    authoritative ``--permission-mode plan`` is then appended as the final
    tokens. Removing every occurrence before appending the forced value
    (rather than patching the first match in place) is the invariant: CLI
    argument parsers generally apply last-flag-wins semantics, so a template
    with duplicate ``--permission-mode`` flags (e.g. a caller-supplied
    override appended after a legitimate one) would otherwise let a later,
    unsanitized occurrence silently win at runtime even though the first
    occurrence was "fixed". See PR #397 round-3 review. This only touches
    the permission-mode flag — it does not discard the rest of the
    template — so a caller with a legitimate reason to vary the executable
    or other flags (e.g. a test double standing in for the ``claude``
    binary) is not blocked, only write access is. A ``None`` template
    resolves to the standard read-only default.
    """
    if command_template is None:
        return _REVIEW_COMMAND_TEMPLATE
    filtered: list[str] = []
    skip_next = False
    for token in command_template:
        if skip_next:
            skip_next = False
            continue
        if token == "--permission-mode":
            skip_next = True
            continue
        if token.startswith("--permission-mode="):
            continue
        filtered.append(token)
    return tuple(filtered) + ("--permission-mode", "plan")


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
    command_template: tuple[str, ...] | None = None,
    env: dict[str, str] | None = None,
    materialize_dirs: tuple[str, ...] = (),
    rework: bool = False,
    review: bool = False,
    head_sha: str | None = None,
    recovery: dict[str, Any] | None = None,
    base_ref: str = "",
    tee_stream_json: bool = False,
    config: OrchestratorConfig | None = None,
) -> ClaudeWorkerRecord:
    """Create an isolated worktree/checkout and launch a headless Claude Code
    worker (or reviewer) in it.

    Never raises: worktree-creation failures and process-launch (``OSError``)
    failures both come back as an error record. If the worktree was created
    but the process failed to launch, the worktree is removed best-effort so
    a failed launch doesn't leak a half-made worktree.

    If ``rework`` is True, the worktree is created in rework mode (reuse existing
    worktree or attach to existing branch instead of creating a new branch).

    If ``review`` is True, this launches a reviewer session in its OWN
    isolated, detached-HEAD checkout (``worktree.create_review_checkout``),
    keyed by PR number under ``sessions_dir`` — never the worker's branch-slug
    worktree under ``worktrees_dir``. ``head_sha`` is required in this mode
    (the PR's live head SHA the reviewer must check out); a missing value
    raises ``ValueError``, which — like every other failure in this
    function — comes back as an error record rather than propagating.
    In review mode, ``command_template`` always ends up carrying
    ``--permission-mode plan`` (read-only) — see
    ``_sanitize_review_command_template``. This is a hard-pinned invariant,
    not a default: any ``--permission-mode`` value the caller supplies
    (including one forwarded from ``ClaudeCodeConfig.command``, a field
    shared with worker dispatch) is overridden to ``"plan"`` before launch,
    so no config combination can grant a reviewer write access. This closes
    the gap where an operator customizing worker behavior (e.g. uncommenting
    the example config's ``acceptEdits`` override) would otherwise silently
    grant write access to reviewer sessions too, defeating
    ``create_review_checkout``'s paired guarantee (no branch checkout AND
    no write mode). See PR #397 round-2 review. Logs/sidecars get a distinct
    ``-review`` suffix so reviewer processes don't mix with worker processes.

    If ``recovery`` is provided (a dict with state file dispatch record), this is
    a dead-worker recovery re-dispatch. The worktree layer will inspect the
    leftover worktree/branch and either clean it (no commits) or reuse it (has
    commits/dirty work). Not applicable to ``review`` mode.
    """
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = _log_path(sessions_dir, issue_number, rework=rework, review=review)
    if review:
        # Hard-pinned, not a default: see _sanitize_review_command_template
        # and PR #397 round-2 review. No caller-supplied command_template
        # (including one forwarded from ClaudeCodeConfig.command) can result
        # in a reviewer session running with write access.
        command_template = _sanitize_review_command_template(command_template)
    elif command_template is None:
        command_template = _WORKER_COMMAND_TEMPLATE

    try:
        if review:
            if not head_sha:
                raise ValueError(
                    f"launch_claude_worker(review=True) requires head_sha for PR #{issue_number}"
                )
            worktree: WorktreeInfo = create_review_checkout(
                repo_root,
                issue_number,
                head_sha,
                reviews_dir=sessions_dir,
            )
        else:
            worktree = create_worktree(
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
        record = _error_record(
            issue_number=issue_number,
            branch=branch,
            worktree_path="",
            prompt_path="",
            command=command_template,
            log_path=str(log_path),
            error=str(exc)
            if isinstance(exc, LiveWorkerRedispatchError)
            else f"worktree creation failed: {exc}",
            failure_kind=failure_kind,
            pid=exc.pid if isinstance(exc, LiveWorkerRedispatchError) else None,
            process_start_time=exc.process_start_time
            if isinstance(exc, LiveWorkerRedispatchError)
            else None,
        )
        return _write_record(sessions_dir, record)

    # A redispatch may have just preserved the prior attempt's branch tip
    # (issue #261) — fold that into whatever post-mortem sidecar already
    # exists for this issue so the ref is discoverable alongside the block
    # diagnosis it belongs to. Best-effort: never blocks or fails dispatch.
    if worktree.attempt_snapshot is not None and worktree.attempt_snapshot.ref_name is not None:
        merge_attempt_snapshot(sessions_dir, issue_number, worktree.attempt_snapshot)

    def _teardown_on_launch_failure() -> None:
        # Review checkouts live in their own PR-keyed dir, never worktrees_dir,
        # so they must be torn down via remove_review_checkout — passing them
        # to remove_worktree would look for the wrong registered worktree path.
        if review:
            remove_review_checkout(repo_root, issue_number, reviews_dir=sessions_dir)
        else:
            remove_worktree(
                repo_root, worktree.path, force=True, branch=None if rework else branch
            )

    prompt_path = worktree.path / PROMPT_FILENAME
    try:
        prompt_path.write_text(prompt_text, encoding="utf-8")
    except OSError as exc:
        _teardown_on_launch_failure()
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
        _teardown_on_launch_failure()
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

    # If tee_stream_json is enabled, extend the command with --output-format stream-json
    # and set up a tee to write to both plaintext log and events file
    events_path = None
    if tee_stream_json:
        command = command + ("--output-format", "stream-json")
        events_path = _events_path(sessions_dir, issue_number, rework=rework, review=review)

    feed_stdin = "{prompt_path}" not in "".join(command_template)
    # Workers inherit the orchestrator's environment, with config-provided
    # overrides merged on top — e.g. PYTEST_XDIST_AUTO_NUM_WORKERS to bound a
    # worker's local `pytest -n auto` so a fleet of them doesn't oversubscribe
    # the shared host (see docs/RUNBOOK.md "Local host saturation ceiling
    # (claude-code adapter)"). `env` is a validated mapping (see config.py).
    # Sanitize the base environment to prevent VIRTUAL_ENV/UV_PROJECT_ENVIRONMENT
    # leaks from the orchestrator, then merge user-provided overrides on top.
    worker_env = {
        **sanitize_env(worktree.path),
        **{str(k): str(v) for k, v in (env or {}).items()},
    }

    try:
        # If tee_stream_json is enabled, we need to tee stdout to both log and events file
        # Use subprocess.PIPE and a thread to write to both files
        if tee_stream_json and events_path:
            # Open handles without 'with' blocks - the thread will manage their lifecycle
            # This is necessary because the thread runs as a daemon and must keep handles
            # open for the entire worker lifetime, not just during launch
            log_handle = log_path.open("w", encoding="utf-8", errors="replace")
            events_handle = events_path.open("w", encoding="utf-8", errors="replace")

            try:
                if feed_stdin:
                    prompt_handle = prompt_path.open("r", encoding="utf-8")
                    try:
                        process = subprocess.Popen(
                            command,
                            cwd=str(worktree.path),
                            stdin=prompt_handle,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            env=worker_env,
                            **no_console_window_kwargs(_CREATE_NEW_PROCESS_GROUP),
                            start_new_session=(os.name != "nt"),  # POSIX: detach into own session
                            text=True,  # Ensure text mode for line-by-line processing
                        )
                    finally:
                        prompt_handle.close()
                else:
                    process = subprocess.Popen(
                        command,
                        cwd=str(worktree.path),
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        env=worker_env,
                        **no_console_window_kwargs(_CREATE_NEW_PROCESS_GROUP),
                        start_new_session=(os.name != "nt"),  # POSIX: detach into own session
                        text=True,  # Ensure text mode for line-by-line processing
                    )
            except OSError:
                # Popen failed before the tee thread could start — nobody else
                # will ever close these, so close them here. The outer
                # `except OSError` below still handles worktree cleanup and
                # the error record.
                log_handle.close()
                events_handle.close()
                raise

            # Start a thread to tee output to both files
            def _tee_output():
                try:
                    for line in process.stdout:
                        log_handle.write(line)
                        log_handle.flush()
                        events_handle.write(line)
                        events_handle.flush()
                except Exception:
                    # Thread dies if process terminates or pipe breaks
                    pass
                finally:
                    # Close handles when thread exits
                    try:
                        log_handle.close()
                    except Exception:
                        pass
                    try:
                        events_handle.close()
                    except Exception:
                        pass

            import threading

            tee_thread = threading.Thread(target=_tee_output, daemon=True)
            tee_thread.start()
        else:
            # Original behavior: direct stdout to log file
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
                            **no_console_window_kwargs(_CREATE_NEW_PROCESS_GROUP),
                            start_new_session=(os.name != "nt"),  # POSIX: detach into own session
                        )
                else:
                    process = subprocess.Popen(
                        command,
                        cwd=str(worktree.path),
                        stdin=subprocess.DEVNULL,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        env=worker_env,
                        **no_console_window_kwargs(_CREATE_NEW_PROCESS_GROUP),
                        start_new_session=(os.name != "nt"),  # POSIX: detach into own session
                    )
    except OSError as exc:
        _teardown_on_launch_failure()
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

    # Capture process creation time immediately after spawn to verify identity later
    process_start_time = _get_process_start_time(process.pid)

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
        process_start_time=process_start_time,
        reclaimed=worktree.reclaimed,
        attempt_ref=worktree.attempt_snapshot.ref_name if worktree.attempt_snapshot else None,
        attempt_ahead_of_main=(
            worktree.attempt_snapshot.ahead_of_main_count if worktree.attempt_snapshot else None
        ),
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
            records.append(ClaudeWorkerRecord.from_dict(data))
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


def is_worker_alive(record: ClaudeWorkerRecord) -> bool:
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


def update_worker_record_with_failure_classification(
    sessions_dir: Path,
    issue_number: int,
    *,
    fallback_kind: str | None = None,
    config: OrchestratorConfig | None = None,
) -> tuple[str | None, str | None]:
    """Update a worker record with failure classification after the session exits.

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
    _write_json_atomic(sidecar_path, payload)
    return resolved_kind, throttled_until


__all__ = [
    "PROMPT_FILENAME",
    "ClaudeWorkerRecord",
    "launch_claude_worker",
    "read_worker_records",
    "probe_claude",
    "is_worker_alive",
    "update_worker_record_with_failure_classification",
    "_sidecar_path",
    "ClaudeProgress",
    "parse_claude_events",
    "_events_path",
]
