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

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from charlie_work.process_utils import (
    is_pid_alive,
    parse_proc_stat_starttime,
    popen_worker,
    start_terminal_status_watcher,
    worker_terminal_status_path,
)
from .config import (
    CLAUDE_CODE_PROMPT_FILENAME,
    ClaudeCodeConfig,
    OrchestratorConfig,
    ReviewDispatchConfig,
)
from .env_sanitize import resolve_pytest_cap, resolve_uv_no_sync, sanitize_env
from .post_mortem import merge_attempt_snapshot
from .state import _canonical_started_at, utc_now
from .subprocess_runner import RunResult, resolve_cli_binary, run_captured
from .throttle_signatures import match_throttle_tail
from .worktree import (
    LiveWorkerRedispatchError,
    ReworkBranchConflictError,
    WorktreeForeignWriterError,
    WorktreeInfo,
    WorktreeProbeFailedError,
    WorktreeUnsafeError,
    create_review_checkout,
    create_worktree,
    remove_review_checkout,
    apply_rework_conflict_notice,
    remove_worktree,
    write_worktree_marker,
)

PROMPT_FILENAME = CLAUDE_CODE_PROMPT_FILENAME

logger = logging.getLogger(__name__)

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

# Pattern for provider authentication failures (issue #484). Matched against the
# log tail of api-kind sessions only — a dead/invalid API key against a custom
# Anthropic-compatible endpoint surfaces as 401/403 or an explicit
# authentication/invalid-api-key error. Auth failures must NOT masquerade as
# generic throttles: they are classified as ``provider_auth`` and enter the
# existing ``throttled_until`` cooldown so routing preflight falls back until
# the key is fixed (a dead key will not self-heal in minutes).
#
# The bare HTTP status codes 401/403 are anchored with word boundaries (\b) so
# a coincidental numeric substring in an unrelated log tail (e.g. "error code
# 14013", "4034 files processed", "issue #4019") cannot trip a false-positive
# 24h cooldown. Every other pattern in this file (throttle markers, quota
# phrases) matches natural-language substrings; the bare codes are the only
# numeric tokens and would otherwise be the sole false-positive vector.
_PROVIDER_AUTH_PATTERN = re.compile(
    r"\b401\b|\b403\b|authentication(?:\s+failed)?|unauthorized|"
    r"invalid[-\s]?api[-\s]?key|invalid[-\s]?authentication|"
    r"permission_denied|auth(?:entication)?\s+error",
    re.IGNORECASE,
)

# Pattern for provider account suspension / insufficient-balance responses
# (issue #1342). Matched against the log tail of api-kind sessions only — a
# suspended provider account (e.g. Moonshot "suspended due to insufficient
# balance, please recharge your account") is a TERMINAL billing failure that
# will not self-heal in minutes, so it must NOT enter the rate-limit backoff
# loop. It is classified as ``provider_suspended`` with NO cooldown (terminal),
# and ``provider_suspended`` sits in
# ``config.DETERMINISTIC_ESCALATION_FAILURE_KINDS`` so the issue escalates to
# an operator on the first occurrence instead of burning the redispatch cap.
#
# Matched by response semantics (HTTP status + error code/message the provider
# documents), not a brittle full-string comparison: the canonical billing-
# suspension signals across Anthropic-compatible providers are
# "insufficient balance/funds/credit", "account (is) suspended" (with billing
# context), and "recharge your account". These are distinct from the
# provider-auth pattern (401/403/invalid-key — a credential problem) and the
# quota-exhaustion pattern ("usage limit" — a usage-ceiling problem).
_PROVIDER_SUSPENDED_PATTERN = re.compile(
    r"insufficient\s+(?:balance|funds|credit)"
    r"|account\s+(?:is\s+)?suspended"
    r"|suspended\s+due\s+to\s+(?:insufficient\s+balance|billing|payment|unpaid)"
    r"|recharge\s+your\s+account"
    r"|please\s+recharge",
    re.IGNORECASE,
)

# Default cooldown durations when we can't parse a specific reset time
_DEFAULT_RATE_LIMIT_COOLDOWN_MINUTES = 15
_DEFAULT_QUOTA_COOLDOWN_HOURS = 24

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
    failure_kind: str | None = (
        None  # "rate_limited" | "quota_exhausted" | "provider_auth" | "budget_exceeded" | ...
    )
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
    session_id: str | None = None  # unique session id for worktree writer marker (issue #400)
    adapter_kind: str = "claude-code"
    provider: str = ""
    xdist_cap: str | None = None  # resolved PYTEST_XDIST_AUTO_NUM_WORKERS at launch (issue #646)
    uv_no_sync: str | None = None  # resolved UV_NO_SYNC at launch, or None if no .venv (#646)

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
            session_id=payload.get("session_id"),
            adapter_kind=str(payload.get("adapter_kind", "claude-code")),
            provider=str(payload.get("provider", "")),
            xdist_cap=payload.get("xdist_cap"),
            uv_no_sync=payload.get("uv_no_sync"),
        )


_ADAPTER_SIDECAR_SUFFIXES: dict[str, str] = {
    "claude-code": "claude",
    "api": "api",
}


def _sidecar_suffix(adapter_kind: str) -> str:
    """Return the dotted sidecar suffix for ``adapter_kind``.

    This is the single mapping from adapter identity to filename suffix;
    every sidecar path derivation in this module routes through it.
    """
    try:
        return _ADAPTER_SIDECAR_SUFFIXES[adapter_kind]
    except KeyError as exc:
        raise ValueError(f"unknown adapter_kind {adapter_kind!r}") from exc


def _sidecar_path(
    sessions_dir: Path, issue_number: int, adapter_kind: str = "claude-code"
) -> Path:
    return sessions_dir / f"issue-{issue_number}.{_sidecar_suffix(adapter_kind)}.json"


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


def _rotate_old_log(log_path: Path) -> None:
    """Rename an existing log file to ``.1`` before a new launch overwrites it.

    Preserves the previous session's log for post-mortem analysis when a
    re-dispatch overwrites the same deterministic log path. Best-effort:
    OSError is swallowed (the old log may not exist on a first dispatch, or
    may be locked on Windows). Only one generation is kept (``.1``); an
    existing ``.1`` file is removed first.
    """
    if not log_path.exists():
        return
    rotated = log_path.with_suffix(log_path.suffix + ".1")
    try:
        if rotated.exists():
            rotated.unlink(missing_ok=True)
        log_path.rename(rotated)
    except OSError:
        pass


def _read_sidecar_inconclusive_count(
    sessions_dir: Path, issue_number: int, adapter_kind: str = "claude-code"
) -> int:
    """Read the existing claude sidecar's Signal-1 deferral counter, if any."""
    sidecar_path = _sidecar_path(sessions_dir, issue_number, adapter_kind)
    try:
        with sidecar_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(payload, dict):
        return 0
    raw = payload.get("inconclusive_probe_deferred_count")
    if raw is None:
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


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


def iter_stream_json_events(text: str) -> Iterator[dict[str, Any]]:
    """Yield parsed stream-json events from tee'd JSONL text.

    Claude Code's ``--output-format stream-json`` emits one JSON object per
    line. Both the plaintext ``.log`` and the ``.events.jsonl`` sidecar carry
    this format when ``tee_stream_json`` is enabled. Non-JSON lines (plain
    stderr noise, truncated tails) are skipped, not raised.
    """
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            yield event


def extract_event_text(event: dict[str, Any]) -> str:
    """Assistant-visible text carried by one stream-json event.

    Real stream-json shapes (``--output-format stream-json``):

    - ``{"type": "assistant", "message": {"content": [{"type": "text", ...}]}}``
      — concatenates ``text`` blocks. ``thinking`` blocks are deliberately
      excluded: they can contain draft verdicts the model later revised.
    - ``{"type": "result", "result": "<final output text>"}``

    Legacy/simplified shape (``{"type": "assistant_message", "content": ...}``
    with string, block-list, or dict content) is still honored so any adapter
    emitting it keeps working. Returns ``""`` for events carrying no text.
    """
    event_type = event.get("type")

    if event_type == "assistant":
        message = event.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return ""

    if event_type == "result":
        result = event.get("result")
        return result if isinstance(result, str) else ""

    if event_type == "assistant_message":
        content = event.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        if isinstance(content, dict):
            return str(content.get("text", ""))

    return ""


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


def iter_claude_events(events_path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON event dicts from a Claude Code stream-json events file.

    This is the single parsing primitive for ``events.jsonl``: it owns the
    tolerant line-by-line read (partial/trailing lines and malformed JSON are
    skipped, never raised; a missing file yields nothing). Higher-level
    consumers — ``parse_claude_events`` (progress metrics) and
    ``api_budget.usage_from_events`` (token usage for spend accounting) —
    iterate this generator so the JSONL parsing logic is implemented once and
    reused, never duplicated.

    Args:
        events_path: Path to the events.jsonl file.

    Yields:
        Each parsed JSON object that is a dict. Non-dict JSON values are
        skipped.
    """
    if not events_path.exists():
        return
    try:
        with events_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    # Skip malformed lines (including partial trailing lines
                    # written while the file is being appended to live).
                    continue
                if isinstance(event, dict):
                    yield event
    except OSError:
        # File read error — treat as no events (absence is not an error).
        return


def parse_claude_events(events_path: Path) -> ClaudeProgress | None:
    """Parse Claude Code's stream-json events file and extract progress metrics.

    Reads the events.jsonl file via ``iter_claude_events`` (the shared parsing
    primitive), accumulating tool_call_count and turn_count, and taking the
    last-seen cumulative usage fields (tokens, cost_usd).

    Two schemas are handled:

    * The real ``--output-format stream-json`` schema: ``assistant`` events
      carry ``message.content`` blocks (tool calls appear as ``tool_use``
      blocks); the terminal ``result`` event reports authoritative
      ``num_turns`` / ``total_cost_usd`` / ``usage``.
    * A legacy/simplified schema (``tool_call`` / ``user_message`` /
      ``assistant_message`` event types and top-level ``tokens`` / ``cost_usd``
      fields), kept for adapters that emit it.

    Tolerates partial/incomplete final lines (file is being appended to live).
    Malformed/unparseable lines are skipped, not raised.

    Returns None if the file doesn't exist (devin workers, or claude workers
    launched without the tee_stream_json flag) — absence is a valid, non-error state.

    Args:
        events_path: Path to the events.jsonl file

    Returns:
        ClaudeProgress with accumulated metrics, or None if file doesn't exist
    """
    tool_call_count = 0
    turn_count = 0
    tokens = None
    cost_usd = None

    for event in iter_claude_events(events_path):
        event_type = event.get("type")

        # Real stream-json schema (--output-format stream-json):
        # assistant events carry message.content blocks; tool calls
        # appear as tool_use blocks; the final result event reports
        # authoritative num_turns / total_cost_usd / usage.
        if event_type == "assistant":
            turn_count += 1
            message = event.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list):
                tool_call_count += sum(
                    1
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "tool_use"
                )
            usage = message.get("usage") if isinstance(message, dict) else None
            if isinstance(usage, dict):
                per_request = sum(
                    v
                    for v in (
                        usage.get("input_tokens"),
                        usage.get("output_tokens"),
                    )
                    if isinstance(v, int)
                )
                if per_request:
                    tokens = (tokens or 0) + per_request
        elif event_type == "result":
            num_turns = event.get("num_turns")
            if isinstance(num_turns, int) and num_turns > 0:
                turn_count = max(turn_count, num_turns)
            total_cost = event.get("total_cost_usd")
            if isinstance(total_cost, (int, float)):
                cost_usd = float(total_cost)
            usage = event.get("usage")
            if isinstance(usage, dict):
                final_tokens = sum(
                    v
                    for v in (
                        usage.get("input_tokens"),
                        usage.get("output_tokens"),
                    )
                    if isinstance(v, int)
                )
                if final_tokens:
                    tokens = final_tokens

        # Legacy/simplified schema kept for adapters that emit it.
        if event_type == "tool_call":
            tool_call_count += 1
        if event_type in ("user_message", "assistant_message"):
            turn_count += 1
        if "tokens" in event and isinstance(event["tokens"], int):
            tokens = event["tokens"]
        if "cost_usd" in event and isinstance(event["cost_usd"], (int, float)):
            cost_usd = float(event["cost_usd"])

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
    *,
    resume_margin_seconds: int = 0,
    adapter_kind: str = "claude-code",
    now: datetime | None = None,
) -> tuple[str | None, str | None]:
    """Classify a session failure by matching the log tail against provider throttle signatures.

    Returns a tuple of (failure_kind, throttled_until_iso):
    - failure_kind: "provider_suspended" | "rate_limited" | "quota_exhausted" |
      "provider_auth" | None
    - throttled_until_iso: ISO timestamp when the cooldown ends, or None if not
      applicable (None for ``provider_suspended`` — terminal, no cooldown)

    This is called after a session exits to detect provider throttling and set a cool-down window.

    ``resume_margin_seconds`` is an extra safety margin past the provider's
    reported reset (or fixed quota cooldown) time. Provider reset estimates are
    floors, not guarantees, and dispatching at T+0 races the actual reset
    (issue #499).

    ``adapter_kind`` selects provider-auth classification (issue #484): when
    ``"api"``, the log tail is also matched against 401/403/authentication
    patterns. Auth failures are checked BEFORE throttle markers so a dead API
    key does not masquerade as a generic throttle. On a ``provider_auth`` match,
    the cooldown reuses the existing quota-exhaustion constant (24h) — a dead
    key needs human intervention, not a 15-minute retry window.

    ``now`` is the injectable clock (mirrors ``devin_shell._classify_session_
    failure`` / ``get_rate_limit_defer_until``): defaults to
    ``datetime.now(UTC)`` when not supplied, so production behavior is
    byte-identical (issue #822).
    """
    if not log_path.exists():
        return None, None

    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None

    resolved_now = now if now is not None else datetime.now(UTC)

    # Check the last 2KB of the log (where error messages appear)
    tail = log_text[-2048:] if len(log_text) > 2048 else log_text

    # Provider account-suspension classification (api only, issue #1342).
    # Checked BEFORE auth/quota/throttle so a suspended account is never
    # retried as a transient rate-limit. A suspended account is a terminal
    # billing failure: it returns ``provider_suspended`` with NO cooldown
    # (the account will not self-heal) and escalates to an operator on the
    # first occurrence via ``DETERMINISTIC_ESCALATION_FAILURE_KINDS``.
    if adapter_kind == "api" and _PROVIDER_SUSPENDED_PATTERN.search(tail):
        return "provider_suspended", None

    # Provider-auth classification (api only, issue #484). Checked before
    # quota/throttle so an auth failure is never relabeled as a generic
    # throttle. A dead API key reuses the quota-exhaustion cooldown (24h)
    # rather than the short rate-limit cooldown — the key will not self-heal.
    if adapter_kind == "api" and _PROVIDER_AUTH_PATTERN.search(tail):
        cooldown = timedelta(hours=_DEFAULT_QUOTA_COOLDOWN_HOURS, seconds=resume_margin_seconds)
        throttled_until = resolved_now + cooldown
        return "provider_auth", throttled_until.replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        )

    # Check for quota exhaustion first (more severe)
    if _QUOTA_EXHAUSTED_PATTERN.search(tail):
        # Quota exhaustion uses a fixed 24-hour cooldown regardless of reset time
        cooldown = timedelta(hours=_DEFAULT_QUOTA_COOLDOWN_HOURS, seconds=resume_margin_seconds)
        throttled_until = resolved_now + cooldown
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
            else _DEFAULT_RATE_LIMIT_COOLDOWN_MINUTES,
            seconds=resume_margin_seconds,
        )
        throttled_until = resolved_now + cooldown
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
    _write_json_atomic(
        _sidecar_path(sessions_dir, record.issue_number, record.adapter_kind),
        record.to_dict(),
    )
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
    inconclusive_probe_deferred_count: int = 0,
    session_id: str | None = None,
    adapter_kind: str = "claude-code",
    provider: str = "",
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
        inconclusive_probe_deferred_count=inconclusive_probe_deferred_count,
        session_id=session_id,
        adapter_kind=adapter_kind,
        provider=provider,
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


def _apply_model_pin(command_template: tuple[str, ...], model: str) -> tuple[str, ...]:
    """Hard-pin ``--model {model}`` onto ``command_template``.

    Neither the worker nor the reviewer command template names a model by
    default, so a bare ``claude -p ...`` subprocess resolves its model from
    ambient global CLI state (e.g. whatever an interactive session on this
    machine last set via ``/model``) rather than anything charlie-work
    controls. That ambient state is not guaranteed to be available for
    headless fleet dispatch — see ``ClaudeCodeConfig.model``. Every
    occurrence of ``--model`` (bare flag + value, or ``--model=value``) is
    stripped first and a single authoritative pin appended, mirroring
    ``_sanitize_review_command_template``'s last-flag-wins handling of
    ``--permission-mode``. Safe to apply uniformly to both workers and
    reviewers: unlike ``--permission-mode``, ``--model`` has no bearing on
    write access or the reviewer's read-only posture.
    """
    filtered: list[str] = []
    skip_next = False
    for token in command_template:
        if skip_next:
            skip_next = False
            continue
        if token == "--model":
            skip_next = True
            continue
        if token.startswith("--model="):
            continue
        filtered.append(token)
    return tuple(filtered) + ("--model", model)


def _apply_effort_pin(command_template: tuple[str, ...], effort: str) -> tuple[str, ...]:
    """Hard-pin ``--effort {effort}`` onto ``command_template``.

    Mirrors ``_apply_model_pin``: strips any existing ``--effort`` flag (both
    space-separated and ``=``-joined forms) and appends a single authoritative
    pin. An empty ``effort`` string is a no-op — the CLI uses its default.
    """
    if not effort:
        return command_template
    filtered: list[str] = []
    skip_next = False
    for token in command_template:
        if skip_next:
            skip_next = False
            continue
        if token == "--effort":
            skip_next = True
            continue
        if token.startswith("--effort="):
            continue
        filtered.append(token)
    return tuple(filtered) + ("--effort", effort)


def _review_effort_arm(pr_number: int, fraction: float, salt: str) -> bool:
    """Deterministically assign a PR to the review_effort experiment's
    treatment arm.

    Pure function of ``(pr_number, fraction, salt)`` — no ``random``/``time``
    involved — so the same PR always lands in the same arm across rework
    rounds and re-dispatches (arm-hopping across rounds would contaminate
    the per-PR quality signal the experiment measures). Hashes
    ``f"{salt}:{pr_number}"`` with sha256 and maps the first 8 bytes to a
    uniform float in [0, 1); the PR is "treatment" iff that value is below
    ``fraction``.

    Returns True for "treatment", False for "control".
    """
    digest = hashlib.sha256(f"{salt}:{pr_number}".encode()).digest()
    point = int.from_bytes(digest[:8], "big") / 2**64
    return point < fraction


def resolve_review_effort(
    pr_number: int,
    review_dispatch: ReviewDispatchConfig,
    claude_code: ClaudeCodeConfig,
) -> tuple[str, str | None]:
    """Resolve the ``--effort`` string for a reviewer session, plus which
    review_effort_experiment arm (if any) the PR was assigned to.

    Returns ``(effort, arm)``:
      - When ``review_dispatch.review_effort_experiment_fraction <= 0.0``
        (the default), the experiment is disabled: ``arm`` is ``None`` and
        ``effort`` is ``review_dispatch.review_effort`` when set, else
        ``claude_code.effort`` — exactly the pre-experiment behavior.
      - Otherwise, ``arm`` is ``"treatment"`` or ``"control"`` per
        ``_review_effort_arm``, and ``effort`` is ``review_dispatch.review_effort``
        for treatment or ``claude_code.effort`` for control.

    Only meaningful for reviewer (``review=True``) launches; callers must
    gate on that themselves (worker launches always use ``claude_code.effort``).
    """
    fraction = review_dispatch.review_effort_experiment_fraction
    if fraction <= 0.0:
        return (review_dispatch.review_effort or claude_code.effort, None)
    if _review_effort_arm(pr_number, fraction, review_dispatch.review_effort_experiment_salt):
        return (review_dispatch.review_effort, "treatment")
    return (claude_code.effort, "control")


def _apply_max_turns_pin(command_template: tuple[str, ...], max_turns: int) -> tuple[str, ...]:
    """Hard-pin ``--max-turns {max_turns}`` onto ``command_template``.

    Mirrors ``_apply_model_pin``: strips any existing ``--max-turns`` flag
    (both space-separated and ``=``-joined forms) and appends a single
    authoritative pin. A ``max_turns`` of 0 or less is a no-op — the CLI
    uses its default (unlimited).
    """
    if max_turns <= 0:
        return command_template
    filtered: list[str] = []
    skip_next = False
    for token in command_template:
        if skip_next:
            skip_next = False
            continue
        if token == "--max-turns":
            skip_next = True
            continue
        if token.startswith("--max-turns="):
            continue
        filtered.append(token)
    return tuple(filtered) + ("--max-turns", str(max_turns))


def run_quota_probe(*, repo_root: Path, config: OrchestratorConfig) -> bool:
    """Run a single, cheap, read-only Haiku-model CLI call to check whether an
    active quota/rate-limit throttle has actually recovered (e.g. an operator
    switched the ambient Claude Code CLI to a different subscription
    account).

    Deliberately NOT ``launch_claude_worker``: no worktree, no branch, no
    sidecar, no Popen -- this is a single bounded, synchronous subprocess via
    ``subprocess_runner.run_captured``, mirroring how other short-lived,
    timeout-bounded external calls in this codebase are made (CLAUDE.md's
    non-blocking-adapter invariant applies to worker/reviewer dispatch, not
    to a deliberate, bounded health-check call like this one). Runs in
    ``repo_root`` itself -- never mutates it: reuses ``_REVIEW_COMMAND_TEMPLATE``
    (``--permission-mode plan``) capped to a single turn, so it is read-only
    and cheap regardless of what the probe prompt asks.

    Returns True ("green") only when the process exits 0 AND its combined
    stdout+stderr shows no quota/auth/rate-limit throttle signature --
    reusing the exact patterns ``_classify_session_failure`` applies to a
    real worker's log tail (single point of enforcement, PR #262 findings
    F1/F5) so a session that exits 0 with an embedded, in-band throttle
    message is never misread as recovery. Never raises: ``run_captured``
    already converts timeouts, missing binaries, and non-zero exits into a
    ``RunResult`` value.
    """
    probe = config.quota_probe
    command = _apply_model_pin(_REVIEW_COMMAND_TEMPLATE, probe.model)
    command = _apply_max_turns_pin(command, 1)
    command = (resolve_cli_binary(command[0]), *command[1:])
    result = run_captured(
        list(command),
        cwd=repo_root,
        timeout_seconds=probe.timeout_seconds,
        stdin=probe.prompt,
    )
    combined = f"{result.stdout}\n{result.stderr}"
    tail = combined[-2048:] if len(combined) > 2048 else combined
    if _PROVIDER_AUTH_PATTERN.search(tail):
        return False
    if _QUOTA_EXHAUSTED_PATTERN.search(tail):
        return False
    matched, _reset_minutes = match_throttle_tail(tail, config.runtime.throttle_error_markers)
    if matched:
        return False
    return result.ok


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
    adapter_kind: str = "claude-code",
    provider: str = "",
    resolved_review_effort: str | None = None,
    model_override: str | None = None,
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

    ``adapter_kind`` selects which sidecar suffix to write (``claude-code`` ->
    ``issue-<n>.claude.json``, ``api`` -> ``issue-<n>.api.json``). ``provider``
    is recorded on the sidecar but does not change the launch command in this
    issue; it is reserved for future adapter-specific command rendering.

    ``resolved_review_effort``, when provided on a ``review=True`` launch, is
    used verbatim as the ``--effort`` pin instead of re-deriving it via
    ``resolve_review_effort``. Callers that already computed the
    review_effort_experiment arm at claim time (e.g. ``dispatch_reviews``,
    which persists the arm to state/telemetry before launching) should pass
    it through here so there is exactly ONE computation of the arm on the
    production path, rather than two calls to the same pure function that
    merely agree by convention. When omitted (direct callers, unit tests),
    the effort is resolved internally as a fallback.

    ``model_override``, when provided, is pinned as the ``--model`` value
    instead of ``resolved_config.claude_code.model``. The api adapter
    (``api_worker.launch_api_worker``) passes the resolved provider's model
    here so the ``--model`` flag — which the Claude Code CLI gives precedence
    over ``ANTHROPIC_MODEL`` — selects the provider's model rather than the
    claude_code section's. When omitted (the default), the claude_code
    section's model is pinned exactly as before — the single enforcement
    point stays ``_apply_model_pin``, never an ``adapter_kind`` branch.
    """
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = _log_path(sessions_dir, issue_number, rework=rework, review=review)
    # Rotate the previous session's log before the new launch overwrites it.
    # This preserves the old log for post-mortem analysis on re-dispatch.
    _rotate_old_log(log_path)
    # Also rotate the previous events.jsonl if it exists.
    if tee_stream_json:
        _events_path_old = _events_path(sessions_dir, issue_number, rework=rework, review=review)
        _rotate_old_log(_events_path_old)
    session_id = str(uuid.uuid4())

    # Issue #426: recovery probes carry a Signal-1-style deferral counter. Seed
    # the recovery dict from the existing sidecar (if any) so consecutive
    # recovery attempts observe the same counter the reaper does.
    if not review and recovery is not None:
        recovery = dict(recovery)
        recovery.setdefault(
            "inconclusive_probe_deferred_count",
            _read_sidecar_inconclusive_count(sessions_dir, issue_number, adapter_kind),
        )

    if review:
        # Hard-pinned, not a default: see _sanitize_review_command_template
        # and PR #397 round-2 review. No caller-supplied command_template
        # (including one forwarded from ClaudeCodeConfig.command) can result
        # in a reviewer session running with write access.
        command_template = _sanitize_review_command_template(command_template)
    elif command_template is None:
        command_template = _WORKER_COMMAND_TEMPLATE
    resolved_config = config or OrchestratorConfig()
    # Issue #1245: the api adapter passes its provider's model so the
    # ``--model`` flag (which the Claude Code CLI prefers over
    # ``ANTHROPIC_MODEL``) selects the provider's model, not the
    # claude_code section's. Default to claude_code.model for every other
    # caller — single enforcement point stays _apply_model_pin, never an
    # adapter_kind branch.
    pinned_model = (
        model_override if model_override is not None else resolved_config.claude_code.model
    )
    command_template = _apply_model_pin(command_template, pinned_model)
    # Reviewer sessions may pin their own effort independently of worker
    # effort (empty string means fall back to claude_code.effort), optionally
    # split into a per-PR randomized treatment/control experiment — see
    # resolve_review_effort. Worker (non-review) launches always use
    # claude_code.effort unconditionally. When the caller already resolved
    # the arm at claim time (resolved_review_effort passed), use it directly
    # so the arm is computed exactly once on the production path; only
    # direct callers/tests that don't pass it fall back to resolving here.
    if review:
        effort = (
            resolved_review_effort
            if resolved_review_effort is not None
            else resolve_review_effort(
                issue_number, resolved_config.review_dispatch, resolved_config.claude_code
            )[0]
        )
    else:
        effort = resolved_config.claude_code.effort
    command_template = _apply_effort_pin(command_template, effort)
    if review:
        # Cap agentic turns for reviewer sessions to prevent unbounded
        # codebase exploration and runaway token spend. 0 = unlimited.
        command_template = _apply_max_turns_pin(
            command_template, resolved_config.review_dispatch.review_max_turns
        )

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
                sessions_dir=sessions_dir,
            )
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as exc:
        if isinstance(exc, WorktreeProbeFailedError):
            # Transient probe contention (e.g. index lock), not a confirmed-dirty
            # worktree. Must stay off DETERMINISTIC_ESCALATION_FAILURE_KINDS so it
            # takes the ordinary redispatch-cap path (issue #288 follow-up, PR #314).
            failure_kind = "worktree_probe_failed"
        elif isinstance(exc, WorktreeUnsafeError):
            failure_kind = "worktree_unsafe"
        elif isinstance(exc, ReworkBranchConflictError):
            failure_kind = "rework_branch_conflict"
        elif isinstance(exc, WorktreeForeignWriterError):
            failure_kind = "worktree_foreign_writer"
        elif isinstance(exc, LiveWorkerRedispatchError):
            failure_kind = "live_worker_redispatch_averted"
        else:
            failure_kind = None
        record = _error_record(
            issue_number=issue_number,
            branch=branch,
            # str() is load-bearing: the exception stores a Path, and an
            # unserializable field here destroys this whole failure record
            # mid-json.dump, downgrading the diagnosis to a generic launch
            # failure that burns the rework cap (issue #1184).
            worktree_path=str(getattr(exc, "worktree_path", ""))
            if isinstance(exc, WorktreeForeignWriterError)
            else "",
            prompt_path="",
            command=command_template,
            log_path=str(log_path),
            error=str(exc)
            if isinstance(exc, (LiveWorkerRedispatchError, WorktreeForeignWriterError))
            else f"worktree creation failed: {exc}",
            failure_kind=failure_kind,
            pid=exc.pid
            if isinstance(exc, LiveWorkerRedispatchError)
            else getattr(exc, "pid", None),
            process_start_time=exc.process_start_time
            if isinstance(exc, LiveWorkerRedispatchError)
            else None,
            inconclusive_probe_deferred_count=exc.inconclusive_probe_deferred_count
            if isinstance(exc, LiveWorkerRedispatchError)
            else 0,
            session_id=session_id,
            adapter_kind=adapter_kind,
            provider=provider,
        )
        return _write_record(sessions_dir, record)

    # A redispatch may have just preserved the prior attempt's branch tip
    # (issue #261) — fold that into whatever post-mortem sidecar already
    # exists for this issue so the ref is discoverable alongside the block
    # diagnosis it belongs to. Best-effort: never blocks or fails dispatch.
    if worktree.attempt_snapshot is not None and worktree.attempt_snapshot.ref_name is not None:
        merge_attempt_snapshot(sessions_dir, issue_number, worktree.attempt_snapshot)

    # The rework pre-merge hit a real conflict (worktree.py's
    # _merge_update_rework_branch): the worktree still launches, but the
    # worker must resolve it before touching the review feedback. Append the
    # notice to this session's disposable prompt file rather than failing
    # closed (see worktree.ReworkMergeConflict).
    if worktree.rework_conflict is not None:
        prompt_text = apply_rework_conflict_notice(prompt_text, worktree.rework_conflict)

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
            adapter_kind=adapter_kind,
            provider=provider,
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
            adapter_kind=adapter_kind,
            provider=provider,
        )
        return _write_record(sessions_dir, record)

    # Resolve argv[0] to the real binary before spawning. On Windows, npm
    # installs `claude` as a `claude.CMD` shim; `Popen(shell=False)` with the
    # bare name goes straight to CreateProcessW, which does not do the
    # PATHEXT-based extension search cmd.exe does, so it cannot find the shim
    # and fails with WinError 2 (issue #487) even though `claude` runs fine
    # from an interactive shell. resolve_cli_binary unwraps the shim to its
    # underlying .exe (rather than stopping at shutil.which's .CMD path,
    # which would route through cmd.exe and mangle `|`-containing args via
    # caret-escaping — see its docstring). No-op on POSIX or for binaries
    # that aren't npm .cmd/.bat shims (e.g. the test doubles below).
    if command:
        command = (resolve_cli_binary(command[0]), *command[1:])

    # If tee_stream_json is enabled, extend the command with --output-format stream-json
    # and set up a tee to write to both plaintext log and events file
    events_path = None
    if tee_stream_json:
        command = command + ("--output-format", "stream-json")
        # The installed Claude Code CLI hard-rejects `--print` +
        # `--output-format=stream-json` without `--verbose` ("Error: When
        # using --print, --output-format=stream-json requires --verbose"),
        # crashing the process before it does any work. Pair the flags
        # unconditionally, but idempotently — a caller-supplied
        # command_template may already carry --verbose.
        if "--verbose" not in command:
            command = command + ("--verbose",)
        events_path = _events_path(sessions_dir, issue_number, rework=rework, review=review)

    feed_stdin = "{prompt_path}" not in "".join(command_template)
    # Workers inherit the orchestrator's environment, with config-provided
    # overrides merged on top — e.g. PYTEST_XDIST_AUTO_NUM_WORKERS to bound a
    # worker's local `pytest -n auto` so a fleet of them doesn't oversubscribe
    # the shared host (see docs/RUNBOOK.md "Local host saturation ceiling
    # (claude-code adapter)"). `env` is a validated mapping (see config.py).
    # Sanitize the base environment to prevent VIRTUAL_ENV/UV_PROJECT_ENVIRONMENT
    # leaks from the orchestrator and to isolate GitHub CLI credentials
    # (GH_TOKEN/GITHUB_TOKEN dropped, GH_CONFIG_DIR forced to a worktree-local
    # empty directory) so workers do not inherit the orchestrator's admin token
    # or stored gh auth state (issue #502). To give workers a scoped GitHub
    # token, set claude_code.worker_env={"GH_TOKEN": "<scoped-PAT>"}.
    sanitized_env = sanitize_env(worktree.path)
    worker_env = {
        **sanitized_env,
        **{str(k): str(v) for k, v in (env or {}).items()},
    }
    # Issue #646: resolve what sanitize_env()+worker_env actually settled on,
    # purely for the launch-time diagnostic log below (does not affect
    # worker_env itself, which already carries the real values).
    xdist_cap, xdist_cap_source = resolve_pytest_cap(sanitized_env, env)
    uv_no_sync, uv_no_sync_source = resolve_uv_no_sync(worktree.path, sanitized_env, env)

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
                        process = popen_worker(
                            command,
                            cwd=str(worktree.path),
                            stdin=prompt_handle,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            env=worker_env,
                            start_new_session=(os.name != "nt"),
                            text=True,
                        )
                    finally:
                        prompt_handle.close()
                else:
                    process = popen_worker(
                        command,
                        cwd=str(worktree.path),
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        env=worker_env,
                        start_new_session=(os.name != "nt"),
                        text=True,
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
                        process = popen_worker(
                            command,
                            cwd=str(worktree.path),
                            stdin=prompt_handle,
                            stdout=log_handle,
                            stderr=subprocess.STDOUT,
                            env=worker_env,
                            start_new_session=(os.name != "nt"),
                        )
                else:
                    process = popen_worker(
                        command,
                        cwd=str(worktree.path),
                        stdin=subprocess.DEVNULL,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        env=worker_env,
                        start_new_session=(os.name != "nt"),
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
            adapter_kind=adapter_kind,
            provider=provider,
        )
        return _write_record(sessions_dir, record)

    # Capture process creation time immediately after spawn to verify identity later
    process_start_time = _get_process_start_time(process.pid)

    # Issue #773: persist this worker's terminal status (exit code + duration)
    # once it exits, so a later orphan-detection pass can tell a clean exit-0
    # no-op apart from a genuine crash instead of inferring it from PID
    # liveness alone. Does not block this function's return; see
    # start_terminal_status_watcher's docstring.
    #
    # Issue #1354: extended to review launches too. A reviewer that dies
    # mid-session leaves no terminal signal in its events.jsonl (the stream
    # is cut before the ``result`` event), so the exit code captured here is
    # the only durable record of HOW the process ended. The review-verdict
    # reaper (``_reap_review_verdicts``) reads it back via
    # ``find_worker_terminal_status`` and folds it into the
    # ``review_verdict_missed`` payload's ``cause`` field. Review sessions
    # pass ``worktree_path=None`` because they use an isolated review
    # checkout (not a worker worktree) whose outcome file is not relevant
    # here -- only the exit code is.
    start_terminal_status_watcher(
        process,
        worker_terminal_status_path(sessions_dir, issue_number, _sidecar_suffix(adapter_kind)),
        worktree_path=worktree.path if not review else None,
    )

    try:
        write_worktree_marker(worktree.path, process.pid, session_id)
    except OSError:
        # Best-effort marker write must not derail a successful launch.
        pass

    # Issue #646: launch-time INFO log so a reader can answer "how many
    # suites were running at <time>, from which worktrees, at what cap"
    # without process forensics. Paired with the exit-side census log in
    # workflow.py (_log_worker_census) — join on session_id/pid/worktree.
    logger.info(
        "worker launch: adapter=claude-code issue=%s worktree=%s pid=%s session_id=%s "
        "xdist_cap=%s(%s) uv_no_sync=%s(%s) at=%s",
        issue_number,
        worktree.path,
        process.pid,
        session_id,
        xdist_cap,
        xdist_cap_source,
        uv_no_sync,
        uv_no_sync_source,
        datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )

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
        session_id=session_id,
        adapter_kind=adapter_kind,
        provider=provider,
        xdist_cap=xdist_cap,
        uv_no_sync=uv_no_sync,
    )
    return _write_record(sessions_dir, record)


def read_worker_records(
    sessions_dir: Path, *, adapter_kind: str | None = "claude-code"
) -> list[ClaudeWorkerRecord]:
    """Load Claude adapter sidecars from ``sessions_dir``.

    By default only ``claude-code`` sidecars (``issue-*.claude.json``) are
    returned, preserving the behavior expected by existing callers. Pass
    ``adapter_kind="api"`` to read only ``issue-*.api.json`` sidecars, or
    ``adapter_kind=None`` to read every known adapter kind.

    Malformed sidecars are skipped rather than raising — a corrupt file from
    a crashed write must not take down reconciliation for every other worker.
    """
    if not sessions_dir.is_dir():
        return []
    if adapter_kind is None:
        suffixes = list(_ADAPTER_SIDECAR_SUFFIXES.values())
    else:
        suffixes = [_sidecar_suffix(adapter_kind)]
    records: list[ClaudeWorkerRecord] = []
    paths = sorted(
        path for suffix in suffixes for path in sessions_dir.glob(f"issue-*.{suffix}.json")
    )
    for path in paths:
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

    argv[0] is resolved through ``resolve_cli_binary`` for the same reason as
    ``launch_claude_worker`` (issue #487): a bare ``"claude"`` on Windows
    cannot be found by ``CreateProcessW`` since npm installs it as a
    ``claude.CMD`` shim.
    """
    resolved_command = (resolve_cli_binary(command[0]), *command[1:]) if command else command
    return run_captured(list(resolved_command), cwd=repo_root, timeout_seconds=15)


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
    adapter_kind: str = "claude-code",
    session_completed: bool = False,
    now: datetime | None = None,
) -> tuple[str | None, str | None]:
    """Update a worker record with failure classification after the session exits.

    This reads the existing sidecar, classifies the failure from the log tail,
    and writes back an updated record with failure_kind set.

    Log-tail classification (``_classify_session_failure``) always runs first.
    If it detects a provider throttle signature (``rate_limited`` /
    ``quota_exhausted``) or a provider-auth failure (``provider_auth``, api
    only — issue #484), that classification wins — including its computed
    ``throttled_until`` cooldown — regardless of ``fallback_kind``. Only when
    the log shows no throttle/auth signature does ``fallback_kind`` apply
    (e.g. the stall watchdog's "stalled" default, or the launch-stall
    watchdog's "launch_stalled" default). This ordering matters: a worker
    that dies because it hit a provider rate limit or a dead API key must be
    classified as such even when the caller only knows "this looked stalled"
    — otherwise ``throttled_until`` never gets set and dispatch keeps
    relaunching workers into the same limit.

    ``session_completed`` (issue #656): when the caller has already confirmed
    via worktree inspection that this session produced complete, committable
    work, log-tail classification is skipped entirely and ``fallback_kind`` is
    used directly. A session that finished real work cannot also have been
    killed by a provider quota/rate-limit/auth failure — that's ground truth,
    not a heuristic — so there is no need to search its log tail at all.
    Without this, the marker search treats the worker's own final-turn
    completion summary (ordinary Claude Code CLI behavior — nearly every
    session ends with a natural-language write-up) as fair game, and generic
    markers like "usage limit" / "rate limit" false-positive on legitimate
    prose about this codebase's own rate-limit/quota domain. Observed live
    2026-07-27: a rework session for issue #651 (the bug about this exact
    false-positive class in the *reviewer* path, fixed by #652) quoted its own
    fix's marker examples in its completion summary and was reclassified
    quota_exhausted, setting a fleet-wide 24h dispatch throttle despite having
    finished cleanly — the same failure mode #652 fixed for reviewers, just in
    the sibling worker-classification path #652 didn't touch.

    ``config`` is optional for backward compatibility; when provided, its
    ``runtime.throttle_error_markers`` and ``runtime.throttle_resume_margin_s``
    are used instead of the defaults.

    ``adapter_kind`` selects the sidecar filename suffix (``issue-<n>.api.json``
    for ``"api"``) and enables provider-auth classification (issue #484).
    Defaults to ``"claude-code"`` for backward compatibility.

    ``now`` is forwarded to ``_classify_session_failure`` (issue #822's
    injectable clock); defaults to ``datetime.now(UTC)`` there when omitted.

    Returns a tuple of (failure_kind, throttled_until_iso) for the caller to
    update runtime state if needed. ``throttled_until_iso`` is only non-None
    when log-tail classification actually matched a throttle/auth signature.
    """
    sidecar_path = _sidecar_path(sessions_dir, issue_number, adapter_kind)
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
    log_path_str = payload.get("log_path") if not session_completed else None
    if log_path_str:
        if config is not None:
            throttle_markers = config.runtime.throttle_error_markers
            resume_margin_seconds = config.runtime.throttle_resume_margin_s
        else:
            throttle_markers = None
            resume_margin_seconds = 0
        classified_kind, throttled_until = _classify_session_failure(
            Path(log_path_str),
            throttle_markers,
            resume_margin_seconds=resume_margin_seconds,
            adapter_kind=adapter_kind,
            now=now,
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
    "iter_claude_events",
    "_events_path",
]
