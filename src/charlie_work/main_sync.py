"""Keep each registered repo's local default branch on ``origin`` (docs/specs/main-sync.md).

Once per full supervisor pass, ``run_main_sync_pass`` walks ``fleet.json`` -- minus the
supervisor's own checkout, which ``self_deploy`` owns -- fetches each repo's default
branch and lets ``main_sync_classify.classify`` pick one action: fast-forward the
primary checkout, move the ref when another branch is checked out, or skip and say why.
A repo may declare ``main_sync.post_sync_command``; an advance whose range touches
``post_sync_when_paths`` stands only if that command exits 0, and is otherwise rolled
back with ``git reset --keep``. Events go through a per-repo ``WriteGate``, so a dry
run writes nothing. Nothing here raises to the supervisor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from . import main_sync_git as git
from .atomic_write import write_json_atomic
from .main_sync_classify import (
    ALARM_ON_SIGHT_REASONS,
    SILENT_SKIP_REASONS,
    CheckoutState,
    classify,
    instruction_paths,
    split_at_first_touch,
)
from .main_sync_config import MainSyncConfig
from .main_sync_state import RepoSyncState, load_sync_state, save_sync_state
from .subprocess_runner import RunResult
from .write_gate import WriteGate

logger = logging.getLogger(__name__)

MARKER_FILENAME = "charlie-main-sync.json"
MARKER_SCHEMA = 1
EX_TEMPFAIL = 75
FAILURE_ALARM_THRESHOLD = 3
DEFER_ALARM_SECONDS = 4 * 3600
BACKOFF_BASE_SECONDS = 300
BACKOFF_CAP_SECONDS = 3600
STDERR_TAIL_CHARS = 2000


@dataclass(frozen=True)
class RepoSyncResult:
    """One repo's outcome this pass.

    ``outcome``: ``up_to_date``, ``advanced``, ``skipped``, ``waiting`` (inside a
    post_sync back-off window), ``deferred``, ``failed`` or ``previewed`` (dry run;
    ``reason`` is the action it would take). ``alarm`` names an alarm raised this
    pass; ``recovered`` marks a success that cleared an earlier alarm.
    """

    repo: str
    outcome: str
    reason: str | None = None
    from_sha: str | None = None
    to_sha: str | None = None
    ref_only: bool = False
    paths: tuple[str, ...] = ()
    detail: str = ""
    alarm: str | None = None
    recovered: bool = False


@dataclass(frozen=True)
class SyncContext:
    repo: str
    repo_root: Path
    config: MainSyncConfig
    fleet_dir: Path
    now: float  # wall-clock epoch seconds
    write_gate: WriteGate

    @property
    def dry_run(self) -> bool:
        return self.write_gate.dry_run


Outcome = tuple[RepoSyncResult, RepoSyncState]


def sync_repo(ctx: SyncContext) -> RepoSyncResult:
    """Sync one repo and persist its counters (never under dry run)."""
    loaded = load_sync_state(ctx.fleet_dir, ctx.repo)
    result, state = _sync(ctx, replace(loaded, budget_skipped=False))
    if not ctx.dry_run and state != loaded:
        save_sync_state(ctx.fleet_dir, ctx.repo, state)
    return result


def _sync(ctx: SyncContext, state: RepoSyncState) -> Outcome:
    if not ctx.config.enabled:
        return _skip(ctx, state, "disabled")
    if not git.has_origin(ctx.repo_root):
        return _skip(ctx, state, "no_origin")
    if not git.is_primary_checkout(ctx.repo_root):
        return _skip(ctx, state, "not_primary")
    branch = git.resolve_default_branch(ctx.repo_root, dry_run=ctx.dry_run)
    if branch is None:
        return _skip(ctx, state, "no_default_branch")
    if not ctx.dry_run:
        fetched = git.fetch_default(ctx.repo_root, branch)
        if not fetched.ok:
            return _skip(ctx, state, "fetch_failed", detail=_tail(fetched))
    snap = git.read_checkout_state(ctx.repo_root, branch)
    if isinstance(snap, str):
        return _skip(ctx, state, snap)
    decision = classify(snap)
    if decision.action == "none":
        return _success(state, RepoSyncResult(ctx.repo, "up_to_date"))
    if decision.action == "skip":
        return _skip(ctx, state, decision.reason or "unknown", snap=snap, paths=decision.paths)
    if ctx.dry_run:
        preview = RepoSyncResult(
            ctx.repo,
            "previewed",
            decision.action,
            snap.local_sha,
            snap.origin_sha,
            ref_only=decision.action == "ref_update",
        )
        return preview, state
    if decision.action == "ref_update":
        return _ref_update(ctx, state, snap)
    return _advance(ctx, state, snap)


def _success(state: RepoSyncState, result: RepoSyncResult) -> Outcome:
    """A success clears every counter; clearing an alarm makes it a recovery."""
    return replace(result, recovered=state.alarmed), RepoSyncState()


def _skip(
    ctx: SyncContext,
    state: RepoSyncState,
    reason: str,
    *,
    snap: CheckoutState | None = None,
    paths: tuple[str, ...] = (),
    detail: str = "",
) -> Outcome:
    local = snap.local_sha if snap else None
    origin = snap.origin_sha if snap else None
    result = RepoSyncResult(ctx.repo, "skipped", reason, local, origin, paths=paths, detail=detail)
    key = f"{reason}:{local or ''}:{origin or ''}"
    if reason in SILENT_SKIP_REASONS or key == state.last_skip_key:
        return result, state
    _emit_skipped(ctx, result)
    state = replace(state, last_skip_key=key)
    if reason in ALARM_ON_SIGHT_REASONS and not state.history_alarmed:
        alarm = _alarm(ctx, reason, f"local {local} shares no history with origin {origin}")
        return replace(result, alarm=alarm), replace(state, history_alarmed=True)
    return result, state


def _ref_update(ctx: SyncContext, state: RepoSyncState, snap: CheckoutState) -> Outcome:
    moved = git.update_ref(ctx.repo_root, snap.default_branch, snap.origin_sha, snap.local_sha)
    if not moved.ok:
        return _skip(ctx, state, "ref_update_failed", snap=snap, detail=_tail(moved))
    result = RepoSyncResult(
        ctx.repo, "advanced", None, snap.local_sha, snap.origin_sha, ref_only=True
    )
    _emit_advanced(ctx, result, commits=snap.behind)
    return _success(state, result)


def _advance(ctx: SyncContext, state: RepoSyncState, snap: CheckoutState) -> Outcome:
    start, target = snap.local_sha, snap.origin_sha
    command = ctx.config.post_sync_command
    commits = git.list_incoming_commits(ctx.repo_root, start, target) if command else ()
    if commits is None:
        return _skip(ctx, state, "git_error", snap=snap)
    safe, needs_post_sync = split_at_first_touch(commits, ctx.config.post_sync_when_paths)
    if not command or not needs_post_sync:
        moved = git.fast_forward(ctx.repo_root, target)
        if not moved.ok:
            return _skip(ctx, state, "ff_failed", snap=snap, detail=_tail(moved))
        return _success(state, _final_advance(ctx, start, target))
    landed = start
    if safe is not None and git.is_ancestor(ctx.repo_root, start, safe):
        if git.fast_forward(ctx.repo_root, safe).ok:
            landed = safe
    if state.next_attempt_at is not None and ctx.now < state.next_attempt_at:
        _settle(ctx, start, landed)
        return RepoSyncResult(ctx.repo, "waiting", "backoff", landed, target), state
    moved = git.fast_forward(ctx.repo_root, target)
    if not moved.ok:
        _settle(ctx, start, landed)
        return _skip(ctx, state, "ff_failed", snap=snap, detail=_tail(moved))
    ran = git.run_post_sync(ctx.repo_root, command, ctx.config.post_sync_timeout_seconds)
    if ran.ok:
        return _success(state, _final_advance(ctx, start, target))
    return _roll_back(ctx, state, start, landed, target, ran)


def _final_advance(ctx: SyncContext, start: str, end: str) -> RepoSyncResult:
    """The working tree now sits at ``end`` for good: write the marker, log the event."""
    result = RepoSyncResult(ctx.repo, "advanced", None, start, end)
    _write_marker(ctx, start, end)
    _emit_advanced(ctx, result, commits=git.count_commits(ctx.repo_root, start, end))
    return result


def _settle(ctx: SyncContext, start: str, landed: str) -> None:
    """Record a partial fast-forward that stands while its remainder waits."""
    if landed != start:
        _final_advance(ctx, start, landed)


def _roll_back(
    ctx: SyncContext,
    state: RepoSyncState,
    start: str,
    landed: str,
    target: str,
    ran: RunResult,
) -> Outcome:
    tail = _tail(ran)
    if not git.reset_keep(ctx.repo_root, landed).ok:
        _emit_rollback_refused(ctx, landed, target, tail)
        _write_marker(ctx, start, target)  # the tree stays at target; sessions must hear it
        alarm = _alarm(ctx, "rollback_refused", f"post_sync failed; reset --keep {landed} refused")
        result = RepoSyncResult(
            ctx.repo, "failed", "rollback_refused", landed, target, detail=tail, alarm=alarm
        )
        return result, replace(state, rollback_alarmed=True)
    _emit_rolled_back(ctx, landed, target)
    _settle(ctx, start, landed)
    if ran.returncode == EX_TEMPFAIL and not ran.timed_out:
        return _defer(ctx, state, landed, target)
    return _fail(ctx, state, landed, target, tail)


def _defer(ctx: SyncContext, state: RepoSyncState, start: str, target: str) -> Outcome:
    count = state.defer_count + 1
    since = state.defer_since if state.defer_since is not None else ctx.now
    retry_at = ctx.now + _backoff(count)
    state = replace(state, defer_count=count, defer_since=since, next_attempt_at=retry_at)
    result = RepoSyncResult(ctx.repo, "deferred", "post_sync_tempfail", start, target)
    _emit_deferred(ctx, result, retry_at)
    if ctx.now - since >= DEFER_ALARM_SECONDS and not state.defer_alarmed:
        alarm = _alarm(ctx, "deferred_4h", f"post_sync has deferred since {_iso(since)}")
        return replace(result, alarm=alarm), replace(state, defer_alarmed=True)
    return result, state


def _fail(ctx: SyncContext, state: RepoSyncState, start: str, target: str, tail: str) -> Outcome:
    streak = state.failure_streak + 1
    state = replace(state, failure_streak=streak, next_attempt_at=ctx.now + _backoff(streak))
    result = RepoSyncResult(ctx.repo, "failed", "post_sync_failed", start, target, detail=tail)
    _emit_failed(ctx, result)
    if streak >= FAILURE_ALARM_THRESHOLD and not state.failure_alarmed:
        alarm = _alarm(ctx, "post_sync_failing", f"{streak} consecutive post_sync failures")
        return replace(result, alarm=alarm), replace(state, failure_alarmed=True)
    return result, state


def _backoff(attempt: int) -> int:
    return min(BACKOFF_BASE_SECONDS * 2 ** (attempt - 1), BACKOFF_CAP_SECONDS)


def _tail(result: RunResult) -> str:
    text = result.stderr or result.error or ("timed out" if result.timed_out else "")
    return str(text)[-STDERR_TAIL_CHARS:]


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).isoformat(timespec="seconds")


def _write_marker(ctx: SyncContext, start: str, end: str) -> None:
    """``<git-common-dir>/charlie-main-sync.json`` -- the hook contract (RUNBOOK)."""
    common = git.git_common_dir(ctx.repo_root)
    if common is None:
        logger.warning("main-sync: no git common dir for %s; marker not written", ctx.repo)
        return
    changed = git.changed_paths(ctx.repo_root, start, end)
    payload = {
        "schema": MARKER_SCHEMA,
        "from": start,
        "to": end,
        "commits": git.count_commits(ctx.repo_root, start, end),
        "instruction_paths": list(instruction_paths(changed)),
        "ts": _iso(ctx.now),
    }
    try:
        write_json_atomic(common / MARKER_FILENAME, payload)
    except OSError as exc:
        logger.warning("main-sync: marker write failed for %s: %s", ctx.repo, exc)


def _payload(result: RepoSyncResult) -> dict[str, object]:
    return {
        "repo": result.repo,
        "from_sha": result.from_sha,
        "to_sha": result.to_sha,
        "reason": result.reason,
    }


# Each emitter passes a literal kind: tests/_instrumentation_kind_scanner.py reads them.
def _emit_advanced(ctx: SyncContext, result: RepoSyncResult, *, commits: int) -> None:
    payload = {**_payload(result), "ref_only": result.ref_only, "commits": commits}
    ctx.write_gate.log_event(kind="main_sync_advanced", payload=payload)


def _emit_skipped(ctx: SyncContext, result: RepoSyncResult) -> None:
    payload = {**_payload(result), "paths": list(result.paths), "detail": result.detail}
    ctx.write_gate.log_event(kind="main_sync_skipped", payload=payload)


def _emit_deferred(ctx: SyncContext, result: RepoSyncResult, retry_at: float) -> None:
    payload = {**_payload(result), "retry_at": _iso(retry_at)}
    ctx.write_gate.log_event(kind="main_sync_deferred", payload=payload)


def _emit_failed(ctx: SyncContext, result: RepoSyncResult) -> None:
    payload = {**_payload(result), "stderr_tail": result.detail}
    ctx.write_gate.log_event(kind="main_sync_failed", payload=payload)


def _emit_rolled_back(ctx: SyncContext, start: str, target: str) -> None:
    payload = {"repo": ctx.repo, "from_sha": start, "to_sha": target, "reset_to": start}
    ctx.write_gate.log_event(kind="main_sync_rolled_back", payload=payload)


def _emit_rollback_refused(ctx: SyncContext, start: str, target: str, tail: str) -> None:
    payload = {"repo": ctx.repo, "from_sha": start, "to_sha": target, "stderr_tail": tail}
    ctx.write_gate.log_event(kind="main_sync_rollback_refused", payload=payload)


def _alarm(ctx: SyncContext, reason: str, detail: str) -> str | None:
    """Log the alarm event; return its name, or None under dry run (no notification)."""
    if ctx.dry_run:
        return None
    payload = {"repo": ctx.repo, "reason": reason, "detail": detail}
    ctx.write_gate.log_event(kind="main_sync_alarm", payload=payload)
    return reason
