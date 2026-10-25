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
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from . import layout
from . import main_sync_git as git
from .fleet_paths import fleet_dir as resolve_fleet_dir
from .fleet_registry import _iter_registered_repo_contexts
from .main_sync_classify import (
    ALARM_ON_SIGHT_REASONS,
    SILENT_SKIP_REASONS,
    CheckoutState,
    classify,
    split_at_first_touch,
)
from .main_sync_config import MainSyncConfig
from .main_sync_events import (  # noqa: F401 - MARKER_FILENAME is re-exported
    MARKER_FILENAME,
    raise_alarm,
    emit_advanced,
    emit_deferred,
    emit_failed,
    emit_rolled_back,
    emit_rollback_refused,
    emit_skipped,
    iso_utc,
    write_marker,
)
from .notify import AttentionEntry
from .main_sync_state import RepoSyncState, load_sync_state, save_sync_state
from .subprocess_runner import RunResult
from .write_gate import WriteGate

logger = logging.getLogger(__name__)

EX_TEMPFAIL = 75
FAILURE_ALARM_THRESHOLD = 3
DEFER_ALARM_SECONDS = 4 * 3600
BACKOFF_BASE_SECONDS = 300
BACKOFF_CAP_SECONDS = 3600
STDERR_TAIL_CHARS = 2000
BUDGET_SECONDS = 120
_PRINTED_OUTCOMES = frozenset({"advanced", "deferred", "failed", "previewed"})


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
    emit_skipped(ctx, result)
    state = replace(state, last_skip_key=key)
    if reason in ALARM_ON_SIGHT_REASONS and not state.history_alarmed:
        alarm = raise_alarm(ctx, reason, f"local {local} shares no history with origin {origin}")
        return replace(result, alarm=alarm), replace(state, history_alarmed=True)
    return result, state


def _ref_update(ctx: SyncContext, state: RepoSyncState, snap: CheckoutState) -> Outcome:
    moved = git.update_ref(ctx.repo_root, snap.default_branch, snap.origin_sha, snap.local_sha)
    if not moved.ok:
        return _skip(ctx, state, "ref_update_failed", snap=snap, detail=_tail(moved))
    result = RepoSyncResult(
        ctx.repo, "advanced", None, snap.local_sha, snap.origin_sha, ref_only=True
    )
    emit_advanced(ctx, result, commits=snap.behind)
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
    write_marker(ctx, start, end)
    emit_advanced(ctx, result, commits=git.count_commits(ctx.repo_root, start, end))
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
        emit_rollback_refused(ctx, landed, target, tail)
        write_marker(ctx, start, target)  # the tree stays at target; sessions must hear it
        alarm = raise_alarm(
            ctx, "rollback_refused", f"post_sync failed; reset --keep {landed} refused"
        )
        result = RepoSyncResult(
            ctx.repo, "failed", "rollback_refused", landed, target, detail=tail, alarm=alarm
        )
        return result, replace(state, rollback_alarmed=True)
    emit_rolled_back(ctx, landed, target)
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
    emit_deferred(ctx, result, retry_at)
    if ctx.now - since >= DEFER_ALARM_SECONDS and not state.defer_alarmed:
        alarm = raise_alarm(ctx, "deferred_4h", f"post_sync has deferred since {iso_utc(since)}")
        return replace(result, alarm=alarm), replace(state, defer_alarmed=True)
    return result, state


def _fail(ctx: SyncContext, state: RepoSyncState, start: str, target: str, tail: str) -> Outcome:
    streak = state.failure_streak + 1
    state = replace(state, failure_streak=streak, next_attempt_at=ctx.now + _backoff(streak))
    result = RepoSyncResult(ctx.repo, "failed", "post_sync_failed", start, target, detail=tail)
    emit_failed(ctx, result)
    if streak >= FAILURE_ALARM_THRESHOLD and not state.failure_alarmed:
        alarm = raise_alarm(ctx, "post_sync_failing", f"{streak} consecutive post_sync failures")
        return replace(result, alarm=alarm), replace(state, failure_alarmed=True)
    return result, state


def _backoff(attempt: int) -> int:
    return min(BACKOFF_BASE_SECONDS * 2 ** (attempt - 1), BACKOFF_CAP_SECONDS)


def _tail(result: RunResult) -> str:
    text = result.stderr or result.error or ("timed out" if result.timed_out else "")
    return str(text)[-STDERR_TAIL_CHARS:]


@dataclass(frozen=True)
class MainSyncPassResult:
    results: tuple[RepoSyncResult, ...] = ()
    skipped_repos: tuple[str, ...] = ()  # registry entries the walker could not resolve
    dry_run: bool = False


def run_main_sync_pass(
    *,
    fleet_dir_override: str | None,
    exclude_root: Path,
    dry_run: bool,
    budget_seconds: float = BUDGET_SECONDS,
    wall_clock: Callable[[], float] = time.time,
    monotonic: Callable[[], float] = time.monotonic,
) -> MainSyncPassResult:
    """One main-sync step over the whole registry. Never raises."""
    try:
        return _run_pass(
            fleet_dir_override, exclude_root, dry_run, budget_seconds, wall_clock, monotonic
        )
    except Exception:  # noqa: BLE001 - a pass step must never take the supervisor loop down
        logger.exception("main-sync: pass aborted")
        return MainSyncPassResult(dry_run=dry_run)


def _run_pass(
    fleet_dir_override: str | None,
    exclude_root: Path,
    dry_run: bool,
    budget_seconds: float,
    wall_clock: Callable[[], float],
    monotonic: Callable[[], float],
) -> MainSyncPassResult:
    fleet_dir = resolve_fleet_dir(override=fleet_dir_override)
    unresolved: list[str] = []
    contexts = list(_iter_registered_repo_contexts(fleet_dir_override, unresolved))
    states = {name: load_sync_state(fleet_dir, name) for name, *_ in contexts}
    contexts.sort(key=lambda c: not states[c[0]].budget_skipped)  # stable: skipped go first
    excluded = _norm(exclude_root)
    deadline = monotonic() + budget_seconds
    results: list[RepoSyncResult] = []
    for name, repo_root, state_dir, repo_config in contexts:
        if _norm(repo_root) == excluded:
            results.append(RepoSyncResult(name, "skipped", "excluded"))
        elif monotonic() >= deadline:
            results.append(RepoSyncResult(name, "skipped", "budget_exhausted"))
            if not dry_run and not states[name].budget_skipped:
                save_sync_state(fleet_dir, name, replace(states[name], budget_skipped=True))
        else:
            state_path = layout.state_file_path(state_dir)
            gate = WriteGate(dry_run=dry_run, state_path=state_path, repo=name)
            ctx = SyncContext(
                name, repo_root, repo_config.main_sync, fleet_dir, wall_clock(), gate
            )
            results.append(_contained(ctx))
    return MainSyncPassResult(tuple(results), tuple(unresolved), dry_run)


def _contained(ctx: SyncContext) -> RepoSyncResult:
    try:
        return sync_repo(ctx)
    except Exception as exc:  # noqa: BLE001 - one repo's surprise must not stop the rest
        logger.exception("main-sync: %s failed", ctx.repo)
        detail = str(exc)[-STDERR_TAIL_CHARS:]
        return RepoSyncResult(ctx.repo, "failed", "internal_error", detail=detail)


def _norm(path: Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def _span(result: RepoSyncResult) -> str:
    if result.from_sha and result.to_sha:
        return f"{result.from_sha[:9]}..{result.to_sha[:9]}"
    return ""


def format_pass_lines(result: MainSyncPassResult) -> tuple[str, ...]:
    """Operator lines: advances, deferrals, failures, previews, alarms, budget skips."""
    lines = []
    for r in result.results:
        printed = r.outcome in _PRINTED_OUTCOMES or r.reason == "budget_exhausted"
        if not printed and r.alarm is None:
            continue
        parts = (
            r.repo,
            r.outcome,
            r.reason or "",
            _span(r),
            "(ref only)" if r.ref_only else "",
            f"ALARM {r.alarm}" if r.alarm else "",
        )
        lines.append(" ".join(p for p in parts if p))
    return tuple(lines)


def health_entries(result: MainSyncPassResult) -> tuple[AttentionEntry, ...]:
    """ERROR per alarm raised this pass; OK for a success that cleared one, so the fleet
    health baseline does not latch ERROR (#817). Nothing under dry run."""
    if result.dry_run:
        return ()
    entries = []
    for r in result.results:
        if r.alarm is not None:
            entries.append(_attention(r, "ERROR", f"main-sync alarm {r.alarm} {_span(r)}".strip()))
        elif r.recovered:
            entries.append(_attention(r, "OK", f"main-sync {r.outcome} {_span(r)}".strip()))
    return tuple(entries)


def _attention(result: RepoSyncResult, health: str, line: str) -> AttentionEntry:
    return AttentionEntry(
        issue_number=-1,
        adapter_kind=f"main-sync {result.repo}",
        health=health,
        previous_health=None,
        last_log_line=line,
        pid=None,
    )
