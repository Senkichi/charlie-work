"""main-sync's write side: the marker file and the ``main_sync_*`` events.

Split out of ``main_sync`` for size. Every emitter passes a literal kind to
``WriteGate.log_event`` so the event-kind scanner can read it.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from . import main_sync_git as git
from .atomic_write import write_json_atomic
from .main_sync_classify import instruction_paths

if TYPE_CHECKING:
    from .main_sync import RepoSyncResult, SyncContext

logger = logging.getLogger(__name__)

MARKER_FILENAME = "charlie-main-sync.json"
MARKER_SCHEMA = 1


def iso_utc(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).isoformat(timespec="seconds")


def write_marker(ctx: SyncContext, start: str, end: str) -> None:
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
        "ts": iso_utc(ctx.now),
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
def emit_advanced(ctx: SyncContext, result: RepoSyncResult, *, commits: int) -> None:
    payload = {**_payload(result), "ref_only": result.ref_only, "commits": commits}
    ctx.write_gate.log_event(kind="main_sync_advanced", payload=payload)


def emit_skipped(ctx: SyncContext, result: RepoSyncResult) -> None:
    payload = {**_payload(result), "paths": list(result.paths), "detail": result.detail}
    ctx.write_gate.log_event(kind="main_sync_skipped", payload=payload)


def emit_deferred(ctx: SyncContext, result: RepoSyncResult, retry_at: float) -> None:
    payload = {**_payload(result), "retry_at": iso_utc(retry_at)}
    ctx.write_gate.log_event(kind="main_sync_deferred", payload=payload)


def emit_failed(ctx: SyncContext, result: RepoSyncResult) -> None:
    payload = {**_payload(result), "stderr_tail": result.detail}
    ctx.write_gate.log_event(kind="main_sync_failed", payload=payload)


def emit_rolled_back(ctx: SyncContext, start: str, target: str) -> None:
    payload = {"repo": ctx.repo, "from_sha": start, "to_sha": target, "reset_to": start}
    ctx.write_gate.log_event(kind="main_sync_rolled_back", payload=payload)


def emit_rollback_refused(ctx: SyncContext, start: str, target: str, tail: str) -> None:
    payload = {"repo": ctx.repo, "from_sha": start, "to_sha": target, "stderr_tail": tail}
    ctx.write_gate.log_event(kind="main_sync_rollback_refused", payload=payload)


def raise_alarm(ctx: SyncContext, reason: str, detail: str) -> str | None:
    """Log the alarm event; return its name, or None under dry run (no notification)."""
    if ctx.dry_run:
        return None
    payload = {"repo": ctx.repo, "reason": reason, "detail": detail}
    ctx.write_gate.log_event(kind="main_sync_alarm", payload=payload)
    return reason
