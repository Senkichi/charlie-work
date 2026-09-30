"""Shared evidence builders for the ``worker_fate`` decision-table tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


from charlie_work.worker_fate import (
    BranchEvidence,
    EvidenceSource,
    FailureEvidence,
    FateEvidence,
    OutcomeEvidence,
    TerminalEvidence,
)
from charlie_work.worker import WorkerHealth

DISPATCHED = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
AFTER = DISPATCHED + timedelta(minutes=5)
BEFORE = DISPATCHED - timedelta(minutes=5)
NOW = DISPATCHED + timedelta(minutes=10)


def _branch(
    *,
    remote_head_sha: str | None = None,
    remote_ahead: int | None = None,
    unpushed: int | None = None,
    local_ahead: int | None = None,
    open_pr_number: int | None = None,
    pr_known: bool = False,
) -> BranchEvidence:
    return BranchEvidence(
        remote_head_sha=remote_head_sha,
        remote_ahead=remote_ahead,
        unpushed=unpushed,
        local_ahead=local_ahead,
        open_pr_number=open_pr_number,
        pr_known=pr_known,
    )


def _outcome(
    *,
    source: EvidenceSource = EvidenceSource.WORKTREE,
    written_at: datetime | None = AFTER,
    outcome: str | None = None,
    push_succeeded: bool | None = None,
    pr_created: bool | None = None,
    head_sha: str | None = None,
    raw: dict | None = None,
) -> OutcomeEvidence:
    return OutcomeEvidence(
        source=source,
        written_at=written_at,
        outcome=outcome,
        push_succeeded=push_succeeded,
        pr_created=pr_created,
        head_sha=head_sha,
        raw=raw if raw is not None else {},
    )


def _terminal(
    *, ended_at: datetime, exit_code: int | None = None, outcome: OutcomeEvidence | None = None
) -> TerminalEvidence:
    return TerminalEvidence(ended_at=ended_at, exit_code=exit_code, outcome=outcome)


def _evidence(
    *,
    issue_number: int = 1,
    adapter: str = "claude-code",
    dispatched_at: datetime | None = DISPATCHED,
    pid_alive: bool = False,
    health: WorkerHealth | None = None,
    terminal=None,
    worktree_outcome: OutcomeEvidence | None = None,
    branch: BranchEvidence | None = None,
    failure: FailureEvidence | None = None,
) -> FateEvidence:
    return FateEvidence(
        issue_number=issue_number,
        adapter=adapter,
        dispatched_at=dispatched_at,
        pid_alive=pid_alive,
        health=health,
        terminal=terminal,
        worktree_outcome=worktree_outcome,
        branch=branch if branch is not None else _branch(),
        failure=failure,
    )
