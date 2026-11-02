"""Live-PID handoff pre-check for the dead-worker sweep (issue #1867).

A live worker PID is not itself proof of in-flight work. A worker that
pushed its branch and wrote a complete ``.worker-outcome.json``
(``push_succeeded: true`` / ``pr_created: false``) has finished the handoff
contract, so a still-running PID at that point is a process that hung on
exit. N4 (wf-review-opus.md): a live PID with a fresh, on-target, declared
push (rule 1's freshness gate: ``written_at`` after ``dispatched_at``)
routes immediately; ``watchdog.worker_outcome_finalize_minutes`` is only the
``<= 0`` kill switch for this lane.

``collect_stale_live_handoff_pids`` is the filesystem-only step, run before
the sweep's early-return check so a pass with no dead-PID orphan and no stale
candidate bails out without ``gh.pr_list()`` or ``state_lock`` (the round-2
review finding). The PR-open and marker stamping live in
``decide_no_pr`` / ``apply_requests_pre``. An entry whose
``live_handoff_routed_outcome_at`` marker equals the current outcome file's
mtime was already routed and is skipped (N4); a newer outcome write re-arms
the lane.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from .. import worker_fate
from ..blocked_worker_escalation import is_exempt_blocked
from ..config import WORKER_OUTCOME_FILENAME
from ..worktree import read_worker_outcome, worktree_path_for_branch

# N4: the entry field recording which outcome-file mtime (ISO) this lane has
# already routed. Cleared with the dispatch epoch (``state.
# clear_dead_worker_failure_kind``) and on operator re-arm
# (``UNESCALATE_ISSUE_RESET_FIELDS``).
ROUTED_OUTCOME_KEY = "live_handoff_routed_outcome_at"


def collect_stale_live_handoff_pids(
    live_pid_entries: dict[int, dict[str, Any]],
    *,
    worker_outcome_finalize_minutes: int,
    repo_root: Path | None,
    worktrees_dir: Path | None,
    now: datetime,
    sessions_dir: Path,
    on_fate: Callable[[worker_fate.WorkerFate], None] | None = None,
) -> dict[int, dict[str, Any]]:
    """Filesystem-only pre-check: which live-PID entries look finalizable.

    Returns candidates keyed by issue number, carrying enough to finalize
    once the sweep attaches the issue/label data (``apply_requests_pre``).
    Does not consult ``pr_by_issue`` -- that requires ``gh.pr_list()``, which
    callers gate on this function's result being non-empty in the first
    place, so it cannot be a precondition here.

    An entry whose ``live_handoff_routed_outcome_at`` marker equals the
    current outcome file's mtime was already routed and is skipped (N4).
    ``on_fate`` receives each resolved fate so the caller can report rule 1's
    stale evidence (``worker_fate.report_stale_evidence``, B6) -- this
    function is lock-free and emits nothing itself.
    """
    candidates: dict[int, dict[str, Any]] = {}
    if (
        not live_pid_entries
        or worker_outcome_finalize_minutes <= 0
        or repo_root is None
        or worktrees_dir is None
    ):
        return candidates

    # Outcome checks first: pure filesystem work with no GitHub cost, and the
    # overwhelmingly common case (a live worker still mid-task) has no
    # outcome file at all.
    for issue_number, live_entry in live_pid_entries.items():
        branch = live_entry.get("branch_name")
        if not branch:
            continue
        worktree_path = worktree_path_for_branch(repo_root, branch, worktrees_dir)
        try:
            outcome_mtime = datetime.fromtimestamp(
                (worktree_path / WORKER_OUTCOME_FILENAME).stat().st_mtime, tz=UTC
            )
        except OSError:
            continue
        outcome_at = outcome_mtime.isoformat()
        if live_entry.get(ROUTED_OUTCOME_KEY) == outcome_at:
            continue
        outcome_age = now - outcome_mtime
        worker_outcome = read_worker_outcome(worktree_path)

        # Obtain this worker's fate from the module (design doc §8, step B4):
        # a live PID with no remote/PR data available at this pure-filesystem
        # stage can only resolve to `Live` (row 5) -- `resolve_fate` cannot
        # independently confirm a push without the `ls-remote`/`gh pr_list`
        # this function deliberately avoids paying for on every pass (see the
        # module docstring). The routing decision below stays on the
        # self-reported claim the legacy code already trusted.
        outcome_evidence = worker_fate.OutcomeEvidence(
            source=worker_fate.EvidenceSource.WORKTREE,
            written_at=outcome_mtime,
            outcome=worker_outcome.get("outcome") if isinstance(worker_outcome, dict) else None,
            push_succeeded=(
                worker_outcome.get("push_succeeded") if isinstance(worker_outcome, dict) else None
            ),
            pr_created=(
                worker_outcome.get("pr_created") if isinstance(worker_outcome, dict) else None
            ),
            head_sha=worker_outcome.get("head_sha") if isinstance(worker_outcome, dict) else None,
            raw=worker_outcome if isinstance(worker_outcome, dict) else {},
        )
        fate = worker_fate.resolve_fate(
            worker_fate.FateEvidence(
                issue_number=issue_number,
                adapter=live_entry.get("adapter") or "unknown",
                dispatched_at=worker_fate.parse_iso_timestamp(live_entry.get("dispatched_at")),
                pid_alive=True,
                health=None,
                terminal=None,
                worktree_outcome=outcome_evidence,
                branch=worker_fate.BranchEvidence(
                    remote_head_sha=None,
                    remote_ahead=None,
                    unpushed=None,
                    open_pr_number=None,
                    pr_known=False,
                ),
                failure=None,
            ),
            now=now,
        )
        if on_fate is not None:
            on_fate(fate)

        # Rule 1: freshness is gated on `dispatched_at`/`head_sha`, not on
        # outcome-age-vs-`now` -- a leftover outcome file from a prior
        # dispatch of this branch is rejected once, by evidence, rather
        # than by an elapsed-time proxy. `resolve_fate` itself can only
        # ever return `Live` here (row 5): with no remote read at this
        # pure-filesystem stage, `_is_pushed` can never confirm the push,
        # so rows 2-4 (`PushedWithoutPr`/`Completed`) are unreachable from
        # this call site's evidence shape. What we actually need is
        # `resolve_fate`'s freshness *step* (rule 1), which survives onto
        # `fate.basis.outcome`: the winning candidate if it passed the
        # `dispatched_at`/`head_sha` check, else `None`. The routing
        # decision below reads that rule-1-gated candidate's self-reported
        # claim directly -- the legacy code trusted the same claim, just
        # without the freshness gate in front of it.
        # Rule 5: a live PID with a fresh declared-push claim routes
        # immediately -- `worker_outcome_finalize_minutes` is only the
        # `<= 0` kill switch checked above, never an age threshold.
        #
        # B9 (wf-review-opus.md): the legacy check required an EXPLICIT
        # ``pr_created is False`` before routing. The worker-fate refactor
        # loosened this to ``pr_created is not True``, which also admits
        # ``None`` -- an outcome that omits ``pr_created`` entirely (an
        # incomplete/ambiguous self-report) now gets finalized as if it had
        # explicitly declared no PR was created. That widening was never
        # one of the nine reviewed flips; restore the strict legacy gate.
        fresh_outcome = fate.basis.outcome
        if (
            fresh_outcome is None
            # Rule 2: a blocked declaration wins everywhere -- never finalize
            # it as a handoff, whatever push flags ride along. Issue #2010: a
            # headless permission-denial "blocked" is exempt (a worker-config
            # defect, not a blocked task) and is routed as on origin/main.
            or (
                isinstance(fate, worker_fate.Blocked)
                and not is_exempt_blocked(
                    fate, sessions_dir=sessions_dir, issue_number=issue_number
                )
            )
            or fresh_outcome.push_succeeded is not True
            or fresh_outcome.pr_created is not False
        ):
            continue
        candidates[issue_number] = {
            "branch": branch,
            "worktree_path": worktree_path,
            "worker_outcome": worker_outcome,
            "worker_pid": live_entry.get("worker_pid"),
            "outcome_age_minutes": outcome_age.total_seconds() / 60,
            "outcome_at": outcome_at,
        }
    return candidates
