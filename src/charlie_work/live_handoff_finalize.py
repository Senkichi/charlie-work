"""PID-independent worker-handoff finalize (issue #1867).

Extracted from ``_detect_and_handle_orphaned_workers`` in ``workflow.py``:
a live worker PID is not itself proof of in-flight work. A worker that
pushed its branch and wrote a complete ``.worker-outcome.json``
(``push_succeeded: true`` / ``pr_created: false``) has finished the handoff
contract -- the file's own instruction is "then stop", so a still-running
PID at that point is a process that hung on exit, not a worker still doing
work. When the outcome file is older than
``watchdog.worker_outcome_finalize_minutes``, this lane opens the PR from
the worker's drafted title/body through the same ``_open_pr_for_orphaned_branch``
the dead-PID lane uses, without waiting for the PID to exit (the swole #163
incident: a completed worker left its PR unopened ~2h because every
finalize path keyed off PID death). Only the PR-open action applies to a
live PID -- none of the dead-PID lanes (label reclaim, escalation,
redispatch cap, drift) do, since those assume the process has exited; the
stall watchdog remains responsible for reaping the hung process itself on
its own cadence.

This module holds three call sites, run by ``_detect_and_handle_orphaned_workers``
in strict order:

1. ``collect_stale_live_handoff_pids`` -- pure filesystem/state work, no
   GitHub call, no lock. Run BEFORE the function's early-return check so a
   pass with no dead-PID orphan and no stale live-handoff candidate can
   still bail out without calling ``gh.pr_list()`` or taking ``state_lock``
   (the round-2 review finding: relaxing the early return to cover any live
   PID, not just a stale one, made the network fetch and lock fire on
   nearly every pass of an active fleet).
2. ``resolve_live_handoff_candidates`` -- run after ``gh.pr_list()`` (needed
   elsewhere in the sweep regardless, for the dead-PID lanes), to drop
   candidates a PR already exists for and attach the issue/label data the
   finalize step needs. Pre-lock, since it may call ``gh.issue_list()``.
3. ``finalize_live_handoff_candidates`` -- the in-lock step: opens the PR
   (or records a fingerprinted stranded-branch drift on failure) and
   updates ``state``/``sweep_events`` in place.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from .config import WORKER_OUTCOME_FILENAME, OrchestratorConfig
from .dead_worker_reap import _open_pr_for_orphaned_branch
from .github import GitHubLike, label_names
from .state import PASSIVE_OPEN_STATUS, utc_now
from .worktree import read_worker_outcome, worktree_path_for_branch


def collect_stale_live_handoff_pids(
    live_pid_entries: dict[int, dict[str, Any]],
    *,
    worker_outcome_finalize_minutes: int,
    repo_root: Path | None,
    worktrees_dir: Path | None,
    now: datetime,
) -> dict[int, dict[str, Any]]:
    """Filesystem-only pre-check: which live-PID entries look finalizable.

    Returns candidates keyed by issue number, carrying enough to finalize
    once the issue/label data is attached by ``resolve_live_handoff_candidates``.
    Does not consult ``pr_by_issue`` -- that requires ``gh.pr_list()``, which
    callers gate on this function's result being non-empty in the first
    place, so it cannot be a precondition here.
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
            outcome_age = now - datetime.fromtimestamp(
                (worktree_path / WORKER_OUTCOME_FILENAME).stat().st_mtime, tz=UTC
            )
        except OSError:
            continue
        if outcome_age <= timedelta(minutes=worker_outcome_finalize_minutes):
            continue
        worker_outcome = read_worker_outcome(worktree_path)
        if not (
            isinstance(worker_outcome, dict)
            and worker_outcome.get("push_succeeded") is True
            and worker_outcome.get("pr_created") is False
        ):
            continue
        candidates[issue_number] = {
            "branch": branch,
            "worktree_path": worktree_path,
            "worker_outcome": worker_outcome,
            "worker_pid": live_entry.get("worker_pid"),
            "outcome_age_minutes": outcome_age.total_seconds() / 60,
        }
    return candidates


def resolve_live_handoff_candidates(
    stale_candidates: dict[int, dict[str, Any]],
    *,
    pr_by_issue: dict[int, dict[str, Any]],
    issues_by_number: dict[int, dict[str, Any]],
    gh: GitHubLike,
    config: OrchestratorConfig,
) -> dict[int, dict[str, Any]]:
    """Drop candidates a PR already exists for and attach issue/label data.

    ``issues_by_number`` is shared with the caller's dead-PID lane and is
    mutated in place (populated via a single bulk ``gh.issue_list()`` call)
    so a pass that already fetched it for the dead-PID lane does not fetch
    it again.
    """
    live_handoff_candidates: dict[int, dict[str, Any]] = {}
    declared = {
        issue_number: candidate
        for issue_number, candidate in stale_candidates.items()
        if issue_number not in pr_by_issue
    }
    if not declared:
        return live_handoff_candidates

    if not issues_by_number:
        for issue in gh.issue_list(state="open"):
            number = issue.get("number")
            if number is not None:
                issues_by_number[int(number)] = issue

    for issue_number, candidate in declared.items():
        issue = issues_by_number.get(issue_number)
        if issue is None:
            # Issue closed or inaccessible -- never open a PR for it.
            continue
        issue_labels = label_names(issue)
        candidate["issue"] = issue
        candidate["issue_labels"] = issue_labels
        candidate["active_labels"] = issue_labels & config.labels.active
        live_handoff_candidates[issue_number] = candidate
    return live_handoff_candidates


def finalize_live_handoff_candidates(
    *,
    gh: GitHubLike,
    config: OrchestratorConfig,
    repo_root: Any,
    state: dict[str, Any],
    state_file: Path,
    live_handoff_candidates: dict[int, dict[str, Any]],
    pr_by_issue: dict[int, dict[str, Any]],
    sweep_events: list[tuple[str, dict[str, Any]]],
    drift_fingerprint: Callable[..., str],
) -> None:
    """In-lock finalize: open the PR, or record a stranded-branch drift.

    Mutates ``state["issues"]`` and appends to ``sweep_events`` in place.
    Only the PR-open action runs here -- none of the dead-PID lanes (label
    reclaim, escalation, redispatch cap, drift) apply to a process that has
    not exited; the stall watchdog remains responsible for reaping the hung
    process on its own cadence.
    """
    for issue_number in live_handoff_candidates:
        entry = state["issues"].get(str(issue_number), {})
        if not isinstance(entry, dict):
            continue
        # Re-verify status (state may have changed between lock windows).
        if entry.get("status") != "dispatched":
            continue
        # A PR may have appeared between the pre-lock snapshot and this
        # lock -- never open a duplicate.
        if issue_number in pr_by_issue:
            continue
        candidate = live_handoff_candidates[issue_number]
        # ``repo_root`` comes from ``getattr(gh, "repo_root", None)`` and is
        # not statically typed; narrow to ``Path | None`` before the salvage
        # helper (``None`` is an error value it already handles).
        salvage_repo_root = repo_root if isinstance(repo_root, Path) else None
        pr_number, pr_error, _closing_ref = _open_pr_for_orphaned_branch(
            gh=gh,
            config=config,
            repo_root=salvage_repo_root,
            branch=candidate["branch"],
            base_ref=config.dispatch.base_ref,
            issue_number=issue_number,
            active_labels=candidate["active_labels"],
            issue_labels=candidate["issue_labels"],
            issue_title=(candidate["issue"] or {}).get("title"),
            state_file=state_file,
            worker_outcome=candidate["worker_outcome"],
        )
        if pr_number is not None:
            entry["status"] = PASSIVE_OPEN_STATUS
            entry["pr_number"] = pr_number
            # Same honest-handoff kind as the dead-PID lane's confirmed-
            # outcome case; the payload additionally records that the worker
            # PID was still running at finalize time (the "log line noting
            # the PID was still running" the issue asks for), plus the
            # outcome-file age that tripped the finalize threshold.
            sweep_events.append(
                (
                    "worker_handoff_pr_opened",
                    {
                        "issue_number": issue_number,
                        "pr_number": pr_number,
                        "branch_name": candidate["branch"],
                        "worker_reported": True,
                        "previous_status": "dispatched",
                        "reason": "worker_outcome_stale_live_pid",
                        "worker_pid": candidate["worker_pid"],
                        "worker_pid_still_running": True,
                        "outcome_age_minutes": round(candidate["outcome_age_minutes"], 1),
                        "label_write_ok": pr_error is None,
                        "pr_error": pr_error,
                    },
                )
            )
            state["issues"][str(issue_number)] = entry
            continue

        # PR creation failed after the bounded retry -- the branch is pushed
        # and stranded. Same fingerprinted-drift pattern as the dead-PID
        # lane: emit the drift event once per unchanged finding while the
        # next pass re-attempts the create itself.
        fingerprint = drift_fingerprint(
            reason="live_worker_handoff_pr_create_failed",
            branch_name=candidate["branch"],
            error=pr_error or "unknown",
        )
        if entry.get("orphan_drift_fingerprint") == fingerprint:
            state["issues"][str(issue_number)] = entry
            continue
        entry["orphan_drift_fingerprint"] = fingerprint
        entry["orphan_drift_at"] = utc_now()
        sweep_events.append(
            (
                "pr_create_failed_branch_stranded",
                {
                    "issue_number": issue_number,
                    "branch_name": candidate["branch"],
                    "previous_status": "dispatched",
                    "reason": "live_worker_handoff_pr_create_failed",
                    "pr_create_error": pr_error,
                    "worker_reported": True,
                    "worker_pid": candidate["worker_pid"],
                    "worker_pid_still_running": True,
                },
            )
        )
        state["issues"][str(issue_number)] = entry
