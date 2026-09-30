"""PID-independent worker-handoff finalize (issue #1867).

Extracted from ``_detect_and_handle_orphaned_workers`` in ``workflow.py``:
a live worker PID is not itself proof of in-flight work. A worker that
pushed its branch and wrote a complete ``.worker-outcome.json``
(``push_succeeded: true`` / ``pr_created: false``) has finished the handoff
contract -- the file's own instruction is "then stop", so a still-running
PID at that point is a process that hung on exit, not a worker still doing
work. N4 (wf-review-opus.md): a live PID with a fresh, on-target, declared
push (rule 1's freshness gate: ``written_at`` after ``dispatched_at``)
routes immediately -- ``watchdog.worker_outcome_finalize_minutes`` no
longer gates this decision (FLIP 5); it only gates the stall watchdog's
separate kill decision elsewhere. This lane opens the PR from the worker's
drafted title/body through the same ``_open_pr_for_orphaned_branch`` the
dead-PID lane uses, without waiting for the PID to exit (the swole #163
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

   N4 (wf-review-opus.md): FLIP 5's "route immediately, ignore
   ``worker_outcome_finalize_minutes``" change reopened a version of that
   same round-2 cost concern from the other direction -- a single live PID
   with a fresh, on-target, declared-push outcome now keeps this
   function's result non-empty on every pass for as long as that PID stays
   alive (there is no threshold left to age it out of the candidate set),
   which defeats the early return and makes ``gh.pr_list()`` and
   ``state_lock`` run every pass until the PID exits, not just once near
   the threshold. Not fixed here -- reintroducing a wait window would
   partially undo the reviewed FLIP 5 latency fix this same lane exists
   for, so the right tradeoff (accept the per-pass cost, or find a
   cheaper filesystem-only staleness marker) is a design decision, not a
   review-fixes nit; see ``wf-review-dispositions.md``.
2. ``resolve_live_handoff_candidates`` -- run after ``gh.pr_list()`` (needed
   elsewhere in the sweep regardless, for the dead-PID lanes), to drop
   candidates a PR already exists for and attach the issue/label data the
   finalize step needs. Pre-lock, since it may call ``gh.issue_list()``.
3. ``finalize_live_handoff_candidates`` -- the in-lock step: opens the PR
   (or records a fingerprinted stranded-branch drift on failure) and
   updates ``state``/``sweep_events`` in place.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from . import worker_fate
from .config import WORKER_OUTCOME_FILENAME, OrchestratorConfig
from .github import GitHubLike, label_names
from .state import PASSIVE_OPEN_STATUS, utc_now
from .worktree import read_worker_outcome, worktree_path_for_branch


def partition_dispatched_by_pid_liveness(
    state: dict[str, Any],
    *,
    worker_pid_alive: Callable[[dict[str, Any]], bool],
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    """Split ``status: dispatched`` issues into dead-PID orphans and
    live-PID entries.

    ``worker_pid_alive`` is taken as a parameter (not imported here)
    deliberately: it must resolve through the caller's own module globals
    (``workflow._worker_pid_alive``) at call time, since that is the name
    the test suite patches to simulate PID liveness -- importing it
    directly into this module would make that patch a no-op.

    Issue #1867: ``live_pid_entries`` is kept separate from the dead-PID
    ``orphaned_issues`` on purpose -- every lane but the live-handoff
    finalize assumes the process has exited, and mixing live entries in
    would let the label-reclaim/redispatch-cap/drift lanes act on a
    running worker.
    """
    orphaned_issues: list[int] = []
    live_pid_entries: dict[int, dict[str, Any]] = {}
    for issue_number_str, entry in state.get("issues", {}).items():
        if not isinstance(entry, dict):
            continue
        if entry.get("status") != "dispatched":
            continue
        if worker_pid_alive(entry):
            live_pid_entries[int(issue_number_str)] = entry
        else:
            orphaned_issues.append(int(issue_number_str))
    return orphaned_issues, live_pid_entries


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
            outcome_mtime = datetime.fromtimestamp(
                (worktree_path / WORKER_OUTCOME_FILENAME).stat().st_mtime, tz=UTC
            )
        except OSError:
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
                    has_remote=True,
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
        # immediately -- `worker_outcome_finalize_minutes` only gates the
        # kill decision elsewhere, never this routing check.
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
    # Resolved through the workflow module object at call time (the
    # ``import charlie_work.workflow as _wf`` seam), so a test patching
    # ``charlie_work.workflow._open_pr_for_orphaned_branch`` also reaches this
    # lane -- the same reason ``worker_pid_alive`` is a parameter. Imported
    # here, not at module level: workflow.py imports this module at import time.
    import charlie_work.workflow as _wf

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
        pr_number, pr_error, _closing_ref = _wf._open_pr_for_orphaned_branch(
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
