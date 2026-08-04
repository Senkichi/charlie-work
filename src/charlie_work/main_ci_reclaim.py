"""Reclaim superseded, not-yet-started main-branch CI runs (issues #863, #815).

## Why this exists

``.github/workflows/reclaim-main-ci.yml`` cancels ``main`` CI runs that were
superseded by a later push but never got past ``queued``/``pending`` -- runs
that would otherwise sit occupying one of this repo's 2 registered
self-hosted runners (#799) testing a commit that can no longer appear in any
PR's check set. That workflow deadlocks under load (#863): it requires
``runs-on: self-hosted`` -- the exact scarce resource it exists to free -- and
combined with ``concurrency: cancel-in-progress: true`` on a shared group,
every queued reaper instance is killed by the *next* push's reaper before it
ever acquires a runner. #863's own evidence shows a manually-issued
``POST actions/runs/{id}/cancel`` succeeding while the workflow-based reaper
starved.

This module ports the same cancellation logic into the Python orchestrator,
which runs every fleet pass with local ``gh`` CLI access and needs no runner
at all -- so it cannot lose the race for the capacity it is trying to
reclaim. The workflow file is kept, not deleted (see its own header comment
and the PR that introduced this module): pushes to ``main`` also arrive from
Aviator's MergeQueue and direct human/operator merges, entirely independent
of whether the local supervisor is running, so the workflow remains the
fallback for the window when this pass cannot run at all. Its
``cancel-in-progress`` was flipped from ``true`` to ``false`` in the same
change, so an accumulating queue of reapers (bounded by push volume during
supervisor downtime, not by steady-state merge traffic) no longer
self-cannibalizes.

This also closes #815 (the reaper gets only one scheduling chance per main
push and can permanently lose the race to the stale run starting first): a
scheduled retry of the *workflow* would still compete for the starved
runners and gain nothing. Running this function on every fleet pass gives
repeated, runner-free attempts at the same still-superseded, still-not-started
run across passes -- exactly the repeated-chances property #815 asked for,
without the capacity problem that made its own proposed fix (adding a
``schedule:`` trigger to the workflow) ineffective.

## Safety invariant

Never cancel a run that has started, and never cancel the run for main's
current tip -- an in-progress main run is a per-commit historical/deployment
record, and main's tip must always be allowed to run to completion. Only a
run whose commit is a strict ancestor of main's current tip is a candidate,
and every candidate is re-checked immediately before cancellation to close
the list-then-cancel race window. This mirrors
``.github/workflows/reclaim-main-ci.yml`` field-for-field:

- ``_CANCELABLE_STATUSES`` mirrors the workflow's ``cancelable`` Set.
- The tip-sha comparison mirrors ``run.head_sha === context.sha``.
- The strict-ancestor check is *additional* to what the workflow does (the
  workflow only compares against the sha that triggered it, so every other
  ``main``-branch run it sees is implicitly assumed superseded by
  construction of the push event). Since this module runs independently of
  any specific push, ancestry must be verified explicitly rather than
  assumed.
- The pre-cancel re-fetch mirrors the workflow's ``getWorkflowRun`` re-check.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .github import GitHubLike, GitHubRunResult
from .subprocess_runner import run_captured

# Ported verbatim (semantics, not literal syntax) from reclaim-main-ci.yml:
#
#   // Defensively broad: GitHub's REST `status` field for a not-yet-started
#   // run has been observed as "queued" live in this repo; "pending"/
#   // "requested"/"waiting" are included because the exact value for every
#   // not-started state isn't documented as a single guaranteed string, and
#   // including one that never occurs is harmless -- listWorkflowRuns simply
#   // returns nothing in that status. Never include "in_progress" or
#   // "completed" here: that is the exemption #810 preserves.
_CANCELABLE_STATUSES = frozenset({"queued", "pending", "requested", "waiting"})

_GIT_TIMEOUT_SECONDS = 30
_FETCH_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class ReclaimedRun:
    """One run this pass actually cancelled."""

    run_id: int
    head_sha: str
    status_before_cancel: str
    created_at: str


@dataclass(frozen=True)
class MainCiReclaimResult:
    """Outcome of one ``reclaim_superseded_main_ci_runs`` call.

    ``ok`` is False only for a pass-level failure (fetch failed, tip
    unresolved, run listing failed) -- never raised, per this codebase's
    errors-as-values convention. A per-run cancel failure does not flip
    ``ok`` to False; it is recorded in ``cancel_errors`` (mirrors the
    workflow's own ``core.warning`` + continue on a failed
    ``cancelWorkflowRun`` call -- one uncancellable run must not abort the
    rest of the pass).
    """

    ok: bool
    error: str | None = None
    tip_sha: str | None = None
    candidates_checked: int = 0
    cancelled: tuple[ReclaimedRun, ...] = ()
    skipped_not_ancestor: int = 0
    skipped_started_before_cancel: int = 0
    cancel_errors: tuple[str, ...] = ()


def _object_exists(repo_root: Path, sha: str) -> bool:
    """Return True if ``sha`` names a commit object present in the local store.

    Gate ancestry questions on this first: ``git merge-base --is-ancestor``
    exits non-zero both for "not an ancestor" and for "I have never heard of
    that object" -- conflating the two would let a not-yet-fetched commit
    read as "not an ancestor, therefore safe to cancel" instead of "unknown,
    must skip". Mirrors ``worktree.py``'s ``_object_exists``; reimplemented
    locally rather than imported since that helper is module-private.
    """
    result = run_captured(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=repo_root,
        timeout_seconds=_GIT_TIMEOUT_SECONDS,
    )
    return result.ok


def _is_strict_ancestor(repo_root: Path, ancestor_sha: str, descendant_sha: str) -> bool:
    """Return True iff ``ancestor_sha`` is a *proper* ancestor of ``descendant_sha``.

    Equal shas are deliberately not "ancestor": ``git merge-base
    --is-ancestor`` treats a commit as its own ancestor, which would
    misclassify main's own tip as a cancellation candidate. The caller
    already excludes an exact tip-sha match directly (mirroring the
    workflow's ``head_sha === context.sha`` check), but this function's own
    name promises "strict", so it enforces the property independently
    rather than relying solely on the caller.

    Returns False -- never raises -- when either object is unknown to this
    checkout (see ``_object_exists``): an ancestry question about an object
    this repo has never fetched has no safe "yes" answer.
    """
    if ancestor_sha == descendant_sha:
        return False
    if not _object_exists(repo_root, ancestor_sha) or not _object_exists(
        repo_root, descendant_sha
    ):
        return False
    result = run_captured(
        ["git", "merge-base", "--is-ancestor", ancestor_sha, descendant_sha],
        cwd=repo_root,
        timeout_seconds=_GIT_TIMEOUT_SECONDS,
    )
    return result.ok


def reclaim_superseded_main_ci_runs(
    gh: GitHubLike,
    repo_root: Path,
    *,
    default_branch: str = "main",
    workflow_filename: str = "ci.yml",
    max_runs_scanned: int = 30,
) -> MainCiReclaimResult:
    """Cancel not-yet-started ``main``-branch CI runs superseded by the current tip.

    Never raises. A pass-level failure (fetch, tip resolution, run listing)
    returns ``ok=False`` with nothing cancelled -- the fail-safe direction is
    always "cancel nothing", never "guess and cancel". See module docstring
    for the full safety invariant and rationale.

    ``git fetch`` (no checkout, does not move HEAD -- safe to run against any
    checkout, including one a supervisor is actively using) is run first so
    the strict-ancestor check below has the objects it needs; without it, a
    commit pushed moments ago by Aviator or a direct merge would read as
    "unknown object" and every candidate would be skipped until some other
    codepath happens to fetch it.
    """
    fetch_result = run_captured(
        ["git", "fetch", "origin", default_branch],
        cwd=repo_root,
        timeout_seconds=_FETCH_TIMEOUT_SECONDS,
    )
    if not fetch_result.ok:
        return MainCiReclaimResult(
            ok=False,
            error=f"git fetch origin {default_branch} failed: {fetch_result.error or fetch_result.stderr}",
        )

    tip_commit = gh.commit(default_branch)
    tip_sha = tip_commit.get("sha") if isinstance(tip_commit, dict) else None
    if not isinstance(tip_sha, str) or not tip_sha:
        return MainCiReclaimResult(
            ok=False, error=f"failed to resolve current tip sha for {default_branch}"
        )
    if not _object_exists(repo_root, tip_sha):
        return MainCiReclaimResult(
            ok=False,
            error=f"tip sha {tip_sha} not present in local object store after fetch",
            tip_sha=tip_sha,
        )

    runs_result = gh.run(
        [
            "api",
            f"repos/{{owner}}/{{repo}}/actions/workflows/{workflow_filename}/runs"
            f"?branch={default_branch}&event=push&per_page={max_runs_scanned}",
        ],
        json_output=True,
        allow_failure=True,
    )
    if isinstance(runs_result, GitHubRunResult):
        if not runs_result.ok or not isinstance(runs_result.value, dict):
            return MainCiReclaimResult(
                ok=False,
                error=f"failed to list workflow runs: {runs_result.error}",
                tip_sha=tip_sha,
            )
        runs_payload = runs_result.value
    elif isinstance(runs_result, dict):
        runs_payload = runs_result
    else:
        return MainCiReclaimResult(
            ok=False,
            error=f"unexpected response listing workflow runs: {type(runs_result).__name__}",
            tip_sha=tip_sha,
        )

    workflow_runs = runs_payload.get("workflow_runs")
    if not isinstance(workflow_runs, list):
        workflow_runs = []

    candidates_checked = 0
    cancelled: list[ReclaimedRun] = []
    skipped_not_ancestor = 0
    skipped_started_before_cancel = 0
    cancel_errors: list[str] = []

    for run in workflow_runs:
        if not isinstance(run, dict):
            continue
        run_id = run.get("id")
        head_sha = run.get("head_sha")
        status = run.get("status")
        created_at = run.get("created_at", "")
        if not isinstance(run_id, int) or not isinstance(head_sha, str):
            continue
        # Never touch the run for main's current tip -- it must always be
        # left to run to completion (mirrors the workflow's
        # `head_sha === context.sha` check).
        if head_sha == tip_sha:
            continue
        if status not in _CANCELABLE_STATUSES:
            continue

        candidates_checked += 1
        if not _is_strict_ancestor(repo_root, head_sha, tip_sha):
            skipped_not_ancestor += 1
            continue

        # Re-fetch immediately before cancelling: the run may have moved
        # from queued/pending to in_progress (or completed) between the
        # list call above and now. Cancelling based on stale status could
        # kill a run that has since started -- exactly what must never
        # happen to a main run. Mirrors the workflow's getWorkflowRun
        # re-check.
        fresh_result = gh.run(
            ["api", f"repos/{{owner}}/{{repo}}/actions/runs/{run_id}"],
            json_output=True,
            allow_failure=True,
        )
        if isinstance(fresh_result, GitHubRunResult):
            fresh = fresh_result.value if fresh_result.ok else None
        elif isinstance(fresh_result, dict):
            fresh = fresh_result
        else:
            fresh = None
        fresh_status = fresh.get("status") if isinstance(fresh, dict) else None
        if fresh_status not in _CANCELABLE_STATUSES:
            skipped_started_before_cancel += 1
            continue

        cancel_result = gh.run(["run", "cancel", str(run_id)], allow_failure=True)
        if isinstance(cancel_result, GitHubRunResult):
            cancel_ok = cancel_result.ok
            cancel_error = cancel_result.error
        else:
            # dry-run short-circuit returns a truthy non-GitHubRunResult value
            cancel_ok = cancel_result is not None
            cancel_error = None
        if not cancel_ok:
            cancel_errors.append(f"run {run_id}: {cancel_error or 'cancel failed'}")
            continue

        cancelled.append(
            ReclaimedRun(
                run_id=run_id,
                head_sha=head_sha,
                status_before_cancel=str(fresh_status),
                created_at=str(created_at),
            )
        )

    return MainCiReclaimResult(
        ok=True,
        tip_sha=tip_sha,
        candidates_checked=candidates_checked,
        cancelled=tuple(cancelled),
        skipped_not_ancestor=skipped_not_ancestor,
        skipped_started_before_cancel=skipped_started_before_cancel,
        cancel_errors=tuple(cancel_errors),
    )
