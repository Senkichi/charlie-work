"""Closed-issue reclaim inside ``clean_worktrees`` (issue #2487).

The existing cleanup lanes are PR-gated: merged PR, or closed-unmerged PR.
A worktree whose linked issue is closed but which carries uncommitted or
untracked worker work -- or which never produced a PR link at all -- fails
both gates and skips forever, which is how the stale-CLI backlog
accumulated (issue #2487).

A closed GitHub issue is a terminal orchestrator decision: nothing will
ever advance it again, so waiting for "the work to become clean" or "a PR
to appear" can never resolve. This module is the third
``clean_worktrees`` lane, consulted only after the resolved-PR lanes have
declined a candidate and only when no PR resolved or the resolved PR was
itself confirmed terminal (``worktree_pr_lanes.pr_lane_verdict``): a
still-open or unconfirmable PR keeps the ordinary wait even on a closed
issue, because an issue can be closed by hand while its PR is still
under review. Once a live ``gh issue view`` confirms the linked issue is
CLOSED, the worktree is reclaimed

1. after the same live-session gate every other lane uses
   (``_cleanup_live_writer_reason`` -- positive evidence only), then
2. by first capturing any uncommitted worker-authored work to a
   ``refs/charlie/rescue/`` ref (the existing issue #849 capture -- it
   stages tracked edits, deletions, and untracked files while excluding
   orchestrator scaffolding), then
3. by removing the worktree with ``remove_worktree(force=True)`` while
   *keeping* the branch, so commits that were never pushed survive.

The capture-then-remove ordering is what makes the sweep strictly
lossless: the content lives in the kept branch ref plus the rescue ref
even after the checkout is gone. Issue state -- never worktree
cleanliness -- is the discriminator: a clean worktree on an open issue
stays skipped; only a closed issue escalates to reclaim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .github import GitHubRunResult, WORKTREE_ISSUE_STATE_FIELDS
from .worktree_pr_lookup import WorktreeCleanGH

if TYPE_CHECKING:
    from .config import OrchestratorConfig
    from .worktree import RescueCapture

#: Result buckets matching ``clean_worktrees``'s result lists.
LaneBucket = Literal["removed", "planned", "skipped", "failed"]


@dataclass(frozen=True)
class ClosedIssueReclaim:
    """Outcome of ``reclaim_closed_issue_worktree`` for one worktree.

    ``bucket`` names the ``clean_worktrees`` result list the ``entry``
    belongs in; ``entry`` is the observability record (skip reasons stay
    human-readable; ``rescue_ref`` is set only when a capture happened).
    """

    bucket: LaneBucket
    entry: dict[str, Any]


def issue_confirmed_closed(gh: WorktreeCleanGH, issue_number: int) -> bool:
    """True only when a live ``gh issue view`` confirms the issue is CLOSED.

    A local-backend ``gh.run`` that fails closed (``ok=False``), an
    unreachable/unsupported issue view, and any transport exception all
    return False, so a worktree whose issue state cannot be confirmed
    closed falls through to the existing PR-gated lanes unchanged.
    """
    try:
        result = gh.run(
            ["issue", "view", str(issue_number), "--json", WORKTREE_ISSUE_STATE_FIELDS],
            json_output=True,
            allow_failure=True,
        )
    except Exception:
        # e.g. an unmapped argv raising GitHubError before allow_failure
        # applies, or a test double that raises rather than returning
        # GitHubRunResult -- the sweep must survive it.
        return False
    return bool(
        isinstance(result, GitHubRunResult)
        and result.ok
        and isinstance(result.value, dict)
        and result.value.get("state") == "CLOSED"
    )


def reclaim_closed_issue_worktree(
    repo_root: Path,
    worktree_path: Path,
    branch: str,
    issue_number: int,
    issue_state: dict[str, Any],
    pr_number: int | None,
    config: OrchestratorConfig,
    *,
    dry_run: bool,
) -> ClosedIssueReclaim:
    """Reclaim one worktree whose linked issue ``gh issue view`` confirmed CLOSED.

    Order is deliberate and mirrors the issue #2487 proposal:

    1. The live-session gate runs FIRST -- a live worker must never lose
       its checkout, and the dirty probe / capture below would touch a
       live worker's index.
    2. Worker-authored uncommitted or untracked content is captured to a
       rescue ref before anything is removed. A failed capture keeps the
       skip -- the work must be durable first.
    3. ``remove_worktree(force=True)`` removes the tree; the branch is
       deliberately NOT passed so unpushed commits survive.
    """
    # Deferred: worktree.py imports this module at top level, so its
    # helpers are reached only after it has finished initializing.
    from .worktree import (
        WorktreeProbeFailedError,
        _capture_worktree_work_to_rescue_ref,
        _cleanup_live_writer_reason,
        _worktree_dirty_reason,
        remove_worktree,
    )

    entry: dict[str, Any] = {
        "worktree": str(worktree_path),
        "branch": branch,
        "issue_number": issue_number,
        "pr_number": pr_number,
        "closed_issue": True,
    }
    live_reason = _cleanup_live_writer_reason(issue_state, worktree_path)
    if live_reason:
        return ClosedIssueReclaim(
            "skipped", {**entry, "reason": f"live worker detected: {live_reason}"}
        )
    try:
        dirty_reason = _worktree_dirty_reason(
            worktree_path,
            config.dispatch.injected_paths,
            config.dispatch.materialize_dirs,
        )
    except WorktreeProbeFailedError as exc:
        # Cannot prove the tree clean -- treat it as dirty: a capture attempt
        # preserves whatever is there, and its failure fails closed.
        dirty_reason = f"worktree status probe failed: {exc}"
    if dry_run:
        return ClosedIssueReclaim("planned", {**entry, "uncommitted_work": dirty_reason or ""})
    rescue_ref: str | None = None
    if dirty_reason:
        capture = _capture_worktree_work_to_rescue_ref(
            repo_root,
            worktree_path,
            issue_number,
            config.dispatch.injected_paths,
            config.dispatch.materialize_dirs,
        )
        if capture.error is not None or capture.ref_name is None:
            return ClosedIssueReclaim(
                "skipped",
                {
                    **entry,
                    "reason": f"closed issue; {dirty_reason}; rescue capture failed: "
                    f"{capture.error or 'no ref produced'}",
                },
            )
        rescue_ref = capture.ref_name
        _emit_rescue_event(repo_root, config, capture, issue_number, worktree_path, dirty_reason)
    if not remove_worktree(repo_root, worktree_path, force=True):
        return ClosedIssueReclaim(
            "failed", {**entry, "rescue_ref": rescue_ref, "reason": "remove_worktree failed"}
        )
    return ClosedIssueReclaim("removed", {**entry, "rescue_ref": rescue_ref})


def _emit_rescue_event(
    repo_root: Path,
    config: OrchestratorConfig,
    capture: RescueCapture,
    issue_number: int,
    worktree_path: Path,
    reason: str,
) -> None:
    """Best-effort ``worktree_rescue_captured`` event -- same payload the
    create_worktree redispatch lane emits. The rescue ref is the durable
    artifact; instrumentation failures are dropped silently.
    """
    try:
        from .instrumentation import log_event
        from .paths import runtime_paths

        state_file = runtime_paths(repo_root, config.runtime.state_dir).state_file
        # write-gate-exempt(issue=2487): clean_worktrees has no write_gate receiver (state_maintenance calls it with gh/config only); same best-effort event disposition as create_worktree's emitter.
        log_event(
            state_file,
            "worktree_rescue_captured",
            {
                "issue_number": issue_number,
                "rescue_ref": capture.ref_name,
                "commit_sha": capture.commit_sha,
                "worktree_path": str(worktree_path),
                "reason": reason,
            },
        )
    except Exception:  # noqa: BLE001 -- instrumentation is best-effort
        pass
