"""Pre-salvage supersession check: skip vestigial salvage PRs for already-landed work (issue #1241).

Single enforcement point for the two salvage lanes (``workflow._attempt_salvage``
and ``reconcile.apply_fixes``'s ``session_unpublished_work_salvaged`` branch).
``workflow`` imports ``reconcile``, so ``reconcile`` cannot import
``workflow._salvage_already_landed`` back without a cycle; this module is the
shared dependency both can import. It owns the live re-check that fires before
a salvage PR is opened: a dead session's snapshot (issue, pr_number, branch)
can be stale by the time staleness trips -- an operator or sibling worker can
have pushed the stranded commit and merged it (closing the issue) inside the
staleness window. Salvage must re-check LIVE terminal state at fire time
instead of trusting the snapshot, or it opens a duplicate PR for already-merged
work (the #1241 incident: PR #1677 opened three minutes after PR #1673 merged).

The check is fail-open: only POSITIVE evidence (issue closed / a merged PR /
commits reachable from main / empty diff) skips salvage. A failed GitHub search
or git fetch falls through to opening the PR -- a duplicate PR is recoverable,
silently dropped work is not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import OrchestratorConfig
from .github import GitHubLike
from .worktree import salvage_branch_empty_diff, salvage_branch_reachable_from_main

# Skip reasons. ``issue_closed`` / ``pr_merged`` / ``empty_diff`` are the
# #1221 checks; ``commits_reachable`` is the #1241 reachability check. The
# event kind a skip emits is derived from the reason via
# ``salvage_skip_event_kind`` so both lanes agree on the observable signal.
REASON_ISSUE_CLOSED = "issue_closed"
REASON_PR_MERGED = "pr_merged"
REASON_EMPTY_DIFF = "empty_diff"
REASON_COMMITS_REACHABLE = "commits_reachable"

# #1221's existing observable skip event. Kept for back-compat: tests and the
# mop-up report already consume it for the issue-closed / merged-PR / empty-diff
# reasons. The #1241 reachability skip emits ``salvage_skipped_superseded``.
SKIP_EVENT_ALREADY_LANDED = "salvage_skipped_already_landed"
SKIP_EVENT_SUPERSEDED = "salvage_skipped_superseded"


def salvage_skip_event_kind(reason: str | None) -> str:
    """Return the event ``kind`` for a salvage skip with the given ``reason``.

    Single source of truth for the skip-event-name mapping so the two salvage
    lanes emit the same observable signal for the same reason. The #1241
    reachability skip gets its own event (``salvage_skipped_superseded``) per
    the issue spec; the #1221 reasons keep their existing event so current
    consumers and tests are undisturbed.
    """
    if reason == REASON_COMMITS_REACHABLE:
        return SKIP_EVENT_SUPERSEDED
    return SKIP_EVENT_ALREADY_LANDED


def check_salvage_superseded(
    *,
    gh: GitHubLike,
    config: OrchestratorConfig,
    repo_root: Path,
    branch: str,
    base_ref: str,
    issue_number: int,
    issue: dict[str, Any] | None = None,
) -> tuple[bool, str | None]:
    """Return ``(superseded, reason)`` if salvage should be skipped.

    Re-checks LIVE terminal state before opening a salvage PR. Any one of these
    means a salvage PR would be vestigial (a duplicate of already-landed work):

    1. the linked issue is CLOSED. ``issue`` may be the caller's already-fetched
       ``gh.issue_view`` result (one call, already made -- the workflow lane);
       when ``None`` (the reconcile lane, which had not fetched) the issue is
       fetched here so the closed-issue check still fires.
    2. a PR binding to this issue is MERGED (``gh.merged_prs_for_issue``). A
       failed search (``ok=False``) is treated as "unknown" and falls through;
       a human reviews salvage PRs anyway.
    3. the branch's tree is identical to current main's tree
       (``salvage_branch_empty_diff``) -- belt-and-suspenders for a squash-merge
       that closed the issue but whose PR search lags.
    4. the branch's tip is already reachable from origin/main
       (``salvage_branch_reachable_from_main``, issue #1241) -- catches a merge
       commit whose tree differs from the salvage head's tree (main advanced
       with other commits) which the empty-diff check misses.

    ``reason`` is a short string identifying which check fired, recorded in the
    skip event for diagnosis. Fails safe (returns ``(False, None)``) on any git
    or GitHub error -- only positive evidence skips.
    """
    # (1) Issue closed. Fetch live state when the caller did not supply it.
    if issue is None:
        try:
            issue = gh.issue_view(issue_number)
        except Exception:
            issue = None
    if issue is not None and str(issue.get("state") or "").upper() == "CLOSED":
        return True, REASON_ISSUE_CLOSED

    # (2) A merged PR binds to this issue.
    merged = gh.merged_prs_for_issue(issue_number, config.dispatch.branch_prefix)
    if getattr(merged, "ok", True) and len(merged) > 0:
        return True, REASON_PR_MERGED

    # (3) The branch's tree is identical to current main's tree.
    if salvage_branch_empty_diff(repo_root, branch, base_ref):
        return True, REASON_EMPTY_DIFF

    # (4) The branch's tip is already reachable from origin/main (#1241).
    if salvage_branch_reachable_from_main(repo_root, branch, base_ref):
        return True, REASON_COMMITS_REACHABLE

    return False, None
