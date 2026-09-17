"""Dispatch candidate sort / filter delegates for ``OrchestratorApp``.

Track 2 Phase B leaf L02 batch 4 (issue #1655, parent #1633, umbrella #1582).
Method bodies moved verbatim from ``OrchestratorApp`` in ``charlie_work.workflow``;
the ``workflow_delegation`` installer re-attaches each ``def`` onto the class.
"""

from __future__ import annotations

from typing import Any


def _sort_by_dependency_depth(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort unblocked candidates by out-degree (number of blocked dependents).

    Prioritizes issues that block the most downstream issues, so a wave
    drains the critical path and maximizes unblocking. Issues are sorted
    descending by their count of currently-blocked dependents, with
    creation date (oldest first) as a tiebreaker.

    This metric is computed against the full ready-labeled issue set
    before filtering, not just the unblocked candidates, to capture
    the true unblocking impact of each issue.

    Args:
        candidates: List of unblocked candidate issue dicts from GitHub API

    Returns:
        List of candidates sorted by out-degree (descending), then by creation date.
    """
    if not candidates:
        return []

    # Fetch the full set of ready-labeled issues to compute out-degree
    # We need issues that are blocked (not just candidates) to count dependents
    ready_issues = self.gh.issue_list(
        labels=[self.config.labels.ready],
        state="OPEN",
    )

    # Build reverse-adjacency map: blocker_number -> [dependents that are still blocked]
    blocker_to_dependents: dict[int, list[int]] = {}

    for issue in ready_issues:
        issue_number = int(issue["number"])
        declared_blockers, open_blockers = self._get_open_blockers(issue)

        # Only count dependents that are currently blocked (have open blockers)
        if not open_blockers:
            continue

        for blocker in declared_blockers:
            if blocker not in blocker_to_dependents:
                blocker_to_dependents[blocker] = []
            blocker_to_dependents[blocker].append(issue_number)

    # Compute out-degree for each candidate (number of blocked dependents)
    out_degree: dict[int, int] = {}
    for issue in candidates:
        issue_number = int(issue["number"])
        out_degree[issue_number] = len(blocker_to_dependents.get(issue_number, []))

    # Sort by out-degree (descending), then by creation date (ascending for oldest-first)
    def sort_key(issue: dict[str, Any]) -> tuple[int, str]:
        issue_number = int(issue["number"])
        # Use negative out_degree for descending sort
        degree = -out_degree.get(issue_number, 0)
        # Parse creation date; if missing, use high sentinel to sort last
        created_at = issue.get("createdAt", "9999-12-31T23:59:59Z")
        return (degree, created_at)

    return sorted(candidates, key=sort_key)


def _sort_by_dispatch_order(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort candidates by dispatch order (oldest-first or newest-first).

    Uses the createdAt field from GitHub API to sort by creation date.
    Default is oldest-first (ascending), but can be configured to newest-first
    (descending) via dispatch.order config.

    Args:
        candidates: List of candidate issue dicts from GitHub API

    Returns:
        Sorted list of candidates by creation date according to dispatch.order
    """
    if self.config.dispatch.order == "newest":
        # Sort by createdAt descending (newest first)
        return sorted(
            candidates,
            key=lambda issue: issue.get("createdAt", ""),
            reverse=True,
        )
    else:
        # Sort by createdAt ascending (oldest first, default)
        return sorted(
            candidates,
            key=lambda issue: issue.get("createdAt", ""),
            reverse=False,
        )


def _sort_review_queue_by_dependency_depth(
    self, queue: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Sort review queue so PRs blocking the most downstream work come first.

    Builds the same blocker->dependents graph used by worker dispatch
    against the currently-blocked ready-labeled issues. PRs whose linked
    issue is a blocker for more downstream issues are dispatched first,
    with PR number as a stable tiebreaker.
    """
    import logging

    logger = logging.getLogger(__name__)
    if not queue:
        return queue

    try:
        ready_issues = self.gh.issue_list(
            labels=[self.config.labels.ready],
            state="OPEN",
        )
        blocker_to_dependents: dict[int, list[int]] = {}
        for issue in ready_issues:
            issue_number = int(issue["number"])
            declared_blockers, open_blockers = self._get_open_blockers(issue)
            if not open_blockers:
                continue
            for blocker in declared_blockers:
                blocker_to_dependents.setdefault(blocker, []).append(issue_number)

        def sort_key(entry: dict[str, Any]) -> tuple[int, int]:
            return (
                -len(blocker_to_dependents.get(entry["issue"], [])),
                entry["pr"],
            )

        return sorted(queue, key=sort_key)
    except Exception:
        logger.warning(
            "Dependency depth sort failed; returning unsorted review queue",
            exc_info=True,
        )
        return queue


def _filter_blocked_issues(
    self, candidates: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[int, list[int]], dict[int, list[int]]]:
    """Filter out issues with open blockers from the candidate list.

    This is a shared helper used by both dry-run and real dispatch paths
    to ensure single-point-of-enforcement for the dependency gate logic.

    Args:
        candidates: List of candidate issue dicts from GitHub API

    Returns:
        Tuple of (filtered_candidates, blocked_issues, open_blockers_by_issue).
        filtered_candidates is the input list with blocked issues removed.
        blocked_issues maps blocked issue numbers to their full declared
        blocker list (open + closed) -- unchanged, this is the exact
        shape the dispatch_skip_blocked event payload has always used.
        open_blockers_by_issue maps the same issue numbers to only the
        currently-open subset: a closed blocker isn't actually blocking
        anymore, so it must not count when deciding whether every
        blocker of an issue is "dead" (see dispatch()'s blocked-chain
        attention check).
    """
    blocked_issues: dict[int, list[int]] = {}
    open_blockers_by_issue: dict[int, list[int]] = {}
    for issue in candidates:
        issue_number = int(issue["number"])
        declared_blockers, open_blockers = self._get_open_blockers(issue)
        if open_blockers:
            blocked_issues[issue_number] = declared_blockers
            open_blockers_by_issue[issue_number] = open_blockers

    filtered_candidates = [
        issue for issue in candidates if int(issue["number"]) not in blocked_issues
    ]
    return filtered_candidates, blocked_issues, open_blockers_by_issue
