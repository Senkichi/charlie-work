"""Queue-sync and backlog-blocker delegates moved out of ``OrchestratorApp``.

Track 2 Phase B, L03 (design doc
``docs/design/2026-09-04-orchestratorapp-mikado-graph-and-delegation-plan.md``,
Sections 3.1/3.2). Bodies relocated verbatim from ``charlie_work.workflow``;
``workflow_delegation._install_delegates`` re-attaches each ``def`` unwrapped
onto ``OrchestratorApp``.

``_fetch_commit_retrying`` and ``_fetch_compare_retrying`` are same-name thin
wrappers around the free functions of the same name re-exported by
``charlie_work.workflow``. They are reached through ``_wf.<name>`` rather than
imported here: a direct import would bind the name to *this module's own def*
and recurse. The ``_wf`` indirection also preserves the existing
``charlie_work.workflow`` monkeypatch seam for both wrappers.
"""

from __future__ import annotations

from typing import Any

import charlie_work.workflow as _wf
from charlie_work.backlog_reachability import (
    _get_open_blockers_for_issue,
    mention_scan_repo_context,
    scan_merged_pr_references,
)


def _merged_pr_referenced_issue_numbers(
    self,
    issues: list[dict[str, Any]],
    merged_prs: list[dict[str, Any]],
) -> tuple[
    set[int],
    set[int],
    set[int],
    dict[int, list[int]],
    dict[int, list[dict[str, Any]]],
]:
    """Thin wrapper for ``scan_merged_pr_references`` (issue #1337 moved
    the implementation to ``backlog_reachability.py`` so the mention-PR
    tracking it added does not grow the monolith past its ratchet
    high-water mark). See ``scan_merged_pr_references``'s docstring for
    the return-tuple semantics.

    Issue #1803: supplies the repo context the mention-qualifier check
    needs (dispatching repo slug + other managed repo names). Skipped
    when either input is empty — the scan's ready-set intersection makes
    the result empty anyway, so resolving ``name_with_owner`` (a gh
    call) would be pure waste, matching the issue #361 fetch guard.

    Consumers: ``dispatch_state`` (dispatch exclusion + mention flagging),
    ``compute_mention_coverage_map`` (the reachability classifier's
    mention-coverage arm), and ``state_merge_train``'s
    ``_finalize_externally_merged_issues`` (issue #1803 rework — the
    externally-merged finalization scan routes here so the temporal
    mergedAt-vs-createdAt check and the qualifier suppression cannot
    drift between the dispatch and finalize paths).
    """
    current_repo: str | None = None
    other_repo_names: frozenset[str] = frozenset()
    if issues and merged_prs:
        current_repo, other_repo_names = mention_scan_repo_context(
            self.gh, self.repo_root, self.fleet_dir_override
        )
    return scan_merged_pr_references(
        issues,
        merged_prs,
        self.config.dispatch.branch_prefix,
        current_repo=current_repo,
        other_repo_names=other_repo_names,
    )


def _get_open_blockers(self, issue: dict[str, Any]) -> tuple[list[int], list[int]]:
    """Check if an issue has any open blocker issues.

    Parses the issue body for blocker declarations and checks GitHub's
    native issue dependencies. Returns both declared blockers and open blockers.

    Args:
        issue: The issue dict from GitHub API

    Returns:
        Tuple of (declared_blockers, open_blockers). Both are lists of issue numbers.
        declared_blockers includes all blockers mentioned in the issue body or
        GitHub dependencies. open_blockers is the subset that are currently open.
    """
    return _get_open_blockers_for_issue(self.gh, issue)


def _fetch_commit_retrying(self, sha: str, *, leg: str) -> tuple[dict[str, Any] | None, str]:
    """Thin delegate to ``queue_sync_coverage._fetch_commit_retrying``.

    Kept as a bound method (rather than inlining the free-function call
    at each call site) so any existing or future monkeypatch of
    ``self._fetch_commit_retrying`` keeps working, mirroring
    ``_write_rework_prompt``'s wrapper shape.
    """
    return _wf._fetch_commit_retrying(self.gh, sha, leg=leg)


def _fetch_compare_retrying(
    self, base: str, head: str, *, leg: str
) -> tuple[dict[str, Any] | None, str]:
    """Thin delegate to ``queue_sync_coverage._fetch_compare_retrying``."""
    return _wf._fetch_compare_retrying(self.gh, base, head, leg=leg)
