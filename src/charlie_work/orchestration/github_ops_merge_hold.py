"""Human-merge-hold check delegates moved out of ``OrchestratorApp``.

Track 2 Phase B, L04 batch 1 (design doc
``docs/design/2026-09-04-orchestratorapp-mikado-graph-and-delegation-plan.md``,
Sections 3.1/3.2). Bodies relocated verbatim from ``charlie_work.workflow``;
``workflow_delegation._install_delegates`` re-attaches each top-level ``def``
unwrapped onto ``OrchestratorApp``. ``linked_issue_number`` and
``_authorized_override_matches`` are reached through ``_wf.``:
``linked_issue_number`` is patched on ``charlie_work.workflow`` by the suite
(Tier D, via the ``workflow_mod`` alias), and ``_authorized_override_matches``
is a ``charlie_work.workflow`` module-level def. All other free names are
imported directly (no test patches them on ``charlie_work.workflow``).
"""

from __future__ import annotations

import charlie_work.workflow as _wf
from typing import Any
from charlie_work.github import GitHubError, label_names


def _human_merge_hold_check(self, issue_number: int | None) -> tuple[bool, bool]:
    """Return ``(human_merge_hold, human_merge_check_unavailable)`` for the bound issue.

    Issue #1598. The single shared detection routine used by both
    ``merge_ready`` (real path) and ``_merge_ready_dry_run`` so the two
    paths cannot drift. Reads the live issue labels at decision time via
    ``issue_view`` (same source the merge-hold check uses), not a cached
    snapshot. When ``human_merge_labels`` is empty (default) or the PR
    has no resolvable linked issue, returns ``(False, False)`` — the
    check is skipped entirely with zero overhead. When the issue fetch
    fails or returns a malformed payload (not a dict, or missing
    ``labels``), returns ``(False, True)`` so callers fail closed (block
    the merge) rather than silently proceeding. The de-escalation write
    when the label is absent is the real path's responsibility, not
    this helper's — dry-run is read-only.
    """
    if not self.config.dispatch.human_merge_labels or issue_number is None:
        return (False, False)
    try:
        _hm_issue = self.gh.issue_view(issue_number)
    except (GitHubError, ValueError):
        return (False, True)
    if not isinstance(_hm_issue, dict) or "labels" not in _hm_issue:
        return (False, True)
    _hm_labels = label_names(_hm_issue)
    human_merge_hold = bool(set(self.config.dispatch.human_merge_labels) & _hm_labels)
    return (human_merge_hold, False)


def _pr_bound_issue_has_human_merge_label(self, pr: dict[str, Any], branch_prefix: str) -> bool:
    """Return True if ``pr``'s bound issue carries a configured human-merge label.

    Issue #1598. Reads the live issue labels at decision time via
    ``issue_view`` (same source the merge-hold check uses), not a cached
    snapshot. Returns False when the PR has no resolvable linked issue or
    when the issue fetch fails — ``merge_ready``'s inline check is the
    authoritative enforcement point and handles the unavailable case
    fail-closed (blocks the merge). This helper is the cheaper
    merge-train-candidate filter: a false negative here merely lets the
    PR become merge-train head, and ``merge_ready`` catches it on the
    actual merge attempt.
    """
    issue_number = _wf.linked_issue_number(
        pr,
        is_cross_repository=pr.get("isCrossRepository"),
        branch_prefix=branch_prefix,
    )
    if issue_number is None:
        return False
    try:
        issue = self.gh.issue_view(issue_number)
    except (GitHubError, ValueError):
        return False
    if not isinstance(issue, dict) or "labels" not in issue:
        return False
    return bool(set(self.config.dispatch.human_merge_labels) & label_names(issue))
