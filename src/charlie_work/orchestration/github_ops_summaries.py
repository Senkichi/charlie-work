"""Issue/PR summary and mention-rearm delegates moved out of ``OrchestratorApp``.

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
from charlie_work.github import label_names


def _summarize_issue(self, issue: dict[str, Any]) -> dict[str, Any]:
    declared_blockers, open_blockers = self._get_open_blockers(issue)
    return {
        "number": issue.get("number"),
        "title": issue.get("title"),
        "url": issue.get("url"),
        "labels": sorted(label_names(issue)),
        "dispatchable": self._is_dispatchable(issue),
        "dependencies": {
            "declared": declared_blockers,
            "open": open_blockers,
        },
    }


def _summarize_pr(self, pr: dict[str, Any]) -> dict[str, Any]:
    return {
        "number": pr.get("number"),
        "title": pr.get("title"),
        "url": pr.get("url"),
        "issue_number": _wf.linked_issue_number(
            pr,
            is_cross_repository=pr.get("isCrossRepository"),
            branch_prefix=self.config.dispatch.branch_prefix,
        ),
        "head": pr.get("headRefName"),
        "is_draft": pr.get("isDraft"),
        "reviewDecision": pr.get("reviewDecision"),
    }


def _mention_rearmed_issue_numbers(
    self,
    mention_only: set[int],
    issues: list[dict[str, Any]],
    state: dict[str, Any],
    already_flagged: set[int],
) -> tuple[set[int], list[int]]:
    """Issue #1336: which mention-only issues have been re-armed by the
    operator and so must NOT be excluded from dispatch.

    Returns ``(rearmed, newly_detected)``:

    * ``rearmed`` -- the full set whose mention-only exclusion lifts
      (issues carrying the durable ``mention_rearmed_at`` marker plus
      issues whose re-arm is detected this pass).
    * ``newly_detected`` -- the subset of ``rearmed`` the caller should
      persist by stamping ``mention_rearmed_at``. Empty for the dry-run
      path, which never writes state.

    An issue lifts its mention-only exclusion only when it was flagged
    in a PRIOR pass (``already_flagged``) AND either the durable
    ``mention_rearmed_at`` marker is already set, or the operator has
    since removed ``agent:human-needed``. The label is read from the
    already-loaded ``issues`` objects (no new per-pass GitHub API call
    in the candidate filter -- the blast-radius concern the #564
    point-2 comment raised when it documented this as out of scope);
    the durable marker written by the caller means subsequent passes
    key the lift off the state signal alone.

    Safe default preserved (#564 / acceptance criterion 2): an issue
    never flagged, flagged this pass, or flagged and still carrying
    ``agent:human-needed`` stays excluded. ``bound`` exclusions are
    never lifted here -- those PRs genuinely bound to the issue by a
    hijack-safe signal.
    """
    issue_labels_by_number = {int(issue["number"]): label_names(issue) for issue in issues}
    rearmed: set[int] = set()
    newly_detected: list[int] = []
    for issue_number in mention_only:
        # Only issues flagged in a PRIOR pass can be re-armed. Issues
        # flagged this pass just had agent:human-needed applied and
        # must stay excluded; never-flagged issues have no flag to
        # re-arm through.
        if issue_number not in already_flagged:
            continue
        entry = state.get("issues", {}).get(str(issue_number), {})
        if not isinstance(entry, dict):
            continue
        if entry.get("mention_rearmed_at"):
            # Durable re-arm marker from a prior pass -- the lift stays.
            rearmed.add(issue_number)
            continue
        # Flagged before, not yet recorded as re-armed: detect the
        # operator's agent:human-needed removal from the already-loaded
        # issue labels. The caller stamps mention_rearmed_at so future
        # passes key off state, not the label snapshot.
        if self.config.labels.human_needed not in issue_labels_by_number.get(issue_number, set()):
            rearmed.add(issue_number)
            newly_detected.append(issue_number)
    return rearmed, newly_detected
