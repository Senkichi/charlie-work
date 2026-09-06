"""Review-decision reader and issue-comment renderer moved out of ``OrchestratorApp``.

Track 2 Phase B, L03 (design doc
``docs/design/2026-09-04-orchestratorapp-mikado-graph-and-delegation-plan.md``,
Sections 3.1/3.2). Bodies relocated verbatim from ``charlie_work.workflow``;
``workflow_delegation._install_delegates`` re-attaches each ``def`` unwrapped
onto ``OrchestratorApp``.

No moved body references a ``charlie_work.workflow`` name, so this module needs no
``import charlie_work.workflow as _wf`` -- there is nothing to rebind (the batch's
#1607 R2/R3 seam demonstrations run on the other two modules).
"""

from __future__ import annotations

from typing import Any

from charlie_work.github import defang_closing_keywords
from charlie_work.issue_comments import render_issue_comments
from charlie_work.review_decision import resolve_decision_payload


def _review_decision(self, pr_number: int) -> dict[str, Any]:
    """Read the resolved review decision for ``pr_number`` as a Mapping.

    Issue #1362 Stage 1: resolution order (the flat
    ``review-decision.json`` file first, falling back to the
    highest-numbered ``rounds/round-K/review-decision.json`` when the
    flat file is missing or unparseable) is delegated to
    ``review_decision``'s helpers -- the same ones
    ``rework_prompts._round_history_entries`` now imports rather than
    defining -- so the fallback logic lives in exactly one place
    instead of being re-derived here, fixing all of this method's
    call sites at once (previously this method had no round fallback
    at all; only ``rework_prompts.py`` did).

    Many callers need the FULL recorded payload (``required_changes``,
    ``summary``, ``escalated``, ``authorized_override``, ...) --
    fields ``review_decision.ReviewDecision`` deliberately does not
    carry, since that dataclass is scoped to control-flow
    approved/stale/missing questions only -- so this method keeps
    returning a plain dict rather than the dataclass. A caller that
    only needs "is this approved and fresh" should call
    ``review_decision.review_decision()`` directly instead (see
    ``already_approved`` in ``loop()``).

    A corrupt flat file with no usable round fallback now resolves to
    ``{"decision": "missing"}`` rather than the old ``"invalid"``
    sentinel -- both are already treated as "not a terminal verdict"
    by every caller that branches on decision value, so this collapses
    two fail-safe outcomes into one without changing whether a caller
    treats the PR as reviewed.
    """
    pr_dir = self.paths.prs / f"pr-{pr_number}"
    return resolve_decision_payload(pr_dir)


def _render_issue_comments(self, issue: dict[str, Any]) -> str:
    """Render the ``$issue_comments`` slot for a worker prompt (issue #872).

    The comments are already in hand: every ``_write_worker_prompt`` call
    site sources its issue from ``gh.issue_view``, and ``ISSUE_VIEW_FIELDS``
    has always requested ``comments``. This adds no API call -- the data was
    being fetched and discarded.

    Bodies are defanged here rather than in ``issue_comments`` itself so
    that module stays free of the GitHub layer; the defang is what stops a
    comment's "closes #123" from auto-closing an unrelated issue when a
    worker copies the text into its PR body.
    """
    dispatch = self.config.dispatch
    return render_issue_comments(
        issue.get("comments"),
        included_associations=dispatch.worker_prompt_comment_associations,
        excluded_authors=dispatch.worker_prompt_excluded_comment_authors,
        max_comments=dispatch.worker_prompt_max_comments,
        max_chars=dispatch.worker_prompt_max_comment_chars,
        sanitize=defang_closing_keywords,
    )
