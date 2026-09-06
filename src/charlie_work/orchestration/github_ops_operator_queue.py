"""Operator-queue command delegate moved out of ``OrchestratorApp``.

Track 2 Phase B, L04 batch 2 (design doc
``docs/design/2026-09-04-orchestratorapp-mikado-graph-and-delegation-plan.md``,
Sections 3.1/3.2). Body relocated verbatim from ``charlie_work.workflow``;
``workflow_delegation._install_delegates`` re-attaches the top-level ``def``
unwrapped onto ``OrchestratorApp``. ``CommandResult`` and ``load_state_locked``
are reached through ``_wf.``: ``CommandResult`` is a ``charlie_work.workflow``
module-level class, and ``load_state_locked`` is patched on
``charlie_work.workflow`` by the suite (Tier D, string form). All other free
names are imported directly (no test patches them on ``charlie_work.workflow``).
"""

from __future__ import annotations

import charlie_work.workflow as _wf
from datetime import UTC, datetime
from typing import Any
from charlie_work.github import label_names
from charlie_work.instrumentation import query_events
from charlie_work.state import (
    DELIBERATELY_UNCLASSIFIED_ESCALATION_EVENT_KINDS,
    ESCALATION_REASON_CLASS_BY_EVENT_KIND,
)


def operator_queue(self) -> _wf.CommandResult:
    """List issues currently parked on ``agent:operator-queue`` (issue #1314 item 1).

    An operator-facing inspection command that joins three data sources
    so the queue is workable without hand-rolling ``gh`` queries:

    - GitHub issues carrying the ``operator_queue`` label (via
      ``gh.issue_list``), for title/URL/label context.
    - ``state.json``'s ``issues`` map, for ``reason_class`` provenance,
      ``escalation_reason``, ``terminal_since`` (age), and the
      de-escalation cap marker (``deescalation_cap_notified_at``).
    - ``events.db``, for the last escalation-transition event per issue
      (the ``kind`` and ``ts`` of the most recent event whose kind is in
      ``ESCALATION_REASON_CLASS_BY_EVENT_KIND`` or
      ``DELIBERATELY_UNCLASSIFIED_ESCALATION_EVENT_KINDS``), so an
      operator can see *when* and *why* the issue was parked.

    Issues on the label but missing from state (a manual label add with
    no escalation event) are included with ``reason_class: null`` and
    ``age_days: null`` so they are visible rather than silently dropped.
    Issues in state but missing from the GitHub label query (a label
    transition that has not yet propagated) are included from state
    alone.

    Returns a ``CommandResult`` with a ``queue`` list sorted by
    ``terminal_since`` ascending (oldest first), each entry carrying
    ``number``, ``title``, ``url``, ``labels``, ``reason_class``,
    ``escalation_reason``, ``terminal_since``, ``age_days``,
    ``deescalation_cap_notified_at``, and ``last_escalation_event``.
    """
    operator_queue_label = self.config.labels.operator_queue
    issues = self.gh.issue_list(operator_queue_label)
    state = _wf.load_state_locked(self.paths.state_file)
    state_issues = state.get("issues", {})
    if not isinstance(state_issues, dict):
        state_issues = {}

    now = datetime.now(UTC)
    escalation_kinds = (
        frozenset(ESCALATION_REASON_CLASS_BY_EVENT_KIND)
        | DELIBERATELY_UNCLASSIFIED_ESCALATION_EVENT_KINDS
    )

    # Build a set of issue numbers from the GitHub label query.
    gh_numbers: set[int] = set()
    issues_by_number: dict[int, dict[str, Any]] = {}
    for issue in issues:
        num = issue.get("number")
        if num is not None:
            gh_numbers.add(int(num))
            issues_by_number[int(num)] = issue

    # Build a set of issue numbers from state that match the operator-queue
    # criteria (status == "escalated", reason_class == "mechanical").
    state_numbers: set[int] = set()
    for num_str, entry in state_issues.items():
        if not isinstance(entry, dict):
            continue
        if (
            entry.get("status") == "escalated"
            and entry.get("reason_class") == "mechanical"
            and str(num_str).isdigit()
        ):
            state_numbers.add(int(num_str))

    all_numbers = sorted(gh_numbers | state_numbers)

    queue: list[dict[str, Any]] = []
    for issue_number in all_numbers:
        tracked_entry = state_issues.get(str(issue_number))
        if not isinstance(tracked_entry, dict):
            tracked_entry = {}

        gh_issue = issues_by_number.get(issue_number, {})

        terminal_since = tracked_entry.get("terminal_since")
        age_days: float | None = None
        if terminal_since:
            try:
                since_dt = datetime.fromisoformat(str(terminal_since).replace("Z", "+00:00"))
                age_days = round((now - since_dt).total_seconds() / 86400.0, 2)
            except (ValueError, TypeError):
                age_days = None

        # Query events.db for the last escalation-transition event.
        last_escalation_event: dict[str, Any] | None = None
        events = query_events(
            self.paths.state_file,
            issue_number=issue_number,
        )
        escalation_events = [e for e in events if e.get("kind") in escalation_kinds]
        if escalation_events:
            last_escalation_event = {
                "kind": escalation_events[-1].get("kind"),
                "ts": escalation_events[-1].get("ts"),
            }

        queue.append(
            {
                "number": issue_number,
                "title": gh_issue.get("title"),
                "url": gh_issue.get("url"),
                "labels": sorted(label_names(gh_issue)) if gh_issue else [],
                "reason_class": tracked_entry.get("reason_class"),
                "escalation_reason": tracked_entry.get("escalation_reason"),
                "terminal_since": terminal_since,
                "age_days": age_days,
                "deescalation_cap_notified_at": tracked_entry.get("deescalation_cap_notified_at"),
                "last_escalation_event": last_escalation_event,
            }
        )

    # Sort by terminal_since ascending (oldest first); issues with no
    # terminal_since sort last.
    queue.sort(key=lambda e: (e["terminal_since"] is None, e["terminal_since"] or ""))

    return _wf.CommandResult(
        True,
        f"operator queue: {len(queue)} issue(s) parked on {operator_queue_label}",
        {"queue": queue, "depth": len(queue)},
    )
