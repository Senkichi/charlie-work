"""Skip the full merge re-check for a PR already handed to the merge queue (#2440).

A PR carrying ``mergequeue_label`` is owned by the queue: ``merge_ready`` on it
only re-reads ``pr_view``/``pr_checks`` and the linked issue to re-prove what
the hand-off already established, and re-POSTs the label. With a deep queue
that is most of the pass's GraphQL budget. This module decides, from data the
pass already holds, when that re-check can be skipped.

The skip holds only while every one of these is true (any change falls through
to the full ``merge_ready``, which is the only path that can react to it):

* the per-pass PR snapshot still shows the queue label;
* the PR's head SHA equals ``prs[n].mergequeue_head_sha`` -- the head recorded
  by the accounting stage right after the hand-off;
* neither the PR nor the linked issue (read from the per-pass open-issue list)
  carries a hold label (``labels.merge_hold`` or a ``human_merge_labels``
  entry), and the issue is not ``priority:critical`` (a late critical label
  needs the skip-line hand-off);
* the last full check (``prs[n].mergequeue_checked_at``) is under
  ``RECHECK_INTERVAL`` old.

``decide`` is pure; ``merge_ready_unless_queued`` is the one gather+apply seam
the reap loop calls instead of ``merge_ready``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from ..github import label_names
from ..host import current as _host_current
from ..issue_priority import is_critical
from .gather import persisted_from
from .issue_labels import open_issue_labels
from .model import PersistedPr

RECHECK_INTERVAL = timedelta(minutes=30)


@dataclass(frozen=True)
class QueueSkipFacts:
    mergequeue_label: str | None
    hold_labels: frozenset[str]
    priority_prefix: str
    pr_labels: frozenset[str]
    pr_head_sha: str | None
    persisted: PersistedPr
    issue_bound: bool
    # ``None`` = the issue was not in the cached list: cannot prove it is unheld.
    issue_labels: frozenset[str] | None
    now: datetime


def parse_stamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def decide(facts: QueueSkipFacts) -> bool:
    """True when the full re-check can be skipped this pass."""
    p = facts.persisted
    label = facts.mergequeue_label
    if not label or label not in facts.pr_labels:
        return False
    if p.status != "mergequeue" or p.mergequeue_revoked_reason:
        return False
    if not facts.pr_head_sha or p.mergequeue_head_sha != facts.pr_head_sha:
        return False
    if facts.pr_labels & facts.hold_labels:
        return False
    if facts.issue_bound:
        if facts.issue_labels is None:
            return False
        if facts.issue_labels & facts.hold_labels:
            return False
        if facts.priority_prefix and is_critical(facts.issue_labels, facts.priority_prefix):
            return False
    checked = parse_stamp(p.mergequeue_checked_at)
    if checked is None:
        return False
    age = facts.now - checked
    return timedelta(0) <= age < RECHECK_INTERVAL


def hold_labels_of(app: Any) -> frozenset[str]:
    return frozenset({app.config.labels.merge_hold, *app.config.dispatch.human_merge_labels})


def merge_ready_unless_queued(
    app: Any,
    pr: dict[str, Any],
    pr_state: dict[str, Any],
    issue_number: int | None,
    *,
    merge: bool | None,
    merge_train_head: int | None,
    errors: list[Any],
    merges: list[Any],
) -> bool:
    """Run ``merge_ready`` for ``pr`` and record it, unless the queue skip holds.

    Returns True when the PR was skipped (zero GitHub calls, no event).
    """
    pr_number = int(pr["number"])
    skipped = False
    if not app.dry_run and (label := app.config.auto_merge.mergequeue_label):
        persisted = persisted_from(pr_state)
        if persisted.status == "mergequeue":
            cached = open_issue_labels(app.gh, issue_number) if issue_number is not None else None
            issue_labels = frozenset(cached) if cached is not None else None
            skipped = decide(
                QueueSkipFacts(
                    mergequeue_label=label,
                    hold_labels=hold_labels_of(app),
                    priority_prefix=app.config.labels.priority_prefix,
                    pr_labels=frozenset(label_names(pr)),
                    pr_head_sha=pr.get("headRefOid"),
                    persisted=persisted,
                    issue_bound=issue_number is not None,
                    issue_labels=issue_labels,
                    now=_host_current().clock.now(),
                )
            )
    if skipped:
        return True
    merge_result = app.merge_ready(pr_number, merge=merge, merge_train_head=merge_train_head)
    app._record_merge_or_error(merge_result, errors, merges)
    return False
