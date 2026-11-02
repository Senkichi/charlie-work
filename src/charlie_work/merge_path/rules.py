"""Pure predicates and status sets shared by the **Merge path** stages.

Moved out of ``workflow.py`` so the decision functions and the legacy
``OrchestratorApp.merge_ready`` body read one definition. Nothing here touches
state, the clock, or GitHub.
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ..checks import CheckSummary
from ..dead_worker_sweep.effects_rework import _is_pr_updated_at_older_than

if TYPE_CHECKING:
    from ..config import AutoMergeConfig

# Statuses that mean an issue already has a rework routed (or an equivalent
# gate) in flight, or is otherwise spoken for -- shared by every "should I
# route this PR to rework_requested" check so they cannot drift apart.
# Originally the cross-PR-revert gate in ``merge_ready``; issue #784 AC-8
# Case 2 reuses it verbatim for ``review_queue``'s stranded-verdict check
# rather than inventing a second list. ``workflow`` re-exports it under its
# historical private name.
REWORK_ALREADY_ROUTED_STATUSES = (
    "escalated",
    "blocked",
    "dispatched",
    "dispatch_pending",
    "manifest_written",
    "rework_requested",
)

# A merge conflict whose rework worker is (about to be) running: wait for it.
CONFLICT_REWORK_IN_FLIGHT_STATUSES = ("dispatched", "dispatch_pending", "manifest_written")

# The merge-conflict route deliberately does NOT exclude "escalated" (issue
# #776): an unrelated-cause escalation must not wall a PR off from its own
# capped remediation. Only a queued manifest or a human decision excludes it.
CONFLICT_ROUTE_EXCLUDED_STATUSES = ("manifest_written", "blocked")

# The readiness-no-CI stall and the check-failure route exclude exactly the
# statuses that already have rework routed or a human in the loop.
CHECK_ROUTE_EXCLUDED_STATUSES = REWORK_ALREADY_ROUTED_STATUSES


def is_pending_only(summary: CheckSummary) -> bool:
    """Return True if the only reason the PR cannot merge is in-flight checks.

    A summary whose only defect is pending checks is not a structural merge
    failure; it should not arm the failed-attempt alarm.
    """
    return (
        bool(summary.pending)
        and not summary.failed
        and not summary.missing
        and not summary.infra_failed
        and not summary.infra_blocked
        and not summary.unavailable
    )


def format_merge_attempt_alarm_message(
    pr_number: int,
    attempts: int,
    summary: CheckSummary,
    mergeable: str | None = None,
    merge_state_status: str | None = None,
) -> str:
    """Human-readable alarm message for an approved PR that cannot merge.

    The message is surfaced in pass warnings, the merge_failed_attempt_alarm
    state event, and the notify digest terminal_reason.
    """
    buckets: list[str] = []
    if summary.missing:
        # Issue #253 signature: required checks missing while the PR is still open
        buckets.append("required checks missing while GitHub shows the PR open")
    if summary.pending:
        buckets.append(f"pending: {', '.join(summary.pending)}")
    if summary.failed:
        buckets.append(f"failed: {', '.join(summary.failed)}")
    if summary.infra_failed:
        buckets.append(f"infra_failed: {', '.join(summary.infra_failed)}")
    if summary.infra_blocked:
        buckets.append(f"infra_blocked: {', '.join(summary.infra_blocked)}")
    if summary.unavailable:
        # gh reported no parseable check list at all (see summarize_checks'
        # `checks is None` branch) — distinct from "all required checks
        # passed", so it must not fall into the passed-but-unmergeable
        # bucket below.
        buckets.append(f"unavailable: {', '.join(summary.unavailable)}")
    if not buckets:
        # Issue #751: every check-summary bucket above is empty, which means
        # the checks the bot tracks are not why this PR is stuck — GitHub's
        # own mergeability signal is (a lagging/absent CONFLICTING reading, a
        # BLOCKED merge state, branch protection, or a merge-base freshness
        # result that isn't one of the explicitly modelled branches above).
        # Surface what `pr_view` already fetched instead of discarding it;
        # only fall back to the generic "unknown" text when GitHub hasn't
        # reported anything usable either, so that case stays distinguishable.
        norm_mergeable = str(mergeable or "").upper()
        norm_merge_state = str(merge_state_status or "").upper()
        known_mergeable = bool(norm_mergeable) and norm_mergeable != "UNKNOWN"
        known_merge_state = bool(norm_merge_state) and norm_merge_state != "UNKNOWN"
        if known_mergeable or known_merge_state:
            buckets.append(
                f"mergeable={norm_mergeable or 'UNKNOWN'}, "
                f"mergeStateStatus={norm_merge_state or 'UNKNOWN'}, "
                "all required checks passed"
            )
        else:
            buckets.append("check summary unknown")
    checks_str = "; ".join(buckets)
    pass_str = "pass" if attempts == 1 else "passes"
    return f"PR #{pr_number} approved but unmergeable for {attempts} {pass_str}: {checks_str}"


def readiness_no_ci_stall(
    *,
    check_names_seen: Collection[str],
    updated_at: str | None,
    now: datetime,
    required_checks: Collection[str],
    minutes: int,
) -> bool:
    """Fact-based form of :func:`is_readiness_no_ci_stall`.

    True when stall detection is on (``minutes > 0`` and required checks are
    configured), none of the required checks appear among the reported check
    names, and ``updated_at`` is older than ``minutes``. ``updatedAt`` is the
    best available proxy for "head SHA pushed" in the ``gh pr view`` fields.
    """
    if minutes <= 0 or not required_checks:
        return False
    if any(name in check_names_seen for name in required_checks):
        return False
    return _is_pr_updated_at_older_than({"updatedAt": updated_at}, now, minutes)


def is_readiness_no_ci_stall(
    pr: dict[str, Any],
    checks: list[dict[str, Any]],
    config: AutoMergeConfig,
    now: datetime,
) -> bool:
    """Detect an approved PR whose required checks have never started.

    Returns True when:
      * ``pr_checks`` returned a parseable (non-None) list;
      * none of the configured ``required_checks`` appear in that list;
      * the PR's ``updatedAt`` is older than ``readiness_no_ci_minutes``.

    The required check names come from ``config.required_checks``; no names are
    hard-coded.
    """
    return readiness_no_ci_stall(
        check_names_seen={str(check.get("name") or "") for check in checks},
        updated_at=pr.get("updatedAt"),
        now=now,
        required_checks=config.required_checks,
        minutes=config.readiness_no_ci_minutes,
    )
