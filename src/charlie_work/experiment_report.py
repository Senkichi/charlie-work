"""Per-PR experiment read-out over ``events.db`` (issue #1701).

The reviewer-effort experiment has been assigning PRs to arms since
2026-07-26 and recording the arm in ``record_review`` session metrics, but
nothing consumed the signal.  This module is the read-out: it reads the
event log, groups everything **by PR** (the arm assignment is a pure
function of ``(pr_number, salt)`` and is stable across rework rounds, so the
PR -- not the review round -- is the unit of analysis), and reports
activity and outcome metrics per arm with intervals, plus the experiment's
stopping rule.

The module is split under the repo's 800-line cap:

* :mod:`charlie_work.experiment_report_intervals` -- Wilson per-arm
  intervals and the Newcombe hybrid-score difference interval.
* :mod:`charlie_work.experiment_report_scan` -- the event vocabulary and
  the payload scanners that turn raw events into ``(pr, arm)``
  observations and per-PR flags.
* :mod:`charlie_work.experiment_report_stopping` -- the stopping rule and
  its constants (the equivalence margin and observed-event floor live
  there).
* :mod:`charlie_work.experiment_report_render` -- the text rendering.

This module keeps the report assembly (:func:`build_report`), the window
semantics, and the public names the command layer and tests import --
re-exported deliberately per the repo's facade pattern.

Design invariants:

* **PR-level unit.**  A PR can have many ``record_review`` rounds.  Rates
  are computed over PRs (first-round decision, escalation, merge, outcome
  flags); only ``rounds_per_pr`` and ``review_cost_per_pr`` aggregate
  rounds.  Per-round rates would inflate ``request_changes`` because each
  kickback creates another round in the same arm.
* **Arm values are derived, never named.**  Arm labels come from the
  recorded values of the ``--experiment``-selected session-metrics key
  (``<experiment>_arm``; a value already ending in ``_arm`` is used
  verbatim).  Nothing in this module knows the experiment's arm names, so
  the same code serves the current experiment and any future one that
  records an ``*_arm`` key (e.g. the planned ``brief_arm``, issue #1276).
* **Outcomes vs. activity.**  Every metric is labelled ``"activity"`` or
  ``"outcome"``.  Outcome metrics measure *whether the review was right*
  and are the only signals allowed to drive the stopping rule.  Only one
  outcome signal is derivable today -- an approved branch found to
  silently revert a base commit.  ``check_failure_rework_requested`` is
  deliberately **not** an outcome: its own rework brief says the approval
  stands ("do not re-litigate the review"), so it is pipeline state, not
  evidence the review was wrong.  ``no_op_rework_repair_requested``
  measures an empty rework *cycle*, not a wrong kickback.  Where no
  outcome metric is derivable, the report says so explicitly and the
  stopping rule stays ``not_met`` -- activity metrics never imply a
  conclusion they cannot support.
* **Read-only.**  The command layer gates on ``events.db`` existence before
  calling :func:`charlie_work.instrumentation.query_events`, because
  ``_get_db`` performs WAL setup / schema migration / legacy-jsonl import
  on first open.  This module itself never opens the database; it consumes
  the event list the caller fetched.
"""

from __future__ import annotations

import statistics
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence

from .experiment_report_intervals import (
    CONFIDENCE_Z,  # noqa: F401  (deliberate re-export)
    rate_difference_interval,
    wilson_interval,  # noqa: F401  (deliberate re-export)
)
from .experiment_report_render import render_text  # noqa: F401  (deliberate re-export)
from .experiment_report_scan import (
    DECISIONS,
    DISPATCH_EVENT_KINDS,
    ESCALATED_SUFFIX,
    MERGE_EVENT_KINDS,
    NO_OUTCOME_STATEMENT,
    NOT_DERIVABLE,
    _dispatch_prs,
    _iter_arm_observations,
    _merge_prs,
    _pr_of,
)
from .experiment_report_stopping import (
    DEFAULT_MIN_PRS_PER_ARM,  # noqa: F401  (deliberate re-export)
    EQUIVALENCE_MARGIN,  # noqa: F401  (deliberate re-export)
    EQUIVALENCE_MULTIPLE,  # noqa: F401  (deliberate re-export)
    MIN_EQUIVALENCE_EVENTS_PER_ARM,  # noqa: F401  (deliberate re-export)
    evaluate_stopping_rule,  # noqa: F401  (deliberate re-export)
)

# Post-approval signal classification -- see the taxonomy docstring in
# experiment_report_scan.py for what each kind measures.  These sets live
# in THIS module (the consumption site) rather than the vocabulary module
# on purpose: tests/test_event_kind_consumers.py counts a kind as consumed
# only when its literal sits in a read position resolvable within the same
# file, and an imported-name comparison is invisible to that audit.

# Post-approval signals that measure review correctness -- the only kind
# allowed into the outcome aggregate and the stopping rule.
REVIEW_DEFECT_KINDS = frozenset(
    {
        "cross_pr_revert_rework_requested",
    }
)

# Post-approval signals that measure pipeline state, not review
# correctness.  Reported as activity; never input to the stopping rule.
PIPELINE_REWORK_KINDS = frozenset(
    {
        "check_failure_rework_requested",
    }
)

POST_APPROVAL_SIGNAL_KINDS = REVIEW_DEFECT_KINDS | PIPELINE_REWORK_KINDS

# A rework cycle that produced no content change relative to the last
# request_changes verdict (janitor._check_no_op_rework consumer) -- a
# rework-cycle measure, not a wrong-kickback measure.
NO_OP_REWORK_KINDS = frozenset(
    {
        "no_op_rework_repair_requested",
    }
)

# The outcome aggregate that feeds the stopping rule.  Kept as a named
# constant so the rule's input is declared once; the metric itself counts
# every kind in REVIEW_DEFECT_KINDS.
STOPPING_RULE_OUTCOME_METRIC = "post_approval_defect_rate"


def metrics_key_for_experiment(experiment: str) -> str:
    """Map ``--experiment`` to the session-metrics key carrying the arm.

    ``review_effort`` -> ``review_effort_arm``; a value already ending in
    ``_arm`` (e.g. ``brief_arm``) is used verbatim.
    """
    return experiment if experiment.endswith("_arm") else f"{experiment}_arm"


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def _parse_ts(ts: Any) -> datetime | None:
    """Parse an ISO-8601 event timestamp; None when absent/malformed."""
    if not isinstance(ts, str) or not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def parse_window_bound(value: str | None) -> datetime | None:
    """Parse a CLI ``--since``/``--until``/``--exclude-window`` bound.

    Naive inputs are interpreted as UTC.  Returns None for None input;
    raises ``ValueError`` for an unparseable string (the caller turns that
    into a user-facing error message).
    """
    if value is None:
        return None
    dt = _parse_ts(value)
    if dt is None:
        raise ValueError(f"not an ISO-8601 timestamp: {value!r}")
    return dt


def _in_window(
    ts: datetime | None,
    since: datetime | None,
    until: datetime | None,
    exclude_windows: Sequence[tuple[datetime, datetime]],
) -> bool:
    """True when an event at *ts* contributes to the report.

    ``--since``/``--until`` are inclusive bounds; each ``--exclude-window``
    is an inclusive [start, end] interval whose events contribute to no
    figure.  An event with an unparseable timestamp is kept (fail-open: it
    is better counted than silently dropped from a read-out).
    """
    if ts is None:
        return True
    if since is not None and ts < since:
        return False
    if until is not None and ts > until:
        return False
    for start, end in exclude_windows:
        if start <= ts <= end:
            return False
    return True


# ---------------------------------------------------------------------------
# Report construction
# ---------------------------------------------------------------------------


def _rate_entry(k: int, n: int, z: float) -> dict[str, Any]:
    ci = wilson_interval(k, n, z)
    return {
        "k": k,
        "n": n,
        "rate": (k / n) if n else None,
        "ci95": list(ci) if ci else None,
    }


def _metric(
    name: str,
    measure: str,
    kind: str,
    description: str,
    per_arm: dict[str, dict[str, Any]],
    differences: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "measure": measure,
        "kind": kind,
        "description": description,
        "per_arm": per_arm,
        "differences": differences or [],
    }


def _rate_metric(
    name: str,
    measure: str,
    description: str,
    arms: list[str],
    k_by_arm: dict[str, int],
    n_by_arm: dict[str, int],
    z: float,
) -> dict[str, Any]:
    per_arm = {a: _rate_entry(k_by_arm.get(a, 0), n_by_arm.get(a, 0), z) for a in arms}
    differences: list[dict[str, Any]] = []
    for i, a in enumerate(arms):
        for b in arms[i + 1 :]:
            ci = rate_difference_interval(
                k_by_arm.get(a, 0),
                n_by_arm.get(a, 0),
                k_by_arm.get(b, 0),
                n_by_arm.get(b, 0),
                z,
            )
            if ci is None:
                continue
            differences.append(
                {
                    "pair": [a, b],
                    "diff": k_by_arm[a] / n_by_arm[a] - k_by_arm[b] / n_by_arm[b],
                    "ci95": list(ci),
                }
            )
    return _metric(name, measure, "rate", description, per_arm, differences)


def _mean_metric(
    name: str,
    description: str,
    arms: list[str],
    values_by_arm: dict[str, list[float]],
    extra_by_arm: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    per_arm: dict[str, dict[str, Any]] = {}
    for a in arms:
        values = values_by_arm.get(a, [])
        entry: dict[str, Any] = {
            "n": len(values),
            "value": (sum(values) / len(values)) if values else None,
        }
        if extra_by_arm and a in extra_by_arm:
            entry.update(extra_by_arm[a])
        per_arm[a] = entry
    return _metric(name, "activity", "mean", description, per_arm)


def _median_metric(
    name: str,
    description: str,
    arms: list[str],
    values_by_arm: dict[str, list[float]],
) -> dict[str, Any]:
    per_arm: dict[str, dict[str, Any]] = {}
    for a in arms:
        values = sorted(values_by_arm.get(a, []))
        per_arm[a] = {
            "n": len(values),
            "value": statistics.median(values) if values else None,
        }
    return _metric(name, "activity", "median", description, per_arm)


def _outcome_coverage(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Assemble the report's outcome-coverage section.

    ``derivable`` is derived from the metrics themselves (every metric
    labelled ``outcome``), never enumerated -- a future outcome signal is
    covered the moment it is recorded.  When the list is empty the section
    carries the explicit ``statement`` the issue requires, and the
    stopping rule reports ``not_met`` rather than implying a conclusion
    from activity metrics.
    """
    derivable = [name for name, m in metrics.items() if m["measure"] == "outcome"]
    return {
        "derivable": derivable,
        "not_derivable": list(NOT_DERIVABLE),
        "statement": None if derivable else NO_OUTCOME_STATEMENT,
    }


def build_report(
    events: list[dict[str, Any]],
    metrics_key: str,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    exclude_windows: Sequence[tuple[datetime, datetime]] = (),
    min_prs_per_arm: int = DEFAULT_MIN_PRS_PER_ARM,
    z: float = CONFIDENCE_Z,
) -> dict[str, Any]:
    """Build the per-PR experiment report from an event list.

    *events* is the caller-fetched event list (oldest-first, as
    ``query_events`` returns).  Window filtering happens here so
    ``--since``/``--until``/``--exclude-window`` share one inclusive
    semantics implemented once.
    """
    kept = [e for e in events if _in_window(_parse_ts(e.get("ts")), since, until, exclude_windows)]

    arm_obs: dict[int, set[str]] = {}
    rounds: dict[int, list[tuple[datetime | None, dict[str, Any]]]] = {}
    issue_of: dict[int, int] = {}
    merged_prs: set[int] = set()
    merge_ts: dict[int, datetime] = {}
    escalated_prs: set[int] = set()
    defect_events: dict[int, set[str]] = {}
    noop_kickback_prs: set[int] = set()
    dispatch_events: list[dict[str, Any]] = []

    for e in kept:
        ts = _parse_ts(e.get("ts"))
        for pr, arm in _iter_arm_observations(e, metrics_key):
            arm_obs.setdefault(pr, set()).add(arm)
        payload = e.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}
        if e["kind"] == "record_review":
            pr = _pr_of(payload, e)
            if pr is not None:
                rounds.setdefault(pr, []).append((ts, payload))
                issue_raw = payload.get("issue_number")
                if (
                    issue_of.get(pr) is None
                    and isinstance(issue_raw, (int, float))
                    and not isinstance(issue_raw, bool)
                ):
                    issue_of[pr] = int(issue_raw)
        elif e["kind"] in MERGE_EVENT_KINDS:
            for pr in _merge_prs(e):
                merged_prs.add(pr)
                if ts is not None and (pr not in merge_ts or ts < merge_ts[pr]):
                    merge_ts[pr] = ts
        elif e["kind"] in DISPATCH_EVENT_KINDS:
            dispatch_events.append(e)
        if e["kind"].endswith(ESCALATED_SUFFIX):
            pr = _pr_of(payload, e)
            if pr is not None:
                escalated_prs.add(pr)
        if e["kind"] in POST_APPROVAL_SIGNAL_KINDS:
            pr = _pr_of(payload, e)
            if pr is not None:
                defect_events.setdefault(pr, set()).add(e["kind"])
        if e["kind"] in NO_OP_REWORK_KINDS:
            pr = _pr_of(payload, e)
            if pr is not None:
                noop_kickback_prs.add(pr)

    # Second pass: dispatch timestamps need the complete PR->issue map.
    issue_to_prs: dict[int, set[int]] = {}
    for pr, issue in issue_of.items():
        issue_to_prs.setdefault(issue, set()).add(pr)
    dispatch_ts: dict[int, datetime] = {}
    for e in dispatch_events:
        ts = _parse_ts(e.get("ts"))
        if ts is None:
            continue
        for pr in _dispatch_prs(e, issue_to_prs):
            if pr not in dispatch_ts or ts < dispatch_ts[pr]:
                dispatch_ts[pr] = ts

    # Arm assignment: a PR belongs to an arm iff every observed arm value
    # for it agrees.  A conflict is a data-integrity finding, not a vote.
    conflicted: dict[int, set[str]] = {pr: arms for pr, arms in arm_obs.items() if len(arms) > 1}
    assigned: dict[int, str] = {
        pr: next(iter(arms)) for pr, arms in arm_obs.items() if len(arms) == 1
    }
    arms = sorted(set(assigned.values()))
    prs_by_arm: dict[str, set[int]] = {a: set() for a in arms}
    for pr, arm in assigned.items():
        prs_by_arm[arm].add(pr)

    # ---- per-arm raw tallies ----
    prs_assigned = {a: len(prs_by_arm[a]) for a in arms}
    prs_with_rounds = {a: sum(1 for p in prs_by_arm[a] if rounds.get(p)) for a in arms}
    first_decision_k: dict[str, dict[str, int]] = {a: {d: 0 for d in DECISIONS} for a in arms}
    first_decision_other: dict[str, int] = {a: 0 for a in arms}
    rounds_per_pr: dict[str, list[float]] = {a: [] for a in arms}
    cost_per_pr: dict[str, list[float]] = {a: [] for a in arms}
    cost_rounds_total: dict[str, int] = {a: 0 for a in arms}
    cost_rounds_with: dict[str, int] = {a: 0 for a in arms}
    escalated_k: dict[str, int] = {a: 0 for a in arms}
    merged_k: dict[str, int] = {a: 0 for a in arms}
    d2m_seconds: dict[str, list[float]] = {a: [] for a in arms}
    approved_n: dict[str, int] = {a: 0 for a in arms}
    review_defect_k: dict[str, int] = {a: 0 for a in arms}
    pipeline_rework_k: dict[str, int] = {a: 0 for a in arms}
    kickback_n: dict[str, int] = {a: 0 for a in arms}
    noop_k: dict[str, int] = {a: 0 for a in arms}

    for pr, arm in assigned.items():
        pr_rounds = rounds.get(pr, [])
        if pr in escalated_prs:
            escalated_k[arm] += 1
        if pr in merged_prs:
            merged_k[arm] += 1
            if pr in dispatch_ts and pr in merge_ts:
                d2m_seconds[arm].append((merge_ts[pr] - dispatch_ts[pr]).total_seconds())
        if not pr_rounds:
            continue
        rounds_per_pr[arm].append(float(len(pr_rounds)))
        decisions = [p.get("decision") for _, p in pr_rounds]
        first = decisions[0]
        if first in DECISIONS:
            first_decision_k[arm][first] += 1
        else:
            first_decision_other[arm] += 1
        total_cost = 0.0
        for _, p in pr_rounds:
            sm = p.get("session_metrics") or {}
            cost = sm.get("cost_usd") if isinstance(sm, dict) else None
            cost_rounds_total[arm] += 1
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                total_cost += float(cost)
                cost_rounds_with[arm] += 1
        cost_per_pr[arm].append(total_cost)
        if "approved" in decisions:
            approved_n[arm] += 1
            for kind in defect_events.get(pr, ()):
                if kind in REVIEW_DEFECT_KINDS:
                    review_defect_k[arm] += 1
                elif kind in PIPELINE_REWORK_KINDS:
                    pipeline_rework_k[arm] += 1
        if "request_changes" in decisions:
            kickback_n[arm] += 1
            if pr in noop_kickback_prs:
                noop_k[arm] += 1

    # ---- metrics ----
    metrics: dict[str, dict[str, Any]] = {}
    metrics["prs_assigned"] = _metric(
        "prs_assigned",
        "activity",
        "count",
        "PRs carrying a consistent arm value for this experiment key",
        {a: {"n": prs_assigned[a], "value": prs_assigned[a]} for a in arms},
    )
    metrics["prs_with_rounds"] = _metric(
        "prs_with_rounds",
        "activity",
        "count",
        "assigned PRs with at least one recorded review round",
        {a: {"n": prs_with_rounds[a], "value": prs_with_rounds[a]} for a in arms},
    )
    for decision in DECISIONS:
        metrics[f"first_round_{decision}_rate"] = _rate_metric(
            f"first_round_{decision}_rate",
            "activity",
            f'share of assigned PRs whose first recorded round decided "{decision}" '
            "(denominator: PRs with >=1 recorded round)",
            arms,
            {a: first_decision_k[a][decision] for a in arms},
            prs_with_rounds,
            z,
        )
    metrics["rounds_per_pr_mean"] = _mean_metric(
        "rounds_per_pr_mean",
        "mean review rounds per assigned PR with >=1 recorded round",
        arms,
        rounds_per_pr,
    )
    metrics["review_cost_per_pr_usd_mean"] = _mean_metric(
        "review_cost_per_pr_usd_mean",
        "mean total reviewer session cost_usd per assigned PR with >=1 recorded round",
        arms,
        cost_per_pr,
        {
            a: {
                "rounds_with_cost": cost_rounds_with[a],
                "rounds_total": cost_rounds_total[a],
            }
            for a in arms
        },
    )
    metrics["escalation_rate"] = _rate_metric(
        "escalation_rate",
        "activity",
        "share of assigned PRs named by any *_escalated event",
        arms,
        escalated_k,
        prs_assigned,
        z,
    )
    metrics["merge_rate"] = _rate_metric(
        "merge_rate",
        "activity",
        "share of assigned PRs that reached a merge event",
        arms,
        merged_k,
        prs_assigned,
        z,
    )
    metrics["dispatch_to_merge_seconds_median"] = _median_metric(
        "dispatch_to_merge_seconds_median",
        "median seconds from earliest dispatch/launch event to earliest merge "
        "event, over merged assigned PRs with a dispatch timestamp",
        arms,
        d2m_seconds,
    )
    metrics["post_approval_defect_rate"] = _rate_metric(
        "post_approval_defect_rate",
        "outcome",
        "share of approved PRs later routed to rework by a post-approval "
        "review-correctness defect signal (cross_pr_revert_rework_requested: "
        "the approved branch silently reverted a base commit) -- an approval "
        "that did not hold up. check_failure_rework_requested is excluded "
        "by construction: its rework brief keeps the approval standing, so "
        "it measures pipeline state, not whether the review was right",
        arms,
        review_defect_k,
        approved_n,
        z,
    )
    metrics["post_approval_revert_bounce_rate"] = _rate_metric(
        "post_approval_revert_bounce_rate",
        "outcome",
        "share of approved PRs later routed to rework for a silent "
        "base-commit revert in the branch (cross_pr_revert_rework_requested) "
        "-- currently the whole of post_approval_defect_rate",
        arms,
        review_defect_k,
        approved_n,
        z,
    )
    metrics["post_approval_ci_failure_rate"] = _rate_metric(
        "post_approval_ci_failure_rate",
        "activity",
        "share of approved PRs later routed to rework for genuinely failing "
        "required checks (check_failure_rework_requested) -- pipeline state, "
        "not a review-correctness outcome: the rework brief keeps the "
        "approval standing, so this never feeds the stopping rule",
        arms,
        pipeline_rework_k,
        approved_n,
        z,
    )
    metrics["no_op_kickback_rate"] = _rate_metric(
        "no_op_kickback_rate",
        "activity",
        "share of request_changes PRs whose next rework cycle produced no "
        "content change relative to the verdict "
        "(no_op_rework_repair_requested) -- an empty or stalled rework cycle "
        "(unpushed commits, a dead session, or nothing left to change), not "
        "evidence the kickback itself was wrong",
        arms,
        noop_k,
        kickback_n,
        z,
    )

    warnings: list[dict[str, Any]] = [
        {
            "pr": pr,
            "arms": sorted(armset),
            "detail": (
                f"conflicting arm values {sorted(armset)} for key "
                f'"{metrics_key}" across events; PR excluded from all arms'
            ),
        }
        for pr, armset in sorted(conflicted.items())
    ]
    unassigned_with_rounds = sum(1 for pr in rounds if pr not in arm_obs and pr not in conflicted)

    coverage = _outcome_coverage(metrics)
    stopping = evaluate_stopping_rule(
        arms,
        prs_assigned,
        metrics[STOPPING_RULE_OUTCOME_METRIC] if coverage["derivable"] else None,
        min_prs_per_arm,
        z,
    )

    return {
        "metrics_key": metrics_key,
        "window": {
            "since": since.isoformat() if since else None,
            "until": until.isoformat() if until else None,
            "exclude_windows": [[s.isoformat(), e.isoformat()] for s, e in exclude_windows],
        },
        "events_scanned": len(kept),
        "arms": arms,
        "per_arm": {
            a: {
                "prs_assigned": prs_assigned[a],
                "prs_with_rounds": prs_with_rounds[a],
                "first_round_decisions": {d: first_decision_k[a][d] for d in DECISIONS},
                "first_round_other_decisions": first_decision_other[a],
            }
            for a in arms
        },
        "metrics": metrics,
        "data_integrity_warnings": warnings,
        "prs_with_rounds_without_arm_value": unassigned_with_rounds,
        "outcome_coverage": coverage,
        "stopping_rule": stopping,
    }
