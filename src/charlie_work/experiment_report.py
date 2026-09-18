"""Per-PR experiment read-out over ``events.db`` (issue #1701).

The reviewer-effort experiment has been assigning PRs to arms since
2026-07-26 and recording the arm in ``record_review`` session metrics, but
nothing consumed the signal.  This module is the read-out: it reads the
event log, groups everything **by PR** (the arm assignment is a pure
function of ``(pr_number, salt)`` and is stable across rework rounds, so the
PR -- not the review round -- is the unit of analysis), and reports
activity and outcome metrics per arm with intervals, plus the experiment's
stopping rule.

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
  ``"outcome"``.  Activity metrics (first-round decision, rounds, cost,
  escalation, merge share, dispatch-to-merge latency) describe *what the
  arms did*; outcome metrics describe *whether the review was right* --
  the only signals allowed to drive the stopping rule.
* **Read-only.**  The command layer gates on ``events.db`` existence before
  calling :func:`charlie_work.instrumentation.query_events`, because
  ``_get_db`` performs WAL setup / schema migration / legacy-jsonl import
  on first open.  This module itself never opens the database; it consumes
  the event list the caller fetched.
"""

from __future__ import annotations

import math
import statistics
from datetime import UTC, datetime
from typing import Any, Iterable, Mapping, Sequence

# Stopping-rule defaults.  These match docs/review-effort-experiment.md;
# --min-prs-per-arm overrides the minimum for ad-hoc analysis, not the rule.
DEFAULT_MIN_PRS_PER_ARM = 100
# Equivalence bound: every arm at twice the minimum with a still-overlapping
# outcome-difference interval ends the experiment as "no detectable
# difference" rather than running forever.
EQUIVALENCE_MULTIPLE = 2
CONFIDENCE_Z = 1.96

# The record_review decision vocabulary (verdict_parsing.py).
_DECISIONS = ("approved", "request_changes", "blocked")

# Event kinds naming a PR's merge completion (the report takes the earliest
# timestamp across them).  `reconcile` is included but only counts when its
# payload kind is `merged_outside_orchestrator`.
_MERGE_EVENT_KINDS = frozenset(
    {
        "merge_succeeded",
        "finalize_externally_merged",
        "reconcile",
    }
)

# Event kinds that timestamp a PR's entry into the pipeline, used as the
# dispatch end of dispatch-to-merge latency: issue dispatch (payload
# `issue_numbers`), review-launch (`launched`), and claim (`pr_numbers` /
# assignment dicts).  The earliest across all of them is the dispatch time.
_DISPATCH_EVENT_KINDS = frozenset(
    {
        "dispatch",
        "review_dispatch",
        "review_dispatch_claim",
    }
)

# Post-approval defect signals: events that fire only after a PR has an
# approved verdict, meaning the approval did not hold up.
#   check_failure_rework_requested  -- required checks genuinely failed on
#                                      the approved head (merge_ready path)
#   cross_pr_revert_rework_requested -- the approved branch was found to
#                                      silently revert a base commit
_POST_APPROVAL_DEFECT_KINDS = frozenset(
    {
        "check_failure_rework_requested",
        "cross_pr_revert_rework_requested",
    }
)

_ESCALATED_SUFFIX = "_escalated"

# Outcome candidates the recorded event stream cannot produce today.  The
# report states this explicitly rather than implying a conclusion from
# activity metrics (issue #1701's contract).
_NOT_DERIVABLE: tuple[dict[str, str], ...] = (
    {
        "candidate": "post-merge main-branch CI failure attributed to the merged PR",
        "reason": (
            "no recorded event links a main-branch check failure after a merge "
            "back to the PR that merged"
        ),
    },
    {
        "candidate": "follow-up fix or revert naming the merged PR",
        "reason": (
            "no recorded event attributes a later fix or revert commit/PR to a "
            "previously merged PR"
        ),
    },
)


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
# Interval math
# ---------------------------------------------------------------------------


def wilson_interval(k: int, n: int, z: float = CONFIDENCE_Z) -> tuple[float, float] | None:
    """Wilson score interval for a binomial rate; None when n == 0."""
    if n <= 0:
        return None
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate_difference_interval(
    k1: int, n1: int, k2: int, n2: int, z: float = CONFIDENCE_Z
) -> tuple[float, float] | None:
    """Wald interval for the difference of two binomial rates (a - b).

    The issue asks for "the difference between arms for each rate with an
    interval"; a difference interval (unlike two CIs compared visually)
    directly answers "is the gap distinguishable from zero".  Clamped to
    [-1, 1]; None when either denominator is 0.
    """
    if n1 <= 0 or n2 <= 0:
        return None
    p1, p2 = k1 / n1, k2 / n2
    diff = p1 - p2
    half = z * math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return (max(-1.0, diff - half), min(1.0, diff + half))


# ---------------------------------------------------------------------------
# Event scanning
# ---------------------------------------------------------------------------


def _pr_of(payload: Mapping[str, Any], event: Mapping[str, Any]) -> int | None:
    """The PR an event refers to: payload pr_number, then the indexed column."""
    for source in (payload, event):
        raw = source.get("pr_number") or source.get("pr")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return int(raw)
    return None


def _iter_arm_observations(
    event: Mapping[str, Any], metrics_key: str
) -> Iterable[tuple[int, str]]:
    """Yield ``(pr_number, arm_value)`` pairs carrying the experiment key.

    Three recorded shapes carry an arm value today, and the scan covers all
    of them generically so future recorders do not need a code change here:

    * ``payload["session_metrics"][key]`` -- per-round values inside
      ``record_review`` (folded from PR state at verdict reap time).
    * ``payload[key]`` -- a top-level arm field on any event.
    * ``payload[*]`` lists of dicts containing the key -- claim-batch
      shapes like ``review_effort_assignments: [{pr_number, <key>, ...}]``.
    """
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return
    session_metrics = payload.get("session_metrics")
    if isinstance(session_metrics, dict) and metrics_key in session_metrics:
        pr = _pr_of(payload, event)
        value = session_metrics[metrics_key]
        if pr is not None and isinstance(value, str) and value:
            yield (pr, value)
    if metrics_key in payload:
        pr = _pr_of(payload, event)
        value = payload[metrics_key]
        if pr is not None and isinstance(value, str) and value:
            yield (pr, value)
    for value in payload.values():
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict) or metrics_key not in item:
                continue
            arm = item[metrics_key]
            pr = item.get("pr_number") or item.get("pr")
            if (
                isinstance(pr, (int, float))
                and not isinstance(pr, bool)
                and isinstance(arm, str)
                and arm
            ):
                yield (int(pr), arm)


def _merge_prs(event: Mapping[str, Any]) -> set[int]:
    """PRs this event marks as merged (empty for non-merge events)."""
    if event["kind"] not in _MERGE_EVENT_KINDS:
        return set()
    payload = event.get("payload") or {}
    if not isinstance(payload, dict):
        return set()
    if event["kind"] == "reconcile" and payload.get("kind") != "merged_outside_orchestrator":
        return set()
    prs: set[int] = set()
    single = _pr_of(payload, event)
    if single is not None:
        prs.add(single)
    for raw in payload.get("pr_numbers") or ():
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            prs.add(int(raw))
    return prs


def _dispatch_prs(event: Mapping[str, Any], issue_to_prs: Mapping[int, set[int]]) -> set[int]:
    """PRs this event marks as dispatched/launched.

    ``dispatch`` names issues (resolved through the PR->issue map built
    from record_review payloads); ``review_dispatch`` and
    ``review_dispatch_claim`` name PRs directly.
    """
    payload = event.get("payload") or {}
    if not isinstance(payload, dict):
        return set()
    prs: set[int] = set()
    if event["kind"] == "dispatch":
        for raw in payload.get("issue_numbers") or ():
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                prs |= issue_to_prs.get(int(raw), set())
        return prs
    if event["kind"] == "review_dispatch":
        for key in ("launched", "failed"):
            for raw in payload.get(key) or ():
                if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                    prs.add(int(raw))
        return prs
    # review_dispatch_claim
    for raw in payload.get("pr_numbers") or ():
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            prs.add(int(raw))
    for value in payload.values():
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict):
                pr = item.get("pr_number") or item.get("pr")
                if isinstance(pr, (int, float)) and not isinstance(pr, bool):
                    prs.add(int(pr))
    return prs


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


def _evaluate_stopping_rule(
    arms: list[str],
    prs_assigned: dict[str, int],
    outcome_metric: dict[str, Any],
    min_prs_per_arm: int,
    z: float,
) -> dict[str, Any]:
    """The experiment's stopping rule, evaluated against this window's data.

    The rule (docs/review-effort-experiment.md):
    * every arm needs >= ``min_prs_per_arm`` assigned PRs before anything is
      decided;
    * once that holds, a ``post_approval_defect_rate`` difference interval
      that excludes 0 ends the experiment in favour of the lower-defect arm;
    * every arm at >= 2x the minimum with all outcome intervals still
      spanning 0 ends it as "no detectable difference";
    * anything else -- including a one-arm or zero-arm report -- is
      ``"not met"``.
    """
    equivalence_min = EQUIVALENCE_MULTIPLE * min_prs_per_arm
    rule: dict[str, Any] = {
        "min_prs_per_arm": min_prs_per_arm,
        "equivalence_min_prs_per_arm": equivalence_min,
        "outcome_metric": outcome_metric["name"],
        "confidence_z": z,
        "post_experiment_config": "reviewer.effort_experiment_fraction: 0.0",
        "met": False,
        "verdict": "not_met",
        "detail": "",
    }
    if len(arms) < 2:
        rule["detail"] = f"fewer than two arms observed ({len(arms)}); nothing to compare"
        return rule
    short = {a: n for a, n in prs_assigned.items() if n < min_prs_per_arm}
    if short:
        detail = ", ".join(f'arm "{a}" has {n} assigned PRs' for a, n in sorted(short.items()))
        rule["detail"] = f"minimum {min_prs_per_arm} assigned PRs per arm not reached: {detail}"
        return rule
    decided = [
        d
        for d in outcome_metric["differences"]
        if d["ci95"] and (d["ci95"][0] > 0 or d["ci95"][1] < 0)
    ]
    if decided:
        best = decided[0]
        a, b = best["pair"]
        # diff = rate_a - rate_b; a strictly positive interval means a is worse.
        better = b if best["diff"] > 0 else a
        worse = a if best["diff"] > 0 else b
        rule.update(
            met=True,
            verdict="difference_detected",
            detail=(
                f'post_approval_defect_rate differs: arm "{better}" is lower '
                f"(pair {a} vs {b}, diff {best['diff']:+.4f}, "
                f"95% CI [{best['ci95'][0]:+.4f}, {best['ci95'][1]:+.4f}]). "
                f'Adopt arm "{better}" (lower post-approval defect rate over '
                f'arm "{worse}") and set reviewer.effort_experiment_fraction: 0.0.'
            ),
        )
        return rule
    all_pairs_intervalled = bool(outcome_metric["differences"]) and all(
        d["ci95"] for d in outcome_metric["differences"]
    )
    if all(n >= equivalence_min for n in prs_assigned.values()) and all_pairs_intervalled:
        rule.update(
            met=True,
            verdict="no_detectable_difference",
            detail=(
                f"every arm has >= {equivalence_min} assigned PRs and every "
                "post_approval_defect_rate difference interval still spans 0. "
                "End the experiment (effort_experiment_fraction: 0.0); pick "
                "the arm with the lower review_cost_per_pr_usd_mean."
            ),
        )
        return rule
    rule["detail"] = (
        "minimum per-arm N reached, but the outcome-metric difference "
        "intervals still span 0 and the equivalence bound "
        f"({equivalence_min} PRs/arm) is not reached -- keep running."
    )
    return rule


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
        elif e["kind"] in _MERGE_EVENT_KINDS:
            for pr in _merge_prs(e):
                merged_prs.add(pr)
                if ts is not None and (pr not in merge_ts or ts < merge_ts[pr]):
                    merge_ts[pr] = ts
        elif e["kind"] in _DISPATCH_EVENT_KINDS:
            dispatch_events.append(e)
        if e["kind"].endswith(_ESCALATED_SUFFIX):
            pr = _pr_of(payload, e)
            if pr is not None:
                escalated_prs.add(pr)
        if e["kind"] in _POST_APPROVAL_DEFECT_KINDS:
            pr = _pr_of(payload, e)
            if pr is not None:
                defect_events.setdefault(pr, set()).add(e["kind"])
        if e["kind"] == "no_op_rework_repair_requested":
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
    first_decision_k: dict[str, dict[str, int]] = {a: {d: 0 for d in _DECISIONS} for a in arms}
    first_decision_other: dict[str, int] = {a: 0 for a in arms}
    rounds_per_pr: dict[str, list[float]] = {a: [] for a in arms}
    cost_per_pr: dict[str, list[float]] = {a: [] for a in arms}
    cost_rounds_total: dict[str, int] = {a: 0 for a in arms}
    cost_rounds_with: dict[str, int] = {a: 0 for a in arms}
    escalated_k: dict[str, int] = {a: 0 for a in arms}
    merged_k: dict[str, int] = {a: 0 for a in arms}
    d2m_seconds: dict[str, list[float]] = {a: [] for a in arms}
    approved_n: dict[str, int] = {a: 0 for a in arms}
    defect_k: dict[str, dict[str, int]] = {
        a: {kind: 0 for kind in sorted(_POST_APPROVAL_DEFECT_KINDS)} for a in arms
    }
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
        if first in _DECISIONS:
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
                defect_k[arm][kind] += 1
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
    for decision in _DECISIONS:
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
    defect_k_total = {a: sum(defect_k[a].values()) for a in arms}
    metrics["post_approval_defect_rate"] = _rate_metric(
        "post_approval_defect_rate",
        "outcome",
        "share of approved PRs later routed to rework by a post-approval "
        "defect signal (check failure on the approved head, or a silent "
        "base-commit revert in the branch)",
        arms,
        defect_k_total,
        approved_n,
        z,
    )
    metrics["post_approval_ci_failure_rate"] = _rate_metric(
        "post_approval_ci_failure_rate",
        "outcome",
        "share of approved PRs later routed to rework for genuinely failing "
        "required checks (check_failure_rework_requested)",
        arms,
        {a: defect_k[a]["check_failure_rework_requested"] for a in arms},
        approved_n,
        z,
    )
    metrics["post_approval_revert_bounce_rate"] = _rate_metric(
        "post_approval_revert_bounce_rate",
        "outcome",
        "share of approved PRs later routed to rework for a silent "
        "base-commit revert in the branch (cross_pr_revert_rework_requested)",
        arms,
        {a: defect_k[a]["cross_pr_revert_rework_requested"] for a in arms},
        approved_n,
        z,
    )
    metrics["no_op_kickback_rate"] = _rate_metric(
        "no_op_kickback_rate",
        "outcome",
        "share of request_changes PRs whose next rework produced no change "
        "relative to the verdict (no_op_rework_repair_requested) -- a "
        "kickback that found nothing real to fix",
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

    derivable = [
        "post_approval_defect_rate",
        "post_approval_ci_failure_rate",
        "post_approval_revert_bounce_rate",
        "no_op_kickback_rate",
    ]
    stopping = _evaluate_stopping_rule(
        arms,
        prs_assigned,
        metrics["post_approval_defect_rate"],
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
                "first_round_decisions": {d: first_decision_k[a][d] for d in _DECISIONS},
                "first_round_other_decisions": first_decision_other[a],
            }
            for a in arms
        },
        "metrics": metrics,
        "data_integrity_warnings": warnings,
        "prs_with_rounds_without_arm_value": unassigned_with_rounds,
        "outcome_coverage": {
            "derivable": derivable,
            "not_derivable": list(_NOT_DERIVABLE),
        },
        "stopping_rule": stopping,
    }


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------


def _fmt_rate(entry: Mapping[str, Any]) -> str:
    if entry["n"] == 0 or entry["rate"] is None:
        return f"0/{entry['n']} = n/a (no PRs in denominator)"
    ci = entry["ci95"]
    ci_s = f"  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
    return f"{entry['k']}/{entry['n']} = {entry['rate']:.3f}{ci_s}"


def _fmt_value(entry: Mapping[str, Any]) -> str:
    if entry["value"] is None:
        return f"n/a (n={entry['n']})"
    return f"{entry['value']:.3f} (n={entry['n']})"


def render_text(report: Mapping[str, Any]) -> str:
    """Render the report dict as the command's human-readable output."""
    lines: list[str] = []
    w = report["window"]
    window_bits = []
    if w["since"]:
        window_bits.append(f"since={w['since']}")
    if w["until"]:
        window_bits.append(f"until={w['until']}")
    for s, e in w["exclude_windows"]:
        window_bits.append(f"exclude=[{s} .. {e}]")
    lines.append(
        f'experiment report: session-metrics key "{report["metrics_key"]}" '
        "(read-only; no state was modified)"
    )
    lines.append(
        f"window: {', '.join(window_bits) if window_bits else 'unfiltered'}; "
        f"events scanned: {report['events_scanned']}"
    )
    if report.get("events_db"):
        lines.append(f"source: {report['events_db']}")
    lines.append(
        "unit of analysis: the PR (arm assignment is stable per PR across "
        "rounds; rates count each PR once)"
    )
    if not report["arms"]:
        lines.append(
            f'NO DATA: no PR carries a "{report["metrics_key"]}" value in the scanned window.'
        )
    for arm in report["arms"]:
        pa = report["per_arm"][arm]
        lines.append(
            f'arm "{arm}": {pa["prs_assigned"]} assigned, '
            f"{pa['prs_with_rounds']} with recorded rounds"
        )
    lines.append("")
    lines.append("metrics:")
    for name, m in report["metrics"].items():
        lines.append(f"  {name} [{m['measure']}]")
        lines.append(f"    {m['description']}")
        for arm in report["arms"]:
            entry = m["per_arm"].get(arm)
            if entry is None:
                continue
            if m["kind"] == "rate":
                rendered = _fmt_rate(entry)
            elif m["kind"] == "count":
                rendered = str(entry["n"])
            else:
                rendered = _fmt_value(entry)
            extra = ""
            if name == "review_cost_per_pr_usd_mean" and "rounds_total" in entry:
                extra = (
                    f"  (cost data on {entry['rounds_with_cost']}/{entry['rounds_total']} rounds)"
                )
            lines.append(f"    {arm}: {rendered}{extra}")
        for d in m["differences"]:
            a, b = d["pair"]
            lines.append(
                f"    diff {a} - {b} = {d['diff']:+.4f}  "
                f"95% CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
            )
    lines.append("")
    lines.append("outcome coverage:")
    lines.append(
        "  derivable and reported above: " + ", ".join(report["outcome_coverage"]["derivable"])
    )
    for item in report["outcome_coverage"]["not_derivable"]:
        lines.append(
            f"  NOT derivable from recorded events: {item['candidate']} ({item['reason']})"
        )
    warnings = report["data_integrity_warnings"]
    if warnings or report["prs_with_rounds_without_arm_value"]:
        lines.append("")
        lines.append("data integrity:")
        for wrn in warnings:
            lines.append(f"  PR #{wrn['pr']}: {wrn['detail']}")
        n = report["prs_with_rounds_without_arm_value"]
        if n:
            lines.append(
                f"  {n} PR(s) with recorded rounds carried no "
                f'"{report["metrics_key"]}" value (outside the experiment)'
            )
    lines.append("")
    rule = report["stopping_rule"]
    status = "MET" if rule["met"] else "NOT MET"
    lines.append(f"stopping rule: {status} ({rule['verdict']})")
    lines.append(
        f"  min PRs/arm: {rule['min_prs_per_arm']} "
        f"(equivalence bound: {rule['equivalence_min_prs_per_arm']}/arm); "
        f"outcome metric: {rule['outcome_metric']}"
    )
    lines.append(f"  {rule['detail']}")
    lines.append(
        f"  when met: set {rule['post_experiment_config']} (docs/review-effort-experiment.md)"
    )
    return "\n".join(lines)
