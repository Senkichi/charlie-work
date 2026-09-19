"""Event scanning for the per-PR experiment read-out (issue #1701).

Extracted from ``experiment_report.py`` under the repo's 800-line module
cap.  This module owns the event vocabulary the report consumes -- the
``record_review`` decision names, the merge/dispatch event kinds that
timestamp a PR's pipeline entry and exit, the post-approval rework
signals, and the payload shapes that carry an arm assignment -- plus the
helpers that pull PR numbers and ``(pr, arm)`` observations out of raw
event dicts.

## Post-approval signal taxonomy

Post-approval rework events are partitioned by *what they measure*,
because only review-correctness signals may drive the stopping rule:

* ``cross_pr_revert_rework_requested`` -- an approved branch was found to
  silently revert a base commit.  Approving that branch was a review
  miss: this is the only derivable **outcome** signal today.
* ``check_failure_rework_requested`` -- required checks genuinely failed
  on the approved head (``merge_ready`` path, issue #674).  The rework
  brief it emits says "The code changes are already approved; do not
  re-litigate the review": the emitter itself states the approval stands,
  so the event is **pipeline state, not a review-correctness measure**.
  A check can fail after a correct approval for reasons the review was
  never asked to judge (a flaky test, a base-branch drift, a broken
  harness).  It is reported as activity and excluded from the outcome
  aggregate and the stopping rule.
* ``no_op_rework_repair_requested`` -- the rework *cycle* produced no
  content change relative to the last ``request_changes`` verdict
  (unchanged patch-id/head).  That measures a stalled or empty rework
  cycle -- unpushed commits, a dead session, or genuinely nothing left
  to change -- not that the kickback was wrong.  Reported as activity.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

# The record_review decision vocabulary (verdict_parsing.py).
DECISIONS = ("approved", "request_changes", "blocked")

# Event kinds naming a PR's merge completion (the report takes the earliest
# timestamp across them).  `reconcile` is included but only counts when its
# payload kind is `merged_outside_orchestrator`.
MERGE_EVENT_KINDS = frozenset(
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
DISPATCH_EVENT_KINDS = frozenset(
    {
        "dispatch",
        "review_dispatch",
        "review_dispatch_claim",
    }
)

ESCALATED_SUFFIX = "_escalated"

# The post-approval classification frozensets (REVIEW_DEFECT_KINDS,
# PIPELINE_REWORK_KINDS, POST_APPROVAL_SIGNAL_KINDS, NO_OP_REWORK_KINDS)
# live in experiment_report.py, the module that compares them against
# event kinds: tests/test_event_kind_consumers.py counts a kind as
# consumed only when its literal sits in a read position resolvable
# within the same file, so the sets cannot be imported from here.

# The explicit statement the report emits when no outcome metric is
# derivable from the recorded events -- the issue's contract, so the
# report never implies a conclusion from activity metrics alone.
NO_OUTCOME_STATEMENT = (
    "no outcome metric is derivable from the recorded event stream; the "
    "stopping rule cannot be met on activity metrics alone"
)

# Outcome candidates the recorded event stream cannot produce today.  The
# report states this explicitly rather than implying a conclusion from
# activity metrics (issue #1701's contract); issue #1717 tracks adding the
# missing recording.
NOT_DERIVABLE: tuple[dict[str, str], ...] = (
    {
        "candidate": "post-merge main-branch CI failure attributed to the merged PR",
        "reason": (
            "no recorded event links a main-branch check failure after a merge "
            "back to the PR that merged (recording tracked by issue #1717)"
        ),
    },
    {
        "candidate": "follow-up fix or revert naming the merged PR",
        "reason": (
            "no recorded event attributes a later fix or revert commit/PR to a "
            "previously merged PR (recording tracked by issue #1717)"
        ),
    },
)


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
    if event["kind"] not in MERGE_EVENT_KINDS:
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
