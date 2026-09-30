"""The pre-lock phase: everything the sweep decides while it may still touch the network.

``pre_flow`` mirrors the first half of ``_detect_and_handle_orphaned_workers``:
stale live-PID handoffs, the open-PR map, the no-open-PR triage, salvage pushes,
branch-head probes, pre-review rework routing (#439), the unverdicted-PR label
read (#1128) and live-handoff candidate resolution (#1867).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .decide_common import Flow, is_pre_review_rework_candidate, label_names
from .decide_no_pr import NoPrTriage, probe_no_pr, triage_no_pr
from .model import (
    CollectLiveHandoff,
    FetchOpenIssues,
    FetchOpenPrs,
    FetchPrView,
    PreOutcome,
    ReadClock,
    ReadReviewDecision,
    ReportStaleEvidence,
    RoutePreReviewRework,
    SalvagePush,
    SweepFacts,
)

_EMPTY_TRIAGE = NoPrTriage(issues={}, fates={}, escalations={}, deferred={}, reclaim_results={})


def dead_orphans(facts: SweepFacts) -> tuple[int, ...]:
    """Dispatched issues whose worker PID is gone, in state order."""
    found: list[int] = []
    for key, entry in (facts.snapshot.get("issues") or {}).items():
        if not isinstance(entry, dict) or entry.get("status") != "dispatched":
            continue
        number = int(key)
        if facts.pid_alive.get(number) is False:
            found.append(number)
    return tuple(found)


def _early_exit(facts: SweepFacts, orphans: tuple[int, ...]) -> PreOutcome:
    return PreOutcome(
        orphans=orphans,
        no_pr_orphans=(),
        pr_by_issue={},
        details={},
        candidates={},
        escalations={},
        deferred={},
        reclaim_results={},
        unreviewed={},
        live_candidates={},
        pr_already_open={},
        salvage_events=(),
        heads={},
        early_exit=True,
        now=facts.now,
    )


def _salvage_flow(
    facts: SweepFacts,
    orphans: tuple[int, ...],
    pr_by_issue: Mapping[int, Mapping[str, Any]],
) -> Flow:
    """#1248: publish stranded commits. Returns ``(events, heads)``."""
    events: list[tuple[str, dict[str, Any]]] = []
    heads: dict[int, str | None] = {
        n: (pr_by_issue[n].get("headRefOid") if n in pr_by_issue else None) for n in orphans
    }
    if not (facts.repo.has_repo_root and facts.repo.has_worktrees):
        return tuple(events), heads
    snapshot_issues = facts.snapshot.get("issues") or {}
    for number in orphans:
        pr = pr_by_issue.get(number)
        if pr is not None and pr.get("isCrossRepository"):
            continue
        entry = snapshot_issues.get(str(number), {})
        branch = pr.get("headRefName") if pr is not None else None
        if not branch and isinstance(entry, dict):
            branch = entry.get("branch_name")
        if not branch:
            continue
        result = yield SalvagePush(number, branch)
        pr_number = int(pr["number"]) if pr is not None else None
        payload: dict[str, Any] = {
            "issue_number": number,
            "pr_number": pr_number,
            "branch": branch,
            "old_remote_sha": result.old_remote_sha,
            "commit_count": result.commit_count,
        }
        if result.pushed:
            payload["new_remote_sha"] = result.new_remote_sha
            events.append(("salvage_pushed_stranded_commits", payload))
            if pr is not None and result.new_remote_sha:
                heads[number] = result.new_remote_sha
        elif result.error:
            payload["error"] = result.error
            events.append(("salvage_push_failed", payload))
    return tuple(events), heads


def _pre_review_flow(
    facts: SweepFacts,
    orphans: tuple[int, ...],
    pr_by_issue: Mapping[int, Mapping[str, Any]],
    heads: Mapping[int, str | None],
) -> Flow:
    """#439: route dead workers with stuck pre-review PRs to rework. Returns the refreshed clock."""
    now = yield ReadClock()
    stale_minutes = facts.config.watchdog.pre_review_rework_stale_minutes
    for number in orphans:
        pr = pr_by_issue.get(number)
        if not pr:
            continue
        pr_number = int(pr["number"])
        resolved = yield ReadReviewDecision(pr_number, heads[number], "pre_review")
        if resolved.decision == "request_changes" and not resolved.stale:
            continue
        view = yield FetchPrView(pr_number)
        is_candidate, reason = is_pre_review_rework_candidate(view or pr, stale_minutes, now)
        if is_candidate:
            yield RoutePreReviewRework(number, pr_number, reason)
    return now


def _unreviewed_flow(
    facts: SweepFacts,
    orphans: tuple[int, ...],
    pr_by_issue: Mapping[int, Mapping[str, Any]],
    heads: Mapping[int, str | None],
) -> Flow:
    """#1128: active labels of dead-worker PRs that carry no verdict yet."""
    waiting: list[int] = []
    for number in orphans:
        if number not in pr_by_issue:
            continue
        resolved = yield ReadReviewDecision(
            int(pr_by_issue[number]["number"]), heads[number], "unreviewed"
        )
        if resolved.missing or resolved.decision == "pending":
            waiting.append(number)
    unreviewed: dict[int, tuple[str, ...]] = {}
    if waiting:
        issues = (yield FetchOpenIssues()).issues_by_number
        for number in waiting:
            issue = issues.get(number)
            if issue is not None:
                active = label_names(issue) & facts.config.labels.active
                unreviewed[number] = tuple(sorted(active))
    return unreviewed


def pre_flow(facts: SweepFacts) -> Flow:
    orphans = dead_orphans(facts)
    found = yield CollectLiveHandoff()
    stale = dict(found.candidates)
    yield ReportStaleEvidence("live_handoff")
    if not orphans and not stale:
        return _early_exit(facts, orphans)

    pr_by_issue = yield FetchOpenPrs()
    no_pr = tuple(n for n in orphans if n not in pr_by_issue)
    triage = _EMPTY_TRIAGE
    if no_pr:
        triage = yield from triage_no_pr(facts, no_pr)
    salvage_events, heads = yield from _salvage_flow(facts, orphans, pr_by_issue)
    details, candidates = yield from probe_no_pr(facts, no_pr, triage)
    yield ReportStaleEvidence("no_pr")

    now = yield from _pre_review_flow(facts, orphans, pr_by_issue, heads)
    unreviewed = yield from _unreviewed_flow(facts, orphans, pr_by_issue, heads)

    pr_already_open = {n: c for n, c in stale.items() if n in pr_by_issue}
    declared = {n: c for n, c in stale.items() if n not in pr_by_issue}
    live_candidates: dict[int, Mapping[str, Any]] = {}
    if declared:
        issues = (yield FetchOpenIssues()).issues_by_number
        live_candidates = {n: c for n, c in declared.items() if n in issues}

    return PreOutcome(
        orphans=orphans,
        no_pr_orphans=no_pr,
        pr_by_issue=pr_by_issue,
        details=details,
        candidates=candidates,
        escalations=triage.escalations,
        deferred=triage.deferred,
        reclaim_results=triage.reclaim_results,
        unreviewed=unreviewed,
        live_candidates=live_candidates,
        pr_already_open=pr_already_open,
        salvage_events=salvage_events,
        heads=heads,
        early_exit=False,
        now=now,
    )
