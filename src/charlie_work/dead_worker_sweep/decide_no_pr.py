"""Decisions for dead workers that have no open PR.

``triage_no_pr`` and ``probe_no_pr`` are the pre-lock halves (fate resolution,
the three operator-queue escalations, the park/reclaim lane, the pushed-branch
probes); ``lock_no_pr_flow`` is the in-lock half (PR-open lane, escalation
bookkeeping, the #1243 redispatch cap, reclaim reporting, drift backfill).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .constants import PASSIVE_OPEN_STATUS
from .decide_common import (
    Draft,
    Flow,
    drift_fingerprint,
    emit,
    flush,
    label_names,
    session_failed_relabeled_payload,
)
from .decide_redispatch import decide_redispatch_cap
from .model import (
    Escalate,
    FateResult,
    FetchOpenIssues,
    OpenPrForBranch,
    ParkBackstop,
    ParkOrReclaim,
    PreOutcome,
    ProbeCrossRepoScope,
    ProbeRemoteBranch,
    ProbeWorktreeHead,
    ProbeZeroArtifact,
    RemoteProbe,
    ResolveFate,
    StripAndFlag,
    SweepFacts,
    UpdateIssue,
)


@dataclass(frozen=True)
class NoPrTriage:
    issues: Mapping[int, Mapping[str, Any]]
    fates: Mapping[int, FateResult]
    escalations: Mapping[int, tuple[str, Mapping[str, Any]]]
    deferred: Mapping[int, str]
    reclaim_results: Mapping[int, Mapping[str, Any]]


def triage_no_pr(facts: SweepFacts, no_pr: tuple[int, ...]) -> Flow:
    """Resolve fates, escalate the structurally hopeless, park or reclaim the rest."""
    active_set = facts.config.labels.active
    issues = (yield FetchOpenIssues("no_pr")).issues_by_number
    fates: dict[int, FateResult] = {}
    for number in no_pr:
        if number in issues:
            fates[number] = yield ResolveFate(number, "no_pr")

    escalations: dict[int, tuple[str, Mapping[str, Any]]] = {}
    for number in no_pr:
        issue = issues.get(number)
        if issue is None:
            continue
        active = label_names(issue) & active_set
        if not active:
            continue
        removed = sorted(active)
        fate = fates[number]
        if fate.blocked_escalatable:
            written = yield StripAndFlag(number, "worker_declared_blocked")
            escalations[number] = (
                "worker_declared_blocked",
                {
                    "removed_labels": removed,
                    "label_write_ok": written.ok,
                    "reason_kind": fate.blocked_reason_kind,
                    "detail": fate.blocked_detail,
                },
            )
            continue
        zero_artifact = False
        if not fate.throttled:
            zero_artifact = yield ProbeZeroArtifact(number)
        if zero_artifact:
            written = yield StripAndFlag(number, "zero_artifact")
            escalations[number] = (
                "zero_artifact",
                {"removed_labels": removed, "label_write_ok": written.ok},
            )
            continue
        scope = yield ProbeCrossRepoScope(number)
        if not scope.passed:
            written = yield StripAndFlag(number, "cross_repo_scope")
            escalations[number] = (
                "cross_repo_scope",
                {
                    "removed_labels": removed,
                    "label_write_ok": written.ok,
                    "reason": scope.reason,
                },
            )
            continue
        yield ParkOrReclaim(number)

    backstop = yield ParkBackstop(orphans=tuple(no_pr), escalated=tuple(escalations))
    return NoPrTriage(
        issues=issues,
        fates=fates,
        escalations=escalations,
        deferred=dict(backstop.deferred),
        reclaim_results=dict(backstop.reclaim_results),
    )


def probe_no_pr(facts: SweepFacts, no_pr: tuple[int, ...], triage: NoPrTriage) -> Flow:
    """Branch-head probes and the pushed-without-PR fate. Returns ``(details, candidates)``."""
    active_set = facts.config.labels.active
    details: dict[int, dict[str, Any]] = {}
    candidates: dict[int, dict[str, Any]] = {}
    for number in no_pr:
        issue = triage.issues.get(number)
        if issue is None:
            continue
        labels = label_names(issue)
        fate = triage.fates[number]
        branch = fate.branch
        worker_outcome = fate.worker_outcome
        reported_push = (
            isinstance(worker_outcome, Mapping)
            and worker_outcome.get("push_succeeded") is True
            and worker_outcome.get("pr_created") is False
        )
        probe = RemoteProbe(ahead_count=None, ahead_error=None, head_sha=None)
        if facts.repo.has_repo_root:
            probe = yield ProbeRemoteBranch(number, branch)
        local_head_sha = None
        if facts.repo.has_repo_root and facts.repo.has_worktrees:
            local_head_sha = yield ProbeWorktreeHead(number, branch)
        details[number] = {
            "issue_labels": sorted(labels),
            "active_labels": sorted(labels & active_set),
            "branch": branch,
            "remote_head_sha": probe.head_sha,
            "local_head_sha": local_head_sha,
        }
        pushed = yield ResolveFate(
            number, "pushed", remote_head_sha=probe.head_sha, ahead_count=probe.ahead_count
        )
        if pushed.pushed_without_pr:
            candidates[number] = {
                "branch": branch,
                "reported_push": reported_push,
                "ahead_count": probe.ahead_count,
                "ahead_error": probe.ahead_error,
            }
    return details, candidates


def lock_no_pr_flow(facts: SweepFacts, pre: PreOutcome, draft: Draft) -> Flow:
    """The in-lock no-open-PR arm. Returns True when the issue joined ``reap_escalations``."""
    number = draft.issue
    entry = draft.work
    config = facts.config
    stamp = facts.stamp

    candidate = pre.candidates.get(number)
    if candidate is not None:
        opened = yield OpenPrForBranch(number, candidate["branch"], "pushed_orphan")
        if opened.pr_number is not None:
            entry["status"] = PASSIVE_OPEN_STATUS
            entry["pr_number"] = opened.pr_number
            payload = {
                "issue_number": number,
                "pr_number": opened.pr_number,
                "branch_name": candidate["branch"],
                "worker_reported": candidate["reported_push"],
                "ahead_count": candidate["ahead_count"],
                "previous_status": "dispatched",
                "label_write_ok": opened.error is None,
                "pr_error": opened.error,
            }
            if candidate["reported_push"]:
                yield emit(
                    "worker_handoff_pr_opened", {**payload, "reason": "worker_handoff_clean_exit"}
                )
            else:
                yield emit(
                    "orphaned_worker_opened_pr",
                    {**payload, "reason": "dead_worker_branch_pushed_no_pr"},
                )
            return False
        fingerprint = drift_fingerprint(
            reason="dead_worker_branch_pushed_pr_create_failed",
            branch_name=candidate["branch"],
            error=opened.error or "unknown",
        )
        if entry.get("orphan_drift_fingerprint") == fingerprint:
            return False
        entry["orphan_drift_fingerprint"] = fingerprint
        entry["orphan_drift_at"] = stamp
        yield emit(
            "pr_create_failed_branch_stranded",
            {
                "issue_number": number,
                "branch_name": candidate["branch"],
                "previous_status": "dispatched",
                "reason": "dead_worker_branch_pushed_pr_create_failed",
                "pr_create_error": opened.error,
                "worker_reported": candidate["reported_push"],
                "ahead_count": candidate["ahead_count"],
            },
        )
        return False

    escalation = pre.escalations.get(number)
    if escalation is not None:
        kind, detail = escalation
        yield from flush(draft)
        reason = {
            "worker_declared_blocked": "worker_declared_blocked",
            "zero_artifact": "zero_artifact_dispatch_loop",
            "cross_repo_scope": "cross_repo_hop",
        }[kind]
        yield Escalate(number, reason=reason, reason_class="mechanical")
        draft.invalidate()
        yield UpdateIssue(number, {"orphan_flagged_at": stamp})
        if kind == "worker_declared_blocked":
            yield emit(
                "worker_declared_blocked",
                {
                    "issue_number": number,
                    "previous_status": "dispatched",
                    "reason": "worker_declared_blocked",
                    "reason_kind": detail["reason_kind"],
                    "detail": detail["detail"],
                    "removed_labels": detail["removed_labels"],
                    "label_write_ok": detail["label_write_ok"],
                },
            )
            return True
        yield emit(
            "session_failed_escalated",
            session_failed_relabeled_payload(
                issue_number=number,
                reason=reason,
                removed_labels=detail["removed_labels"],
                added_ready=False,
                label_write_ok=detail["label_write_ok"],
            ),
        )
        return False

    head_details = pre.details.get(number, {})
    verdict = decide_redispatch_cap(
        entry,
        head_details=head_details,
        window_minutes=config.watchdog.redispatch_window_minutes,
        max_auto_redispatch=config.watchdog.max_auto_redispatch,
        stamp=stamp,
        # Sweep-start clock: decide reads no clock of its own. The original sampled wall
        # time at this call; the skew (seconds) is negligible against the 240-minute window.
        now=pre.now,
    )
    history = list(verdict.orphan_redispatch_at)
    if verdict.exceeded:
        yield from flush(draft)
        yield Escalate(
            number,
            reason="orphan_sweep_redispatch_cap_exceeded",
            reason_class="mechanical",
            issue_extra={
                "dispatched_at": None,
                "orphan_redispatch_head_sha": verdict.current_head,
                "orphan_redispatch_at": history,
                "orphan_redispatch_counted_dispatch": None,
                "orphan_flagged_at": None,
                "orphan_drift_fingerprint": None,
                "orphan_drift_at": None,
            },
        )
        draft.invalidate()
        yield emit(
            "orphan_sweep_redispatch_escalated",
            {
                "issue_number": number,
                "previous_status": "dispatched",
                "reason": "orphan_sweep_redispatch_cap_exceeded",
                "redispatch_count": verdict.redispatch_count,
                "branch_head_sha": head_details.get("remote_head_sha"),
                "worktree_head_sha": head_details.get("local_head_sha"),
                "orphan_redispatch_at_len": len(history),
            },
        )
        return True

    entry["orphan_redispatch_head_sha"] = verdict.current_head
    entry["orphan_redispatch_at"] = history
    entry["orphan_redispatch_counted_dispatch"] = verdict.dispatch_identity

    reclaim = pre.reclaim_results.get(number)
    if reclaim is not None:
        yield emit(
            "session_failed_relabeled",
            session_failed_relabeled_payload(
                issue_number=number,
                reason="dead_worker_no_open_pr_orphan_sweep",
                **reclaim,
            ),
        )
        if reclaim["label_write_ok"]:
            entry["orphan_flagged_at"] = stamp
            return False

    if entry.get("orphan_drift_at") is None and entry.get("orphan_flagged_at") is not None:
        entry["orphan_drift_at"] = entry["orphan_flagged_at"]
    if entry.get("orphan_flagged_at"):
        return False
    entry["orphan_flagged_at"] = stamp
    entry["orphan_drift_at"] = stamp
    yield emit(
        "orphaned_worker_drift",
        {
            "issue_number": number,
            "previous_status": "dispatched",
            "reason": "dead_worker_no_open_pr",
        },
    )
    return False
