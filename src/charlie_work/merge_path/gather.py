"""Gather steps of the **Merge path**: reads only, in the order the legacy body read.

Each function observes one stage's facts and hands them to the matching
``decide_*``. A read happens only when the stage below would consume it (the
guards live in ``decide.py`` next to the decisions), so a PR that stops at the
conflict gate never costs a ``pr_list`` or a base-currency probe. Nothing here
writes: state is read through ``MergePathPorts.load_state_locked`` and every
other read goes through the app's own read helpers, so ``setattr(app, ...)``
patches keep working.

``mode`` exists only for the two read-breadth differences the dry-run preview
has always had (``"preview"``): it gates the base-currency check for
front_of_train/broadcast only, and it reads the merge-hold whenever a merge
could be attempted rather than only when a mergequeue label is set.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Literal

from ..checks import CheckSummary, summarize_checks
from ..cross_pr_revert import CrossPrRevertStatus
from ..escalation import _escalation_flags
from ..github import GitHubError, label_names
from ..host import current as _host_current
from ..janitor import check_operator_containment
from .decide import (
    SYNC_STRATEGIES,
    branch_precheck,
    decide_admission,
    decide_branch,
    decide_readiness,
    is_already_in_mergequeue,
    needs_carry_forward,
)
from .issue_labels import open_issue_labels
from .model import (
    UNAVAILABLE,
    Admission,
    AdmissionFacts,
    BranchFacts,
    BranchGate,
    HoldFacts,
    MergePathConfig,
    PersistedPr,
    PlanKind,
    Readiness,
    ReadinessFacts,
    RevertStatus,
    StageKind,
    SyncOutcome,
    VerdictFact,
)
from .ports import MergePathPorts

if TYPE_CHECKING:
    from ..config import OrchestratorConfig

Mode = Literal["live", "preview"]

_REVERT_STATUS = {
    CrossPrRevertStatus.CLEAN: RevertStatus.CLEAN,
    CrossPrRevertStatus.REVERT_DETECTED: RevertStatus.DETECTED,
    CrossPrRevertStatus.UNDETERMINED: RevertStatus.UNDETERMINED,
}


# --------------------------------------------------------------------------- #
# Small value builders
# --------------------------------------------------------------------------- #


def config_slice(config: OrchestratorConfig) -> MergePathConfig:
    """The config fields the decisions read, copied once per pass."""
    auto = config.auto_merge
    return MergePathConfig(
        auto_merge_enabled=auto.enabled,
        mergequeue_label=auto.mergequeue_label,
        update_branch_strategy=auto.update_branch_strategy,
        require_approved_review=auto.require_approved_review,
        failed_attempt_alarm=auto.failed_attempt_alarm,
        max_conflict_rework_attempts=config.review.max_conflict_rework_attempts,
        human_merge_labels=tuple(config.dispatch.human_merge_labels),
        review_dispatch_enabled=config.review_dispatch.enabled,
        required_checks=tuple(auto.required_checks),
        readiness_no_ci_minutes=auto.readiness_no_ci_minutes,
        merge_strategy=auto.strategy,
        skip_line_label=auto.mergequeue_skip_line_label,
        priority_prefix=config.labels.priority_prefix,
    )


def persisted_from(entry: dict[str, Any] | None) -> PersistedPr:
    """View of ``state.prs[n]`` (``None`` / missing entry = a fresh PR)."""
    entry = entry or {}
    return PersistedPr(
        status=entry.get("status"),
        mergequeue_revoked_reason=entry.get("mergequeue_revoked_reason"),
        failed_attempts=entry.get("consecutive_failed_merge_attempts", 0),
        stale_base_deferrals=entry.get("consecutive_stale_base_deferrals", 0),
        mergequeue_since=entry.get("mergequeue_since"),
        mergequeue_head_sha=entry.get("mergequeue_head_sha"),
        mergequeue_checked_at=entry.get("mergequeue_checked_at"),
    )


def verdict_fact(decision: dict[str, Any]) -> VerdictFact:
    return VerdictFact(
        approved=decision.get("decision") == "approved",
        reviewed_head_sha=decision.get("reviewed_head_sha"),
        raw=MappingProxyType(decision),
    )


def escalation_from(
    snapshot: dict[str, Any], pr_number: int, issue_number: int | None
) -> tuple[bool, bool]:
    """``(pr_escalated, issue_escalated)`` from a state snapshot."""
    return _escalation_flags(
        snapshot.get("prs", {}).get(str(pr_number), {}),
        snapshot.get("issues", {}).get(str(issue_number), {})
        if issue_number is not None
        else None,
    )


# --------------------------------------------------------------------------- #
# Stage 1: admission
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Opening:
    """Everything stage 1 observed, kept for the renderers and later stages."""

    facts: AdmissionFacts
    admission: Admission
    pr: dict[str, Any]
    decision: dict[str, Any]
    issue_number: int | None
    # The carry-forward verdict, when it was read (live needs its tier/ids).
    carry_check: Any = None


def gather_skip(pr_number: int, cfg: MergePathConfig, entry: dict[str, Any]) -> Opening | None:
    """The SKIP opening for a PR already recorded as merged, else ``None``.

    Reads nothing from GitHub, so the live driver can ask it inside its first
    state lock before any network call.
    """
    persisted = persisted_from(entry)
    if persisted.status != "merged":
        return None
    facts = AdmissionFacts(
        pr_number=pr_number,
        config=cfg,
        persisted=persisted,
        pr_found=False,
        live_labels=frozenset(),
        live_head_sha=None,
        issue_number=entry.get("issue_number"),
        verdict=VerdictFact(False, None),
        carry_forward=None,
    )
    return Opening(facts, decide_admission(facts), {}, {}, entry.get("issue_number"))


def gather_opening(
    app: Any,
    ports: MergePathPorts,
    pr_number: int,
    cfg: MergePathConfig,
    entry: dict[str, Any],
    *,
    escalation: Callable[[int | None], tuple[bool, bool]] | None = None,
    preview: bool = False,
) -> Opening:
    """Skip / not-found / carry-forward / re-review facts for one PR.

    ``escalation`` is read lazily, only when the first verdict is a re-review
    (the one place admission consumes the flags); the preview never passes it.
    """
    skipped = gather_skip(pr_number, cfg, entry)
    if skipped is not None:
        return skipped
    persisted = persisted_from(entry)

    pr = app.gh.pr_view(pr_number)
    if not pr:
        facts = AdmissionFacts(
            pr_number=pr_number,
            config=cfg,
            persisted=persisted,
            pr_found=False,
            live_labels=frozenset(),
            live_head_sha=None,
            issue_number=None,
            verdict=VerdictFact(False, None),
            carry_forward=None,
        )
        return Opening(facts, decide_admission(facts), {}, {}, None)

    issue_number = ports.linked_issue_number(
        pr,
        is_cross_repository=pr.get("isCrossRepository"),
        branch_prefix=app.config.dispatch.branch_prefix,
        branch_issue_validator=app._make_branch_issue_validator(),
    )
    decision = app._review_decision(pr_number)
    verdict = verdict_fact(decision)
    live_head = pr.get("headRefOid")
    check = None
    carry: bool | None = None
    if needs_carry_forward(verdict, live_head):
        check = app._check_carry_forward(pr_number, decision)
        carry = bool(check.carry_forward)
    facts = AdmissionFacts(
        pr_number=pr_number,
        config=cfg,
        persisted=persisted,
        pr_found=True,
        live_labels=frozenset(label_names(pr)),
        live_head_sha=live_head,
        issue_number=issue_number,
        verdict=verdict,
        carry_forward=carry,
    )
    admission = decide_admission(facts, preview_unknown_live_head_proceeds=preview)
    if admission.kind is PlanKind.REQUEST_REWORK and escalation is not None:
        pr_esc, issue_esc = escalation(issue_number)
        facts = replace(facts, pr_escalated=pr_esc, issue_escalated=issue_esc)
        admission = decide_admission(facts)
    return Opening(facts, admission, pr, decision, issue_number, check)


# --------------------------------------------------------------------------- #
# Stage 2: branch gate
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BranchRead:
    facts: BranchFacts
    gate: BranchGate


def gather_branch(
    app: Any,
    ports: MergePathPorts,
    opening: Opening,
    *,
    train_head_param: int | None,
    mode: Mode,
) -> BranchRead:
    """Conflict, train head and base-currency facts, then the first ``decide_branch``.

    The live call can answer ``request_sync``; the preview, which never
    syncs, gets the verdict for ``SyncOutcome.NOT_ATTEMPTED`` directly.
    """
    adm = opening.admission
    f0 = opening.facts
    cfg = f0.config
    pr = opening.pr
    pr_number = f0.pr_number
    persisted = f0.persisted
    facts = BranchFacts(
        pr_number=pr_number,
        config=cfg,
        admission=adm,
        persisted=persisted,
        merge_conflict=False,
        issue_status=None,
        train_head_param=train_head_param,
    )
    if not adm.approved:
        return BranchRead(facts, decide_branch(facts))

    conflict = bool(app._is_merge_conflict(pr))
    issue_status: str | None = None
    if conflict and mode == "live" and opening.issue_number is not None:
        snap = ports.load_state_locked(app.paths.state_file)
        issue_status = snap.get("issues", {}).get(str(opening.issue_number), {}).get("status")
    facts = replace(facts, merge_conflict=conflict, issue_status=issue_status)
    stop, _, sync_failed = branch_precheck(facts)
    if stop is not None:
        return BranchRead(facts, stop)

    strategy = cfg.update_branch_strategy
    if strategy == "front_of_train" and train_head_param is None:
        try:
            listed = app.gh.pr_list()
        except GitHubError:
            facts = replace(facts, train_head=UNAVAILABLE)
        else:
            facts = replace(facts, train_head=app._merge_train_head(listed))
        stop, _, sync_failed = branch_precheck(facts)
        if stop is not None:
            return BranchRead(facts, stop)

    if not sync_failed and strategy != "off":
        gated = bool(
            app._is_base_currency_gated(pr.get("baseRefName") or app.config.runners.default_branch)
        )
        if mode == "preview":
            # Legacy dry-run drift (D-b): only the sync strategies are gated.
            gated = gated and strategy in SYNC_STRATEGIES
        facts = replace(facts, base_currency_gated=gated)
        if gated:
            base_current = app._is_base_current(pr)
            facts = replace(facts, base_current_read=True, base_current=base_current)
            if mode == "live" and strategy in SYNC_STRATEGIES:
                if not is_already_in_mergequeue(persisted, adm):
                    facts = replace(
                        facts, should_update_branch=app._should_update_pr_branch(pr, base_current)
                    )
    if mode == "preview":
        return BranchRead(facts, decide_branch(facts, SyncOutcome.NOT_ATTEMPTED))
    return BranchRead(facts, decide_branch(facts))


def regather_after_sync(
    app: Any,
    read: BranchRead,
    *,
    pr_before: dict[str, Any],
    pr_after: dict[str, Any],
    outcome: SyncOutcome,
) -> BranchGate:
    """Second ``decide_branch`` call, with the sync outcome and a fresh freshness read.

    ``base_current`` is re-read only when the sync moved the head, exactly
    as the legacy body re-read it.
    """
    facts = read.facts
    if (
        outcome is not SyncOutcome.FAILED
        and facts.base_currency_gated
        and pr_after.get("headRefOid") != pr_before.get("headRefOid")
    ):
        facts = replace(facts, base_current_read=True, base_current=app._is_base_current(pr_after))
    return decide_branch(facts, outcome)


# --------------------------------------------------------------------------- #
# Stage 3: readiness
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChecksRead:
    summary: CheckSummary
    unavailable: bool
    enriched: list[dict[str, Any]]
    containment_warnings: tuple[str, ...]


def gather_checks(app: Any, pr_number: int) -> ChecksRead:
    """``pr_checks`` + enrichment + the operator-containment scan of the diff."""
    required = app.config.auto_merge.required_checks
    checks = app.gh.pr_checks(pr_number)
    unavailable = checks is None
    if unavailable:
        summary = summarize_checks(None, required)
        enriched: list[dict[str, Any]] = []
    else:
        # Issue #1383: shared data-boundary enrichment.
        enriched = app._enrich_checks_infra_blocked(checks, required)
        summary = summarize_checks(enriched, required)
    diff = app.gh.pr_diff(pr_number)
    warnings = check_operator_containment(app.repo_root, diff, pr_number)
    return ChecksRead(summary, unavailable, enriched, tuple(warnings))


@dataclass(frozen=True)
class RevertRead:
    status: RevertStatus
    reason: str | None


def gather_revert(
    app: Any, ports: MergePathPorts, opening: Opening, gate: BranchGate
) -> RevertRead:
    """Cross-PR revert gate; only ever run for an approved PR whose sync has not failed."""
    if not (opening.admission.approved and not gate.sync_failed):
        return RevertRead(RevertStatus.CLEAN, None)
    verdict = ports.detect_cross_pr_revert(opening.pr, app.repo_root)
    status = _REVERT_STATUS.get(verdict.status, RevertStatus.UNDETERMINED)
    if verdict.blocks_merge and status is RevertStatus.CLEAN:
        status = RevertStatus.UNDETERMINED
    return RevertRead(status, verdict.reason)


def readiness_facts(
    opening: Opening,
    gate: BranchGate,
    revert: RevertRead,
    checks: ChecksRead,
    *,
    now: datetime | None = None,
    issue_status: str | None = None,
    issue_reason_class: str | None = None,
    human_merge: tuple[bool, bool] = (False, False),
) -> ReadinessFacts:
    """Assemble stage-3 facts. Escalation is left unread: infra remediation re-reads it itself."""
    pr = opening.pr
    return ReadinessFacts(
        pr_number=opening.facts.pr_number,
        config=opening.facts.config,
        branch=gate,
        issue_number=opening.issue_number,
        issue_status=issue_status,
        issue_reason_class=issue_reason_class,
        revert=revert.status,
        revert_reason=revert.reason,
        checks=checks.summary,
        checks_unavailable=checks.unavailable,
        check_names_seen=frozenset(str(c.get("name") or "") for c in checks.enriched),
        now=now or _host_current().clock.now(),
        pr_updated_at=pr.get("updatedAt"),
        is_draft=bool(pr.get("isDraft")),
        human_merge_hold=human_merge[0],
        human_merge_check_unavailable=human_merge[1],
        containment_warnings=checks.containment_warnings,
    )


def decide_readiness_lazily(
    facts: ReadinessFacts, read_status: Callable[[], tuple[str | None, str | None]]
) -> tuple[Readiness, bool]:
    """``decide_readiness`` with the issue status read only when it can matter.

    The first call assumes no status; only a stall reads the persisted status
    (legacy read it there). The cross-PR revert route reads its own status
    earlier, ahead of ``pr_checks`` (``decide_revert``), and passes it in
    ``facts``; a revert blocks the sync gate, so it never coincides with a
    stall. Returns the readiness and whether the status was read.
    """
    optimistic = decide_readiness(facts)
    if not optimistic.readiness_stall:
        return optimistic, False
    status, reason_class = read_status()
    return decide_readiness(
        replace(facts, issue_status=status, issue_reason_class=reason_class)
    ), True


# --------------------------------------------------------------------------- #
# Stage 4: holds
# --------------------------------------------------------------------------- #


def gather_merge_hold(app: Any, pr: dict[str, Any], issue_number: int | None) -> tuple[bool, bool]:
    """``(merge_hold, unavailable)``: the PR's label, else the linked issue's.

    The issue's labels come from the per-pass open-issue list when it shows the
    issue, else a live ``issue_view``.
    """
    labels = app.config.labels
    held = labels.merge_hold in label_names(pr)
    if held or issue_number is None:
        return held, False
    cached = open_issue_labels(app.gh, issue_number)
    if cached is not None:
        return labels.merge_hold in cached, False
    try:
        issue = app.gh.issue_view(issue_number)
    except (GitHubError, ValueError):
        return False, True
    if not isinstance(issue, dict) or "labels" not in issue:
        return False, True
    return labels.merge_hold in label_names(issue), False


def holds_from_snapshot(
    snapshot: dict[str, Any],
    cfg: MergePathConfig,
    pr_number: int,
    issue_number: int | None,
    *,
    should_merge: bool,
) -> HoldFacts:
    """Escalation flags, counters and issue status from one state snapshot."""
    pr_esc, issue_esc = escalation_from(snapshot, pr_number, issue_number)
    issue_entry = (
        snapshot.get("issues", {}).get(str(issue_number), {}) if issue_number is not None else {}
    )
    return HoldFacts(
        config=cfg,
        persisted=persisted_from(snapshot.get("prs", {}).get(str(pr_number), {})),
        issue_status=issue_entry.get("status"),
        pr_escalated=pr_esc,
        issue_escalated=issue_esc,
        should_merge=should_merge,
    )


def proceed(readiness: Readiness) -> Readiness:
    """Stall/infra outcomes either ended the pass or were declined: carry on."""
    return replace(readiness, kind=StageKind.PROCEED)
