"""Fact/plan builders for the merge-path decision-table tests.

Every builder starts from a fully pinned "happy path" (approved, head unchanged,
checks green, no holds, issue bound) and applies keyword overrides with
``dataclasses.replace`` semantics, so a table row names only what it varies.
No builder reads a clock, the environment, or GitHub.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from charlie_work.checks import CheckSummary
from charlie_work.merge_path.model import (
    AccountingFacts,
    Admission,
    AdmissionFacts,
    BranchFacts,
    BranchGate,
    GateInputs,
    HoldFacts,
    MergePathConfig,
    PersistedPr,
    Readiness,
    ReadinessFacts,
    RevertStatus,
    StageKind,
    VerdictFact,
)

PR = 7
ISSUE = 70
HEAD = "a" * 40
NEW_HEAD = "b" * 40
LABEL = "merge-queue"
NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
NOW_ISO = "2026-01-01T12:00:00+00:00"
REQUIRED = ("Tests passed", "Lint & Format")


def with_(base: Any, **over: Any) -> Any:
    return replace(base, **over) if over else base


def cfg(**over: Any) -> MergePathConfig:
    return with_(
        MergePathConfig(
            auto_merge_enabled=True,
            mergequeue_label=None,
            update_branch_strategy="front_of_train",
            require_approved_review=True,
            failed_attempt_alarm=3,
            max_conflict_rework_attempts=2,
            human_merge_labels=(),
            review_dispatch_enabled=True,
            required_checks=REQUIRED,
            readiness_no_ci_minutes=15,
            merge_strategy="squash",
        ),
        **over,
    )


def verdict(**over: Any) -> VerdictFact:
    return with_(VerdictFact(approved=True, reviewed_head_sha=HEAD), **over)


def persisted(**over: Any) -> PersistedPr:
    return with_(PersistedPr(), **over)


def summary(**over: Any) -> CheckSummary:
    base = CheckSummary(
        required=REQUIRED,
        passed=REQUIRED,
        pending=(),
        failed=(),
        missing=(),
        infra_failed=(),
    )
    return with_(base, **over)


def admission_facts(**over: Any) -> AdmissionFacts:
    return with_(
        AdmissionFacts(
            pr_number=PR,
            config=cfg(),
            persisted=persisted(),
            pr_found=True,
            live_labels=frozenset(),
            live_head_sha=HEAD,
            issue_number=ISSUE,
            verdict=verdict(),
            carry_forward=None,
        ),
        **over,
    )


def admission(**over: Any) -> Admission:
    """A PROCEED admission for an approved PR."""
    return with_(Admission(kind=StageKind.PROCEED, approved=True), **over)


def branch_facts(**over: Any) -> BranchFacts:
    return with_(
        BranchFacts(
            pr_number=PR,
            config=cfg(),
            admission=admission(),
            persisted=persisted(),
            merge_conflict=False,
            issue_status=None,
            train_head_param=PR,
            train_head=None,
            base_currency_gated=False,
            base_current_read=False,
            base_current=None,
            should_update_branch=False,
        ),
        **over,
    )


def branch_gate(**over: Any) -> BranchGate:
    return with_(BranchGate(kind=StageKind.PROCEED, admission=admission()), **over)


def readiness_facts(**over: Any) -> ReadinessFacts:
    return with_(
        ReadinessFacts(
            pr_number=PR,
            config=cfg(),
            branch=branch_gate(),
            issue_number=ISSUE,
            issue_status=None,
            issue_reason_class=None,
            revert=RevertStatus.CLEAN,
            revert_reason=None,
            checks=summary(),
            checks_unavailable=False,
            check_names_seen=frozenset(REQUIRED),
            now=NOW,
            pr_updated_at="2026-01-01T11:59:00Z",
            is_draft=False,
            human_merge_hold=False,
            human_merge_check_unavailable=False,
        ),
        **over,
    )


def readiness(**over: Any) -> Readiness:
    """A PROCEED readiness with a green gate; override ``gate=`` for other gates."""
    return with_(
        Readiness(
            kind=StageKind.PROCEED,
            gate=GateInputs(
                summary_ready=True,
                approved=True,
                require_approved_review=True,
                sync_failed=False,
            ),
            branch=branch_gate(),
            issue_number=ISSUE,
            summary=summary(),
            checks_unavailable=False,
            pending_only=False,
        ),
        **over,
    )


def gate(**over: Any) -> GateInputs:
    return with_(
        GateInputs(
            summary_ready=True,
            approved=True,
            require_approved_review=True,
            sync_failed=False,
        ),
        **over,
    )


def hold_facts(**over: Any) -> HoldFacts:
    return with_(
        HoldFacts(
            config=cfg(),
            persisted=persisted(),
            issue_status=None,
            pr_escalated=False,
            issue_escalated=False,
            should_merge=True,
        ),
        **over,
    )


def accounting_facts(**over: Any) -> AccountingFacts:
    return with_(
        AccountingFacts(
            pr_number=PR,
            issue_number=ISSUE,
            config=cfg(),
            locked=persisted(),
            deadline_spent=False,
            now_iso=NOW_ISO,
            live_head_sha=HEAD,
            mergeable="MERGEABLE",
            merge_state_status="CLEAN",
        ),
        **over,
    )
