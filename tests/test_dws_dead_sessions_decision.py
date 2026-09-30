"""Decision-table tests for the dead-session lane's pure decisions.

Every function in ``decide_dead_sessions`` is total over already-read facts, so each
row here is one input and the answer the lane has always given. (Named ``test_dws_*``
so the dormant module guard does not mistake this for a one-module-one-test file.)
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from charlie_work.config import (
    DETERMINISTIC_ESCALATION_FAILURE_KINDS,
    DETERMINISTIC_JUDGMENT_ESCALATION_FAILURE_KINDS,
)
from charlie_work.dead_worker_sweep.decide_dead_sessions import (
    REDISPATCH_CAP_REASON,
    EscalationClass,
    dead_fallback_kind,
    escalation_class,
    launch_failure_escalates,
    launch_failure_redispatch_at,
    redispatch_verdict,
    stamp,
    wants_unsafe_salvage,
)
from charlie_work.throttle_signatures import PROVIDER_THROTTLE_FAILURE_KINDS
from charlie_work.worktree import WORKTREE_UNSAFE_KINDS

NOW = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
STAMP = "2026-09-30T12:00:00Z"


def test_stamp_uses_the_z_suffix() -> None:
    assert stamp(NOW) == STAMP


@pytest.mark.parametrize("kind", sorted(DETERMINISTIC_ESCALATION_FAILURE_KINDS))
def test_a_deterministic_mechanical_kind_escalates_immediately(kind: str) -> None:
    assert escalation_class(kind) == EscalationClass(immediate=True, reason_class="mechanical")


@pytest.mark.parametrize("kind", sorted(DETERMINISTIC_JUDGMENT_ESCALATION_FAILURE_KINDS))
def test_a_deterministic_judgment_kind_escalates_as_judgment(kind: str) -> None:
    assert escalation_class(kind) == EscalationClass(immediate=True, reason_class="judgment")


@pytest.mark.parametrize("kind", [None, "stalled", "rate_limited", "worktree_probe_failed"])
def test_other_kinds_take_the_ordinary_cap_path(kind: str | None) -> None:
    assert escalation_class(kind) == EscalationClass(immediate=False, reason_class="mechanical")


def test_judgment_wins_when_a_kind_is_in_both_sets(monkeypatch: pytest.MonkeyPatch) -> None:
    from charlie_work.dead_worker_sweep import decide_dead_sessions as mod

    monkeypatch.setattr(
        mod,
        "DETERMINISTIC_ESCALATION_FAILURE_KINDS",
        frozenset({"both"}) | DETERMINISTIC_ESCALATION_FAILURE_KINDS,
    )
    monkeypatch.setattr(
        mod,
        "DETERMINISTIC_JUDGMENT_ESCALATION_FAILURE_KINDS",
        frozenset({"both"}) | DETERMINISTIC_JUDGMENT_ESCALATION_FAILURE_KINDS,
    )
    assert escalation_class("both").reason_class == "judgment"


@pytest.mark.parametrize(
    ("kind", "has_open_pr", "expected"),
    [
        ("worker_blocked", False, True),
        ("worker_blocked", True, False),
        ("worktree_unsafe_local_commits", False, True),
        ("stalled", False, False),
        (None, False, False),
    ],
)
def test_a_launch_failure_escalates_only_when_deterministic_and_no_pr(
    kind: str | None, has_open_pr: bool, expected: bool
) -> None:
    assert launch_failure_escalates(kind, has_open_pr=has_open_pr) is expected


def test_the_launch_failure_redispatch_window_always_appends() -> None:
    assert launch_failure_redispatch_at((), NOW) == (STAMP,)
    assert launch_failure_redispatch_at(["a", "b"], NOW) == ("a", "b", STAMP)


@pytest.mark.parametrize(
    ("is_completed", "worktree_unknown", "expected"),
    [
        (True, False, "unpublished_work"),
        (True, True, "unpublished_work"),
        (False, False, "stalled"),
        (False, True, None),
    ],
)
def test_the_fallback_kind_for_an_unclassified_dead_session(
    is_completed: bool, worktree_unknown: bool, expected: str | None
) -> None:
    assert (
        dead_fallback_kind(is_completed=is_completed, worktree_unknown=worktree_unknown)
        == expected
    )


@pytest.mark.parametrize("kind", sorted(WORKTREE_UNSAFE_KINDS))
def test_unsafe_salvage_needs_repo_branch_and_an_active_label(kind: str) -> None:
    full = {
        "has_repo_root": True,
        "branch": "agent/issue-1",
        "active_labels": {"agent:in-progress"},
    }
    assert wants_unsafe_salvage(kind, **full) is True
    for missing in ({"has_repo_root": False}, {"branch": ""}, {"active_labels": set()}):
        assert wants_unsafe_salvage(kind, **{**full, **missing}) is False


def test_unsafe_salvage_ignores_kinds_outside_the_unsafe_family() -> None:
    assert (
        wants_unsafe_salvage(
            "stalled", has_repo_root=True, branch="b", active_labels={"agent:in-progress"}
        )
        is False
    )


def _verdict(windowed=(), kind=None, cap=3):
    return redispatch_verdict(windowed, kind, now=NOW, max_auto_redispatch=cap)


def test_a_first_relabel_is_recorded_and_does_not_escalate() -> None:
    v = _verdict(kind="stalled")
    assert (v.redispatch_at, v.escalate, v.reason) == ((STAMP,), False, None)


def test_exceeding_the_cap_escalates_with_the_cap_reason() -> None:
    v = _verdict(["a", "b", "c"], "stalled", cap=3)
    assert v.redispatch_at == ("a", "b", "c", STAMP)
    assert (v.escalate, v.reason, v.reason_class) == (True, REDISPATCH_CAP_REASON, "mechanical")


def test_reaching_the_cap_exactly_does_not_escalate() -> None:
    assert _verdict(["a", "b"], "stalled", cap=3).escalate is False


@pytest.mark.parametrize("kind", sorted(PROVIDER_THROTTLE_FAILURE_KINDS))
def test_a_provider_throttle_death_never_consumes_the_cap(kind: str) -> None:
    v = _verdict(["a", "b", "c"], kind, cap=3)
    assert v.redispatch_at == ("a", "b", "c")
    assert v.escalate is False


def test_a_deterministic_kind_bypasses_the_cap_and_names_itself() -> None:
    v = _verdict(kind="worker_blocked")
    assert (v.escalate, v.reason, v.reason_class) == (True, "worker_blocked", "mechanical")
    assert v.redispatch_at == (STAMP,)


def test_a_judgment_kind_escalates_as_judgment() -> None:
    v = _verdict(kind="worktree_unsafe_local_commits")
    assert (v.escalate, v.reason, v.reason_class) == (
        True,
        "worktree_unsafe_local_commits",
        "judgment",
    )


def test_the_verdict_does_not_mutate_its_window() -> None:
    window = ["a"]
    _verdict(window, "stalled")
    assert window == ["a"]
