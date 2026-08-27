"""Regression tests for issue #1461.

Issue #1461: cross-lane escalation clobbers the single ``escalation_reason``
field, blinding each lane's re-escalation dedup guard.  The fix tracks an
append-only ``escalation_reasons_seen`` list on the issue/PR entry in
``_escalate_issue``, and each lane's dedup guard checks membership in that
list instead of equality against the clobberable single field.  ``terminal_since``
is no longer reset on a re-escalation of the same reason within an episode.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from _janitor_routing_fixtures import _conflicting_app, _set_decision
from charlie_work.config import ReviewConfig
from charlie_work.escalation import _escalate_issue
from charlie_work.state import clear_escalation, load_state, save_state
from charlie_work.workflow import OrchestratorApp


# ---------------------------------------------------------------------------
# Unit tests for _escalate_issue's escalation_reasons_seen tracking
# ---------------------------------------------------------------------------


def test_escalate_issue_appends_reason_to_seen_on_issue_and_pr() -> None:
    """A first escalation records the reason in ``escalation_reasons_seen``
    on both the issue and PR entries."""
    state: dict[str, Any] = {
        "issues": {"1": {"number": 1}},
        "prs": {"10": {"number": 10, "issue_number": 1}},
    }
    state = _escalate_issue(
        state,
        1,
        reason="conflict_rework_attempts_cap_exceeded",
        reason_class="mechanical",
        pr_number=10,
    )
    assert state["issues"]["1"]["escalation_reasons_seen"] == [
        "conflict_rework_attempts_cap_exceeded"
    ]
    assert state["prs"]["10"]["escalation_reasons_seen"] == [
        "conflict_rework_attempts_cap_exceeded"
    ]


def test_escalate_issue_appends_distinct_reasons_without_duplicates() -> None:
    """A second escalation with a different reason appends to the list; a
    re-escalation with the same reason does not duplicate."""
    state: dict[str, Any] = {"issues": {"1": {"number": 1}}, "prs": {}}
    state = _escalate_issue(state, 1, reason="reason_a", reason_class="mechanical")
    state = _escalate_issue(state, 1, reason="reason_b", reason_class="mechanical")
    state = _escalate_issue(state, 1, reason="reason_a", reason_class="mechanical")
    assert state["issues"]["1"]["escalation_reasons_seen"] == ["reason_a", "reason_b"]


def test_escalate_issue_does_not_reset_terminal_since_on_same_reason_reescalation() -> None:
    """Issue #1461: a re-escalation of the same reason within an episode
    must NOT reset ``terminal_since`` -- the cross-lane clobber path used to
    reset the staleness clock on every pass."""
    old_ts = "2020-01-01T00:00:00Z"
    state: dict[str, Any] = {
        "issues": {
            "1": {
                "number": 1,
                "terminal_since": old_ts,
                "escalation_reasons_seen": ["same_reason"],
            }
        },
        "prs": {},
    }
    state = _escalate_issue(state, 1, reason="same_reason", reason_class="mechanical")
    assert state["issues"]["1"]["terminal_since"] == old_ts


def test_escalate_issue_resets_terminal_since_on_new_reason() -> None:
    """A genuinely new reason (not in ``escalation_reasons_seen``) is a fresh
    episode and ``terminal_since`` must move forward."""
    old_ts = "2020-01-01T00:00:00Z"
    state: dict[str, Any] = {
        "issues": {
            "1": {"number": 1, "terminal_since": old_ts, "escalation_reasons_seen": ["old_reason"]}
        },
        "prs": {},
    }
    before = datetime.now(UTC)
    state = _escalate_issue(state, 1, reason="new_reason", reason_class="mechanical")
    after = datetime.now(UTC)
    stamped = state["issues"]["1"]["terminal_since"]
    parsed = datetime.fromisoformat(stamped.replace("Z", "+00:00"))
    assert stamped != old_ts
    assert before - timedelta(seconds=1) <= parsed <= after + timedelta(seconds=1)


def test_escalate_issue_resets_terminal_since_after_deescalation_clears_seen() -> None:
    """After a de-escalation (which clears ``escalation_reasons_seen``), a
    re-escalation with the same reason string is a fresh episode and
    ``terminal_since`` must move forward."""
    old_ts = "2020-01-01T00:00:00Z"
    state: dict[str, Any] = {
        "issues": {
            "1": {
                "number": 1,
                "terminal_since": old_ts,
                "escalation_reason": "same_reason",
                "escalation_reasons_seen": ["same_reason"],
            }
        },
        "prs": {},
    }
    # Simulate de-escalation
    clear_escalation(state["issues"]["1"])
    assert "escalation_reasons_seen" not in state["issues"]["1"]
    # Re-escalate with the same reason
    before = datetime.now(UTC)
    state = _escalate_issue(state, 1, reason="same_reason", reason_class="mechanical")
    after = datetime.now(UTC)
    stamped = state["issues"]["1"]["terminal_since"]
    parsed = datetime.fromisoformat(stamped.replace("Z", "+00:00"))
    assert stamped != old_ts
    assert before - timedelta(seconds=1) <= parsed <= after + timedelta(seconds=1)


# ---------------------------------------------------------------------------
# Unit tests for clear_escalation
# ---------------------------------------------------------------------------


def test_clear_escalation_removes_escalation_reasons_seen() -> None:
    """``clear_escalation`` must pop ``escalation_reasons_seen`` alongside
    ``escalation_reason`` and ``reason_class``."""
    entry: dict[str, Any] = {
        "escalation_reason": "some_reason",
        "reason_class": "mechanical",
        "escalation_reasons_seen": ["some_reason"],
    }
    clear_escalation(entry)
    assert "escalation_reason" not in entry
    assert "reason_class" not in entry
    assert "escalation_reasons_seen" not in entry


# ---------------------------------------------------------------------------
# Integration test: janitor lane dedup survives cross-lane clobber
# ---------------------------------------------------------------------------


def _force_issue_status(app: OrchestratorApp, issue_number: int, status: str | None) -> None:
    state = load_state(app.paths.state_file)
    record = {**state["issues"].get(str(issue_number), {}), "number": issue_number}
    if status is None:
        record.pop("status", None)
    else:
        record["status"] = status
    state["issues"][str(issue_number)] = record
    save_state(app.paths.state_file, state)


def test_janitor_conflict_lane_dedup_survives_cross_lane_clobber(tmp_path: Path) -> None:
    """Issue #1461: after the conflict lane escalates and a different lane
    clobbers ``escalation_reason``, the conflict lane's dedup guard must
    still recognize its own prior escalation via ``escalation_reasons_seen``
    and NOT re-escalate (no duplicate ``janitor_rework_escalated`` event, no
    attempts counter inflation).

    This is the exact scenario from the issue's evidence:
    - 09:34 conflict lane escalates (escalation_reason=conflict_rework_attempts_cap_exceeded)
    - 10:54 watchdog lane clobbers (escalation_reason=redispatch_cap_exceeded)
    - 13:37 conflict lane re-fires (attempts 3->4) -- THE BUG

    With the fix, the conflict lane's guard sees
    ``conflict_rework_attempts_cap_exceeded`` in ``escalation_reasons_seen``
    and returns None instead of re-escalating.
    """
    app = _conflicting_app(
        tmp_path,
        review=ReviewConfig(max_conflict_rework_attempts=2),
    )
    _set_decision(app, 456, "request_changes")

    # Pass 1: routes to rework (attempt 1)
    _force_issue_status(app, 123, "reviewing")
    result1 = app.review(456)
    assert result1.ok is True
    assert result1.data["routed_to_rework"] is True

    # Pass 2: routes to rework again (attempt 2)
    _force_issue_status(app, 123, "reviewing")
    result2 = app.review(456)
    assert result2.ok is True

    # Pass 3: cap exceeded -> conflict lane escalates
    _force_issue_status(app, 123, "reviewing")
    result3 = app.review(456)
    assert result3.ok is False
    assert result3.data["escalated"] is True

    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "conflict_rework_attempts_cap_exceeded"
    assert "conflict_rework_attempts_cap_exceeded" in state["issues"]["123"].get(
        "escalation_reasons_seen", []
    )
    conflict_attempts = state["prs"]["456"]["conflict_rework_attempts"]
    escalated_count = sum(
        1 for e in state.get("events", []) if e.get("kind") == "janitor_rework_escalated"
    )
    assert escalated_count == 1

    # Simulate a cross-lane clobber: a different lane (e.g. the watchdog
    # redispatch cap) escalates the same issue, overwriting
    # escalation_reason but NOT removing the conflict lane's entry from
    # escalation_reasons_seen.
    state = load_state(app.paths.state_file)
    state["issues"]["123"]["escalation_reason"] = "redispatch_cap_exceeded"
    state["prs"]["456"]["escalation_reason"] = "redispatch_cap_exceeded"
    save_state(app.paths.state_file, state)

    # Pass 4: the conflict lane's next pass.  Before the fix, the clobbered
    # escalation_reason blinded the dedup guard and the lane re-escalated
    # (attempts 3->4, duplicate event).  After the fix, the guard sees
    # ``conflict_rework_attempts_cap_exceeded`` in escalation_reasons_seen
    # and returns None (no re-escalation, no attempt inflation).
    _force_issue_status(app, 123, "reviewing")
    result4 = app.review(456)
    # The review() pass should not re-escalate; it should be a pass_skip
    # (the issue is escalated, review()'s early return fires).
    assert result4.ok is True
    assert result4.data.get("pass_skipped") is True
    assert result4.data.get("routed_to_rework") is not True

    state = load_state(app.paths.state_file)
    # attempts counter must NOT have inflated
    assert state["prs"]["456"]["conflict_rework_attempts"] == conflict_attempts
    # no duplicate janitor_rework_escalated event
    escalated_count_after = sum(
        1 for e in state.get("events", []) if e.get("kind") == "janitor_rework_escalated"
    )
    assert escalated_count_after == escalated_count
    # escalation_reasons_seen still contains the conflict lane's reason
    assert "conflict_rework_attempts_cap_exceeded" in state["issues"]["123"].get(
        "escalation_reasons_seen", []
    )


# ---------------------------------------------------------------------------
# Integration test: unescalate clears escalation_reasons_seen
# ---------------------------------------------------------------------------


def test_unescalate_clears_escalation_reasons_seen(tmp_path: Path) -> None:
    """``charlie unescalate`` must clear ``escalation_reasons_seen`` so a
    re-arm gives every lane a genuinely fresh dedup slate."""
    app = _conflicting_app(
        tmp_path,
        review=ReviewConfig(max_conflict_rework_attempts=2),
    )
    _set_decision(app, 456, "request_changes")

    # Escalate via the conflict lane
    _force_issue_status(app, 123, "reviewing")
    app.review(456)
    _force_issue_status(app, 123, "reviewing")
    app.review(456)
    _force_issue_status(app, 123, "reviewing")
    result = app.review(456)
    assert result.ok is False
    assert result.data["escalated"] is True

    state = load_state(app.paths.state_file)
    assert "escalation_reasons_seen" in state["issues"]["123"]
    assert "escalation_reasons_seen" in state["prs"]["456"]

    # Unescalate
    unescalate_result = app.unescalate(issue_number=123)
    assert unescalate_result.ok is True

    state = load_state(app.paths.state_file)
    assert "escalation_reasons_seen" not in state["issues"].get("123", {})
    assert "escalation_reasons_seen" not in state["prs"].get("456", {})
