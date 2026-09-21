"""Unit tests for the issue #1769 dispatch-cadence durable-marker primitives.

These are the state.py building blocks ``ci_findings.check_dispatch_staleness``
and ``orchestration.dispatch_state`` compose: a durable "last non-empty
dispatch" baseline immune to any lookback-count bound, and an edge +
bounded-low-rate-reminder alert marker. Exercised directly (not just through
``check_dispatch_staleness``) so a future change to either primitive's
semantics fails here first, close to the invariant it protects.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from charlie_work.state import (
    arm_dispatch_stale_alert,
    clear_dispatch_stale_alert,
    is_dispatch_stale_alert_due,
    last_non_empty_dispatch,
    record_non_empty_dispatch,
)


def test_last_non_empty_dispatch_absent_by_default() -> None:
    assert last_non_empty_dispatch({}) is None


def test_record_and_read_round_trip() -> None:
    state = record_non_empty_dispatch({}, "2026-09-20T22:05:32Z", [1761, 5])

    marker = last_non_empty_dispatch(state)

    assert marker == {"ts": "2026-09-20T22:05:32Z", "issue_numbers": [5, 1761]}


def test_record_non_empty_dispatch_does_not_mutate_input() -> None:
    original: dict = {"issues": {}}

    updated = record_non_empty_dispatch(original, "2026-09-20T22:05:32Z", [1])

    assert original == {"issues": {}}
    assert updated is not original
    assert "dispatch_cadence" not in original


def test_record_non_empty_dispatch_overwrites_prior_baseline() -> None:
    state = record_non_empty_dispatch({}, "2026-09-20T20:00:00Z", [1])
    state = record_non_empty_dispatch(state, "2026-09-20T22:00:00Z", [2])

    assert last_non_empty_dispatch(state) == {
        "ts": "2026-09-20T22:00:00Z",
        "issue_numbers": [2],
    }


def test_is_dispatch_stale_alert_due_true_when_never_alerted() -> None:
    now = datetime.now(UTC)

    assert is_dispatch_stale_alert_due({}, now=now, reminder_minutes=60) is True


def test_is_dispatch_stale_alert_due_false_within_window() -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    last = (now - timedelta(minutes=10)).isoformat().replace("+00:00", "Z")
    state = arm_dispatch_stale_alert({}, last)

    assert is_dispatch_stale_alert_due(state, now=now, reminder_minutes=60) is False


def test_is_dispatch_stale_alert_due_true_after_window_elapses() -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    last = (now - timedelta(minutes=61)).isoformat().replace("+00:00", "Z")
    state = arm_dispatch_stale_alert({}, last)

    assert is_dispatch_stale_alert_due(state, now=now, reminder_minutes=60) is True


def test_is_dispatch_stale_alert_due_true_on_malformed_timestamp() -> None:
    """A corrupt marker must not silently wedge the alarm off forever."""
    state = {"dispatch_cadence": {"last_stale_alert_at": "not-a-timestamp"}}

    assert is_dispatch_stale_alert_due(state, now=datetime.now(UTC), reminder_minutes=60) is True


def test_is_dispatch_stale_alert_due_true_when_reminder_minutes_non_positive() -> None:
    """A zero/negative reminder interval means "no suppression", not a divide
    or comparison edge case -- always due."""
    now = datetime.now(UTC)
    state = arm_dispatch_stale_alert({}, now.isoformat().replace("+00:00", "Z"))

    assert is_dispatch_stale_alert_due(state, now=now, reminder_minutes=0) is True


def test_arm_dispatch_stale_alert_does_not_mutate_input() -> None:
    original: dict = {}

    updated = arm_dispatch_stale_alert(original, "2026-09-21T00:00:00Z")

    assert original == {}
    assert updated["dispatch_cadence"]["last_stale_alert_at"] == "2026-09-21T00:00:00Z"


def test_clear_dispatch_stale_alert_resets_marker() -> None:
    state = arm_dispatch_stale_alert({}, "2026-09-21T00:00:00Z")

    cleared = clear_dispatch_stale_alert(state)

    assert cleared["dispatch_cadence"]["last_stale_alert_at"] is None


def test_clear_dispatch_stale_alert_leaves_baseline_marker_untouched() -> None:
    state = record_non_empty_dispatch({}, "2026-09-20T22:05:32Z", [1761])
    state = arm_dispatch_stale_alert(state, "2026-09-21T02:00:00Z")

    cleared = clear_dispatch_stale_alert(state)

    assert last_non_empty_dispatch(cleared) == {
        "ts": "2026-09-20T22:05:32Z",
        "issue_numbers": [1761],
    }
    assert cleared["dispatch_cadence"]["last_stale_alert_at"] is None


def test_clear_dispatch_stale_alert_is_a_noop_when_already_clear() -> None:
    state = {"foo": "bar"}

    assert clear_dispatch_stale_alert(state) is state
