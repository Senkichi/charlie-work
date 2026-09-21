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
from pathlib import Path

from charlie_work.instrumentation import log_event
from charlie_work.state import (
    arm_dispatch_stale_alert,
    backfill_dispatch_baseline,
    clear_dispatch_stale_alert,
    dispatch_baseline_needs_backfill,
    is_dispatch_stale_alert_due,
    last_non_empty_dispatch,
    mark_dispatch_baseline_backfill_attempted,
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


def test_clear_dispatch_stale_alert_resets_marker_on_genuine_resolution() -> None:
    state = arm_dispatch_stale_alert({}, "2026-09-21T00:00:00Z")

    cleared = clear_dispatch_stale_alert(state, reason="within_threshold")

    assert cleared["dispatch_cadence"]["last_stale_alert_at"] is None


def test_clear_dispatch_stale_alert_leaves_baseline_marker_untouched() -> None:
    state = record_non_empty_dispatch({}, "2026-09-20T22:05:32Z", [1761])
    state = arm_dispatch_stale_alert(state, "2026-09-21T02:00:00Z")

    cleared = clear_dispatch_stale_alert(state, reason="empty_backlog")

    assert last_non_empty_dispatch(cleared) == {
        "ts": "2026-09-20T22:05:32Z",
        "issue_numbers": [1761],
    }
    assert cleared["dispatch_cadence"]["last_stale_alert_at"] is None


def test_clear_dispatch_stale_alert_is_a_noop_when_already_clear() -> None:
    state = {"foo": "bar"}

    assert clear_dispatch_stale_alert(state, reason="within_threshold") is state


def test_clear_dispatch_stale_alert_resets_on_all_ready_blocked_by_dependencies() -> None:
    """A deliberately-idle, observed-good state clears the marker just like
    ``empty_backlog`` -- it is a known resolution, not an unknown reading."""
    state = arm_dispatch_stale_alert({}, "2026-09-21T00:00:00Z")

    cleared = clear_dispatch_stale_alert(state, reason="all_ready_blocked_by_dependencies")

    assert cleared["dispatch_cadence"]["last_stale_alert_at"] is None


def test_clear_dispatch_stale_alert_resets_on_current_pass_dispatched() -> None:
    state = arm_dispatch_stale_alert({}, "2026-09-21T00:00:00Z")

    cleared = clear_dispatch_stale_alert(state, reason="current_pass_dispatched")

    assert cleared["dispatch_cadence"]["last_stale_alert_at"] is None


def test_clear_dispatch_stale_alert_leaves_marker_when_reason_unknown() -> None:
    """Review finding #4 (BLOCKER-adjacent MAJOR): a failed/unknown reading
    (a GitHub API blip surfacing as ``backlog_not_observed``) must not be
    mistaken for a genuine resolution -- clearing here would reset the edge
    and the very next real observation would re-fire immediately."""
    state = arm_dispatch_stale_alert({}, "2026-09-21T00:00:00Z")

    cleared = clear_dispatch_stale_alert(state, reason="backlog_not_observed")

    assert cleared["dispatch_cadence"]["last_stale_alert_at"] == "2026-09-21T00:00:00Z"


def test_clear_dispatch_stale_alert_leaves_marker_for_no_baseline() -> None:
    state = arm_dispatch_stale_alert({}, "2026-09-21T00:00:00Z")

    cleared = clear_dispatch_stale_alert(state, reason="no_baseline")

    assert cleared["dispatch_cadence"]["last_stale_alert_at"] == "2026-09-21T00:00:00Z"


def test_clear_dispatch_stale_alert_leaves_marker_for_threshold_disabled() -> None:
    state = arm_dispatch_stale_alert({}, "2026-09-21T00:00:00Z")

    cleared = clear_dispatch_stale_alert(state, reason="threshold_disabled")

    assert cleared["dispatch_cadence"]["last_stale_alert_at"] == "2026-09-21T00:00:00Z"


def test_clear_dispatch_stale_alert_leaves_marker_when_reason_omitted() -> None:
    """A caller that has not been updated to pass ``reason`` gets the
    conservative default -- treated the same as an unknown reading, not a
    resolution."""
    state = arm_dispatch_stale_alert({}, "2026-09-21T00:00:00Z")

    cleared = clear_dispatch_stale_alert(state)

    assert cleared["dispatch_cadence"]["last_stale_alert_at"] == "2026-09-21T00:00:00Z"


# ---------------------------------------------------------------------------
# Issue #1769 review BLOCKER: the durable baseline is only ever written
# going forward (record_non_empty_dispatch). A repo already mid-stall the
# moment this marker is introduced never takes that write path on its own,
# so `dispatch_baseline_needs_backfill`/`backfill_dispatch_baseline` recover
# a real baseline from events.db history exactly once.
# ---------------------------------------------------------------------------


def test_dispatch_baseline_needs_backfill_true_for_fresh_state() -> None:
    assert dispatch_baseline_needs_backfill({}) is True


def test_dispatch_baseline_needs_backfill_false_once_baseline_exists() -> None:
    state = record_non_empty_dispatch({}, "2026-09-20T22:05:32Z", [1])

    assert dispatch_baseline_needs_backfill(state) is False


def test_dispatch_baseline_needs_backfill_false_once_attempted() -> None:
    state = mark_dispatch_baseline_backfill_attempted({})

    assert dispatch_baseline_needs_backfill(state) is False


def test_mark_dispatch_baseline_backfill_attempted_does_not_mutate_input() -> None:
    original: dict = {}

    updated = mark_dispatch_baseline_backfill_attempted(original)

    assert original == {}
    assert updated["dispatch_cadence"]["baseline_backfill_attempted"] is True


def test_backfill_dispatch_baseline_recovers_newest_non_empty_dispatch(tmp_path: Path) -> None:
    """The BLOCKER's required scenario: a state dict with no ``dispatch_cadence``
    key at all, plus real ``dispatch.db`` history from before this marker
    existed. Recovery must find the newest event whose payload is non-empty,
    not merely the newest ``dispatch`` event."""
    state_path = tmp_path / "state.json"
    log_event(state_path, "dispatch", {"issue_numbers": [1]})
    log_event(state_path, "dispatch", {"issue_numbers": []})
    log_event(state_path, "dispatch", {"issue_numbers": []})

    backfilled = backfill_dispatch_baseline({}, state_path)

    marker = last_non_empty_dispatch(backfilled)
    assert marker is not None
    assert marker["issue_numbers"] == [1]
    assert backfilled["dispatch_cadence"]["baseline_backfill_attempted"] is True


def test_backfill_dispatch_baseline_marks_attempted_when_nothing_found(tmp_path: Path) -> None:
    """A genuinely fresh repo (no non-empty dispatch ever) still gets marked
    as attempted, so a future pass does not re-pay the full-table scan."""
    state_path = tmp_path / "state.json"
    log_event(state_path, "dispatch", {"issue_numbers": []})

    backfilled = backfill_dispatch_baseline({}, state_path)

    assert last_non_empty_dispatch(backfilled) is None
    assert backfilled["dispatch_cadence"]["baseline_backfill_attempted"] is True


def test_backfill_dispatch_baseline_is_a_noop_once_baseline_exists(tmp_path: Path) -> None:
    """A repo that already has a real baseline must never be overwritten by
    a stale events.db reading -- the durable marker, once real, wins."""
    state_path = tmp_path / "state.json"
    log_event(state_path, "dispatch", {"issue_numbers": [999]})
    state = record_non_empty_dispatch({}, "2026-09-20T22:05:32Z", [1761])

    backfilled = backfill_dispatch_baseline(state, state_path)

    assert backfilled is state
    assert last_non_empty_dispatch(backfilled) == {
        "ts": "2026-09-20T22:05:32Z",
        "issue_numbers": [1761],
    }


def test_backfill_dispatch_baseline_is_a_noop_once_already_attempted(tmp_path: Path) -> None:
    """Once marked attempted (found or not), a later call must not re-scan
    events.db even if new non-empty dispatch rows have since been written --
    the one-time-per-repo contract."""
    state_path = tmp_path / "state.json"
    state = mark_dispatch_baseline_backfill_attempted({})
    log_event(state_path, "dispatch", {"issue_numbers": [5]})

    backfilled = backfill_dispatch_baseline(state, state_path)

    assert backfilled is state
    assert last_non_empty_dispatch(backfilled) is None
