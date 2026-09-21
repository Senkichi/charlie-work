"""Tests for issue #946 / #1769: staleness detector on dispatch cadence.

The detector reads a durable ``dispatch_cadence`` marker in the caller's
already-loaded state dict for the most recent dispatch pass whose payload
``issue_numbers`` was non-empty. Issue #1769: this used to scan the most
recent 100 ``dispatch`` rows in events.db instead -- bounded by event COUNT,
not by "does a non-empty one exist" -- so a sustained real stall could
scroll the true baseline out of that window and the alarm went silent
(``no_baseline``) exactly when it mattered most. See
``test_stale_survives_far_more_passes_than_the_old_100_row_lookback`` below
for the direct regression.

When the baseline reading is older than a configurable threshold and the
unfiltered backlog is observed to be non-empty, ``check_dispatch_staleness``
returns a stale diagnostic. The dispatch path records that as a
``dispatch_stale`` warning event, but only when ``should_emit`` says so:
issue #1769 also makes this edge-triggered (fires once at stall onset) plus
a bounded low-rate reminder, rather than an unconditional re-fire every pass
the condition still holds -- see the "should_emit" tests near the bottom.

``classify_backlog_reachability`` already answers the second half: the
``observed: False`` case must not be treated as "backlog empty".
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from charlie_work.config import DispatchConfig
from charlie_work.instrumentation import log_event, query_events
from charlie_work.state import arm_dispatch_stale_alert, record_non_empty_dispatch
from charlie_work.workflow import check_dispatch_staleness


def _iso_now(now: datetime) -> str:
    return now.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _state_with_baseline(ts: str, issue_numbers: list[int]) -> dict[str, Any]:
    """Build a bare state dict carrying a durable non-empty-dispatch baseline."""
    return record_non_empty_dispatch({}, ts, issue_numbers)


def _backlog(*, nonempty: bool, observed: bool = True) -> dict[str, object]:
    if not observed:
        return {"observed": False, "open_total": 0, "dispatchable": 0}
    return {"observed": True, "open_total": 1 if nonempty else 0, "dispatchable": 0}


def test_stale_when_last_nonempty_dispatch_exceeds_threshold() -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    threshold_minutes = 60
    config = DispatchConfig(dispatch_staleness_minutes=threshold_minutes)
    old = _iso_now(now - timedelta(minutes=threshold_minutes + 10))
    state = _state_with_baseline(old, [1, 2])

    result = check_dispatch_staleness(state, config, _backlog(nonempty=True), now=now)

    assert result["stale"] is True
    assert result["age_seconds"] == (threshold_minutes + 10) * 60
    assert result["threshold_seconds"] == threshold_minutes * 60
    assert result["last_dispatch_at"] == old
    assert result["last_dispatch_issue_numbers"] == [1, 2]
    assert result["backlog_observed"] is True
    assert result["backlog_open_total"] == 1


def test_not_stale_when_last_nonempty_dispatch_within_threshold() -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    threshold_minutes = 60
    config = DispatchConfig(dispatch_staleness_minutes=threshold_minutes)
    recent = _iso_now(now - timedelta(minutes=threshold_minutes - 10))
    state = _state_with_baseline(recent, [3])

    result = check_dispatch_staleness(state, config, _backlog(nonempty=True), now=now)

    assert result["stale"] is False
    assert result["should_emit"] is False
    assert result["age_seconds"] == (threshold_minutes - 10) * 60
    assert result["last_dispatch_at"] == recent


def test_not_stale_when_no_baseline() -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    config = DispatchConfig(dispatch_staleness_minutes=60)

    result = check_dispatch_staleness({}, config, _backlog(nonempty=True), now=now)

    assert result["stale"] is False
    assert result["last_dispatch_at"] is None
    assert result["reason"] == "no_baseline"


def test_most_recent_nonempty_dispatch_wins_past_empty_events() -> None:
    """An empty-payload dispatch pass never calls ``record_non_empty_dispatch``,
    so it cannot overwrite -- or even touch -- a real baseline. Simulates two
    genuinely non-empty passes (old, then recent): the marker reflects only
    the most recent one, the same "most recent non-empty wins" contract the
    old events.db scan had, but as a plain overwrite instead of a scan.
    """
    now = datetime.now(UTC).replace(microsecond=0)
    config = DispatchConfig(dispatch_staleness_minutes=60)
    old = _iso_now(now - timedelta(minutes=90))
    state = _state_with_baseline(old, [1])
    recent = _iso_now(now - timedelta(minutes=30))
    state = record_non_empty_dispatch(state, recent, [2])

    result = check_dispatch_staleness(state, config, _backlog(nonempty=True), now=now)

    assert result["stale"] is False
    assert result["age_seconds"] == 30 * 60
    assert result["last_dispatch_at"] == recent


def test_not_stale_when_backlog_not_observed() -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    config = DispatchConfig(dispatch_staleness_minutes=60)
    old = _iso_now(now - timedelta(minutes=90))
    state = _state_with_baseline(old, [1])

    result = check_dispatch_staleness(
        state,
        config,
        _backlog(nonempty=True, observed=False),
        now=now,
    )

    assert result["stale"] is False
    assert result["reason"] == "backlog_not_observed"


def test_not_stale_when_backlog_genuinely_empty() -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    config = DispatchConfig(dispatch_staleness_minutes=60)
    old = _iso_now(now - timedelta(minutes=90))
    state = _state_with_baseline(old, [1])

    result = check_dispatch_staleness(
        state,
        config,
        _backlog(nonempty=False),
        now=now,
    )

    assert result["stale"] is False
    assert result["backlog_open_total"] == 0
    assert result["reason"] == "empty_backlog"


def test_zero_threshold_disables() -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    config = DispatchConfig(dispatch_staleness_minutes=0)
    old = _iso_now(now - timedelta(minutes=90))
    state = _state_with_baseline(old, [1])

    result = check_dispatch_staleness(state, config, _backlog(nonempty=True), now=now)

    assert result["stale"] is False
    assert result["threshold_seconds"] == 0


def test_current_pass_dispatched_short_circuits() -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    config = DispatchConfig(dispatch_staleness_minutes=60)
    old = _iso_now(now - timedelta(minutes=90))
    state = _state_with_baseline(old, [1])

    result = check_dispatch_staleness(
        state,
        config,
        _backlog(nonempty=True),
        now=now,
        recent_issue_numbers=[2],
    )

    assert result["stale"] is False
    assert result["age_seconds"] == 0
    assert result["last_dispatch_at"] == _iso_now(now)


def test_dispatch_stale_event_classified_as_warning(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    log_event(state_path, "dispatch_stale", {"age_seconds": 123})

    events = query_events(state_path, kind="dispatch_stale")

    assert len(events) == 1
    assert events[0]["level"] == "warning"


# ---------------------------------------------------------------------------
# Issue #1110: staleness must not fire when every ready issue is blocked by an
# open dependency. A deliberately sequenced cohort tail is permanently -- and
# correctly -- unselectable by dispatch, so a cadence alarm for it is a false
# positive that pattern-matches the #944 four-day stall this detector exists
# to catch.
# ---------------------------------------------------------------------------


def _backlog_blocked(*, open_total: int = 2, blocked: int = 2) -> dict[str, object]:
    """A backlog where every ready issue is dependency-blocked."""
    return {
        "observed": True,
        "open_total": open_total,
        "dispatchable": 0,
        "blocked_by_open_dependency": blocked,
    }


def test_not_stale_when_all_ready_issues_blocked_by_dependencies() -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    config = DispatchConfig(dispatch_staleness_minutes=60)
    old = _iso_now(now - timedelta(minutes=90))
    state = _state_with_baseline(old, [1])

    result = check_dispatch_staleness(
        state,
        config,
        _backlog_blocked(open_total=2, blocked=2),
        now=now,
    )

    assert result["stale"] is False
    assert result["reason"] == "all_ready_blocked_by_dependencies"
    assert result["backlog_dispatchable"] == 0
    assert result["backlog_blocked_by_open_dependency"] == 2


def test_still_stale_when_no_ready_issues_at_all() -> None:
    # The #944 case: open issues exist but none are ready (all missing_ready,
    # terminal, etc.). dispatchable == 0 and blocked_by_open_dependency == 0.
    # The alarm MUST still fire -- this is the four-day stall this detector
    # exists to catch. The #1110 fix must not suppress it.
    now = datetime.now(UTC).replace(microsecond=0)
    config = DispatchConfig(dispatch_staleness_minutes=60)
    old = _iso_now(now - timedelta(minutes=90))
    state = _state_with_baseline(old, [1])

    backlog = {
        "observed": True,
        "open_total": 87,
        "dispatchable": 0,
        "blocked_by_open_dependency": 0,
        "missing_ready": 87,
    }
    result = check_dispatch_staleness(state, config, backlog, now=now)

    assert result["stale"] is True
    assert result["reason"] == "dispatch_stale"
    assert result["backlog_dispatchable"] == 0
    assert result["backlog_blocked_by_open_dependency"] == 0


def test_stale_when_dispatchable_issues_exist_despite_some_blocked() -> None:
    # A backlog with both genuinely dispatchable issues and dependency-blocked
    # ones: the post-gate count is > 0, so the alarm fires normally. The #1110
    # fix only suppresses the alarm when ALL ready issues are blocked.
    now = datetime.now(UTC).replace(microsecond=0)
    config = DispatchConfig(dispatch_staleness_minutes=60)
    old = _iso_now(now - timedelta(minutes=90))
    state = _state_with_baseline(old, [1])

    backlog = {
        "observed": True,
        "open_total": 3,
        "dispatchable": 1,
        "blocked_by_open_dependency": 2,
    }
    result = check_dispatch_staleness(state, config, backlog, now=now)

    assert result["stale"] is True
    assert result["reason"] == "dispatch_stale"


# ---------------------------------------------------------------------------
# Issue #1769 acceptance criterion 3: the durable baseline marker cannot fall
# out of a rolling window, however many empty dispatch passes accumulate
# after the last real one -- the exact false-negative this fix closes.
# ---------------------------------------------------------------------------


def test_stale_survives_far_more_passes_than_the_old_100_row_lookback() -> None:
    """Reproduces the live charlie-work scenario from issue #1769: a real
    non-empty dispatch, followed by many more than 100 subsequent empty
    dispatch passes, must still report ``stale: True`` (never
    ``no_baseline``) once the threshold has elapsed. The old events.db
    ``limit=100`` scan would have scrolled the real baseline out of its
    window well before pass 100; the durable marker cannot, because nothing
    but a genuinely non-empty dispatch pass ever rewrites it, and none of
    the simulated passes below are non-empty.
    """
    now = datetime.now(UTC).replace(microsecond=0)
    threshold_minutes = 240
    config = DispatchConfig(dispatch_staleness_minutes=threshold_minutes)
    old = _iso_now(now - timedelta(minutes=threshold_minutes + 60))
    state = _state_with_baseline(old, [1761])

    result: dict[str, Any] = {}
    for _ in range(145):
        result = check_dispatch_staleness(state, config, _backlog(nonempty=True), now=now)
        assert result["stale"] is True, result
        assert result["reason"] == "dispatch_stale"
        assert result["last_dispatch_at"] == old

    assert result["age_seconds"] == (threshold_minutes + 60) * 60


# ---------------------------------------------------------------------------
# Issue #1769 section 6 policy: edge-triggered + bounded low-rate reminder,
# never an unconditional re-fire every pass while the condition still holds.
# ---------------------------------------------------------------------------


def test_should_emit_true_on_first_stale_detection() -> None:
    """A fresh stall onset (no prior alert recorded) is always the edge."""
    now = datetime.now(UTC).replace(microsecond=0)
    threshold_minutes = 60
    config = DispatchConfig(dispatch_staleness_minutes=threshold_minutes)
    old = _iso_now(now - timedelta(minutes=threshold_minutes + 10))
    state = _state_with_baseline(old, [1])

    result = check_dispatch_staleness(state, config, _backlog(nonempty=True), now=now)

    assert result["stale"] is True
    assert result["should_emit"] is True


def test_should_emit_false_while_within_reminder_window() -> None:
    """Once alerted, a still-stale pass does not re-fire before the
    reminder interval (the staleness threshold itself, reused rather than a
    second config knob) has elapsed."""
    now = datetime.now(UTC).replace(microsecond=0)
    threshold_minutes = 60
    config = DispatchConfig(dispatch_staleness_minutes=threshold_minutes)
    old = _iso_now(now - timedelta(minutes=threshold_minutes + 10))
    state = _state_with_baseline(old, [1])
    state = arm_dispatch_stale_alert(
        state, _iso_now(now - timedelta(minutes=threshold_minutes - 1))
    )

    result = check_dispatch_staleness(state, config, _backlog(nonempty=True), now=now)

    assert result["stale"] is True
    assert result["should_emit"] is False


def test_should_emit_true_again_once_reminder_interval_elapses() -> None:
    """A stall that outlives the reminder interval fires again -- a long
    stall is not fully silent forever after the first alert."""
    now = datetime.now(UTC).replace(microsecond=0)
    threshold_minutes = 60
    config = DispatchConfig(dispatch_staleness_minutes=threshold_minutes)
    old = _iso_now(now - timedelta(minutes=threshold_minutes * 3))
    state = _state_with_baseline(old, [1])
    state = arm_dispatch_stale_alert(
        state, _iso_now(now - timedelta(minutes=threshold_minutes + 1))
    )

    result = check_dispatch_staleness(state, config, _backlog(nonempty=True), now=now)

    assert result["stale"] is True
    assert result["should_emit"] is True


def test_should_emit_stays_false_across_many_passes_within_reminder_window() -> None:
    """No-spam control: many consecutive still-stale passes inside one
    reminder window must not re-fire -- only crossing the interval boundary
    does (covered by the two tests above)."""
    now = datetime.now(UTC).replace(microsecond=0)
    threshold_minutes = 60
    config = DispatchConfig(dispatch_staleness_minutes=threshold_minutes)
    old = _iso_now(now - timedelta(minutes=threshold_minutes * 5))
    state = _state_with_baseline(old, [1])
    state = arm_dispatch_stale_alert(state, _iso_now(now))

    emit_count = 0
    for _ in range(50):
        result = check_dispatch_staleness(state, config, _backlog(nonempty=True), now=now)
        assert result["stale"] is True
        if result["should_emit"]:
            emit_count += 1

    assert emit_count == 0


def test_should_emit_false_when_not_stale() -> None:
    """``should_emit`` is always False alongside ``stale: False`` -- there is
    nothing to (re-)emit for a healthy pass, regardless of any stale prior
    alert marker left over from a since-resolved episode."""
    now = datetime.now(UTC).replace(microsecond=0)
    threshold_minutes = 60
    config = DispatchConfig(dispatch_staleness_minutes=threshold_minutes)
    recent = _iso_now(now - timedelta(minutes=threshold_minutes - 5))
    state = _state_with_baseline(recent, [1])
    state = arm_dispatch_stale_alert(
        state, _iso_now(now - timedelta(minutes=threshold_minutes * 10))
    )

    result = check_dispatch_staleness(state, config, _backlog(nonempty=True), now=now)

    assert result["stale"] is False
    assert result["should_emit"] is False
