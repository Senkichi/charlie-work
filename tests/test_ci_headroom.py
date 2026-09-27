"""Unit tests for ``ci_headroom_available`` (issue #1770, step 1 of 2).

The function under test is a pure(ish) read over ``events.db``: it never
calls GitHub and never mutates anything but the event log it writes its own
diagnostic to. Every case here builds its own ``runner_allocation`` event (or
deliberately omits one) with ``instrumentation.log_event`` and asserts both
the return value and, where relevant, the ``ci_headroom_unavailable``
diagnostic event.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from charlie_work.ci_headroom import (
    ALLOCATION_EVENT_KIND,
    ci_headroom_available,
)
from charlie_work.instrumentation import log_event, query_events


def _log_allocation(
    state_path: Path,
    targets: list[dict[str, Any]],
    *,
    budget: int = 8,
) -> None:
    """Write a ``runner_allocation`` event shaped like ``plan_summary``'s output."""
    log_event(
        state_path,
        ALLOCATION_EVENT_KIND,
        {
            "budget": budget,
            "budget_reason": "test",
            "targets": targets,
            "changes": [],
            "notes": [],
        },
    )


def _target(
    repo: str,
    *,
    capacity: int,
    demand: int,
    running: int | None = None,
    target: int | None = None,
    pinned: bool = False,
) -> dict[str, Any]:
    return {
        "repo": repo,
        "capacity": capacity,
        "demand": demand,
        "running": running if running is not None else min(capacity, demand),
        "target": target if target is not None else min(capacity, demand),
        "pinned": pinned,
    }


def _unavailable_events(state_path: Path) -> list[dict[str, Any]]:
    # Literal, not the imported constant: this repo's event-kind-consumer
    # scanner (tests/test_event_kind_consumers.py) statically resolves
    # `query_events(kind=...)` literals to register a real consumer for a
    # kind -- an imported name would not be recognized as one.
    return query_events(state_path, kind="ci_headroom_unavailable")


# ---------------------------------------------------------------------------
# Normal computation
# ---------------------------------------------------------------------------


def test_headroom_available_computes_from_freshest_target(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    _log_allocation(state_path, [_target("Senkichi/job-cannon", capacity=5, demand=3)])

    result = ci_headroom_available(
        "Senkichi/job-cannon", headroom_ratio=1.5, fleet_state_path=state_path
    )

    # floor(5 * 1.5) - 3 = floor(7.5) - 3 = 7 - 3 = 4
    assert result == 4
    assert _unavailable_events(state_path) == []


def test_headroom_ratio_floors_a_fractional_ceiling(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    _log_allocation(state_path, [_target("Senkichi/swole", capacity=2, demand=0)])

    result = ci_headroom_available(
        "Senkichi/swole", headroom_ratio=1.5, fleet_state_path=state_path
    )

    # floor(2 * 1.5) - 0 = floor(3.0) - 0 = 3
    assert result == 3


def test_headroom_saturated_clamps_to_zero_not_negative(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    _log_allocation(state_path, [_target("Senkichi/job-cannon", capacity=5, demand=18)])

    result = ci_headroom_available(
        "Senkichi/job-cannon", headroom_ratio=1.5, fleet_state_path=state_path
    )

    # floor(5 * 1.5) - 18 = 7 - 18 = -11 -> clamped to 0, never negative.
    assert result == 0
    assert _unavailable_events(state_path) == []


def test_headroom_selects_the_matching_repo_among_several_targets(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    _log_allocation(
        state_path,
        [
            _target("Senkichi/swole", capacity=2, demand=2),
            _target("Senkichi/job-cannon", capacity=5, demand=1),
            _target("Senkichi/fresh-eyes", capacity=1, demand=0),
        ],
    )

    result = ci_headroom_available(
        "Senkichi/job-cannon", headroom_ratio=1.0, fleet_state_path=state_path
    )

    assert result == 4  # floor(5 * 1.0) - 1


def test_headroom_uses_the_freshest_event_when_several_exist(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    _log_allocation(state_path, [_target("Senkichi/job-cannon", capacity=5, demand=1)])
    _log_allocation(state_path, [_target("Senkichi/job-cannon", capacity=5, demand=4)])

    result = ci_headroom_available(
        "Senkichi/job-cannon", headroom_ratio=1.0, fleet_state_path=state_path
    )

    # Must reflect the SECOND (freshest) event's demand=4, not the first's demand=1:
    # floor(5 * 1.0) - 4 = 1, not floor(5 * 1.0) - 1 = 4.
    assert result == 1


# ---------------------------------------------------------------------------
# Fail-open: missing / stale / malformed / unconfigured / pinned data
# ---------------------------------------------------------------------------


def test_headroom_none_and_logged_when_no_allocation_event_exists(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"

    result = ci_headroom_available(
        "Senkichi/job-cannon", headroom_ratio=1.5, fleet_state_path=state_path
    )

    assert result is None
    events = _unavailable_events(state_path)
    assert len(events) == 1
    assert events[0]["payload"]["reason"] == "no_data"
    assert events[0]["repo"] == "Senkichi/job-cannon"
    assert events[0]["level"] == "warning"


def test_headroom_none_and_logged_when_event_is_stale(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    _log_allocation(state_path, [_target("Senkichi/job-cannon", capacity=5, demand=1)])

    # The event was just written at "now"; ask as-of 40 minutes later, past
    # the 30-minute default staleness window.
    future = datetime.now(UTC) + timedelta(minutes=40)
    result = ci_headroom_available(
        "Senkichi/job-cannon", headroom_ratio=1.5, fleet_state_path=state_path, now=future
    )

    assert result is None
    events = _unavailable_events(state_path)
    assert len(events) == 1
    assert events[0]["payload"]["reason"] == "stale"


def test_headroom_not_stale_within_the_max_data_age_window(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    _log_allocation(state_path, [_target("Senkichi/job-cannon", capacity=5, demand=1)])

    just_inside_window = datetime.now(UTC) + timedelta(minutes=20)
    result = ci_headroom_available(
        "Senkichi/job-cannon",
        headroom_ratio=1.5,
        fleet_state_path=state_path,
        now=just_inside_window,
    )

    assert result == 6  # floor(5 * 1.5) - 1
    assert _unavailable_events(state_path) == []


def test_headroom_respects_custom_max_data_age_minutes(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    _log_allocation(state_path, [_target("Senkichi/job-cannon", capacity=5, demand=1)])

    ten_minutes_later = datetime.now(UTC) + timedelta(minutes=10)
    result = ci_headroom_available(
        "Senkichi/job-cannon",
        headroom_ratio=1.5,
        fleet_state_path=state_path,
        max_data_age_minutes=5,
        now=ten_minutes_later,
    )

    assert result is None
    assert _unavailable_events(state_path)[0]["payload"]["reason"] == "stale"


def test_headroom_none_and_logged_for_unconfigured_repo(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    _log_allocation(state_path, [_target("Senkichi/swole", capacity=2, demand=1)])

    # Senkichi/job-cannon never appears in this pass's targets at all.
    result = ci_headroom_available(
        "Senkichi/job-cannon", headroom_ratio=1.5, fleet_state_path=state_path
    )

    assert result is None
    events = _unavailable_events(state_path)
    assert len(events) == 1
    assert events[0]["payload"]["reason"] == "unconfigured"


def test_headroom_none_and_logged_when_demand_is_pinned(tmp_path: Path) -> None:
    """A pinned target's demand=0 is a bookkeeping placeholder, not a real
    zero -- trusting it would silently read as "fully open" for a repo whose
    live demand ci_fleet could not actually measure this pass."""
    state_path = tmp_path / "state.json"
    _log_allocation(
        state_path,
        [_target("Senkichi/job-cannon", capacity=5, demand=0, pinned=True)],
    )

    result = ci_headroom_available(
        "Senkichi/job-cannon", headroom_ratio=1.5, fleet_state_path=state_path
    )

    assert result is None
    events = _unavailable_events(state_path)
    assert len(events) == 1
    assert events[0]["payload"]["reason"] == "pinned"


def test_headroom_none_and_logged_when_targets_is_not_a_list(tmp_path: Path) -> None:
    """A payload whose ``targets`` value is not a list is treated the same as
    "repo not found" (``"unconfigured"``): either way there is no target
    entry to read this repo's numbers from."""
    state_path = tmp_path / "state.json"
    log_event(state_path, ALLOCATION_EVENT_KIND, {"targets": "not-a-list"})

    result = ci_headroom_available(
        "Senkichi/job-cannon", headroom_ratio=1.5, fleet_state_path=state_path
    )

    assert result is None
    assert _unavailable_events(state_path)[0]["payload"]["reason"] == "unconfigured"


def test_headroom_none_and_logged_when_event_payload_is_null_in_db(tmp_path: Path) -> None:
    """A ``runner_allocation`` row with a NULL payload must not crash the lane.

    Regression for issue #1883: ``instrumentation._row_to_event`` fed the raw
    column straight into ``json.loads``, so a NULL payload raised
    ``TypeError: the JSON object must be str, bytes or bytearray, not
    NoneType`` — outside ``query_events``' ``sqlite3.Error`` catch, escaping
    this function's documented never-raises contract and aborting the fleet
    pass with an unclassified ``fleet_pass_config_error``. NULL can only
    arrive via an ``events`` table this module's schema did not create
    (``CREATE TABLE IF NOT EXISTS`` preserves it) or a corrupted file; the
    row now degrades to an empty payload, which fails open as
    ``"unconfigured"`` like any other unusable reading.
    """
    import sqlite3

    state_path = tmp_path / "state.json"
    db_path = state_path.parent / "events.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, kind TEXT, payload TEXT,
            repo TEXT, correlation_id TEXT, pr_number INTEGER,
            issue_number INTEGER, level TEXT DEFAULT 'info'
        );
        """
    )
    conn.execute(
        "INSERT INTO events (ts, kind, payload) VALUES (?, ?, NULL)",
        (datetime.now(UTC).isoformat(), ALLOCATION_EVENT_KIND),
    )
    conn.commit()
    conn.close()

    result = ci_headroom_available(
        "Senkichi/job-cannon", headroom_ratio=1.5, fleet_state_path=state_path
    )

    assert result is None
    events = _unavailable_events(state_path)
    assert len(events) == 1
    assert events[0]["payload"]["reason"] == "unconfigured"


def test_headroom_none_and_logged_for_non_integer_capacity_or_demand(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    _log_allocation(
        state_path,
        [{"repo": "Senkichi/job-cannon", "capacity": "5", "demand": 1, "pinned": False}],
    )

    result = ci_headroom_available(
        "Senkichi/job-cannon", headroom_ratio=1.5, fleet_state_path=state_path
    )

    assert result is None
    assert _unavailable_events(state_path)[0]["payload"]["reason"] == "malformed"


def test_headroom_none_when_capacity_or_demand_is_a_bool(tmp_path: Path) -> None:
    """``isinstance(True, int)`` is True in Python -- guard against a bool
    slipping through as though it were a real integer count."""
    state_path = tmp_path / "state.json"
    _log_allocation(
        state_path,
        [{"repo": "Senkichi/job-cannon", "capacity": 5, "demand": True, "pinned": False}],
    )

    result = ci_headroom_available(
        "Senkichi/job-cannon", headroom_ratio=1.5, fleet_state_path=state_path
    )

    assert result is None
    assert _unavailable_events(state_path)[0]["payload"]["reason"] == "malformed"


def test_headroom_none_and_logged_when_no_events_db_directory_exists(tmp_path: Path) -> None:
    """A brand-new repo/host with no fleet events.db at all fails open, not closed."""
    state_path = tmp_path / "does-not-exist" / "state.json"

    result = ci_headroom_available(
        "Senkichi/job-cannon", headroom_ratio=1.5, fleet_state_path=state_path
    )

    assert result is None


# ---------------------------------------------------------------------------
# min_in_flight_demand: floors demand at the caller's own in-flight count
# (issue #1770 review finding 3)
# ---------------------------------------------------------------------------


def test_min_in_flight_demand_floors_a_stale_low_reading(tmp_path: Path) -> None:
    """The allocation event's demand (0) predates work the caller already
    knows is in flight -- min_in_flight_demand must floor the effective
    demand used in the computation, not just clamp the final result."""
    state_path = tmp_path / "state.json"
    _log_allocation(state_path, [_target("Senkichi/job-cannon", capacity=5, demand=0)])

    result = ci_headroom_available(
        "Senkichi/job-cannon",
        headroom_ratio=1.0,
        fleet_state_path=state_path,
        min_in_flight_demand=3,
    )

    # max(0, floor(5*1.0) - max(0, 3)) = 2, not floor(5*1.0) - 0 = 5.
    assert result == 2


def test_min_in_flight_demand_below_measured_demand_is_a_no_op(tmp_path: Path) -> None:
    """The default 0 -- and any value below the measured demand -- must not
    change the result: min_in_flight_demand is a floor, never a ceiling."""
    state_path = tmp_path / "state.json"
    _log_allocation(state_path, [_target("Senkichi/job-cannon", capacity=5, demand=4)])

    result = ci_headroom_available(
        "Senkichi/job-cannon",
        headroom_ratio=1.0,
        fleet_state_path=state_path,
        min_in_flight_demand=1,
    )

    assert result == 1  # floor(5*1.0) - max(4, 1) = 5 - 4 = 1, unaffected.


# ---------------------------------------------------------------------------
# diagnostic_state_path / diagnostic_repo: route the fail-open diagnostic to
# a different store/key than the allocation-event read (issue #1770 review
# finding 8)
# ---------------------------------------------------------------------------


def test_diagnostic_state_path_and_repo_override_where_the_event_lands(tmp_path: Path) -> None:
    fleet_state_path = tmp_path / "fleet" / "state.json"
    per_repo_state_path = tmp_path / "per-repo" / "state.json"
    # No allocation event at fleet_state_path -> "no_data" fail-open path.

    result = ci_headroom_available(
        "Senkichi/job-cannon",
        headroom_ratio=1.5,
        fleet_state_path=fleet_state_path,
        diagnostic_state_path=per_repo_state_path,
        diagnostic_repo="job-cannon",
    )

    assert result is None
    assert _unavailable_events(fleet_state_path) == []
    per_repo_events = query_events(per_repo_state_path, kind="ci_headroom_unavailable")
    assert len(per_repo_events) == 1
    assert per_repo_events[0]["repo"] == "job-cannon"
    assert per_repo_events[0]["payload"]["reason"] == "no_data"


# ---------------------------------------------------------------------------
# Edge-triggered / rate-limited emission of ci_headroom_unavailable (issue
# #1770 review finding 2)
# ---------------------------------------------------------------------------


def test_unavailable_event_not_repeated_for_same_reason_within_the_window(
    tmp_path: Path,
) -> None:
    """A second call with the same persisting reason, shortly after the
    first, must not write a second event -- an unconditional write here
    would turn one stuck condition into an unbounded warning stream."""
    state_path = tmp_path / "state.json"
    # No allocation event ever -> "no_data" on every call.
    t0 = datetime.now(UTC)

    ci_headroom_available(
        "Senkichi/job-cannon", headroom_ratio=1.5, fleet_state_path=state_path, now=t0
    )
    ci_headroom_available(
        "Senkichi/job-cannon",
        headroom_ratio=1.5,
        fleet_state_path=state_path,
        now=t0 + timedelta(minutes=5),
    )

    events = _unavailable_events(state_path)
    assert len(events) == 1


def test_unavailable_event_repeats_once_the_interval_elapses(tmp_path: Path) -> None:
    """The same reason, persisting past max_data_age_minutes since the last
    write, re-emits -- the dedup window is bounded, not permanent."""
    state_path = tmp_path / "state.json"
    t0 = datetime.now(UTC)

    ci_headroom_available(
        "Senkichi/job-cannon", headroom_ratio=1.5, fleet_state_path=state_path, now=t0
    )
    ci_headroom_available(
        "Senkichi/job-cannon",
        headroom_ratio=1.5,
        fleet_state_path=state_path,
        now=t0 + timedelta(minutes=31),
    )

    events = _unavailable_events(state_path)
    assert len(events) == 2


def test_unavailable_event_repeats_immediately_on_a_reason_transition(tmp_path: Path) -> None:
    """A reason change (e.g. no_data -> unconfigured) re-emits immediately,
    even well inside the dedup window -- the operator-facing signal is the
    reason itself, and suppressing a transition would hide it."""
    state_path = tmp_path / "state.json"
    t0 = datetime.now(UTC)

    # First call: no allocation event at all -> "no_data".
    first = ci_headroom_available(
        "Senkichi/job-cannon", headroom_ratio=1.5, fleet_state_path=state_path, now=t0
    )
    assert first is None

    # Now an allocation event exists, but not for this repo -> "unconfigured".
    _log_allocation(state_path, [_target("Senkichi/swole", capacity=2, demand=1)])
    second = ci_headroom_available(
        "Senkichi/job-cannon",
        headroom_ratio=1.5,
        fleet_state_path=state_path,
        now=t0 + timedelta(minutes=1),
    )
    assert second is None

    events = _unavailable_events(state_path)
    assert len(events) == 2
    assert events[0]["payload"]["reason"] == "no_data"
    assert events[1]["payload"]["reason"] == "unconfigured"
