"""Regression tests for the claim-stage vs stalled-sweep livelock (issue #571).

Live trace 2026-07-25 00:25-00:35Z: a quota-probe reviewer for PR #540 died
on a 429 within seconds; every subsequent pass the stalled-review sweep saw a
claim younger than the 5-minute stale timeout (skip), then ~27s later the
claim stage saw the same claim as stale, silently freed it, and relaunched —
resetting the clock just after the sweep looked. With pass cadence roughly
equal to the timeout the sweep never dispositioned the dead probe: no
throttle classification, frozen probe-failure counter (no exponential
backoff), one 429'd launch per pass for the whole closed quota window.

Fix: a ``review_dispatch_dispatched`` claim whose reviewer is dead stays
NOT-dispatchable through the sweep's window and only frees after the 3x
orphan backstop (``_REVIEW_DEAD_CLAIM_BACKSTOP_TIMEOUT_MINUTES``), which
exists solely for sidecar-less crashes the worker-iterating sweep cannot see.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from charlie_work.workflow import _is_review_dispatchable


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _state_with_dead_dispatched_claim(now: datetime, minutes_old: float) -> dict:
    return {
        "prs": {
            "540": {
                "number": 540,
                "issue_number": 480,
                "review_dispatch_status": "review_dispatch_dispatched",
                "review_dispatched_at": _iso(now - timedelta(minutes=minutes_old)),
                "reviewer_pid": 999999999,
                "reviewer_process_start_time": 1.0,
            }
        }
    }


def test_dead_reviewer_claim_stays_held_through_sweep_window() -> None:
    """Inside the sweep's disposition window (past the 5-minute stale timeout
    but before the orphan backstop) a dead reviewer's claim must NOT be
    claim-stage dispatchable — the sweep classifies it first.

    ``now`` is sampled once (issue #828) and threaded into both the fixture
    and the assertion so the 9-minute margin (15min backstop - 6min claim
    age) can never be eaten by wall-clock drift between fixture construction
    and the call under test.
    """
    now = datetime.now(UTC)
    state = _state_with_dead_dispatched_claim(now, minutes_old=6)

    assert _is_review_dispatchable(state, 540, {}, now=now) is False


def test_dead_reviewer_claim_frees_after_orphan_backstop() -> None:
    """A dead claim older than the orphan backstop (sweep never saw it, e.g.
    no sidecar was ever written) must still self-heal."""
    now = datetime.now(UTC)
    state = _state_with_dead_dispatched_claim(now, minutes_old=20)

    assert _is_review_dispatchable(state, 540, {}, now=now) is True


def test_live_reviewer_claim_never_dispatchable() -> None:
    """A dispatched claim with a live reviewer stays held regardless of age.
    (Liveness for our own pid: process_start_time of a real running process
    is not asserted here — a plainly dead pid plus young age covers the
    guard; this test pins the fresh-claim case.)

    ``now`` is sampled once and threaded through (issue #828); the margin
    here (15min backstop - 1min claim age = 14min) is wide, but the seam is
    exercised identically to the tight-margin test above for consistency.
    """
    now = datetime.now(UTC)
    state = _state_with_dead_dispatched_claim(now, minutes_old=1)

    assert _is_review_dispatchable(state, 540, {}, now=now) is False
