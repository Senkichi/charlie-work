"""``should_exit`` drained/activity predicate tests.

Split out of ``tests/test_supervise.py`` (issue #1562, Track 1) --
bodies are verbatim relocations; shared helpers live in
``tests/_supervise_fixtures.py``.
"""

from __future__ import annotations

from _supervise_fixtures import _active_result, _drained_result
from charlie_work.supervise import should_exit
from charlie_work.workflow import CommandResult


def test_should_exit_drained_returns_true() -> None:
    result = _drained_result()
    assert should_exit(result, live_count=0) is True


def test_should_exit_live_workers_returns_false() -> None:
    result = _drained_result()
    assert should_exit(result, live_count=2) is False


def test_should_exit_open_prs_returns_false() -> None:
    result = _active_result(open_prs=1)
    assert should_exit(result, live_count=0) is False


def test_should_exit_dispatched_returns_false() -> None:
    result = _active_result(dispatched=1)
    assert should_exit(result, live_count=0) is False


def test_should_exit_rework_dispatched_returns_false() -> None:
    result = _active_result(rework=1)
    assert should_exit(result, live_count=0) is False


def test_should_exit_merged_returns_false() -> None:
    result = _active_result(merged=1)
    assert should_exit(result, live_count=0) is False


def test_should_exit_provider_throttled_dispatch_returns_false() -> None:
    """Regression for finding #2: dispatch() defers with selected_count=0 and
    deferred_reason="provider_throttled" while queued issues are still
    waiting on the throttle cooldown to clear. With zero live workers and no
    open PRs, the old should_exit() misread this as fully drained.
    """
    result = CommandResult(
        False,
        "dispatch deferred: provider throttled until 2026-07-10T00:00:00Z",
        {
            "dispatch": {"selected_count": 0, "deferred_reason": "provider_throttled"},
            "dispatch_rework": {"selected_count": 0},
            "merges": [],
            "open_tracked_prs": 0,
        },
    )
    assert should_exit(result, live_count=0) is False


def test_should_exit_failed_merge_attempts_only_still_blocked_by_open_prs() -> None:
    """Regression: merge_ready() appends one "merges" entry per approved PR
    regardless of outcome. All attempts failing (can_merge=False) must not be
    misread as merge "activity" that would let should_exit ignore the still-
    open PRs -- open_tracked_prs > 0 keeps the loop alive on its own, and the
    honest (successes-only) merged count must not accidentally short-circuit
    that.
    """
    result = _active_result(merge_failed=3, open_prs=2)
    assert should_exit(result, live_count=0) is False


def test_should_exit_all_failed_merge_attempts_no_other_activity_exits() -> None:
    """The honest fix: failed merge attempts alone (no live workers, no
    dispatches, no open PRs) are not "activity" -- should_exit returns True.
    The old implementation (len(data["merges"]) as the merged count) would
    have kept the loop alive here since 3 failed-attempt entries still made
    len(merges) == 3 look nonzero.
    """
    result = _active_result(merge_failed=3, open_prs=0)
    assert should_exit(result, live_count=0) is True


def test_should_exit_provider_throttled_rework_returns_false() -> None:
    """Same as above but the throttle hits dispatch_rework instead of dispatch."""
    result = CommandResult(
        False,
        "rework dispatch deferred: provider throttled until 2026-07-10T00:00:00Z",
        {
            "dispatch": {"selected_count": 0},
            "dispatch_rework": {"selected_count": 0, "deferred_reason": "provider_throttled"},
            "merges": [],
            "open_tracked_prs": 0,
        },
    )
    assert should_exit(result, live_count=0) is False


def test_should_exit_fleet_lock_held_returns_false() -> None:
    """Dispatch deferred because another repo holds the fleet lock is not drained."""
    result = CommandResult(
        True,
        "dispatch deferred: fleet lock held",
        {
            "dispatch": {"selected_count": 0, "deferred_reason": "fleet_lock_held"},
            "dispatch_rework": {"selected_count": 0},
            "merges": [],
            "open_tracked_prs": 0,
        },
    )
    assert should_exit(result, live_count=0) is False
