"""Failure-streak and zero-pass-streak alarm tests for the supervisor.

Split out of ``tests/test_supervise.py`` (issue #1562, Track 1) --
bodies are verbatim relocations; shared helpers live in
``tests/_supervise_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from _supervise_fixtures import (
    _make_fake_runner,
    no_fleet_live_sessions as no_fleet_live_sessions,
)
from charlie_work import layout
from charlie_work.instrumentation import query_events
from charlie_work.subprocess_runner import RunResult
from charlie_work.supervise import (
    SelfDeployResult,
    _pending_sync_marker_path,
    _record_self_deploy_failure_streak,
    _self_deploy_failure_counter_path,
    _self_deploy_state_path,
    _zero_pass_streak_counter_path,
    record_zero_pass_streak,
    self_deploy,
)


def _fake_pull_failure_runner() -> Callable[..., RunResult]:
    runner, _ = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),
            RunResult(
                returncode=1,
                stdout="",
                stderr="fatal: unable to access remote",
                error="command exited 1",
            ),
        ]
    )
    return runner


def test_self_deploy_failure_streak_fires_alarm_once_at_threshold(
    tmp_path: Path, no_fleet_live_sessions: None
) -> None:
    """Issue #817 item 5: three consecutive self_deploy failures cross the
    default threshold (3) and fire exactly one ``self_deploy_alarm`` event --
    not one per failure past the threshold -- and a subsequent success
    resets the counter so a later failure streak starts counting from zero
    again instead of re-alarming immediately.
    """
    state_path = _self_deploy_state_path(tmp_path)
    counter_path = _self_deploy_failure_counter_path(tmp_path)

    self_deploy(tmp_path, run_command=_fake_pull_failure_runner(), failure_alarm_threshold=3)
    self_deploy(tmp_path, run_command=_fake_pull_failure_runner(), failure_alarm_threshold=3)
    assert query_events(state_path, kind="self_deploy_alarm") == []

    self_deploy(tmp_path, run_command=_fake_pull_failure_runner(), failure_alarm_threshold=3)
    alarms = query_events(state_path, kind="self_deploy_alarm")
    assert len(alarms) == 1
    assert alarms[0]["payload"]["consecutive_failures"] == 3
    assert alarms[0]["level"] == "error"

    # A fourth consecutive failure must not fire a second alarm.
    self_deploy(tmp_path, run_command=_fake_pull_failure_runner(), failure_alarm_threshold=3)
    assert len(query_events(state_path, kind="self_deploy_alarm")) == 1

    counter = json.loads(counter_path.read_text(encoding="utf-8"))
    assert counter["consecutive_failures"] == 4

    # A success resets the streak.
    success_runner, _ = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),
            RunResult(0, "Already up to date.\n", ""),
            RunResult(0, "abc123\n", ""),
        ]
    )
    self_deploy(tmp_path, run_command=success_runner, failure_alarm_threshold=3)
    counter_after_success = json.loads(counter_path.read_text(encoding="utf-8"))
    assert counter_after_success["consecutive_failures"] == 0

    # Two more failures after the reset must not re-fire the alarm yet
    # (streak restarted at 0, threshold is 3).
    self_deploy(tmp_path, run_command=_fake_pull_failure_runner(), failure_alarm_threshold=3)
    self_deploy(tmp_path, run_command=_fake_pull_failure_runner(), failure_alarm_threshold=3)
    assert len(query_events(state_path, kind="self_deploy_alarm")) == 1


def _deferred_result() -> SelfDeployResult:
    """A genuine deferral: a pending sync was postponed because a fleet
    worker was live (issue #858). ``deferred=True`` is what the streak
    recorder actually keys on -- ``ok=True, synced=False`` alone is *not*
    sufficient, since "already up to date" and "code-only update" share that
    shape without anything pending to sync."""
    return SelfDeployResult(
        ok=True,
        pulled=True,
        changed=True,
        synced=False,
        message="sync deferred: 1 runner active",
        deferred=True,
    )


def _failed_result() -> SelfDeployResult:
    return SelfDeployResult(
        ok=False, pulled=False, changed=False, synced=False, error="uv sync failed"
    )


def _synced_result() -> SelfDeployResult:
    return SelfDeployResult(
        ok=True, pulled=True, changed=True, synced=True, message="updated and synced: abc123"
    )


def test_self_deploy_deferred_sync_does_not_reset_failure_streak(tmp_path: Path) -> None:
    """Issue #858: a deferred sync (``ok=True, synced=False``) carries no
    information about whether the pending sync would have succeeded, so it
    must leave the consecutive-failure streak unchanged -- neither reset
    (today's bug: any ``ok`` result resets to 0, so an interleaved deferral
    silently rewinds the counter and the alarm threshold is never reached)
    nor incremented.

    Replays the live sequence from the issue: fail, fail, defer, fail with
    threshold=3. Before the fix this fires zero alarms, because the deferral
    between the second and third failure resets the counter to 0. The fix
    must fire exactly one alarm, on the final failure.
    """
    state_path = _self_deploy_state_path(tmp_path)
    counter_path = _self_deploy_failure_counter_path(tmp_path)

    _record_self_deploy_failure_streak(tmp_path, _failed_result(), threshold=3)
    _record_self_deploy_failure_streak(tmp_path, _failed_result(), threshold=3)
    counter = json.loads(counter_path.read_text(encoding="utf-8"))
    assert counter["consecutive_failures"] == 2
    assert query_events(state_path, kind="self_deploy_alarm") == []

    # A deferral interleaves here -- it must not touch the streak.
    _record_self_deploy_failure_streak(tmp_path, _deferred_result(), threshold=3)
    counter = json.loads(counter_path.read_text(encoding="utf-8"))
    assert counter["consecutive_failures"] == 2, (
        "a deferred sync (ok=True, synced=False) must leave the streak "
        "unchanged, not reset it to 0"
    )
    assert query_events(state_path, kind="self_deploy_alarm") == []

    # Third genuine failure must cross the threshold and fire exactly once.
    _record_self_deploy_failure_streak(tmp_path, _failed_result(), threshold=3)
    counter = json.loads(counter_path.read_text(encoding="utf-8"))
    assert counter["consecutive_failures"] == 3
    alarms = query_events(state_path, kind="self_deploy_alarm")
    assert len(alarms) == 1
    assert alarms[0]["payload"]["consecutive_failures"] == 3


def test_self_deploy_pure_deferral_run_never_alarms_or_resets(tmp_path: Path) -> None:
    """Issue #858 acceptance criterion 5: a long run of pure deferrals never
    fires the alarm and never resets a pre-existing nonzero streak -- a
    healthy fleet that defers for hours (because workers are always live)
    must not have its outage history silently erased.
    """
    state_path = _self_deploy_state_path(tmp_path)
    counter_path = _self_deploy_failure_counter_path(tmp_path)

    _record_self_deploy_failure_streak(tmp_path, _failed_result(), threshold=3)
    _record_self_deploy_failure_streak(tmp_path, _failed_result(), threshold=3)
    counter = json.loads(counter_path.read_text(encoding="utf-8"))
    assert counter["consecutive_failures"] == 2

    for _ in range(50):
        _record_self_deploy_failure_streak(tmp_path, _deferred_result(), threshold=3)

    counter = json.loads(counter_path.read_text(encoding="utf-8"))
    assert counter["consecutive_failures"] == 2
    assert query_events(state_path, kind="self_deploy_alarm") == []


def test_self_deploy_genuine_sync_resets_failure_streak(tmp_path: Path) -> None:
    """A genuinely synced result (``ok=True, synced=True``) still resets the
    streak to 0 -- unchanged from today's behavior, and distinct from a
    deferral, which must not reset it.
    """
    counter_path = _self_deploy_failure_counter_path(tmp_path)

    _record_self_deploy_failure_streak(tmp_path, _failed_result(), threshold=3)
    _record_self_deploy_failure_streak(tmp_path, _failed_result(), threshold=3)
    counter = json.loads(counter_path.read_text(encoding="utf-8"))
    assert counter["consecutive_failures"] == 2

    _record_self_deploy_failure_streak(tmp_path, _synced_result(), threshold=3)
    counter = json.loads(counter_path.read_text(encoding="utf-8"))
    assert counter["consecutive_failures"] == 0


def test_self_deploy_streak_survives_a_real_deferral_end_to_end(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Issue #858, driven through the public ``self_deploy()`` entry point
    rather than hand-built ``SelfDeployResult`` values -- this is the wiring
    check: it proves the *real* deferral branch in ``_self_deploy_attempt``
    (live_count > 0, ~supervise.py:877) actually produces a result the
    streak recorder classifies as deferred, not just that the recorder's
    arithmetic is correct for a hand-built one.

    Replays the live incident shape: a failing ``uv sync`` interleaved with a
    genuine deferral once fleet workers come up, threshold=3.

    1. dep-change pull, 0 runners, uv sync fails -> streak 1
    2. marker still pending, 0 runners, uv sync fails again -> streak 2
    3. marker still pending, 2 runners live -> real deferral -> streak stays 2
    4. marker still pending, 0 runners, uv sync fails again -> streak 3, one alarm
    """
    state_path = _self_deploy_state_path(tmp_path)
    counter_path = _self_deploy_failure_counter_path(tmp_path)
    live_counts = iter([0, 0, 2, 0])

    def _fake_count(_fleet_dir_override: str | None) -> tuple[int, list[str]]:
        return next(live_counts), []

    monkeypatch.setattr("charlie_work.fleet_registry.count_fleet_live_sessions", _fake_count)

    # Pass 1: dependency-changing pull, 0 runners -> attempts sync, sync fails.
    runner1, _ = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # before HEAD
            RunResult(0, "", ""),  # pull ok
            RunResult(0, "def456\n", ""),  # after HEAD
            RunResult(0, "pyproject.toml\nuv.lock\n", ""),  # diff
            RunResult(1, "", "uv sync failed: resolution error"),  # uv sync fails
        ]
    )
    result1 = self_deploy(tmp_path, run_command=runner1, failure_alarm_threshold=3)
    assert result1.ok is False
    assert result1.deferred is False
    assert json.loads(counter_path.read_text(encoding="utf-8"))["consecutive_failures"] == 1
    assert _pending_sync_marker_path(layout.default_state_root(tmp_path)).exists()

    # Pass 2: no new commits, marker still pending, 0 runners -> retries sync, fails again.
    runner2, _ = _make_fake_runner(
        [
            RunResult(0, "def456\n", ""),  # before HEAD
            RunResult(0, "Already up to date.\n", ""),  # pull
            RunResult(0, "def456\n", ""),  # after HEAD (unchanged)
            RunResult(1, "", "uv sync failed: resolution error"),  # uv sync fails
        ]
    )
    result2 = self_deploy(tmp_path, run_command=runner2, failure_alarm_threshold=3)
    assert result2.ok is False
    assert result2.deferred is False
    assert json.loads(counter_path.read_text(encoding="utf-8"))["consecutive_failures"] == 2
    assert query_events(state_path, kind="self_deploy_alarm") == []

    # Pass 3: no new commits, marker still pending, 2 runners now live -> real deferral.
    runner3, _ = _make_fake_runner(
        [
            RunResult(0, "def456\n", ""),  # before HEAD
            RunResult(0, "Already up to date.\n", ""),  # pull
            RunResult(0, "def456\n", ""),  # after HEAD (unchanged)
        ]
    )
    result3 = self_deploy(tmp_path, run_command=runner3, failure_alarm_threshold=3)
    assert result3.ok is True
    assert result3.deferred is True
    assert json.loads(counter_path.read_text(encoding="utf-8"))["consecutive_failures"] == 2, (
        "a real deferral through the public entry point must not reset or increment the streak"
    )
    assert query_events(state_path, kind="self_deploy_alarm") == []

    # Pass 4: no new commits, marker still pending, 0 runners -> retries sync, fails
    # a third time -> crosses threshold=3, fires exactly one alarm.
    runner4, _ = _make_fake_runner(
        [
            RunResult(0, "def456\n", ""),  # before HEAD
            RunResult(0, "Already up to date.\n", ""),  # pull
            RunResult(0, "def456\n", ""),  # after HEAD (unchanged)
            RunResult(1, "", "uv sync failed: resolution error"),  # uv sync fails
        ]
    )
    result4 = self_deploy(tmp_path, run_command=runner4, failure_alarm_threshold=3)
    assert result4.ok is False
    assert json.loads(counter_path.read_text(encoding="utf-8"))["consecutive_failures"] == 3
    alarms = query_events(state_path, kind="self_deploy_alarm")
    assert len(alarms) == 1
    assert alarms[0]["payload"]["consecutive_failures"] == 3


def test_record_self_deploy_failure_streak_creates_state_dir_when_absent(
    tmp_path: Path,
) -> None:
    """``_record_self_deploy_failure_streak`` must create its own state
    directory before acquiring ``state_lock`` -- it must not rely on
    ``_log_self_deploy_outcome``'s ``log_event()`` call (which shares the same
    parent directory) having already created it as a side effect. Calling it
    directly, with no prior state-dir-creating call in this process, isolates
    the bug: without the pre-lock ``mkdir``, ``state_lock`` tries to create a
    sibling ``.lock`` file in a nonexistent directory and raises, which would
    violate ``self_deploy``'s documented never-raises contract.
    """
    counter_path = _self_deploy_failure_counter_path(tmp_path)
    assert not counter_path.parent.exists()

    failure = SelfDeployResult(ok=False, pulled=False, changed=False, synced=False, error="boom")
    _record_self_deploy_failure_streak(tmp_path, failure, threshold=3)  # must not raise

    assert counter_path.exists()
    counter = json.loads(counter_path.read_text(encoding="utf-8"))
    assert counter["consecutive_failures"] == 1


def test_self_deploy_failure_alarm_threshold_zero_disables(
    tmp_path: Path, no_fleet_live_sessions: None
) -> None:
    """``failure_alarm_threshold<=0`` disables the alarm entirely, matching
    ``AutoMergeConfig.failed_attempt_alarm``'s "0 disables" convention.
    """
    state_path = _self_deploy_state_path(tmp_path)
    for _ in range(5):
        self_deploy(tmp_path, run_command=_fake_pull_failure_runner(), failure_alarm_threshold=0)
    assert query_events(state_path, kind="self_deploy_alarm") == []


def test_zero_pass_streak_fires_alarm_once_at_threshold(tmp_path: Path) -> None:
    """Issue #855: three consecutive zero-repo-pass cycles (with at least one
    repo configured) cross the default threshold (3) and fire exactly one
    ``supervisor_zero_pass_alarm`` event -- not one per cycle past the
    threshold -- and a subsequent cycle with repo_passes > 0 resets the
    counter so a later 0-pass streak starts counting from zero again instead
    of re-alarming immediately. Mirrors
    test_self_deploy_failure_streak_fires_alarm_once_at_threshold exactly.
    """
    state_path = _self_deploy_state_path(tmp_path)
    counter_path = _zero_pass_streak_counter_path(tmp_path)

    record_zero_pass_streak(tmp_path, repo_passes=0, repos_configured=True, threshold=3)
    record_zero_pass_streak(tmp_path, repo_passes=0, repos_configured=True, threshold=3)
    assert query_events(state_path, kind="supervisor_zero_pass_alarm") == []

    record_zero_pass_streak(tmp_path, repo_passes=0, repos_configured=True, threshold=3)
    alarms = query_events(state_path, kind="supervisor_zero_pass_alarm")
    assert len(alarms) == 1
    assert alarms[0]["payload"]["consecutive_zero_pass_cycles"] == 3
    assert alarms[0]["payload"]["threshold"] == 3
    assert alarms[0]["level"] == "error"

    # A fourth consecutive zero-pass cycle must not fire a second alarm.
    record_zero_pass_streak(tmp_path, repo_passes=0, repos_configured=True, threshold=3)
    assert len(query_events(state_path, kind="supervisor_zero_pass_alarm")) == 1

    counter = json.loads(counter_path.read_text(encoding="utf-8"))
    assert counter["consecutive_zero_pass_cycles"] == 4

    # A cycle with repo_passes > 0 resets the streak.
    record_zero_pass_streak(tmp_path, repo_passes=2, repos_configured=True, threshold=3)
    counter_after_reset = json.loads(counter_path.read_text(encoding="utf-8"))
    assert counter_after_reset["consecutive_zero_pass_cycles"] == 0

    # Two more zero-pass cycles after the reset must not re-fire the alarm
    # yet (streak restarted at 0, threshold is 3).
    record_zero_pass_streak(tmp_path, repo_passes=0, repos_configured=True, threshold=3)
    record_zero_pass_streak(tmp_path, repo_passes=0, repos_configured=True, threshold=3)
    assert len(query_events(state_path, kind="supervisor_zero_pass_alarm")) == 1


def test_zero_pass_streak_never_fires_when_repos_not_configured(tmp_path: Path) -> None:
    """Issue #855 acceptance criterion 4: a fleet with zero configured repos
    is a configuration state, not an incident. Any number of zero-repo-pass
    cycles with ``repos_configured=False`` must never fire the alarm, and
    must not move the counter in either direction (the counter file is not
    even created).
    """
    state_path = _self_deploy_state_path(tmp_path)
    counter_path = _zero_pass_streak_counter_path(tmp_path)

    for _ in range(10):
        record_zero_pass_streak(tmp_path, repo_passes=0, repos_configured=False, threshold=3)

    assert query_events(state_path, kind="supervisor_zero_pass_alarm") == []
    assert not counter_path.exists()


def test_zero_pass_streak_never_fires_when_repo_passes_positive(tmp_path: Path) -> None:
    """A cycle that performs repo work (repo_passes > 0) never emits the
    alarm, however many times it repeats.
    """
    state_path = _self_deploy_state_path(tmp_path)

    for _ in range(10):
        record_zero_pass_streak(tmp_path, repo_passes=1, repos_configured=True, threshold=3)

    assert query_events(state_path, kind="supervisor_zero_pass_alarm") == []


def test_record_zero_pass_streak_creates_state_dir_when_absent(tmp_path: Path) -> None:
    """``record_zero_pass_streak`` must create its own state directory
    before acquiring ``state_lock`` -- it must not rely on some other call
    having already created it as a side effect. Calling it directly, with no
    prior state-dir-creating call in this process, isolates the bug: without
    the pre-lock ``mkdir``, ``state_lock`` tries to create a sibling
    ``.lock`` file in a nonexistent directory and raises, which would
    propagate out of run_fleet_supervise's post-loop bookkeeping. Mirrors
    test_record_self_deploy_failure_streak_creates_state_dir_when_absent.
    """
    counter_path = _zero_pass_streak_counter_path(tmp_path)
    assert not counter_path.parent.exists()

    # must not raise
    record_zero_pass_streak(tmp_path, repo_passes=0, repos_configured=True, threshold=3)

    assert counter_path.exists()
    counter = json.loads(counter_path.read_text(encoding="utf-8"))
    assert counter["consecutive_zero_pass_cycles"] == 1


def test_zero_pass_alarm_threshold_zero_disables(tmp_path: Path) -> None:
    """``threshold<=0`` disables the alarm entirely, matching
    ``self_deploy_failure_alarm``'s "0 disables" convention.
    """
    state_path = _self_deploy_state_path(tmp_path)
    for _ in range(5):
        record_zero_pass_streak(tmp_path, repo_passes=0, repos_configured=True, threshold=0)
    assert query_events(state_path, kind="supervisor_zero_pass_alarm") == []
