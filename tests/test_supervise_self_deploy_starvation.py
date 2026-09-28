"""``self_deploy`` deferred-sync starvation bound tests (issue #1855).

A pending ``uv sync`` deferral must not extend indefinitely under continuous
fleet load: the pending-sync marker carries ``written_at`` (the first
deferral of the episode), a marker older than ``starvation_seconds`` reports
``starved=True`` and emits exactly one ``self_deploy_sync_starved`` warning
event, and the sync lands the first pass the fleet drains to zero -- with no
live worker ever killed.

Sibling of ``tests/test_supervise_self_deploy.py`` -- split to keep that file
under the file-size ratchet cap; shared fakes live in
``tests/_supervise_fixtures.py``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from _supervise_fixtures import _make_fake_runner
from charlie_work import layout
from charlie_work.instrumentation import query_events
from charlie_work.subprocess_runner import RunResult
from charlie_work.supervise import (
    _parse_marker_timestamp,
    _pending_sync_marker_path,
    _self_deploy_state_path,
    self_deploy,
)


def _stale_written_at(seconds_ago: float) -> str:
    """An ISO-8601 ``Z`` marker timestamp ``seconds_ago`` in the past."""
    return (
        (datetime.now(UTC) - timedelta(seconds=seconds_ago))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _plant_marker(tmp_path: Path, *, written_at: str | None = None) -> Path:
    """Write a pending-sync marker under tmp_path's default state root.

    ``written_at=None`` produces the pre-#1855 marker shape (no timestamp).
    """
    marker_path = _pending_sync_marker_path(layout.default_state_root(tmp_path))
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"from_sha": "abc123", "to_sha": "def456"}
    if written_at is not None:
        payload["written_at"] = written_at
    marker_path.write_text(json.dumps(payload), encoding="utf-8")
    return marker_path


def _idle_pass_runner() -> Any:
    """Runner responses for a pass where HEAD sits at def456 (marker replayed)."""
    runner, calls = _make_fake_runner(
        [
            RunResult(0, "def456\n", ""),  # before HEAD
            RunResult(0, "Already up to date.\n", ""),  # pull
            RunResult(0, "def456\n", ""),  # after HEAD (unchanged)
            RunResult(0, "", ""),  # uv sync (only reached when fleet idle)
        ]
    )
    return runner, calls


def test_self_deploy_deferral_below_bound_is_not_starved(tmp_path: Path, monkeypatch: Any) -> None:
    """A pending marker younger than the bound stays a plain deferral."""
    monkeypatch.setattr(
        "charlie_work.fleet_registry.count_fleet_live_sessions",
        lambda _fleet_dir_override: (2, []),
    )
    planted_at = _stale_written_at(60)
    marker_path = _plant_marker(tmp_path, written_at=planted_at)
    runner, calls = _idle_pass_runner()

    result = self_deploy(tmp_path, run_command=runner, starvation_seconds=3600)

    assert result.ok is True
    assert result.deferred is True
    assert result.starved is False
    assert all(c[0] != ["uv", "sync"] for c in calls)
    assert query_events(_self_deploy_state_path(tmp_path), kind="self_deploy_sync_starved") == []
    # The rewrite carries the original episode timestamp forward verbatim.
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["written_at"] == planted_at


def test_self_deploy_starved_after_bound_trips(tmp_path: Path, monkeypatch: Any) -> None:
    """A marker aged past the bound reports ``starved`` and emits one
    ``self_deploy_sync_starved`` warning event.

    The deferral itself stays non-destructive -- live workers are counted
    but never touched; the caller uses ``starved`` to stop admitting new
    dispatches so the fleet drains to zero on its own.
    """
    monkeypatch.setattr(
        "charlie_work.fleet_registry.count_fleet_live_sessions",
        lambda _fleet_dir_override: (2, []),
    )
    planted_at = _stale_written_at(7200)
    marker_path = _plant_marker(tmp_path, written_at=planted_at)
    runner, calls = _idle_pass_runner()

    result = self_deploy(tmp_path, run_command=runner, starvation_seconds=3600)

    assert result.ok is True
    assert result.deferred is True
    assert result.starved is True
    assert result.synced is False
    assert all(c[0] != ["uv", "sync"] for c in calls)

    events = query_events(_self_deploy_state_path(tmp_path), kind="self_deploy_sync_starved")
    assert len(events) == 1
    assert events[0]["level"] == "warning"
    payload = events[0]["payload"]
    assert payload["pending_seconds"] >= 3600
    assert payload["starvation_seconds"] == 3600
    assert payload["live_count"] == 2
    assert payload["from_sha"] == "abc123"
    assert payload["to_sha"] == "def456"

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    # The episode timestamp is NOT reset by a starved pass -- it measures the
    # whole episode, and the notification latch is now set.
    assert marker["written_at"] == planted_at
    assert marker["starved_notified"] is True


def test_self_deploy_sync_starved_fires_once_per_episode(tmp_path: Path, monkeypatch: Any) -> None:
    """Repeated starved passes emit exactly one event.

    The ``starved_notified`` latch in the marker (not process memory) is what
    deduplicates across supervisor restarts mid-episode.
    """
    monkeypatch.setattr(
        "charlie_work.fleet_registry.count_fleet_live_sessions",
        lambda _fleet_dir_override: (2, []),
    )
    marker_path = _plant_marker(tmp_path, written_at=_stale_written_at(7200))

    for _ in range(3):
        runner, _ = _idle_pass_runner()
        result = self_deploy(tmp_path, run_command=runner, starvation_seconds=3600)
        assert result.starved is True

    events = query_events(_self_deploy_state_path(tmp_path), kind="self_deploy_sync_starved")
    assert len(events) == 1
    # The latch survived every per-pass marker rewrite.
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["starved_notified"] is True


def test_self_deploy_starved_episode_syncs_once_fleet_drains(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The deferred sync runs the first pass with zero live workers -- even
    mid-episode -- and clears the marker, ending the episode.

    Pass N starves (workers still live); pass N+1 finds the fleet idle, runs
    ``uv sync`` straight off the marker, and returns ``starved=False`` --
    ``starved`` describes this attempt only, so the caller's drain posture
    lifts immediately.
    """
    live_counts = iter([2, 0])
    monkeypatch.setattr(
        "charlie_work.fleet_registry.count_fleet_live_sessions",
        lambda _fleet_dir_override: (next(live_counts), []),
    )
    marker_path = _plant_marker(tmp_path, written_at=_stale_written_at(7200))

    starved_runner, _ = _idle_pass_runner()
    starved = self_deploy(tmp_path, run_command=starved_runner, starvation_seconds=3600)
    assert starved.deferred is True
    assert starved.starved is True
    assert marker_path.exists()

    drained_runner, drained_calls = _idle_pass_runner()
    drained = self_deploy(tmp_path, run_command=drained_runner, starvation_seconds=3600)

    assert drained.synced is True
    assert drained.starved is False
    assert drained.deferred is False
    assert drained_calls[-1][0] == ["uv", "sync"]
    assert not marker_path.exists()
    # Still exactly one starvation event for the whole episode.
    assert (
        len(query_events(_self_deploy_state_path(tmp_path), kind="self_deploy_sync_starved")) == 1
    )


def test_self_deploy_legacy_marker_without_written_at_rearms(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A pre-#1855 marker (no ``written_at``) is not starved -- its age is
    unverifiable, so the rewrite re-arms a fresh window instead of tripping
    the bound on a possibly-stale file (or crashing on ``.get``)."""
    monkeypatch.setattr(
        "charlie_work.fleet_registry.count_fleet_live_sessions",
        lambda _fleet_dir_override: (1, []),
    )
    marker_path = _plant_marker(tmp_path)  # no written_at -- legacy shape
    runner, _ = _idle_pass_runner()

    result = self_deploy(tmp_path, run_command=runner, starvation_seconds=3600)

    assert result.deferred is True
    assert result.starved is False
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert _parse_marker_timestamp(marker["written_at"]) is not None
    assert query_events(_self_deploy_state_path(tmp_path), kind="self_deploy_sync_starved") == []


def test_self_deploy_nonpositive_bound_disables_starvation(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """``starvation_seconds <= 0`` disables the bound entirely -- an
    arbitrarily old marker stays a plain deferral with no event."""
    monkeypatch.setattr(
        "charlie_work.fleet_registry.count_fleet_live_sessions",
        lambda _fleet_dir_override: (2, []),
    )
    _plant_marker(tmp_path, written_at=_stale_written_at(10**6))
    runner, _ = _idle_pass_runner()

    result = self_deploy(tmp_path, run_command=runner, starvation_seconds=0)

    assert result.deferred is True
    assert result.starved is False
    assert query_events(_self_deploy_state_path(tmp_path), kind="self_deploy_sync_starved") == []
