"""``self_deploy`` deferred-sync gate tests (issues #2230, #2312).

A dependency-changing update that lands while fleet live sessions are
active must defer the *whole* transition -- merge and ``uv sync`` -- with
HEAD parked at the pre-deploy commit so the checked-out source stays
consistent with the installed venv. The pending-sync marker replays the
held range on every later pass until the fleet drains.

Sibling of ``tests/test_supervise_self_deploy.py`` -- split to keep that
file under the file-size ratchet cap; shared fakes live in
``tests/_supervise_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from _supervise_fixtures import (
    _make_fake_runner,
    no_fleet_live_sessions as no_fleet_live_sessions,
)
from charlie_work import layout
from charlie_work.subprocess_runner import RunResult
from charlie_work.supervise import (
    SelfDeployResult,
    _parse_marker_timestamp,
    _pending_sync_marker_path,
    self_deploy,
)


def test_self_deploy_defers_sync_when_fleet_runners_active(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A dependency-changing update defers BEFORE the merge while fleet live
    sessions are active (issue #2312).

    The deferral decision now sits between the fetch and the merge, so a
    deferred pass leaves HEAD parked at ``before_sha`` -- the checkout's
    source and its installed venv stay consistent, and a supervisor
    restarted mid-deferral cannot crash on missing deps. ``head_changed``
    is therefore False here (previously True: the old ordering merged first
    and parked a *broken* environment).
    """
    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # HEAD
            RunResult(0, "", ""),  # fetch ok
            RunResult(0, "def456\n", ""),  # origin/main
            RunResult(1, "", ""),  # merge-base: target not an ancestor of HEAD
            RunResult(0, "pyproject.toml\nuv.lock\n", ""),  # diff
            RunResult(0, "", ""),  # uv sync (should not be reached)
        ]
    )

    def _fake_count(_fleet_dir_override: str | None) -> tuple[int, list[str]]:
        return 2, []

    monkeypatch.setattr("charlie_work.fleet_registry.count_fleet_live_sessions", _fake_count)

    result = self_deploy(tmp_path, run_command=runner)
    assert result == SelfDeployResult(
        ok=True,
        pulled=True,
        changed=True,
        synced=False,
        head_changed=False,
        from_sha="abc123",
        to_sha="def456",
        message="sync deferred: 2 runners active",
        deferred=True,
    )
    assert all(c[0] != ["uv", "sync", "--locked", "--inexact"] for c in calls)
    # Issue #2312: the merge must never have been issued -- HEAD stays parked.
    assert all(c[0][:2] != ["git", "merge"] for c in calls)
    assert [c[0] for c in calls] == [
        ["git", "rev-parse", "HEAD"],
        ["git", "fetch", "origin", "main"],
        ["git", "rev-parse", "origin/main"],
        ["git", "merge-base", "--is-ancestor", "def456", "HEAD"],
        ["git", "diff", "--name-only", "abc123..def456"],
    ]

    marker_path = _pending_sync_marker_path(layout.default_state_root(tmp_path))
    assert marker_path.exists()
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["from_sha"] == "abc123"
    assert marker["to_sha"] == "def456"
    # Issue #1855: the first deferral of an episode stamps ``written_at`` --
    # the timestamp the starvation bound measures -- and a fresh episode has
    # no ``starved_notified`` latch.
    assert _parse_marker_timestamp(marker["written_at"]) is not None
    assert "starved_notified" not in marker


def test_self_deploy_deferred_marker_replay_defers_again_without_merging(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A deferred pass holds HEAD parked; the next pass replays the marker.

    Issue #2312's companion invariant: once a deferral has parked HEAD below
    the marker's ``to_sha``, subsequent passes must keep holding the merge --
    not just the sync -- until the fleet drains. The marker from a pre-#2312
    deploy (HEAD already at ``to_sha``, covered by
    ``test_self_deploy_loud_warning_on_repeated_deferral``) still defers on
    the same ``live_count > 0`` gate; only the shape of what is held differs.
    """
    marker_path = _pending_sync_marker_path(layout.default_state_root(tmp_path))
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        json.dumps({"from_sha": "abc123", "to_sha": "def456"}), encoding="utf-8"
    )

    monkeypatch.setattr(
        "charlie_work.fleet_registry.count_fleet_live_sessions",
        lambda _fleet_dir_override: (1, []),
    )
    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # HEAD still parked at from_sha
            RunResult(0, "", ""),  # fetch ok
            RunResult(0, "def456\n", ""),  # origin/main
            RunResult(1, "", ""),  # merge-base: target not an ancestor of HEAD
            RunResult(0, "uv.lock\n", ""),  # diff
        ]
    )

    result = self_deploy(tmp_path, run_command=runner)

    assert result.ok is True
    assert result.deferred is True
    assert result.head_changed is False
    assert all(c[0][:2] != ["git", "merge"] for c in calls)
    assert all(c[0] != ["uv", "sync", "--locked", "--inexact"] for c in calls)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["to_sha"] == "def456"


def test_self_deploy_defers_via_host_sessions_port(
    tmp_path: Path, fake_host: Any, no_fleet_live_sessions: None
) -> None:
    """Issue #2230: ``_self_deploy_attempt`` must reach the fleet live-worker
    count through ``host.current().sessions.fleet_live_workers`` -- the port
    every other consumer uses -- not by calling
    ``fleet_registry.count_fleet_live_sessions`` directly.

    ``fake_host(sessions=FakeSessionCounter(...))`` only intercepts the port.
    If the call site bypassed it (e.g. reverted to the pre-#2230 direct
    ``fleet_registry`` call), this fake would never be consulted: the pinned
    ``no_fleet_live_sessions`` zero would apply instead, the sync would not
    defer, and the ``counter.calls`` assertion below would fail outright.
    """
    from charlie_work.host.fakes import FakeSessionCounter

    counter = FakeSessionCounter(fleet_workers=(2, []))
    fake_host(sessions=counter)

    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # HEAD
            RunResult(0, "", ""),  # fetch ok
            RunResult(0, "def456\n", ""),  # origin/main
            RunResult(1, "", ""),  # merge-base: target not an ancestor of HEAD
            RunResult(0, "pyproject.toml\nuv.lock\n", ""),  # diff
            RunResult(0, "", ""),  # uv sync (should not be reached)
        ]
    )

    result = self_deploy(tmp_path, run_command=runner)

    assert result.deferred is True
    assert result.synced is False
    assert result.head_changed is False
    assert result.message == "sync deferred: 2 runners active"
    assert all(c[0][:2] != ["git", "merge"] for c in calls)
    assert all(c[0] != ["uv", "sync", "--locked", "--inexact"] for c in calls)
    assert counter.calls == [("fleet_live_workers", (None,))]


def test_self_deploy_honors_state_root_override(tmp_path: Path, monkeypatch: Any) -> None:
    """Issue #720: a configured state_root moves the pending-sync marker out of default."""
    custom_state_root = tmp_path / ".var" / "devin-orchestrator"

    def _fake_count(_fleet_dir_override: str | None) -> tuple[int, list[str]]:
        return 1, []

    monkeypatch.setattr("charlie_work.fleet_registry.count_fleet_live_sessions", _fake_count)

    runner, _ = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # HEAD
            RunResult(0, "", ""),  # fetch ok
            RunResult(0, "def456\n", ""),  # origin/main
            RunResult(1, "", ""),  # merge-base: target not an ancestor of HEAD
            RunResult(0, "pyproject.toml\nuv.lock\n", ""),  # diff
        ]
    )

    result = self_deploy(tmp_path, state_root=custom_state_root, run_command=runner)
    assert result.deferred is True
    assert _pending_sync_marker_path(custom_state_root).exists()
    assert not _pending_sync_marker_path(layout.default_state_root(tmp_path)).exists()


def test_self_deploy_proceeds_when_zero_fleet_runners(
    tmp_path: Path, no_fleet_live_sessions: None
) -> None:
    """A dependency-changing pull merges and runs uv sync when no fleet live
    sessions are active."""
    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # HEAD
            RunResult(0, "", ""),  # fetch ok
            RunResult(0, "def456\n", ""),  # origin/main
            RunResult(1, "", ""),  # merge-base: target not an ancestor of HEAD
            RunResult(0, "pyproject.toml\nuv.lock\n", ""),  # diff
            RunResult(0, "", ""),  # merge --ff-only ok
            RunResult(0, "", ""),  # uv sync --locked --inexact ok
        ]
    )

    result = self_deploy(tmp_path, run_command=runner)
    assert result.ok is True
    assert result.pulled is True
    assert result.changed is True
    assert result.synced is True
    assert result.head_changed is True
    assert result.from_sha == "abc123"
    assert result.to_sha == "def456"
    assert "updated and synced" in result.message
    assert calls[-1][0] == ["uv", "sync", "--locked", "--inexact"]
    assert not _pending_sync_marker_path(layout.default_state_root(tmp_path)).exists()


def test_self_deploy_retries_sync_after_deferral(
    tmp_path: Path, monkeypatch: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """A deferred dependency sync is retried on the next pass once runners are idle.

    Post-#2312 shape: pass N parks HEAD at ``from_sha`` (the merge is held,
    not just the sync), so pass N+1 replays the marker *and* lands the held
    merge before ``uv sync --locked --inexact`` -- the deferral covers the whole
    "advance checkout + sync env" transition as one unit.
    """
    live_counts = iter([2, 0])

    def _fake_count(_fleet_dir_override: str | None) -> tuple[int, list[str]]:
        return next(live_counts), []

    monkeypatch.setattr("charlie_work.fleet_registry.count_fleet_live_sessions", _fake_count)

    # Pass N: dependency-changing update, two active runners -> defer before
    # the merge; HEAD stays at abc123 and the marker records the held range.
    first_runner, first_calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # HEAD
            RunResult(0, "", ""),  # fetch ok
            RunResult(0, "def456\n", ""),  # origin/main
            RunResult(1, "", ""),  # merge-base: target not an ancestor of HEAD
            RunResult(0, "pyproject.toml\nuv.lock\n", ""),  # diff
            RunResult(0, "", ""),  # merge/uv sync (not reached)
        ]
    )

    first = self_deploy(tmp_path, run_command=first_runner)
    assert first.synced is False
    assert first.head_changed is False
    assert first.message == "sync deferred: 2 runners active"

    marker_path = _pending_sync_marker_path(layout.default_state_root(tmp_path))
    assert marker_path.exists()

    # Pass N+1: HEAD still parked at abc123, no new origin commits, runners
    # now idle -> merge the held range, then sync from marker and clear it.
    second_runner, second_calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # HEAD (parked)
            RunResult(0, "", ""),  # fetch ok
            RunResult(0, "def456\n", ""),  # origin/main (unchanged)
            RunResult(1, "", ""),  # merge-base: target not an ancestor of HEAD
            RunResult(0, "uv.lock\n", ""),  # diff abc123..def456
            RunResult(0, "", ""),  # merge --ff-only ok
            RunResult(0, "", ""),  # uv sync --locked --inexact ok
        ]
    )

    second = self_deploy(tmp_path, run_command=second_runner)
    assert second == SelfDeployResult(
        ok=True,
        pulled=True,
        changed=True,
        synced=True,
        head_changed=True,
        from_sha="abc123",
        to_sha="def456",
        message="updated and synced: def456",
    )
    assert ["git", "merge", "--ff-only", "origin/main"] in [c[0] for c in second_calls]
    assert second_calls[-1][0] == ["uv", "sync", "--locked", "--inexact"]
    assert not marker_path.exists()


def test_self_deploy_loud_warning_on_repeated_deferral(
    tmp_path: Path, monkeypatch: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """When a pending-sync marker survives repeated passes, a warning is printed.

    Also the outage regression guard: HEAD did not move on *this* attempt
    (before == after == "def456") even though a marker from an earlier
    deferral is still present and live workers are still active. Callers
    must see ``head_changed is False`` here -- gating a watchdog restart on
    ``from_sha != to_sha`` instead (the marker's original range, from
    "abc123" to "def456") previously caused the supervisor to exit and
    relaunch every single pass without ever reaching zero live workers to
    complete the deferred sync (the total-fleet-outage bug this test guards
    against).
    """
    monkeypatch.setattr(
        "charlie_work.fleet_registry.count_fleet_live_sessions",
        lambda _fleet_dir_override: (3, []),
    )

    # Create marker from a previous deferral.
    marker_path = _pending_sync_marker_path(layout.default_state_root(tmp_path))
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        json.dumps({"from_sha": "abc123", "to_sha": "def456"}), encoding="utf-8"
    )

    runner, _ = _make_fake_runner(
        [
            RunResult(0, "def456\n", ""),  # HEAD (already at the marker's to_sha
            # -- the pre-#2312 residue shape: the deploy merged but the sync
            # was deferred, so only the marker replays)
            RunResult(0, "", ""),  # fetch ok
            RunResult(0, "def456\n", ""),  # origin/main (unchanged)
            RunResult(0, "", ""),  # uv sync (not reached)
        ]
    )

    result = self_deploy(tmp_path, run_command=runner)
    assert result.synced is False
    assert result.head_changed is False
    assert "3 runners active" in result.message
    assert marker_path.exists()

    out = capsys.readouterr().out
    assert "WARNING: pending dependency sync still deferred" in out
    assert "3 runners active" in out
    assert "abc123..def456" in out
