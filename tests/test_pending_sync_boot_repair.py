"""Boot-time pending-sync repair tests (issue #2312).

``heal_pending_sync_at_boot`` is the backstop for the skew the old
self-deploy ordering could create: a deploy whose ``uv sync`` was deferred
under live workers had already advanced HEAD, so a supervisor restarting
onto the new tree loaded config against the *old* venv and crashed before
any self-deploy pass could retry the sync. The boot repair runs before
config load: a pending-sync marker whose ``to_sha`` is the checked-out
HEAD -- or a ``uv sync --locked --check`` probe reporting drift -- triggers
``uv sync --locked`` when no fleet workers are live, clears the marker, and
emits one ``self_deploy_boot_sync`` event.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from _supervise_fixtures import _make_fake_runner
from charlie_work import layout
from charlie_work.instrumentation import query_events
from charlie_work.pending_sync import (
    BootSyncRepair,
    heal_pending_sync_at_boot,
)
from charlie_work.subprocess_runner import RunResult


def _plant_marker(state_root: Path, *, from_sha: str = "abc123", to_sha: str = "def456") -> Path:
    """Write a pending-sync marker under ``state_root``."""
    marker_path = layout.pending_sync_path(state_root)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps({"from_sha": from_sha, "to_sha": to_sha}), encoding="utf-8")
    return marker_path


def _state_root(tmp_path: Path) -> Path:
    return layout.default_state_root(tmp_path)


def _heal(
    tmp_path: Path,
    *,
    live_count: int = 0,
    run_command: Any,
) -> BootSyncRepair:
    """Invoke the boot repair with the marker/state paths under ``tmp_path``."""
    return heal_pending_sync_at_boot(
        tmp_path,
        state_root=_state_root(tmp_path),
        live_count=lambda: live_count,
        run_command=run_command,
    )


def test_boot_repair_syncs_and_clears_marker_at_head(tmp_path: Path) -> None:
    """A pending-sync marker whose ``to_sha`` equals HEAD means the deploy's
    merge landed but its sync never did -- the boot repair runs
    ``uv sync --locked``, clears the marker, and emits the event."""
    state_root = _state_root(tmp_path)
    marker_path = _plant_marker(state_root)
    state_path = layout.state_file_path(state_root)

    runner, calls = _make_fake_runner(
        [
            RunResult(0, "def456\n", ""),  # git rev-parse HEAD (marker check)
            RunResult(0, "", ""),  # uv sync --locked
        ]
    )

    outcome = _heal(tmp_path, run_command=runner)

    assert outcome == BootSyncRepair(
        detected_via="marker", synced=True, deferred=False, detail=None
    )
    assert [c[0] for c in calls] == [
        ["git", "rev-parse", "HEAD"],
        ["uv", "sync", "--locked"],
    ]
    assert not marker_path.exists()

    events = query_events(state_path, kind="self_deploy_boot_sync")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["detected_via"] == "marker"
    assert payload["synced"] is True
    assert payload["deferred"] is False
    assert payload["cleared_marker"] is True


def test_boot_repair_defers_to_live_workers(tmp_path: Path) -> None:
    """Live fleet workers hold the repair: no ``uv sync``, marker kept, and
    the event still records that the skew was found."""
    state_root = _state_root(tmp_path)
    marker_path = _plant_marker(state_root)
    state_path = layout.state_file_path(state_root)

    runner, calls = _make_fake_runner(
        [
            RunResult(0, "def456\n", ""),  # git rev-parse HEAD (marker check)
        ]
    )

    outcome = _heal(tmp_path, live_count=2, run_command=runner)

    assert outcome.deferred is True
    assert outcome.synced is False
    assert outcome.detected_via == "marker"
    assert [c[0] for c in calls] == [["git", "rev-parse", "HEAD"]]
    assert marker_path.exists()

    events = query_events(state_path, kind="self_deploy_boot_sync")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["deferred"] is True
    assert payload["live_count"] == 2
    assert payload["synced"] is False


def test_boot_repair_ignores_marker_not_at_head_when_env_consistent(
    tmp_path: Path,
) -> None:
    """A marker whose ``to_sha`` is NOT the checked-out HEAD is an in-flight
    deferral (HEAD parked at the pre-deploy commit): the venv still matches
    the parked tree, self_deploy owns the retry -- the boot repair leaves it
    alone when the probe agrees the env is in sync."""
    state_root = _state_root(tmp_path)
    marker_path = _plant_marker(state_root, to_sha="zzz999")
    (tmp_path / "uv.lock").write_bytes(b"lock\n")
    state_path = layout.state_file_path(state_root)

    runner, calls = _make_fake_runner(
        [
            RunResult(0, "def456\n", ""),  # git rev-parse HEAD
            RunResult(0, "", ""),  # uv sync --locked --check: env consistent
        ]
    )

    outcome = _heal(tmp_path, run_command=runner)

    assert outcome.detected_via is None
    assert outcome.synced is False
    assert [c[0] for c in calls] == [
        ["git", "rev-parse", "HEAD"],
        ["uv", "sync", "--locked", "--check"],
    ]
    # The marker belongs to the still-pending deploy; the repair must not
    # consume it.
    assert marker_path.exists()
    assert query_events(state_path, kind="self_deploy_boot_sync") == []


def test_boot_repair_probe_detects_skew_without_marker(tmp_path: Path) -> None:
    """No marker at all, but ``uv sync --check`` reports the venv drifted from
    the locked state (e.g. a crash between the merge and the marker write --
    or an out-of-band pull): the probe alone triggers the repair."""
    state_root = _state_root(tmp_path)
    (tmp_path / "uv.lock").write_bytes(b"lock\n")
    state_path = layout.state_file_path(state_root)

    runner, calls = _make_fake_runner(
        [
            RunResult(1, "", "environment is not in sync"),  # probe
            RunResult(0, "", ""),  # uv sync --locked
        ]
    )

    outcome = _heal(tmp_path, run_command=runner)

    assert outcome == BootSyncRepair(
        detected_via="probe", synced=True, deferred=False, detail=None
    )
    assert [c[0] for c in calls] == [
        ["uv", "sync", "--locked", "--check"],
        ["uv", "sync", "--locked"],
    ]
    events = query_events(state_path, kind="self_deploy_boot_sync")
    assert len(events) == 1
    assert events[0]["payload"]["detected_via"] == "probe"
    assert events[0]["payload"]["cleared_marker"] is False


def test_boot_repair_failed_sync_keeps_marker(tmp_path: Path) -> None:
    """A failed ``uv sync`` leaves the marker pending so the next boot (or the
    next self_deploy pass) retries, and the event carries the failure."""
    state_root = _state_root(tmp_path)
    marker_path = _plant_marker(state_root)
    state_path = layout.state_file_path(state_root)

    runner, _calls = _make_fake_runner(
        [
            RunResult(0, "def456\n", ""),  # git rev-parse HEAD
            RunResult(1, "", "ResolutionImpossible: no matching version"),
        ]
    )

    outcome = _heal(tmp_path, run_command=runner)

    assert outcome.synced is False
    assert outcome.detected_via == "marker"
    assert outcome.detail is not None
    assert "ResolutionImpossible" in outcome.detail
    assert marker_path.exists()
    events = query_events(state_path, kind="self_deploy_boot_sync")
    assert len(events) == 1
    assert events[0]["payload"]["synced"] is False
    assert "ResolutionImpossible" in events[0]["payload"]["error"]


def test_boot_repair_noop_when_nothing_pending(tmp_path: Path) -> None:
    """No marker and a clean probe: no sync, no event -- the common boot is silent."""
    state_root = _state_root(tmp_path)
    (tmp_path / "uv.lock").write_bytes(b"lock\n")
    state_path = layout.state_file_path(state_root)

    runner, calls = _make_fake_runner(
        [
            RunResult(0, "", ""),  # uv sync --locked --check: consistent
        ]
    )

    outcome = _heal(tmp_path, run_command=runner)

    assert outcome == BootSyncRepair(detected_via=None, synced=False, deferred=False, detail=None)
    assert [c[0] for c in calls] == [["uv", "sync", "--locked", "--check"]]
    assert query_events(state_path, kind="self_deploy_boot_sync") == []


def test_boot_repair_skips_probe_without_lockfile(tmp_path: Path) -> None:
    """A checkout with no ``uv.lock`` cannot have a lock-managed venv skew the
    repair owns; without a marker there is nothing to do at all."""
    runner, calls = _make_fake_runner([])

    outcome = _heal(tmp_path, run_command=runner)

    assert outcome.detected_via is None
    assert calls == []


def test_boot_repair_never_raises(tmp_path: Path) -> None:
    """The repair is a startup backstop: any internal failure -- here a live-count
    probe that itself throws -- returns an outcome instead of breaking boot."""
    state_root = _state_root(tmp_path)
    _plant_marker(state_root)

    def _exploding_count() -> int:
        raise RuntimeError("fleet registry exploded")

    runner, _calls = _make_fake_runner(
        [
            RunResult(0, "def456\n", ""),  # git rev-parse HEAD
        ]
    )

    outcome = heal_pending_sync_at_boot(
        tmp_path,
        state_root=state_root,
        live_count=_exploding_count,
        run_command=runner,
    )

    assert outcome.synced is False
    assert outcome.detail is not None
    assert "fleet registry exploded" in outcome.detail
