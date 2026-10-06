"""Boot-time pending-sync repair tests (issue #2312).

``heal_pending_sync_at_boot`` is the backstop for the skew the old
self-deploy ordering could create: a deploy whose ``uv sync`` was deferred
under live workers had already advanced HEAD, so a supervisor restarting
onto the new tree loaded config against the *old* venv and crashed before
any self-deploy pass could retry the sync. The boot repair runs before
config load: a pending-sync marker whose ``to_sha`` is the checked-out
HEAD -- or a ``uv sync --locked --check --inexact`` probe reporting a
clean "would change" exit -- triggers ``uv sync --locked --inexact`` when
no fleet workers are live, clears the marker, and emits one
``self_deploy_boot_sync`` event.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from _supervise_fixtures import _make_fake_runner
from charlie_work import layout
from charlie_work.instrumentation import query_events
from charlie_work.pending_sync import (
    BootSyncRepair,
    heal_pending_sync_at_boot,
)
from charlie_work.subprocess_runner import RunResult, run_captured


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
    ``uv sync --locked --inexact``, clears the marker, and emits the event."""
    state_root = _state_root(tmp_path)
    marker_path = _plant_marker(state_root)
    state_path = layout.state_file_path(state_root)

    runner, calls = _make_fake_runner(
        [
            RunResult(0, "def456\n", ""),  # git rev-parse HEAD (marker check)
            RunResult(0, "", ""),  # uv sync --locked --inexact
        ]
    )

    outcome = _heal(tmp_path, run_command=runner)

    assert outcome == BootSyncRepair(
        detected_via="marker", synced=True, deferred=False, detail=None
    )
    assert [c[0] for c in calls] == [
        ["git", "rev-parse", "HEAD"],
        ["uv", "sync", "--locked", "--inexact"],
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
            RunResult(0, "", ""),  # uv sync --locked --check --inexact: env consistent
        ]
    )

    outcome = _heal(tmp_path, run_command=runner)

    assert outcome.detected_via is None
    assert outcome.synced is False
    assert [c[0] for c in calls] == [
        ["git", "rev-parse", "HEAD"],
        ["uv", "sync", "--locked", "--check", "--inexact"],
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
            RunResult(0, "", ""),  # uv sync --locked --inexact
        ]
    )

    outcome = _heal(tmp_path, run_command=runner)

    assert outcome == BootSyncRepair(
        detected_via="probe", synced=True, deferred=False, detail=None
    )
    assert [c[0] for c in calls] == [
        ["uv", "sync", "--locked", "--check", "--inexact"],
        ["uv", "sync", "--locked", "--inexact"],
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
            RunResult(0, "", ""),  # uv sync --locked --check --inexact: consistent
        ]
    )

    outcome = _heal(tmp_path, run_command=runner)

    assert outcome == BootSyncRepair(detected_via=None, synced=False, deferred=False, detail=None)
    assert [c[0] for c in calls] == [["uv", "sync", "--locked", "--check", "--inexact"]]
    assert query_events(state_path, kind="self_deploy_boot_sync") == []


def test_boot_repair_skips_probe_without_lockfile(tmp_path: Path) -> None:
    """A checkout with no ``uv.lock`` cannot have a lock-managed venv skew the
    repair owns; without a marker there is nothing to do at all."""
    runner, calls = _make_fake_runner([])

    outcome = _heal(tmp_path, run_command=runner)

    assert outcome.detected_via is None
    assert calls == []


def test_boot_repair_marker_at_head_syncs_without_lockfile(tmp_path: Path) -> None:
    """Marker present, no ``uv.lock``: the replaying marker alone decides --
    the lockfile gates only the probe, so the marker path still syncs and
    clears. (The mirror of ``test_boot_repair_skips_probe_without_lockfile``,
    which covers the no-marker case.)"""
    state_root = _state_root(tmp_path)
    marker_path = _plant_marker(state_root)

    runner, calls = _make_fake_runner(
        [
            RunResult(0, "def456\n", ""),  # git rev-parse HEAD (marker check)
            RunResult(0, "", ""),  # uv sync --locked --inexact
        ]
    )

    outcome = _heal(tmp_path, run_command=runner)

    assert outcome == BootSyncRepair(
        detected_via="marker", synced=True, deferred=False, detail=None
    )
    assert [c[0] for c in calls] == [
        ["git", "rev-parse", "HEAD"],
        ["uv", "sync", "--locked", "--inexact"],
    ]
    assert not marker_path.exists()


def test_boot_repair_marker_not_at_head_noop_without_lockfile(tmp_path: Path) -> None:
    """Marker present but not replaying, no ``uv.lock``: the marker's merge is
    still owned by self_deploy and the probe is skipped, so the whole pass is
    a silent no-op consuming exactly one command."""
    state_root = _state_root(tmp_path)
    marker_path = _plant_marker(state_root, to_sha="zzz999")

    runner, calls = _make_fake_runner(
        [
            RunResult(0, "def456\n", ""),  # git rev-parse HEAD (marker check)
        ]
    )

    outcome = _heal(tmp_path, run_command=runner)

    assert outcome == BootSyncRepair()
    assert [c[0] for c in calls] == [["git", "rev-parse", "HEAD"]]
    assert marker_path.exists()


def test_boot_repair_probe_uv_error_is_not_drift(tmp_path: Path) -> None:
    """Only a clean "would change" exit (1) counts as drift. A uv error --
    unparseable lockfile, unsupported ``--check``, resolution failure -- must
    not trigger a blind sync: the repair reports the failure and emits the
    event, leaving the marker/markerless state untouched."""
    state_root = _state_root(tmp_path)
    (tmp_path / "uv.lock").write_bytes(b"lock\n")
    state_path = layout.state_file_path(state_root)

    runner, calls = _make_fake_runner(
        [
            RunResult(
                returncode=2,
                stdout="",
                stderr="error: Failed to parse `uv.lock`",
                error="command exited 2",
            ),
        ]
    )

    outcome = _heal(tmp_path, run_command=runner)

    assert outcome.detected_via is None
    assert outcome.synced is False
    assert outcome.detail is not None
    assert "Failed to parse" in outcome.detail
    # No sync was attempted -- the probe is the only call.
    assert [c[0] for c in calls] == [["uv", "sync", "--locked", "--check", "--inexact"]]
    events = query_events(state_path, kind="self_deploy_boot_sync")
    assert len(events) == 1
    assert events[0]["payload"]["synced"] is False
    assert "Failed to parse" in events[0]["payload"]["error"]


def test_boot_repair_probe_timeout_is_not_drift(tmp_path: Path) -> None:
    """A probe that times out (or never spawned -- ``returncode=None`` in
    both shapes) is inconclusive, not drift: no sync, a detail report."""
    state_root = _state_root(tmp_path)
    (tmp_path / "uv.lock").write_bytes(b"lock\n")
    state_path = layout.state_file_path(state_root)

    runner, calls = _make_fake_runner(
        [
            RunResult(
                returncode=None,
                stdout="",
                stderr="",
                timed_out=True,
                error="command timed out after 30s",
            ),
        ]
    )

    outcome = _heal(tmp_path, run_command=runner)

    assert outcome.detected_via is None
    assert outcome.synced is False
    assert outcome.detail is not None
    assert "timed out" in outcome.detail
    assert [c[0] for c in calls] == [["uv", "sync", "--locked", "--check", "--inexact"]]
    events = query_events(state_path, kind="self_deploy_boot_sync")
    assert len(events) == 1
    assert "timed out" in events[0]["payload"]["error"]


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


def _write_scratch_wheel(wheel_path: Path) -> None:
    """Build a minimal valid ``localdep-0.1`` wheel -- a real locked package
    the test can remove to create genuine install drift without any network.
    """
    import base64
    import hashlib
    import zipfile

    records: list[str] = []

    def add(zf: zipfile.ZipFile, arcname: str, data: bytes) -> None:
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        records.append(f"{arcname},sha256={digest},{len(data)}")
        zf.writestr(arcname, data)

    with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as zf:
        add(zf, "localdep/__init__.py", b'"""local dep"""\n')
        add(
            zf,
            "localdep-0.1.dist-info/METADATA",
            b"Metadata-Version: 2.1\nName: localdep\nVersion: 0.1\n",
        )
        add(
            zf,
            "localdep-0.1.dist-info/WHEEL",
            b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        records.append("localdep-0.1.dist-info/RECORD,,")
        zf.writestr("localdep-0.1.dist-info/RECORD", "\n".join(records) + "\n")


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv binary not available")
def test_boot_repair_real_uv_preserves_installed_extras(tmp_path: Path) -> None:
    """Positive control against REAL uv on an extras-bearing scratch venv.

    The documented dev environment is ``uv sync --all-extras``, so its venv
    carries packages (pytest/ruff/pluggy) the locked requirement set does
    not. Under exact mode, ``uv sync --check`` reports those as drift and a
    repair sync would uninstall them every idle boot; under ``--inexact``
    the extra is ignored. The control is the SAME scratch venv probed both
    ways: exact-mode exit 1 proves uv genuinely flags the extra, and the
    inexact heal must then be a clean no-op that leaves it installed. A
    vendored-wheel dependency gives the project one real locked package, so
    the second leg can prove removing it is still detected and repaired --
    without pruning the extra.
    """
    scratch = tmp_path / "uvproj"
    (scratch / "vendor").mkdir(parents=True)
    _write_scratch_wheel(scratch / "vendor" / "localdep-0.1-py3-none-any.whl")
    (scratch / "pyproject.toml").write_text(
        "[project]\n"
        'name = "boot-probe-test"\n'
        'version = "0.0.1"\n'
        'requires-python = ">=3.10"\n'
        'dependencies = ["localdep"]\n'
        "\n"
        "[tool.uv.sources]\n"
        'localdep = { path = "vendor/localdep-0.1-py3-none-any.whl" }\n',
        encoding="utf-8",
    )
    state_root = tmp_path / "state"
    state_root.mkdir()

    # A real, working locked venv (the only dep is a vendored wheel: no network).
    lock = run_captured(["uv", "lock"], cwd=scratch, timeout_seconds=60)
    if not lock.ok:
        pytest.skip(f"uv lock unavailable in this environment: {lock.stderr or lock.error}")
    sync = run_captured(["uv", "sync", "--locked"], cwd=scratch, timeout_seconds=120)
    if not sync.ok or not (scratch / ".venv").exists():
        pytest.skip(f"uv sync unavailable in this environment: {sync.stderr or sync.error}")

    venv = scratch / ".venv"
    candidates = [venv / "Lib" / "site-packages", *sorted(venv.glob("lib/python*/site-packages"))]
    site_packages = next((p for p in candidates if p.is_dir()), None)
    assert site_packages is not None, f"no site-packages under {venv}"
    assert (site_packages / "localdep-0.1.dist-info").is_dir()

    # Plant a package that is not in the lock -- exactly the shape a dev
    # extra (pytest/ruff/pluggy) has relative to the locked requirement set.
    extra = site_packages / "extraneous_pkg-0.1.dist-info"
    extra.mkdir()
    (extra / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: extraneous-pkg\nVersion: 0.1\n", encoding="utf-8"
    )
    (extra / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        encoding="utf-8",
    )
    (extra / "RECORD").write_text("", encoding="utf-8")

    # Positive control: the pre-fix exact-mode probe DOES flag the extra --
    # proving the inexact probe's clean exit below is meaningful, not vacuous.
    exact_probe = run_captured(
        ["uv", "sync", "--locked", "--check"], cwd=scratch, timeout_seconds=60
    )
    assert exact_probe.returncode == 1, (
        f"exact-mode --check should flag the extraneous package: {exact_probe}"
    )

    outcome = heal_pending_sync_at_boot(
        scratch,
        state_root=state_root,
        live_count=lambda: 0,
        run_command=run_captured,
    )

    # The extra is a dev extra, not drift: no detection, no sync, and the
    # dist-info is still on disk (a repair sync would have pruned it).
    assert outcome == BootSyncRepair(detected_via=None, synced=False, deferred=False, detail=None)
    assert extra.is_dir()
    assert query_events(layout.state_file_path(state_root), kind="self_deploy_boot_sync") == []

    # Second leg: removing a LOCKED package is genuine drift the inexact
    # probe still catches and repairs -- the fix narrows what counts as
    # drift, it does not disable detection. The repair sync restores the
    # locked package AND leaves the extraneous dev-extra in place.
    shutil.rmtree(site_packages / "localdep-0.1.dist-info")
    shutil.rmtree(site_packages / "localdep")
    outcome = heal_pending_sync_at_boot(
        scratch,
        state_root=state_root,
        live_count=lambda: 0,
        run_command=run_captured,
    )
    assert outcome.detected_via == "probe"
    assert outcome.synced is True
    assert (site_packages / "localdep-0.1.dist-info").is_dir()
    assert extra.is_dir()
