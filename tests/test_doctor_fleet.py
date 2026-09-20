"""Fleet-supervisor and fleet-dir virtualization checks for ``run_doctor``.

Split out of ``tests/test_doctor.py`` (issue #1563, Track 1 shoulder) --
bodies are verbatim relocations; shared helpers live in
``tests/_doctor_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from charlie_work.config import AutoMergeConfig
from charlie_work.doctor import run_doctor
from charlie_work.paths import runtime_paths
from _doctor_fixtures import (
    FakeDoctorGitHub,
    _config,
    _make_fleet_json,
    _patch_resolve_to_diverge,
)


def test_doctor_skips_fleet_supervisor_check_when_fleet_not_configured(
    tmp_path: Path,
) -> None:
    """No fleet supervisor warning when fleet.json does not exist."""
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    names = {check.name for check in checks}
    assert "fleet supervisor" not in names
    assert ok is True


def test_doctor_warns_when_fleet_configured_but_not_supervised(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """fleet.json has repos but no supervisor lock held → warning."""
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    gh = FakeDoctorGitHub(labels=config.labels.all)
    _make_fleet_json(tmp_path, paths.root)

    # All locks are acquirable, so no supervisor is running.
    monkeypatch.setattr(
        "charlie_work.doctor.try_acquire_supervisor_lock",
        lambda _path: MagicMock(),
    )

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    fleet_check = by_name["fleet supervisor"]
    assert fleet_check.ok is False
    assert fleet_check.severity == "warning"
    assert "run `charlie fleet supervise`" in fleet_check.detail
    assert ok is True


def test_doctor_passes_when_fleet_supervisor_lock_held(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """fleet-supervisor.lock held means the fleet is being driven."""
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    gh = FakeDoctorGitHub(labels=config.labels.all)
    _make_fleet_json(tmp_path, paths.root)

    # Touch the fleet supervisor lock file so the existence check triggers the probe.
    fleet_lock_path = tmp_path / "fleet" / "fleet-supervisor.lock"
    fleet_lock_path.parent.mkdir(parents=True, exist_ok=True)
    fleet_lock_path.write_text("", encoding="utf-8")

    def _fake_lock(path: Path) -> MagicMock | None:
        if path.name == "fleet-supervisor.lock":
            return None  # held
        return MagicMock()  # repo locks free

    monkeypatch.setattr(
        "charlie_work.doctor.try_acquire_supervisor_lock",
        _fake_lock,
    )

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    fleet_check = by_name["fleet supervisor"]
    assert fleet_check.ok is True
    assert "fleet supervisor appears to be running" in fleet_check.detail
    assert ok is True


def test_doctor_fleet_supervisor_per_repo_aware(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A single held repo lock does not hide unsupervised repos in the fleet."""
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)
    fleet_json = fleet_dir_path / "fleet.json"
    state_dir1 = tmp_path / "state1"
    state_dir1.mkdir(parents=True, exist_ok=True)
    state_dir2 = tmp_path / "state2"
    state_dir2.mkdir(parents=True, exist_ok=True)
    fleet_json.write_text(
        json.dumps(
            {
                "version": 1,
                "repos": {
                    "owner/repo1": {
                        "repo_root": str(tmp_path),
                        "state_dir": str(state_dir1),
                    },
                    "owner/repo2": {
                        "repo_root": str(tmp_path),
                        "state_dir": str(state_dir2),
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    # Create the per-repo lock files so the probe attempts to acquire them.
    (state_dir1 / "supervisor.lock").write_text("", encoding="utf-8")
    (state_dir2 / "supervisor.lock").write_text("", encoding="utf-8")

    def _fake_lock(path: Path) -> MagicMock | None:
        # Only repo1 has a live per-repo supervisor.
        if "state1" in str(path):
            return None
        return MagicMock()

    monkeypatch.setattr(
        "charlie_work.doctor.try_acquire_supervisor_lock",
        _fake_lock,
    )

    ok, checks = run_doctor(
        tmp_path,
        paths,
        config,
        tmp_path / "c.yaml",
        gh,
        fleet_dir_override=str(fleet_dir_path),
    )

    by_name = {check.name: check for check in checks}
    fleet_check = by_name["fleet supervisor"]
    assert fleet_check.ok is False
    assert fleet_check.severity == "warning"
    assert "owner/repo1" in fleet_check.detail
    assert "owner/repo2" in fleet_check.detail
    assert ok is True


def test_doctor_warns_when_fleet_dir_is_virtualized(tmp_path: Path, monkeypatch: Any) -> None:
    """A literal/resolved divergence fires a warning naming both paths."""
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)
    redirected = tmp_path / "Packages" / "app" / "LocalCache" / "Local" / "charlie-work"
    _patch_resolve_to_diverge(monkeypatch, fleet_dir_path, redirected)

    ok, checks = run_doctor(
        tmp_path, paths, config, tmp_path / "c.yaml", gh, fleet_dir_override=str(fleet_dir_path)
    )

    by_name = {check.name: check for check in checks}
    virt = by_name["fleet dir virtualization"]
    assert virt.ok is False
    assert virt.severity == "warning"
    # Both paths must be named so the operator can see where it landed.
    assert str(fleet_dir_path) in virt.detail
    assert str(redirected) in virt.detail
    # Reference the #590 failure and state the write-forks-a-copy consequence.
    assert "#590" in virt.detail
    assert "private copy" in virt.detail
    # Warning-only: a virtualized fleet dir is not fatal for an interactive human.
    assert ok is True


def test_doctor_is_silent_when_fleet_dir_is_not_virtualized(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Equal literal and resolved paths produce no virtualization check."""
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)
    # No divergence injection: resolve() returns the literal path unchanged.
    ok, checks = run_doctor(
        tmp_path, paths, config, tmp_path / "c.yaml", gh, fleet_dir_override=str(fleet_dir_path)
    )

    names = {check.name for check in checks}
    assert "fleet dir virtualization" not in names
    assert ok is True


def test_fleet_dir_virtualization_probe_is_repo_agnostic(tmp_path: Path, monkeypatch: Any) -> None:
    """The probe fires for any fleet_dir override, not just charlie-work's layout.

    The redirection is a per-process property of the host, so a repo whose
    fleet dir is unrelated to charlie-work's own layout must still be flagged.
    """
    from charlie_work.fleet_paths import fleet_dir_virtualization

    # An arbitrary override path with no charlie-work-specific component.
    literal = tmp_path / "some-other-repo" / "fleet-state"
    redirected = tmp_path / "Packages" / "other-app" / "LocalCache" / "some-other-repo"
    _patch_resolve_to_diverge(monkeypatch, literal, redirected)

    diverged = fleet_dir_virtualization(override=str(literal))
    assert diverged is not None
    assert diverged[0] == literal
    assert diverged[1] == redirected


def test_fleet_dir_virtualization_returns_none_when_equal(tmp_path: Path) -> None:
    """No divergence -> None (the probe must stay silent)."""
    from charlie_work.fleet_paths import fleet_dir_virtualization

    literal = tmp_path / "fleet"
    literal.mkdir(parents=True, exist_ok=True)
    assert fleet_dir_virtualization(override=str(literal)) is None


def test_run_doctor_wires_the_virtualization_probe(tmp_path: Path, monkeypatch: Any) -> None:
    """Pin the wiring, not just the probe body.

    Deleting the ``_check_fleet_dir_virtualization`` call in ``run_doctor``
    would leave the probe-body tests green while the probe silently stopped
    running for operators.
    """
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)
    redirected = tmp_path / "Packages" / "app" / "LocalCache" / "Local" / "charlie-work"
    _patch_resolve_to_diverge(monkeypatch, fleet_dir_path, redirected)

    _, checks = run_doctor(
        tmp_path, paths, config, tmp_path / "c.yaml", gh, fleet_dir_override=str(fleet_dir_path)
    )

    virt_checks = [c for c in checks if c.name == "fleet dir virtualization"]
    assert len(virt_checks) == 1
    assert virt_checks[0].severity == "warning"
