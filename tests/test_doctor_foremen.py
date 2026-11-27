"""Foreman-store checks for ``run_doctor`` (issue #3123).

``_check_foremen`` decodes ``<fleet_dir>/foremen/<name>/`` by file shape —
ADR-0010 forbids importing ``charlie_work.foreman`` from ``doctor.py`` — and
reports one ``foreman <name>`` check per foreman: charter parses, names its
directory, and points its single ``repos`` entry at a fleet-registry key;
state.json parses with a known ``state``; items.json parses with decodable
records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from charlie_work.config import AutoMergeConfig
from charlie_work.doctor import run_doctor
from charlie_work.paths import runtime_paths

from _doctor_fixtures import FakeDoctorGitHub, _config, _make_fleet_json


def _run(tmp_path: Path, make_foremen: Any = None) -> tuple[bool, dict[str, Any]]:
    """Build the doctor fixture fleet, optionally seed foremen, run, return checks by name."""
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)
    fleet_dir = _make_fleet_json(tmp_path, paths.root).parent
    if make_foremen is not None:
        make_foremen(fleet_dir)
    _ok, checks = run_doctor(
        tmp_path,
        paths,
        config,
        tmp_path / "c.yaml",
        gh,
        fleet_dir_override=str(fleet_dir),
    )
    return _ok, {check.name: check for check in checks}


def _foreman(
    fleet_dir: Path,
    name: str = "alpha",
    *,
    charter: str | None = "name: alpha\nrepos:\n  - owner/repo\n",
    state: dict | str | None = None,
    items: Any = "absent",
) -> Path:
    """Create ``foremen/<name>/``; ``items`` controls items.json ("absent" skips it)."""
    fdir = fleet_dir / "foremen" / name
    fdir.mkdir(parents=True, exist_ok=True)
    if charter is not None:
        (fdir / "charter.yaml").write_text(charter, encoding="utf-8")
    if state is not None:
        raw = state if isinstance(state, str) else json.dumps(state)
        (fdir / "state.json").write_text(raw, encoding="utf-8")
    if items != "absent":
        (fdir / "items.json").write_text(json.dumps(items), encoding="utf-8")
    return fdir


def test_doctor_foremen_absent_when_no_foremen(tmp_path: Path) -> None:
    _ok, checks = _run(tmp_path)
    assert not [name for name in checks if name.startswith("foreman ")]


def test_doctor_foremen_ignores_non_foreman_dirs(tmp_path: Path) -> None:
    def seed(fleet_dir: Path) -> None:
        for stray in ("_shared", "UPPER"):
            (fleet_dir / "foremen" / stray).mkdir(parents=True)

    _ok, checks = _run(tmp_path, seed)
    assert not [name for name in checks if name.startswith("foreman ")]


def test_doctor_foremen_healthy(tmp_path: Path) -> None:
    def seed(fleet_dir: Path) -> None:
        _foreman(
            fleet_dir,
            state={"state": "active"},
            items={"items": [{"id": "i-1", "kind": "decision"}]},
        )

    _ok, checks = _run(tmp_path, seed)
    assert checks["foreman alpha"].ok is True


def test_doctor_foremen_items_absent_is_fine(tmp_path: Path) -> None:
    def seed(fleet_dir: Path) -> None:
        _foreman(fleet_dir, state={"state": "paused"})

    _ok, checks = _run(tmp_path, seed)
    assert checks["foreman alpha"].ok is True


def test_doctor_foremen_charter_name_mismatch(tmp_path: Path) -> None:
    def seed(fleet_dir: Path) -> None:
        _foreman(
            fleet_dir,
            charter="name: beta\nrepos:\n  - owner/repo\n",
            state={"state": "active"},
        )

    _ok, checks = _run(tmp_path, seed)
    check = checks["foreman alpha"]
    assert check.ok is False
    assert "does not match directory" in check.detail


def test_doctor_foremen_repos_not_registered(tmp_path: Path) -> None:
    def seed(fleet_dir: Path) -> None:
        _foreman(
            fleet_dir,
            charter="name: alpha\nrepos:\n  - ghost/repo\n",
            state={"state": "active"},
        )

    _ok, checks = _run(tmp_path, seed)
    check = checks["foreman alpha"]
    assert check.ok is False
    assert "registered repo" in check.detail


def test_doctor_foremen_repos_arity(tmp_path: Path) -> None:
    def seed(fleet_dir: Path) -> None:
        _foreman(
            fleet_dir,
            charter="name: alpha\nrepos:\n  - owner/repo\n  - owner/other\n",
            state={"state": "active"},
        )

    _ok, checks = _run(tmp_path, seed)
    check = checks["foreman alpha"]
    assert check.ok is False
    assert "exactly one" in check.detail


def test_doctor_foremen_corrupt_state(tmp_path: Path) -> None:
    def seed(fleet_dir: Path) -> None:
        _foreman(fleet_dir, state="{not json")

    _ok, checks = _run(tmp_path, seed)
    check = checks["foreman alpha"]
    assert check.ok is False
    assert "state.json" in check.detail


def test_doctor_foremen_unknown_state_value(tmp_path: Path) -> None:
    def seed(fleet_dir: Path) -> None:
        _foreman(fleet_dir, state={"state": "bogus"})

    _ok, checks = _run(tmp_path, seed)
    check = checks["foreman alpha"]
    assert check.ok is False
    assert "not one of" in check.detail


def test_doctor_foremen_bad_item_records(tmp_path: Path) -> None:
    def seed(fleet_dir: Path) -> None:
        _foreman(
            fleet_dir,
            state={"state": "active"},
            items={"items": [{"id": "i-1", "kind": "nonsense"}]},
        )

    _ok, checks = _run(tmp_path, seed)
    check = checks["foreman alpha"]
    assert check.ok is False
    assert "failed to decode" in check.detail
