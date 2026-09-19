"""``check_disk_space`` tests for ``scripts/heartbeat_check.py``.

Split out of ``tests/test_heartbeat_check.py`` (issue #1556, Track-1):
free-above/warn-below/anomaly-below thresholds, drive-anchor
dedup, state-dir-derived volumes, stat failure, and the no-repos
fleet-dir fallback (issue #1359).
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from _heartbeat_check_fixtures import (
    _load_heartbeat_check,
    _make_repo,
)


@pytest.fixture(scope="module")
def hb() -> ModuleType:
    return _load_heartbeat_check()


# ---------------------------------------------------------------------------
# Disk-free check (issue #1359)
# ---------------------------------------------------------------------------


class _FakeUsage:
    """Stand-in for the namedtuple returned by shutil.disk_usage."""

    def __init__(self, total: int, free: int) -> None:
        self.total = total
        self.free = free
        self.used = total - free


def _set_fleet_dir_to(hb: ModuleType, monkeypatch: Any, path: Path) -> None:
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(path))


def _disk_usage_stub(total: int, free: int):
    return lambda path: _FakeUsage(total, free)


def test_check_disk_space_ok_when_free_above_thresholds(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    _set_fleet_dir_to(hb, monkeypatch, tmp_path)
    # 500 GB free of 1000 GB (50%) -- well above both thresholds.
    monkeypatch.setattr(hb.shutil, "disk_usage", _disk_usage_stub(1000 * 1024**3, 500 * 1024**3))
    repo = _make_repo(hb, tmp_path)
    report = hb.Report()
    hb.check_disk_space(report, [repo])
    assert not report.anomaly
    line = report.lines[0]
    assert line.startswith("OK disk-space ")
    assert "free=500.0GB" in line
    assert "(50.0%)" in line


def test_check_disk_space_warn_between_soft_and_hard(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    _set_fleet_dir_to(hb, monkeypatch, tmp_path)
    # 50 GB free of 1000 GB (5.0%): below the 100 GB soft threshold but above
    # the 20 GB hard threshold -> WARN, never an anomaly.
    monkeypatch.setattr(hb.shutil, "disk_usage", _disk_usage_stub(1000 * 1024**3, 50 * 1024**3))
    repo = _make_repo(hb, tmp_path)
    report = hb.Report()
    hb.check_disk_space(report, [repo])
    assert not report.anomaly
    line = report.lines[0]
    assert line.startswith("WARN disk-space ")
    assert "below soft threshold" in line


def test_check_disk_space_anomaly_below_hard_bytes(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    _set_fleet_dir_to(hb, monkeypatch, tmp_path)
    # 10 GB free of 1000 GB: below the 20 GB hard threshold -> ANOMALY.
    monkeypatch.setattr(hb.shutil, "disk_usage", _disk_usage_stub(1000 * 1024**3, 10 * 1024**3))
    repo = _make_repo(hb, tmp_path)
    report = hb.Report()
    hb.check_disk_space(report, [repo])
    assert report.anomaly
    line = report.lines[0]
    assert line.startswith("ANOMALY disk-space ")
    assert "below hard threshold" in line


def test_check_disk_space_anomaly_below_hard_ratio(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Free bytes above the 20 GB floor but free ratio below 1% still trips."""
    _set_fleet_dir_to(hb, monkeypatch, tmp_path)
    # 25 GB free of 3000 GB (~0.83%): 25 > 20 GB so the byte branch alone would
    # not fire, but 0.83% < 1% -> ANOMALY via the ratio branch.
    monkeypatch.setattr(hb.shutil, "disk_usage", _disk_usage_stub(3000 * 1024**3, 25 * 1024**3))
    repo = _make_repo(hb, tmp_path)
    report = hb.Report()
    hb.check_disk_space(report, [repo])
    assert report.anomaly
    assert "below hard threshold" in report.lines[0]


def test_check_disk_space_dedupes_by_drive_anchor(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Two repos whose state_dirs share a volume report one line, not two."""
    _set_fleet_dir_to(hb, monkeypatch, tmp_path)
    monkeypatch.setattr(hb.shutil, "disk_usage", _disk_usage_stub(1000 * 1024**3, 500 * 1024**3))
    repo_a = hb.RepoInfo(
        slug="a/repo",
        repo_root=tmp_path,
        state_dir=tmp_path / "state_a",
        config_path=Path(""),
    )
    repo_b = hb.RepoInfo(
        slug="b/repo",
        repo_root=tmp_path,
        state_dir=tmp_path / "state_b",
        config_path=Path(""),
    )
    report = hb.Report()
    hb.check_disk_space(report, [repo_a, repo_b])
    disk_lines = [ln for ln in report.lines if "disk-space" in ln]
    assert len(disk_lines) == 1


def test_check_disk_space_derives_volumes_from_state_dirs_not_hardcoded(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """The probed path must come from the registered state_dir, not a constant."""
    _set_fleet_dir_to(hb, monkeypatch, tmp_path)
    probed: list[str] = []
    state_dir = tmp_path / "registered-state"

    def _capture(path: str) -> _FakeUsage:
        probed.append(path)
        return _FakeUsage(1000 * 1024**3, 500 * 1024**3)

    monkeypatch.setattr(hb.shutil, "disk_usage", _capture)
    repo = hb.RepoInfo(
        slug="a/repo",
        repo_root=tmp_path,
        state_dir=state_dir,
        config_path=Path(""),
    )
    report = hb.Report()
    hb.check_disk_space(report, [repo])
    # The state_dir path (resolved) must be among the probed paths -- the
    # volume set is configuration-derived, not a hardcoded drive letter.
    assert any(str(state_dir.resolve()) == p or state_dir.resolve() == Path(p) for p in probed)
    assert not report.anomaly


def test_check_disk_space_anomaly_when_stat_fails(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    _set_fleet_dir_to(hb, monkeypatch, tmp_path)

    def _raise(path: str) -> _FakeUsage:
        raise OSError("no such volume")

    monkeypatch.setattr(hb.shutil, "disk_usage", _raise)
    repo = _make_repo(hb, tmp_path)
    report = hb.Report()
    hb.check_disk_space(report, [repo])
    assert report.anomaly
    assert "cannot stat volume" in report.lines[0]


def test_check_disk_space_no_repos_still_checks_fleet_dir(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """With zero registered repos the fleet-dir volume is still probed."""
    _set_fleet_dir_to(hb, monkeypatch, tmp_path)
    monkeypatch.setattr(hb.shutil, "disk_usage", _disk_usage_stub(1000 * 1024**3, 500 * 1024**3))
    report = hb.Report()
    hb.check_disk_space(report, [])
    assert not report.anomaly
    assert len(report.lines) == 1
    assert "disk-space" in report.lines[0]
