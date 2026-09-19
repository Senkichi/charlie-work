"""In-progress staleness tests for ``scripts/heartbeat_check.py``.

Split out of ``tests/test_heartbeat_check.py`` (issue #1556, Track-1):
``check_in_progress_staleness`` plus the worktree-mtime liveness
signal (issue #1379): fresh/stale/missing worktrees, ``.git``
exclusion, reparse-point mtime exclusion, and the worktrees-dir
override.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from _heartbeat_check_fixtures import (
    _iso,
    _load_heartbeat_check,
    _make_repo,
)


@pytest.fixture(scope="module")
def hb() -> ModuleType:
    return _load_heartbeat_check()


def test_check_in_progress_staleness_anomaly_when_unchanged_across_beats(
    hb: ModuleType, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    updated = _iso(5)
    prev = {"in_progress": {"99": updated}}
    new: dict[str, Any] = {}
    report = hb.Report()
    hb.check_in_progress_staleness(report, repo, [(99, updated)], prev, new, skip_delta=False)
    assert report.anomaly
    assert "99" in report.lines[0]


# --------------------------------------------------------------------------
# Issue #1379: worktree mtime as a second liveness signal for in-progress-stale
# --------------------------------------------------------------------------


_BRANCH_99 = "agent/issue-99-some-fix-title-here"


def _set_file_mtime(path: Path, dt: datetime) -> None:
    """Set a file's mtime to ``dt`` (derived-relative, never hardcoded)."""
    ts = dt.timestamp()
    os.utime(path, (ts, ts))


def _write_state_issues(repo: Any, issues: dict[str, Any]) -> None:
    """Write a state.json with the given ``issues`` map under repo.state_dir."""
    state_json = repo.state_dir / "state.json"
    state_json.parent.mkdir(parents=True, exist_ok=True)
    state_json.write_text(json.dumps({"issues": issues}), encoding="utf-8")


def _make_worktree_with_file(
    hb: ModuleType, repo: Any, branch: str, file_age_min: float, now: datetime
) -> Path:
    """Create the worktree dir for ``branch`` with one file aged ``file_age_min``.

    The worktree path is derived via ``hb._worktree_path_for_branch`` (the same
    slugify the production check uses), so a slug mismatch between the check
    and the orchestrator's worktree creation surfaces here as a missing-dir
    failure rather than a silent false ANOMALY.
    """
    wt = hb._worktree_path_for_branch(repo, branch)
    wt.mkdir(parents=True, exist_ok=True)
    src = wt / "src"
    src.mkdir(exist_ok=True)
    f = src / "main.py"
    f.write_text("# worker activity\n", encoding="utf-8")
    _set_file_mtime(f, now - timedelta(minutes=file_age_min))
    return wt


def test_in_progress_stale_ok_when_worktree_fresh(hb: ModuleType, tmp_path: Path) -> None:
    """AC1: zero recent events but worktree file modified inside the window -> OK."""
    repo = _make_repo(hb, tmp_path)
    now = datetime.now(timezone.utc)
    updated = _iso(5, base=now)
    prev = {"in_progress": {"99": updated}}
    new: dict[str, Any] = {}
    _write_state_issues(repo, {"99": {"branch_name": _BRANCH_99}})
    # Worktree file 5 minutes old -- well inside the 30-minute window.
    _make_worktree_with_file(hb, repo, _BRANCH_99, file_age_min=5, now=now)
    report = hb.Report()
    hb.check_in_progress_staleness(
        report, repo, [(99, updated)], prev, new, skip_delta=False, now=now
    )
    assert not report.anomaly
    line = report.lines[0]
    assert line.startswith("OK ")
    assert "worktree-fresh=1" in line
    assert "worktree mtime" in line
    assert "99" in line


def test_in_progress_stale_anomaly_when_worktree_stale(hb: ModuleType, tmp_path: Path) -> None:
    """AC2: zero recent events AND no worktree file newer than window -> ANOMALY with both ages."""
    repo = _make_repo(hb, tmp_path)
    now = datetime.now(timezone.utc)
    updated = _iso(5, base=now)
    prev = {"in_progress": {"99": updated}}
    new: dict[str, Any] = {}
    _write_state_issues(repo, {"99": {"branch_name": _BRANCH_99}})
    # Worktree file 60 minutes old -- outside the 30-minute window (dead worker).
    _make_worktree_with_file(hb, repo, _BRANCH_99, file_age_min=60, now=now)
    report = hb.Report()
    hb.check_in_progress_staleness(
        report, repo, [(99, updated)], prev, new, skip_delta=False, now=now
    )
    assert report.anomaly
    line = report.lines[0]
    assert line.startswith("ANOMALY ")
    assert "no events across 2 beats" in line
    assert "worktree idle" in line
    assert "99" in line


def test_in_progress_stale_anomaly_when_worktree_missing(hb: ModuleType, tmp_path: Path) -> None:
    """AC3: worktree dir does not exist -> events-only ANOMALY with 'no worktree found'."""
    repo = _make_repo(hb, tmp_path)
    now = datetime.now(timezone.utc)
    updated = _iso(5, base=now)
    prev = {"in_progress": {"99": updated}}
    new: dict[str, Any] = {}
    _write_state_issues(repo, {"99": {"branch_name": _BRANCH_99}})
    # Deliberately do NOT create the worktree dir.
    report = hb.Report()
    hb.check_in_progress_staleness(
        report, repo, [(99, updated)], prev, new, skip_delta=False, now=now
    )
    assert report.anomaly
    line = report.lines[0]
    assert line.startswith("ANOMALY ")
    assert "no worktree found" in line
    assert "99" in line


def test_in_progress_stale_ok_mixed_fresh_and_stale(hb: ModuleType, tmp_path: Path) -> None:
    """Mixed: one worktree-fresh (OK) and one worktree-stale (ANOMALY) -> ANOMALY wins."""
    repo = _make_repo(hb, tmp_path)
    now = datetime.now(timezone.utc)
    updated = _iso(5, base=now)
    prev = {"in_progress": {"99": updated, "100": updated}}
    new: dict[str, Any] = {}
    branch_100 = "agent/issue-100-another-fix-title"
    _write_state_issues(
        repo,
        {"99": {"branch_name": _BRANCH_99}, "100": {"branch_name": branch_100}},
    )
    # #99 is alive (5m), #100 is dead (60m).
    _make_worktree_with_file(hb, repo, _BRANCH_99, file_age_min=5, now=now)
    _make_worktree_with_file(hb, repo, branch_100, file_age_min=60, now=now)
    report = hb.Report()
    hb.check_in_progress_staleness(
        report, repo, [(99, updated), (100, updated)], prev, new, skip_delta=False, now=now
    )
    assert report.anomaly
    line = report.lines[0]
    assert "#100" in line
    assert "worktree idle" in line
    # #99 must NOT appear in the ANOMALY detail (it was exonerated by worktree mtime).
    assert "#99" not in line


def test_in_progress_stale_excludes_git_dir(hb: ModuleType, tmp_path: Path) -> None:
    """``.git/`` file activity must not count as worker activity (issue #1379)."""
    repo = _make_repo(hb, tmp_path)
    now = datetime.now(timezone.utc)
    updated = _iso(5, base=now)
    prev = {"in_progress": {"99": updated}}
    new: dict[str, Any] = {}
    _write_state_issues(repo, {"99": {"branch_name": _BRANCH_99}})
    wt = _make_worktree_with_file(hb, repo, _BRANCH_99, file_age_min=60, now=now)
    # Plant a fresh file under .git/ (background git op) -- must NOT exonerate.
    git_dir = wt / ".git"
    git_dir.mkdir(exist_ok=True)
    fresh_git_file = git_dir / "HEAD"
    fresh_git_file.write_text("ref: refs/heads/main\n", encoding="utf-8")
    _set_file_mtime(fresh_git_file, now - timedelta(minutes=1))
    report = hb.Report()
    hb.check_in_progress_staleness(
        report, repo, [(99, updated)], prev, new, skip_delta=False, now=now
    )
    assert report.anomaly
    assert "worktree idle" in report.lines[0]


def _make_junction(junction: Path, target: Path) -> None:
    """Create a real Windows directory junction ``junction`` -> ``target``.

    ``os.symlink`` creates a symlink, not a junction -- and ``os.path.islink``
    returns ``False`` for a junction, which is exactly the gap issue #1379's
    review found. A real junction (via ``mklink /J``) reproduces the
    production ``.venv`` junction shape this fleet uses, so the test exercises
    the actual failure path rather than a symlink stand-in that ``islink``
    would already catch.
    """
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(target.resolve())],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


_windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="junctions are a Windows reparse-point type"
)


@_windows_only
def test_in_progress_stale_excludes_junction_mtime(hb: ModuleType, tmp_path: Path) -> None:
    """A ``.venv`` junction's fresh file must NOT exonerate a dead worker.

    Issue #1379 review: ``os.path.islink`` does not detect Windows junctions
    and ``os.walk(followlinks=False)`` recurses straight through one, so a
    shared ``.venv`` junction's unrelated mtimes could mask a genuinely dead
    worker. This plants a real junction (``mklink /J``, not ``os.symlink``)
    with a fresh file inside it and asserts the dead worktree still flags.
    """
    repo = _make_repo(hb, tmp_path)
    now = datetime.now(timezone.utc)
    updated = _iso(5, base=now)
    prev = {"in_progress": {"99": updated}}
    new: dict[str, Any] = {}
    _write_state_issues(repo, {"99": {"branch_name": _BRANCH_99}})
    # Worktree itself is dead: only a 60m-old file (outside the 30m window).
    wt = _make_worktree_with_file(hb, repo, _BRANCH_99, file_age_min=60, now=now)
    # Plant a .venv junction pointing at a shared target with a FRESH file
    # (1m old -- well inside the window). Without the reparse-point fix this
    # fresh file would be walked and exonerate the dead worker (false OK).
    shared_venv = tmp_path / "shared-venv"
    shared_venv.mkdir()
    fresh_venv_file = shared_venv / "pyvenv.cfg"
    fresh_venv_file.write_text("home = shared\n", encoding="utf-8")
    _set_file_mtime(fresh_venv_file, now - timedelta(minutes=1))
    _make_junction(wt / ".venv", shared_venv)
    # Sanity: the junction really is a junction (islink returns False), so
    # this test genuinely exercises the islink-blind path, not a symlink.
    assert not os.path.islink(wt / ".venv")
    assert os.path.isdir(wt / ".venv")
    report = hb.Report()
    hb.check_in_progress_staleness(
        report, repo, [(99, updated)], prev, new, skip_delta=False, now=now
    )
    assert report.anomaly
    assert "worktree idle" in report.lines[0]


def test_in_progress_stale_honors_worktrees_dir_override(hb: ModuleType, tmp_path: Path) -> None:
    """``claude_code.worktrees_dir`` override must be honored (issue #1379 review).

    The worktrees root defaults to ``<state_dir>/worktrees``; a repo that sets
    ``claude_code.worktrees_dir`` places worktrees elsewhere. The check must
    look at the configured location, not the default -- otherwise it
    under-reports (a fresh worktree at the override location reads as missing
    -> false ANOMALY). Mirrors ``charlie_work.paths.resolved_layout``.
    """
    repo = _make_repo(hb, tmp_path)
    now = datetime.now(timezone.utc)
    updated = _iso(5, base=now)
    prev = {"in_progress": {"99": updated}}
    new: dict[str, Any] = {}
    _write_state_issues(repo, {"99": {"branch_name": _BRANCH_99}})
    # Configure an explicit (relative) worktrees_dir override.
    override_root = tmp_path / "custom-worktrees"
    repo.config_path.write_text(
        "claude_code:\n  worktrees_dir: custom-worktrees\n", encoding="utf-8"
    )
    # Build the worktree at the OVERRIDE location (not the default
    # state_dir/worktrees), with a fresh file inside the 30m window.
    wt = hb._worktree_path_for_branch(repo, _BRANCH_99, worktrees_dir=override_root)
    wt.mkdir(parents=True, exist_ok=True)
    src = wt / "src"
    src.mkdir(exist_ok=True)
    f = src / "main.py"
    f.write_text("# worker activity\n", encoding="utf-8")
    _set_file_mtime(f, now - timedelta(minutes=5))
    # The default location must NOT exist, so a check that ignores the
    # override would read "no worktree found" (false ANOMALY).
    default_wt = repo.state_dir / "worktrees" / hb._slugify_branch(_BRANCH_99)
    assert not default_wt.exists()
    report = hb.Report()
    hb.check_in_progress_staleness(
        report, repo, [(99, updated)], prev, new, skip_delta=False, now=now
    )
    assert not report.anomaly
    line = report.lines[0]
    assert line.startswith("OK ")
    assert "worktree-fresh=1" in line
