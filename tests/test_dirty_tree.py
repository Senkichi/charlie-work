"""Unit tests for :mod:`charlie_work.dirty_tree` -- the issue #729 clean-tree gate.

These tests exercise the real ``git status --porcelain`` parsing against a
genuine (``git init``-ed) temporary repository, not a stub. The gate's whole
purpose is to detect tracked-file divergence between the working tree and
HEAD, so the test must observe real git behavior -- a stub would only test the
stub.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from charlie_work.dirty_tree import check_working_tree_clean


def _git(repo: Path, *args: str) -> None:
    """Run a git command in *repo*, raising on failure."""
    subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A real git repo with one committed file, configured for deterministic tests."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "test")
    _git(repo, "config", "user.email", "test@example.test")
    (repo / "tracked.txt").write_text("initial\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "initial")
    return repo


def test_clean_tree_reports_no_dirty_paths(git_repo: Path) -> None:
    """A tree with no modifications since HEAD reports clean."""
    report = check_working_tree_clean(repo_root=git_repo)

    assert report.ok is True
    assert report.clean is True
    assert report.dirty_paths == ()


def test_modified_tracked_file_is_detected(git_repo: Path) -> None:
    """The exact #729 scenario: a tracked file modified in the working tree."""
    (git_repo / "tracked.txt").write_text("tampered\n", encoding="utf-8")

    report = check_working_tree_clean(repo_root=git_repo)

    assert report.ok is True
    assert report.clean is False
    assert "tracked.txt" in report.dirty_paths


def test_staged_new_file_is_detected(git_repo: Path) -> None:
    """A staged new file is tracked divergence from HEAD -- it has not been reviewed."""
    (git_repo / "new_guard.py").write_text("# unreviewed\n", encoding="utf-8")
    _git(git_repo, "add", "new_guard.py")

    report = check_working_tree_clean(repo_root=git_repo)

    assert report.ok is True
    assert report.clean is False
    assert "new_guard.py" in report.dirty_paths


def test_untracked_file_does_not_trip_gate(git_repo: Path) -> None:
    """Untracked files are excluded -- the gate is about modifications to the
    reviewed tree, not the presence of scratch work.
    """
    (git_repo / "scratch.txt").write_text("scratch\n", encoding="utf-8")

    report = check_working_tree_clean(repo_root=git_repo)

    assert report.ok is True
    assert report.clean is True
    assert report.dirty_paths == ()


def test_deleted_tracked_file_is_detected(git_repo: Path) -> None:
    """A deleted tracked file is divergence from HEAD."""
    (git_repo / "tracked.txt").unlink()

    report = check_working_tree_clean(repo_root=git_repo)

    assert report.ok is True
    assert report.clean is False
    assert "tracked.txt" in report.dirty_paths


def test_probe_failure_on_non_repo_is_fail_closed(tmp_path: Path) -> None:
    """A directory that is not a git repo cannot be checked -- fail closed,
    never silently report clean.
    """
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()

    report = check_working_tree_clean(repo_root=not_a_repo)

    assert report.ok is False
    assert report.clean is False
    assert report.error is not None
    assert "could not check working tree cleanliness" in report.error
