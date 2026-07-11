"""Unit tests for attempt_refs.py (issue #261).

Covers the plumbing in isolation (snapshot creation, attempt numbering,
listing, ahead-of-main computation, graceful failure) — the end-to-end
redispatch integration (attempt ref survives a real ``create_worktree``
branch reset) lives in test_worktree.py alongside the other recovery-path
tests it shares fixtures with.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from charlie_work.attempt_refs import (
    ATTEMPT_REF_PREFIX,
    AttemptSnapshot,
    list_attempt_refs,
    snapshot_attempt_ref,
)


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_repo(repo_root: Path) -> None:
    repo_root.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "init", "--initial-branch=main")
    _git(repo_root, "config", "user.email", "test@example.test")
    _git(repo_root, "config", "user.name", "Test User")
    (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
    _git(repo_root, "add", "README.md")
    _git(repo_root, "commit", "-m", "initial commit")


def _commit(repo_root: Path, name: str, content: str) -> str:
    (repo_root / name).write_text(content, encoding="utf-8")
    _git(repo_root, "add", name)
    _git(repo_root, "commit", "-m", f"add {name}")
    return _git(repo_root, "rev-parse", "HEAD").stdout.strip()


def test_snapshot_attempt_ref_preserves_tip(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    _git(repo_root, "checkout", "-b", "agent/issue-1")
    tip = _commit(repo_root, "work.txt", "some work\n")

    snapshot = snapshot_attempt_ref(repo_root, "agent/issue-1", issue_number=1, base_ref="main")

    assert snapshot.error is None
    assert snapshot.old_tip == tip
    assert snapshot.ref_name == f"{ATTEMPT_REF_PREFIX}/issue-1/attempt-1"
    assert snapshot.ahead_of_main_count == 1

    resolved = _git(repo_root, "rev-parse", snapshot.ref_name).stdout.strip()
    assert resolved == tip


def test_snapshot_attempt_ref_no_commits_returns_all_none(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    # Branch that does not resolve to a commit (never created).
    snapshot = snapshot_attempt_ref(repo_root, "does-not-exist", issue_number=1, base_ref="main")

    assert snapshot.ref_name is None
    assert snapshot.old_tip is None
    assert snapshot.ahead_of_main_count is None
    assert snapshot.error is None  # not an error — simply nothing to preserve


def test_snapshot_attempt_number_increments_across_calls(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    _git(repo_root, "checkout", "-b", "agent/issue-7")
    tip1 = _commit(repo_root, "attempt1.txt", "attempt 1\n")

    first = snapshot_attempt_ref(repo_root, "agent/issue-7", issue_number=7, base_ref="main")
    assert first.ref_name == f"{ATTEMPT_REF_PREFIX}/issue-7/attempt-1"
    assert first.old_tip == tip1

    # Simulate a second attempt: reset the branch and commit again.
    tip2 = _commit(repo_root, "attempt2.txt", "attempt 2\n")
    second = snapshot_attempt_ref(repo_root, "agent/issue-7", issue_number=7, base_ref="main")
    assert second.ref_name == f"{ATTEMPT_REF_PREFIX}/issue-7/attempt-2"
    assert second.old_tip == tip2
    assert second.old_tip != first.old_tip

    refs = list_attempt_refs(repo_root, 7)
    assert refs == (
        f"{ATTEMPT_REF_PREFIX}/issue-7/attempt-1",
        f"{ATTEMPT_REF_PREFIX}/issue-7/attempt-2",
    )


def test_list_attempt_refs_empty_when_none_exist(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    assert list_attempt_refs(repo_root, 999) == ()


def test_snapshot_attempt_ref_never_raises_on_non_git_directory(tmp_path: Path) -> None:
    """A repo_root that is not a git repository must degrade to an
    AttemptSnapshot with old_tip=None/error set, never raise — attempt
    preservation is fire-and-forget insurance around a redispatch and must
    never itself become the reason a redispatch fails.
    """
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()

    snapshot = snapshot_attempt_ref(not_a_repo, "some-branch", issue_number=1, base_ref="main")

    assert isinstance(snapshot, AttemptSnapshot)
    assert snapshot.ref_name is None
    assert snapshot.old_tip is None


def test_list_attempt_refs_never_raises_on_non_git_directory(tmp_path: Path) -> None:
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()

    assert list_attempt_refs(not_a_repo, 1) == ()
