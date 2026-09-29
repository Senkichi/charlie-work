"""Issue #2019: dead-worker uncommitted source edits are not shim dirt."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from charlie_work import config
from charlie_work.worktree import (
    WORKTREE_UNSAFE_KIND_SHIM_DIRT,
    WORKTREE_UNSAFE_KIND_UNCOMMITTED_WORK,
    WORKTREE_UNSAFE_KINDS,
    WorktreeUnsafeError,
    _capture_worktree_work_to_rescue_ref,
    _worktree_dirty_reason,
    _worktree_refuse_to_reset_reason,
    _worktree_unsafe_kind_from_reason,
)


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    _git(tmp_path, "init", "-q", "-b", "main", str(root))
    (root / "src" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "chore: init")
    return root


def test_dirty_tracked_source_file_classifies_as_uncommitted_work(repo: Path) -> None:
    (repo / "src" / "mod.py").write_text("x = 2\n", encoding="utf-8")

    reason = _worktree_refuse_to_reset_reason(repo, "main", "main", repo)

    assert reason is not None
    assert "src/mod.py" in reason
    assert _worktree_unsafe_kind_from_reason(reason) == WORKTREE_UNSAFE_KIND_UNCOMMITTED_WORK
    assert WorktreeUnsafeError(reason).kind == WORKTREE_UNSAFE_KIND_UNCOMMITTED_WORK
    assert WORKTREE_UNSAFE_KIND_UNCOMMITTED_WORK in WORKTREE_UNSAFE_KINDS


def test_shim_only_dirt_still_classifies_as_shim_dirt(repo: Path) -> None:
    (repo / ".adapter").mkdir()
    (repo / ".adapter" / "settings.json").write_text("{}\n", encoding="utf-8")

    reason = _worktree_dirty_reason(repo)

    assert reason == "worktree has uncommitted modifications"
    assert _worktree_unsafe_kind_from_reason(reason) == WORKTREE_UNSAFE_KIND_SHIM_DIRT


def test_mixed_dirt_names_source_paths_only(repo: Path) -> None:
    (repo / ".adapter").mkdir()
    (repo / ".adapter" / "settings.json").write_text("{}\n", encoding="utf-8")
    (repo / "src" / "mod.py").write_text("x = 3\n", encoding="utf-8")

    reason = _worktree_dirty_reason(repo)

    assert reason is not None
    assert "src/mod.py" in reason
    assert ".adapter" not in reason
    assert _worktree_unsafe_kind_from_reason(reason) == WORKTREE_UNSAFE_KIND_UNCOMMITTED_WORK


def test_uncommitted_work_kind_is_judgment_class_not_mechanical() -> None:
    kind = WORKTREE_UNSAFE_KIND_UNCOMMITTED_WORK
    assert kind in config.DETERMINISTIC_JUDGMENT_ESCALATION_FAILURE_KINDS
    assert kind not in config.DETERMINISTIC_ESCALATION_FAILURE_KINDS


def test_rescue_capture_preserves_source_edit_on_ref(repo: Path) -> None:
    (repo / "src" / "mod.py").write_text("x = 4\n", encoding="utf-8")

    capture = _capture_worktree_work_to_rescue_ref(repo, repo, 2019)

    assert capture.error is None
    assert capture.ref_name is not None
    assert capture.ref_name.startswith("refs/charlie/rescue/issue-2019-")
    assert _git(repo, "show", f"{capture.ref_name}:src/mod.py") == "x = 4\n"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
