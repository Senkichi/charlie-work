"""Helpers shared by more than one ``test_worktree*`` module (issue #1555).

Hoisted out of the former single ``test_worktree.py`` when it was split into
seam-named siblings. Kept in a dedicated fixture module -- not ``conftest.py`` --
so each sibling imports exactly what it uses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from _worktree_fixtures import _clone_repo, _init_repo
from charlie_work.worktree import RescueCapture


def _force_capture_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch rescue capture to always fail so the refusal path is exercised.

    Issue #849 added rescue capture before ``WorktreeUnsafeError`` — when
    capture succeeds, the reset proceeds and no error is raised. The existing
    refusal tests below test the safety property that a correct refusal
    prevents data loss; they must run with capture disabled to exercise the
    refusal path. The rescue capture path itself is tested by the
    ``test_rescue_capture_*`` tests at the end of this file.
    """
    monkeypatch.setattr(
        "charlie_work.worktree._capture_worktree_work_to_rescue_ref",
        lambda *args, **kwargs: RescueCapture(
            ref_name=None, commit_sha=None, error="forced capture failure for refusal test"
        ),
    )


def _init_repo_with_remote(tmp_path: Path) -> tuple[Path, Path]:
    """Create a bare remote and a local clone, return (remote, repo)."""
    remote = tmp_path / "remote"
    _init_repo(remote, bare=True)
    repo = tmp_path / "repo"
    _clone_repo(remote, repo)
    return remote, repo


def _make_state(issue_number: int, pr_number: int, *, status: str = "merged") -> dict[str, Any]:
    return {
        "issues": {str(issue_number): {"number": issue_number}},
        "prs": {
            str(pr_number): {
                "number": pr_number,
                "issue_number": issue_number,
                "status": status,
                "merged": status == "merged",
            }
        },
        "events": [],
    }


def _create_shared_venv(repo_root: Path, pth_target: Path | None = None) -> Path:
    """Create a fake shared venv with an editable .pth pointing at the target src."""
    pth = repo_root / "shared-venv" / "Lib" / "site-packages" / "_editable_impl_charlie_work.pth"
    pth.parent.mkdir(parents=True)
    target = pth_target or (repo_root / "src")
    pth.write_text(str(target.resolve()) + "\n", encoding="utf-8")
    return repo_root / "shared-venv"
