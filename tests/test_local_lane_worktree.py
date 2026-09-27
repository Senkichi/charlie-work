"""``ensure_branch_worktree`` regression coverage (issues #1844, #1876, #1884).

Split out of ``tests/test_local_lane.py`` under the issue #1442 file-size
ratchet: the regression test #1857 added for #1876's create-vs-reuse
spelling fix landed in the over-cap monolith, so the
``ensure_branch_worktree``-specific cases live here -- the same
sibling-module move the collect-only gate's ``missing_sibling`` clause
exists to allow (leaf names reappear under ``tests/``, so nothing reads as
dropped).

The ``TestLocalLanePrimitives`` class name is pinned, not chosen: the
collect-only gate (issue #1538) compares leaf names that include the class
component (``TestLocalLanePrimitives::test_...``), so a verbatim relocation
must keep the wrapping class name identical -- renaming or unwrapping it
would read as a removal plus an addition.

``_git`` comes from ``tests/_worktree_fixtures.py`` -- the sanctioned
shared-plumbing import (``test_*.py -> test_*.py`` imports are forbidden by
``test_zero_cross_test_import_guard.py``). The repo builders stay local
because ``_init_repo``'s empty-commit seed deliberately differs from the
fixtures module's README-commit shape, and the ``repo`` fixture must sit in
``tempfile.mkdtemp`` -- pytest's ``tmp_path`` nests under this repo's own
worktree (``.var/worker-tmp/...``) and the paths git derives for
``worktree add`` (``<repo>/.git/worktrees/<name>`` and the target's
``gitdir:`` back-pointer) overflow git's internal ``$GIT_DIR`` buffer at
that depth -- ``fatal: '$GIT_DIR' too big``. The conftest's
``_isolate_git_env`` already whitelists the system temp dir via
``GIT_CEILING_DIRECTORIES``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

from _worktree_fixtures import _git
from charlie_work.local_lane import ensure_branch_worktree, worktree_for_branch


@pytest.fixture
def repo() -> Path:
    root = Path(tempfile.mkdtemp(prefix="cw-lane-"))
    yield root


def _init_repo(repo_root: Path) -> None:
    """A fresh git repo on ``main`` with one commit and NO origin remote."""
    repo_root.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "init", "--initial-branch=main")
    _git(repo_root, "config", "user.email", "test@example.test")
    _git(repo_root, "config", "user.name", "test")
    _git(repo_root, "commit", "--allow-empty", "-m", "chore: seed")


def _commit_file(repo_root: Path, relpath: str, content: str, message: str) -> str:
    path = repo_root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _git(repo_root, "add", relpath)
    _git(repo_root, "commit", "-m", message)
    return _git(repo_root, "rev-parse", "HEAD").stdout.strip()


def _make_branch(repo_root: Path, branch: str, relpath: str, content: str) -> str:
    """Branch off main with one committed file; return the new head sha."""
    _git(repo_root, "checkout", "-b", branch)
    head = _commit_file(repo_root, relpath, content, f"feat: {relpath}")
    _git(repo_root, "checkout", "main")
    return head


def _short_spelling(path: Path) -> Path:
    """The 8.3 short-name spelling of ``path`` on Windows, else ``path`` itself.

    windows-latest spells ``%TEMP%`` short (``RUNNER~1``), which is how the
    create-vs-reuse spelling split in ``ensure_branch_worktree`` first
    surfaced in CI. Where no short name exists (non-Windows host, or a
    volume with 8.3 generation disabled) the original path comes back and
    callers exercise the plain-path case.
    """
    if sys.platform != "win32":
        return path
    import ctypes

    buf = ctypes.create_unicode_buffer(1024)
    if ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, len(buf)):
        return Path(buf.value)
    return path


class TestLocalLanePrimitives:
    def test_ensure_branch_worktree_creates_and_reuses(self, repo: Path) -> None:
        _init_repo(repo)
        _make_branch(repo, "agent/issue-7-x", "a.py", "a = 1\n")
        worktrees_dir = repo / "wt"

        wt = ensure_branch_worktree(repo, "agent/issue-7-x", worktrees_dir)

        assert wt is not None and wt.is_dir()
        assert worktree_for_branch(repo, "agent/issue-7-x") == wt
        # Second call returns the same worktree rather than re-adding.
        assert ensure_branch_worktree(repo, "agent/issue-7-x", worktrees_dir) == wt

    def test_ensure_branch_worktree_returns_gits_recorded_spelling(self, repo: Path) -> None:
        """Create-path and reuse-path return the same canonical spelling.

        Regression for the windows-latest CI failure on #1844's head:
        ``%TEMP%`` there is the 8.3 short-name form, so the ``target``
        constructed from ``repo_root`` differed from the canonical path git
        records at ``worktree add`` time and
        ``worktree_for_branch(...) == wt`` failed. Spelling ``repo`` through
        its 8.3 name reproduces that split on any host with 8.3 generation
        enabled; elsewhere ``_short_spelling`` is the identity and the
        assertions still pin the create-vs-reuse invariant.
        """
        _init_repo(repo)
        _make_branch(repo, "agent/issue-7-x", "a.py", "a = 1\n")
        spelled = _short_spelling(repo)
        worktrees_dir = spelled / "wt"

        wt = ensure_branch_worktree(spelled, "agent/issue-7-x", worktrees_dir)

        assert wt is not None and wt.is_dir()
        assert wt == worktree_for_branch(repo, "agent/issue-7-x")
        assert ensure_branch_worktree(spelled, "agent/issue-7-x", worktrees_dir) == wt
