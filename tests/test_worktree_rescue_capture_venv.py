"""Regression tests for the rescue-capture ignored-exclusion bug.

On git 2.45.1.windows.1, ``git add -A -- . ':(exclude)<path>'`` exits 1 with
"The following paths are ignored by one of your .gitignore files" advice
whenever ``<path>`` both exists and is already gitignored -- even though the
resulting staged tree is correct either way (an already-ignored path was
never going to be staged by ``-A`` regardless of the exclude pathspec).
``_capture_worktree_work_to_rescue_ref`` unconditionally excludes ``.venv``,
which is gitignored in every real charlie-work checkout, so this trips on
essentially every rescue capture attempt: ``add_result.ok`` is False, the
function returns ``RescueCapture(error=...)``, and ``_capture_or_raise``
(worktree.py) raises ``WorktreeUnsafeError`` -- discarding a capture that
would otherwise have succeeded and turning a recoverable dirty-worktree
refusal into a hard one.

New file (not tests/test_worktree.py) to avoid merge conflicts with sibling
PRs also touching that large, frequently-changed file.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from _worktree_fixtures import _git
from charlie_work import worktree as worktree_module
from charlie_work.worktree import (
    RESCUE_REF_PREFIX,
    RescueCapture,
    WorktreeUnsafeError,
    _LAUNCHER_OWNED_PR_BODY_RE,
    _build_rescue_capture_exclusions,
    _capture_worktree_work_to_rescue_ref,
    _filter_redundant_add_exclusions,
    _worker_authored_dirty,
    create_worktree,
)


def _init_repo_with_gitignore(repo_root: Path, gitignore_lines: list[str]) -> None:
    """A minimal non-bare repo with an initial commit and a .gitignore.

    ``_capture_worktree_work_to_rescue_ref`` only needs a valid git HEAD in
    ``worktree_path`` (and writes the rescue ref via ``update-ref`` in
    ``repo_root``, which the tests below pass as the same directory) -- it
    does not require an actual ``git worktree add``-linked checkout, so a
    plain repo is sufficient and keeps the fixture minimal.
    """
    repo_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "--initial-branch=main"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    _git(repo_root, "config", "user.email", "test@example.test")
    _git(repo_root, "config", "user.name", "Test User")
    (repo_root / ".gitignore").write_text("\n".join(gitignore_lines) + "\n", encoding="utf-8")
    (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
    _git(repo_root, "add", ".gitignore", "README.md")
    _git(repo_root, "commit", "-m", "initial commit")


# --------------------------------------------------------------------------
# Fix A: _filter_redundant_add_exclusions / _build_rescue_capture_exclusions
# --------------------------------------------------------------------------


def test_filter_redundant_add_exclusions_drops_ignored_literal(tmp_path: Path) -> None:
    """A literal exclusion target that exists and is gitignored is dropped
    (it would trip the rc=1 advice/error); a literal target that exists and
    is NOT gitignored is kept; a literal target that does not exist at all
    is dropped (nothing to protect)."""
    repo_root = tmp_path / "repo"
    _init_repo_with_gitignore(repo_root, [".venv/"])
    (repo_root / ".venv").mkdir()
    (repo_root / ".venv" / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")
    (repo_root / "kept_dir").mkdir()
    (repo_root / "kept_dir" / "f.txt").write_text("x\n", encoding="utf-8")

    filtered = _filter_redundant_add_exclusions(
        repo_root, [".venv", "kept_dir", "does_not_exist"]
    )

    assert filtered == ["kept_dir"]


def test_filter_redundant_add_exclusions_keeps_glob_targets_unconditionally(
    tmp_path: Path,
) -> None:
    """Glob pathspecs (containing *, ?, or [) are passed through untouched
    regardless of ignore/existence status -- empirically, git's ignored-file
    advice/error does not trigger for a glob exclusion even when every match
    is gitignored, so pruning them would add complexity for no benefit."""
    repo_root = tmp_path / "repo"
    _init_repo_with_gitignore(repo_root, ["PR_BODY_*.md"])

    filtered = _filter_redundant_add_exclusions(repo_root, ["PR_BODY*.md", ".venv"])

    assert filtered == ["PR_BODY*.md"]


def test_build_rescue_capture_exclusions_omits_ignored_venv_keeps_globs(
    tmp_path: Path,
) -> None:
    """The full exclusion-builder drops the now-redundant ``.venv`` literal
    when it is gitignored, but always keeps the PR-body glob patterns."""
    repo_root = tmp_path / "repo"
    _init_repo_with_gitignore(repo_root, [".venv/"])
    (repo_root / ".venv").mkdir()
    (repo_root / ".venv" / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")

    exclusions = _build_rescue_capture_exclusions(repo_root, (), ())

    assert ":(exclude).venv" not in exclusions
    assert ":(exclude)PR_BODY*.md" in exclusions
    assert ":(exclude).pr_body*.md" in exclusions


# --------------------------------------------------------------------------
# Fix A regression: _capture_worktree_work_to_rescue_ref with a real,
# gitignored .venv present -- the exact shape that trips rc=1.
# --------------------------------------------------------------------------


def test_rescue_capture_succeeds_with_gitignored_venv_present(tmp_path: Path) -> None:
    """RED on origin/main, GREEN after the fix (see the red/green evidence
    in the PR description). A worktree with a real, gitignored ``.venv``
    directory (mirroring the shared-venv junction in every live
    charlie-work worktree) plus genuine worker-authored dirt must capture
    successfully -- before the fix, the unconditional ``:(exclude).venv``
    pathspec trips git's ignored-file advice/error (git 2.45.1.windows.1),
    ``add_result.ok`` is False, and capture returns an error even though
    the worker's dirty content was never at risk.
    """
    repo_root = tmp_path / "repo"
    _init_repo_with_gitignore(repo_root, [".venv/"])
    (repo_root / ".venv").mkdir()
    (repo_root / ".venv" / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")

    # Genuine worker-authored dirt: a tracked modification + an untracked file.
    (repo_root / "README.md").write_text("modified by worker\n", encoding="utf-8")
    (repo_root / "worker_output.txt").write_text("real work\n", encoding="utf-8")

    capture = _capture_worktree_work_to_rescue_ref(repo_root, repo_root, issue_number=99999)

    assert capture.error is None, f"capture failed: {capture.error}"
    assert capture.ref_name is not None
    assert capture.ref_name.startswith(RESCUE_REF_PREFIX)

    tree_paths = _git(repo_root, "ls-tree", "-r", "--name-only", capture.ref_name).stdout
    assert "worker_output.txt" in tree_paths
    assert ".venv/pyvenv.cfg" not in tree_paths
    assert (
        _git(repo_root, "show", f"{capture.ref_name}:README.md").stdout == "modified by worker\n"
    )


# --------------------------------------------------------------------------
# Fix B: _capture_or_raise propagates the capture error into the raised
# WorktreeUnsafeError message instead of discarding it.
# --------------------------------------------------------------------------


def test_capture_or_raise_includes_capture_error_in_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuine capture failure (forced here rather than relying on the
    rc=1 bug so this test is independent of the fix A behavior above) must
    still raise WorktreeUnsafeError -- capture failure must never downgrade
    the safety property -- and the raised message must now include the
    capture's own error detail so a capture-stage failure is distinguishable
    from a bare dirty-worktree refusal.
    """
    remote_repo = tmp_path / "remote"
    remote_repo.mkdir()
    subprocess.run(
        ["git", "init", "--bare", "--initial-branch=main"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    seed = tmp_path / "seed"
    _init_repo_with_gitignore(seed, ["*.pyc"])
    _git(seed, "remote", "add", "origin", str(remote_repo))
    _git(seed, "push", "origin", "main")

    repo_root = tmp_path / "repo"
    subprocess.run(
        ["git", "clone", str(remote_repo), str(repo_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(repo_root, "config", "user.email", "test@example.test")
    _git(repo_root, "config", "user.name", "Test User")

    branch_name = "agent/issue-77-capture-error-message"
    info = create_worktree(repo_root, branch_name, base_ref="origin/main")
    (info.path / "dirty.txt").write_text("uncommitted worker edit\n", encoding="utf-8")

    monkeypatch.setattr(
        worktree_module,
        "_capture_worktree_work_to_rescue_ref",
        lambda *args, **kwargs: RescueCapture(
            ref_name=None, commit_sha=None, error="simulated write-tree failure xyz123"
        ),
    )

    with pytest.raises(WorktreeUnsafeError) as exc_info:
        create_worktree(
            repo_root,
            branch_name,
            rework=True,
            base_ref="",
            issue_number=77,
        )

    message = str(exc_info.value)
    assert "worktree has uncommitted modifications" in message
    assert "simulated write-tree failure xyz123" in message


# --------------------------------------------------------------------------
# Fix C: launcher-owned PR-body regex + capture exclusions recognize the
# real ".pr_body_<issue>.md" spelling (confirmed live on the issue #1418
# rework worktree, alongside PR_BODY.md).
# --------------------------------------------------------------------------


def test_launcher_owned_pr_body_regex_matches_dot_pr_body_variant() -> None:
    assert _LAUNCHER_OWNED_PR_BODY_RE.match(".pr_body_1418.md")
    assert _LAUNCHER_OWNED_PR_BODY_RE.match(".pr_body.md")
    # Still matches every pre-existing spelling.
    assert _LAUNCHER_OWNED_PR_BODY_RE.match("PR_BODY.md")
    assert _LAUNCHER_OWNED_PR_BODY_RE.match("PR_BODY_42.md")
    assert _LAUNCHER_OWNED_PR_BODY_RE.match(".worker-pr-body.md")
    assert _LAUNCHER_OWNED_PR_BODY_RE.match("_pr_body.md")
    # Negative control: an unrelated markdown file must not match.
    assert not _LAUNCHER_OWNED_PR_BODY_RE.match("README.md")


def test_worker_authored_dirty_ignores_dot_pr_body_scratch_file(tmp_path: Path) -> None:
    """Issue #1418 scenario: the launcher writes ``.pr_body_<issue>.md``
    (leading dot, digits) alongside ``PR_BODY.md``. Before the fix, this
    exact spelling did not match ``_LAUNCHER_OWNED_PR_BODY_RE``, so the dirty
    check flagged it as worker-authored dirt even though it is launcher
    residue."""
    repo_root = tmp_path / "repo"
    _init_repo_with_gitignore(repo_root, ["*.pyc"])
    (repo_root / ".pr_body_1418.md").write_text("draft\n", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, ()) is False


def test_rescue_capture_excludes_dot_pr_body_scratch_file(tmp_path: Path) -> None:
    """The capture-exclusion list (separate from the dirty-check regex) must
    also exclude the ``.pr_body_<issue>.md`` spelling so it never pollutes a
    rescue tree reached via other genuine worker dirt."""
    repo_root = tmp_path / "repo"
    _init_repo_with_gitignore(repo_root, ["*.pyc"])
    (repo_root / ".pr_body_1418.md").write_text("draft\n", encoding="utf-8")
    (repo_root / "worker_output.txt").write_text("real work\n", encoding="utf-8")

    capture = _capture_worktree_work_to_rescue_ref(repo_root, repo_root, issue_number=1418)

    assert capture.error is None, f"capture failed: {capture.error}"
    tree_paths = _git(repo_root, "ls-tree", "-r", "--name-only", capture.ref_name).stdout
    assert "worker_output.txt" in tree_paths
    assert ".pr_body_1418.md" not in tree_paths
