"""Issue #1820: a ``closed`` local issue still blocks when its worker branch never merged.

``LocalFileGitHub.are_issues_open`` is the single point that resolves blocker
state for the local-file backend. ``state: closed`` frontmatter alone is not
proof the blocker's work reached HEAD -- a human can close the file before (or
without ever) merging the ``{branch_prefix}-<n>-*`` branch. These tests run a
real git repo: the closed blocker must keep counting as open while a
resolvable worker-branch tip is not an ancestor of HEAD, and must fall back to
the plain frontmatter verdict when no such branch exists (never dispatched, or
merged and the ref cleaned up -- the default lane's own end state under
``auto_merge.delete_branch``).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from charlie_work.backlog_reachability import _get_open_blockers_for_issue
from charlie_work.local_issues import LocalFileGitHub


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    result = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, env=env)
    assert result.returncode == 0, (args, result.stderr)
    return result


def _init_repo(repo_root: Path) -> None:
    """A fresh git repo with one commit on ``main``.

    ``core.longpaths`` because pytest's basetemp under this repo's
    ``.var/worker-tmp`` pushes ``.git/objects`` paths past the Windows
    MAX_PATH limit; the ``GIT_*`` scrub stops an ambient ``GIT_DIR`` from
    redirecting the commands at the *outer* repo (same reasons as
    ``test_local_issues_loop_gates._init_repo``).
    """
    repo_root.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "init", "--initial-branch=main")
    _git(repo_root, "config", "core.longpaths", "true")
    _git(
        repo_root,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "--allow-empty",
        "-m",
        "chore: seed",
    )


def _write_issue(issues_dir: Path, number: int, *, state: str, body: str = "Body.") -> Path:
    issues_dir.mkdir(parents=True, exist_ok=True)
    path = issues_dir / f"{number:03d}_issue.md"
    path.write_text(f"---\nstate: {state}\nlabels: []\n---\n{body}\n", encoding="utf-8")
    return path


def _commit_on_branch(repo_root: Path, branch: str, filename: str) -> None:
    _git(repo_root, "switch", "-c", branch)
    (repo_root / filename).write_text("x = 1\n", encoding="utf-8")
    _git(repo_root, "add", filename)
    _git(repo_root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "work")
    _git(repo_root, "switch", "main")


@pytest.fixture
def local_repo(tmp_path: Path) -> tuple[Path, Path]:
    """A git repo whose issue #1 is closed and issue #2 is ``Blocked by #1``."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / "docs" / "issues"
    _write_issue(issues_dir, 1, state="closed", body="Add helper.")
    _write_issue(issues_dir, 2, state="open", body="Blocked by #1\n\nUse the helper.")
    return repo_root, issues_dir


def test_closed_blocker_with_unmerged_worker_branch_still_counts_as_open(
    local_repo: tuple[Path, Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The issue's reproduction: ``state: closed`` + unmerged ``agent/issue-1-*``
    branch must still count as an open blocker."""
    repo_root, issues_dir = local_repo
    _commit_on_branch(repo_root, "agent/issue-1-add-helper", "helper.py")
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    # The anomaly is surfaced once per client instance, not once per call --
    # are_issues_open runs several times per pass (prefetch + per-issue).
    with caplog.at_level("WARNING", logger="charlie_work.local_issues"):
        assert gh.are_issues_open([1]) == {1}
        gh.are_issues_open([1])
        gh.are_issues_open([1])
    warnings = [r for r in caplog.records if "worker branch is not merged" in r.getMessage()]
    assert len(warnings) == 1

    declared, open_blockers = _get_open_blockers_for_issue(gh, gh.issue_view(2))
    assert declared == [1]
    assert open_blockers == [1]


def test_closed_blocker_merged_branch_counts_as_closed(
    local_repo: tuple[Path, Path],
) -> None:
    """A worker branch merged into HEAD satisfies the blocker."""
    repo_root, issues_dir = local_repo
    _commit_on_branch(repo_root, "agent/issue-1-add-helper", "helper.py")
    _git(repo_root, "merge", "--no-ff", "-m", "merge", "agent/issue-1-add-helper")
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    assert gh.are_issues_open([1]) == set()
    _declared, open_blockers = _get_open_blockers_for_issue(gh, gh.issue_view(2))
    assert open_blockers == []


def test_closed_blocker_merged_then_branch_deleted_counts_as_closed(
    local_repo: tuple[Path, Path],
) -> None:
    """The default local merge lane's own end state: merged, then
    ``git branch -D`` under ``auto_merge.delete_branch``. The ref is gone, so
    there is nothing left to check -- dependents must not wedge forever."""
    repo_root, issues_dir = local_repo
    _commit_on_branch(repo_root, "agent/issue-1-add-helper", "helper.py")
    _git(repo_root, "merge", "--no-ff", "-m", "merge", "agent/issue-1-add-helper")
    _git(repo_root, "branch", "-D", "agent/issue-1-add-helper")
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    assert gh.are_issues_open([1]) == set()


def test_closed_issue_never_dispatched_falls_back_to_frontmatter(
    local_repo: tuple[Path, Path],
) -> None:
    """Closed-as-not-applicable: no worker branch was ever recorded, so
    ``state: closed`` keeps its historical verdict."""
    repo_root, issues_dir = local_repo
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    assert gh.are_issues_open([1]) == set()
    _declared, open_blockers = _get_open_blockers_for_issue(gh, gh.issue_view(2))
    assert open_blockers == []


def test_park_comment_branch_off_convention_still_blocks(
    local_repo: tuple[Path, Path],
) -> None:
    """The park comment names the branch exactly, so a worker branch outside
    the ``{prefix}-<n>-*`` convention is still honored."""
    repo_root, issues_dir = local_repo
    _commit_on_branch(repo_root, "custom/helper-branch", "helper.py")
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)
    comment = repo_root / "comment.md"
    comment.write_text(
        "Work for this issue is committed on branch `custom/helper-branch`. "
        "This repo has no remote.",
        encoding="utf-8",
    )
    gh.issue_comment(1, comment)

    assert gh.are_issues_open([1]) == {1}


def test_park_comment_branch_deleted_after_merge_counts_as_closed(
    local_repo: tuple[Path, Path],
) -> None:
    """A recorded branch name whose ref no longer resolves falls back to the
    frontmatter verdict -- merged-and-deleted cannot be told apart from
    deleted-unmerged at the ref level, and the former is the default lane's
    normal end state."""
    repo_root, issues_dir = local_repo
    _commit_on_branch(repo_root, "agent/issue-1-add-helper", "helper.py")
    _git(repo_root, "merge", "--no-ff", "-m", "merge", "agent/issue-1-add-helper")
    _git(repo_root, "branch", "-D", "agent/issue-1-add-helper")
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)
    comment = repo_root / "comment.md"
    comment.write_text(
        "Work for this issue is committed on branch `agent/issue-1-add-helper`. "
        "This repo has no remote.",
        encoding="utf-8",
    )
    gh.issue_comment(1, comment)

    assert gh.are_issues_open([1]) == set()


def test_configured_branch_prefix_drives_the_convention_scan(tmp_path: Path) -> None:
    """``github_client_for`` passes ``dispatch.branch_prefix`` through; a
    non-default prefix must scope the ``refs/heads/{prefix}-<n>-*`` scan."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / "docs" / "issues"
    _write_issue(issues_dir, 1, state="closed")
    _commit_on_branch(repo_root, "bot/task-1-add-helper", "helper.py")
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir, branch_prefix="bot/task")

    assert gh.are_issues_open([1]) == {1}
    # A client still on the default prefix sees no worker branch for #1.
    default_gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)
    assert default_gh.are_issues_open([1]) == set()


def test_sibling_issue_number_is_not_a_worker_branch(
    local_repo: tuple[Path, Path],
) -> None:
    """``agent/issue-12-x`` must not count as issue #1's worker branch."""
    repo_root, issues_dir = local_repo
    _commit_on_branch(repo_root, "agent/issue-12-other", "helper.py")
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    assert gh.are_issues_open([1]) == set()


def test_open_issue_is_unaffected_by_the_merge_check(local_repo: tuple[Path, Path]) -> None:
    """An open issue reports open regardless of its branch state -- the extra
    check only re-arms *closed* issues."""
    repo_root, issues_dir = local_repo
    _write_issue(issues_dir, 3, state="open")
    _commit_on_branch(repo_root, "agent/issue-3-wip", "wip.py")
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    assert gh.are_issues_open([3]) == {3}


def test_branch_pointing_at_base_counts_as_landed(local_repo: tuple[Path, Path]) -> None:
    """A worker branch whose tip adds nothing over its fork point is already
    contained in HEAD -- nothing to block on."""
    repo_root, issues_dir = local_repo
    _git(repo_root, "branch", "agent/issue-1-empty")
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    assert gh.are_issues_open([1]) == set()
