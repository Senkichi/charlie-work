"""``clean_worktrees``: open/closed-unmerged PR lanes and their fail-closed probes."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from _worktree_fixtures import _FakeGH, _git, _init_repo
from _worktree_shared_fixtures import _init_repo_with_remote, _make_state, _create_shared_venv
from charlie_work.config import DevinConfig, OrchestratorConfig
from charlie_work.process_utils import get_process_start_time
from charlie_work.subprocess_runner import RunResult
from charlie_work.worktree import (
    _default_worktrees_dir,
    clean_worktrees,
    create_worktree,
    push_branch,
)


def test_clean_worktrees_skips_open_pr(tmp_path: Path) -> None:
    """Review finding 2: an open-PR worktree must never be removed.

    This is the highest-severity failure mode -- destroying a worktree whose
    PR is still open and possibly still being worked on.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    info = create_worktree(repo_root, "agent/issue-8-open", base_ref="HEAD")
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig()
    state = _make_state(issue_number=8, pr_number=108, status="open")

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(pr_state="OPEN"),
    )

    assert len(result.data["removed"]) == 0
    assert len(result.data["skipped"]) == 1
    assert "not merged" in result.data["skipped"][0]["reason"].lower()
    assert info.path.exists()


def test_clean_worktrees_removes_closed_unmerged_pr_with_nothing_to_lose(
    tmp_path: Path,
) -> None:
    """Issue #990: a closed-and-never-merged PR is a terminal decision, not a
    pending state -- it will never become ``MERGED``, so a worktree that only
    waits on that condition is pinned forever (measured on the live host: 6
    of 16 worktrees, all with confirmed-CLOSED PRs). When the worktree is
    clean and has no commits beyond where it started (nothing a removal could
    lose), it must be reclaimed exactly like a merged one instead of skipped
    as "PR not merged" in perpetuity. No origin remote is configured here on
    purpose -- the worktree never advanced past its base, so eligibility is
    provable without a network round-trip at all.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    info = create_worktree(repo_root, "agent/issue-9-closed", base_ref="HEAD")
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig()
    state = _make_state(issue_number=9, pr_number=109, status="closed")

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(pr_state="CLOSED", merged_at=None),
    )

    assert result.ok is True, result.message
    assert result.data["skipped"] == []
    assert len(result.data["removed"]) == 1
    assert result.data["removed"][0]["issue_number"] == 9
    assert not info.path.exists()


def test_clean_worktrees_skips_closed_unmerged_pr_on_probe_failure(tmp_path: Path) -> None:
    """Issue #1011, through the actual destructive caller: when the closed-
    unmerged-PR reclaim path's local-tip probe (``git rev-parse --verify -q
    HEAD`` run inside the worktree) fails for a reason other than a
    confirmed-missing ref, ``clean_worktrees`` must skip -- not remove -- so
    that ``remove_worktree(..., force=True, branch=branch)`` never runs
    ``git branch -D`` on a branch whose "nothing to lose" status was never
    actually confirmed. Before this fix, the helper would have returned
    ``None`` here (misreading the broken probe as a confirmed-absent ref) and
    this caller would have deleted the worktree AND the branch ref, leaving
    only reflog-based recovery. This is the worktree-HEAD probe shape
    (``cwd == wt_path``), the sibling of the branch-name-in-repo-root shape
    already covered by ``test_local_tip_probe_failure_refuses_to_reset``.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    branch_name = "agent/issue-11-closed-probe-fail"
    info = create_worktree(repo_root, branch_name, base_ref="HEAD")
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig()
    state = _make_state(issue_number=11, pr_number=111, status="closed")

    import charlie_work.worktree

    original_run_captured = charlie_work.worktree.run_captured

    def mock_run_captured(*args: object, **kwargs: object) -> object:
        if (
            isinstance(args[0], list)
            and args[0] == ["git", "rev-parse", "--verify", "-q", "HEAD"]
            and Path(kwargs.get("cwd", "")) == info.path
        ):
            return RunResult(
                returncode=128,
                stdout="",
                stderr="fatal: not a git repository (or any of the parent directories): .git",
                error="command exited 128",
            )
        return original_run_captured(*args, **kwargs)

    charlie_work.worktree.run_captured = mock_run_captured
    try:
        result = clean_worktrees(
            repo_root,
            worktrees_dir,
            state,
            config,
            _FakeGH(pr_state="CLOSED", merged_at=None),
        )
    finally:
        charlie_work.worktree.run_captured = original_run_captured

    assert result.ok is True, result.message
    assert result.data["removed"] == []
    assert len(result.data["skipped"]) == 1
    assert "safety probe failed" in result.data["skipped"][0]["reason"]
    assert info.path.exists()


def test_clean_worktrees_skips_closed_pr_with_merged_at_set(tmp_path: Path) -> None:
    """Pins the other half of the ``gh_closed_unmerged`` conjunct (issue #990):
    a PR can be ``state == "CLOSED"`` while ``mergedAt`` is non-null -- GitHub
    sets both together for a PR that was merged and then closed (or reported
    as closed by a stale/racing ``gh`` call). That is not the terminal
    "closed, never merged" case the new eligibility path exists for, so it
    must still fall into the fail-closed "PR not merged" branch rather than
    being routed into the closed-unmerged removal path on ``state`` alone.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    info = create_worktree(repo_root, "agent/issue-10-closed-merged-at", base_ref="HEAD")
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig()
    state = _make_state(issue_number=10, pr_number=110, status="closed")

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(pr_state="CLOSED"),  # default merged_at is non-null
    )

    assert result.ok is True, result.message
    assert result.data["removed"] == []
    assert len(result.data["skipped"]) == 1
    assert "not merged" in result.data["skipped"][0]["reason"]
    assert info.path.exists()


def test_clean_worktrees_removes_closed_unmerged_pr_with_fully_pushed_branch(
    tmp_path: Path,
) -> None:
    """The realistic shape from issue #990 (e.g. PR #980, closed as superseded
    by a competing fix): the worker's committed work is still pushed to its
    own remote branch, because closing a PR -- unlike a squash-merge with
    ``delete_branch=True`` -- never deletes the branch. Eligibility here goes
    through ``_worktree_refuse_to_reset_reason``'s ordinary remote-ahead
    check: local HEAD equals the remote branch tip, so nothing would be lost
    by removing the worktree.
    """
    remote, repo_root = _init_repo_with_remote(tmp_path)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _git(repo_root, "push", "origin", "main")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    branch_name = "agent/issue-12-closed-pushed"
    info = create_worktree(repo_root, branch_name, base_ref="origin/main")
    (info.path / "feature.txt").write_text("work later superseded\n", encoding="utf-8")
    _git(info.path, "add", "feature.txt")
    _git(info.path, "commit", "-m", "work for issue 12, later superseded")
    push_ok, push_error = push_branch(repo_root, branch_name, worktree_path=info.path)
    assert push_ok, push_error

    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    state = _make_state(issue_number=12, pr_number=112, status="closed")

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(pr_state="CLOSED", merged_at=None),
    )

    assert result.ok is True, result.message
    assert result.data["skipped"] == []
    assert len(result.data["removed"]) == 1
    assert not info.path.exists()


def test_clean_worktrees_skips_closed_unmerged_pr_with_unpushed_commits(
    tmp_path: Path,
) -> None:
    """Pins the case issue #990 calls out explicitly: a closed-unmerged PR is
    precisely where a human may have abandoned work worth keeping, so a
    worktree holding a commit that never made it to the remote branch must
    still wait -- reclaiming it would destroy the only copy. This is the
    "must still wait" counterpart to the fully-pushed removal case above.
    """
    remote, repo_root = _init_repo_with_remote(tmp_path)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _git(repo_root, "push", "origin", "main")

    branch_name = "agent/issue-13-closed-unpushed"
    info = create_worktree(repo_root, branch_name, base_ref="origin/main")
    (info.path / "feature.txt").write_text("pushed work\n", encoding="utf-8")
    _git(info.path, "add", "feature.txt")
    _git(info.path, "commit", "-m", "pushed work for issue 13")
    push_ok, push_error = push_branch(repo_root, branch_name, worktree_path=info.path)
    assert push_ok, push_error

    # A further commit made AFTER the push -- never made it to the remote.
    (info.path / "orphaned.txt").write_text("never pushed\n", encoding="utf-8")
    _git(info.path, "add", "orphaned.txt")
    _git(info.path, "commit", "-m", "abandoned work, never pushed")

    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig()
    state = _make_state(issue_number=13, pr_number=113, status="closed")

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(pr_state="CLOSED", merged_at=None),
    )

    assert result.data["removed"] == []
    assert len(result.data["skipped"]) == 1
    reason = result.data["skipped"][0]["reason"].lower()
    assert "closed-unmerged pr" in reason
    assert "not on remote branch" in reason
    assert info.path.exists()


def test_clean_worktrees_skips_closed_unmerged_pr_with_dirty_worktree(
    tmp_path: Path,
) -> None:
    """A dirty working tree must still block reclamation on the
    closed-unmerged path exactly as it does on the merged path."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    info = create_worktree(repo_root, "agent/issue-14-closed-dirty", base_ref="HEAD")
    (info.path / "dirty_file.txt").write_text("local changes", encoding="utf-8")
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig()
    state = _make_state(issue_number=14, pr_number=114, status="closed")

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(pr_state="CLOSED", merged_at=None),
    )

    assert result.data["removed"] == []
    assert len(result.data["skipped"]) == 1
    assert "uncommitted" in result.data["skipped"][0]["reason"].lower()
    assert info.path.exists()


def test_clean_worktrees_skips_closed_unmerged_pr_with_live_worker(tmp_path: Path) -> None:
    """A closed-unmerged PR does not bypass the liveness gate: an active
    worker still using the worktree must block reclamation exactly like the
    merged path -- issue #990 only widens which terminal PR states qualify,
    it never removes the liveness check.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    info = create_worktree(repo_root, "agent/issue-15-closed-live", base_ref="HEAD")
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig()
    issue_state = {
        "number": 15,
        "worker_pid": os.getpid(),
        "worker_process_start_time": get_process_start_time(os.getpid()),
        "started_at": datetime.now(UTC).isoformat(),
    }
    state = _make_state(issue_number=15, pr_number=115, status="closed")
    state["issues"]["15"] = issue_state

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(pr_state="CLOSED", merged_at=None),
    )

    assert result.ok is True
    assert result.data["removed"] == []
    assert len(result.data["skipped"]) == 1
    assert "live" in result.data["skipped"][0]["reason"].lower()
    assert info.path.exists()


def test_clean_worktrees_state_json_merged_alone_is_not_sufficient(tmp_path: Path) -> None:
    """Review finding 3: state.json claiming "merged" must NOT authorize
    removal when the live gh pr view disagrees. Mutating the eligibility
    check to ``state_merged or gh_merged`` (the pre-fix behavior) makes this
    test fail, since state.json here says "merged" but gh reports OPEN.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    info = create_worktree(repo_root, "agent/issue-10-stale-state", base_ref="HEAD")
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig()
    # state.json (stale/wrong) claims the PR is merged.
    state = _make_state(issue_number=10, pr_number=110, status="merged")

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(pr_state="OPEN"),
    )

    assert len(result.data["removed"]) == 0
    assert len(result.data["skipped"]) == 1
    reason = result.data["skipped"][0]["reason"].lower()
    assert "state.json" in reason
    assert "gh pr view" in reason
    assert info.path.exists()


def test_clean_worktrees_gh_unavailable_fails_closed(tmp_path: Path) -> None:
    """Review finding 3: when gh itself is unavailable/erroring, clean_worktrees
    must fail CLOSED (skip) rather than falling back to trusting state.json,
    even though state.json says "merged".
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    info = create_worktree(repo_root, "agent/issue-11-gh-down", base_ref="HEAD")
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig()
    state = _make_state(issue_number=11, pr_number=111, status="merged")

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(available=False, error="gh: network is unreachable"),
    )

    assert len(result.data["removed"]) == 0
    assert len(result.data["skipped"]) == 1
    reason = result.data["skipped"][0]["reason"]
    assert "unavailable" in reason.lower() or "cannot confirm" in reason.lower()
    assert "network is unreachable" in reason
    assert info.path.exists()
