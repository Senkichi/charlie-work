"""Tests for the ``clean_worktrees`` head-branch PR fallback (issue #1713).

A worktree whose state entry was pruned, or that was created outside the
normal dispatch path, has no linked PR in ``state.json`` -- the state-only
``_find_linked_pr_number`` lookup can never resolve it, so it used to sit
skipped as "no linked PR in state.json" forever. The fallback resolves the
PR by head branch instead (``gh pr list --head <branch> --state all``) and
adopts it only when exactly one PR matches; every downstream safety gate
(live ``gh pr view`` merge confirmation, containment, dirty tree, liveness)
is unchanged.

New module rather than ``tests/test_worktree.py``: that file's module-level
attachment point is saturated, so new coverage lands here and imports the
shared cleanup-lane fake ``_FakeGH`` from ``tests/_worktree_fixtures.py``
(the ``test_*.py -> test_*.py`` import the first draft used is banned by
``tests/test_zero_cross_test_import_guard.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from _worktree_fixtures import _FakeGH, _git, _init_repo

from charlie_work.config import OrchestratorConfig
from charlie_work.worktree import _default_worktrees_dir, clean_worktrees, create_worktree


def _unlinked_state(issue_number: int) -> dict[str, Any]:
    """state.json shape with the issue present but no PR link anywhere."""
    return {
        "issues": {str(issue_number): {"number": issue_number}},
        "prs": {},
        "events": [],
    }


def _repo_with_worktree(tmp_path: Path, branch: str) -> tuple[Path, Path, Path, str]:
    """Init a repo plus one dispatch-prefix worktree.

    Returns ``(repo_root, worktree_path, worktrees_dir, head_sha)``.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")

    info = create_worktree(repo_root, branch, base_ref="HEAD")
    head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    return repo_root, info.path, _default_worktrees_dir(repo_root), head_sha


def _pr_list_calls(gh: _FakeGH) -> list[list[str]]:
    return [call for call in gh.calls if call[:2] == ["pr", "list"]]


def test_head_fallback_reclaims_worktree_when_branch_has_one_merged_pr(
    tmp_path: Path,
) -> None:
    """Acceptance: no state link + exactly one merged PR on the branch +
    clean tree + HEAD contained in the merged head -> removed."""
    branch = "agent/issue-11-head-fallback"
    repo_root, wt_path, worktrees_dir, head_sha = _repo_with_worktree(tmp_path, branch)
    gh = _FakeGH(head_sha=head_sha, head_branch_prs={branch: [911]})

    result = clean_worktrees(
        repo_root, worktrees_dir, _unlinked_state(11), OrchestratorConfig(), gh
    )

    assert result.ok is True, result.message
    assert [entry["pr_number"] for entry in result.data["removed"]] == [911]
    assert not wt_path.exists()
    # The fallback ran exactly once, before the `gh pr view` confirmation.
    assert _pr_list_calls(gh) == [
        ["pr", "list", "--head", branch, "--state", "all", "--json", "number"]
    ]
    assert any(call[:2] == ["pr", "view"] for call in gh.calls)


def test_head_fallback_plans_removal_under_dry_run(tmp_path: Path) -> None:
    """The same single-match fallback must show up in a ``--dry-run`` plan --
    the operator's preview of what a real sweep would reclaim."""
    branch = "agent/issue-12-head-fallback-dry-run"
    repo_root, wt_path, worktrees_dir, head_sha = _repo_with_worktree(tmp_path, branch)
    gh = _FakeGH(head_sha=head_sha, head_branch_prs={branch: [912]})

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        _unlinked_state(12),
        OrchestratorConfig(),
        gh,
        dry_run=True,
    )

    assert result.ok is True, result.message
    assert [entry["pr_number"] for entry in result.data["planned"]] == [912]
    assert wt_path.exists()


def test_head_fallback_skips_stray_commit_beyond_merged_head(tmp_path: Path) -> None:
    """A fallback-resolved PR does not relax the containment gate: a worktree
    HEAD carrying a commit the merged PR head does not contain is skipped
    with the existing stray-commit reason."""
    branch = "agent/issue-13-head-fallback-stray"
    repo_root, wt_path, worktrees_dir, _head = _repo_with_worktree(tmp_path, branch)
    merged_head_sha = _git(wt_path, "rev-parse", "HEAD").stdout.strip()
    _git(wt_path, "config", "user.email", "test@example.test")
    _git(wt_path, "config", "user.name", "Test User")
    (wt_path / "stray.txt").write_text("post-merge work", encoding="utf-8")
    _git(wt_path, "add", "stray.txt")
    _git(wt_path, "commit", "-m", "stray commit beyond the merged head")
    gh = _FakeGH(head_sha=merged_head_sha, head_branch_prs={branch: [913]})

    result = clean_worktrees(
        repo_root, worktrees_dir, _unlinked_state(13), OrchestratorConfig(), gh
    )

    assert result.data["removed"] == []
    assert len(result.data["skipped"]) == 1
    assert "not contained in" in result.data["skipped"][0]["reason"]
    assert wt_path.exists()


def test_head_fallback_skips_dirty_worktree(tmp_path: Path) -> None:
    """A fallback-resolved PR does not relax the dirty-tree gate either."""
    branch = "agent/issue-14-head-fallback-dirty"
    repo_root, wt_path, worktrees_dir, head_sha = _repo_with_worktree(tmp_path, branch)
    (wt_path / "dirty_file.txt").write_text("worker-authored changes", encoding="utf-8")
    gh = _FakeGH(head_sha=head_sha, head_branch_prs={branch: [914]})

    result = clean_worktrees(
        repo_root, worktrees_dir, _unlinked_state(14), OrchestratorConfig(), gh
    )

    assert result.data["removed"] == []
    assert len(result.data["skipped"]) == 1
    assert "uncommitted" in result.data["skipped"][0]["reason"]
    assert wt_path.exists()


def test_head_fallback_keeps_skip_when_no_pr_matches(tmp_path: Path) -> None:
    """Zero PRs on the head branch -> the lookup proves nothing; keep the
    fail-closed skip, with the lookup outcome named in the reason."""
    branch = "agent/issue-15-head-fallback-none"
    repo_root, wt_path, worktrees_dir, _head = _repo_with_worktree(tmp_path, branch)
    gh = _FakeGH()

    result = clean_worktrees(
        repo_root, worktrees_dir, _unlinked_state(15), OrchestratorConfig(), gh
    )

    assert result.data["removed"] == []
    assert len(result.data["skipped"]) == 1
    reason = result.data["skipped"][0]["reason"]
    assert reason.startswith("no linked PR in state.json")
    assert "no PR found for head branch" in reason
    assert wt_path.exists()
    # The fallback really ran -- the skip is a verified zero-match, not the
    # old unconditional state-only miss.
    assert _pr_list_calls(gh)


def test_head_fallback_keeps_skip_when_multiple_prs_match(tmp_path: Path) -> None:
    """Two+ PRs sharing the head branch is ambiguous (e.g. a rework PR after
    an earlier closed one) -- picking one could compare against the wrong
    merged head, so the skip stays."""
    branch = "agent/issue-16-head-fallback-multi"
    repo_root, wt_path, worktrees_dir, _head = _repo_with_worktree(tmp_path, branch)
    gh = _FakeGH(head_branch_prs={branch: [916, 917]})

    result = clean_worktrees(
        repo_root, worktrees_dir, _unlinked_state(16), OrchestratorConfig(), gh
    )

    assert result.data["removed"] == []
    assert len(result.data["skipped"]) == 1
    reason = result.data["skipped"][0]["reason"]
    assert reason.startswith("no linked PR in state.json")
    assert "2 PRs share head branch" in reason
    assert wt_path.exists()


def test_head_fallback_keeps_skip_when_gh_list_fails(tmp_path: Path) -> None:
    """An erroring ``gh pr list`` carries no information about whether a PR
    exists -- fail closed exactly like the ``gh pr view`` error path."""
    branch = "agent/issue-17-head-fallback-gh-down"
    repo_root, wt_path, worktrees_dir, _head = _repo_with_worktree(tmp_path, branch)
    gh = _FakeGH(available=False, error="gh: connection refused")

    result = clean_worktrees(
        repo_root, worktrees_dir, _unlinked_state(17), OrchestratorConfig(), gh
    )

    assert result.data["removed"] == []
    assert len(result.data["skipped"]) == 1
    reason = result.data["skipped"][0]["reason"]
    assert reason.startswith("no linked PR in state.json")
    assert "head-branch PR lookup failed" in reason
    assert wt_path.exists()


def test_head_fallback_not_attempted_when_state_links_pr(tmp_path: Path) -> None:
    """The fallback exists for missing links only: when state.json already
    names the PR, no ``gh pr list`` call is issued at all."""
    branch = "agent/issue-18-head-fallback-linked"
    repo_root, _wt_path, worktrees_dir, head_sha = _repo_with_worktree(tmp_path, branch)
    gh = _FakeGH(head_sha=head_sha)
    state = {
        "issues": {"18": {"number": 18}},
        "prs": {"918": {"number": 918, "issue_number": 18, "status": "merged", "merged": True}},
        "events": [],
    }

    result = clean_worktrees(repo_root, worktrees_dir, state, OrchestratorConfig(), gh)

    assert [entry["pr_number"] for entry in result.data["removed"]] == [918]
    assert _pr_list_calls(gh) == []
