"""Tests for the ``clean_worktrees`` closed-issue lane (issue #2487).

Worktrees whose linked GitHub issue is closed used to sit skipped forever:
with no linked PR they hit "no linked PR in state.json", and with
uncommitted work they hit the dirty-tree safety skip -- both conditions
that can never resolve once the issue is closed. The lane
(``worktree_closed_reclaim.reclaim_closed_issue_worktree``) reclaims them
instead, in a strictly lossless order: live-session gate, rescue capture
of any uncommitted work to ``refs/charlie/rescue/`` (issue #849), then
``remove_worktree(force=True)`` with the branch kept.

Issue state -- never worktree cleanliness -- is the discriminator: an
open-issue worktree in the identical shape stays skipped, so the sweep can
never mistake a fresh checkout for a finished one.

New module rather than ``tests/test_worktree.py``: that file's module-level
attachment point is saturated (file-size ratchet, issue #1442), so new
coverage lands here and imports the shared cleanup-lane fake ``_FakeGH``
from ``tests/_worktree_fixtures.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from _worktree_fixtures import (
    _FakeGH,
    _git,
    _init_repo,
    _wt_scratch as _register_wt_scratch,  # noqa: F401 -- registers the wt_scratch fixture (shallow tmp dir for real ``git worktree add``)
)

from charlie_work.config import OrchestratorConfig
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.worktree import (
    RESCUE_REF_PREFIX,
    RescueCapture,
    _default_worktrees_dir,
    clean_worktrees,
    create_worktree,
)


def _unlinked_state(issue_number: int) -> dict[str, Any]:
    """state.json shape with the issue present but no PR link anywhere."""
    return {
        "issues": {str(issue_number): {"number": issue_number}},
        "prs": {},
        "events": [],
    }


def _linked_state(issue_number: int, pr_number: int, *, status: str = "merged") -> dict[str, Any]:
    """state.json shape with the issue linked to a PR record."""
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


def _repo_with_worktree(scratch: Path, branch: str) -> tuple[Path, Path, Path]:
    """Init a repo plus one dispatch-prefix worktree at the main tip.

    Returns ``(repo_root, worktree_path, worktrees_dir)``. The worktree's
    HEAD equals ``main``'s tip -- the "HEAD is in main" shape.
    """
    repo_root = scratch / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    info = create_worktree(repo_root, branch, base_ref="HEAD")
    return repo_root, info.path, _default_worktrees_dir(repo_root)


def _issue_view_calls(gh: _FakeGH) -> list[list[str]]:
    return [call for call in gh.calls if call[:2] == ["issue", "view"]]


def test_closed_issue_dirty_worktree_rescued_removed_and_branch_kept(
    wt_scratch: Path,
) -> None:
    """Acceptance #1: closed issue + uncommitted work -> the work lands in a
    rescue ref, the worktree is removed, and the branch is kept so unpushed
    commits survive."""
    branch = "agent/issue-21-closed-dirty"
    repo_root, wt_path, worktrees_dir = _repo_with_worktree(wt_scratch, branch)
    dirty_contents = "uncommitted worker work\n"
    (wt_path / "dirty.txt").write_text(dirty_contents, encoding="utf-8")
    config = OrchestratorConfig()
    state_file = runtime_paths(repo_root, config.runtime.state_dir).state_file
    gh = _FakeGH(issue_states={21: "CLOSED"})

    result = clean_worktrees(repo_root, worktrees_dir, _unlinked_state(21), config, gh)

    assert result.ok is True, result.message
    assert len(result.data["removed"]) == 1
    entry = result.data["removed"][0]
    assert entry["worktree"] == str(wt_path)
    assert entry["branch"] == branch
    assert entry["issue_number"] == 21
    assert entry["closed_issue"] is True
    assert not wt_path.exists()

    # The branch survives -- unpushed commits are not lost.
    assert _git(repo_root, "rev-parse", "--verify", f"refs/heads/{branch}").stdout.strip()

    # The uncommitted work survives in a durable rescue ref.
    rescue_ref = entry["rescue_ref"]
    assert rescue_ref.startswith(RESCUE_REF_PREFIX)
    assert _git(repo_root, "show", f"{rescue_ref}:dirty.txt").stdout == dirty_contents

    # The capture is also observable via the worktree_rescue_captured event.
    events = query_events(state_file, kind="worktree_rescue_captured")
    matching = [e for e in events if e.get("payload", {}).get("rescue_ref") == rescue_ref]
    assert len(matching) == 1
    assert matching[0]["payload"]["issue_number"] == 21


def test_closed_issue_clean_worktree_removed_without_pr(wt_scratch: Path) -> None:
    """Acceptance #2 shape: closed issue + clean worktree + no PR link -- the
    combination that used to skip as "no linked PR" forever. Removed with no
    rescue capture needed."""
    branch = "agent/issue-22-closed-clean"
    repo_root, wt_path, worktrees_dir = _repo_with_worktree(wt_scratch, branch)
    gh = _FakeGH(issue_states={22: "CLOSED"})

    result = clean_worktrees(
        repo_root, worktrees_dir, _unlinked_state(22), OrchestratorConfig(), gh
    )

    assert result.ok is True, result.message
    assert len(result.data["removed"]) == 1
    entry = result.data["removed"][0]
    assert entry["worktree"] == str(wt_path)
    assert entry["rescue_ref"] is None
    assert not wt_path.exists()
    assert _git(repo_root, "rev-parse", "--verify", f"refs/heads/{branch}").stdout.strip()


def test_open_issue_clean_worktree_at_main_is_still_skipped(wt_scratch: Path) -> None:
    """Acceptance #3: an open issue with a clean worktree whose HEAD is in
    main remains skipped -- issue state, not cleanliness, is what
    discriminates a finished worktree from a fresh one."""
    branch = "agent/issue-23-open-clean"
    repo_root, wt_path, worktrees_dir = _repo_with_worktree(wt_scratch, branch)
    gh = _FakeGH(issue_states={23: "OPEN"})

    result = clean_worktrees(
        repo_root, worktrees_dir, _unlinked_state(23), OrchestratorConfig(), gh
    )

    assert result.data["removed"] == []
    assert len(result.data["skipped"]) == 1
    assert "no linked PR" in result.data["skipped"][0]["reason"]
    assert wt_path.exists()
    # The discriminator really ran -- the skip is a verified OPEN, not an
    # unreachable lane.
    assert _issue_view_calls(gh) == [["issue", "view", "23", "--json", "state"]]


def test_closed_issue_worktree_with_live_worker_is_skipped(wt_scratch: Path) -> None:
    """A live worker must never lose its checkout, even when the linked
    issue is closed: the liveness gate runs before any capture or removal."""
    branch = "agent/issue-24-closed-live"
    repo_root, wt_path, worktrees_dir = _repo_with_worktree(wt_scratch, branch)
    (wt_path / "wip.txt").write_text("live worker wip\n", encoding="utf-8")
    state = _unlinked_state(24)
    state["issues"]["24"]["worker_pid"] = os.getpid()
    gh = _FakeGH(issue_states={24: "CLOSED"})

    result = clean_worktrees(repo_root, worktrees_dir, state, OrchestratorConfig(), gh)

    assert result.data["removed"] == []
    assert len(result.data["skipped"]) == 1
    assert "live worker" in result.data["skipped"][0]["reason"]
    assert wt_path.exists()


def test_closed_issue_capture_failure_keeps_skip(
    wt_scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed capture must not degrade into destruction: the worktree
    stays, skipped with the capture error named."""
    from charlie_work import worktree as worktree_module

    branch = "agent/issue-25-closed-capture-fail"
    repo_root, wt_path, worktrees_dir = _repo_with_worktree(wt_scratch, branch)
    (wt_path / "wip.txt").write_text("must not be lost\n", encoding="utf-8")
    monkeypatch.setattr(
        worktree_module,
        "_capture_worktree_work_to_rescue_ref",
        lambda *args, **kwargs: RescueCapture(
            ref_name=None, commit_sha=None, error="forced capture failure"
        ),
    )
    gh = _FakeGH(issue_states={25: "CLOSED"})

    result = clean_worktrees(
        repo_root, worktrees_dir, _unlinked_state(25), OrchestratorConfig(), gh
    )

    assert result.data["removed"] == []
    assert len(result.data["skipped"]) == 1
    assert "rescue capture failed" in result.data["skipped"][0]["reason"]
    assert wt_path.exists()
    assert (wt_path / "wip.txt").read_text(encoding="utf-8") == "must not be lost\n"


def test_closed_issue_lane_is_dry_run_safe(wt_scratch: Path) -> None:
    """``--dry-run`` must plan without mutating: no capture ref, no removal."""
    branch = "agent/issue-26-closed-dryrun"
    repo_root, wt_path, worktrees_dir = _repo_with_worktree(wt_scratch, branch)
    (wt_path / "wip.txt").write_text("uncommitted\n", encoding="utf-8")
    gh = _FakeGH(issue_states={26: "CLOSED"})

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        _unlinked_state(26),
        OrchestratorConfig(),
        gh,
        dry_run=True,
    )

    assert result.ok is True, result.message
    assert len(result.data["planned"]) == 1
    assert result.data["planned"][0]["closed_issue"] is True
    assert wt_path.exists()
    # No rescue ref was created for a dry-run plan.
    refs = _git(repo_root, "for-each-ref", "--format=%(refname)", RESCUE_REF_PREFIX).stdout
    assert refs.strip() == ""


def test_closed_issue_merged_pr_uses_merged_lane_and_deletes_branch(
    wt_scratch: Path,
) -> None:
    """Ordering regression (review finding): closing keywords auto-close the
    issue when its PR merges, so closed-issue + merged-PR is the COMMON
    merged shape. The merged lane must still see it -- it is the lane that
    deletes the branch -- and the closed-issue lane must never run ahead
    of it, or every merged branch leaks."""
    branch = "agent/issue-31-merged"
    repo_root, wt_path, worktrees_dir = _repo_with_worktree(wt_scratch, branch)
    head_sha = _git(wt_path, "rev-parse", "HEAD").stdout.strip()
    gh = _FakeGH(head_sha=head_sha, issue_states={31: "CLOSED"})

    result = clean_worktrees(
        repo_root, worktrees_dir, _linked_state(31, 131), OrchestratorConfig(), gh
    )

    assert result.ok is True, result.message
    assert len(result.data["removed"]) == 1
    entry = result.data["removed"][0]
    assert entry["pr_number"] == 131
    # The merged lane's removal record carries no closed_issue flag --
    # proving the PR lane, not the closed-issue lane, handled it.
    assert "closed_issue" not in entry
    assert not wt_path.exists()
    # The merged lane confirmed the PR and deleted the branch, exactly as
    # it did before the closed-issue lane existed.
    assert any(call[:2] == ["pr", "view"] for call in gh.calls)
    assert _git(repo_root, "branch", "--list", branch).stdout.strip() == ""


def test_closed_issue_open_pr_keeps_worktree_skipped(wt_scratch: Path) -> None:
    """Open-PR gate (review finding): an issue can be closed by hand while
    its PR is still under review rework. The closed-issue lane must not
    reclaim the checkout out from under a live review -- a resolved PR
    that is not terminal keeps the ordinary "PR not merged" wait."""
    branch = "agent/issue-32-open-pr"
    repo_root, wt_path, worktrees_dir = _repo_with_worktree(wt_scratch, branch)
    gh = _FakeGH(pr_state="OPEN", issue_states={32: "CLOSED"})

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        _linked_state(32, 132, status="open"),
        OrchestratorConfig(),
        gh,
    )

    assert result.data["removed"] == []
    assert len(result.data["skipped"]) == 1
    assert result.data["skipped"][0]["reason"] == "PR not merged"
    assert wt_path.exists()
    assert _git(repo_root, "rev-parse", "--verify", f"refs/heads/{branch}").stdout.strip()
    # The lane never even consulted the issue: a live PR owns the wait.
    assert _issue_view_calls(gh) == []


def test_closed_issue_merged_pr_dirty_worktree_rescued_and_removed(
    wt_scratch: Path,
) -> None:
    """The merged lane declines a dirty worktree even when the PR merged;
    on a confirmed-CLOSED issue that decline is terminal, so the
    closed-issue lane captures the uncommitted work to a rescue ref and
    removes the checkout -- branch kept, as with no PR at all."""
    branch = "agent/issue-33-merged-dirty"
    repo_root, wt_path, worktrees_dir = _repo_with_worktree(wt_scratch, branch)
    head_sha = _git(wt_path, "rev-parse", "HEAD").stdout.strip()
    dirty_contents = "uncommitted worker work\n"
    (wt_path / "wip.txt").write_text(dirty_contents, encoding="utf-8")
    gh = _FakeGH(head_sha=head_sha, issue_states={33: "CLOSED"})

    result = clean_worktrees(
        repo_root, worktrees_dir, _linked_state(33, 133), OrchestratorConfig(), gh
    )

    assert result.ok is True, result.message
    assert len(result.data["removed"]) == 1
    entry = result.data["removed"][0]
    # The closed-issue lane handled it AFTER the merged lane declined on
    # the dirty tree -- the live `gh pr view` really ran first.
    assert entry["closed_issue"] is True
    assert entry["pr_number"] == 133
    assert any(call[:2] == ["pr", "view"] for call in gh.calls)
    assert not wt_path.exists()
    assert _git(repo_root, "rev-parse", "--verify", f"refs/heads/{branch}").stdout.strip()
    rescue_ref = entry["rescue_ref"]
    assert rescue_ref.startswith(RESCUE_REF_PREFIX)
    assert _git(repo_root, "show", f"{rescue_ref}:wip.txt").stdout == dirty_contents


def test_issue_view_failure_falls_through_to_pr_lanes(wt_scratch: Path) -> None:
    """An unreachable ``gh`` cannot confirm CLOSED, so the worktree keeps the
    pre-existing behavior -- the closed-issue lane never activates on an
    error."""
    branch = "agent/issue-27-gh-down"
    repo_root, wt_path, worktrees_dir = _repo_with_worktree(wt_scratch, branch)
    gh = _FakeGH(available=False, error="gh: connection refused")

    result = clean_worktrees(
        repo_root, worktrees_dir, _unlinked_state(27), OrchestratorConfig(), gh
    )

    assert result.data["removed"] == []
    assert len(result.data["skipped"]) == 1
    assert "no linked PR" in result.data["skipped"][0]["reason"]
    assert wt_path.exists()
