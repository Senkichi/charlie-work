"""``clean_worktrees``: merged-PR lane, live-worker/marker guards, dry run and the
orphan sweep."""

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from _worktree_fixtures import _FakeGH, _clone_repo, _git, _init_repo
from _worktree_shared_fixtures import _init_repo_with_remote, _make_state, _create_shared_venv
from charlie_work.config import DevinConfig, OrchestratorConfig
from charlie_work.process_utils import get_process_start_time
from charlie_work import worktree as worktree_module
from charlie_work.worktree import (
    OPERATOR_MARKER_KIND,
    OPERATOR_MARKER_SESSION_ID,
    WorktreeCleanResult,
    LiveWorkerRedispatchError,
    _default_worktrees_dir,
    clean_worktrees,
    create_worktree,
    is_junction,
    push_branch,
    write_worktree_marker,
    _slugify,
)


def test_clean_worktrees_removes_merged_worktree_and_verifies_shared_venv(
    tmp_path: Path,
) -> None:
    """Junction-safe removal deletes the worktree and leaves the shared venv valid."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    info = create_worktree(repo_root, "agent/issue-1-merged", base_ref="HEAD")
    head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    state = _make_state(issue_number=1, pr_number=101)

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=head_sha),
    )

    assert isinstance(result, WorktreeCleanResult)
    assert result.ok is True
    assert len(result.data["removed"]) == 1
    assert not info.path.exists()
    assert result.data["venv_ok"] is True
    assert "configured checkouts" in result.data["venv_message"]


def test_clean_worktrees_skips_dirty_worktree(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    info = create_worktree(repo_root, "agent/issue-2-dirty", base_ref="HEAD")
    head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    (info.path / "dirty_file.txt").write_text("local changes", encoding="utf-8")
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    state = _make_state(issue_number=2, pr_number=102)

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=head_sha),
    )

    assert result.ok is True
    assert len(result.data["skipped"]) == 1
    assert "uncommitted" in result.data["skipped"][0]["reason"]
    assert info.path.exists()


def test_clean_worktrees_reports_registered_and_out_of_scope_counts(
    tmp_path: Path,
) -> None:
    """Issue #1012: a worktree outside the dispatch branch prefix is never a
    dispatch candidate at all -- it must count toward
    ``worktrees_out_of_scope``, not silently vanish with zero accounting, so
    a durable payload can distinguish "never a candidate" from "considered
    and skipped"."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    # A real dispatch candidate that gets skipped for cause (dirty tree).
    info = create_worktree(repo_root, "agent/issue-2-dirty", base_ref="HEAD")
    head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    (info.path / "dirty_file.txt").write_text("local changes", encoding="utf-8")

    # An operator-created worktree whose branch never matches the dispatch
    # prefix -- this is the exact "never a candidate" shape from issue #1012's
    # "How it surfaced" section.
    create_worktree(repo_root, "operator/manual-fix", base_ref="HEAD")

    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    state = _make_state(issue_number=2, pr_number=102)

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=head_sha),
    )

    assert result.ok is True
    # `git worktree list` reports 3 entries: the repo's own main checkout,
    # the dirty candidate, and the operator one. The main checkout is
    # outside worktrees_dir and the operator worktree's branch is off the
    # dispatch prefix, so 2 of the 3 registered worktrees never became
    # candidates -- only the dirty one entered per-worktree eligibility
    # logic and landed in `skipped`.
    assert result.data["worktrees_registered"] == 3
    assert result.data["worktrees_out_of_scope"] == 2
    assert len(result.data["skipped"]) == 1
    assert "uncommitted" in result.data["skipped"][0]["reason"]


def test_clean_worktrees_ignores_injected_only_dirtiness(tmp_path: Path) -> None:
    """Issue #381: merged worktree with only injected prompt files dirty should be removed."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    info = create_worktree(repo_root, "agent/issue-381-clean", base_ref="HEAD")
    head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    for injected in config.dispatch.injected_paths:
        prompt = info.path / injected
        prompt.parent.mkdir(parents=True, exist_ok=True)
        prompt.write_text("injected prompt", encoding="utf-8")
    state = _make_state(issue_number=381, pr_number=1381)

    result = clean_worktrees(
        repo_root,
        _default_worktrees_dir(repo_root),
        state,
        config,
        _FakeGH(head_sha=head_sha),
    )

    assert result.ok is True
    assert len(result.data["removed"]) == 1
    assert result.data["removed"][0]["branch"] == "agent/issue-381-clean"
    assert not info.path.exists()


def test_clean_worktrees_skips_stray_post_merge_commit(tmp_path: Path) -> None:
    """A worktree HEAD that no longer matches the merged PR's headRefOid (a
    stray commit made after GitHub recorded the merge) must be skipped with a
    distinct reason -- it must NOT be silently removed just because the PR
    state.json/gh say "merged" (review finding 1: eligibility requires HEAD ==
    merged PR head, not just "PR is merged somewhere").
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    info = create_worktree(repo_root, "agent/issue-3-stray-commit", base_ref="HEAD")
    _git(info.path, "config", "user.email", "test@example.test")
    _git(info.path, "config", "user.name", "Test User")
    # This is the SHA GitHub would report as the merged PR's headRefOid --
    # captured BEFORE the stray commit below.
    merged_head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    (info.path / "new_file.txt").write_text("stray post-merge commit", encoding="utf-8")
    _git(info.path, "add", "new_file.txt")
    _git(info.path, "commit", "-m", "stray commit made after the PR was merged")
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    state = _make_state(issue_number=3, pr_number=103)

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=merged_head_sha),
    )

    assert result.ok is True
    assert len(result.data["skipped"]) == 1
    reason = result.data["skipped"][0]["reason"].lower()
    assert "stray" in reason
    assert info.path.exists()


def test_clean_worktrees_removes_worktree_behind_merged_head(tmp_path: Path) -> None:
    """The mirror of the stray-commit case, and the other half of the bug that
    made ``worktree-clean`` inert.

    The eligibility question is containment, not equality: does the worktree
    hold commits that did NOT get merged? A worktree sitting *behind* the
    merged head holds none — everything in it is reachable from what landed.
    That is the ordinary shape here, because the merge path advances the PR
    branch after the worker's last local commit (Aviator merge-queue rebases,
    merge-train updates, base-into-branch merges).

    The old ``local_head_sha != merged_head_sha`` test could not tell the two
    directions apart and reported 46 of 47 mismatching worktrees on the live
    host as "stray post-merge commit(s)" — a false reason attached to a
    permanent skip.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    info = create_worktree(repo_root, "agent/issue-3-behind", base_ref="HEAD")
    _git(info.path, "config", "user.email", "test@example.test")
    _git(info.path, "config", "user.name", "Test User")
    # The worktree's HEAD is where the worker stopped.
    local_head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    # The branch then advanced (merge queue rebase / base-into-branch update)
    # and GitHub recorded THAT tip as the merged headRefOid. Build it as a real
    # descendant commit, then put the worktree back where the worker left it.
    (info.path / "queued.txt").write_text("advanced by the merge queue", encoding="utf-8")
    _git(info.path, "add", "queued.txt")
    _git(info.path, "commit", "-m", "merge queue advanced the branch")
    merged_head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    _git(info.path, "reset", "--hard", local_head_sha)
    assert _git(info.path, "rev-parse", "HEAD").stdout.strip() == local_head_sha

    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    state = _make_state(issue_number=3, pr_number=103)

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=merged_head_sha),
    )

    assert result.ok is True, result.message
    assert result.data["skipped"] == []
    assert [entry["issue_number"] for entry in result.data["removed"]] == [3]
    assert not info.path.exists()


def test_clean_worktrees_skips_when_merged_head_object_is_absent(tmp_path: Path) -> None:
    """An unknown merged head must fail closed with a legible reason.

    ``git merge-base --is-ancestor`` exits non-zero both for "not an ancestor"
    and for "no such object", so the containment check gates on object
    presence first. Without that gate the skip would still happen, but under
    the wrong reason — and a wrong reason string is what makes a defect like
    this survive a reading of the output.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    info = create_worktree(repo_root, "agent/issue-3-absent", base_ref="HEAD")
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    state = _make_state(issue_number=3, pr_number=103)
    # A well-formed SHA that names no object in this repo.
    absent_sha = "0" * 39 + "1"

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=absent_sha),
    )

    assert result.data["removed"] == []
    assert len(result.data["skipped"]) == 1
    assert "not present in the local object store" in result.data["skipped"][0]["reason"]
    assert info.path.exists()


def test_clean_worktrees_fetches_missing_merged_pr_head_and_reclaims(tmp_path: Path) -> None:
    """Regression test for the bug this fix closes.

    A squash-merged PR whose remote branch was deleted post-merge leaves
    ``headRefOid`` unreachable from anything ``repo_root`` has ever fetched
    -- nothing local ever pointed at it, so it was never fetched in the
    first place (a live sweep hit this for 37 of 63 candidates and could
    never reclaim any of them). GitHub still serves the commit via
    ``refs/pull/<N>/head`` even after the branch is gone, so
    ``clean_worktrees`` must fetch that ref on demand and re-check before
    falling into the "not present in the local object store" skip.
    """
    remote, repo_root = _init_repo_with_remote(tmp_path)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _git(repo_root, "push", "origin", "main")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    branch_name = "agent/issue-9-fetch-pr-head"
    info = create_worktree(repo_root, branch_name, base_ref="origin/main")
    (info.path / "feature.txt").write_text("real work\n", encoding="utf-8")
    _git(info.path, "add", "feature.txt")
    _git(info.path, "commit", "-m", "real work for issue 9")
    push_ok, push_error = push_branch(repo_root, branch_name, worktree_path=info.path)
    assert push_ok, push_error

    # A separate clone advances the branch further (mirrors a merge-queue
    # rebase landing a new commit past the worker's last local commit) and
    # pushes the result ONLY to refs/pull/<pr>/head -- never as a branch --
    # mirroring how GitHub exposes a merged PR's head once its branch is
    # deleted post-merge (this repo's AutoMergeConfig default: squash +
    # delete_branch=True).
    third_clone = tmp_path / "third"
    _clone_repo(remote, third_clone)
    _git(third_clone, "fetch", "origin", branch_name)
    _git(third_clone, "checkout", "-b", branch_name, f"origin/{branch_name}")
    (third_clone / "queued.txt").write_text("advanced by the merge queue\n", encoding="utf-8")
    _git(third_clone, "add", "queued.txt")
    _git(third_clone, "commit", "-m", "merge queue advanced the branch")
    merged_head_sha = _git(third_clone, "rev-parse", "HEAD").stdout.strip()
    _git(third_clone, "push", "origin", "HEAD:refs/pull/109/head")
    _git(third_clone, "push", "origin", "--delete", branch_name)

    # Sanity-check the regression shape: repo_root has never fetched this
    # object (it never fetched refs/pull/*, and the branch that once could
    # have carried it was deleted before repo_root ever saw the advance).
    cat_file = subprocess.run(
        ["git", "cat-file", "-e", f"{merged_head_sha}^{{commit}}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert cat_file.returncode != 0

    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    state = _make_state(issue_number=9, pr_number=109)

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=merged_head_sha),
    )

    assert result.data["skipped"] == [], result.data["skipped"]
    assert [entry["issue_number"] for entry in result.data["removed"]] == [9]
    assert not info.path.exists()


def test_clean_worktrees_skips_when_pr_head_fetch_fails(tmp_path: Path) -> None:
    """The fetch-on-demand fallback must itself fail closed, without raising.

    Distinct from ``test_clean_worktrees_skips_when_merged_head_object_is_absent``
    (which has no origin remote at all, so the fetch attempt is skipped
    entirely): here an origin remote exists, but nothing was ever pushed to
    ``refs/pull/<pr>/head``, so the fetch this fix adds actually runs and
    fails ("couldn't find remote ref"). That failure must degrade to the
    same fail-closed skip and reason as before -- never an exception that
    aborts the whole sweep.
    """
    remote, repo_root = _init_repo_with_remote(tmp_path)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _git(repo_root, "push", "origin", "main")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    info = create_worktree(repo_root, "agent/issue-11-fetch-fails", base_ref="origin/main")
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    state = _make_state(issue_number=11, pr_number=111)
    # A well-formed SHA that names no object anywhere -- not locally, and
    # never pushed to refs/pull/111/head on the remote either.
    absent_sha = "0" * 39 + "2"

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=absent_sha),
    )

    assert result.data["removed"] == []
    assert len(result.data["skipped"]) == 1
    assert "not present in the local object store" in result.data["skipped"][0]["reason"]
    assert info.path.exists()


def test_clean_worktrees_removes_squash_merged_worktree_with_deleted_remote_branch(
    tmp_path: Path,
) -> None:
    """Production-mirroring regression test (review finding 1).

    This repo's ``AutoMergeConfig`` defaults are ``strategy="squash"`` with
    ``delete_branch=True``. After a real merge: the remote branch is gone, and
    the squash commit landed on ``main`` is NOT an ancestor of the worker
    branch's own commit (its tree matches, but it is a distinct commit with a
    different parent). The old ``_worktree_refuse_to_reset_reason``-based
    eligibility check counted the worker's own committed work as "local
    commit(s) not on remote branch" in exactly this shape and refused to clean
    up, forever. ``clean_worktrees`` must still remove the worktree here
    because gh confirms MERGED and the worktree's local HEAD equals the
    merged PR's headRefOid -- local-ahead-of-a-deleted-remote is expected.
    """
    remote, repo_root = _init_repo_with_remote(tmp_path)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _git(repo_root, "push", "origin", "main")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    branch_name = "agent/issue-7-squash-merged"
    info = create_worktree(repo_root, branch_name, base_ref="origin/main")
    (info.path / "feature.txt").write_text("real work\n", encoding="utf-8")
    _git(info.path, "add", "feature.txt")
    _git(info.path, "commit", "-m", "real work for issue 7")
    branch_head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    push_ok, push_error = push_branch(repo_root, branch_name, worktree_path=info.path)
    assert push_ok, push_error

    # Simulate a squash-merge into main: a NEW commit on main whose parent is
    # main's own tip, not an ancestor of branch_head_sha.
    _git(repo_root, "merge", "--squash", branch_name)
    _git(repo_root, "commit", "-m", f"squash merge of {branch_name}")
    _git(repo_root, "push", "origin", "main")
    # Simulate delete_branch=True: the remote branch is gone after merge.
    _git(repo_root, "push", "origin", "--delete", branch_name)

    # Sanity-check the exact regression shape: the squash commit on main is
    # NOT an ancestor of the branch's own commit.
    is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", branch_head_sha, "main"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert is_ancestor.returncode != 0, "test setup invalid: squash commit must not be an ancestor"

    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    state = _make_state(issue_number=7, pr_number=107)

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=branch_head_sha),
    )

    assert len(result.data["skipped"]) == 0, result.data["skipped"]
    assert len(result.data["removed"]) == 1
    assert result.ok is True
    assert not info.path.exists()


def test_clean_worktrees_skips_live_worker(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    info = create_worktree(repo_root, "agent/issue-4-live", base_ref="HEAD")
    head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    issue_state = {
        "number": 4,
        "worker_pid": os.getpid(),
        "worker_process_start_time": get_process_start_time(os.getpid()),
        "started_at": datetime.now(UTC).isoformat(),
    }
    state = _make_state(issue_number=4, pr_number=104)
    state["issues"]["4"] = issue_state

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=head_sha),
    )

    assert result.ok is True
    assert len(result.data["skipped"]) == 1
    assert "live" in result.data["skipped"][0]["reason"].lower()
    assert info.path.exists()


def test_clean_worktrees_removes_merged_worktree_with_no_recorded_worker_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression that made ``worktree-clean`` a no-op on the live host.

    ``clean_worktrees`` used to gate removal on ``_probe_recovery_liveness``,
    the *redispatch* lane's guard. That probe is fail-closed on an inconclusive
    Devin activity probe, and on this repo's hosts it is inconclusive forever:
    finished workers have no recorded pid (so the per-PID log source reports
    ``no pid``), the Devin ``sessions.db`` holds no rows for claude-code/api
    workers, and the probe's only escape hatch — the
    ``inconclusive_probe_deferred_count`` cap — is never persisted by this
    lane, so the counter is pinned at 0. Result: 70 of 81 merged worktrees
    skipped with ``live worker detected: probe_error``, on every pass.

    The old tests missed it because ``real_activity_for_worker`` *raised* under
    tmp_path (no sessions.db at all), and the probe swallows exceptions as
    "not live" — the opposite of what a real host with a real db does. So this
    test forces the production shape directly: the redispatch probe declares a
    live worker, and the merged worktree must still be removed, because the
    cleanup lane must not consult that probe at all.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    def _always_probe_error(*args: Any, **kwargs: Any) -> None:
        raise LiveWorkerRedispatchError(
            issue_number=4,
            pid=None,
            process_start_time=None,
            probe_result="probe_error",
            inconclusive_probe_deferred_count=1,
        )

    monkeypatch.setattr(worktree_module, "_probe_recovery_liveness", _always_probe_error)

    info = create_worktree(repo_root, "agent/issue-4-no-pid", base_ref="HEAD")
    head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    # No worker_pid / last_known_worker_pid: the shape of a worker that
    # finished long ago and whose pid was pruned from state.
    state = _make_state(issue_number=4, pr_number=104)

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=head_sha),
    )

    assert result.ok is True, result.message
    assert result.data["skipped"] == []
    assert [entry["issue_number"] for entry in result.data["removed"]] == [4]
    assert not info.path.exists()


def test_clean_worktrees_skips_worktree_with_live_writer_marker(tmp_path: Path) -> None:
    """No recorded pid, but a writer marker whose pid is alive: still in use.

    This is the signal that replaces the redispatch probe — positive evidence
    of a running process rather than an absence of records.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    # Register the worktree with git directly rather than create_worktree:
    # the cleanup lane is the subject here, and a pristine fresh repo
    # exercises none of create_worktree's leftover/reclaim checks (each of
    # which costs a git subprocess — the dominant per-test cost on Windows,
    # issue #2363). ``src/`` likewise stays uncommitted: the shared-venv
    # verifier reads the filesystem, not the index.
    branch = "agent/issue-4-marker"
    worktrees_dir = _default_worktrees_dir(repo_root)
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    wt_path = worktrees_dir / _slugify(branch)
    _git(repo_root, "worktree", "add", "-b", branch, str(wt_path), "HEAD")
    head_sha = _git(wt_path, "rev-parse", "HEAD").stdout.strip()
    write_worktree_marker(wt_path, os.getpid(), "session-live")
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    state = _make_state(issue_number=4, pr_number=104)

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=head_sha),
    )

    assert result.data["removed"] == []
    assert len(result.data["skipped"]) == 1
    assert "live writer marker" in result.data["skipped"][0]["reason"]
    assert wt_path.exists()


def test_clean_worktrees_skips_operator_claimed_worktree(tmp_path: Path) -> None:
    """Operator markers carry a sentinel pid; state.json is the authority.

    A pid-only liveness check would read the sentinel as dead and delete a
    worktree an operator is actively editing.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    info = create_worktree(repo_root, "agent/issue-4-operator", base_ref="HEAD")
    head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    write_worktree_marker(info.path, 0, OPERATOR_MARKER_SESSION_ID, kind=OPERATOR_MARKER_KIND)
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    state = _make_state(issue_number=4, pr_number=104)
    state["issues"]["4"]["operator_claimed_at"] = datetime.now(UTC).isoformat()

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=head_sha),
    )

    assert result.data["removed"] == []
    assert len(result.data["skipped"]) == 1
    assert "operator-claimed" in result.data["skipped"][0]["reason"]
    assert info.path.exists()


def test_clean_worktrees_released_operator_marker_does_not_block_removal(
    tmp_path: Path,
) -> None:
    """A leftover operator marker with no live claim in state must not pin the
    worktree forever — the marker alone is not evidence of a live operator."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    info = create_worktree(repo_root, "agent/issue-4-released", base_ref="HEAD")
    head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    write_worktree_marker(info.path, 0, OPERATOR_MARKER_SESSION_ID, kind=OPERATOR_MARKER_KIND)
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    # No operator_claimed_at: the claim was released.
    state = _make_state(issue_number=4, pr_number=104)

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=head_sha),
    )

    assert result.data["skipped"] == []
    assert [entry["issue_number"] for entry in result.data["removed"]] == [4]
    assert not info.path.exists()


def test_clean_worktrees_dry_run_reports_plan_and_does_not_remove(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    info = create_worktree(repo_root, "agent/issue-5-dry-run", base_ref="HEAD")
    head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    state = _make_state(issue_number=5, pr_number=105)

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=head_sha),
        dry_run=True,
    )

    assert result.ok is True
    assert len(result.data["planned"]) == 1
    assert len(result.data["removed"]) == 0
    assert info.path.exists()


def test_clean_worktrees_surfaces_poisoned_venv_after_removal(tmp_path: Path) -> None:
    """If removal leaves the shared venv .pth pointing at a worktree, report it."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")

    info = create_worktree(repo_root, "agent/issue-6-poison", base_ref="HEAD")
    head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    _create_shared_venv(repo_root, pth_target=info.path / "src")

    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    state = _make_state(issue_number=6, pr_number=106)

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=head_sha),
    )

    assert result.ok is False
    assert len(result.data["removed"]) == 1
    assert result.data["venv_ok"] is False
    # charlie_work's package root is derivable, so the per-package message names
    # the expected root rather than the generic "outside all roots" fallback.
    assert "expected its package root" in result.data["venv_message"]


def test_clean_worktrees_removes_junctioned_worktree_and_preserves_shared_venv_contents(
    tmp_path: Path,
) -> None:
    """Non-blocking review item: junction safety through the clean_worktrees
    path itself (not just remove_worktree in isolation). A worktree with a
    live ``.venv`` junction removed via clean_worktrees must delete only the
    reparse point, never the shared venv's real contents.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    # Mirror this repo's own .gitignore (.venv/ is ignored); otherwise the
    # .venv junction itself shows up as an untracked path under `git status
    # --porcelain` and the dirty check would (incorrectly, for this test's
    # purposes) treat every junctioned worktree as dirty.
    (repo_root / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    _git(repo_root, "add", ".gitignore")
    _git(repo_root, "commit", "-m", "add .gitignore")
    venv_source = tmp_path / "shared-venv"
    venv_source.mkdir()
    marker = venv_source / "site-packages-marker.txt"
    marker.write_text("shared contents\n", encoding="utf-8")

    info = create_worktree(
        repo_root, "agent/issue-12-junction", base_ref="HEAD", venv_source=venv_source
    )
    assert is_junction(info.path / ".venv")
    head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()

    worktrees_dir = _default_worktrees_dir(repo_root)
    # No devin.venv_source configured here: this test is only about the
    # removal path's junction safety (remove_worktree), not the separate
    # post-removal poisoned-.pth check, which has its own dedicated tests.
    config = OrchestratorConfig()
    state = _make_state(issue_number=12, pr_number=112)

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=head_sha),
    )

    assert len(result.data["removed"]) == 1
    assert not info.path.exists()
    # The shared venv itself, and its contents, must survive the junction-safe
    # removal driven through clean_worktrees.
    assert venv_source.exists()
    assert marker.exists()
    assert marker.read_text(encoding="utf-8") == "shared contents\n"


@pytest.mark.skipif(os.name != "nt", reason="Windows reparse point regression")
def test_clean_worktrees_orphan_sweep_removes_unregistered_tree_with_reparse_point(
    tmp_path: Path,
) -> None:
    """Issue #462: orphan directories left after a failed git worktree remove are
    detected and removed without following reparse points.
    """
    import _winapi

    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    _git(repo_root, "add", ".gitignore")
    _git(repo_root, "commit", "-m", "ignore venv")

    shared_venv = tmp_path / "shared-venv"
    shared_venv.mkdir()
    marker = shared_venv / "site-packages-marker.txt"
    marker.write_text("shared contents\n", encoding="utf-8")

    worktrees_dir = _default_worktrees_dir(repo_root)
    orphan_dir = worktrees_dir / "agent-issue-462-orphan"
    orphan_dir.mkdir(parents=True)
    (orphan_dir / "stale.txt").write_text("stale\n", encoding="utf-8")
    _winapi.CreateJunction(str(shared_venv), str(orphan_dir / ".venv"))

    config = OrchestratorConfig()
    state = _make_state(issue_number=462, pr_number=462)
    result = clean_worktrees(repo_root, worktrees_dir, state, config, _FakeGH())

    assert result.ok is True
    assert not orphan_dir.exists()
    assert shared_venv.exists()
    assert marker.read_text(encoding="utf-8") == "shared contents\n"
    assert any(str(orphan_dir) == r["worktree"] for r in result.data["orphans"]["removed"])


def test_clean_worktrees_orphan_sweep_spares_live_foreign_worktree(
    tmp_path: Path,
) -> None:
    """2026-08-09 incident: the orphan sweep deleted the sibling ci_runners
    worktree (another repo's live checkout inside this repo's worktrees dir,
    provisioned by worker launch shims for the ci-fleet editable) because it
    can never appear in this repo's own ``git worktree list``. A live foreign
    worktree must be spared; a dangling one (admin dir gone) is residue and
    stays sweepable.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    worktrees_dir = _default_worktrees_dir(repo_root)
    worktrees_dir.mkdir(parents=True, exist_ok=True)

    foreign_repo = tmp_path / "ci_runners"
    _init_repo(foreign_repo)
    foreign_sibling = worktrees_dir / "ci_runners"
    _git(foreign_repo, "worktree", "add", "--detach", str(foreign_sibling))
    assert (foreign_sibling / ".git").is_file()

    plain_orphan = worktrees_dir / "agent-issue-999-residue"
    plain_orphan.mkdir()
    (plain_orphan / "stale.txt").write_text("stale\n", encoding="utf-8")

    config = OrchestratorConfig()
    state = _make_state(issue_number=999, pr_number=999)
    result = clean_worktrees(repo_root, worktrees_dir, state, config, _FakeGH())

    assert result.ok is True
    assert foreign_sibling.is_dir()
    assert not plain_orphan.exists()
    removed_paths = {r["worktree"] for r in result.data["orphans"]["removed"]}
    assert str(plain_orphan) in removed_paths
    assert str(foreign_sibling) not in removed_paths

    # Dangle the foreign worktree's registration: with the admin dir gone it
    # is residue, and the next sweep must reclaim it.
    shutil.rmtree(foreign_repo / ".git" / "worktrees")
    result = clean_worktrees(repo_root, worktrees_dir, state, config, _FakeGH())
    assert result.ok is True
    assert not foreign_sibling.exists()
    removed_paths = {r["worktree"] for r in result.data["orphans"]["removed"]}
    assert str(foreign_sibling) in removed_paths


def test_clean_worktrees_skips_orphan_sweep_when_worktree_list_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A git worktree list failure must not be read as zero registered worktrees.

    The orphan sweep must be skipped and surfaced as an attention event so a
    transient git hiccup cannot silently destroy live worker state.
    """
    from charlie_work.subprocess_runner import RunResult
    import charlie_work.worktree as worktree_mod

    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    info = create_worktree(repo_root, "agent/issue-1-live", base_ref="HEAD")
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig()
    state = _make_state(issue_number=1, pr_number=101)

    original_run = worktree_mod.run_captured

    def fake_run(args: list[str], **kwargs: Any) -> RunResult:
        if args[:4] == ["git", "worktree", "list", "--porcelain"]:
            return RunResult(
                returncode=1,
                stdout="",
                stderr="simulated git failure",
                error="simulated git failure",
            )
        return original_run(args, **kwargs)

    monkeypatch.setattr(worktree_mod, "run_captured", fake_run)

    result = clean_worktrees(repo_root, worktrees_dir, state, config, _FakeGH())

    assert info.path.exists()
    assert not result.data["orphans"]["removed"]
    assert not result.data["orphans"]["failed"]
    assert any(e["type"] == "worktree_list_failed" for e in result.data["attention_events"])
    assert result.ok is False
