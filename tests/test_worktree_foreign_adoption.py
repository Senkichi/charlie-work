"""Issue #1476: rework dispatch adopts a safe foreign checkout of the PR branch.

When the rework target branch is already checked out in a worktree the
orchestrator did not create (e.g. ``.claude/worktrees/<name>`` — issue #1317 /
PR #1449's ``worktree-agent-*`` checkout), rework dispatch previously raised
``worktree_foreign_writer`` on every pass: git refuses a second checkout of the
same branch, so the issue retried until the blocked-environment cap escalated
it — on a checkout that was live, held real PR commits, and was never going
away. The fix verifies the foreign checkout is safe (clean, not the repo's own
main checkout, no live foreign writer) and dispatches the rework session
directly into it; anything unsafe stays refused. An adopted checkout is owned
by someone else, so every teardown path must leave it — and its uncommitted
work — on disk.

This module covers the ordinary-rework gate and adoption semantics: the
safety refusals, the branch matcher, registered-path resolution, the adoption
event, and borrowed-checkout mutation containment (scaffolding repair, .venv
junctions). Recovery-mode adoption, the liveness probe, and launch-failure
teardown live in ``tests/test_worktree_foreign_adoption_recovery.py`` — the
file-size ratchet (issue #1442) required the split, and the shared
remote/clone helpers moved to ``tests/_worktree_fixtures.py``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from _worktree_fixtures import (
    _clone_with_pushed_branch,
    _git,
    _init_repo,
    _push_sibling_commit,
)

from charlie_work.config import OrchestratorConfig
from charlie_work.worktree import (
    RESCUE_REF_PREFIX,
    ReworkBranchConflictError,
    WorktreeForeignWriterError,
    _create_junction_or_symlink,
    _merge_update_rework_branch,
    _slugify,
    create_worktree,
    find_branch_worktree,
    is_junction,
    worktree_path_for_branch,
    write_worktree_marker,
)


def test_rework_adopts_clean_foreign_worktree(tmp_path: Path) -> None:
    """A clean foreign checkout of the rework branch is adopted for the rework
    session instead of blocking dispatch forever."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch = "agent/issue-1476-foreign"
    foreign_wt = tmp_path / "operator-worktree"
    _git(repo_root, "worktree", "add", str(foreign_wt), "-b", branch)

    managed_dir = tmp_path / "charlie-worktrees"
    info = create_worktree(repo_root, branch, rework=True, worktrees_dir=managed_dir)

    assert info.path == foreign_wt
    assert info.foreign_adopted is True

    _git(repo_root, "worktree", "remove", str(foreign_wt), "--force")


def test_rework_adopts_foreign_worktree_behind_origin_tip(tmp_path: Path) -> None:
    """A foreign checkout strictly behind ``origin/<branch>`` is adopted and
    fast-forwarded to the remote tip — the same mutation the managed path
    performs, and a non-destructive one."""
    branch = "agent/issue-1476-behind"
    remote, repo_root = _clone_with_pushed_branch(tmp_path, branch)
    foreign_wt = tmp_path / "operator-worktree"
    _git(repo_root, "worktree", "add", str(foreign_wt), branch)

    remote_tip = _push_sibling_commit(remote, tmp_path, branch, "advanced.txt")

    info = create_worktree(repo_root, branch, rework=True, worktrees_dir=tmp_path / "managed")

    assert info.path == foreign_wt
    assert info.foreign_adopted is True
    assert _git(foreign_wt, "rev-parse", "HEAD").stdout.strip() == remote_tip


def test_rework_refuses_diverged_foreign_worktree(tmp_path: Path) -> None:
    """A foreign checkout whose branch diverged non-FF from origin is refused:
    catching it up requires ``git reset --hard``, which the orchestrator may
    only run on worktrees it owns. The checkout and its commits stay intact."""
    branch = "agent/issue-1476-diverged"
    remote, repo_root = _clone_with_pushed_branch(tmp_path, branch)
    foreign_wt = tmp_path / "operator-worktree"
    _git(repo_root, "worktree", "add", str(foreign_wt), branch)

    # Diverge: the foreign checkout commits locally while a second clone
    # pushes a different commit to the same branch.
    (foreign_wt / "local.txt").write_text("local commit\n", encoding="utf-8")
    _git(foreign_wt, "add", "local.txt")
    _git(foreign_wt, "commit", "-m", "foreign local commit")
    local_sha = _git(foreign_wt, "rev-parse", "HEAD").stdout.strip()
    _push_sibling_commit(remote, tmp_path, branch, "remote.txt")

    with pytest.raises(WorktreeForeignWriterError) as exc_info:
        create_worktree(repo_root, branch, rework=True, worktrees_dir=tmp_path / "managed")

    assert exc_info.value.worktree_path == foreign_wt
    assert "diverged" in str(exc_info.value)
    # The foreign checkout is untouched — no reset, no removal.
    assert foreign_wt.is_dir()
    assert _git(foreign_wt, "rev-parse", "HEAD").stdout.strip() == local_sha


def test_rework_refuses_dirty_foreign_worktree(tmp_path: Path) -> None:
    """Uncommitted changes in a foreign checkout are never adopted, reset, or
    rescue-captured — they belong to whoever owns the checkout. Refusal leaves
    the working tree byte-identical and creates no rescue ref."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch = "agent/issue-1476-dirty"
    foreign_wt = tmp_path / "operator-worktree"
    _git(repo_root, "worktree", "add", str(foreign_wt), "-b", branch)
    (foreign_wt / "README.md").write_text("uncommitted foreign edit\n", encoding="utf-8")

    with pytest.raises(WorktreeForeignWriterError) as exc_info:
        create_worktree(repo_root, branch, rework=True, worktrees_dir=tmp_path / "managed")

    assert exc_info.value.worktree_path == foreign_wt
    assert "uncommitted" in str(exc_info.value)
    assert (foreign_wt / "README.md").read_text(encoding="utf-8") == ("uncommitted foreign edit\n")
    # No rescue capture was attempted — adopting never writes refs for foreign
    # work it refuses to touch.
    refs = _git(repo_root, "for-each-ref", RESCUE_REF_PREFIX).stdout.strip()
    assert refs == ""


def test_rework_refuses_foreign_worktree_with_live_writer(tmp_path: Path) -> None:
    """A live foreign-writer marker still wins over adoption — the existing
    #400/#1423 marker guard is unchanged."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch = "agent/issue-1476-writer"
    foreign_wt = tmp_path / "operator-worktree"
    _git(repo_root, "worktree", "add", str(foreign_wt), "-b", branch)

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    write_worktree_marker(foreign_wt, os.getpid(), "foreign-session")

    with pytest.raises(WorktreeForeignWriterError) as exc_info:
        create_worktree(
            repo_root,
            branch,
            rework=True,
            worktrees_dir=tmp_path / "managed",
            sessions_dir=sessions_dir,
        )

    assert exc_info.value.worktree_path == foreign_wt
    assert exc_info.value.pid == os.getpid()


def test_rework_refuses_prunable_foreign_worktree(tmp_path: Path) -> None:
    """A registered worktree whose directory is gone cannot host a session —
    refuse rather than let git prune-drift surface mid-launch."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch = "agent/issue-1476-prunable"
    foreign_wt = tmp_path / "operator-worktree"
    _git(repo_root, "worktree", "add", str(foreign_wt), "-b", branch)
    shutil.rmtree(foreign_wt)

    with pytest.raises(WorktreeForeignWriterError) as exc_info:
        create_worktree(repo_root, branch, rework=True, worktrees_dir=tmp_path / "managed")

    assert exc_info.value.worktree_path == foreign_wt
    assert "missing" in str(exc_info.value)

    _git(repo_root, "worktree", "prune")


def test_rework_refuses_branch_checked_out_in_main_checkout(tmp_path: Path) -> None:
    """The repo's own main checkout is never adopted: dispatching a worker
    there would run worker tooling inside the checkout that hosts the
    orchestrator (and usually the operator's own session)."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch = "agent/issue-1476-main"
    _git(repo_root, "checkout", "-b", branch)

    with pytest.raises(WorktreeForeignWriterError) as exc_info:
        create_worktree(repo_root, branch, rework=True, worktrees_dir=tmp_path / "managed")

    assert exc_info.value.worktree_path == repo_root
    assert "main checkout" in str(exc_info.value)

    _git(repo_root, "checkout", "main")
    _git(repo_root, "branch", "-D", branch)


def test_find_branch_worktree_normalizes_refs_heads_prefix() -> None:
    """``git worktree list --porcelain`` emits ``refs/heads/<name>`` values —
    the matcher must normalize that prefix. Detached entries (no branch line)
    never match."""
    worktrees = [
        {"worktree": "/repo", "branch": "refs/heads/main"},
        {
            "worktree": "/repo/.var/worktrees/agent-issue-5-x",
            "branch": "refs/heads/agent/issue-5-x",
        },
        {"worktree": "/repo/.var/worktrees/detached"},
    ]
    assert find_branch_worktree(worktrees, "main")["worktree"] == "/repo"
    assert find_branch_worktree(worktrees, "agent/issue-5-x")["worktree"] == (
        "/repo/.var/worktrees/agent-issue-5-x"
    )
    assert find_branch_worktree(worktrees, "missing/branch") is None


def test_find_branch_worktree_rejects_suffix_collision() -> None:
    """``team/agent/issue-5-x`` must NOT alias ``agent/issue-5-x``: the matcher
    also gates adoption now, so a suffix match would dispatch a worker into a
    checkout holding a different branch's work."""
    worktrees = [
        {"worktree": "/repo", "branch": "refs/heads/main"},
        {
            "worktree": "/repo/.claude/worktrees/team-agent-issue-5-x",
            "branch": "refs/heads/team/agent/issue-5-x",
        },
    ]
    assert find_branch_worktree(worktrees, "agent/issue-5-x") is None
    assert find_branch_worktree(worktrees, "team/agent/issue-5-x")["worktree"] == (
        "/repo/.claude/worktrees/team-agent-issue-5-x"
    )


def test_rework_does_not_adopt_branch_suffix_collision(tmp_path: Path) -> None:
    """End to end: a foreign checkout of ``team/agent/...`` is left untouched
    when the rework branch is ``agent/...`` — the create goes to the managed
    destination instead of borrowing the wrong checkout."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch = "agent/issue-1476-suffix"
    foreign_wt = tmp_path / "operator-worktree"
    _git(repo_root, "worktree", "add", str(foreign_wt), "-b", f"team/{branch}")
    _git(repo_root, "branch", branch)

    managed_dir = tmp_path / "managed"
    info = create_worktree(repo_root, branch, rework=True, worktrees_dir=managed_dir)

    assert info.path == managed_dir / _slugify(branch)
    assert info.foreign_adopted is False
    assert foreign_wt.is_dir()
    assert _git(foreign_wt, "branch", "--show-current").stdout.strip() == (f"team/{branch}")

    _git(repo_root, "worktree", "remove", str(foreign_wt), "--force")


def test_worktree_path_for_branch_prefers_registered_worktree(tmp_path: Path) -> None:
    """``worktree_path_for_branch`` locates the worktree actually serving the
    branch — including a foreign checkout — so every 'locate the worktree'
    caller (salvage, outcome reads, operator-claim markers, de-escalation
    probes) sees the adopted path. When nothing is registered it still returns
    the managed destination."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch = "agent/issue-1476-resolve"
    foreign_wt = tmp_path / "operator-worktree"
    _git(repo_root, "worktree", "add", str(foreign_wt), "-b", branch)

    managed_dir = tmp_path / "charlie-worktrees"
    assert worktree_path_for_branch(repo_root, branch, managed_dir) == foreign_wt
    assert worktree_path_for_branch(repo_root, "other/branch", managed_dir) == (
        managed_dir / "other-branch"
    )

    _git(repo_root, "worktree", "remove", str(foreign_wt), "--force")


def test_rework_foreign_adoption_emits_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adoption is observable: a ``worktree_foreign_adopted`` instrumentation
    event records which foreign path the dispatch borrowed."""
    emitted: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "charlie_work.instrumentation.log_event",
        lambda _state_file, kind, payload, **_kw: emitted.append((kind, payload)),
    )

    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch = "agent/issue-1476-event"
    foreign_wt = tmp_path / "operator-worktree"
    _git(repo_root, "worktree", "add", str(foreign_wt), "-b", branch)

    info = create_worktree(
        repo_root,
        branch,
        rework=True,
        worktrees_dir=tmp_path / "managed",
        issue_number=1476,
        config=OrchestratorConfig(),
    )

    assert info.foreign_adopted is True
    adoption_events = [p for kind, p in emitted if kind == "worktree_foreign_adopted"]
    assert len(adoption_events) == 1
    assert adoption_events[0]["worktree_path"] == str(foreign_wt)
    assert adoption_events[0]["issue_number"] == 1476

    _git(repo_root, "worktree", "remove", str(foreign_wt), "--force")


def test_merge_update_rework_branch_without_scaffolding_repair_escalates(
    tmp_path: Path,
) -> None:
    """``allow_scaffolding_repair=False`` — the value ``create_worktree``
    passes for an adopted foreign checkout — escalates even a
    scaffolding-named pre-merge blocker instead of deleting it: in a borrowed
    checkout the file belongs to the owner. The default repair path still
    clears the same fixture."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    _git(repo_root, "checkout", "-b", "feature")
    (repo_root / "work.txt").write_text("worker output\n", encoding="utf-8")
    _git(repo_root, "add", "work.txt")
    _git(repo_root, "commit", "-m", "feature work")

    _git(repo_root, "checkout", "main")
    (repo_root / ".orchestrator-prompt.md").write_text("base prompt v1\n", encoding="utf-8")
    _git(repo_root, "add", ".orchestrator-prompt.md")
    _git(repo_root, "commit", "-m", "base adds tracked prompt file")

    _git(repo_root, "checkout", "feature")
    # An untracked scaffolding-named file shadows the path the base now
    # tracks — normally repairable residue, but never in a borrowed checkout.
    (repo_root / ".orchestrator-prompt.md").write_text("owner's local copy\n", encoding="utf-8")

    injected = (".orchestrator-prompt.md",)
    with pytest.raises(ReworkBranchConflictError) as exc_info:
        _merge_update_rework_branch(
            repo_root,
            repo_root,
            "feature",
            "main",
            injected_paths=injected,
            allow_scaffolding_repair=False,
        )

    assert exc_info.value.stage == "pre_merge"
    assert ".orchestrator-prompt.md" in exc_info.value.conflicted_paths
    # Nothing was deleted — the owner's file is untouched.
    assert (repo_root / ".orchestrator-prompt.md").read_text(encoding="utf-8") == (
        "owner's local copy\n"
    )

    # Control on the same fixture: the managed-path default still repairs.
    result = _merge_update_rework_branch(
        repo_root, repo_root, "feature", "main", injected_paths=injected
    )
    assert result is None
    assert (repo_root / ".orchestrator-prompt.md").read_text(encoding="utf-8") == (
        "base prompt v1\n"
    )


def test_rework_adoption_creates_no_venv_junction(tmp_path: Path) -> None:
    """A borrowed checkout gets no scaffolding written into it: with
    ``venv_source`` set, a managed worktree would get a ``.venv`` junction —
    an adopted foreign checkout must not."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch = "agent/issue-1476-no-junction"
    foreign_wt = tmp_path / "operator-worktree"
    _git(repo_root, "worktree", "add", str(foreign_wt), "-b", branch)
    venv_source = tmp_path / "shared-venv"
    venv_source.mkdir()

    info = create_worktree(
        repo_root,
        branch,
        rework=True,
        worktrees_dir=tmp_path / "managed",
        venv_source=venv_source,
    )

    assert info.foreign_adopted is True
    assert info.venv_junction is None
    assert not (foreign_wt / ".venv").exists()
    assert not is_junction(foreign_wt / ".venv")

    _git(repo_root, "worktree", "remove", str(foreign_wt), "--force")


def test_rework_adoption_preserves_owner_venv_junction(tmp_path: Path) -> None:
    """Nor is the owner's own ``.venv`` junction removed: the managed path
    unlinks a leftover junction when ``venv_source`` is None, but a borrowed
    checkout's junction belongs to whoever created the checkout."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch = "agent/issue-1476-keep-junction"
    foreign_wt = tmp_path / "operator-worktree"
    _git(repo_root, "worktree", "add", str(foreign_wt), "-b", branch)
    owner_venv = tmp_path / "owner-venv"
    owner_venv.mkdir()
    _create_junction_or_symlink(foreign_wt / ".venv", owner_venv)

    info = create_worktree(
        repo_root,
        branch,
        rework=True,
        worktrees_dir=tmp_path / "managed",
        venv_source=None,
    )

    assert info.foreign_adopted is True
    assert info.venv_junction is None
    assert is_junction(foreign_wt / ".venv")

    _git(repo_root, "worktree", "remove", str(foreign_wt), "--force")
