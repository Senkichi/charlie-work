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
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from _worktree_fixtures import _clone_repo, _git, _init_repo

from charlie_work import claude_code, devin_shell
from charlie_work.claude_code import launch_claude_worker
from charlie_work.config import OrchestratorConfig
from charlie_work.devin_shell import launch_devin_session
from charlie_work.worktree import (
    RESCUE_REF_PREFIX,
    WorktreeForeignWriterError,
    WorktreeInfo,
    create_worktree,
    worktree_path_for_branch,
    write_worktree_marker,
)


def _clone_with_pushed_branch(tmp_path: Path, branch: str) -> tuple[Path, Path]:
    """Bare origin + clone + ``branch`` pushed to origin; return (remote, repo)."""
    remote = tmp_path / "remote.git"
    _init_repo(remote, bare=True)
    repo_root = tmp_path / "repo"
    _clone_repo(remote, repo_root)
    _git(repo_root, "branch", branch)
    _git(repo_root, "push", "origin", branch)
    return remote, repo_root


def _push_sibling_commit(remote: Path, tmp_path: Path, branch: str, filename: str) -> str:
    """Advance ``origin/<branch>`` by one commit from a second clone; return the
    pushed SHA."""
    other = tmp_path / "other-clone"
    if not other.exists():
        _clone_repo(remote, other)
        _git(other, "checkout", "-b", branch, f"origin/{branch}")
    else:
        _git(other, "fetch", "origin", branch)
        _git(other, "reset", "--hard", f"origin/{branch}")
    (other / filename).write_text(f"{filename}\n", encoding="utf-8")
    _git(other, "add", filename)
    _git(other, "commit", "-m", f"remote advance {filename}")
    pushed_sha = _git(other, "rev-parse", "HEAD").stdout.strip()
    _git(other, "push", "origin", branch)
    return pushed_sha


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


def test_rework_recovery_refuses_foreign_worktree_without_marker(tmp_path: Path) -> None:
    """Recovery semantics presume the leftover worktree is ours — its dirt is a
    dead worker's partial work to continue from. With no writer marker proving
    a prior orchestrator session ran there, that presumption is false."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch = "agent/issue-1476-recovery"
    foreign_wt = tmp_path / "operator-worktree"
    _git(repo_root, "worktree", "add", str(foreign_wt), "-b", branch)
    (foreign_wt / "partial.txt").write_text("foreign edits\n", encoding="utf-8")

    recovery = {"branch_name": branch}
    with pytest.raises(WorktreeForeignWriterError) as exc_info:
        create_worktree(
            repo_root,
            branch,
            rework=True,
            recovery=recovery,
            worktrees_dir=tmp_path / "managed",
        )

    assert exc_info.value.worktree_path == foreign_wt
    assert (foreign_wt / "partial.txt").read_text(encoding="utf-8") == "foreign edits\n"


def test_rework_recovery_adopts_marked_foreign_worktree(tmp_path: Path) -> None:
    """A writer marker is the durable proof a prior orchestrator session owned
    the checkout, so recovery may continue the dead worker's dirt there — the
    stale marker is cleaned by the ordinary marker guard."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch = "agent/issue-1476-recovery-marked"
    foreign_wt = tmp_path / "operator-worktree"
    _git(repo_root, "worktree", "add", str(foreign_wt), "-b", branch)
    (foreign_wt / "partial.txt").write_text("dead worker's partial work\n", encoding="utf-8")

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    # A dead worker's leftover marker: pid no longer alive, session unknown to
    # the sidecar dir — proof of prior orchestrator occupancy, not a live writer.
    write_worktree_marker(foreign_wt, 99999999, "dead-session")

    recovery = {"branch_name": branch}
    info = create_worktree(
        repo_root,
        branch,
        rework=True,
        recovery=recovery,
        worktrees_dir=tmp_path / "managed",
        sessions_dir=sessions_dir,
    )

    assert info.path == foreign_wt
    assert info.foreign_adopted is True
    assert (foreign_wt / "partial.txt").read_text(encoding="utf-8") == (
        "dead worker's partial work\n"
    )


def test_recovery_dispatch_routes_to_foreign_adoption(tmp_path: Path) -> None:
    """The load-bearing path: a dead-worker re-dispatch (``rework=False`` +
    recovery record) that finds the branch checked out foreign must route
    through the adoption gate — not the `git branch -D` restart path, which
    always fails on a checked-out branch. With a worker marker proving prior
    orchestrator ownership, recovery adopts and continues."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch = "agent/issue-1476-recovery-route"
    foreign_wt = tmp_path / "operator-worktree"
    _git(repo_root, "worktree", "add", str(foreign_wt), "-b", branch)
    (foreign_wt / "partial.txt").write_text("dead worker's partial work\n", encoding="utf-8")

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    write_worktree_marker(foreign_wt, 99999999, "dead-session")

    info = create_worktree(
        repo_root,
        branch,
        rework=False,
        recovery={"branch_name": branch},
        worktrees_dir=tmp_path / "managed",
        sessions_dir=sessions_dir,
    )

    assert info.path == foreign_wt
    assert info.foreign_adopted is True
    assert (foreign_wt / "partial.txt").read_text(encoding="utf-8") == (
        "dead worker's partial work\n"
    )

    _git(repo_root, "worktree", "remove", str(foreign_wt), "--force")


def test_recovery_dispatch_refuses_unmarked_foreign(tmp_path: Path) -> None:
    """Same routing, opposite verdict: without a worker writer marker the
    foreign checkout cannot be proven to be ours, so the recovery re-dispatch
    refuses instead of crashing on `git branch -D`."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch = "agent/issue-1476-recovery-refuse"
    foreign_wt = tmp_path / "operator-worktree"
    _git(repo_root, "worktree", "add", str(foreign_wt), "-b", branch)

    with pytest.raises(WorktreeForeignWriterError) as exc_info:
        create_worktree(
            repo_root,
            branch,
            rework=False,
            recovery={"branch_name": branch},
            worktrees_dir=tmp_path / "managed",
        )

    assert exc_info.value.worktree_path == foreign_wt
    assert "ownership" in str(exc_info.value)

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


def test_claude_launch_failure_preserves_adopted_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The claude adapter's launch-failure teardown calls ``remove_worktree``
    on every managed worktree — an adopted foreign checkout must be exempt:
    the checkout belongs to whoever created it."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    foreign_wt = tmp_path / "foreign-wt"
    foreign_wt.mkdir()

    adopted = WorktreeInfo(
        path=foreign_wt, branch="agent/issue-1476", venv_junction=None, foreign_adopted=True
    )
    monkeypatch.setattr(claude_code, "create_worktree", lambda *a, **k: adopted)
    removed: list[Path] = []
    monkeypatch.setattr(
        claude_code, "remove_worktree", lambda _root, wt, **_kw: removed.append(wt) or True
    )

    record = launch_claude_worker(
        1476,
        "agent/issue-1476",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=("this-binary-does-not-exist-xyz",),
        rework=True,
    )

    assert not record.ok
    assert removed == []
    assert foreign_wt.is_dir()


def test_devin_launch_failure_preserves_adopted_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same ownership guarantee for the devin adapter's ``_teardown_worktree``."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    foreign_wt = tmp_path / "foreign-wt"
    foreign_wt.mkdir()
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("prompt\n", encoding="utf-8")

    adopted = WorktreeInfo(
        path=foreign_wt, branch="agent/issue-1476", venv_junction=None, foreign_adopted=True
    )
    monkeypatch.setattr(devin_shell, "create_worktree", lambda *a, **k: adopted)
    removed: list[Path] = []
    monkeypatch.setattr(
        devin_shell, "remove_worktree", lambda _root, wt, **_kw: removed.append(wt) or True
    )

    record = launch_devin_session(
        1476,
        "agent/issue-1476",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=("this-binary-does-not-exist-xyz",),
        rework=True,
    )

    assert record.error is not None
    assert record.pid is None
    assert removed == []
    assert foreign_wt.is_dir()


def test_devin_launch_failure_still_removes_managed_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: the adoption exemption must not leak — a managed worktree
    (``foreign_adopted=False``) is still torn down on launch failure."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    managed_wt = tmp_path / "managed-wt"
    managed_wt.mkdir()
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("prompt\n", encoding="utf-8")

    managed = WorktreeInfo(path=managed_wt, branch="agent/issue-1476", venv_junction=None)
    monkeypatch.setattr(devin_shell, "create_worktree", lambda *a, **k: managed)
    removed: list[Path] = []
    monkeypatch.setattr(
        devin_shell, "remove_worktree", lambda _root, wt, **_kw: removed.append(wt) or True
    )

    record = launch_devin_session(
        1476,
        "agent/issue-1476",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=("this-binary-does-not-exist-xyz",),
        rework=True,
    )

    assert record.error is not None
    assert removed == [managed_wt]
