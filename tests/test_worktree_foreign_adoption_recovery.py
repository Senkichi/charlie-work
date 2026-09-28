"""Issue #1476: recovery-mode adoption of a foreign checkout, and teardown.

Split out of ``tests/test_worktree_foreign_adoption.py`` under the #1476
rework — the file-size ratchet (issue #1442) caps modules at 800 lines, so the
recovery/teardown/probe half lives here while the ordinary-rework gate stays
in the original module. Shared remote/clone helpers
(``_clone_with_pushed_branch``, ``_push_sibling_commit``) moved to
``tests/_worktree_fixtures.py``.

Recovery semantics presume a leftover worktree is ours — its dirt is a dead
worker's partial work to continue from. For a foreign checkout that
presumption only holds when the checkout carries a worker-kind writer marker
proving prior orchestrator ownership; anything else is refused, and an adopted
checkout is exempt from every teardown path that would delete it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from _sessions_db_fixtures import make_sessions_db
from _worktree_fixtures import (
    _clone_repo,
    _clone_with_pushed_branch,
    _git,
    _init_repo,
    _push_sibling_commit,
)

from charlie_work import claude_code, devin_shell
from charlie_work import worktree as worktree_mod
from charlie_work.claude_code import launch_claude_worker
from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    PostMortemConfig,
    WorkerRoleConfig,
)
from charlie_work.devin_shell import launch_devin_session
from charlie_work.subprocess_runner import RunResult
from charlie_work.worktree import (
    OPERATOR_MARKER_KIND,
    OPERATOR_MARKER_SESSION_ID,
    LiveWorkerRedispatchError,
    WorktreeForeignWriterError,
    WorktreeInfo,
    create_worktree,
    read_worktree_marker,
    write_worktree_marker,
)


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


@pytest.mark.parametrize(
    "marker_kwargs",
    [
        pytest.param(
            {
                "pid": 0,
                "session_id": OPERATOR_MARKER_SESSION_ID,
                "kind": OPERATOR_MARKER_KIND,
            },
            id="operator-kind-marker",
        ),
        pytest.param(
            {"pid": 4242, "session_id": "operator-9f8e7d"},
            id="operator-session-id",
        ),
        pytest.param(
            {"pid": 0, "session_id": "worker-session"},
            id="nonpositive-pid",
        ),
    ],
)
def test_rework_recovery_refuses_non_worker_marker(tmp_path: Path, marker_kwargs: dict) -> None:
    """Only a worker-kind marker proves prior orchestrator ownership. An
    operator claim marker, an ``operator-*`` session id, and a nonpositive pid
    each leave the checkout foreign — recovery must refuse rather than treat
    someone else's dirt as a dead worker's partial work."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch = "agent/issue-1476-marker-kind"
    foreign_wt = tmp_path / "operator-worktree"
    _git(repo_root, "worktree", "add", str(foreign_wt), "-b", branch)
    (foreign_wt / "partial.txt").write_text("foreign edits\n", encoding="utf-8")
    write_worktree_marker(foreign_wt, **marker_kwargs)

    with pytest.raises(WorktreeForeignWriterError) as exc_info:
        create_worktree(
            repo_root,
            branch,
            rework=True,
            recovery={"branch_name": branch},
            worktrees_dir=tmp_path / "managed",
        )

    assert exc_info.value.worktree_path == foreign_wt
    assert "ownership" in str(exc_info.value)
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


def test_recovery_liveness_probe_targets_foreign_checkout_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recovery liveness probe must inspect the checkout the prior worker
    actually ran in. ``real_activity_for_worker`` keys both its sessions.db
    lookup and its file-mtime check on the worktree path it is given; a branch
    checked out in a foreign worktree records that activity under the foreign
    path, so probing the managed slug path reports a permanently-absent worker
    while the real one is still moving — re-opening the #282
    dispatch-over-a-live-worker hole on the recovery route this issue added.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch = "agent/issue-1476-probe-path"
    foreign_wt = tmp_path / "operator-worktree"
    _git(repo_root, "worktree", "add", str(foreign_wt), "-b", branch)
    (foreign_wt / "partial.txt").write_text("worker output\n", encoding="utf-8")

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    # Dead worker's leftover marker — proof of prior orchestrator occupancy.
    write_worktree_marker(foreign_wt, 99999999, "dead-session")

    # Fresh sessions.db activity keyed at the FOREIGN path, where the prior
    # worker ran — invisible to a probe pointed at the managed slug path.
    db_path = tmp_path / "sessions.db"
    now = datetime.now(UTC).isoformat()
    make_sessions_db(
        db_path,
        session_id="session-1",
        working_directory=str(foreign_wt),
        created_at=now,
        rows=[{"role": "tool", "content": "tool result", "created_at": now}],
    )

    probed_paths: list[str] = []
    real_probe = worktree_mod.real_activity_for_worker

    def recording_probe(pm_config, worktree_path, *args, **kwargs):  # type: ignore[no-untyped-def]
        probed_paths.append(worktree_path)
        return real_probe(pm_config, worktree_path, *args, **kwargs)

    monkeypatch.setattr(worktree_mod, "real_activity_for_worker", recording_probe)

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        post_mortem=PostMortemConfig(db_path=str(db_path)),
    )

    with pytest.raises(LiveWorkerRedispatchError) as exc_info:
        create_worktree(
            repo_root,
            branch,
            rework=False,
            recovery={
                "branch_name": branch,
                "status": "dispatched",
                "worker_pid": 999999,
                "worker_process_start_time": 0.0,
                "started_at": now,
            },
            worktrees_dir=tmp_path / "managed",
            sessions_dir=sessions_dir,
            config=config,
        )

    # The probe was pointed at the foreign checkout — the path the prior
    # worker actually ran in — and its fresh activity aborted the redispatch.
    assert probed_paths == [str(foreign_wt)]
    assert exc_info.value.probe_result.endswith("_activity")

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


def test_recovery_adoption_keeps_marker_after_post_gate_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dead worker's writer marker is the only proof a foreign checkout is
    ours, so it must survive a post-gate failure. The marker guard cleans a
    dead-pid marker at gate time; if a later step then fails — the rework
    fetch here, or a launch failure whose teardown skips borrowed checkouts —
    a markerless retry is refused as "cannot prove prior orchestrator
    ownership" and the #1476 stuck loop returns. Cleanup is deferred to the
    post-launch marker overwrite, so the second recovery call still adopts."""
    branch = "agent/issue-1476-marker-durable"
    _remote, repo_root = _clone_with_pushed_branch(tmp_path, branch)
    foreign_wt = tmp_path / "operator-worktree"
    _git(repo_root, "worktree", "add", str(foreign_wt), branch)

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    # Dead worker's leftover marker — proof of prior orchestrator occupancy.
    write_worktree_marker(foreign_wt, 99999999, "dead-session")

    # Fail only the rework fetch — a post-gate failure AFTER the marker guard
    # has already run. ls-remote still works so the branch resolves on origin.
    fetch_fails = {"enabled": True}
    real_run_remote = worktree_mod._run_remote_captured

    def flaky_remote(command: list[str], cwd: Path, **kwargs: object) -> RunResult:
        if fetch_fails["enabled"] and command[:3] == ["git", "fetch", "origin"]:
            return RunResult(
                returncode=128,
                stdout="",
                stderr="fatal: simulated fetch outage",
                error="command exited 128",
            )
        return real_run_remote(command, cwd, **kwargs)

    monkeypatch.setattr(worktree_mod, "_run_remote_captured", flaky_remote)

    with pytest.raises(RuntimeError, match="Fetch failed"):
        create_worktree(
            repo_root,
            branch,
            rework=False,
            recovery={"branch_name": branch},
            worktrees_dir=tmp_path / "managed",
            sessions_dir=sessions_dir,
        )

    # The ownership proof survived the failed pass.
    assert read_worktree_marker(foreign_wt) is not None

    fetch_fails["enabled"] = False
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

    _git(repo_root, "worktree", "remove", str(foreign_wt), "--force")


def test_recovery_routes_unpushed_branch_at_foreign_path_to_rework(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recovery with an origin remote where the branch is ABSENT from origin
    (killed before first push) but checked out at a foreign path is routed to
    the rework adoption gate — and the rework fetch is skipped because the
    branch provably has no origin ref to fetch."""
    remote = tmp_path / "remote.git"
    _init_repo(remote, bare=True)
    repo_root = tmp_path / "repo"
    _clone_repo(remote, repo_root)
    branch = "agent/issue-1476-unpushed"
    _git(repo_root, "branch", branch)  # local-only: never pushed to origin
    foreign_wt = tmp_path / "operator-worktree"
    _git(repo_root, "worktree", "add", str(foreign_wt), branch)

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    write_worktree_marker(foreign_wt, 99999999, "dead-session")

    remote_calls: list[list[str]] = []
    real_run_remote = worktree_mod._run_remote_captured

    def recording_remote(command: list[str], cwd: Path, **kwargs: object) -> RunResult:
        remote_calls.append(list(command))
        return real_run_remote(command, cwd, **kwargs)

    monkeypatch.setattr(worktree_mod, "_run_remote_captured", recording_remote)

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
    assert info.reclaimed == "fetch-fallback"
    # The ls-remote probe ran (that is how remote_exists became False) but no
    # fetch was attempted — fetching a branch absent from origin hard-fails
    # and would re-create the #1476 retry loop for an adopted checkout.
    assert ["git", "ls-remote", "origin", f"refs/heads/{branch}"] in remote_calls
    assert ["git", "fetch", "origin", branch] not in remote_calls

    _git(repo_root, "worktree", "remove", str(foreign_wt), "--force")


def test_recovery_unpushed_branch_at_foreign_path_refused_unmarked(
    tmp_path: Path,
) -> None:
    """Same routing as the marked case, opposite verdict: with no worker
    marker the foreign checkout cannot be proven ours, so the remote-absent
    recovery route refuses instead of falling through to `git branch -D`
    (which fails on a checked-out branch)."""
    remote = tmp_path / "remote.git"
    _init_repo(remote, bare=True)
    repo_root = tmp_path / "repo"
    _clone_repo(remote, repo_root)
    branch = "agent/issue-1476-unpushed-refuse"
    _git(repo_root, "branch", branch)  # local-only: never pushed to origin
    foreign_wt = tmp_path / "operator-worktree"
    _git(repo_root, "worktree", "add", str(foreign_wt), branch)

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


def test_recovery_resets_diverged_marked_foreign_checkout(tmp_path: Path) -> None:
    """The one path where a foreign checkout is hard-reset: recovery adoption
    of a marked checkout whose branch diverged non-FF from origin. Recovery
    treats the checkout as ours (the worker marker is the ownership proof),
    snapshots the pre-reset tip, and resets to origin — where the same
    divergence in a non-recovery adoption refuses
    (``test_rework_refuses_diverged_foreign_worktree``)."""
    branch = "agent/issue-1476-recovery-diverged"
    remote, repo_root = _clone_with_pushed_branch(tmp_path, branch)
    foreign_wt = tmp_path / "operator-worktree"
    _git(repo_root, "worktree", "add", str(foreign_wt), branch)

    # Diverge: a local commit in the foreign checkout plus a different commit
    # pushed to the same branch from a second clone.
    (foreign_wt / "local.txt").write_text("dead worker commit\n", encoding="utf-8")
    _git(foreign_wt, "add", "local.txt")
    _git(foreign_wt, "commit", "-m", "dead worker local commit")
    local_sha = _git(foreign_wt, "rev-parse", "HEAD").stdout.strip()
    remote_tip = _push_sibling_commit(remote, tmp_path, branch, "remote.txt")

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
        issue_number=1476,
        config=OrchestratorConfig(),
    )

    assert info.path == foreign_wt
    assert info.foreign_adopted is True
    assert _git(foreign_wt, "rev-parse", "HEAD").stdout.strip() == remote_tip
    assert info.reclaimed is not None
    assert info.reclaimed.startswith("reset-origin:")
    # The attempt snapshot preserved the diverged local tip before the reset.
    assert info.attempt_snapshot is not None
    assert info.attempt_snapshot.old_tip == local_sha
    assert info.attempt_snapshot.ref_name is not None
    assert _git(repo_root, "rev-parse", info.attempt_snapshot.ref_name).stdout.strip() == local_sha

    _git(repo_root, "worktree", "remove", str(foreign_wt), "--force")
