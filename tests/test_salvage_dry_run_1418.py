"""Issue #1418: dry-run threading tests for ``salvage_push_stranded_commits``.

Extracted from ``tests/test_charlie_work.py`` to satisfy the file-size ratchet
(issue #1442): the four regression tests for the two call sites that were not
threading ``dry_run`` into ``salvage_push_stranded_commits`` were added to the
already-over-cap ``test_charlie_work.py`` monolith, tripping the high-water-mark
ratchet.  Moving them here keeps the monolith at its recorded mark while
preserving the test coverage verbatim.

The helpers ``_init_repo_with_remote_inline`` and ``_wg`` are inlined here
(following the same self-contained pattern ``test_charlie_work.py`` established
when it inlined ``_init_repo_with_remote_inline`` instead of importing from
``test_worktree.py``) so this module stays self-contained.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from _fakes_github import FakeGitHub
from charlie_work.config import DevinConfig, OrchestratorConfig, WatchdogConfig, WorkerRoleConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp
from charlie_work.write_gate import WriteGate


def _wg(state_file: Path, *, dry_run: bool = False) -> WriteGate:
    return WriteGate(dry_run=dry_run, state_path=state_file, repo="charlie-work")


def _init_repo_with_remote_inline(tmp_path: Path) -> tuple[Path, Path]:
    """Create a bare origin remote + local clone with one commit on main.

    Inlined here (instead of importing from test_worktree.py) so this test
    module stays self-contained for the salvage-push regression tests.
    Returns ``(remote, repo_root)``.

    Mirrors ``test_worktree._init_repo(bare=True)``: a bare repo cannot
    receive commits directly, so a temporary non-bare repo is initialized
    with ``--initial-branch=main``, seeded with one commit, then cloned
    with ``--bare`` to produce the remote.
    """
    # Build a temp non-bare repo with one commit on main, then clone --bare.
    temp_repo = tmp_path / "remote-temp"
    temp_repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "--initial-branch=main"],
        cwd=temp_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.test"],
        cwd=temp_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=temp_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    (temp_repo / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=temp_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "initial commit"],
        cwd=temp_repo,
        check=True,
        capture_output=True,
        text=True,
    )

    remote = tmp_path / "remote"
    subprocess.run(
        ["git", "clone", "--bare", str(temp_repo), str(remote)],
        check=True,
        capture_output=True,
        text=True,
    )
    shutil.rmtree(temp_repo, ignore_errors=True)

    repo_root = tmp_path / "repo"
    subprocess.run(
        ["git", "clone", str(remote), str(repo_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.test"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return remote, repo_root


def test_reap_restore_rework_requested_salvage_threads_dry_run(tmp_path: Path) -> None:
    """Issue #1418: ``_reap_restore_rework_requested`` must thread the in-scope
    WriteGate's ``dry_run`` flag into ``salvage_push_stranded_commits``.  Under
    a ``dry_run=True`` gate the salvage call site must NOT issue a real
    ``git push`` to ``origin`` -- the same dry-run leak class fixed for the
    fresh-dispatch lane in #1326 (PR #1413).  This site was introduced on
    ``main`` by the #1239 salvage (#1392, commit 482be64) after #1413 opened,
    so it was out of scope for that PR.

    The test monkeypatches ``salvage_push_stranded_commits`` with a stand-in
    that records the ``dry_run`` kwarg and asserts it received ``dry_run=True``.
    The real remote branch head must NOT advance -- no push was issued.
    """
    import json
    from datetime import UTC, datetime

    from charlie_work.paths import resolved_layout
    from charlie_work.state import load_state, save_state, state_lock
    from charlie_work.worker import WorkerView
    from charlie_work.workflow import _reap_restore_rework_requested
    from charlie_work.worktree import SalvagePushResult, worktree_path_for_branch

    remote, repo_root = _init_repo_with_remote_inline(tmp_path)
    branch = "agent/issue-1418-reap-dry-run"

    run = lambda args: subprocess.run(  # noqa: E731
        args, cwd=repo_root, check=True, capture_output=True, text=True
    )

    run(["git", "branch", branch])
    config = OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
        worker=WorkerRoleConfig(harness="command"),
    )
    layout = resolved_layout(config, repo_root)
    wt_path = worktree_path_for_branch(repo_root, branch, layout.worktrees)
    wt_path.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "worktree", "add", str(wt_path), branch])

    # Push the branch so a real salvage would have something to advance.
    from charlie_work.worktree import push_branch

    ok, error = push_branch(repo_root, branch, worktree_path=wt_path)
    assert ok, error
    pr_head_sha = run(["git", "rev-parse", branch]).stdout.strip()

    # Add a stranded commit to the worktree (would be pushed by a real salvage).
    (wt_path / "fix.txt").write_text("fixed\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "fix.txt"],
        cwd=wt_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "completed rework (died before push)"],
        cwd=wt_path,
        check=True,
        capture_output=True,
        text=True,
    )

    paths = runtime_paths(repo_root, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [{"name": config.labels.in_progress}]
    fake_gh.prs[0]["headRefOid"] = pr_head_sha

    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "status": "dispatched",
            "worker_pid": 99999,
            "worker_process_start_time": 1234567890.0,
            "branch_name": branch,
        }
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": pr_head_sha,
        }
        save_state(paths.state_file, state)

    pr_decision_dir = paths.prs / "pr-456"
    pr_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "request_changes", "reviewed_head_sha": pr_head_sha}),
        encoding="utf-8",
    )

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-123.log"
    log_path.write_text("worker died\n", encoding="utf-8")

    worker = WorkerView(
        adapter_kind="devin",
        issue_number=123,
        repo_key="",
        pid=99999,
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        process_start_time=1234567890.0,
        log_path=str(log_path),
        worktree_path=str(wt_path),
        error=None,
        failure_kind="worker_died",
        reclaimed=None,
        branch=branch,
    )

    open_prs_by_issue = {123: [fake_gh.prs[0]]}

    salvage_calls: list[dict[str, object]] = []

    def fake_salvage(repo_root, branch, worktree_path, *, base_ref="", dry_run=False):
        salvage_calls.append({"branch": branch, "base_ref": base_ref, "dry_run": dry_run})
        return SalvagePushResult(pushed=False, skip_reason="dry_run_fake")

    # Issue #1317: _reap_restore_rework_requested moved verbatim to
    # dead_worker_reap.py, so its bare-name call to salvage_push_stranded_commits
    # resolves via that module's globals -- patch it there, not on workflow.py.
    import charlie_work.dead_worker_reap as reap_module

    with patch.object(reap_module, "salvage_push_stranded_commits", fake_salvage):
        _reap_restore_rework_requested(
            paths.state_file,
            fake_gh,
            config,
            open_prs_by_issue,
            worker,
            failure_kind="worker_died",
            repo_root=repo_root,
            write_gate=_wg(paths.state_file, dry_run=True),
        )

    assert len(salvage_calls) == 1, salvage_calls
    assert salvage_calls[0]["dry_run"] is True, salvage_calls[0]

    # The real remote branch head MUST NOT have advanced -- no push was issued.
    remote_sha = subprocess.run(
        ["git", "rev-parse", branch],
        cwd=remote,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert remote_sha == pr_head_sha, (
        f"dry-run salvage issued a real push: remote {remote_sha} != {pr_head_sha}"
    )


def test_salvage_rework_stranded_commits_threads_dry_run(tmp_path: Path) -> None:
    """Issue #1418: ``OrchestratorApp._salvage_rework_stranded_commits`` must
    thread ``self.write_gate.dry_run`` into ``salvage_push_stranded_commits``.
    Under a ``dry_run=True`` app the death-loop salvage call site must NOT
    issue a real ``git push`` to ``origin`` -- the same dry-run leak class
    fixed for the fresh-dispatch lane in #1326 (PR #1413).  This site was
    introduced on ``main`` by the #1239 salvage (#1392, commit 482be64) after
    #1413 opened, so it was out of scope for that PR.

    The test constructs the app with ``dry_run=True`` and monkeypatches
    ``salvage_push_stranded_commits`` with a stand-in that records the
    ``dry_run`` kwarg, asserting it received ``dry_run=True``.  The real
    remote branch head must NOT advance -- no push was issued.
    """
    from datetime import UTC, datetime

    from charlie_work.paths import resolved_layout
    from charlie_work.worktree import (
        SalvagePushResult,
        push_branch,
        worktree_path_for_branch,
    )

    remote, repo_root = _init_repo_with_remote_inline(tmp_path)
    branch = "agent/issue-1418-rework-dry-run"

    run = lambda args: subprocess.run(  # noqa: E731
        args, cwd=repo_root, check=True, capture_output=True, text=True
    )

    run(["git", "branch", branch])
    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "print('ok')")),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
        worker=WorkerRoleConfig(harness="command"),
    )
    layout = resolved_layout(config, repo_root)
    wt_path = worktree_path_for_branch(repo_root, branch, layout.worktrees)
    wt_path.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "worktree", "add", str(wt_path), branch])

    ok, error = push_branch(repo_root, branch, worktree_path=wt_path)
    assert ok, error
    pr_head_sha = run(["git", "rev-parse", branch]).stdout.strip()

    # Add a stranded commit to the worktree (would be pushed by a real salvage).
    (wt_path / "fix.txt").write_text("fixed\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "fix.txt"],
        cwd=wt_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "completed rework (died before push)"],
        cwd=wt_path,
        check=True,
        capture_output=True,
        text=True,
    )

    paths = runtime_paths(repo_root, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.repo_root = repo_root
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]
            self.prs[0]["headRefOid"] = pr_head_sha

    fake_gh = ReworkGitHub()
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
            "redispatch_at": [now_iso, now_iso],
            "worker_death_at": [now_iso, now_iso],
            "branch_name": branch,
        }
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": pr_head_sha,
        }
        save_state(paths.state_file, state)

    # Construct the app with dry_run=True so self.write_gate.dry_run is True.
    app = OrchestratorApp(repo_root, paths, config, fake_gh, dry_run=True)

    issue_entry = {
        "number": 123,
        "status": "rework_requested",
        "branch_name": branch,
    }
    pr_data = fake_gh.prs[0]

    salvage_calls: list[dict[str, object]] = []

    def fake_salvage(repo_root, branch, worktree_path, *, base_ref="", dry_run=False):
        salvage_calls.append({"branch": branch, "base_ref": base_ref, "dry_run": dry_run})
        return SalvagePushResult(pushed=False, skip_reason="dry_run_fake")

    import charlie_work.workflow as workflow_module

    with patch.object(workflow_module, "salvage_push_stranded_commits", fake_salvage):
        result = app._salvage_rework_stranded_commits(123, pr_data, issue_entry)

    # The salvage stand-in reported no push, so the method returns False.
    assert result is False
    assert len(salvage_calls) == 1, salvage_calls
    assert salvage_calls[0]["dry_run"] is True, salvage_calls[0]

    # The real remote branch head MUST NOT have advanced -- no push was issued.
    remote_sha = subprocess.run(
        ["git", "rev-parse", branch],
        cwd=remote,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert remote_sha == pr_head_sha, (
        f"dry-run salvage issued a real push: remote {remote_sha} != {pr_head_sha}"
    )


def test_push_branch_and_salvage_dry_run_no_mock_exercises_short_circuit(
    tmp_path: Path,
) -> None:
    """Issue #1418 rework: a direct test that does NOT mock
    ``push_branch`` or ``salvage_push_stranded_commits``.

    The two call-site tests above monkeypatch ``salvage_push_stranded_commits``
    to verify the call sites thread ``dry_run``.  That leaves the real dry-run
    short-circuit logic in ``worktree.py`` unexercised by this PR's own diff.
    This test calls both functions directly with ``dry_run=True`` against a real
    local git repo (bare "origin" + clone + linked worktree) and asserts:

    1. ``push_branch(..., dry_run=True)`` returns ``(True, None)`` and the
       branch does NOT appear on origin afterward.
    2. ``salvage_push_stranded_commits(..., dry_run=True)`` returns
       ``SalvagePushResult(pushed=True, ...)`` (the "nothing happened" success
       shape that lets downstream classification proceed under dry-run) and the
       remote branch tip is unchanged.

    No ``git push`` subprocess fires in either case — verified by checking the
    bare remote's refs before and after each call.
    """
    from charlie_work.worktree import SalvagePushResult, push_branch, salvage_push_stranded_commits

    remote, repo_root = _init_repo_with_remote_inline(tmp_path)
    branch = "agent/issue-1418-direct-dry-run"

    run = lambda args: subprocess.run(  # noqa: E731
        args, cwd=repo_root, check=True, capture_output=True, text=True
    )

    run(["git", "branch", branch])
    run(["git", "worktree", "add", str(tmp_path / "wt"), branch])
    wt_path = tmp_path / "wt"

    # Add a stranded commit in the worktree (never pushed to origin).
    (wt_path / "feature.txt").write_text("feature\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "feature.txt"],
        cwd=wt_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "stranded commit, never pushed"],
        cwd=wt_path,
        check=True,
        capture_output=True,
        text=True,
    )
    local_tip = run(["git", "rev-parse", branch]).stdout.strip()

    def _remote_refs() -> str:
        return subprocess.run(
            ["git", "show-ref"],
            cwd=remote,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    # --- push_branch(dry_run=True) ---
    show_ref_before = _remote_refs()
    assert branch not in show_ref_before

    ok, error = push_branch(repo_root, branch, worktree_path=wt_path, dry_run=True)

    # Dry-run returns the natural "nothing happened" success shape.
    assert ok is True
    assert error is None

    # The branch must STILL NOT exist on origin -- no real push happened.
    show_ref_after = _remote_refs()
    assert branch not in show_ref_after

    # --- salvage_push_stranded_commits(dry_run=True) ---
    # The branch still does not exist on origin, so salvage takes the
    # "branch never made it to origin" path: it probes the base branch,
    # counts commits beyond it, then calls push_branch(dry_run=True).
    result = salvage_push_stranded_commits(repo_root, branch, wt_path, dry_run=True)

    assert isinstance(result, SalvagePushResult)
    # Dry-run: push_branch returned (True, None), so salvage reports pushed=True
    # with the local tip as the would-be new remote SHA -- but no real push
    # reached origin.
    assert result.pushed is True
    assert result.error is None
    assert result.new_remote_sha == local_tip
    assert result.commit_count is not None and result.commit_count >= 1

    # The branch must STILL NOT exist on origin -- no real push happened.
    show_ref_final = _remote_refs()
    assert branch not in show_ref_final


def test_salvage_rework_stranded_commits_dry_run_no_mock_end_to_end(
    tmp_path: Path,
) -> None:
    """Issue #1418 rework: end-to-end test through the real call site.

    Unlike ``test_salvage_rework_stranded_commits_threads_dry_run`` (which
    mocks ``salvage_push_stranded_commits``), this test lets the real function
    run.  It constructs an ``OrchestratorApp`` with ``dry_run=True``, sets up a
    real git repo with a pushed branch and a stranded commit in the worktree,
    and calls ``app._salvage_rework_stranded_commits`` without any monkeypatch.
    The real ``salvage_push_stranded_commits`` runs with ``dry_run=True``
    (threaded by the call site fixed in this PR), reaches ``push_branch`` which
    short-circuits before ``git push``, and the method returns ``True`` -- but
    the remote branch tip MUST NOT have advanced.
    """
    from datetime import UTC, datetime

    from charlie_work.paths import resolved_layout
    from charlie_work.state import load_state, save_state, state_lock
    from charlie_work.worktree import push_branch, worktree_path_for_branch

    remote, repo_root = _init_repo_with_remote_inline(tmp_path)
    branch = "agent/issue-1418-e2e-dry-run"

    run = lambda args: subprocess.run(  # noqa: E731
        args,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    run(["git", "branch", branch])
    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "print('ok')")),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
        worker=WorkerRoleConfig(harness="command"),
    )
    layout = resolved_layout(config, repo_root)
    wt_path = worktree_path_for_branch(repo_root, branch, layout.worktrees)
    wt_path.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "worktree", "add", str(wt_path), branch])

    # Push the branch so salvage has a real remote tip to probe.
    ok, error = push_branch(repo_root, branch, worktree_path=wt_path)
    assert ok, error
    pr_head_sha = run(["git", "rev-parse", branch]).stdout.strip()

    # Add a stranded commit to the worktree (would be pushed by a real salvage).
    (wt_path / "fix.txt").write_text("fixed\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "fix.txt"],
        cwd=wt_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "completed rework (died before push)"],
        cwd=wt_path,
        check=True,
        capture_output=True,
        text=True,
    )

    paths = runtime_paths(repo_root, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.repo_root = repo_root
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]
            self.prs[0]["headRefOid"] = pr_head_sha

    fake_gh = ReworkGitHub()
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
            "redispatch_at": [now_iso, now_iso],
            "worker_death_at": [now_iso, now_iso],
            "branch_name": branch,
        }
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": pr_head_sha,
        }
        save_state(paths.state_file, state)

    # Construct the app with dry_run=True so self.write_gate.dry_run is True.
    app = OrchestratorApp(repo_root, paths, config, fake_gh, dry_run=True)

    issue_entry = {
        "number": 123,
        "status": "rework_requested",
        "branch_name": branch,
    }
    pr_data = fake_gh.prs[0]

    # NO mock -- the real salvage_push_stranded_commits runs with dry_run=True.
    result = app._salvage_rework_stranded_commits(123, pr_data, issue_entry)

    # The real salvage returned pushed=True (dry-run "nothing happened" shape),
    # so the method returns True -- but no real push reached origin.
    assert result is True

    # The real remote branch head MUST NOT have advanced -- no push was issued.
    remote_sha = subprocess.run(
        ["git", "rev-parse", branch],
        cwd=remote,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert remote_sha == pr_head_sha, (
        f"dry-run salvage issued a real push: remote {remote_sha} != {pr_head_sha}"
    )
