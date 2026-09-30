"""Shared fixtures for the operator re-arm (``unescalate``) tests.

Hoisted out of ``test_fix_unescalate.py`` (issue #1284): a state.json
event filter and a minimal ``OrchestratorApp`` builder pointed at a
nonexistent post-mortem sessions.db (so real-activity probing is always
inconclusive), both imported by other test modules that exercise the same
escalation-recovery paths.
"""

from __future__ import annotations

from pathlib import Path

from _fakes_github import FakeGitHub
from charlie_work.config import OrchestratorConfig, PostMortemConfig
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp


def _events(state, kind: str) -> list[dict]:
    return [e for e in state.get("events", []) if e.get("kind") == kind]


def _app(tmp_path: Path) -> OrchestratorApp:
    # Isolate post_mortem.db_path from the real Devin sessions.db. The default
    # (db_path="") resolves to %APPDATA%\devin\cli\sessions.db at read time;
    # on a self-hosted CI runner that file exists with real session data, so
    # issue_worker_liveness's real-activity probe could surface a stale
    # timestamp for the test PID and flip the verdict from inconclusive-defer
    # (live=True, refuse) to conclusive-stale (live=False, proceed) -- dropping
    # the ``issue_worker_alive`` key the refusal branch sets. Pointing at a
    # nonexistent path under tmp_path makes every probe source error out
    # (inconclusive), which is the condition both #625 tests depend on.
    config = OrchestratorConfig(
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db"))
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    return OrchestratorApp(tmp_path, paths, config, fake_gh)


def _dry_run_app(tmp_path: Path) -> OrchestratorApp:
    """Same fixture as ``_app`` but constructed with ``dry_run=True``.

    Used by issue #1327's regression test: the de-escalation sweep must not
    clear ``status`` in ``state.json`` under dry-run, since the paired GitHub
    label transition is already gated at the sink and the two systems of
    record must stay synchronized.
    """
    config = OrchestratorConfig(
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db"))
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    return OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)


def _stranded_worktree_bed(
    tmp_path: Path, branch: str, *, origin: bool = True
) -> tuple[OrchestratorApp, Path, Path, Path | None]:
    """Repo + layout-path worktree holding ONE committed, never-pushed commit.

    Rule 3 (Worker fate) fixture: no worker-authored dirt, so
    ``_worktree_refuse_to_reset_reason`` returns the
    ``WORKTREE_UNSAFE_KIND_LOCAL_COMMITS`` reason. ``origin=True`` clones a
    bare remote (the branch does not exist on it yet); ``origin=False`` is a
    plain no-remote repo. Returns ``(app, repo_root, worktree_path, remote)``.
    """
    import pytest
    from _worktree_fixtures import _git, _init_bare_remote_and_clone, _init_repo

    from charlie_work.worktree import worktree_path_for_branch

    remote: Path | None = None
    if origin:
        remote, repo = _init_bare_remote_and_clone(tmp_path)
    else:
        repo = tmp_path / "repo"
        _init_repo(repo)
    app = _app(repo)
    wt_path = worktree_path_for_branch(repo, branch, app._layout.worktrees)
    # git builds the child worktree's .git path in a fixed-size buffer; a
    # deeply nested basetemp overflows it with no test-side remedy.
    if len(str(wt_path / ".git")) > 240:
        pytest.skip(f"worktree target path too deep for git's internal buffers: {wt_path}")
    wt_path.parent.mkdir(parents=True, exist_ok=True)
    _git(repo, "worktree", "add", str(wt_path), "-b", branch)
    _git(wt_path, "config", "user.email", "test@example.test")
    _git(wt_path, "config", "user.name", "Test User")
    (wt_path / "work.txt").write_text("stranded work\n", encoding="utf-8")
    _git(wt_path, "add", "work.txt")
    _git(wt_path, "commit", "-m", "stranded local-only commit")
    return app, repo, wt_path, remote


def _stranded_state(branch: str, *, reason: str = "worktree_unsafe_local_commits") -> dict:
    return {
        "issues": {
            "123": {
                "number": 123,
                "status": "escalated",
                "escalation_reason": reason,
                "branch_name": branch,
            }
        }
    }
