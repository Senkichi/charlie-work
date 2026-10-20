"""Shared builders for the dead-worker sweep characterization tests.

Hoisted out of ``tests/test_dead_worker_sweep_characterization.py`` (the
``tests/_*.py`` hoisted-fixture convention: test modules never import each
other). Two beds cover the sweep's two families:

- ``no_pr_bed``: a dead-PID ``dispatched`` issue with NO open PR (the
  reclaim / escalate / pushed-branch family);
- ``_orphan_sweep_fixtures._dead_worker_rework_bed``: the same issue with an
  open PR carrying a verdict (the with-PR family), reused as-is.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from _fakes_github import FakeGitHub
from _host_fixtures import host_probe
from _rework_dispatch_fixtures import _wg
from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import resolved_layout, runtime_paths
from charlie_work.state import load_state, save_state
from charlie_work.worktree import worktree_path_for_branch


class NoPrGitHub(FakeGitHub):
    """FakeGitHub whose open-PR listing is always empty."""

    def pr_list(self):
        return []


def iso(dt: datetime | None = None, *, minutes_ago: int = 0) -> str:
    base = dt or datetime.now(UTC)
    return (base - timedelta(minutes=minutes_ago)).isoformat().replace("+00:00", "Z")


def sweep_config(**watchdog: Any) -> OrchestratorConfig:
    return OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, **watchdog),
    )


def sessions_dir_for(tmp_path: Path) -> Path:
    path = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_issue(paths: Any, number: int, **fields: Any) -> None:
    """Merge ``fields`` into a (possibly new) dead-PID ``dispatched`` entry."""
    state = load_state(paths.state_file)
    entry = state["issues"].get(str(number)) or {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    state["issues"][str(number)] = {**entry, **fields}
    save_state(paths.state_file, state)


def no_pr_bed(
    tmp_path: Path,
    number: int,
    *,
    title: str = "test issue",
    labels: tuple[str, ...] | None = None,
    repo_root: Path | None = None,
    config: OrchestratorConfig | None = None,
    **entry_fields: Any,
) -> tuple[OrchestratorConfig, Any, NoPrGitHub]:
    """Dead ``dispatched`` issue ``number`` with no open PR.

    ``labels`` defaults to ``(in_progress,)`` (the active label the reclaim
    strips); the entry is seeded via ``seed_issue`` with ``entry_fields``.
    """
    config = config or sweep_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    seed_issue(paths, number, **entry_fields)
    gh = NoPrGitHub(repo_root=repo_root or tmp_path)
    names = (config.labels.in_progress,) if labels is None else labels
    gh.issues = [
        {
            "number": number,
            "title": title,
            "url": f"https://example.test/issues/{number}",
            "body": "",
            "labels": [{"name": n} for n in names],
            "state": "OPEN",
        }
    ]
    gh.prs = []
    return config, paths, gh


def run_sweep(
    tmp_path: Path,
    paths: Any,
    config: OrchestratorConfig,
    gh: Any,
    monkeypatch,
    *,
    review_callback: Any = None,
    pid_alive: bool = False,
    dry_run: bool = False,
    fleet_dir: Path | None = None,
    patches: tuple[Any, ...] = (),
) -> None:
    """Drive the real ``_detect_and_handle_orphaned_workers`` once.

    ``patches`` are already-built ``patch(...)`` context managers entered for
    the duration of the call (e.g. the ``rework_outcome`` head probe).
    """
    from contextlib import ExitStack

    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    sessions_dir = sessions_dir_for(tmp_path)
    kwargs: dict[str, Any] = {}
    if fleet_dir is not None:
        kwargs["fleet_dir_override"] = str(fleet_dir)
    with ExitStack() as stack:
        stack.enter_context(host_probe(monkeypatch, alive=pid_alive))
        for p in patches:
            stack.enter_context(p)
        _detect_and_handle_orphaned_workers(
            sessions_dir,
            paths.state_file,
            config,
            gh,
            review_callback=review_callback,
            write_gate=_wg(paths.state_file, dry_run=dry_run),
            **kwargs,
        )


def events_of(paths: Any, kind: str) -> list[dict[str, Any]]:
    return [e for e in load_state(paths.state_file).get("events", []) if e.get("kind") == kind]


def issue_entry(paths: Any, number: int) -> dict[str, Any]:
    return load_state(paths.state_file)["issues"][str(number)]


def write_terminal_exit(
    tmp_path: Path,
    number: int,
    *,
    exit_code: int = 0,
    started_at: str = "2024-01-01T00:00:00Z",
    ended_at: str = "2024-01-01T00:05:00Z",
) -> None:
    """Durable terminal-status record a worker's exit watcher would write.

    ``started_at`` must not predate the entry's ``dispatched_at``: a record
    from an earlier dispatch is ignored as stale.
    """
    (sessions_dir_for(tmp_path) / f"issue-{number}.claude.terminal.json").write_text(
        json.dumps(
            {
                "pid": 99999,
                "exit_code": exit_code,
                "started_at": started_at,
                "ended_at": ended_at,
                "duration_seconds": 300.0,
            }
        ),
        encoding="utf-8",
    )


def write_blocked_outcome(
    tmp_path: Path, config: OrchestratorConfig, branch: str, reason_kind: str, detail: str
) -> None:
    worktree = worktree_path_for_branch(
        tmp_path, branch, resolved_layout(config, tmp_path).worktrees
    )
    worktree.mkdir(parents=True, exist_ok=True)
    (worktree / ".worker-outcome.json").write_text(
        json.dumps({"outcome": "blocked", "reason_kind": reason_kind, "detail": detail}),
        encoding="utf-8",
    )


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


def make_repo_with_pushed_branch(tmp_path: Path, branch: str) -> Path:
    """Real repo + bare origin with ``branch`` pushed one commit ahead of main."""
    remote = tmp_path / "remote"
    remote.mkdir(parents=True, exist_ok=True)
    _git(["git", "init", "--bare", str(remote)], cwd=tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    _git(["git", "init", "--initial-branch=main", str(repo)], cwd=tmp_path)
    _git(["git", "config", "user.email", "test@example.test"], cwd=repo)
    _git(["git", "config", "user.name", "Test User"], cwd=repo)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git(["git", "add", "README.md"], cwd=repo)
    _git(["git", "commit", "-m", "initial"], cwd=repo)
    _git(["git", "remote", "add", "origin", str(remote)], cwd=repo)
    _git(["git", "push", "-u", "origin", "main"], cwd=repo)
    _git(["git", "checkout", "-b", branch], cwd=repo)
    (repo / "fix.txt").write_text("fix\n", encoding="utf-8")
    _git(["git", "add", "fix.txt"], cwd=repo)
    _git(["git", "commit", "-m", "fix"], cwd=repo)
    _git(["git", "push", "-u", "origin", branch], cwd=repo)
    _git(["git", "checkout", "main"], cwd=repo)
    return repo


def write_aged_outcome(
    repo_root: Path,
    config: OrchestratorConfig,
    branch: str,
    outcome: dict[str, Any],
    *,
    age_seconds: float,
) -> None:
    """Write ``.worker-outcome.json`` into the branch's worktree, backdated."""
    worktree = worktree_path_for_branch(
        repo_root, branch, resolved_layout(config, repo_root).worktrees
    )
    worktree.mkdir(parents=True, exist_ok=True)
    path = worktree / ".worker-outcome.json"
    path.write_text(json.dumps(outcome), encoding="utf-8")
    old = time.time() - age_seconds
    os.utime(path, (old, old))
