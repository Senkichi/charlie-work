"""Shared no-PR dispatched-issue builders for the worker-fate characterization tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from _fakes_github import FakeGitHub

from charlie_work.config import (
    OrchestratorConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state
from charlie_work.write_gate import WriteGate


def _wg(state_file: Path, *, dry_run: bool = False) -> WriteGate:
    return WriteGate(dry_run=dry_run, state_path=state_file, repo="charlie-work")


class _FakeGitHubNoPR(FakeGitHub):
    def pr_list(self):
        return []


def _seed_no_pr_dispatched_issue(
    tmp_path: Path, config: OrchestratorConfig, issue_number: int, branch: str
) -> tuple[Path, Path]:
    """Seed a dead-PID ``dispatched`` issue with no open PR. Returns (state_file, sessions_dir)."""
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    state = load_state(paths.state_file)
    state["issues"][str(issue_number)] = {
        "status": "dispatched",
        "dispatched_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "branch_name": branch,
    }
    save_state(paths.state_file, state)
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return paths.state_file, sessions_dir


def _no_pr_fake_gh(tmp_path: Path, config: OrchestratorConfig, issue_number: int) -> FakeGitHub:
    fake_gh = _FakeGitHubNoPR(repo_root=tmp_path)
    fake_gh.issues = [
        {
            "number": issue_number,
            "title": "test issue",
            "url": f"https://example.test/issues/{issue_number}",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []
    # A successful pr_create by default: only the A9 test drives a PR-open
    # path, but leaving the fake's default None-return in place would make
    # gh.pr_create "fail" and trigger real-time retry/backoff sleeps in
    # pr_create_retry -- pin a return value up front so no test pays that
    # cost by accident.
    fake_gh.pr_create_return = 5501
    return fake_gh
