"""Shared dead-worker orphan-sweep fixtures (issues #1911/#1915).

Hoisted out of ``tests/test_orphaned_worker_completed_outcome_sweep.py`` so
the #1915 review-drain sibling module
(``tests/test_orphaned_worker_review_drain.py``) reuses the same
bed/outcome/sweep helpers -- test modules must not import each other, so
shared fixtures live in the ``tests/_*.py`` hoisted-fixture convention
(``tests/test_zero_cross_test_import_guard.py``).

``_dead_worker_rework_bed`` seeds a dead-PID ``dispatched`` issue whose open
PR carries a ``decision`` verdict on the unchanged live head ``abc123``.
``janitor_green=True`` upgrades the PR fixture to the full field set
``run_janitor`` inspects (``state``/``mergeStateStatus``/``body``/title), so
a real ``OrchestratorApp.review()`` can be used as the sweep's
``review_callback`` instead of a stub; the minimal dict still parses as an
open PR either way, so the fake-callback tests keep their behavior.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from _dead_session_fixtures import _write_flat_review_decision
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import _wg
from charlie_work.config import (
    WORKER_OUTCOME_FILENAME,
    DevinConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
)
from charlie_work.worktree import worktree_path_for_branch


def _janitor_green_orphan_pr(issue_number: int, pr_number: int, head_sha: str) -> dict[str, Any]:
    """Full janitor-green PR shape for a real ``OrchestratorApp.review()``."""
    return {
        "number": pr_number,
        "title": f"Fix #{issue_number}: thing",
        "url": f"https://example.test/pull/{pr_number}",
        "headRefName": f"agent/issue-{issue_number}",
        "baseRefName": "main",
        "headRefOid": head_sha,
        "mergeStateStatus": "CLEAN",
        "body": f"Closes #{issue_number}\n\nBody.",
        "labels": [],
        "isCrossRepository": False,
        "state": "OPEN",
    }


def _dead_worker_rework_bed(
    tmp_path: Path,
    *,
    decision: str = "request_changes",
    pr_state_status: str | None = None,
    config: OrchestratorConfig | None = None,
    janitor_green: bool = False,
    flat_decision_extra: dict[str, Any] | None = None,
) -> tuple[Any, Any, Any, str]:
    """Seed the shared #1911 scenario.

    Returns ``(config, paths, fake_gh, dispatched_at)``: a dead-PID dispatched
    issue 207 whose open PR 100 carries a ``decision`` verdict on the
    unchanged live head ``abc123``, with ``gh.repo_root`` pointed at
    ``tmp_path`` so the sweep resolves a real worktrees dir.
    ``pr_state_status`` sets the PR state entry's ``status`` field -- the
    post-approval rework lanes mark it ``rework_requested`` while leaving
    ``decision="approved"`` (issue #1109).

    ``config`` overrides the default bed config (e.g. to pin
    ``auto_merge.required_checks`` for a stale-CI verdict);
    ``janitor_green`` swaps the minimal PR dict for the full shape a real
    ``review()`` needs; ``flat_decision_extra`` is merged into the flat
    ``review-decision.json`` payload (e.g. ``required_changes`` for the
    stale-CI citation shape, issue #1111).
    """
    if config is None:
        config = OrchestratorConfig(
            devin=DevinConfig(),
            worker=WorkerRoleConfig(harness="devin-shell"),
            watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
        )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    dispatched_at = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")

    state = load_state(paths.state_file)
    state["issues"]["207"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": dispatched_at,
        "branch_name": "agent/issue-207",
    }
    state["prs"]["100"] = {
        "decision": decision,
        "reviewed_head_sha": "abc123",
    }
    if pr_state_status is not None:
        state["prs"]["100"]["status"] = pr_state_status
    save_state(paths.state_file, state)
    if flat_decision_extra is None:
        _write_flat_review_decision(paths, 100, decision, "abc123")
    else:
        pr_dir = paths.prs / "pr-100"
        pr_dir.mkdir(parents=True, exist_ok=True)
        (pr_dir / "review-decision.json").write_text(
            json.dumps(
                {
                    "decision": decision,
                    "reviewed_head_sha": "abc123",
                    **flat_decision_extra,
                }
            ),
            encoding="utf-8",
        )

    pr: dict[str, Any] = (
        _janitor_green_orphan_pr(207, 100, "abc123")
        if janitor_green
        else {
            "number": 100,
            "headRefOid": "abc123",  # Unchanged since request_changes
            "isCrossRepository": False,
            "headRepository": {"owner": {"login": "test"}, "name": "repo"},
            "headRefName": "agent/issue-207",
        }
    )

    fake_gh = FakeGitHub(repo_root=tmp_path)
    fake_gh.prs = [pr]
    fake_gh.issues.append(
        {
            "number": 207,
            "title": "Test issue",
            "url": "https://example.test/issues/207",
            "body": "",
            "labels": [],
            "state": "OPEN",
        }
    )
    return config, paths, fake_gh, dispatched_at


def _write_outcome(
    paths: Any, tmp_path: Path, payload: dict[str, Any], *, mtime: datetime | None = None
) -> Path:
    worktree_path = worktree_path_for_branch(tmp_path, "agent/issue-207", paths.worktrees)
    worktree_path.mkdir(parents=True, exist_ok=True)
    outcome_path = worktree_path / WORKER_OUTCOME_FILENAME
    outcome_path.write_text(json.dumps(payload), encoding="utf-8")
    if mtime is not None:
        ts = mtime.timestamp()
        os.utime(outcome_path, (ts, ts))
    return outcome_path


def _run_orphan_sweep(
    tmp_path: Path,
    paths: Any,
    config: Any,
    fake_gh: Any,
    *,
    review_callback: Any = None,
) -> None:
    from unittest.mock import patch

    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        _detect_and_handle_orphaned_workers(
            sessions_dir,
            paths.state_file,
            config,
            fake_gh,
            review_callback=review_callback,
            write_gate=_wg(paths.state_file),
        )
