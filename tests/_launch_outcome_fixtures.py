"""Shared scaffolding for the launch-outcome characterisation suites (issue #3098/#3099).

Scripted ``FakeWorkerLauncher`` launches and lane-parametrized app builders for
``tests/test_worker_launch_outcome.py`` and ``tests/test_worker_launch_failures.py``.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _fakes_github import FakeGitHub
from charlie_work.adapters import SessionDispatchResult
from charlie_work.config import (
    WRITER_MARKER_FILENAME,
    DevinConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.host.fakes import FakeWorkerLauncher
from charlie_work.instrumentation import query_events
from charlie_work.labels import TransitionOutcome
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp
from charlie_work.worktree import worktree_path_for_branch

FRESH = "fresh"
REWORK = "rework"
LANES = pytest.mark.parametrize("lane", [FRESH, REWORK])

_FOREIGN_WRITER_ERROR = "worktree has a live foreign writer"


class _LabelFailGitHub(FakeGitHub):
    """Every label add fails (error-as-value), so no transition is APPLIED."""

    def __init__(self, *, needs_rework: str | None = None) -> None:
        super().__init__()
        if needs_rework is not None:
            self.issues[0]["labels"] = [{"name": needs_rework}]

    def add_issue_label(self, number: int, label: str) -> bool:
        return False


class _ReworkGitHub(FakeGitHub):
    def __init__(self, needs_rework: str) -> None:
        super().__init__()
        self.issues[0]["labels"] = [{"name": needs_rework}]


def _config(**watchdog: int) -> OrchestratorConfig:
    return OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; sys.exit(1)")),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(redispatch_window_minutes=240, **watchdog),
    )


def _seed_issue(app_paths: Any, entry: dict[str, Any] | None) -> None:
    app_paths.root.mkdir(parents=True, exist_ok=True)
    if entry is None:
        return
    with state_lock(app_paths.state_file):
        state = load_state(app_paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            **entry,
        }
        save_state(app_paths.state_file, state)


def _fresh_app(
    tmp_path: Path,
    *,
    config: OrchestratorConfig,
    entry: dict[str, Any] | None = None,
    label_fail: bool = False,
) -> OrchestratorApp:
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    _seed_issue(paths, entry)
    gh: FakeGitHub = _LabelFailGitHub() if label_fail else FakeGitHub()
    # The default fixture PR links issue 123 and would exclude it; close it.
    gh.prs[0]["state"] = "CLOSED"
    return OrchestratorApp(tmp_path, paths, config, gh)


def _rework_app(
    tmp_path: Path,
    *,
    config: OrchestratorConfig,
    entry: dict[str, Any] | None = None,
    label_fail: bool = False,
    dry_run: bool = False,
) -> OrchestratorApp:
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    _seed_issue(paths, {"status": "rework_requested", **(entry or {})})
    needs_rework = config.labels.needs_rework
    gh: FakeGitHub = (
        _LabelFailGitHub(needs_rework=needs_rework) if label_fail else _ReworkGitHub(needs_rework)
    )
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    (pr_dir / "rework-prompt.md").write_text("Fix the issues", encoding="utf-8")
    return OrchestratorApp(tmp_path, paths, config, gh, dry_run=dry_run)


def _app(
    lane: str,
    tmp_path: Path,
    *,
    config: OrchestratorConfig,
    entry: dict[str, Any] | None = None,
    label_fail: bool = False,
) -> OrchestratorApp:
    build = _fresh_app if lane == FRESH else _rework_app
    return build(tmp_path, config=config, entry=entry, label_fail=label_fail)


def _run(app: OrchestratorApp, lane: str) -> Any:
    return app.dispatch(limit=1) if lane == FRESH else app.dispatch_rework()


def _launch(
    fake_host: Any,
    *,
    ok: bool,
    failure_kind: str | None = None,
    pid: int | None = None,
    worktree_path: str | None = None,
) -> None:
    def _fake(_repo_root, _manifest, _results, _settings, requests):
        return [
            SessionDispatchResult(
                issue_number=r.issue_number,
                issue_title=r.issue_title,
                prompt_path=str(r.prompt_path),
                branch_name=r.branch_name,
                adapter="command",
                ok=ok,
                error=None if ok else _FOREIGN_WRITER_ERROR,
                failure_kind=failure_kind,
                pid=pid,
                worktree_path=worktree_path,
            )
            for r in requests
        ]

    fake_host(worker_launch=FakeWorkerLauncher([_fake]))


def _assert_label_error(app: OrchestratorApp, result: Any, edge: str) -> None:
    """The persisted ``label_error``, the result's ``label_errors`` and the reason."""
    assert result.data["label_errors"] == [123]
    reason = result.data["failures"][123]
    assert reason.startswith(f"label transition '{edge}' partial_failure; add failures: [")
    label_error = load_state(app.paths.state_file)["issues"]["123"]["label_error"]
    assert label_error["edge"] == edge
    assert label_error["outcome"] == TransitionOutcome.PARTIAL_FAILURE.value
    assert label_error["add_failures"]
    assert not label_error["remove_failures"]
    assert reason == f"label transition '{edge}' partial_failure; add failures: " + str(
        [label for _issue, label in label_error["add_failures"]]
    )


def _events(app: OrchestratorApp, kind: str) -> list[dict[str, Any]]:
    return query_events(app.paths.state_file, kind=kind)


def _marker(app: OrchestratorApp, tmp_path: Path) -> Path:
    """Plant a writer marker at the worktree derived from the PR head ref."""
    branch = str(app.gh.prs[0]["headRefName"])
    wt_path = worktree_path_for_branch(tmp_path, branch, app._layout.worktrees)
    wt_path.mkdir(parents=True, exist_ok=True)
    marker = {
        "pid": 1234,
        "session_id": "foreign-session",
        "started_at": (datetime.now(UTC) - timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
        "kind": "worker",
        "process_start_time": 1234567890.0,
    }
    (wt_path / WRITER_MARKER_FILENAME).write_text(json.dumps(marker), encoding="utf-8")
    return wt_path
