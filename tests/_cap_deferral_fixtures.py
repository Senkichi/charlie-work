"""Shared fixtures for the issue #2051 cap-deferral regression tests.

Hoisted out of ``test_cap_escalation_deferred_live_worker.py`` when the
fleet-review rework coverage pushed that module past the 800-line
file-size cap -- same pattern as ``_janitor_routing_fixtures.py`` and
``_unescalate_fixtures.py``. Contains the worker-liveness issue-field
shapes, the dispatch_rework cap fixtures (one per escalated lane), and
the janitor conflict-cap app wrapper the consumer tests share.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _fakes_github import FakeGitHub
from _janitor_routing_fixtures import _conflicting_app
from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    PostMortemConfig,
    ReviewConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp


def _config(tmp_path: Path, **kwargs: Any) -> OrchestratorConfig:
    """Config whose post_mortem sessions.db never resolves.

    Mirrors ``tests/_unescalate_fixtures.py``'s ``_app``: the default
    ``db_path=""`` resolves to the real %APPDATA%\\devin\\cli\\sessions.db,
    which on a runner can hold a stale timestamp for a test PID and flip
    the liveness probe from inconclusive to conclusive-stale. Pointing at
    a nonexistent path makes every probe source error out, so the state-
    side verdict lands on the wall-clock-deadline branch the tests
    control via ``dispatched_at`` freshness.
    """
    return OrchestratorConfig(
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db")),
        **kwargs,
    )


def _live_worker_fields() -> dict[str, Any]:
    """Issue-state fields for a genuinely live state-tracked worker.

    ``os.getpid()`` is this test process -- a real live PID -- and a fresh
    ``dispatched_at`` keeps the session within the watchdog wall-clock
    deadline, so ``issue_worker_liveness``'s inconclusive activity probe
    defers to ``live=True`` (same shape as
    ``test_unescalate_refuses_entirely_when_worker_session_alive``).
    """
    return {
        "worker_pid": os.getpid(),
        "dispatched_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }


def _wedged_worker_fields() -> dict[str, Any]:
    """Issue-state fields for an alive-but-wedged worker.

    The test process is alive, but ``dispatched_at`` is days old -- past
    the wall-clock deadline -- so ``issue_worker_liveness`` reports
    ``live=False`` and the cap escalation must proceed (the same shape as
    ``test_unescalate_proceeds_when_worker_alive_but_wedged``).
    """
    return {
        "worker_pid": os.getpid(),
        "dispatched_at": "2020-01-01T00:00:00+00:00",
    }


def _dead_pid() -> int:
    """A PID that is guaranteed not alive for the rest of this test."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert proc.pid is not None
    return proc.pid


def _events_of_kind(state: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [e for e in state.get("events", []) if e.get("kind") == kind]


def _no_op_cap_app(
    tmp_path: Path,
    issue_extra: dict[str, Any] | None = None,
    *,
    dry_run: bool = False,
) -> tuple[OrchestratorApp, FakeGitHub]:
    """A ``dispatch_rework`` fixture with the no-op cap already exhausted.

    Issue 123 is ``rework_requested`` with ``redispatch_at`` at
    ``max_auto_redispatch`` and PR 456's head still at the recorded
    ``reviewed_head_sha`` -- the exact ``no_op_rework_escalated`` input
    shape from ``test_dispatch_rework_no_op_rework_cap_escalates`` --
    plus whatever worker-liveness fields ``issue_extra`` adds.
    """
    config = _config(
        tmp_path,
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]

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
            **(issue_extra or {}),
        }
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": "sha-abc123",
        }
        save_state(paths.state_file, state)

    return OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=dry_run), fake_gh


def _worker_death_cap_app(
    tmp_path: Path, issue_extra: dict[str, Any] | None = None
) -> tuple[OrchestratorApp, FakeGitHub]:
    """The death-loop cap input: every counted redispatch ended in a
    worker death, so ``paired_death_count`` reaches
    ``max_auto_redispatch`` while the genuine no-op count stays at zero.
    The issue carries no ``branch_name``, so the #1239 stranded-commit
    salvage cannot publish anything and the death lane escalates (or,
    under this PR's guard, defers).
    """
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return _no_op_cap_app(
        tmp_path,
        issue_extra={"worker_death_at": [now_iso, now_iso], **(issue_extra or {})},
    )


def _blocked_environment_cap_app(
    tmp_path: Path, issue_extra: dict[str, Any] | None = None
) -> tuple[OrchestratorApp, FakeGitHub]:
    """The blocked-environment cap input: ``blocked_environment_at`` is
    at ``max_auto_redispatch`` and no reapable foreign-writer marker
    exists at the derived worktree path, so the pre-filter escalates (or
    defers under this PR's guard) before any head check runs.
    """
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return _no_op_cap_app(
        tmp_path,
        issue_extra={"blocked_environment_at": [now_iso, now_iso], **(issue_extra or {})},
    )


def _conflicting_issue_app(tmp_path: Path, **kwargs: Any) -> OrchestratorApp:
    """The janitor conflict-cap fixture with an isolated post_mortem db."""
    return _conflicting_app(
        tmp_path,
        review=ReviewConfig(max_conflict_rework_attempts=1),
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db")),
        **kwargs,
    )
