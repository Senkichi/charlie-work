"""Shared fixtures for the superseded-worker reap test modules (issue #1494).

Hoisted out of ``tests/test_charlie_work_superseded_worker_reap.py`` during
the PR #1926 rework review when that module grew past the 800-line cap and
was split into a unit-level module and a lane-integration module. The
``tests/_*.py`` hoisted-fixture convention is the sanctioned import target
for shared test helpers (issue #1284; test modules may not import each
other -- see ``tests/test_zero_cross_test_import_guard.py``).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from _fakes_github import FakeGitHub
from charlie_work import host as host_pkg
from charlie_work.adapters import SessionDispatchResult
from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.worker import WorkerView
from charlie_work.workflow import OrchestratorApp
from charlie_work.write_gate import WriteGate


PRIOR_PID = 48840
PRIOR_START = 1234567890.0


class _AliveDictProbe:
    """Host probe reading a shared ``alive`` dict lazily.

    Unlike ``FakeProcessProbe`` (which snapshots its map), this reads the
    caller's dict on every call so the fake ``kill_process_tree`` flipping a
    pid to dead is observed by the post-kill recheck.
    """

    def __init__(self, alive: dict[int, bool]) -> None:
        self._alive = alive

    def is_alive(self, pid: int | None, start_time: float | None = None) -> bool:
        return self._alive.get(pid, False)

    def start_time(self, pid: int) -> float | None:
        return None


def _worker_view(
    issue_number: int,
    pid: int | None,
    *,
    process_start_time: float | None = PRIOR_START,
    error: str | None = None,
    worktree_path: str = "",
    session_id: str | None = "prior-session",
) -> WorkerView:
    return WorkerView(
        adapter_kind="devin",
        issue_number=issue_number,
        repo_key="",
        pid=pid,
        started_at="2026-01-01T00:00:00Z",
        process_start_time=process_start_time,
        log_path="",
        worktree_path=worktree_path,
        error=error,
        failure_kind=None,
        reclaimed=None,
        session_id=session_id,
    )


def _patch_reap_internals(
    monkeypatch: pytest.MonkeyPatch,
    *,
    workers: list[WorkerView],
    alive: dict[int, bool],
    kill_calls: list[tuple[int, float | None]],
    events: list[tuple[str, dict]],
    orphans: list[dict] | None = None,
    orphan_kill_calls: list[int] | None = None,
) -> None:
    """Patch every seam ``_reap_superseded_workers`` touches.

    ``alive`` maps pid -> liveness; the fake ``kill_process_tree`` flips the
    pid to dead so the post-kill recheck observes the reap. Patching the
    module-level names ``write_gate`` delegates to keeps a real ``WriteGate``
    under test (dry-run short-circuit included).

    The stall detector is no-oped: ``_dispatch_rework_impl`` runs
    ``_detect_and_handle_stalled_sessions`` before candidate selection, and
    a live seeded worker would otherwise be reaped by that lane first —
    leaving the launch-trigger helper nothing to do. Issue #1494 is exactly
    the case the stall lane does NOT cover, so these tests isolate it.
    """

    monkeypatch.setattr(
        "charlie_work.workflow._detect_and_handle_stalled_sessions",
        lambda *a, **k: [],
    )
    monkeypatch.setattr("charlie_work.worker.iter_workers", lambda _sessions_dir: list(workers))
    monkeypatch.setattr(
        host_pkg,
        "_ACTIVE",
        dataclasses.replace(host_pkg.current(), probe=_AliveDictProbe(alive)),
    )

    def _fake_kill_process_tree(pid: int, st: float | None = None) -> list[int]:
        kill_calls.append((pid, st))
        if alive.get(pid, False):
            alive[pid] = False
            return [pid]
        return []

    monkeypatch.setattr("charlie_work.write_gate.kill_process_tree", _fake_kill_process_tree)
    monkeypatch.setattr(
        "charlie_work.write_gate.log_event",
        lambda _state_path, kind, payload, **_kw: events.append((kind, payload)),
    )
    monkeypatch.setattr(
        "charlie_work.superseded_worker_reap.sweep_orphan_processes",
        lambda _wt: list(orphans or []),
    )
    monkeypatch.setattr(
        "charlie_work.write_gate.kill_orphan_pid",
        lambda pid: orphan_kill_calls.append(pid) if orphan_kill_calls is not None else None,
    )


def _gate(tmp_path: Path, *, dry_run: bool = False) -> WriteGate:
    return WriteGate(dry_run=dry_run, state_path=tmp_path / "state.json", repo="test")


def _rework_app(tmp_path: Path, config: OrchestratorConfig):
    """Issue 123 in rework_requested + open PR 456 + rework prompt on disk —
    the standard dispatch_rework candidate fixture."""
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
        }
        save_state(paths.state_file, state)

    fake_gh = ReworkGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    (pr_dir / "rework-prompt.md").write_text("Fix the issues", encoding="utf-8")
    return app, fake_gh, paths


def _ok_dispatch_factory(order: list[str]):
    def _fake(_repo_root, _manifest, _results, _settings, requests):
        order.append("dispatch_sessions")
        return [
            SessionDispatchResult(
                issue_number=request.issue_number,
                issue_title=request.issue_title,
                prompt_path=str(request.prompt_path),
                branch_name=request.branch_name,
                adapter="command",
                ok=True,
                pid=99999,
                process_start_time=2.0,
            )
            for request in requests
        ]

    return _fake
