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
