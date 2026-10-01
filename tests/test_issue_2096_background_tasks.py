"""Issue #2096: headless claude-code workers must not background their suite.

Two layers: the launch env disables background tasks (enforced at the
harness), and a fast clean exit with an uncommitted tree and no commit is
classified ``worker_exited_with_background_work`` (detection backstop).
"""

from __future__ import annotations

import json
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path

import pytest
from _claude_adapter_fixtures import _install_fake_create_worktree
from _dead_session_fixtures import _git, _make_classify_state
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import _wg
from _worktree_fixtures import _init_bare_remote_and_clone, _setup_completed_worktree

from charlie_work.claude_code import ClaudeWorkerRecord, launch_claude_worker
from charlie_work.config import OrchestratorConfig
from charlie_work.dead_worker_sweep.decide_dead_sessions import (
    BACKGROUND_EXIT_FAILURE_KIND,
    dead_fallback_kind,
    exited_with_background_work,
)

_ENV = "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"


def _probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: dict[str, str] | None) -> str:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _install_fake_create_worktree(monkeypatch, tmp_path)
    script = tmp_path / "probe.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import os
            from pathlib import Path
            Path("probe.txt").write_text(os.environ.get("{_ENV}", "<unset>"), encoding="utf-8")
            """
        ),
        encoding="utf-8",
    )
    record = launch_claude_worker(
        96,
        "agent/issue-96-bg",
        "prompt",
        repo_root=repo_root,
        sessions_dir=tmp_path / "sessions",
        command_template=(sys.executable, str(script)),
        env=env,
    )
    assert record.ok
    probe = Path(record.worktree_path) / "probe.txt"
    return _wait(probe)


def _wait(probe: Path) -> str:
    import time

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if probe.exists() and probe.read_text(encoding="utf-8"):
            return probe.read_text(encoding="utf-8")
        time.sleep(0.05)
    raise AssertionError("worker never wrote probe")


def test_worker_env_disables_background_tasks_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    assert _probe(tmp_path, monkeypatch, None) == "1"


def test_operator_worker_env_overrides_background_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    assert _probe(tmp_path, monkeypatch, {_ENV: "0"}) == "0"


def test_exited_with_background_work_predicate() -> None:
    ok = {"exit_code": 0, "duration_seconds": 38.0}
    kw = {"adapter_kind": "claude-code", "dirty": True, "ahead_count": 0}
    assert exited_with_background_work(ok, **kw)
    assert not exited_with_background_work(None, **kw)
    assert not exited_with_background_work({**ok, "exit_code": 1}, **kw)
    assert not exited_with_background_work({**ok, "duration_seconds": 5000.0}, **kw)
    assert not exited_with_background_work(ok, **{**kw, "dirty": False})
    assert not exited_with_background_work(ok, **{**kw, "ahead_count": 1})
    assert not exited_with_background_work(ok, **{**kw, "adapter_kind": "devin"})


def test_dead_fallback_kind_background_exit() -> None:
    assert (
        dead_fallback_kind(is_completed=False, worktree_unknown=False, background_exit=True)
        == BACKGROUND_EXIT_FAILURE_KIND
    )
    assert dead_fallback_kind(is_completed=False, worktree_unknown=False) == "stalled"


def test_fake_worker_exit_zero_dirty_no_commit_emits_classification_event(
    tmp_path: Path,
) -> None:
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    _remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 40)
    # Un-commit the work: dirty tree, nothing ahead of the base.
    _git(worktree_path, "reset", "--mixed", "origin/main")
    sessions_dir, state_file = _make_classify_state(tmp_path)
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    record = ClaudeWorkerRecord(
        issue_number=40,
        branch=branch,
        worktree_path=str(worktree_path),
        prompt_path=str(tmp_path / "prompt.md"),
        command=("claude", "-p"),
        pid=None,
        started_at=now,
        log_path=str(sessions_dir / "issue-40.claude.log"),
        error=None,
    )
    (sessions_dir / "issue-40.claude.json").write_text(
        json.dumps(record.to_dict()), encoding="utf-8"
    )
    (sessions_dir / "issue-40.claude.terminal.json").write_text(
        json.dumps({"pid": 1, "exit_code": 0, "duration_seconds": 38.0}), encoding="utf-8"
    )
    assert _git(worktree_path, "status", "--porcelain").stdout.strip()

    config = OrchestratorConfig()
    gh = FakeGitHub(repo_root=repo_root)
    gh.issues = [
        {
            "number": 40,
            "title": "t",
            "url": "https://example.test/issues/40",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, state_file, gh, config, write_gate=_wg(state_file)
    )

    state = json.loads(state_file.read_text(encoding="utf-8"))
    events = [e for e in state["events"] if e["kind"] == BACKGROUND_EXIT_FAILURE_KIND]
    assert len(events) == 1
    assert events[0]["payload"]["issue_number"] == 40
