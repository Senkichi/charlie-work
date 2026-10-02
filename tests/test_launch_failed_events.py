"""``launch_failed`` event emission at the launch seam (issue #2246).

Every worker/reviewer launch path that comes back as an error value must emit
exactly one ``launch_failed`` event at launch time — at the seam where the
record's ``.error`` is set, not a sweep cycle later. The event carries
``role`` ("worker"/"reviewer"), ``harness``, ``model`` (the role-chain entry
actually tried), ``issue_number``/``pr_number``, a bounded ``error_class``,
and the (truncated) error text.

These tests drive the REAL launch functions — a nonexistent binary and a
failing ``Popen`` per adapter — and read the events back out of the SQLite
``events.db`` that lives beside ``state.json`` under ``runtime.state_dir``.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _devin_shell_fixtures import _fake_worktree

from charlie_work import claude_code, devin_review_resume, devin_shell
from charlie_work.adapters import AdapterSettings, SessionRequest, dispatch_sessions
from charlie_work.api_worker import launch_api_worker
from charlie_work.claude_code import launch_claude_worker
from charlie_work.config import (
    ApiBudgetConfig,
    ApiProviderConfig,
    ApiWorkerConfig,
    OrchestratorConfig,
)
from charlie_work.devin_shell import launch_devin_session
from charlie_work.host.launch import RealReviewLauncher
from charlie_work.instrumentation import _LEVEL_BY_KIND, query_events
from charlie_work.paths import runtime_paths
from charlie_work.worktree import WorktreeInfo


_MISSING_BINARY = ("this-binary-does-not-exist-xyz-2246",)


def _state_path(repo_root: Path) -> Path:
    """Where the launch seam's events land: the repo's runtime state file."""
    paths = runtime_paths(repo_root, OrchestratorConfig().runtime.state_dir)
    return paths.state_file


def _launch_failed(repo_root: Path) -> list[dict[str, Any]]:
    return query_events(_state_path(repo_root), kind="launch_failed")


def _install_fake_worktree(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Fake ``create_worktree`` on both adapters (each binds its own name)."""
    for module in (claude_code, devin_shell):
        monkeypatch.setattr(
            module, "create_worktree", lambda *a, **k: _fake_worktree(tmp_path, "b")
        )


def _install_fake_review_checkout(monkeypatch: pytest.MonkeyPatch, reviews_dir: Path) -> None:
    """Fake ``create_review_checkout``/``remove_review_checkout`` on both adapters."""

    def fake_checkout(repo_root: Path, pr_number: int, head_sha: str, *, reviews_dir: Path):
        checkout = reviews_dir / f"pr-{pr_number}"
        checkout.mkdir(parents=True, exist_ok=True)
        return WorktreeInfo(path=checkout, branch=f"pr-{pr_number}", venv_junction=None)

    for module in (claude_code, devin_shell):
        monkeypatch.setattr(module, "create_review_checkout", fake_checkout)
        monkeypatch.setattr(module, "remove_review_checkout", lambda *a, **k: True)


def _api_worker_config() -> ApiWorkerConfig:
    provider = ApiProviderConfig(
        base_url="https://api.moonshot.ai/anthropic",
        api_key_env="MOONSHOT_API_KEY",
        model="kimi-k3",
        input_usd_per_mtok=3.0,
        output_usd_per_mtok=15.0,
        cached_input_usd_per_mtok=0.30,
    )
    return ApiWorkerConfig(
        enabled=True,
        provider="kimi-k3",
        max_concurrent_sessions=1,
        providers={"kimi-k3": provider},
        budget=ApiBudgetConfig(),
        worker_template="worker_claude_code.md",
        rework_template="rework.md",
    )


# --- worker launches ---------------------------------------------------------


def test_launch_devin_session_missing_binary_emits_launch_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("do the thing", encoding="utf-8")
    _install_fake_worktree(monkeypatch, tmp_path)

    record = launch_devin_session(
        7,
        "agent/issue-7-x",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_MISSING_BINARY,
    )

    assert record.error is not None and record.pid is None
    events = _launch_failed(repo_root)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["role"] == "worker"
    assert payload["harness"] == "devin-shell"
    assert payload["model"] == ""
    assert payload["issue_number"] == 7
    assert payload["error_class"] == "spawn"
    assert "failed to launch devin" in payload["error"]
    assert events[0]["level"] == "error"


def test_launch_devin_session_failing_popen_emits_launch_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("do the thing", encoding="utf-8")
    _install_fake_worktree(monkeypatch, tmp_path)

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise OSError("exec format error")

    monkeypatch.setattr(devin_shell, "popen_worker", boom)

    record = launch_devin_session(
        8,
        "agent/issue-8-x",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        worker_model="swe-2-high",
    )

    assert record.error is not None and record.pid is None
    events = _launch_failed(repo_root)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["role"] == "worker"
    assert payload["harness"] == "devin-shell"
    assert payload["model"] == "swe-2-high"
    assert payload["issue_number"] == 8
    assert payload["error_class"] == "spawn"
    assert "exec format error" in payload["error"]


def test_launch_claude_worker_missing_binary_emits_launch_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_worktree(monkeypatch, tmp_path)

    record = launch_claude_worker(
        9,
        "agent/issue-9-x",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_MISSING_BINARY,
    )

    assert record.error is not None and record.pid is None
    events = _launch_failed(repo_root)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["role"] == "worker"
    assert payload["harness"] == "claude-code"
    assert payload["issue_number"] == 9
    assert payload["error_class"] == "spawn"
    assert "failed to launch claude" in payload["error"]
    assert events[0]["level"] == "error"


def test_launch_claude_worker_failing_popen_emits_launch_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_worktree(monkeypatch, tmp_path)

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise OSError("permission denied on spawn")

    monkeypatch.setattr(claude_code, "popen_worker", boom)

    record = launch_claude_worker(
        10,
        "agent/issue-10-x",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        config=OrchestratorConfig(),
    )

    assert record.error is not None and record.pid is None
    events = _launch_failed(repo_root)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["role"] == "worker"
    assert payload["harness"] == "claude-code"
    assert payload["issue_number"] == 10
    assert payload["error_class"] == "spawn"


def test_launch_api_worker_missing_binary_emits_launch_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_worktree(monkeypatch, tmp_path)
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test")

    record = launch_api_worker(
        11,
        "agent/issue-11-x",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        api_worker_config=_api_worker_config(),
        command_template=_MISSING_BINARY,
    )

    assert record.error is not None and record.pid is None
    events = _launch_failed(repo_root)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["role"] == "worker"
    assert payload["harness"] == "api"
    # The api adapter pins the provider's configured model, not worker.model.
    assert payload["model"] == "kimi-k3"
    assert payload["issue_number"] == 11
    assert payload["error_class"] == "spawn"
    assert events[0]["level"] == "error"


def test_launch_api_worker_failing_popen_emits_launch_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_worktree(monkeypatch, tmp_path)
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test")

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise OSError("spawn blew up")

    # api_worker delegates to claude_code.launch_claude_worker, which resolves
    # popen_worker from claude_code's module namespace.
    monkeypatch.setattr(claude_code, "popen_worker", boom)

    record = launch_api_worker(
        12,
        "agent/issue-12-x",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        api_worker_config=_api_worker_config(),
    )

    assert record.error is not None and record.pid is None
    events = _launch_failed(repo_root)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["role"] == "worker"
    assert payload["harness"] == "api"
    assert payload["model"] == "kimi-k3"
    assert payload["error_class"] == "spawn"


def test_launch_api_worker_disabled_config_emits_launch_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Early error-value paths (before delegation) also emit once."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"

    disabled = dataclasses.replace(_api_worker_config(), enabled=False)

    record = launch_api_worker(
        13,
        "agent/issue-13-x",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        api_worker_config=disabled,
    )

    assert record.error is not None
    events = _launch_failed(repo_root)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["harness"] == "api"
    assert payload["error_class"] == "config"
    assert payload["issue_number"] == 13


# --- reviewer launches -------------------------------------------------------


def test_review_launch_claude_worker_emits_launch_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()
    _install_fake_review_checkout(monkeypatch, reviews_dir)

    record = launch_claude_worker(
        500,
        "agent/issue-500-x",
        "review this",
        repo_root=repo_root,
        sessions_dir=reviews_dir,
        command_template=_MISSING_BINARY,
        review=True,
        head_sha="deadbeef",
    )

    assert record.error is not None and record.pid is None
    events = _launch_failed(repo_root)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["role"] == "reviewer"
    assert payload["harness"] == "claude-code"
    # In review mode the record's issue_number field IS the PR number.
    assert payload["pr_number"] == 500
    assert payload["issue_number"] is None
    assert payload["error_class"] == "spawn"


def test_review_launch_devin_session_emits_launch_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()
    _install_fake_review_checkout(monkeypatch, reviews_dir)
    prompt_path = tmp_path / "review-prompt.md"
    prompt_path.write_text("review this", encoding="utf-8")

    record = launch_devin_session(
        501,
        "agent/issue-501-x",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=reviews_dir,
        command_template=_MISSING_BINARY,
        review=True,
        head_sha="deadbeef",
        worker_model="swe-2-high",
    )

    assert record.error is not None and record.pid is None
    events = _launch_failed(repo_root)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["role"] == "reviewer"
    assert payload["harness"] == "devin-shell"
    assert payload["model"] == "swe-2-high"
    assert payload["pr_number"] == 501
    assert payload["issue_number"] is None
    assert payload["error_class"] == "spawn"


def test_review_launch_unsupported_harness_emits_launch_failed(tmp_path: Path) -> None:
    """The host port's synthetic error record is part of the same seam."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    record = RealReviewLauncher().launch(
        "bogus-harness",
        pr_number=502,
        branch="agent/issue-502-x",
        prompt_path=tmp_path / "p.md",
        prompt_text="review",
        head_sha="deadbeef",
        repo_root=repo_root,
        reviews_dir=tmp_path / "reviews",
        config=OrchestratorConfig(),
        worker_env={},
        materialize_dirs=(),
        resolved_review_effort=None,
        max_turns_override=None,
        model_override="claude-opus-4-6",
        api_worker_config=None,
    )

    assert record.error is not None and record.pid is None
    events = _launch_failed(repo_root)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["role"] == "reviewer"
    assert payload["harness"] == "bogus-harness"
    assert payload["model"] == "claude-opus-4-6"
    assert payload["pr_number"] == 502
    assert payload["error_class"] == "config"


def test_review_resume_popen_failure_emits_launch_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """devin_review_resume's Popen failure is a reviewer launch that came
    back as an error value -- it must emit launch_failed too (the site the
    issue cited)."""
    from _review_fixtures import _dispatch_reviews_app

    app = _dispatch_reviews_app(tmp_path)
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()
    checkout = reviews_dir / "pr-2087"
    checkout.mkdir()
    log_path = reviews_dir / "issue-2087.log"
    log_path.write_text(
        "warning: rejected a tool call that requires confirmation. "
        "Running in non-interactive mode.\n",
        encoding="utf-8",
    )
    (reviews_dir / "issue-2087.json").write_text(
        json.dumps(
            {
                "issue_number": 2087,
                "branch": "agent/issue-2081-x",
                "worktree_path": str(checkout),
                "prompt_path": str(reviews_dir / "p.md"),
                "command": ["devin", "--model", "swe-2-high", "--print"],
                "pid": 40001,
                "started_at": "2026-09-30T18:18:26Z",
                "log_path": str(log_path),
                "process_start_time": 1_790_792_306.5,
                "session_id": "sess-1",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        devin_review_resume,
        "run_captured",
        lambda *a, **k: SimpleNamespace(
            ok=True,
            stdout=json.dumps([{"id": "sess-1", "last_activity_at": 4_000_000_000}]),
            returncode=0,
        ),
    )
    monkeypatch.setattr("charlie_work.worker_fate.is_alive", lambda *a, **k: False)
    monkeypatch.setattr("charlie_work.stalled_review_reap.is_pid_alive", lambda *a, **k: False)

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise OSError("devin binary gone")

    monkeypatch.setattr(devin_review_resume, "popen_worker", boom)

    worker = SimpleNamespace(adapter_kind="devin", log_path=str(log_path))
    ok = devin_review_resume.resume_exec_rejected_review(app, worker, 2087, reviews_dir)

    assert ok is False
    events = query_events(app.paths.state_file, kind="launch_failed")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["role"] == "reviewer"
    assert payload["harness"] == "devin-shell"
    assert payload["model"] == "swe-2-high"
    assert payload["pr_number"] == 2087
    assert payload["error_class"] == "spawn"


# --- adapter-dispatch error values (no record produced) ----------------------


def test_command_adapter_failure_emits_launch_failed(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    manifest_path = tmp_path / "manifest.json"
    results_path = tmp_path / "results.json"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("prompt", encoding="utf-8")

    request = SessionRequest(
        issue_number=20,
        issue_title="Test",
        prompt_path=prompt_path,
        branch_name="agent/issue-20",
    )
    settings = AdapterSettings(
        adapter="command",
        dispatch_command=_MISSING_BINARY,
        command_timeout_seconds=30,
    )

    results = dispatch_sessions(repo_root, manifest_path, results_path, settings, [request])

    assert results[0].ok is False
    events = _launch_failed(repo_root)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["role"] == "worker"
    assert payload["harness"] == "command"
    assert payload["issue_number"] == 20
    assert payload["error_class"] == "spawn"


def test_unsupported_adapter_emits_launch_failed(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    manifest_path = tmp_path / "manifest.json"
    results_path = tmp_path / "results.json"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("prompt", encoding="utf-8")

    request = SessionRequest(
        issue_number=21,
        issue_title="Test",
        prompt_path=prompt_path,
        branch_name="agent/issue-21",
    )
    settings = AdapterSettings(adapter="bogus-adapter")

    results = dispatch_sessions(repo_root, manifest_path, results_path, settings, [request])

    assert results[0].ok is False
    events = _launch_failed(repo_root)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["harness"] == "bogus-adapter"
    assert payload["issue_number"] == 21
    assert payload["error_class"] == "config"


def test_devin_shell_adapter_exception_emits_launch_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The defensive except-shim in _run_devin_shell_adapter is an adapter-level
    launch error value (adapters.py's own seam for it)."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    manifest_path = tmp_path / "manifest.json"
    results_path = tmp_path / "results.json"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("prompt", encoding="utf-8")

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("adapter exploded pre-record")

    monkeypatch.setattr(devin_shell, "launch_devin_session", boom)

    request = SessionRequest(
        issue_number=22,
        issue_title="Test",
        prompt_path=prompt_path,
        branch_name="agent/issue-22",
    )
    settings = AdapterSettings(adapter="devin-shell", worker_model="swe-2-high")

    results = dispatch_sessions(repo_root, manifest_path, results_path, settings, [request])

    assert results[0].ok is False
    assert "launch failed" in (results[0].error or "")
    events = _launch_failed(repo_root)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["harness"] == "devin-shell"
    assert payload["model"] == "swe-2-high"
    assert payload["issue_number"] == 22
    assert payload["error_class"] == "internal"


# --- invariants ---------------------------------------------------------------


def test_successful_launch_emits_no_launch_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exactly-once means zero on the success path too."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("x", encoding="utf-8")
    _install_fake_worktree(monkeypatch, tmp_path)
    script = tmp_path / "fake_devin.py"
    script.write_text("import sys\nprint('ok')\n", encoding="utf-8")

    record = launch_devin_session(
        23,
        "agent/issue-23-x",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script), "{prompt_path}"),
    )

    assert record.error is None and record.pid is not None
    assert _launch_failed(repo_root) == []


def test_launch_failed_is_registered_error_level() -> None:
    """The kind is registered in the level registry as ``error``: a failed
    launch ended the dispatch and burns the rework cap — error-tier like
    ``review_verdict_missed``, not a routine operational warning."""
    assert _LEVEL_BY_KIND.get("launch_failed") == "error"
