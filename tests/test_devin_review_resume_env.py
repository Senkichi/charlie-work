"""Env-sanitization test for the devin review-resume path (issue #2795).

The resumed exec-rejected reviewer's Popen env is rebuilt through
``sanitize_env(checkout, config=app.config)``; this module pins that the
*configured* api provider key never reaches it. Split out of
``test_launch_failed_events.py`` (that module is an over-cap monolith with a
zero-growth ratchet mark) so the plumbing test lives beside its own subject.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from charlie_work import devin_review_resume
from charlie_work.config import (
    ApiProviderConfig,
    ApiWorkerConfig,
    OrchestratorConfig,
    ReviewDispatchConfig,
)
from charlie_work.paths import runtime_paths


def _provider_api_worker_config() -> ApiWorkerConfig:
    """One enabled api provider whose key env var the resume must strip (#2795)."""
    return ApiWorkerConfig(
        enabled=True,
        provider="kimi-k3",
        max_concurrent_sessions=1,
        providers={
            "kimi-k3": ApiProviderConfig(
                base_url="https://api.moonshot.ai/anthropic",
                api_key_env="MOONSHOT_API_KEY",
                model="kimi-k3",
                input_usd_per_mtok=3.0,
                output_usd_per_mtok=15.0,
                cached_input_usd_per_mtok=0.30,
            )
        },
    )


def test_review_resume_strips_configured_provider_key_from_popen_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The resumed reviewer's Popen env must not carry the provider key (#2795).

    resume_exec_rejected_review rebuilds the child env through
    ``sanitize_env(checkout, config=app.config)`` and hands it straight to
    ``popen_worker``. The provider key lives in the supervisor's ambient env
    (the api adapter resolves it there), so the config plumbing is what keeps
    the key out of the resumed reviewer's env. Reverting devin_review_resume.py
    (the ``config=app.config`` wiring removed) makes this fail: the captured
    Popen env still carries ``MOONSHOT_API_KEY``.
    """
    from _fakes_github import FakeGitHub
    from charlie_work.workflow import OrchestratorApp

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
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-configured-provider-key")

    captured: dict[str, Any] = {}

    class FakeProcess:
        pid = 40042

    def fake_popen_worker(command: list[str], **kwargs: Any) -> FakeProcess:
        captured["command"] = list(command)
        captured["env"] = kwargs.get("env")
        return FakeProcess()

    monkeypatch.setattr(devin_review_resume, "popen_worker", fake_popen_worker)
    monkeypatch.setattr(devin_review_resume, "_get_process_start_time", lambda *a, **k: 1.0)
    monkeypatch.setattr(
        devin_review_resume, "maybe_start_terminal_status_watcher", lambda *a, **k: None
    )

    # The config must carry the api provider for the resume env to strip its
    # key. Build the app with that config resolved at construction time —
    # config is never reassigned on a live OrchestratorApp (workflow caches
    # _layout from it).
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True),
        api_worker=_provider_api_worker_config(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    worker = SimpleNamespace(adapter_kind="devin", log_path=str(log_path))
    ok = devin_review_resume.resume_exec_rejected_review(app, worker, 2087, reviews_dir)

    assert ok is True, captured
    env = captured["env"]
    assert env is not None, "popen_worker must receive the sanitized env kwarg"
    assert "MOONSHOT_API_KEY" not in env, (
        "the resumed reviewer's env must not carry the configured provider key (issue #2795)"
    )
    # Positive control: the env handed to Popen is sanitize_env's output
    # (worktree-local GH_CONFIG_DIR proves the sanitized base, not raw
    # os.environ, reached the child).
    assert "GH_CONFIG_DIR" in env
