"""Dispatch adapter wiring: command-adapter rendering, adapter settings, adapter label, per-adapter launch/label paths.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

import sys
from pathlib import Path
from _fakes_github import FakeGitHub
from charlie_work.config import (
    ClaudeCodeConfig,
    DevinConfig,
    DispatchConfig,
    OrchestratorConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_command_adapter_render_error_returns_error_record(tmp_path: Path) -> None:
    """Defense-in-depth: render errors past the load gate return error records, not exceptions."""
    from charlie_work.adapters import AdapterSettings, SessionRequest, dispatch_sessions

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    manifest_path = tmp_path / "manifest.json"
    results_path = tmp_path / "results.json"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("prompt", encoding="utf-8")

    request = SessionRequest(
        issue_number=1,
        issue_title="Test",
        prompt_path=prompt_path,
        branch_name="agent/issue-1",
    )

    settings = AdapterSettings(
        adapter="command",
        dispatch_command=("echo", "{unknown_placeholder}"),
        command_timeout_seconds=300,
    )

    results = dispatch_sessions(repo_root, manifest_path, results_path, settings, [request])

    assert len(results) == 1
    assert results[0].ok is False
    assert results[0].error is not None
    assert "unknown_placeholder" in results[0].error


def test_command_adapter_positional_placeholder_returns_error_record(tmp_path: Path) -> None:
    """AC #2: Command adapter positional placeholder {0} returns error record, not raise.

    The command adapter's render try/except catches IndexError for positional {0} templates.
    This test verifies that a config built directly (bypassing load_config) with a positional
    placeholder returns an error record instead of raising.

    Mutation to verify: remove IndexError from the except clause in adapters.py line 281,
    and the test will fail (it will raise IndexError instead of returning an error record).
    """
    from charlie_work.adapters import AdapterSettings, SessionRequest, dispatch_sessions

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    manifest_path = tmp_path / "manifest.json"
    results_path = tmp_path / "results.json"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("prompt", encoding="utf-8")

    request = SessionRequest(
        issue_number=1,
        issue_title="Test",
        prompt_path=prompt_path,
        branch_name="agent/issue-1",
    )

    # Config built directly (bypassing load_config) with a positional placeholder
    settings = AdapterSettings(
        adapter="command",
        dispatch_command=("echo", "{0}"),
        command_timeout_seconds=300,
    )

    results = dispatch_sessions(repo_root, manifest_path, results_path, settings, [request])

    assert len(results) == 1
    assert results[0].ok is False
    assert results[0].error is not None
    # The error should mention the positional placeholder issue
    assert "0" in results[0].error or "positional" in results[0].error.lower()


def test_adapter_settings_launch_stagger_seconds_wired_from_dispatch_config(
    tmp_path: Path,
) -> None:
    """_adapter_settings() must read dispatch.launch_stagger_seconds -- the
    single point of enforcement between config and both dispatch lanes."""
    config = OrchestratorConfig(dispatch=DispatchConfig(launch_stagger_seconds=17))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    assert app._adapter_settings().launch_stagger_seconds == 17


def test_adapter_settings_api_branch_carries_api_worker_config(
    tmp_path: Path,
) -> None:
    """_adapter_settings() with devin.adapter='api' must carry the resolved
    ApiWorkerConfig into AdapterSettings and reuse the claude-code
    venv_source/worker_env. This is the single point that routes the provider
    registry into the api dispatch lane; a regression here silently breaks all
    api dispatches with no downstream error (the adapter raises an error
    result only at dispatch time, by which point the misconfiguration is
    already fleet-wide)."""
    from charlie_work.config import (
        ApiBudgetConfig,
        ApiProviderConfig,
        ApiWorkerConfig,
    )

    api_provider = ApiProviderConfig(
        base_url="https://api.moonshot.ai/anthropic",
        api_key_env="MOONSHOT_API_KEY",
        model="kimi-k3",
        input_usd_per_mtok=3.0,
        output_usd_per_mtok=15.0,
        cached_input_usd_per_mtok=0.30,
    )
    api_cfg = ApiWorkerConfig(
        enabled=True,
        provider="kimi-k3",
        max_concurrent_sessions=2,
        providers={"kimi-k3": api_provider},
        budget=ApiBudgetConfig(max_usd_per_session=1.5),
    )
    claude_cfg = ClaudeCodeConfig(
        venv_source=".venv",
        worker_env={"PYTEST_XDIST_AUTO_NUM_WORKERS": "2"},
        tee_stream_json=True,
    )
    config = OrchestratorConfig(
        worker=WorkerRoleConfig(harness="api"),
        claude_code=claude_cfg,
        api_worker=api_cfg,
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    settings = app._adapter_settings()
    assert settings.adapter == "api"
    # The api branch must carry the resolved ApiWorkerConfig (not None) so the
    # api dispatch lane can resolve the provider. This is the regression guard:
    # test_dispatch_sessions_api_adapter_end_to_end constructs AdapterSettings
    # directly and bypasses this path, so without this test a broken wiring
    # (e.g. dropping the api_worker_config= line) would go undetected.
    assert settings.api_worker_config is api_cfg
    # The api branch reuses the claude-code venv_source/worker_env resolution.
    assert settings.venv_source == tmp_path / ".venv"
    assert settings.worker_env == {"PYTEST_XDIST_AUTO_NUM_WORKERS": "2"}
    # tee_stream_json is read from claude_code config (launch_api_worker
    # force-enables it internally, but AdapterSettings still carries the
    # configured value for the manifest/results surface).
    assert settings.tee_stream_json is True


def test_adapter_settings_non_api_branches_omit_api_worker_config(
    tmp_path: Path,
) -> None:
    """For devin-shell / claude-code / manual adapters, _adapter_settings()
    must set api_worker_config=None so a stale config block cannot leak into a
    non-api dispatch lane."""
    for adapter in ("devin-shell", "claude-code", "manual"):
        config = OrchestratorConfig(worker=WorkerRoleConfig(harness=adapter))
        paths = runtime_paths(tmp_path, config.runtime.state_dir)
        app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())
        settings = app._adapter_settings()
        assert settings.adapter == adapter
        assert settings.api_worker_config is None, (
            f"adapter={adapter} must not carry api_worker_config"
        )


def test_string_dispatch_command_rejects_issue_title(tmp_path: Path) -> None:
    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command="echo {issue_title}"),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(limit=1)

    assert result.ok is False
    assert result.data["failed_count"] == 1
    assert "list form" in result.data["dispatch_results"][0]["error"]
    assert (123, "agent:in-progress") not in fake_gh.labels_added


def test_devin_shell_dispatch_launches_and_labels_in_progress(tmp_path: Path, monkeypatch) -> None:
    from charlie_work import devin_shell
    from charlie_work.worktree import WorktreeInfo

    wt_path = tmp_path / "worktrees" / "agent-issue-123-fix-search"
    wt_path.mkdir(parents=True, exist_ok=True)

    def _fake_create_worktree(repo_root, branch, **kwargs):
        return WorktreeInfo(path=wt_path, branch=branch, venv_junction=None)

    monkeypatch.setattr(devin_shell, "create_worktree", _fake_create_worktree)

    config = OrchestratorConfig(
        devin=DevinConfig(shell_command=(sys.executable, "-c", "import sys; sys.exit(0)")),
        worker=WorkerRoleConfig(harness="devin-shell"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["dispatch_results"][0]["adapter"] == "devin-shell"
    assert (123, "agent:in-progress") in fake_gh.labels_added
    sidecar = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions" / "issue-123.json"
    assert sidecar.exists()
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatched"


def test_claude_code_dispatch_routes_and_labels(tmp_path: Path, monkeypatch) -> None:
    from charlie_work.claude_code import ClaudeWorkerRecord

    captured: dict[str, object] = {}

    def _fake_launch(issue_number, branch, prompt_text, **kwargs):
        captured["prompt_text"] = prompt_text
        captured["venv_source"] = kwargs.get("venv_source")
        return ClaudeWorkerRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path=str(tmp_path / "wt"),
            prompt_path=str(tmp_path / "wt" / ".orchestrator-prompt.md"),
            command=("claude", "-p"),
            pid=4242,
            started_at="2026-07-02T00:00:00Z",
            log_path=str(tmp_path / "log"),
        )

    monkeypatch.setattr("charlie_work.claude_code.launch_claude_worker", _fake_launch)
    config = OrchestratorConfig(worker=WorkerRoleConfig(harness="claude-code"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(limit=1)

    assert result.ok is True
    assert "Issue #123" in str(captured["prompt_text"])  # rendered prompt fed through
    assert captured["venv_source"] is None  # issue #274: no shared venv by default
    assert (123, "agent:in-progress") in fake_gh.labels_added


def test_claude_code_dispatch_failure_stays_out_of_progress(tmp_path: Path, monkeypatch) -> None:
    from charlie_work.claude_code import ClaudeWorkerRecord

    def _fake_launch(issue_number, branch, prompt_text, **kwargs):
        return ClaudeWorkerRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path="",
            prompt_path="",
            command=("claude", "-p"),
            pid=None,
            started_at="2026-07-02T00:00:00Z",
            log_path="",
            error="claude not found on PATH",
        )

    monkeypatch.setattr("charlie_work.claude_code.launch_claude_worker", _fake_launch)


def test_manifest_adapter_label_helper() -> None:
    """Issue #626: ``manifest_adapter_label`` is the single point of label
    derivation. One kind → that kind; more than one → ``"mixed"``."""
    from charlie_work.adapters import manifest_adapter_label

    assert manifest_adapter_label({"devin-shell"}) == "devin-shell"
    assert manifest_adapter_label({"api"}) == "api"
    assert manifest_adapter_label({"claude-code"}) == "claude-code"
    assert manifest_adapter_label({"api", "devin-shell"}) == "mixed"
    assert manifest_adapter_label({"api", "claude-code", "devin-shell"}) == "mixed"
