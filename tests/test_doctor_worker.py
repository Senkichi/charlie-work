"""Worker-facing doctor checks: worker model, GitHub token, role config.

Split out of ``tests/test_doctor.py`` (issue #1563, Track 1 shoulder) --
bodies are verbatim relocations; shared helpers live in
``tests/_doctor_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path
from charlie_work.config import (
    AutoMergeConfig,
    DevinConfig,
    ReviewerRoleConfig,
    WorkerRoleConfig,
)
from charlie_work.doctor import run_doctor
from charlie_work.paths import runtime_paths
from _doctor_fixtures import (
    FakeDoctorGitHub,
    _config,
)


def test_doctor_reports_config_driven_worker_model(tmp_path: Path) -> None:
    """When devin.worker_model is set, doctor must report the config-driven model."""
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(
            sessions_dir="sessions",
            worker_model="claude-sonnet-4-5",
        ),
        worker=WorkerRoleConfig(harness="devin-shell", model="claude-sonnet-4-5"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    assert "devin-shell worker model" in by_name
    model_check = by_name["devin-shell worker model"]
    assert model_check.ok is True
    assert "config-driven: claude-sonnet-4-5" in model_check.detail
    assert ok is True


def test_doctor_reports_cli_default_when_worker_model_empty(tmp_path: Path) -> None:
    """When devin.worker_model is empty (default), doctor must report CLI default."""
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(
            sessions_dir="sessions",
            worker_model="",
        ),
        worker=WorkerRoleConfig(harness="devin-shell", model=""),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    assert "devin-shell worker model" in by_name
    model_check = by_name["devin-shell worker model"]
    assert model_check.ok is True
    assert "CLI default" in model_check.detail
    assert model_check.severity == "warning"
    assert ok is True  # warning-only, not a blocking failure


def test_doctor_omits_worker_model_check_for_non_devin_shell_adapters(tmp_path: Path) -> None:
    """When adapter is not devin-shell, the worker model check must not appear."""
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(
            sessions_dir="sessions",
            worker_model="claude-sonnet-4-5",
        ),
        worker=WorkerRoleConfig(harness="claude-code", model="claude-sonnet-4-5"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    names = {check.name for check in checks}
    assert "devin-shell worker model" not in names


def test_worker_github_token_check_is_retired(tmp_path: Path) -> None:
    """Issue #1853: doctor must not report a "worker GitHub token" finding.

    Workers are credential-free by design (operator decision, 2026-09-23):
    ``gh`` is unavailable to them and every PR mutation flows through
    ``.worker-outcome.json`` instead. The #873/#1001 check was retired —
    a missing ``worker_env`` token is the intended state, not a defect.
    """
    for adapter in ("devin-shell", "claude-code", "manual", "command"):
        config = _config(
            auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
            devin=DevinConfig(sessions_dir="sessions"),
            worker=WorkerRoleConfig(harness=adapter),
        )
        paths = runtime_paths(tmp_path, config.runtime.state_dir)
        gh = FakeDoctorGitHub(labels=config.labels.all)

        _ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

        names = {check.name for check in checks}
        assert "worker GitHub token" not in names, adapter
        assert "worker GitHub token (claude-code-routed)" not in names, adapter


def test_doctor_reports_role_config_summary(tmp_path: Path) -> None:
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        worker=WorkerRoleConfig(harness="devin-shell", model="claude-sonnet-4-5"),
        reviewer=ReviewerRoleConfig(harness="claude-code", model="claude-opus-4-1"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    assert "role config" in by_name
    role_check = by_name["role config"]
    assert role_check.ok is True
    assert "worker: harness=devin-shell model=claude-sonnet-4-5" in role_check.detail
    assert "reviewer: harness=claude-code model=claude-opus-4-1" in role_check.detail
    assert ok is True


def test_doctor_role_config_summary_reports_cross_family_yes_when_models_differ(
    tmp_path: Path,
) -> None:
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        worker=WorkerRoleConfig(harness="devin-shell", model="glm-5-2"),
        reviewer=ReviewerRoleConfig(harness="claude-code", model="claude-opus-4-1"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    assert "cross-family: yes" in by_name["role config"].detail


# ---------------------------------------------------------------------------
# claude-code worker tool scope / MCP isolation (issue #2836)
# ---------------------------------------------------------------------------


def _doctor_checks_for(tmp_path: Path, config) -> dict:
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)
    _ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)
    return {check.name: check for check in checks}


def test_doctor_reports_claude_code_tool_scope_defaults(tmp_path: Path) -> None:
    from charlie_work.config import ClaudeCodeConfig

    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(sessions_dir="sessions"),
        worker=WorkerRoleConfig(harness="claude-code"),
        claude_code=ClaudeCodeConfig(),
    )

    by_name = _doctor_checks_for(tmp_path, config)

    check = by_name["claude-code worker tool scope"]
    assert check.ok is True
    assert "--tools Bash,Read,Edit,Write,Grep,Glob,Monitor,TaskStop" in check.detail
    assert "MCP isolation: on" in check.detail


def test_doctor_warns_when_isolate_mcp_off(tmp_path: Path) -> None:
    from charlie_work.config import ClaudeCodeConfig

    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(sessions_dir="sessions"),
        worker=WorkerRoleConfig(harness="claude-code"),
        claude_code=ClaudeCodeConfig(isolate_mcp=False),
    )

    by_name = _doctor_checks_for(tmp_path, config)

    check = by_name["claude-code worker tool scope"]
    assert check.ok is False
    assert check.severity == "warning"
    assert "isolate_mcp is false" in check.detail


def test_doctor_warns_when_worker_tools_empty(tmp_path: Path) -> None:
    from charlie_work.config import ClaudeCodeConfig

    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(sessions_dir="sessions"),
        worker=WorkerRoleConfig(harness="claude-code"),
        claude_code=ClaudeCodeConfig(worker_tools=""),
    )

    by_name = _doctor_checks_for(tmp_path, config)

    check = by_name["claude-code worker tool scope"]
    assert check.ok is False
    assert check.severity == "warning"
    assert "worker_tools is empty" in check.detail


def test_doctor_tool_scope_reports_command_template_pin(tmp_path: Path) -> None:
    from charlie_work.config import ClaudeCodeConfig

    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(sessions_dir="sessions"),
        worker=WorkerRoleConfig(harness="claude-code"),
        claude_code=ClaudeCodeConfig(command=("claude", "-p", "--tools", "Bash,Read")),
    )

    by_name = _doctor_checks_for(tmp_path, config)

    check = by_name["claude-code worker tool scope"]
    assert check.ok is True
    assert "--tools Bash,Read" in check.detail
