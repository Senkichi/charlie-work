"""Worker-facing doctor checks: worker model, GitHub token, role config.

Split out of ``tests/test_doctor.py`` (issue #1563, Track 1 shoulder) --
bodies are verbatim relocations; shared helpers live in
``tests/_doctor_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path
from charlie_work.config import (
    AutoMergeConfig,
    ClaudeCodeConfig,
    DevinConfig,
    RescueConfig,
    ReviewerRoleConfig,
    WorkerRoleConfig,
)
from charlie_work.doctor import run_doctor
from charlie_work.paths import runtime_paths
from _doctor_fixtures import (
    FakeDoctorGitHub,
    _api_worker_config,
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


def test_worker_github_token_ok_when_configured_devin_shell(tmp_path: Path) -> None:
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(
            sessions_dir="sessions",
            worker_env={"GH_TOKEN": "placeholder-not-a-real-token"},
        ),
        worker=WorkerRoleConfig(harness="devin-shell"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    assert by_name["worker GitHub token"].ok is True
    assert "GH_TOKEN" in by_name["worker GitHub token"].detail
    assert ok is True


def test_worker_github_token_warns_when_missing_devin_shell(tmp_path: Path) -> None:
    """Missing token is severity=warning (not error).

    The fix is a deferred operator action (#873), so this must surface the
    finding without making every un-tokened production config
    unconditionally doctor-red.
    """
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(sessions_dir="sessions"),
        worker=WorkerRoleConfig(harness="devin-shell"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    check = by_name["worker GitHub token"]
    assert check.ok is False
    assert check.severity == "warning"
    assert "devin.worker_env" in check.detail
    assert ok is True  # warning-only, must not block overall doctor ok


def test_worker_github_token_claude_code_adapter_sources_claude_code_worker_env(
    tmp_path: Path,
) -> None:
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(sessions_dir="sessions"),
        worker=WorkerRoleConfig(harness="claude-code"),
        claude_code=ClaudeCodeConfig(worker_env={"GITHUB_TOKEN": "placeholder-not-a-real-token"}),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    assert by_name["worker GitHub token"].ok is True
    assert "claude_code.worker_env" in by_name["worker GitHub token"].detail
    assert ok is True


def test_worker_github_token_omitted_for_manual_and_command_adapters(tmp_path: Path) -> None:
    """Neither manual nor command has this failure mode.

    manual writes a session manifest and never launches a worker subprocess;
    command has no sanitize_env call at all, so the check must not appear
    for either.
    """
    for adapter in ("manual", "command"):
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


def test_worker_github_token_api_routed_check_fires_alongside_default_adapter(
    tmp_path: Path, monkeypatch
) -> None:
    """A devin-shell default with api_worker enabled still has an api-routed subset.

    Whenever api_worker.enabled is True, some sessions in a pass may launch
    via the api harness (Task 3 of this plan removed per-issue *routing*
    policy, not the api harness itself — api_worker.enabled alone is enough
    to put api sessions in a pass, e.g. via a future non-routing dispatch
    path). That subset sources
    claude_code.worker_env — a devin-shell default with a devin.worker_env
    token must not hide a missing claude_code.worker_env token for the
    api-routed subset.
    """
    # Unrelated api_worker checks (issue #483) also fire once enabled=True;
    # satisfy the api-key-env-var one so this test's `ok` assertion isolates
    # the worker-github-token behavior under test, not that pre-existing probe.
    monkeypatch.setenv("MOONSHOT_API_KEY", "placeholder-not-a-real-key")
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(
            sessions_dir="sessions",
            worker_env={"GH_TOKEN": "placeholder-not-a-real-token"},
        ),
        worker=WorkerRoleConfig(harness="devin-shell"),
        api_worker=_api_worker_config(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    assert by_name["worker GitHub token"].ok is True  # devin-shell path is fine
    routed = by_name["worker GitHub token (claude-code-routed)"]
    assert routed.ok is False  # claude_code.worker_env has no token
    assert routed.severity == "warning"
    assert "claude_code.worker_env" in routed.detail
    assert ok is True


def test_worker_github_token_rescue_routed_check_fires_when_api_worker_disabled(
    tmp_path: Path,
) -> None:
    """rescue.enabled alone must trigger the claude-code-routed check.

    _rescue_adapter_settings (workflow.py) always forces adapter="claude-code"
    for the bounded rescue tier once rescue.enabled is True, independent of
    api_worker.enabled — they are unrelated toggles. A devin-shell default
    with rescue enabled but api_worker left at its default (disabled) must
    still surface a missing claude_code.worker_env token, or a rescue-tier
    dispatch can stall silently with doctor reporting fully healthy.
    """
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(
            sessions_dir="sessions",
            worker_env={"GH_TOKEN": "placeholder-not-a-real-token"},
        ),
        worker=WorkerRoleConfig(harness="devin-shell"),
        rescue=RescueConfig(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    assert by_name["worker GitHub token"].ok is True  # devin-shell path is fine
    routed = by_name["worker GitHub token (claude-code-routed)"]
    assert routed.ok is False  # claude_code.worker_env has no token
    assert routed.severity == "warning"
    assert "claude_code.worker_env" in routed.detail
    assert ok is True


def test_worker_github_token_no_secret_in_output(tmp_path: Path) -> None:
    """The token VALUE must never appear in any doctor check detail.

    Only presence/absence and the variable NAME may be reported (#873).
    """
    secret = "ghp_super-secret-token-value-1234567890"
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(
            sessions_dir="sessions",
            worker_env={"GH_TOKEN": secret},
        ),
        # worker.harness must be set explicitly so the "worker GitHub token"
        # check actually fires -- doctor.py reads worker.harness, and this
        # test constructs OrchestratorConfig directly in Python, bypassing
        # whatever load-time defaulting/validation build_config_from_data
        # would otherwise perform. Without this the check is silently absent
        # and the loop below trivially finds no leak in nothing.
        worker=WorkerRoleConfig(harness="devin-shell"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    _ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    for check in checks:
        assert secret not in check.detail, f"Secret leaked in check {check.name!r}: {check.detail}"
        assert secret not in check.name


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
