from __future__ import annotations

import json
from pathlib import Path

from charlie_work.config import (
    AutoMergeConfig,
    ClaudeCodeConfig,
    CrossFamilyConfig,
    DevinConfig,
    OrchestratorConfig,
    RuntimeConfig,
)
from charlie_work.doctor import _check_name_matches, run_doctor, workflow_job_names
from charlie_work.paths import runtime_paths
from charlie_work.subprocess_runner import RunResult


class FakeDoctorGitHub:
    def __init__(self, labels: list[str] | None = None) -> None:
        self.labels = labels if labels is not None else []

    def run(self, args, **kwargs):
        return ""

    def label_list(self):
        return [{"name": name} for name in self.labels]


def _write_workflow(repo_root: Path, body: str) -> None:
    workflows = repo_root / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yml").write_text(body, encoding="utf-8")


def test_workflow_job_names_prefers_name_over_id(tmp_path: Path) -> None:
    _write_workflow(
        tmp_path,
        "name: CI\njobs:\n  test:\n    name: Tests passed\n    runs-on: ubuntu-latest\n"
        "  lint:\n    runs-on: ubuntu-latest\n",
    )

    names = workflow_job_names(tmp_path)

    assert names == {"Tests passed", "lint"}


def test_workflow_job_names_empty_without_workflows(tmp_path: Path) -> None:
    assert workflow_job_names(tmp_path) == set()


def test_check_name_matches_exact_and_matrix_prefix() -> None:
    assert _check_name_matches("Tests passed", {"Tests passed"}) is True
    # Matrix job configured as "Test"; check runs report "Test (windows-latest)".
    assert _check_name_matches("Test (windows-latest)", {"Test"}) is True
    assert _check_name_matches("Test", {"Test (windows-latest)"}) is True
    assert _check_name_matches("Pre-commit", {"Tests passed", "lint"}) is False


def _config(**kwargs) -> OrchestratorConfig:
    return OrchestratorConfig(**kwargs)


def test_doctor_flags_mismatched_required_check(tmp_path: Path) -> None:
    _write_workflow(tmp_path, "jobs:\n  test:\n    name: Test\n    runs-on: ubuntu-latest\n")
    config = _config(auto_merge=AutoMergeConfig(required_checks=("Tests passed", "Test")))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    assert by_name["check name: Test"].ok is True
    assert by_name["check name: Tests passed"].ok is False
    assert ok is False


def test_doctor_flags_empty_required_checks_with_auto_merge(tmp_path: Path) -> None:
    config = _config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    assert by_name["required checks configured"].ok is False
    assert ok is False


def test_doctor_flags_missing_labels_as_bootstrap_hint(tmp_path: Path) -> None:
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=["automated-ready"])

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    assert by_name["github labels"].ok is False
    assert "bootstrap-labels" in by_name["github labels"].detail


def test_doctor_missing_config_is_warning_not_blocking(tmp_path: Path) -> None:
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, None, gh)

    by_name = {check.name: check for check in checks}
    assert by_name["config file"].ok is False
    assert by_name["config file"].severity == "warning"
    assert ok is True  # warnings alone do not block


def test_doctor_flags_missing_prompts_dir_and_template(tmp_path: Path) -> None:
    config = _config(runtime=RuntimeConfig(prompts_dir="nope-prompts"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    assert by_name["prompts dir"].ok is False
    assert by_name["worker template: worker.md"].ok is True  # package fallback still resolves


def test_doctor_cross_family_missing_binary_is_warning(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "charlie_work.doctor.shutil.which",
        lambda name: None if name == "devin" else f"C:/fake/{name}",
    )
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        cross_family=CrossFamilyConfig(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    assert by_name["cross-family binary"].ok is False
    assert by_name["cross-family binary"].severity == "warning"
    assert ok is True


def _write_sidecar(sessions_dir: Path, name: str, payload: dict) -> None:
    (sessions_dir / name).write_text(json.dumps(payload), encoding="utf-8")


def test_doctor_adapter_probe_runs_devin_probe_and_surfaces_sessions(
    tmp_path: Path, monkeypatch
) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    # A failed launch (error set) and a launched-but-dead one (implausible PID
    # that OpenProcess/os.kill can never find -> is_session_alive False).
    _write_sidecar(
        sessions_dir,
        "issue-1.json",
        {
            "issue_number": 1,
            "branch": "agent/issue-1",
            "prompt_path": "p.md",
            "command": ["devin"],
            "pid": None,
            "started_at": "2026-01-01T00:00:00Z",
            "log_path": "issue-1.log",
            "error": "devin not found",
        },
    )
    _write_sidecar(
        sessions_dir,
        "issue-2.json",
        {
            "issue_number": 2,
            "branch": "agent/issue-2",
            "prompt_path": "p.md",
            "command": ["devin"],
            "pid": 999_999_999,
            "started_at": "2026-01-01T00:00:00Z",
            "log_path": "issue-2.log",
            "error": None,
        },
    )
    monkeypatch.setattr(
        "charlie_work.devin_shell.probe_devin",
        lambda repo_root, **kwargs: RunResult(returncode=0, stdout="devin 1.2.3", stderr=""),
    )

    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(adapter="devin-shell", sessions_dir="sessions"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh, adapter_probe=True)

    by_name = {check.name: check for check in checks}
    assert by_name["devin CLI probe"].ok is True
    assert "devin 1.2.3" in by_name["devin CLI probe"].detail
    sessions = by_name["launched sessions"]
    assert sessions.ok is False  # one failed record present
    assert "1 failed" in sessions.detail
    assert "1 exited" in sessions.detail
    assert sessions.severity == "warning"  # never blocks the run
    assert ok is True  # a warning-only sessions finding must not fail doctor


def test_doctor_adapter_probe_reports_failed_devin_binary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "charlie_work.devin_shell.probe_devin",
        lambda repo_root, **kwargs: RunResult(
            returncode=None, stdout="", stderr="", error="devin: not found"
        ),
    )
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(adapter="devin-shell", sessions_dir="sessions"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh, adapter_probe=True)

    by_name = {check.name: check for check in checks}
    assert by_name["devin CLI probe"].ok is False
    assert "not found" in by_name["devin CLI probe"].detail
    assert ok is False  # a broken adapter CLI is an error-severity block
    # No sessions dir was created -> surfaced as a benign warning, not a crash.
    assert by_name["launched sessions"].severity == "warning"


def test_doctor_adapter_probe_claude_code_probes_claude(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "charlie_work.claude_code.probe_claude",
        lambda repo_root, **kwargs: RunResult(returncode=0, stdout="claude 2.0", stderr=""),
    )
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(adapter="claude-code", sessions_dir="sessions"),
        # Empty venv_source skips the venv-existence check so this test stays
        # scoped to the probe path.
        claude_code=ClaudeCodeConfig(venv_source=""),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh, adapter_probe=True)

    by_name = {check.name: check for check in checks}
    assert by_name["claude CLI probe"].ok is True
    assert "claude 2.0" in by_name["claude CLI probe"].detail


def test_doctor_without_adapter_probe_omits_probe_checks(tmp_path: Path) -> None:
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(adapter="devin-shell", sessions_dir="sessions"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    names = {check.name for check in checks}
    assert "devin CLI probe" not in names
    assert "launched sessions" not in names


# ---------------------------------------------------------------------------
# Issue-17 regression tests
# ---------------------------------------------------------------------------


def test_doctor_corrupt_state_is_not_quarantined(tmp_path: Path) -> None:
    """doctor must report a failure on corrupt state WITHOUT renaming the file."""
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    # Write a corrupt (non-JSON) state file.
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    paths.state_file.write_text("NOT JSON {{{", encoding="utf-8")

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    # The original file must still exist — doctor must not quarantine it.
    assert paths.state_file.exists(), "doctor must not rename/quarantine the state file"
    # Doctor should surface the corruption as a check failure.
    by_name = {check.name: check for check in checks}
    assert by_name["state file"].ok is False


def test_doctor_surfaces_existing_quarantine_files_as_warning(tmp_path: Path) -> None:
    """Pre-existing *.corrupt-* files must appear as a warning-severity check."""
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    # Simulate a previously-quarantined corrupt state file.
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    corrupt = paths.state_file.parent / f"{paths.state_file.name}.corrupt-20260101T000000Z"
    corrupt.write_text("{}", encoding="utf-8")

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    assert "state file quarantine" in by_name
    quarantine_check = by_name["state file quarantine"]
    assert quarantine_check.ok is False
    assert quarantine_check.severity == "warning"
    assert "1 quarantined" in quarantine_check.detail
    # A warning-only finding must not block doctor.
    assert ok is True


def test_doctor_no_quarantine_check_when_none_exist(tmp_path: Path) -> None:
    """No quarantine check emitted when there are no corrupt-* files."""
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    names = {check.name for check in checks}
    assert "state file quarantine" not in names


def test_doctor_adapter_probe_uses_configured_devin_binary(tmp_path: Path, monkeypatch) -> None:
    """probe_devin must be called with the binary from devin.shell_command."""
    captured: list[tuple[str, ...]] = []

    def fake_probe_devin(repo_root, **kwargs):
        captured.append(kwargs.get("command", ()))
        return RunResult(returncode=0, stdout="custom-devin 9.9", stderr="")

    monkeypatch.setattr("charlie_work.devin_shell.probe_devin", fake_probe_devin)

    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(
            adapter="devin-shell",
            sessions_dir="sessions",
            shell_command=("my-devin-wrapper", "--prompt-file", "{prompt_path}", "--print"),
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh, adapter_probe=True)

    assert len(captured) == 1
    assert captured[0][0] == "my-devin-wrapper", (
        "probe must use the configured binary, not the hardcoded default"
    )


def test_doctor_adapter_probe_uses_configured_claude_binary(tmp_path: Path, monkeypatch) -> None:
    """probe_claude must be called with the binary from claude_code.command."""
    captured: list[tuple[str, ...]] = []

    def fake_probe_claude(repo_root, **kwargs):
        captured.append(kwargs.get("command", ()))
        return RunResult(returncode=0, stdout="my-claude 5.0", stderr="")

    monkeypatch.setattr("charlie_work.claude_code.probe_claude", fake_probe_claude)

    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(adapter="claude-code", sessions_dir="sessions"),
        claude_code=ClaudeCodeConfig(
            command=("my-claude-wrapper", "-p", "--permission-mode", "acceptEdits"),
            venv_source="",
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh, adapter_probe=True)

    assert len(captured) == 1
    assert captured[0][0] == "my-claude-wrapper", (
        "probe must use the configured binary, not the hardcoded default"
    )
