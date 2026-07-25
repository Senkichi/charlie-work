from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from charlie_work.config import (
    ApiBudgetConfig,
    AutoMergeConfig,
    ClaudeCodeConfig,
    CrossFamilyConfig,
    DevinConfig,
    OrchestratorConfig,
    RuntimeConfig,
)
from charlie_work.config import ApiProviderConfig, ApiWorkerConfig
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


def test_doctor_surfaces_post_mortem_terminal_cause_and_attempt_ref(
    tmp_path: Path, monkeypatch
) -> None:
    """Issue #261 F6: a dead session with a written post-mortem sidecar and a
    preserved attempt ref must surface both in the "dead session post-mortems"
    doctor check — otherwise operators have no visibility into a
    push-gate-hook kill or salvaged unpushed commits without reading raw
    sidecar JSON by hand."""
    import subprocess

    from charlie_work.attempt_refs import snapshot_attempt_ref
    from charlie_work.post_mortem import PostMortemRecord

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    _write_sidecar(
        sessions_dir,
        "issue-7.json",
        {
            "issue_number": 7,
            "branch": "agent/issue-7",
            "prompt_path": "p.md",
            "command": ["devin"],
            "pid": None,
            "started_at": "2026-01-01T00:00:00Z",
            "log_path": "issue-7.log",
            "error": "devin not found",
        },
    )
    # A written post-mortem sidecar for the same dead session.
    record = PostMortemRecord(
        issue_number=7,
        generated_at="2026-01-01T00:05:00+00:00",
        db_path=str(tmp_path / "sessions.db"),
        matched=True,
        session_id="sess-7",
        failure_kind="worker_blocked",
        terminal_tool="bash",
        terminal_reason="push-gate hook rejected: rm -rf attempted",
    )
    (sessions_dir / "issue-7.post-mortem.json").write_text(
        json.dumps(record.to_dict()), encoding="utf-8"
    )

    # A real attempt ref preserved for issue 7, in a real git repo at repo_root
    # (list_attempt_refs shells out to `git for-each-ref`).
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.test"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True
    )
    (tmp_path / "README.md").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    snapshot = snapshot_attempt_ref(tmp_path, "HEAD", issue_number=7)
    assert snapshot.ref_name is not None  # sanity: the fixture actually wrote a ref

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
    post_mortems = by_name["dead session post-mortems"]
    assert "issue #7" in post_mortems.detail
    assert "failure_kind=worker_blocked" in post_mortems.detail
    assert "terminal_tool=bash" in post_mortems.detail
    assert "push-gate hook rejected" in post_mortems.detail
    assert snapshot.ref_name in post_mortems.detail
    assert post_mortems.severity == "warning"  # never blocks the run


def test_doctor_surface_post_mortems_absent_degrades_silently(tmp_path: Path, monkeypatch) -> None:
    """Issue #261 F6: a dead session with NO post-mortem sidecar and no
    attempt refs must not surface a "dead session post-mortems" check at
    all — post-mortem extraction is best-effort/opportunistic (issue #261),
    never a doctor failure or a misleading empty finding."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    _write_sidecar(
        sessions_dir,
        "issue-9.json",
        {
            "issue_number": 9,
            "branch": "agent/issue-9",
            "prompt_path": "p.md",
            "command": ["devin"],
            "pid": None,
            "started_at": "2026-01-01T00:00:00Z",
            "log_path": "issue-9.log",
            "error": "devin not found",
        },
    )
    # No .post-mortem.json sidecar written, and no repo_root git refs — the
    # extraction never ran / found nothing.

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
    assert "dead session post-mortems" not in by_name
    assert ok is True  # a failed launch alone is a warning, never a hard failure here


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


def test_doctor_reports_config_driven_worker_model(tmp_path: Path) -> None:
    """When devin.worker_model is set, doctor must report the config-driven model."""
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(
            adapter="devin-shell",
            sessions_dir="sessions",
            worker_model="claude-sonnet-4-5",
        ),
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
            adapter="devin-shell",
            sessions_dir="sessions",
            worker_model="",
        ),
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
            adapter="claude-code",
            sessions_dir="sessions",
            worker_model="claude-sonnet-4-5",
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    names = {check.name for check in checks}
    assert "devin-shell worker model" not in names


def test_gh_field_lists_use_constants_no_inline_literals() -> None:
    """All gh --json field lists must use module-level constants, not inline literals.

    This test scans src/charlie_work/*.py for gh.run() calls with --json arguments
    and verifies that the field list value (the argument after --json) references a
    constant from github.py rather than a string literal. This prevents the contract
    drift issue described in #64 by ensuring all field lists are centralized and not
    scattered as inline strings.
    """
    import ast
    from pathlib import Path

    import charlie_work.github as github_module

    # Get the expected constant names from github.py
    expected_constants = {
        "ISSUE_LIST_FIELDS",
        "ISSUE_VIEW_FIELDS",
        "PR_LIST_FIELDS",
        "PR_VIEW_FIELDS",
        "PR_CHECKS_FIELDS",
        "LABEL_LIST_FIELDS",
        "RECONCILE_PR_FIELDS",
        "RECONCILE_ISSUE_FIELDS",
    }

    # Verify constants exist
    for const in expected_constants:
        assert hasattr(github_module, const), f"Missing constant: {const}"
        value = getattr(github_module, const)
        assert isinstance(value, str), f"{const} must be a string"
        assert value, f"{const} must not be empty"

    # Scan all Python files in src/charlie_work/ for gh.run() calls with --json
    src_dir = Path(__file__).parent.parent / "src" / "charlie_work"
    violations: list[tuple[str, int, str]] = []

    for py_file in src_dir.glob("*.py"):
        if py_file.name == "github.py":
            # Constant definitions are allowed in github.py
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (OSError, SyntaxError):
            continue

        # Walk the AST to find gh.run() calls
        for node in ast.walk(tree):
            # Look for Call nodes where the function is gh.run
            if isinstance(node, ast.Call):
                # Check if this is a call to something named 'run'
                if isinstance(node.func, ast.Attribute) and node.func.attr == "run":
                    # Look for --json in the arguments and check the next argument
                    args = node.args
                    for i, arg in enumerate(args):
                        # Handle string constants (Python 3.8+)
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            if arg.value == "--json" and i + 1 < len(args):
                                # Check if the next argument is a string literal (violation)
                                next_arg = args[i + 1]
                                if isinstance(next_arg, ast.Constant) and isinstance(
                                    next_arg.value, str
                                ):
                                    # This is a field list as a string literal - violation
                                    violations.append(
                                        (str(py_file), next_arg.lineno, next_arg.value)
                                    )
                                elif isinstance(next_arg, ast.JoinedStr):
                                    # f-string field list - violation
                                    violations.append(
                                        (str(py_file), next_arg.lineno, "f-string field list")
                                    )
                    # Also check keyword arguments with list values
                    for keyword in node.keywords:
                        if keyword.arg in ("args",) and isinstance(keyword.value, ast.List):
                            list_items = keyword.value.elts
                            for i, item in enumerate(list_items):
                                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                                    if item.value == "--json" and i + 1 < len(list_items):
                                        next_item = list_items[i + 1]
                                        if isinstance(next_item, ast.Constant) and isinstance(
                                            next_item.value, str
                                        ):
                                            violations.append(
                                                (str(py_file), next_item.lineno, next_item.value)
                                            )
                                        elif isinstance(next_item, ast.JoinedStr):
                                            violations.append(
                                                (
                                                    str(py_file),
                                                    next_item.lineno,
                                                    "f-string field list",
                                                )
                                            )

    if violations:
        violation_msg = "\n".join(
            f"  {file}:{line}: {repr(literal)}" for file, line, literal in violations
        )
        raise AssertionError(
            f"Found {len(violations)} field list string literal(s) in gh.run() calls that should use constants:\n"
            f"{violation_msg}\n"
            f"Use the constants from github.py instead (e.g., ISSUE_LIST_FIELDS)."
        )


def test_fake_github_payloads_align_with_field_constants() -> None:
    """FakeGitHub test double payloads must not have keys outside the field constants.

    This prevents the third instance of the #64 bug class: a fake growing a key
    the real gh CLI never returns, which lets dead code pass CI and crash live.
    The dispatch_rework KeyError hotfix on main was caused by exactly this.
    """
    import charlie_work.github as github_module

    # Get the field sets from the constants
    pr_list_fields = set(github_module.PR_LIST_FIELDS.split(","))
    issue_list_fields = set(github_module.ISSUE_LIST_FIELDS.split(","))
    reconcile_pr_fields = set(github_module.RECONCILE_PR_FIELDS.split(","))
    reconcile_issue_fields = set(github_module.RECONCILE_ISSUE_FIELDS.split(","))

    # Check the main FakeGitHub in test_charlie_work.py
    # Import dynamically to avoid module import issues
    test_charlie_work_path = Path(__file__).parent / "test_charlie_work.py"
    spec = importlib.util.spec_from_file_location("test_charlie_work", test_charlie_work_path)
    test_charlie_work = importlib.util.module_from_spec(spec)
    sys.modules["test_charlie_work"] = test_charlie_work
    spec.loader.exec_module(test_charlie_work)

    MainFakeGitHub = test_charlie_work.FakeGitHub
    fake_gh = MainFakeGitHub()

    # Verify PR payload keys are subset of PR_LIST_FIELDS
    pr_keys = set(fake_gh.prs[0].keys())
    extra_pr_keys = pr_keys - pr_list_fields
    assert not extra_pr_keys, (
        f"FakeGitHub.prs[0] has keys not in PR_LIST_FIELDS: {extra_pr_keys}. "
        f"Either remove these keys from the fake or add them to PR_LIST_FIELDS."
    )

    # Verify issue payload keys are subset of ISSUE_LIST_FIELDS
    issue_keys = set(fake_gh.issues[0].keys())
    extra_issue_keys = issue_keys - issue_list_fields
    assert not extra_issue_keys, (
        f"FakeGitHub.issues[0] has keys not in ISSUE_LIST_FIELDS: {extra_issue_keys}. "
        f"Either remove these keys from the fake or add them to ISSUE_LIST_FIELDS."
    )

    # Check the FakeGitHub in test_reconcile.py
    test_reconcile_path = Path(__file__).parent / "test_reconcile.py"
    spec = importlib.util.spec_from_file_location("test_reconcile", test_reconcile_path)
    test_reconcile = importlib.util.module_from_spec(spec)
    sys.modules["test_reconcile"] = test_reconcile
    spec.loader.exec_module(test_reconcile)

    _pr = test_reconcile._pr
    _issue = test_reconcile._issue

    # Verify _pr helper keys are subset of RECONCILE_PR_FIELDS (not PR_LIST_FIELDS)
    # because test_reconcile uses the reconcile field list which includes 'state'
    sample_pr = _pr(1, "OPEN")
    pr_keys = set(sample_pr.keys())
    extra_pr_keys = pr_keys - reconcile_pr_fields
    assert not extra_pr_keys, (
        f"test_reconcile._pr has keys not in RECONCILE_PR_FIELDS: {extra_pr_keys}. "
        f"Either remove these keys from the fake or add them to RECONCILE_PR_FIELDS."
    )

    # Verify _issue helper keys are subset of RECONCILE_ISSUE_FIELDS
    sample_issue = _issue(1, [])
    issue_keys = set(sample_issue.keys())
    extra_issue_keys = issue_keys - reconcile_issue_fields
    assert not extra_issue_keys, (
        f"test_reconcile._issue has keys not in RECONCILE_ISSUE_FIELDS: {extra_issue_keys}. "
        f"Either remove these keys from the fake or add them to RECONCILE_ISSUE_FIELDS."
    )


def _make_fleet_json(tmp_path: Path, state_dir: Path) -> Path:
    """Create a fleet.json with one repo pointing at the given state_dir."""
    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)
    fleet_json = fleet_dir_path / "fleet.json"
    fleet_json.write_text(
        json.dumps(
            {
                "version": 1,
                "repos": {
                    "owner/repo": {
                        "repo_root": str(tmp_path),
                        "state_dir": str(state_dir),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return fleet_json


def test_doctor_skips_fleet_supervisor_check_when_fleet_not_configured(
    tmp_path: Path,
) -> None:
    """No fleet supervisor warning when fleet.json does not exist."""
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    names = {check.name for check in checks}
    assert "fleet supervisor" not in names
    assert ok is True


def test_doctor_warns_when_fleet_configured_but_not_supervised(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """fleet.json has repos but no supervisor lock held → warning."""
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    gh = FakeDoctorGitHub(labels=config.labels.all)
    _make_fleet_json(tmp_path, paths.root)

    # All locks are acquirable, so no supervisor is running.
    monkeypatch.setattr(
        "charlie_work.doctor.try_acquire_supervisor_lock",
        lambda _path: MagicMock(),
    )

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    fleet_check = by_name["fleet supervisor"]
    assert fleet_check.ok is False
    assert fleet_check.severity == "warning"
    assert "run `charlie fleet supervise`" in fleet_check.detail
    assert ok is True


def test_doctor_passes_when_fleet_supervisor_lock_held(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """fleet-supervisor.lock held means the fleet is being driven."""
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    gh = FakeDoctorGitHub(labels=config.labels.all)
    _make_fleet_json(tmp_path, paths.root)

    # Touch the fleet supervisor lock file so the existence check triggers the probe.
    fleet_lock_path = tmp_path / "fleet" / "fleet-supervisor.lock"
    fleet_lock_path.parent.mkdir(parents=True, exist_ok=True)
    fleet_lock_path.write_text("", encoding="utf-8")

    def _fake_lock(path: Path) -> MagicMock | None:
        if path.name == "fleet-supervisor.lock":
            return None  # held
        return MagicMock()  # repo locks free

    monkeypatch.setattr(
        "charlie_work.doctor.try_acquire_supervisor_lock",
        _fake_lock,
    )

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    fleet_check = by_name["fleet supervisor"]
    assert fleet_check.ok is True
    assert "fleet supervisor appears to be running" in fleet_check.detail
    assert ok is True


def test_doctor_fleet_supervisor_per_repo_aware(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A single held repo lock does not hide unsupervised repos in the fleet."""
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)
    fleet_json = fleet_dir_path / "fleet.json"
    state_dir1 = tmp_path / "state1"
    state_dir1.mkdir(parents=True, exist_ok=True)
    state_dir2 = tmp_path / "state2"
    state_dir2.mkdir(parents=True, exist_ok=True)
    fleet_json.write_text(
        json.dumps(
            {
                "version": 1,
                "repos": {
                    "owner/repo1": {
                        "repo_root": str(tmp_path),
                        "state_dir": str(state_dir1),
                    },
                    "owner/repo2": {
                        "repo_root": str(tmp_path),
                        "state_dir": str(state_dir2),
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    # Create the per-repo lock files so the probe attempts to acquire them.
    (state_dir1 / "supervisor.lock").write_text("", encoding="utf-8")
    (state_dir2 / "supervisor.lock").write_text("", encoding="utf-8")

    def _fake_lock(path: Path) -> MagicMock | None:
        # Only repo1 has a live per-repo supervisor.
        if "state1" in str(path):
            return None
        return MagicMock()

    monkeypatch.setattr(
        "charlie_work.doctor.try_acquire_supervisor_lock",
        _fake_lock,
    )

    ok, checks = run_doctor(
        tmp_path,
        paths,
        config,
        tmp_path / "c.yaml",
        gh,
        fleet_dir_override=str(fleet_dir_path),
    )

    by_name = {check.name: check for check in checks}
    fleet_check = by_name["fleet supervisor"]
    assert fleet_check.ok is False
    assert fleet_check.severity == "warning"
    assert "owner/repo1" in fleet_check.detail
    assert "owner/repo2" in fleet_check.detail
    assert ok is True


# ---------------------------------------------------------------------------
# api_worker doctor probes (issue #483)
# ---------------------------------------------------------------------------


def _api_provider(
    *,
    api_key_env: str = "MOONSHOT_API_KEY",
    base_url: str = "https://api.moonshot.ai/anthropic",
) -> ApiProviderConfig:
    return ApiProviderConfig(
        base_url=base_url,
        api_key_env=api_key_env,
        model="kimi-k3",
        input_usd_per_mtok=3.0,
        output_usd_per_mtok=15.0,
        cached_input_usd_per_mtok=0.30,
    )


def _api_worker_config(
    *,
    enabled: bool = True,
    provider: ApiProviderConfig | None = None,
    budget: ApiBudgetConfig | None = None,
    provider_name: str = "kimi-k3",
) -> ApiWorkerConfig:
    return ApiWorkerConfig(
        enabled=enabled,
        provider=provider_name,
        providers={provider_name: provider or _api_provider()},
        budget=budget or ApiBudgetConfig(),
    )


def test_doctor_api_worker_not_configured_emits_nothing(tmp_path: Path) -> None:
    """Default (absent) api_worker section → no api_worker checks at all."""
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    names = {check.name for check in checks}
    assert not any(name.startswith("api_worker") for name in names)
    assert ok is True


def test_doctor_api_worker_disabled_emits_notice(tmp_path: Path) -> None:
    """Configured but disabled → single notice line, warning severity, ok=True."""
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        api_worker=_api_worker_config(enabled=False),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    notice = by_name["api_worker configured but disabled"]
    assert notice.ok is True
    assert notice.severity == "warning"
    assert "disabled" in notice.detail
    # No enabled-mode checks should appear.
    assert "api_worker api key" not in by_name
    assert "api_worker base url" not in by_name
    assert ok is True


def test_doctor_api_worker_enabled_all_checks_ok(tmp_path: Path, monkeypatch) -> None:
    """Enabled with key present, valid URL, no ledger → all four checks pass."""
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-value")
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        api_worker=_api_worker_config(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    assert by_name["api_worker api key"].ok is True
    assert "MOONSHOT_API_KEY" in by_name["api_worker api key"].detail
    # No secret value leaks into the detail.
    assert "sk-test-value" not in by_name["api_worker api key"].detail

    assert by_name["api_worker base url"].ok is True
    assert "https" in by_name["api_worker base url"].detail

    assert by_name["api_worker budget ledger"].ok is True
    assert "not yet created" in by_name["api_worker budget ledger"].detail

    headroom = by_name["api_worker budget headroom"]
    assert headroom.ok is True
    assert headroom.severity == "warning"
    assert "$0.00 spent today" in headroom.detail
    assert "$15.00" in headroom.detail  # lifetime cap default
    assert ok is True


def test_doctor_api_worker_missing_env_var(tmp_path: Path, monkeypatch) -> None:
    """Enabled but env var absent → api key check fails (error severity)."""
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        api_worker=_api_worker_config(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    assert by_name["api_worker api key"].ok is False
    assert "NOT set" in by_name["api_worker api key"].detail
    assert "MOONSHOT_API_KEY" in by_name["api_worker api key"].detail
    assert ok is False  # error-severity failure blocks


def test_doctor_api_worker_bad_url(tmp_path: Path, monkeypatch) -> None:
    """Enabled but base_url is not https → base url check fails."""
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-value")
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        api_worker=_api_worker_config(
            enabled=True,
            provider=_api_provider(base_url="http://insecure.example.com"),
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    assert by_name["api_worker base url"].ok is False
    assert "not a valid https URL" in by_name["api_worker base url"].detail
    assert ok is False


def test_doctor_api_worker_corrupt_ledger(tmp_path: Path, monkeypatch) -> None:
    """Enabled with a corrupt ledger file → ledger check fails (corrupt detected)."""
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-value")
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        api_worker=_api_worker_config(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    # Write a corrupt ledger file.
    ledger_file = paths.root / "api-budget.json"
    ledger_file.write_text("{ this is not valid json", encoding="utf-8")
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    ledger_check = by_name["api_worker budget ledger"]
    assert ledger_check.ok is False
    assert "corrupt" in ledger_check.detail.lower()
    assert "quarantined" in ledger_check.detail.lower()


def test_doctor_api_worker_headroom_math(tmp_path: Path, monkeypatch) -> None:
    """Enabled with a ledger showing spend → headroom check reports correct math."""
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-value")
    budget = ApiBudgetConfig(
        max_usd_per_session=0.0,
        preflight_reserve_usd=1.0,
        max_usd_per_day=5.0,
        lifetime_usd=15.0,
    )
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        api_worker=_api_worker_config(enabled=True, budget=budget),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)

    # Write a ledger with today's spend and lifetime spend.
    from datetime import UTC, datetime

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    ledger_data = {
        "days": {
            today: {
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_tokens": 0,
                "usd": 3.50,
            }
        },
        "lifetime_usd": 10.00,
        "sessions": [],
    }
    ledger_file = paths.root / "api-budget.json"
    ledger_file.write_text(json.dumps(ledger_data), encoding="utf-8")
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    headroom = by_name["api_worker budget headroom"]
    # 3.50 spent today + 1.0 reserve = 4.50 <= 5.0 cap → daily ok
    # 10.00 < 15.0 → lifetime ok
    assert headroom.ok is True
    assert "$3.50 spent today" in headroom.detail
    assert "$5.00 cap" in headroom.detail
    assert "$1.50 remaining" in headroom.detail  # 5.0 - 3.50
    assert "$10.00 spent lifetime" in headroom.detail
    assert "$5.00 remaining" in headroom.detail  # 15.0 - 10.00
    assert ok is True


def test_doctor_api_worker_headroom_exhausted(tmp_path: Path, monkeypatch) -> None:
    """Enabled with spend at caps → headroom check fails (exhausted)."""
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-value")
    budget = ApiBudgetConfig(
        max_usd_per_session=0.0,
        preflight_reserve_usd=1.0,
        max_usd_per_day=5.0,
        lifetime_usd=15.0,
    )
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        api_worker=_api_worker_config(enabled=True, budget=budget),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)

    from datetime import UTC, datetime

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    # Daily: 4.50 spent + 1.0 reserve = 5.50 > 5.0 → daily exhausted
    # Lifetime: 15.00 >= 15.0 → lifetime exhausted
    ledger_data = {
        "days": {
            today: {
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_tokens": 0,
                "usd": 4.50,
            }
        },
        "lifetime_usd": 15.00,
        "sessions": [],
    }
    ledger_file = paths.root / "api-budget.json"
    ledger_file.write_text(json.dumps(ledger_data), encoding="utf-8")
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    headroom = by_name["api_worker budget headroom"]
    assert headroom.ok is False
    assert "exhausted" in headroom.detail
    # Headroom is a warning, so it doesn't block ok by itself — but the
    # daily exhaustion means the api key check etc. still pass. The headroom
    # check is warning severity, so ok should still be True unless another
    # error-severity check fails.
    assert headroom.severity == "warning"


def test_doctor_api_worker_no_secret_in_output(tmp_path: Path, monkeypatch) -> None:
    """The API key VALUE must never appear in any doctor check detail."""
    secret = "sk-super-secret-key-value-9999"
    monkeypatch.setenv("MOONSHOT_API_KEY", secret)
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        api_worker=_api_worker_config(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    _ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    for check in checks:
        if check.name.startswith("api_worker"):
            assert secret not in check.detail, (
                f"Secret leaked in check {check.name!r}: {check.detail}"
            )
