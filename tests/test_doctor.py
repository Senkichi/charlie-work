from __future__ import annotations

from pathlib import Path

from devin_orchestrator.config import (
    AutoMergeConfig,
    CrossFamilyConfig,
    OrchestratorConfig,
    RuntimeConfig,
)
from devin_orchestrator.doctor import _check_name_matches, run_doctor, workflow_job_names
from devin_orchestrator.paths import runtime_paths


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
        "devin_orchestrator.doctor.shutil.which",
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
