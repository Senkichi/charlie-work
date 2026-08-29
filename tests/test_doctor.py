from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from _script_loader import load_script_module
from charlie_work.config import (
    ApiBudgetConfig,
    AutoMergeConfig,
    ClaudeCodeConfig,
    CrossFamilyConfig,
    DevinConfig,
    OrchestratorConfig,
    RescueConfig,
    ReviewDispatchConfig,
    RuntimeConfig,
)
from charlie_work.config import ApiProviderConfig, ApiWorkerConfig
from charlie_work.doctor import (
    _check_name_match_kind,
    _check_name_matches,
    _tolerance_match_base_names,
    run_doctor,
    workflow_has_matrix,
    workflow_job_matrix_flags,
    workflow_job_names,
)
from charlie_work.instrumentation import log_event
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


def _write_workflow_named(repo_root: Path, filename: str, body: str) -> None:
    """Write a workflow file by name, tolerating an existing workflows dir.

    Unlike ``_write_workflow`` (which always writes ``ci.yml`` and would raise
    on a second ``mkdir``), this lets a test plant several workflow files in
    one repo -- the multi-workflow scenario the issue #1508 matrix-scoping fix
    targets.
    """
    workflows = repo_root / ".github" / "workflows"
    workflows.mkdir(parents=True, exist_ok=True)
    (workflows / filename).write_text(body, encoding="utf-8")


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


def test_check_name_match_kind_classifies_exact_tolerance_and_none() -> None:
    # Issue #1508: the doctor verifier needs to distinguish an exact match
    # (which the merge gate in checks.py also accepts) from a tolerance-only
    # match (which the merge gate rejects) so it can flag stale suffixed
    # required-check entries when no strategy.matrix exists.
    assert _check_name_match_kind("Tests passed", {"Tests passed"}) == "exact"
    assert _check_name_match_kind("Test (windows-latest)", {"Test"}) == "tolerance"
    assert _check_name_match_kind("Test", {"Test (windows-latest)"}) == "tolerance"
    # Reusable workflow reports as "<caller> / <callee>"; required is the caller.
    assert _check_name_match_kind("caller", {"caller / callee"}) == "tolerance"
    assert _check_name_match_kind("Pre-commit", {"Tests passed", "lint"}) is None


def test_workflow_has_matrix_detects_strategy_matrix(tmp_path: Path) -> None:
    _write_workflow(
        tmp_path,
        "name: CI\njobs:\n  test:\n    name: Tests\n    runs-on: ubuntu-latest\n"
        "    strategy:\n"
        "      matrix:\n"
        "        os: [ubuntu-latest, windows-latest]\n",
    )
    assert workflow_has_matrix(tmp_path) is True


def test_workflow_has_matrix_false_without_matrix(tmp_path: Path) -> None:
    _write_workflow(
        tmp_path,
        "name: CI\njobs:\n  test:\n    name: Tests\n    runs-on: ubuntu-latest\n",
    )
    assert workflow_has_matrix(tmp_path) is False


def test_workflow_has_matrix_false_without_workflows(tmp_path: Path) -> None:
    assert workflow_has_matrix(tmp_path) is False


def test_workflow_job_matrix_flags_scoped_per_job(tmp_path: Path) -> None:
    # Issue #1508: matrix-ness is a PER-JOB property, not repo-wide. Two
    # workflows in one repo -- only the second job has strategy.matrix -- so
    # the flags dict must mark "Matrix job" True and "Plain job" False
    # independently. The repo-wide workflow_has_matrix() stays True (some job
    # has a matrix), which is exactly the combination the old global-boolean
    # verifier misused to justify a tolerance match against the plain job.
    _write_workflow_named(
        tmp_path,
        "plain.yml",
        "name: Plain\njobs:\n  plain:\n    name: Plain job\n    runs-on: ubuntu-latest\n",
    )
    _write_workflow_named(
        tmp_path,
        "matrix.yml",
        "name: Matrix\njobs:\n  mx:\n    name: Matrix job\n    runs-on: ubuntu-latest\n"
        "    strategy:\n      matrix:\n        os: [ubuntu-latest, windows-latest]\n",
    )

    flags = workflow_job_matrix_flags(tmp_path)

    assert flags == {"Plain job": False, "Matrix job": True}
    # The repo-wide predicate is still True (some job has a matrix) -- this is
    # the trap: a global boolean cannot distinguish which job justified it.
    assert workflow_has_matrix(tmp_path) is True


def test_workflow_job_matrix_flags_empty_without_workflows(tmp_path: Path) -> None:
    assert workflow_job_matrix_flags(tmp_path) == {}


def test_tolerance_match_base_names_returns_matched_jobs() -> None:
    # The base names are the workflow job display names the suffix/delimiter
    # expands from -- the jobs whose own matrix flag decides justification.
    assert _tolerance_match_base_names("Tests (windows-latest)", {"Tests"}) == ["Tests"]
    assert _tolerance_match_base_names("Tests", {"Tests (windows-latest)"}) == [
        "Tests (windows-latest)"
    ]
    # Reusable-workflow delimiter: "<caller> / <callee>".
    assert _tolerance_match_base_names("caller", {"caller / callee"}) == ["caller / callee"]
    # A bare prefix is NOT a tolerance match.
    assert _tolerance_match_base_names("Tests passed", {"Tests"}) == []


def _config(**kwargs) -> OrchestratorConfig:
    # The real default (review_dispatch.enabled=False, cross_family.auto_verdict=
    # False) is exactly the "no automated review-to-verdict path" gap the new
    # doctor check flags -- so every pre-existing test in this module that
    # doesn't care about that check would otherwise trip it incidentally.
    # Default review_dispatch on here; tests that specifically exercise the
    # new check override it explicitly (see
    # test_doctor_flags_no_automated_review_to_verdict_path).
    kwargs.setdefault("review_dispatch", ReviewDispatchConfig(enabled=True))
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


def test_doctor_flags_tolerance_only_match_when_no_matrix(tmp_path: Path) -> None:
    # Issue #1508: a suffixed required-check entry ("Tests (windows-latest)")
    # that matches a non-matrix job ("Tests") only via tolerance is a stale
    # entry the exact-match merge gate (checks.py) would report `missing`
    # forever. Doctor must FAIL it, not pass it -- the same trap #1507
    # corrected in the README.
    _write_workflow(tmp_path, "jobs:\n  test:\n    name: Tests\n    runs-on: ubuntu-latest\n")
    config = _config(auto_merge=AutoMergeConfig(required_checks=("Tests (windows-latest)",)))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    check = by_name["check name: Tests (windows-latest)"]
    assert check.ok is False
    assert "matrix-suffix tolerance" in check.detail
    assert "no strategy.matrix" in check.detail
    assert ok is False


def test_doctor_passes_tolerance_match_when_matrix_exists(tmp_path: Path) -> None:
    # When the workflow actually has a strategy.matrix, the suffix tolerance
    # is justified by a real matrix expansion, so doctor passes the suffixed
    # required-check entry.
    _write_workflow(
        tmp_path,
        "jobs:\n  test:\n    name: Tests\n    runs-on: ubuntu-latest\n"
        "    strategy:\n"
        "      matrix:\n"
        "        os: [ubuntu-latest, windows-latest]\n",
    )
    config = _config(auto_merge=AutoMergeConfig(required_checks=("Tests (windows-latest)",)))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    check = by_name["check name: Tests (windows-latest)"]
    assert check.ok is True
    assert "matrix-suffix tolerance" in check.detail
    assert "strategy.matrix" in check.detail


def test_doctor_flags_tolerance_match_when_matrix_is_in_a_sibling_workflow(tmp_path: Path) -> None:
    # Issue #1508 rework: the matrix justification must be scoped to the
    # SPECIFIC job that produced the tolerance match, not computed repo-wide.
    # Two workflow files live in the repo: ``plain.yml`` defines a non-matrix
    # job "Tests", and ``matrix.yml`` defines an UNRELATED matrix job
    # "Build (matrix)". The repo therefore HAS a strategy.matrix somewhere, so
    # the old global-boolean verifier passed a tolerance-only required-check
    # entry ("Tests (windows-latest)") matching the plain "Tests" job -- a
    # false pass, because the exact-match merge gate (checks.py) would report
    # "Tests (windows-latest)" missing forever (the "Tests" job has no matrix
    # to expand into that suffixed name). Doctor must FAIL it.
    _write_workflow_named(
        tmp_path,
        "plain.yml",
        "name: Plain\njobs:\n  test:\n    name: Tests\n    runs-on: ubuntu-latest\n",
    )
    _write_workflow_named(
        tmp_path,
        "matrix.yml",
        "name: Matrix\njobs:\n  build:\n    name: Build (matrix)\n    runs-on: ubuntu-latest\n"
        "    strategy:\n      matrix:\n        os: [ubuntu-latest, windows-latest]\n",
    )
    config = _config(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests (windows-latest)", "Build (matrix) (ubuntu-latest)")
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    # The non-matrix workflow's tolerance-only match MUST fail -- the sibling
    # workflow's matrix does not justify it.
    plain_check = by_name["check name: Tests (windows-latest)"]
    assert plain_check.ok is False
    assert "matrix-suffix tolerance" in plain_check.detail
    assert "no strategy.matrix" in plain_check.detail
    # The matrix workflow's own tolerance match still passes -- scoping works
    # both ways: a real matrix on the matched job justifies the suffix.
    matrix_check = by_name["check name: Build (matrix) (ubuntu-latest)"]
    assert matrix_check.ok is True
    assert "matrix-suffix tolerance" in matrix_check.detail
    assert "strategy.matrix" in matrix_check.detail
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


def test_doctor_flags_no_automated_review_to_verdict_path(tmp_path: Path) -> None:
    # review_dispatch.enabled=False + cross_family.auto_verdict=False (both
    # real defaults on OrchestratorConfig) is exactly the 7-day-outage config
    # shape -- no automated path ever calls record_review(). _config()
    # defaults review_dispatch on for the rest of this module, so this test
    # overrides it back off explicitly.
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        review_dispatch=ReviewDispatchConfig(enabled=False),
    )
    assert config.review_dispatch.enabled is False
    assert config.cross_family.auto_verdict is False
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    check = by_name["review-to-verdict path"]
    assert check.ok is False
    assert check.severity == "error"
    assert "review_dispatch.enabled" in check.detail
    assert "cross_family.auto_verdict" in check.detail
    assert ok is False


def test_doctor_review_to_verdict_path_ok_when_auto_verdict_enabled(tmp_path: Path) -> None:
    # review_dispatch off, cross_family.auto_verdict on: the check must pass
    # on auto_verdict alone, independent of review_dispatch.
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        review_dispatch=ReviewDispatchConfig(enabled=False),
        cross_family=CrossFamilyConfig(auto_verdict=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    assert by_name["review-to-verdict path"].ok is True
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


# --- worker GitHub token (issue #873 Part 2) ---------------------------------
#
# sanitize_env (issue #502) strips GH_TOKEN/GITHUB_TOKEN from every worker
# subprocess; the only sanctioned way back in is devin.worker_env /
# claude_code.worker_env. These checks must never touch sanitize_env, must
# never read the process environment, and must never log a token value.


def test_worker_github_token_ok_when_configured_devin_shell(tmp_path: Path) -> None:
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(
            adapter="devin-shell",
            sessions_dir="sessions",
            worker_env={"GH_TOKEN": "placeholder-not-a-real-token"},
        ),
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
        devin=DevinConfig(adapter="devin-shell", sessions_dir="sessions"),
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
        devin=DevinConfig(adapter="claude-code", sessions_dir="sessions"),
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
            devin=DevinConfig(adapter=adapter, sessions_dir="sessions"),
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
    """A devin-shell default with api_worker enabled still routes some issues to api.

    routing.select_adapter can send individual issues to the api adapter
    (policy:rework/policy:complexity) whenever api_worker.enabled is True,
    regardless of the configured default adapter. That subset sources
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
            adapter="devin-shell",
            sessions_dir="sessions",
            worker_env={"GH_TOKEN": "placeholder-not-a-real-token"},
        ),
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
            adapter="devin-shell",
            sessions_dir="sessions",
            worker_env={"GH_TOKEN": "placeholder-not-a-real-token"},
        ),
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
            adapter="devin-shell",
            sessions_dir="sessions",
            worker_env={"GH_TOKEN": secret},
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    _ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    for check in checks:
        assert secret not in check.detail, f"Secret leaked in check {check.name!r}: {check.detail}"
        assert secret not in check.name


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
    test_charlie_work = load_script_module(test_charlie_work_path, "test_charlie_work")

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
    test_reconcile = load_script_module(test_reconcile_path, "test_reconcile")

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
    """Enabled with a ledger showing spend -> headroom check reports correct math.

    Regression for issue #828 (originally #822's class): production derives
    its own `today = now.strftime("%Y-%m-%d")` ledger key independently of
    this test's fixture write. If the wall clock crosses UTC midnight between
    the write and `run_doctor`'s read, the lookup misses and the report shows
    $0.00 instead of the expected spend -- a real (if rare) production defect,
    not just a test flake. `now` is frozen and passed to both the fixture and
    `run_doctor` so the ledger key always matches regardless of any stall or
    midnight boundary in between.
    """
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

    frozen_now = datetime(2026, 7, 29, 12, 0, 0, tzinfo=UTC)
    today = frozen_now.strftime("%Y-%m-%d")
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

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh, now=frozen_now)

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
    """Enabled with spend at caps -> headroom check fails (exhausted).

    Regression for issue #828 (originally #822's class): same UTC-midnight
    ledger-key race as ``test_doctor_api_worker_headroom_math`` above -- see
    that test's docstring. ``now`` is frozen and passed to both the fixture
    and ``run_doctor``.
    """
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

    frozen_now = datetime(2026, 7, 29, 12, 0, 0, tzinfo=UTC)
    today = frozen_now.strftime("%Y-%m-%d")
    # Daily: 4.50 spent + 1.0 reserve = 5.50 > 5.0 -> daily exhausted
    # Lifetime: 15.00 >= 15.0 -> lifetime exhausted
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

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh, now=frozen_now)

    by_name = {check.name: check for check in checks}
    headroom = by_name["api_worker budget headroom"]
    assert headroom.ok is False
    # Both daily and lifetime are exhausted in this fixture; the detail must
    # report each leg's exhaustion, not just one occurrence of the word.
    assert headroom.detail.count("exhausted") >= 2
    assert "$4.50 spent today" in headroom.detail
    assert "$0.50 remaining" in headroom.detail  # max(0, 5.0 - 4.50)
    assert "$15.00 spent lifetime" in headroom.detail
    assert "$0.00 remaining" in headroom.detail  # max(0, 15.0 - 15.00)
    # Headroom is warning severity, so it does not block the overall ok —
    # the api key / base url / ledger checks all pass, so ok stays True.
    assert headroom.severity == "warning"
    assert ok is True


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


# --- host-wide runner allocation staleness probe (issue #590) ---------------
#
# The probe exists because every way the allocation prologue can decline to act is
# silent, so "configured but inert" has to be *detected* rather than inferred from
# logs that have proven lossy.


def _doctor_allocation_config(*, enabled: bool = True, budget: int = 8) -> Any:
    from dataclasses import replace

    from charlie_work.config import OrchestratorConfig

    base = OrchestratorConfig()
    return replace(
        base,
        runner_allocation=replace(
            base.runner_allocation, enabled=enabled, max_running_runners=budget
        ),
    )


def _collect_allocation_checks(
    config: Any, fleet_dir: Path, *, now: Any = None
) -> list[tuple[str, bool, str]]:
    from charlie_work.doctor import _check_runner_allocation

    collected: list[tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str, *, severity: str = "error") -> None:
        collected.append((name, ok, detail))

    _check_runner_allocation(add, config, fleet_dir_override=str(fleet_dir), now=now)
    return collected


def test_allocation_probe_is_silent_when_the_feature_is_disabled(tmp_path: Path) -> None:
    checks = _collect_allocation_checks(_doctor_allocation_config(enabled=False), tmp_path)
    assert checks == []


def test_allocation_probe_reports_enabled_but_never_run(tmp_path: Path) -> None:
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    assert len(checks) == 1
    name, ok, detail = checks[0]
    assert name == "runner allocation"
    assert ok is False
    assert "never run" in detail


def test_allocation_probe_passes_on_a_fresh_pass(tmp_path: Path) -> None:
    import datetime

    from ci_fleet.charlie_work_adapter import ALLOCATION_STATE_FILENAME

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (tmp_path / ALLOCATION_STATE_FILENAME).write_text(
        json.dumps({"version": 1, "updated_at": now, "source": "prologue", "repos": {}}),
        encoding="utf-8",
    )
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    assert len(checks) == 1
    _, ok, detail = checks[0]
    assert ok is True
    assert "budget 8" in detail
    # Only an unattended write is evidence, so the ok line has to say which it saw.
    assert "unattended" in detail


def test_allocation_probe_flags_a_stale_pass(tmp_path: Path) -> None:
    """A configured-but-inert allocator is the exact shape of issue #590."""
    import datetime

    from ci_fleet.charlie_work_adapter import ALLOCATION_STATE_FILENAME

    config = _doctor_allocation_config()
    stale_by = config.supervisor.full_pass_interval_seconds * 3 + 60
    old = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=stale_by)
    (tmp_path / ALLOCATION_STATE_FILENAME).write_text(
        json.dumps(
            {"version": 1, "updated_at": old.isoformat(), "source": "prologue", "repos": {}}
        ),
        encoding="utf-8",
    )
    checks = _collect_allocation_checks(config, tmp_path)
    assert len(checks) == 1
    _, ok, detail = checks[0]
    assert ok is False
    assert "not running unattended" in detail


def test_allocation_probe_survives_a_corrupt_state_file(tmp_path: Path) -> None:
    from ci_fleet.charlie_work_adapter import ALLOCATION_STATE_FILENAME

    (tmp_path / ALLOCATION_STATE_FILENAME).write_text("{not json", encoding="utf-8")
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    assert len(checks) == 1
    _, ok, detail = checks[0]
    assert ok is False
    assert "updated_at" in detail


def _write_allocation_stamp(
    fleet_dir: Path,
    *,
    age_seconds: float,
    source: Any,
    full_pass_interval_seconds: int | None = None,
    skip_reason: str | None = None,
    now: Any = None,
) -> None:
    """Write a state file aged ``age_seconds`` (negative = future-dated).

    ``now`` is the reference instant the age is computed against; defaults to
    the real wall clock. Pass a frozen value (issue #828) when the caller
    also passes the same value to ``_collect_allocation_checks`` so a tight
    downstream assertion cannot race an unbounded CI stall between the write
    here and the probe's own clock read.
    """
    import datetime

    from ci_fleet.charlie_work_adapter import ALLOCATION_STATE_FILENAME

    reference = now if now is not None else datetime.datetime.now(datetime.timezone.utc)
    when = reference - datetime.timedelta(seconds=age_seconds)
    payload: dict[str, Any] = {"version": 1, "updated_at": when.isoformat(), "repos": {}}
    if source is not None:
        payload["source"] = source
    if full_pass_interval_seconds is not None:
        payload["full_pass_interval_seconds"] = full_pass_interval_seconds
    if skip_reason is not None:
        payload["skip_reason"] = skip_reason
    (fleet_dir / ALLOCATION_STATE_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


def test_allocation_probe_does_not_accept_a_manual_pass_as_evidence(tmp_path: Path) -> None:
    """A fresh manual allocate must not make the probe read healthy.

    CLAUDE.md requires post-reboot procedures to delegate to `charlie runners
    allocate`, so this is the routine case -- and it writes the same host-wide file
    the unattended pass does. Treating its timestamp as proof would blind the probe
    for three intervals during exactly the window an operator is diagnosing #590.
    """
    _write_allocation_stamp(tmp_path, age_seconds=5, source="cli")
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    assert len(checks) == 1
    _, ok, detail = checks[0]
    assert ok is False
    assert "manual" in detail
    assert "cannot confirm" in detail


def test_allocation_probe_reports_unrecorded_provenance_rather_than_assuming(
    tmp_path: Path,
) -> None:
    """A file written before provenance tracking is unknown, not unattended."""
    _write_allocation_stamp(tmp_path, age_seconds=5, source=None)
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    assert len(checks) == 1
    _, ok, detail = checks[0]
    assert ok is False
    assert "unrecorded" in detail


def test_allocation_probe_names_an_unrecognised_writer(tmp_path: Path) -> None:
    """A future writer that forgets to extend AllocationSource is named, not hidden."""
    _write_allocation_stamp(tmp_path, age_seconds=5, source="some-new-path")
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    _, ok, detail = checks[0]
    assert ok is False
    assert "some-new-path" in detail


def test_allocation_probe_clamps_a_future_dated_stamp(tmp_path: Path) -> None:
    """Clock skew must not print a negative age, and must not read as stale.

    Regression for issue #828 (originally #822's class): the stamp is
    future-dated by 600s, and the clamp (doctor.py's non-negative `max(0, ...)`
    -- preserved, not removed) only reads as exactly "0s ago" while the
    probe's own `now` sample has not yet caught up to the future-dated
    `updated_at`. Two independently-sampled `now()`s would flip this the
    moment a stall pushes the probe's read past 600s after the fixture write
    -- the same order of magnitude as an observed CI stall. `now` is frozen
    and passed to both the fixture write and the probe so the comparison is
    exact regardless of any stall in between.
    """
    import datetime

    frozen_now = datetime.datetime(2026, 7, 29, 12, 0, 0, tzinfo=datetime.timezone.utc)
    _write_allocation_stamp(tmp_path, age_seconds=-600, source="prologue", now=frozen_now)
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path, now=frozen_now)
    _, ok, detail = checks[0]
    assert ok is True
    assert "-" not in detail
    assert "0s ago" in detail


# ---------------------------------------------------------------------------
# Recorded driving interval + skip reason (issue #606)
# ---------------------------------------------------------------------------


def test_allocation_probe_measures_staleness_against_the_recorded_interval(
    tmp_path: Path,
) -> None:
    """The bound comes from the interval the pass was driven at, not re-resolved.

    A per-repo layer setting a different interval would otherwise make the probe
    measure against a cadence the daemon is not running at. Here the recorded
    interval is far shorter than the config default, so a stamp that is fresh
    under the config bound is stale under the recorded one.
    """
    # Recorded interval 10s -> stale_after 30s. Config default is 300s -> 900s.
    # Age 60s is stale under the recorded bound but fresh under config.
    _write_allocation_stamp(
        tmp_path,
        age_seconds=60,
        source="prologue",
        full_pass_interval_seconds=10,
    )
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    _, ok, detail = checks[0]
    assert ok is False
    assert "30s staleness bound" in detail
    assert "not running unattended" in detail


def test_allocation_probe_falls_back_to_config_interval_when_none_recorded(
    tmp_path: Path,
) -> None:
    """A file written before interval recording uses the config bound."""
    config = _doctor_allocation_config()
    # Age just past the config bound (300*3=900), no recorded interval.
    _write_allocation_stamp(
        tmp_path,
        age_seconds=config.supervisor.full_pass_interval_seconds * 3 + 60,
        source="prologue",
    )
    checks = _collect_allocation_checks(config, tmp_path)
    _, ok, detail = checks[0]
    assert ok is False
    assert "not running unattended" in detail


def test_allocation_probe_reports_a_recorded_skip_reason(tmp_path: Path) -> None:
    """A fresh unattended skip names the cause instead of asserting #590."""
    _write_allocation_stamp(
        tmp_path,
        age_seconds=5,
        source="prologue",
        full_pass_interval_seconds=300,
        skip_reason="no configured runners found under /actions-runners",
    )
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    _, ok, detail = checks[0]
    assert ok is False
    assert "declined to act" in detail
    assert "no configured runners found under /actions-runners" in detail
    # A fresh unattended skip is the daemon *reaching* allocation and declining —
    # not the "never reached it" shape #590 describes — so #590 must not appear.
    assert "#590" not in detail


def test_allocation_probe_joins_skip_reason_and_staleness_when_stale(
    tmp_path: Path,
) -> None:
    """A stale skip reports both the recorded reason and the #590 reading."""
    _write_allocation_stamp(
        tmp_path,
        age_seconds=2000,
        source="prologue",
        full_pass_interval_seconds=300,
        skip_reason="no configured runners found under /actions-runners",
    )
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    _, ok, detail = checks[0]
    assert ok is False
    assert "declined to act" in detail
    assert "no configured runners found under /actions-runners" in detail
    # Stale: the daemon last found this and has not been back, so #590 joins.
    assert "not running unattended" in detail
    assert "#590" in detail


def test_allocation_probe_reports_a_manual_skip_with_the_recorded_reason(
    tmp_path: Path,
) -> None:
    """A CLI skip names the reason; the writer is still flagged as non-daemon.

    A fresh manual skip records *why* it declined (issue #606) but, like a
    fresh manual non-skip, cannot confirm the daemon is rebalancing — the
    writer overwrites the same host-wide file, so its skip reason is not
    evidence the unattended pass reached allocation (issue #590). Both
    signals must appear together.
    """
    _write_allocation_stamp(
        tmp_path,
        age_seconds=5,
        source="cli",
        full_pass_interval_seconds=300,
        skip_reason="no configured runners found under /actions-runners",
    )
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    _, ok, detail = checks[0]
    assert ok is False
    assert "declined to act" in detail
    assert "no configured runners found under /actions-runners" in detail
    assert "manual" in detail
    # The writer is non-unattended, so the #590 "cannot confirm" framing must
    # join the recorded reason — exactly the distinction this probe preserves.
    assert "cannot confirm" in detail
    assert "#590" in detail


def test_allocation_probe_cannot_confirm_clause_for_a_fresh_manual_skip(
    tmp_path: Path,
) -> None:
    """Regression: a fresh manual (CLI) skip surfaces the #590 clause.

    The skip_reason branch returns before the source-mismatch check, so the
    'cannot confirm the daemon is rebalancing (issue #590)' framing must be
    appended within that branch when the writer is not unattended — otherwise
    an operator's manual run reads as a named daemon skip and blinds the probe
    for three intervals during the very window #590 is being diagnosed.
    """
    _write_allocation_stamp(
        tmp_path,
        age_seconds=5,
        source="cli",
        full_pass_interval_seconds=300,
        skip_reason="no configured runners found under /actions-runners",
    )
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    _, ok, detail = checks[0]
    assert ok is False
    # The #590 clause is the authoritative signal that a manual write cannot
    # stand in for daemon health.
    assert "cannot confirm the daemon is rebalancing" in detail
    assert "issue #590" in detail
    # Fresh (5s < 900s bound): staleness must NOT also be cited — only the
    # source-mismatch clause applies, so "not running unattended" is absent.
    assert "not running unattended" not in detail


def test_allocation_probe_joins_source_and_staleness_for_a_stale_manual_skip(
    tmp_path: Path,
) -> None:
    """A stale manual skip cites both clauses, with #590 named once.

    Both the source mismatch (manual write cannot confirm the daemon) and
    staleness (daemon has not been back) apply; the clauses join so #590 is
    cited once rather than duplicated.
    """
    _write_allocation_stamp(
        tmp_path,
        age_seconds=2000,
        source="cli",
        full_pass_interval_seconds=300,
        skip_reason="no configured runners found under /actions-runners",
    )
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    _, ok, detail = checks[0]
    assert ok is False
    assert "declined to act" in detail
    assert "cannot confirm the daemon is rebalancing" in detail
    assert "not running unattended" in detail
    # #590 cited exactly once, not duplicated.
    assert detail.count("#590") == 1


def test_run_doctor_wires_the_allocation_probe(tmp_path: Path) -> None:
    """Pin the wiring, not just the probe body.

    Every other allocation test calls ``_check_runner_allocation`` directly, so
    deleting its call in ``run_doctor`` would leave them all green while the probe
    silently stopped running for operators.
    """
    config = _doctor_allocation_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    _, checks = run_doctor(
        tmp_path, paths, config, tmp_path / "c.yaml", gh, fleet_dir_override=str(tmp_path)
    )

    allocation_checks = [c for c in checks if c.name == "runner allocation"]
    assert len(allocation_checks) == 1
    # No state file was written, so the probe should report the never-run case.
    assert allocation_checks[0].ok is False
    assert "never run" in allocation_checks[0].detail
    # Warning-only: the probe must never change doctor's exit code.
    assert allocation_checks[0].severity == "warning"


# ---------------------------------------------------------------------------
# fleet dir virtualization probe (issue #624)
#
# MSIX/container redirection makes the literal fleet-dir path string identical
# in both the container and the host while naming different files. The probe
# keys on the literal path disagreeing with its resolved form -- never on a
# hardcoded package moniker. The tests inject the divergence by patching the
# resolution step (per the issue's test guidance), not by building a real
# MSIX redirect.


def _patch_resolve_to_diverge(monkeypatch: Any, literal: Path, redirected: Path) -> None:
    """Make ``Path.resolve()`` return ``redirected`` for ``literal`` only.

    Every other path resolves normally, so the rest of ``run_doctor`` is
    unaffected. This is the "patch the resolution step" injection the issue
    prescribes: the real ``fleet_dir_virtualization`` logic runs end-to-end,
    only the filesystem's answer is forged.
    """
    import os as _os
    import pathlib

    real_resolve = pathlib.Path.resolve

    def fake_resolve(self, *args, **kwargs):
        result = real_resolve(self, *args, **kwargs)
        if _os.path.normcase(_os.fspath(result)) == _os.path.normcase(_os.fspath(literal)):
            return redirected
        return result

    monkeypatch.setattr(pathlib.Path, "resolve", fake_resolve)


def test_doctor_warns_when_fleet_dir_is_virtualized(tmp_path: Path, monkeypatch: Any) -> None:
    """A literal/resolved divergence fires a warning naming both paths."""
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)
    redirected = tmp_path / "Packages" / "app" / "LocalCache" / "Local" / "charlie-work"
    _patch_resolve_to_diverge(monkeypatch, fleet_dir_path, redirected)

    ok, checks = run_doctor(
        tmp_path, paths, config, tmp_path / "c.yaml", gh, fleet_dir_override=str(fleet_dir_path)
    )

    by_name = {check.name: check for check in checks}
    virt = by_name["fleet dir virtualization"]
    assert virt.ok is False
    assert virt.severity == "warning"
    # Both paths must be named so the operator can see where it landed.
    assert str(fleet_dir_path) in virt.detail
    assert str(redirected) in virt.detail
    # Reference the #590 failure and state the write-forks-a-copy consequence.
    assert "#590" in virt.detail
    assert "private copy" in virt.detail
    # Warning-only: a virtualized fleet dir is not fatal for an interactive human.
    assert ok is True


def test_doctor_is_silent_when_fleet_dir_is_not_virtualized(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Equal literal and resolved paths produce no virtualization check."""
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)
    # No divergence injection: resolve() returns the literal path unchanged.
    ok, checks = run_doctor(
        tmp_path, paths, config, tmp_path / "c.yaml", gh, fleet_dir_override=str(fleet_dir_path)
    )

    names = {check.name for check in checks}
    assert "fleet dir virtualization" not in names
    assert ok is True


def test_fleet_dir_virtualization_probe_is_repo_agnostic(tmp_path: Path, monkeypatch: Any) -> None:
    """The probe fires for any fleet_dir override, not just charlie-work's layout.

    The redirection is a per-process property of the host, so a repo whose
    fleet dir is unrelated to charlie-work's own layout must still be flagged.
    """
    from charlie_work.fleet_paths import fleet_dir_virtualization

    # An arbitrary override path with no charlie-work-specific component.
    literal = tmp_path / "some-other-repo" / "fleet-state"
    redirected = tmp_path / "Packages" / "other-app" / "LocalCache" / "some-other-repo"
    _patch_resolve_to_diverge(monkeypatch, literal, redirected)

    diverged = fleet_dir_virtualization(override=str(literal))
    assert diverged is not None
    assert diverged[0] == literal
    assert diverged[1] == redirected


def test_fleet_dir_virtualization_returns_none_when_equal(tmp_path: Path) -> None:
    """No divergence -> None (the probe must stay silent)."""
    from charlie_work.fleet_paths import fleet_dir_virtualization

    literal = tmp_path / "fleet"
    literal.mkdir(parents=True, exist_ok=True)
    assert fleet_dir_virtualization(override=str(literal)) is None


def test_run_doctor_wires_the_virtualization_probe(tmp_path: Path, monkeypatch: Any) -> None:
    """Pin the wiring, not just the probe body.

    Deleting the ``_check_fleet_dir_virtualization`` call in ``run_doctor``
    would leave the probe-body tests green while the probe silently stopped
    running for operators.
    """
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)
    redirected = tmp_path / "Packages" / "app" / "LocalCache" / "Local" / "charlie-work"
    _patch_resolve_to_diverge(monkeypatch, fleet_dir_path, redirected)

    _, checks = run_doctor(
        tmp_path, paths, config, tmp_path / "c.yaml", gh, fleet_dir_override=str(fleet_dir_path)
    )

    virt_checks = [c for c in checks if c.name == "fleet dir virtualization"]
    assert len(virt_checks) == 1
    assert virt_checks[0].severity == "warning"


# ---------------------------------------------------------------------------
# A5: state-dir split-brain tripwires (issue #712)
# ---------------------------------------------------------------------------


def test_doctor_state_dir_split_brain_silent_without_override(tmp_path: Path) -> None:
    """No ``state_dir`` override -> the default tree *is* the configured tree,
    so there is nothing to diagnose and the check must not appear at all."""
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    _, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    assert [c for c in checks if c.name == "state dir split-brain"] == []


def test_doctor_state_dir_split_brain_silent_when_explicit_state_dir_matches_default(
    tmp_path: Path,
) -> None:
    """``state_dir: .var/charlie-work`` spelled explicitly (charlie-work's own
    ``orchestrator.config.yaml`` does exactly this) must compare equal to
    ``layout.default_state_root`` and stay silent — a spelling-only match, not
    a real override. Guards a ``.resolve()`` asymmetry: ``runtime_paths``
    resolves ``paths.root`` but ``default_state_root`` does not, so a config
    that merely repeats the default value must not misread as a divergence.
    """
    config = _config(
        runtime=RuntimeConfig(state_dir=".var/charlie-work"),
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    _, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    assert [c for c in checks if c.name == "state dir split-brain"] == []
    by_name = {c.name: c for c in checks}
    assert by_name["worktrees root"].ok is True


def test_doctor_state_dir_split_brain_silent_when_overridden_with_no_residue(
    tmp_path: Path,
) -> None:
    """``state_dir`` overridden but the default tree is empty/absent -> silent.

    Guards against a false positive firing on every overridden repo regardless
    of whether the default tree actually holds anything.
    """
    config = _config(
        runtime=RuntimeConfig(state_dir="custom-state"),
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    _, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    assert [c for c in checks if c.name == "state dir split-brain"] == []


def test_doctor_state_dir_split_brain_fires_on_residue(tmp_path: Path) -> None:
    """Empirical proof this detector fires on the live #712 shape: an
    overridden ``state_dir`` plus a default tree still holding ``state.json``,
    a non-empty ``worktrees/``, and a non-empty ``events.db`` — exactly what
    job-cannon's ``.var/charlie-work`` looks like today (74 uncollected
    worktrees; 0-byte ``events.db`` was the pre-#718 shape, so this test uses a
    non-empty one to prove the size>0 branch too).
    """
    config = _config(
        runtime=RuntimeConfig(state_dir="custom-state"),
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    default_root = tmp_path / ".var" / "charlie-work"
    (default_root / "worktrees" / "some-branch").mkdir(parents=True)
    (default_root / "state.json").write_text("{}", encoding="utf-8")
    (default_root / "events.db").write_bytes(b"\x00" * 32)

    _, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {c.name: c for c in checks}
    check = by_name["state dir split-brain"]
    assert check.ok is False
    assert check.severity == "warning"
    assert str(default_root) in check.detail
    assert str(paths.root) in check.detail
    assert "state.json" in check.detail
    assert "worktrees" in check.detail
    assert "events.db" in check.detail


def test_doctor_state_dir_split_brain_fires_on_worktrees_only_production_shape(
    tmp_path: Path,
) -> None:
    """The *actual* job-cannon shape, not just the easier all-three-signals
    shape above: no ``state.json`` (never written there), a 0-byte
    ``events.db`` (schema never applied — the #718 shape), and only a
    non-empty ``worktrees/`` holding the 74 uncollected worktrees. The
    ``events.db`` size>0 branch must NOT be required for this to fire — the
    ``worktrees/`` clause alone has to be sufficient, since that is the only
    signal present in production today.
    """
    config = _config(
        runtime=RuntimeConfig(state_dir="custom-state"),
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    default_root = tmp_path / ".var" / "charlie-work"
    (default_root / "worktrees" / "some-branch").mkdir(parents=True)
    (default_root / "worktrees" / "other-branch").mkdir(parents=True)
    (default_root / "events.db").write_bytes(b"")

    _, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {c.name: c for c in checks}
    check = by_name["state dir split-brain"]
    assert check.ok is False
    assert check.severity == "warning"
    assert "worktrees" in check.detail
    assert "2 entries" in check.detail
    assert "state.json" not in check.detail
    assert "events.db" not in check.detail


def test_doctor_state_dir_split_brain_ignores_zero_byte_events_db(tmp_path: Path) -> None:
    """A freshly-``sqlite3.connect``-ed, schema-never-applied ``events.db`` is
    0 bytes (the #718 shape) — it must not itself count as residue, since a
    0-byte file proves nothing was ever written there."""
    config = _config(
        runtime=RuntimeConfig(state_dir="custom-state"),
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    default_root = tmp_path / ".var" / "charlie-work"
    default_root.mkdir(parents=True)
    (default_root / "events.db").write_bytes(b"")

    _, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    assert [c for c in checks if c.name == "state dir split-brain"] == []


def test_doctor_worktrees_root_agrees_by_default(tmp_path: Path) -> None:
    """No override -> the reported root is the plain default; ``ok`` is
    unconditionally True (see the ``_check_worktrees_root_agreement``
    docstring for why a pass/fail comparison is no longer meaningful post-A2)."""
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    _, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {c.name: c for c in checks}
    check = by_name["worktrees root"]
    assert check.ok is True
    assert check.severity == "error"  # default severity; ok=True never blocks
    assert str(paths.root / "worktrees") in check.detail


def test_doctor_worktrees_root_no_longer_diverges_when_state_dir_overridden(
    tmp_path: Path,
) -> None:
    """Regression proof for #712: before A2, an overridden ``runtime.state_dir``
    with no explicit ``claude_code.worktrees_dir`` made dispatch's fallback
    (the unconditional default tree) and ``worktree-clean``'s sweep (the
    configured tree) resolve to two different directories, and this check
    reported ``ok=False``. A2 unified both call sites behind
    ``resolved_layout(config, repo_root).worktrees``, so the same config now
    reports a single root (the configured tree) and ``ok=True`` -- the stale
    default-tree path must no longer appear anywhere in the detail.
    """
    config = _config(
        runtime=RuntimeConfig(state_dir="custom-state"),
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    _, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {c.name: c for c in checks}
    check = by_name["worktrees root"]
    assert check.ok is True
    assert check.severity == "error"  # default severity; ok=True never blocks
    assert str(paths.root / "worktrees") in check.detail
    stale_default_worktrees = tmp_path / ".var" / "charlie-work" / "worktrees"
    assert str(stale_default_worktrees) not in check.detail


def test_doctor_worktrees_root_agrees_when_explicit_worktrees_dir_matches_state_dir(
    tmp_path: Path,
) -> None:
    """An explicit ``claude_code.worktrees_dir`` (axis 2) still reports a
    single agreeing root -- unaffected by which of the two override axes is
    in play, since both resolve through the same ``resolved_layout`` call."""
    config = _config(
        runtime=RuntimeConfig(state_dir="custom-state"),
        claude_code=ClaudeCodeConfig(worktrees_dir="custom-state/worktrees"),
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    _, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {c.name: c for c in checks}
    assert by_name["worktrees root"].ok is True


def test_check_recent_lane_failures_surfaces_past_event(tmp_path: Path) -> None:
    """#6-G / G-AC3 + G-AC5: a past fleet_pass_config_error event recorded to
    this repo's events.db (by fleet_dispatch._record_lane_failure_event, when
    a lane failed to start on a prior pass) surfaces as a doctor finding.

    Severity is a warning, not a hard error, per doctor.py's own convention
    for "this recently happened, may already be fixed" reports (e.g.
    _check_runner_allocation's staleness checks) — it must not, by itself,
    flip run_doctor()'s overall ok to False the way a currently-broken
    required check does.
    """
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    log_event(
        paths.state_file,
        "fleet_pass_config_error",
        {
            "repo_key": "owner/repo",
            "error": "ConfigError: unknown key(s) in config section 'cross_family': auto_verdict",
        },
        repo="owner/repo",
    )

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {c.name: c for c in checks}
    check = by_name["recent lane failures"]
    assert check.ok is False
    assert check.severity == "warning"
    assert "cross_family" in check.detail
    # A warning-severity finding must not by itself block the overall result.
    assert ok is True


def test_check_recent_lane_failures_silent_when_no_events(tmp_path: Path) -> None:
    """No fleet_pass_config_error events -> no "recent lane failures" finding
    at all, so a healthy repo's doctor output stays unchanged (no new noise)."""
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    _, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {c.name: c for c in checks}
    assert "recent lane failures" not in by_name


def test_doctor_surfaces_in_progress_corroboration_alive_but_polling(
    tmp_path: Path, monkeypatch
) -> None:
    """Issue #1346: doctor's "in-progress worker corroboration" check must
    visibly distinguish an alive-but-polling worker (stale sidecar log mtime,
    fresh events.jsonl corroboration) from a genuinely stalled one (stale log
    AND stale corroboration).

    Both workers share the same stale sidecar log mtime -- the only signal a
    log-mtime monitor sees -- so pre-#1346 they were indistinguishable to an
    operator running `charlie doctor`. After #1346 the check reports the
    watchdog's corroboration verdict (same ``real_activity_probe_for`` +
    ``classify_worker_health`` code path) and buckets the alive-but-polling
    worker under "alive-but-polling" (ok=True, informational) while the
    stalled worker lands under "stalled/dead" (ok=False).
    """
    import os
    import time
    from datetime import UTC, datetime, timedelta
    from unittest.mock import patch

    from charlie_work.config import PostMortemConfig, WatchdogConfig

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    old_time = datetime.now(UTC) - timedelta(minutes=30)
    fresh_time = datetime.now(UTC) - timedelta(minutes=1)

    def _plant_claude_worker(issue_number: int, *, fresh_corroboration: bool) -> None:
        log_path = sessions_dir / f"issue-{issue_number}.claude.log"
        log_path.write_text("working\n", encoding="utf-8")
        # Stale sidecar log mtime for BOTH workers -- the signal log-mtime
        # monitors cannot disambiguate.
        os.utime(log_path, (time.time(), old_time.timestamp()))

        events_path = sessions_dir / f"issue-{issue_number}.events.jsonl"
        ts = fresh_time if fresh_corroboration else old_time
        events_path.write_text(
            f'{{"type": "tool_call", "timestamp": "{ts.isoformat()}"}}\n',
            encoding="utf-8",
        )
        os.utime(events_path, (time.time(), ts.timestamp()))

        _write_sidecar(
            sessions_dir,
            f"issue-{issue_number}.claude.json",
            {
                "issue_number": issue_number,
                "branch": f"agent/issue-{issue_number}",
                "worktree_path": str(tmp_path / f"wt-{issue_number}"),
                "prompt_path": "p.md",
                "command": ["claude", "p.md"],
                "pid": 80000 + issue_number,
                "started_at": (datetime.now(UTC) - timedelta(minutes=10)).isoformat(),
                "log_path": str(log_path),
                "error": None,
                "failure_kind": None,
                "process_start_time": 1710000000.0,
                "reclaimed": None,
            },
        )

    # Worker 1346: alive-but-polling (stale log, fresh corroboration).
    _plant_claude_worker(1346, fresh_corroboration=True)
    # Worker 1347: genuinely stalled (stale log, stale corroboration).
    _plant_claude_worker(1347, fresh_corroboration=False)

    monkeypatch.setattr(
        "charlie_work.devin_shell.probe_devin",
        lambda repo_root, **kwargs: RunResult(returncode=0, stdout="devin 1.2.3", stderr=""),
    )
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(adapter="devin-shell", sessions_dir="sessions"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db")),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        ok, checks = run_doctor(
            tmp_path, paths, config, tmp_path / "c.yaml", gh, adapter_probe=True
        )

    by_name = {c.name: c for c in checks}
    assert "in-progress worker corroboration" in by_name
    check = by_name["in-progress worker corroboration"]

    # The alive-but-polling worker is healthy per the watchdog; the stalled
    # worker is not. The check fails only on the genuinely stalled one.
    assert check.ok is False
    assert check.severity == "warning"
    assert "alive-but-polling (1)" in check.detail
    assert "issue #1346" in check.detail
    assert "fresh=True" in check.detail
    assert "stalled/dead (1)" in check.detail
    assert "issue #1347" in check.detail
    assert "fresh=False" in check.detail
    # A warning-severity finding must not by itself block the overall result.
    assert ok is True


def test_doctor_in_progress_corroboration_silent_when_no_workers(
    tmp_path: Path, monkeypatch
) -> None:
    """Issue #1346: with no in-progress workers, the corroboration check
    reports an empty detail and ok=True (no noise for a healthy repo)."""
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

    # sessions dir exists but is empty.
    (tmp_path / "sessions").mkdir()

    _, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh, adapter_probe=True)

    by_name = {c.name: c for c in checks}
    check = by_name["in-progress worker corroboration"]
    assert check.ok is True
    assert "no in-progress workers" in check.detail
