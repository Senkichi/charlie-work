"""Required-check / workflow-matrix flagging checks for ``run_doctor``.

Split out of ``tests/test_doctor.py`` (issue #1563, Track 1 shoulder) --
bodies are verbatim relocations; shared helpers live in
``tests/_doctor_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path
from charlie_work.config import (
    AutoMergeConfig,
    RescueConfig,
    ReviewDispatchConfig,
    RuntimeConfig,
)
from charlie_work.doctor import (
    _check_name_match_kind,
    _tolerance_match_base_names,
    run_doctor,
    workflow_job_matrix_flags,
    workflow_job_names,
)
from charlie_work.paths import runtime_paths
from _doctor_fixtures import (
    FakeDoctorGitHub,
    _config,
    _write_workflow,
    _write_workflow_named,
)


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


def test_workflow_job_matrix_flags_scoped_per_job(tmp_path: Path) -> None:
    # Issue #1508: matrix-ness is a PER-JOB property, not repo-wide. Two
    # workflows in one repo -- only the second job has strategy.matrix -- so
    # the flags dict must mark "Matrix job" True and "Plain job" False
    # independently. The repo-wide predicate (any flag True) stays True (some
    # job has a matrix), which is exactly the combination the old
    # global-boolean verifier misused to justify a tolerance match against the
    # plain job.
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
    assert any(flags.values()) is True


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


def test_doctor_flags_no_automated_review_to_verdict_path(tmp_path: Path) -> None:
    # review_dispatch.enabled=False + rescue.enabled=False (both real
    # defaults on OrchestratorConfig) is exactly the 7-day-outage config
    # shape -- no automated path ever calls record_review(). _config()
    # defaults review_dispatch on for the rest of this module, so this test
    # overrides it back off explicitly.
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        review_dispatch=ReviewDispatchConfig(enabled=False),
    )
    assert config.review_dispatch.enabled is False
    assert config.rescue.enabled is False
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    check = by_name["review-to-verdict path"]
    assert check.ok is False
    assert check.severity == "error"
    assert "review_dispatch.enabled" in check.detail
    assert "rescue.enabled" in check.detail
    assert ok is False


def test_doctor_review_to_verdict_path_ok_when_rescue_enabled(tmp_path: Path) -> None:
    # review_dispatch off, rescue.enabled on: the check must pass on
    # rescue.enabled alone, independent of review_dispatch.
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        review_dispatch=ReviewDispatchConfig(enabled=False),
        rescue=RescueConfig(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    assert by_name["review-to-verdict path"].ok is True
    assert ok is True
