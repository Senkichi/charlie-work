"""Issue #2316: a required check no static job name spells is a warning, not an error,
when a workflow posts the context itself or names a job by expression.

The merge gate (``checks.py``) matches the PR's *real* check runs exactly; the
doctor cross-check can only read static ``jobs.<id>.name`` values. A plain stale
name must still fail.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from charlie_work.config import AutoMergeConfig
from charlie_work.doctor import DoctorCheck, run_doctor
from charlie_work.doctor_external_checks import (
    DynamicCheckSources,
    is_check_run_post,
    is_expression_name,
    scan_dynamic_check_sources,
)
from charlie_work.paths import runtime_paths
from _doctor_fixtures import (
    FakeDoctorGitHub,
    _config,
    _write_workflow,
    _write_workflow_named,
)

# A job that folds lint into the test job and reports the old lint context
# itself, so branch protection and the queue config keep the "lint" name.
_POSTING_WORKFLOW = (
    "jobs:\n"
    "  test:\n"
    "    name: Tests passed\n"
    "    runs-on: ubuntu-latest\n"
    "    steps:\n"
    "      - run: pytest\n"
    "      - name: Report lint context\n"
    "        run: |\n"
    "          gh api repos/${{ github.repository }}/check-runs --method POST \\\n"
    "            -f name=lint -f head_sha=${{ github.sha }} -f conclusion=success\n"
)

# A kill-switch path renames a standalone job so its skipped run cannot shadow
# the required context: the real name only exists at run time.
_EXPRESSION_WORKFLOW = (
    "jobs:\n"
    "  test:\n"
    "    name: Tests passed\n"
    "    runs-on: ubuntu-latest\n"
    "  lint:\n"
    "    name: ${{ vars.LINT_KILL_SWITCH == 'on' && 'lint (skipped)' || 'lint' }}\n"
    "    runs-on: ubuntu-latest\n"
)


def _run(tmp_path: Path, required: tuple[str, ...], monkeypatch: pytest.MonkeyPatch):
    # Pin gh-on-PATH so overall ``ok`` depends only on the config under test.
    monkeypatch.setattr(shutil, "which", lambda name: "/fake/gh" if name == "gh" else None)
    config = _config(auto_merge=AutoMergeConfig(required_checks=required))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)
    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)
    return ok, {check.name: check for check in checks}, checks


def _hard_failures(checks: list[DoctorCheck]) -> list[str]:
    return [c.name for c in checks if not c.ok and c.severity == "error"]


# -- is_check_run_post ---------------------------------------------------------


@pytest.mark.parametrize(
    "run_text",
    [
        "gh api repos/o/r/check-runs --method POST -f name=lint",
        "gh api repos/o/r/check-runs --method=POST -f name=lint",
        "gh api repos/o/r/check-runs -X POST -f name=lint",
        "gh api repos/o/r/check-runs -XPOST -f name=lint",
        "gh api repos/o/r/check-runs --method post -f name=lint",
        # No explicit method: gh defaults to POST once any parameter is added.
        "gh api repos/o/r/check-runs -f name=lint -f head_sha=abc",
        "gh api repos/o/r/check-runs -F name=lint",
        "gh api repos/o/r/check-runs --raw-field name=lint",
        "gh api repos/o/r/check-runs --input body.json",
        # Backslash continuations join into one logical command.
        "gh api repos/o/r/check-runs \\\n  --method POST \\\n  -f name=lint",
        # The command is not the first line of the script.
        "set -euo pipefail\necho start\ngh api repos/o/r/check-runs --method POST -f name=x\n",
        # Expression-valued repo segment.
        "gh api repos/${{ github.repository }}/check-runs --method POST -f name=lint",
    ],
)
def test_is_check_run_post_detects_posting_shapes(run_text: str) -> None:
    assert is_check_run_post(run_text) is True


@pytest.mark.parametrize(
    "run_text",
    [
        "",
        "pytest -q",
        # A read: no parameters, so gh defaults to GET.
        "gh api repos/o/r/commits/abc/check-runs",
        "gh api repos/o/r/check-runs --jq '.check_runs[].name'",
        # Explicit non-POST method wins over parameters (parameters become the query string).
        "gh api repos/o/r/check-runs --method GET -f name=lint",
        "gh api repos/o/r/check-runs -X GET -f name=lint",
        # POSTs, but not to the check-runs API.
        "gh api repos/o/r/statuses/abc --method POST -f state=success",
        "gh api repos/o/r/issues/1/comments -f body=hi",
        # Mentions check-runs but is not a gh api call.
        "echo 'see check-runs docs' --method POST",
        # The method flag and endpoint are on different commands.
        "gh api repos/o/r/check-runs\ngh api repos/o/r/issues --method POST -f title=t",
    ],
)
def test_is_check_run_post_ignores_non_posting_shapes(run_text: str) -> None:
    assert is_check_run_post(run_text) is False


# -- is_expression_name --------------------------------------------------------


def test_is_expression_name_only_for_template_expressions() -> None:
    assert is_expression_name("${{ matrix.os }}") is True
    assert is_expression_name("Tests (${{ matrix.os }})") is True
    assert is_expression_name("Tests passed") is False
    assert is_expression_name("costs $5 {{not an expression}}") is False
    assert is_expression_name(None) is False
    assert is_expression_name(7) is False


# -- scan_dynamic_check_sources ------------------------------------------------


def test_scan_names_the_file_and_job_for_each_shape(tmp_path: Path) -> None:
    _write_workflow_named(tmp_path, "post.yml", _POSTING_WORKFLOW)
    _write_workflow_named(tmp_path, "expr.yml", _EXPRESSION_WORKFLOW)

    sources = scan_dynamic_check_sources(tmp_path)

    assert sources.posting_steps == ("post.yml: job `test`",)
    assert sources.expression_job_names == ("expr.yml: job `lint`",)
    assert bool(sources) is True
    described = sources.describe()
    assert "post.yml: job `test`" in described
    assert "expr.yml: job `lint`" in described


def test_scan_is_empty_for_static_workflows(tmp_path: Path) -> None:
    _write_workflow(
        tmp_path,
        "jobs:\n  test:\n    name: Tests passed\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - run: pytest\n      - run: gh api repos/o/r/commits/abc/check-runs\n",
    )

    sources = scan_dynamic_check_sources(tmp_path)

    assert sources == DynamicCheckSources()
    assert bool(sources) is False
    assert sources.describe() == ""


def test_scan_without_workflows_or_with_broken_yaml(tmp_path: Path) -> None:
    assert scan_dynamic_check_sources(tmp_path) == DynamicCheckSources()
    _write_workflow_named(tmp_path, "broken.yml", "jobs: [unclosed\n")
    _write_workflow_named(tmp_path, "scalar.yml", "just a string\n")
    _write_workflow_named(tmp_path, "nojobs.yml", "name: x\n")
    assert scan_dynamic_check_sources(tmp_path) == DynamicCheckSources()


def test_scan_tolerates_odd_job_and_step_shapes(tmp_path: Path) -> None:
    # jobs.<id> that is not a mapping, steps that are not a list, a step that is
    # not a mapping, and a non-string ``run`` must not raise.
    _write_workflow(
        tmp_path,
        "jobs:\n"
        "  a: just-a-string\n"
        "  b:\n    steps: not-a-list\n"
        "  c:\n    steps:\n      - plain-string-step\n      - run: 12\n",
    )

    assert scan_dynamic_check_sources(tmp_path) == DynamicCheckSources()


# -- run_doctor: the severity decision -----------------------------------------


def test_doctor_warns_when_a_step_posts_the_required_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_workflow(tmp_path, _POSTING_WORKFLOW)

    ok, by_name, checks = _run(tmp_path, ("Tests passed", "lint"), monkeypatch)

    assert by_name["check name: Tests passed"].ok is True
    posted = by_name["check name: lint"]
    assert posted.ok is False
    assert posted.severity == "warning"
    assert "ci.yml: job `test`" in posted.detail
    assert "check-runs" in posted.detail
    assert _hard_failures(checks) == []
    assert ok is True


def test_doctor_warns_when_a_job_name_is_an_expression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_workflow(tmp_path, _EXPRESSION_WORKFLOW)

    ok, by_name, checks = _run(tmp_path, ("Tests passed", "lint"), monkeypatch)

    expr = by_name["check name: lint"]
    assert expr.ok is False
    assert expr.severity == "warning"
    assert "ci.yml: job `lint`" in expr.detail
    assert "${{ ... }}" in expr.detail
    assert _hard_failures(checks) == []
    assert ok is True


def test_doctor_still_errors_on_a_plain_stale_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No check-runs POST and no expression name anywhere: the stale name is a
    # real mismatch the exact-match merge gate would report missing forever.
    _write_workflow(
        tmp_path,
        "jobs:\n  test:\n    name: Tests passed\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - run: gh api repos/o/r/commits/abc/check-runs\n",
    )

    ok, by_name, checks = _run(tmp_path, ("Tests passed", "lint"), monkeypatch)

    stale = by_name["check name: lint"]
    assert stale.ok is False
    assert stale.severity == "error"
    assert "no job in .github/workflows/ reports this name" in stale.detail
    assert "check name: lint" in _hard_failures(checks)
    assert ok is False


def test_doctor_downgrade_does_not_mask_a_static_name_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The downgrade qualifies only the no-match branch: a required name that
    # matches a static job stays a plain pass even in a repo with a posting step.
    _write_workflow(tmp_path, _POSTING_WORKFLOW)

    _, by_name, _ = _run(tmp_path, ("Tests passed",), monkeypatch)

    assert by_name["check name: Tests passed"].ok is True
    assert by_name["check name: Tests passed"].detail == "matches a workflow job"


def test_doctor_tolerance_only_match_stays_an_error_despite_dynamic_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # #1508's stale-suffix case is not "unverifiable": the matched job is static
    # and has no matrix, so the suffixed name can never be produced. A posting
    # step elsewhere in the repo must not soften it.
    _write_workflow(tmp_path, _POSTING_WORKFLOW)

    ok, by_name, checks = _run(tmp_path, ("Tests passed (windows-latest)",), monkeypatch)

    stale = by_name["check name: Tests passed (windows-latest)"]
    assert stale.ok is False
    assert stale.severity == "error"
    assert "matrix-suffix tolerance" in stale.detail
    assert "check name: Tests passed (windows-latest)" in _hard_failures(checks)
    assert ok is False
