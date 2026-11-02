"""Covered-push classifier in ci.yml's ``coverage`` job (DD-5).

A push to main whose merge commit was made by the merge-queue bot already ran
the full suite on the queue's draft PR, so the shards are skipped. The step
body is executed with bash against a fake ``gh`` (``_ci_step_runner``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _ci_step_runner import find_step, load_workflow, run_step

WF = load_workflow()
JOBS = WF["jobs"]
STEP = find_step(WF, "coverage", "Classify covered push")

PUSH = {"EVENT_NAME": "push", "REPO": "Senkichi/charlie-work", "SHA": "abc123"}


def _run(tmp_path: Path, **env: str):
    return run_step(STEP, tmp_path, {**PUSH, **env})


def test_body_has_no_expressions_and_uses_two_call_lookup() -> None:
    body = STEP["run"]
    assert "${{" not in body
    assert "merge_commit_sha" in body
    assert ".merged_by.login" in body


def test_job_wiring() -> None:
    assert JOBS["coverage"]["outputs"]["covered"] == "${{ steps.covered.outputs.covered }}"
    assert JOBS["tests-shard"]["if"] == "needs.coverage.outputs.covered != 'true'"
    gate = JOBS["collect-only-gate"]
    assert gate["needs"] == "coverage"
    assert gate["if"] == "needs.coverage.outputs.covered != 'true'"


@pytest.mark.parametrize("login", ["aviator-app[bot]", "aviator-app"])
def test_queue_bot_merge_is_covered(tmp_path: Path, login: str) -> None:
    result = _run(tmp_path, FAKE_PULLS_OUT="42", FAKE_PR_OUT=login)
    assert result.returncode == 0, result.stderr
    assert result.outputs["covered"] == "true"
    assert any("repos/Senkichi/charlie-work/pulls/42" in call for call in result.gh_calls)


def test_human_merge_is_not_covered(tmp_path: Path) -> None:
    assert (
        _run(tmp_path, FAKE_PULLS_OUT="42", FAKE_PR_OUT="Senkichi").outputs["covered"] == "false"
    )


def test_direct_push_without_pr_is_not_covered(tmp_path: Path) -> None:
    result = _run(tmp_path, FAKE_PULLS_OUT="")
    assert result.outputs["covered"] == "false"
    assert len(result.gh_calls) == 1


@pytest.mark.parametrize(
    "env", [{"FAKE_PULLS_RC": "1"}, {"FAKE_PULLS_OUT": "42", "FAKE_PR_RC": "1"}]
)
def test_api_errors_fail_open_to_running(tmp_path: Path, env: dict[str, str]) -> None:
    result = _run(tmp_path, **env)
    assert result.returncode == 0
    assert result.outputs["covered"] == "false"


def test_non_numeric_pr_is_not_covered(tmp_path: Path) -> None:
    assert (
        _run(tmp_path, FAKE_PULLS_OUT="4;2", FAKE_PR_OUT="aviator-app[bot]").outputs["covered"]
        == "false"
    )


def test_pull_request_event_makes_no_api_call(tmp_path: Path) -> None:
    result = _run(tmp_path, EVENT_NAME="pull_request")
    assert result.outputs["covered"] == "false"
    assert result.gh_calls == []


def test_kill_switch(tmp_path: Path) -> None:
    result = _run(
        tmp_path, SKIP_COVERED="false", FAKE_PULLS_OUT="42", FAKE_PR_OUT="aviator-app[bot]"
    )
    assert result.outputs["covered"] == "false"
    assert result.gh_calls == []


def test_custom_bot_login(tmp_path: Path) -> None:
    result = _run(
        tmp_path, BOT_LOGIN="other-queue[bot]", FAKE_PULLS_OUT="7", FAKE_PR_OUT="other-queue"
    )
    assert result.outputs["covered"] == "true"
