from __future__ import annotations

from charlie_work.checks import (
    CheckDebounceResult,
    _run_id_from_link,
    classify_check_failures,
)


REQUIRED = ("Tests passed", "Lint & Format")


def _link(run_id: int, job_id: int) -> str:
    return f"https://github.com/owner/repo/actions/runs/{run_id}/job/{job_id}"


def test_run_id_from_link_extracts_run_id() -> None:
    link = "https://github.com/owner/repo/actions/runs/29525590823/job/87713099471?check_suite_focus=true"
    assert _run_id_from_link(link) == 29525590823


def test_run_id_from_link_returns_none_for_external_status() -> None:
    assert _run_id_from_link("https://external.ci/some/path") is None
    assert _run_id_from_link(None) is None
    assert _run_id_from_link("") is None


def test_classify_first_failure_requests_rerun_and_records_attempt() -> None:
    checks = [
        {"name": "Tests passed", "state": "FAILURE", "link": _link(100, 1)},
        {"name": "Lint & Format", "bucket": "pass"},
    ]
    result = classify_check_failures(
        checks,
        REQUIRED,
        pr_state=None,
        head_sha="sha-1",
    )
    assert result == CheckDebounceResult(
        rerun_run_ids=(100,),
        check_rerun_attempts={"sha-1": {"Tests passed": [100]}},
        definitive_failed=(),
    )


def test_classify_second_failure_is_definitive_and_does_not_rerun() -> None:
    checks = [
        {"name": "Tests passed", "state": "FAILURE", "link": _link(100, 1)},
        {"name": "Lint & Format", "bucket": "pass"},
    ]
    pr_state = {"check_rerun_attempts": {"sha-1": {"Tests passed": [100]}}}
    result = classify_check_failures(
        checks,
        REQUIRED,
        pr_state,
        head_sha="sha-1",
    )
    assert result.rerun_run_ids == ()
    assert result.definitive_failed == ("Tests passed",)
    # Attempts unchanged.
    assert result.check_rerun_attempts == {"sha-1": {"Tests passed": [100]}}


def test_classify_passing_check_clears_attempt_marker() -> None:
    checks = [
        {"name": "Tests passed", "state": "SUCCESS"},
        {"name": "Lint & Format", "bucket": "pass"},
    ]
    pr_state = {"check_rerun_attempts": {"sha-1": {"Tests passed": [100]}}}
    result = classify_check_failures(
        checks,
        REQUIRED,
        pr_state,
        head_sha="sha-1",
    )
    assert result.rerun_run_ids == ()
    assert result.definitive_failed == ()
    assert result.check_rerun_attempts == {"sha-1": {}}


def test_classify_new_head_resets_attempts() -> None:
    checks = [
        {"name": "Tests passed", "state": "FAILURE", "link": _link(100, 1)},
        {"name": "Lint & Format", "bucket": "pass"},
    ]
    pr_state = {"check_rerun_attempts": {"sha-old": {"Tests passed": [100]}}}
    result = classify_check_failures(
        checks,
        REQUIRED,
        pr_state,
        head_sha="sha-new",
    )
    assert result.rerun_run_ids == (100,)
    assert result.check_rerun_attempts == {"sha-new": {"Tests passed": [100]}}


def test_classify_external_status_failure_is_definitive_without_rerun() -> None:
    checks = [
        {"name": "Tests passed", "state": "FAILURE", "link": "https://external.ci/run"},
        {"name": "Lint & Format", "bucket": "pass"},
    ]
    result = classify_check_failures(
        checks,
        REQUIRED,
        pr_state=None,
        head_sha="sha-1",
    )
    assert result.rerun_run_ids == ()
    assert result.definitive_failed == ("Tests passed",)


def test_classify_record_attempts_false_does_not_consume_attempt() -> None:
    checks = [
        {"name": "Tests passed", "state": "FAILURE", "link": _link(100, 1)},
        {"name": "Lint & Format", "bucket": "pass"},
    ]
    result = classify_check_failures(
        checks,
        REQUIRED,
        pr_state=None,
        head_sha="sha-1",
        record_attempts=False,
    )
    assert result.rerun_run_ids == ()
    assert result.definitive_failed == ("Tests passed",)
    assert result.check_rerun_attempts == {"sha-1": {}}


def test_classify_groups_multiple_failed_checks_by_run_id() -> None:
    checks = [
        {"name": "Tests passed", "state": "FAILURE", "link": _link(100, 1)},
        {"name": "Lint & Format", "state": "FAILURE", "link": _link(100, 2)},
    ]
    result = classify_check_failures(
        checks,
        REQUIRED,
        pr_state=None,
        head_sha="sha-1",
    )
    # Both checks share the same workflow run id; only one rerun is requested.
    assert result.rerun_run_ids == (100,)
    assert result.check_rerun_attempts == {
        "sha-1": {"Tests passed": [100], "Lint & Format": [100]}
    }
