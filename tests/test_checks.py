from __future__ import annotations

from charlie_work.checks import (
    CheckDebounceResult,
    InfraRerunResult,
    _run_id_from_link,
    classify_check_failures,
    classify_infra_failures,
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


# classify_infra_failures (issue #841): CANCELLED/INFRA_FAILURE/TIMED_OUT
# required checks. Unlike classify_check_failures, attempts are COUNTS per
# run id (not set membership) because `gh run rerun` reuses the same run id
# on every retry -- verified live on two production reruns, both landing as
# run_attempt=2 on the SAME run id, never a new one. That means "has this run
# id been attempted before" can't distinguish attempt 1 from attempt 2; only
# a count can, which is what the attempt_cap enforcement below relies on.


def test_classify_infra_first_cancel_triggers_rerun() -> None:
    checks = [
        {"name": "Tests passed", "state": "CANCELLED", "link": _link(100, 1)},
        {"name": "Lint & Format", "bucket": "pass"},
    ]
    result = classify_infra_failures(
        checks,
        REQUIRED,
        pr_state=None,
        head_sha="sha-1",
    )
    assert result == InfraRerunResult(
        rerun_run_ids=(100,),
        infra_rerun_attempts={"sha-1": {"Tests passed": {"100": 1}}},
        definitive_failed=(),
    )


def test_classify_infra_second_cancel_still_under_cap_triggers_rerun_again() -> None:
    """MUTATION CHECK: with the default cap of 2, a run id retried once (count=1)
    must still be eligible for a second rerun. This fails if the code used
    set-membership (attempted-once == never-again) instead of a count, which
    would be wrong here because `gh run rerun` reuses the same run id."""
    checks = [
        {"name": "Tests passed", "state": "CANCELLED", "link": _link(100, 1)},
        {"name": "Lint & Format", "bucket": "pass"},
    ]
    pr_state = {"infra_rerun_attempts": {"sha-1": {"Tests passed": {"100": 1}}}}
    result = classify_infra_failures(
        checks,
        REQUIRED,
        pr_state,
        head_sha="sha-1",
    )
    assert result.rerun_run_ids == (100,)
    assert result.definitive_failed == ()
    assert result.infra_rerun_attempts == {"sha-1": {"Tests passed": {"100": 2}}}


def test_classify_infra_cap_exhausted_is_definitive_and_does_not_rerun() -> None:
    """Criterion 2: once attempt_cap (default 2) is reached, the check is
    definitive so the caller escalates instead of retrying forever."""
    checks = [
        {"name": "Tests passed", "state": "CANCELLED", "link": _link(100, 1)},
        {"name": "Lint & Format", "bucket": "pass"},
    ]
    pr_state = {"infra_rerun_attempts": {"sha-1": {"Tests passed": {"100": 2}}}}
    result = classify_infra_failures(
        checks,
        REQUIRED,
        pr_state,
        head_sha="sha-1",
    )
    assert result.rerun_run_ids == ()
    assert result.definitive_failed == ("Tests passed",)
    # Attempts unchanged -- the cap-exceeded pass does not itself count as a
    # further attempt.
    assert result.infra_rerun_attempts == {"sha-1": {"Tests passed": {"100": 2}}}


def test_classify_infra_custom_attempt_cap() -> None:
    """The cap is configurable (auto_merge.infra_rerun_attempt_cap), not hardcoded."""
    checks = [
        {"name": "Tests passed", "state": "CANCELLED", "link": _link(100, 1)},
        {"name": "Lint & Format", "bucket": "pass"},
    ]
    pr_state = {"infra_rerun_attempts": {"sha-1": {"Tests passed": {"100": 1}}}}
    result = classify_infra_failures(
        checks,
        REQUIRED,
        pr_state,
        head_sha="sha-1",
        attempt_cap=1,
    )
    assert result.rerun_run_ids == ()
    assert result.definitive_failed == ("Tests passed",)


def test_classify_infra_passing_check_clears_attempt_marker() -> None:
    checks = [
        {"name": "Tests passed", "state": "SUCCESS"},
        {"name": "Lint & Format", "bucket": "pass"},
    ]
    pr_state = {"infra_rerun_attempts": {"sha-1": {"Tests passed": {"100": 1}}}}
    result = classify_infra_failures(
        checks,
        REQUIRED,
        pr_state,
        head_sha="sha-1",
    )
    assert result.rerun_run_ids == ()
    assert result.definitive_failed == ()
    assert result.infra_rerun_attempts == {"sha-1": {}}


def test_classify_infra_new_head_resets_attempts() -> None:
    """Criterion 4 (discriminator): a benign supersede-cancel from an OLD head
    must not leak into a NEW head's rerun budget. `gh pr checks` only ever
    reports the PR's current head, so there is no per-check head_sha to
    compare -- the discriminator is this head-SHA-scoped attempts dict itself
    (mirrors classify_check_failures's identical existing pattern). This test
    fails if the head-SHA keying (`attempts_state = {head_sha: ...}`) is
    deleted and attempts are tracked globally instead."""
    checks = [
        {"name": "Tests passed", "state": "CANCELLED", "link": _link(200, 1)},
        {"name": "Lint & Format", "bucket": "pass"},
    ]
    pr_state = {"infra_rerun_attempts": {"sha-old": {"Tests passed": {"100": 2}}}}
    result = classify_infra_failures(
        checks,
        REQUIRED,
        pr_state,
        head_sha="sha-new",
    )
    # A fresh run id (200) on a fresh head is eligible even though the OLD
    # head's run id (100) had already exhausted its cap.
    assert result.rerun_run_ids == (200,)
    assert result.infra_rerun_attempts == {"sha-new": {"Tests passed": {"200": 1}}}
    # The old head's attempts are dropped, not merged forward.
    assert "sha-old" not in result.infra_rerun_attempts


def test_classify_infra_external_status_failure_is_definitive_without_rerun() -> None:
    checks = [
        {"name": "Tests passed", "state": "CANCELLED", "link": "https://external.ci/run"},
        {"name": "Lint & Format", "bucket": "pass"},
    ]
    result = classify_infra_failures(
        checks,
        REQUIRED,
        pr_state=None,
        head_sha="sha-1",
    )
    assert result.rerun_run_ids == ()
    assert result.definitive_failed == ("Tests passed",)


def test_classify_infra_record_attempts_false_does_not_consume_attempt() -> None:
    """When another janitor blocker co-occurs (record_attempts=False, mirroring
    is_infra_failure_block), the infra rerun must not fire and must not
    consume an attempt -- the same PR will get a fresh look once the other
    blocker clears."""
    checks = [
        {"name": "Tests passed", "state": "CANCELLED", "link": _link(100, 1)},
        {"name": "Lint & Format", "bucket": "pass"},
    ]
    result = classify_infra_failures(
        checks,
        REQUIRED,
        pr_state=None,
        head_sha="sha-1",
        record_attempts=False,
    )
    assert result.rerun_run_ids == ()
    assert result.definitive_failed == ("Tests passed",)
    assert result.infra_rerun_attempts == {"sha-1": {}}


def test_classify_infra_no_required_checks_returns_empty_result() -> None:
    result = classify_infra_failures([], (), pr_state=None, head_sha="sha-1")
    assert result == InfraRerunResult()


def test_classify_infra_timed_out_state_is_treated_as_infra() -> None:
    """TIMED_OUT (the documented GitHub Actions conclusion, distinct from this
    repo's observed CANCELLED-on-timeout behavior) must also route through
    the infra rerun path, not the code-failure path."""
    checks = [
        {"name": "Tests passed", "state": "TIMED_OUT", "link": _link(100, 1)},
        {"name": "Lint & Format", "bucket": "pass"},
    ]
    result = classify_infra_failures(
        checks,
        REQUIRED,
        pr_state=None,
        head_sha="sha-1",
    )
    assert result.rerun_run_ids == (100,)
