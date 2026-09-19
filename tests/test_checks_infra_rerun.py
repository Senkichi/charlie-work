"""Infra-failure rerun-debounce tests for ``charlie_work.checks``.

Split out of ``tests/test_checks.py`` (issue #1565, Track-1 shoulder):
``classify_infra_failures``'s CANCELLED/INFRA_FAILURE/TIMED_OUT rerun
path with its count-based per-run-id attempt cap (issue #841), and the
caller-level pin for the bucket-over-terminal-state fix (issue #985).
"""

from __future__ import annotations

from _checks_fixtures import REQUIRED, _link

from charlie_work.checks import (
    InfraRerunResult,
    classify_infra_failures,
    summarize_checks,
)


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


def test_classify_infra_failures_excludes_run_whose_bucket_overrides_terminal_state() -> None:
    """Caller-level pin: two runs share one required name. One is a genuine
    infra failure; the other carries a terminal `state` but `bucket == "pass"`
    (the disagreement input). Before #985's fix, `_is_infra_run` read only
    `state` and both run ids were queued for rerun; after the fix, only the
    genuinely infra run id is."""
    checks = [
        {"name": "Tests passed", "state": "INFRA_FAILURE", "link": _link(100, 1)},
        {"name": "Tests passed", "state": "CANCELLED", "bucket": "pass", "link": _link(200, 1)},
        {"name": "Lint & Format", "bucket": "pass"},
    ]
    # The aggregator (summarize_checks -> _classify_check_run) already treats
    # the second run as PASS, so the name is still correctly infra_failed
    # (driven by the first run) rather than fully passing.
    summary = summarize_checks(checks, REQUIRED)
    assert summary.infra_failed == ("Tests passed",)

    result = classify_infra_failures(checks, REQUIRED, pr_state=None, head_sha="sha-1")
    assert result.rerun_run_ids == (100,)
    assert result.infra_rerun_attempts == {"sha-1": {"Tests passed": {"100": 1}}}
