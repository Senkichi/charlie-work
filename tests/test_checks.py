from __future__ import annotations

import pytest

from charlie_work.checks import (
    CheckDebounceResult,
    InfraRerunResult,
    _CheckClassification,
    _classify_check_run,
    _is_failing_run,
    _is_infra_run,
    _run_id_from_link,
    classify_check_failures,
    classify_infra_failures,
    summarize_checks,
)
from charlie_work.workflow import _non_required_check_findings


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


def test_summarize_checks_skipped_required_check_not_failing() -> None:
    """A required check with conclusion SKIPPED is a legitimate non-outcome
    and must not be counted as failing."""
    checks = [{"name": "Tests passed", "state": "SKIPPED"}]
    summary = summarize_checks(checks, ("Tests passed",))
    assert "Tests passed" not in summary.failed
    assert "Tests passed" not in summary.infra_failed
    assert summary.ready is True


def test_summarize_checks_neutral_required_check_not_failing() -> None:
    """A required check with conclusion NEUTRAL is a legitimate non-outcome
    and must not be counted as failing."""
    checks = [{"name": "Tests passed", "state": "NEUTRAL"}]
    summary = summarize_checks(checks, ("Tests passed",))
    assert "Tests passed" not in summary.failed
    assert "Tests passed" not in summary.infra_failed
    assert summary.ready is True


def test_is_failing_run_skipped_and_neutral_return_false() -> None:
    """SKIPPED and NEUTRAL single check runs are not code failures."""
    assert _is_failing_run({"name": "x", "state": "SKIPPED"}) is False
    assert _is_failing_run({"name": "x", "state": "NEUTRAL"}) is False


def test_non_required_check_findings_skipped_and_neutral_not_listed() -> None:
    """workflow.py must continue to treat SKIPPED/NEUTRAL as non-outcomes
    for non-required checks (regression guard for the shared helper)."""
    checks = [{"name": "Optional Job", "state": "SKIPPED"}]
    failing, cancelled = _non_required_check_findings(checks, ("Tests",))
    assert failing == ()
    assert cancelled == ()

    checks = [{"name": "Optional Job", "state": "NEUTRAL"}]
    failing, cancelled = _non_required_check_findings(checks, ("Tests",))
    assert failing == ()
    assert cancelled == ()


@pytest.mark.parametrize("state", ["SKIPPED", "NEUTRAL"])
def test_all_three_call_sites_agree_skipped_and_neutral_are_not_failures(
    state: str,
) -> None:
    """The shared classification helper must keep summarize_checks,
    _is_failing_run, and _non_required_check_findings consistent."""
    check = {"name": "Check", "state": state}
    summary = summarize_checks([check], ("Check",))
    assert "Check" not in summary.failed
    assert "Check" not in summary.infra_failed
    assert _is_failing_run(check) is False
    failing, cancelled = _non_required_check_findings([check], ("Other",))
    assert failing == ()
    assert cancelled == ()


@pytest.mark.parametrize(
    "state,summary_bucket,is_failing,is_infra,non_required_failing,non_required_cancelled",
    [
        ("SUCCESS", "passed", False, False, (), ()),
        ("PENDING", "pending", False, False, (), ()),
        ("FAILURE", "failed", True, False, ("Check",), ()),
        ("CANCELLED", "infra_failed", False, True, (), ("Check",)),
        ("INFRA_FAILURE", "infra_failed", False, True, ("Check",), ()),
        ("TIMED_OUT", "infra_failed", False, True, ("Check",), ()),
    ],
)
def test_known_check_states_classify_consistently(
    state: str,
    summary_bucket: str,
    is_failing: bool,
    is_infra: bool,
    non_required_failing: tuple[str, ...],
    non_required_cancelled: tuple[str, ...],
) -> None:
    """Existing state classifications must be unchanged at all four call sites
    (``summarize_checks``, ``_is_failing_run``, ``_is_infra_run``,
    ``_non_required_check_findings``) after the SKIPPED/NEUTRAL carve-out is
    moved to a shared helper (issue #850) and after ``_is_infra_run`` is
    collapsed onto the same helper (issue #985)."""
    check = {"name": "Check", "state": state}
    summary = summarize_checks([check], ("Check",))
    assert getattr(summary, summary_bucket) == ("Check",)
    assert _is_failing_run(check) is is_failing
    assert _is_infra_run(check) is is_infra
    failing, cancelled = _non_required_check_findings([check], ("Other",))
    assert failing == non_required_failing
    assert cancelled == non_required_cancelled


def test_is_infra_run_and_is_failing_run_mutually_exclusive() -> None:
    """Documents the counterpart relationship `_is_infra_run`'s docstring
    claims: no single check run is ever both a code failure and an infra
    failure. This holds even on the pre-#985 implementation (both read
    `state` and CANCELLED/INFRA_FAILURE/TIMED_OUT don't overlap FAILURE), so
    it is not on its own a regression guard for #985 -- see
    `test_is_infra_run_defers_to_bucket_over_terminal_state` for the case
    that actually discriminates the fix."""
    for state in ("SUCCESS", "PENDING", "FAILURE", "CANCELLED", "INFRA_FAILURE", "TIMED_OUT"):
        check = {"name": "Check", "state": state}
        assert not (_is_infra_run(check) and _is_failing_run(check))


def test_is_infra_run_defers_to_bucket_over_terminal_state() -> None:
    """The disagreement case from issue #985: `_classify_check_run` resolves
    `bucket == "pass"`/`"pending"` *before* it ever looks at a terminal
    `state`, so a run carrying a terminal state alongside a pass/pending
    bucket is PASS/PENDING, not infra. The pre-#985 `_is_infra_run` read only
    `state` and would have returned True here -- this is the input on which
    the two implementations disagreed."""
    assert _is_infra_run({"name": "x", "state": "CANCELLED", "bucket": "pass"}) is False
    assert _is_infra_run({"name": "x", "state": "INFRA_FAILURE", "bucket": "pending"}) is False
    # Sanity check on the classifier itself: the same input resolves to PASS,
    # not to CANCELLED/INFRA -- that's *why* _is_infra_run must return False.
    assert (
        _classify_check_run({"state": "CANCELLED", "bucket": "pass"}) == _CheckClassification.PASS
    )
    assert (
        _classify_check_run({"state": "INFRA_FAILURE", "bucket": "pending"})
        == _CheckClassification.PENDING
    )


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


def test_is_infra_run_delegates_to_classify_check_run() -> None:
    """Guard against re-duplication: `_is_infra_run` must call the shared
    `_classify_check_run` helper rather than re-inlining its own copy of the
    terminal-state check (the exact regression this test's issue, #985, was
    filed against). Matches the `inspect.getsource` delegation-guard pattern
    used elsewhere in this repo (see
    `test_dispatch_rework_reaps_unconditionally_when_max_concurrent_zero`)."""
    import inspect

    from charlie_work import checks as checks_module

    source = inspect.getsource(checks_module._is_infra_run)
    assert "_classify_check_run" in source, (
        "_is_infra_run must delegate to _classify_check_run, not reimplement it"
    )
    # Matching only what's forbidden fails open (a rewritten literal check
    # would still pass a substring search for "_classify_check_run" if that
    # name merely appeared in a comment) -- so also assert the raw
    # GitHub-state literals the pre-#985 body hardcoded (distinct from the
    # legitimate `_CheckClassification.CANCELLED`/`.INFRA` enum references
    # the fixed body uses) are gone.
    for literal in ("INFRA_FAILURE", "TIMED_OUT", '"CANCELLED"', "'CANCELLED'"):
        assert literal not in source, (
            f"_is_infra_run must not re-inline the {literal!r} state literal; "
            "route through _CheckClassification instead"
        )


def test_summarize_checks_requires_all_configured_checks() -> None:
    checks = [
        {"name": "Tests passed", "state": "SUCCESS"},
        {"name": "Lint & Format", "bucket": "pass"},
        {"name": "Pre-commit", "state": "FAILURE"},
    ]

    summary = summarize_checks(checks, ("Tests passed", "Lint & Format", "Pre-commit"))

    assert summary.ready is False
    assert summary.passed == ("Tests passed", "Lint & Format")
    assert summary.failed == ("Pre-commit",)
    assert summary.infra_failed == ()


def test_summarize_checks_duplicate_runs_failure_then_success() -> None:
    """Regression test for issue #1: duplicate runs with FAILURE then SUCCESS should classify as failed."""
    checks = [
        {"name": "test", "state": "FAILURE"},
        {"name": "test", "state": "SUCCESS"},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is False
    assert summary.failed == ("test",)
    assert summary.passed == ()
    assert summary.infra_failed == ()


def test_summarize_checks_duplicate_runs_success_then_failure() -> None:
    """Regression test for issue #1: duplicate runs with SUCCESS then FAILURE should classify as failed."""
    checks = [
        {"name": "test", "state": "SUCCESS"},
        {"name": "test", "state": "FAILURE"},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is False
    assert summary.failed == ("test",)
    assert summary.passed == ()
    assert summary.infra_failed == ()


def test_summarize_checks_duplicate_runs_all_success() -> None:
    """Duplicate runs with all SUCCESS should classify as passed."""
    checks = [
        {"name": "test", "state": "SUCCESS"},
        {"name": "test", "state": "SUCCESS"},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is True
    assert summary.passed == ("test",)
    assert summary.failed == ()
    assert summary.infra_failed == ()


def test_summarize_checks_duplicate_runs_pending_then_success() -> None:
    """Duplicate runs with PENDING then SUCCESS should classify as pending."""
    checks = [
        {"name": "test", "state": "PENDING"},
        {"name": "test", "state": "SUCCESS"},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is False
    assert summary.pending == ("test",)
    assert summary.passed == ()
    assert summary.failed == ()
    assert summary.infra_failed == ()


def test_summarize_checks_duplicate_runs_failure_then_pending() -> None:
    """Duplicate runs with FAILURE then PENDING should classify as failed (worst-of)."""
    checks = [
        {"name": "test", "state": "FAILURE"},
        {"name": "test", "state": "PENDING"},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is False
    assert summary.failed == ("test",)
    assert summary.pending == ()
    assert summary.infra_failed == ()


def test_summarize_checks_empty_state_and_bucket_classifies_as_pending() -> None:
    """Regression test for issue #95: null/empty state+bucket should classify as pending."""
    checks = [
        {"name": "test", "state": None, "bucket": None},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is False
    assert summary.pending == ("test",)
    assert summary.failed == ()
    assert summary.infra_failed == ()


def test_summarize_checks_empty_string_state_and_bucket_classifies_as_pending() -> None:
    """Regression test for issue #95: empty string state+bucket should classify as pending."""
    checks = [
        {"name": "test", "state": "", "bucket": ""},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is False
    assert summary.pending == ("test",)
    assert summary.failed == ()


def test_summarize_checks_cancelled_classifies_as_infra_failed() -> None:
    """Regression test for issue #210: CANCELLED state should classify as infrastructure failure."""
    checks = [
        {"name": "test", "state": "CANCELLED"},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is False
    assert summary.infra_failed == ("test",)
    assert summary.failed == ()
    assert summary.pending == ()


def test_summarize_checks_timed_out_classifies_as_infra_failed() -> None:
    """Issue #841: TIMED_OUT is a documented GitHub Actions conclusion value
    distinct from this repo's observed CANCELLED-on-timeout behavior, but it
    must not fall through the catch-all into a code failure -- there is no
    code fix for a job that ran out of time."""
    checks = [
        {"name": "test", "state": "TIMED_OUT"},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is False
    assert summary.infra_failed == ("test",)
    assert summary.failed == ()


def test_summarize_checks_cancelled_case_insensitive() -> None:
    """CANCELLED state classification should be case-insensitive."""
    checks = [
        {"name": "test", "state": "cancelled"},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is False
    assert summary.infra_failed == ("test",)
    assert summary.failed == ()


def test_summarize_checks_mixed_cancelled_and_failure() -> None:
    """Mixed CANCELLED and FAILURE states should classify each separately."""
    checks = [
        {"name": "test1", "state": "CANCELLED"},
        {"name": "test2", "state": "FAILURE"},
    ]

    summary = summarize_checks(checks, ("test1", "test2"))

    assert summary.ready is False
    assert summary.infra_failed == ("test1",)
    assert summary.failed == ("test2",)
    assert summary.pending == ()


def test_summarize_checks_duplicate_runs_cancelled_then_success() -> None:
    """Duplicate runs with CANCELLED then SUCCESS should classify as infra_failed (worst-of)."""
    checks = [
        {"name": "test", "state": "CANCELLED"},
        {"name": "test", "state": "SUCCESS"},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is False
    assert summary.infra_failed == ("test",)
    assert summary.failed == ()
    assert summary.pending == ()


def test_summarize_checks_failure_takes_priority_over_cancelled() -> None:
    """FAILURE should take priority over CANCELLED in worst-of semantics."""
    checks = [
        {"name": "test", "state": "CANCELLED"},
        {"name": "test", "state": "FAILURE"},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is False
    assert summary.failed == ("test",)
    assert summary.infra_failed == ()
    assert summary.pending == ()


def test_summarize_checks_infra_failure_marker_classifies_as_infra_failed() -> None:
    """INFRA_FAILURE marker state should classify as infrastructure failure."""
    checks = [
        {"name": "test", "state": "INFRA_FAILURE"},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is False
    assert summary.infra_failed == ("test",)
    assert summary.failed == ()
    assert summary.pending == ()


def test_summarize_checks_infra_failure_case_insensitive() -> None:
    """INFRA_FAILURE state classification should be case-insensitive."""
    checks = [
        {"name": "test", "state": "infra_failure"},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is False
    assert summary.infra_failed == ("test",)
    assert summary.failed == ()


def test_summarize_checks_failure_takes_priority_over_infra_failure() -> None:
    """FAILURE should take priority over INFRA_FAILURE in worst-of semantics."""
    checks = [
        {"name": "test", "state": "INFRA_FAILURE"},
        {"name": "test", "state": "FAILURE"},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is False
    assert summary.failed == ("test",)
    assert summary.infra_failed == ()


def test_summarize_checks_none_returns_unavailable_required_checks() -> None:
    """Command-level gh failure (checks=None) marks every required check unavailable."""
    summary = summarize_checks(None, ("Tests",))

    assert summary.ready is False
    assert summary.unavailable == ("Tests",)
    assert summary.passed == ()
    assert summary.pending == ()
    assert summary.failed == ()
    assert summary.missing == ()
