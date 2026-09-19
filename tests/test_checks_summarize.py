"""``summarize_checks`` worst-of bucketing tests for ``charlie_work.checks``.

Split out of ``tests/test_checks.py`` (issue #1565, Track-1 shoulder):
required-check aggregation across duplicate runs (issue #1), empty-state
pending (issue #95), CANCELLED/INFRA_FAILURE/TIMED_OUT infra bucketing
(issues #210/#841), unavailable checks, and the INFRA_BLOCKED bucket's
precedence (issue #1383).
"""

from __future__ import annotations

from charlie_work.checks import summarize_checks


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


def test_summarize_checks_infra_blocked_marker_routed_to_infra_blocked_bucket() -> None:
    """A check with state=INFRA_BLOCKED is routed to the infra_blocked bucket, not failed."""
    checks = [{"name": "Tests passed", "state": "INFRA_BLOCKED"}]
    summary = summarize_checks(checks, ("Tests passed",))

    assert summary.infra_blocked == ("Tests passed",)
    assert summary.failed == ()
    assert summary.infra_failed == ()
    assert summary.ready is False


def test_summarize_checks_infra_blocked_does_not_affect_ready_when_empty() -> None:
    """An empty infra_blocked bucket does not block readiness."""
    checks = [{"name": "Tests passed", "state": "SUCCESS"}]
    summary = summarize_checks(checks, ("Tests passed",))

    assert summary.infra_blocked == ()
    assert summary.ready is True


def test_summarize_checks_failed_takes_precedence_over_infra_blocked() -> None:
    """When a check name has both a FAILURE and an INFRA_BLOCKED run, failed wins (worst-of)."""
    checks = [
        {"name": "Tests passed", "state": "INFRA_BLOCKED"},
        {"name": "Tests passed", "state": "FAILURE"},
    ]
    summary = summarize_checks(checks, ("Tests passed",))

    assert summary.failed == ("Tests passed",)
    assert summary.infra_blocked == ()
    assert summary.missing == ()
