"""Per-run state-classification consistency tests for ``charlie_work.checks``.

Split out of ``tests/test_checks.py`` (issue #1565, Track-1 shoulder):
the shared ``_classify_check_run`` helper and its ``_is_failing_run`` /
``_is_infra_run`` / ``_non_required_check_findings`` call sites --
SKIPPED/NEUTRAL carve-outs (issue #850), the bucket-over-terminal-state
resolution (issue #985), and the no-re-duplication delegation guard.
"""

from __future__ import annotations

import pytest

from charlie_work.checks import (
    _CheckClassification,
    _classify_check_run,
    _is_failing_run,
    _is_infra_run,
    summarize_checks,
)
from charlie_work.workflow import _non_required_check_findings


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
