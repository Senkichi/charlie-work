"""Stale-CI verdict detection and skip-orchestration tests for the janitor.

Split out of ``tests/test_janitor.py`` (issue #1558, Track 1):
``is_stale_ci_verdict`` and the ``run_janitor`` wiring that skips the
no-op-rework check on a stale all-green citation verdict while failing
closed on red required checks, escalated decisions, prose findings, and
unavailable summaries.
"""

from __future__ import annotations

from pathlib import Path

from _janitor_fixtures import (
    _STALE_REQUIRED,
    _all_green_summary,
    _config,
    _green_checks,
    _green_pr,
    _stale_decision,
)

from charlie_work.checks import CheckSummary
from charlie_work.janitor import (
    is_stale_ci_verdict,
    run_janitor,
)


def test_is_stale_ci_verdict_true_when_all_green() -> None:
    decision = _stale_decision(["Tests passed: .github:18 — Process completed with exit code 1."])
    assert is_stale_ci_verdict(decision, _all_green_summary()) is True


def test_is_stale_ci_verdict_false_when_summary_none() -> None:
    decision = _stale_decision(["Tests passed: .github:18 — Process completed with exit code 1."])
    assert is_stale_ci_verdict(decision, None) is False


def test_is_stale_ci_verdict_false_when_non_citation_decision() -> None:
    decision = _stale_decision(["src/foo.py:42 — off-by-one error in the loop bound."])
    assert is_stale_ci_verdict(decision, _all_green_summary()) is False


def test_is_stale_ci_verdict_false_when_required_check_still_failed() -> None:
    decision = _stale_decision(["Tests passed: .github:18 — Process completed with exit code 1."])
    summary = CheckSummary(
        required=_STALE_REQUIRED,
        passed=("Pre-commit",),
        pending=(),
        failed=("Tests passed",),
        missing=(),
        infra_failed=(),
        unavailable=(),
    )
    assert is_stale_ci_verdict(decision, summary) is False


def test_is_stale_ci_verdict_false_when_required_check_pending() -> None:
    decision = _stale_decision(["Tests passed: .github:18 — Process completed with exit code 1."])
    summary = CheckSummary(
        required=_STALE_REQUIRED,
        passed=("Pre-commit",),
        pending=("Tests passed",),
        failed=(),
        missing=(),
        infra_failed=(),
        unavailable=(),
    )
    assert is_stale_ci_verdict(decision, summary) is False


def test_is_stale_ci_verdict_false_when_required_check_missing() -> None:
    decision = _stale_decision(["Tests passed: .github:18 — Process completed with exit code 1."])
    summary = CheckSummary(
        required=_STALE_REQUIRED,
        passed=("Pre-commit",),
        pending=(),
        failed=(),
        missing=("Tests passed",),
        infra_failed=(),
        unavailable=(),
    )
    assert is_stale_ci_verdict(decision, summary) is False


def test_is_stale_ci_verdict_false_when_required_check_infra_failed() -> None:
    decision = _stale_decision(["Tests passed: .github:18 — Process completed with exit code 1."])
    summary = CheckSummary(
        required=_STALE_REQUIRED,
        passed=("Pre-commit",),
        pending=(),
        failed=(),
        missing=(),
        infra_failed=("Tests passed",),
        unavailable=(),
    )
    assert is_stale_ci_verdict(decision, summary) is False


def test_run_janitor_stale_ci_skips_no_op_rework_check() -> None:
    """A stale-CI request_changes verdict (all findings cite required checks
    that are green now) must skip the no-op-rework check entirely: no
    no-op failure, ``no_op_check_skipped_stale_ci`` True, and the skip
    warning present. Without the review_decision this exact pr_state/head
    combination fails via the SHA-fallback path (see
    test_no_op_rework_fallback_to_sha_without_patch_id) -- the skip is what
    changes here."""
    pr = _green_pr(headRefOid="abc123")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",  # Same head -> would trigger SHA-fallback no-op
    }
    decision = _stale_decision(["Tests passed: .github:18 — Process completed with exit code 1."])

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        review_decision=decision,
    )

    assert verdict.ok is True, f"Expected ok=True but got {verdict.failures}"
    assert verdict.no_op_check_skipped_stale_ci is True
    assert verdict.is_no_op_rework is False
    assert verdict.failures == ()
    assert any("No-op rework check skipped" in w for w in verdict.warnings)
    assert any("stale-CI" in w for w in verdict.warnings)


def test_run_janitor_no_review_decision_preserves_no_op_check() -> None:
    """Positive control: a review_decision mapping matching the recorded
    verdict (the normal shape every real caller passes -- run_janitor's own
    ``review_decision`` param, resolved via ``self._review_decision(pr_number)``
    at every production call site) leaves the no-op check active --
    ``no_op_check_skipped_stale_ci`` stays False, and the SHA-fallback no-op
    failure fires exactly as it did before issue #1116."""
    pr = _green_pr(headRefOid="abc123")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        review_decision=pr_state,
    )

    assert verdict.ok is False
    assert verdict.no_op_check_skipped_stale_ci is False
    assert verdict.is_no_op_rework is True
    assert any("PR head unchanged since request_changes verdict" in f for f in verdict.failures)
    assert not any("No-op rework check skipped" in w for w in verdict.warnings)


def test_run_janitor_no_review_decision_skips_no_op_check() -> None:
    """Issue #1362 Stage 1: without a resolved review_decision mapping (the
    file-first reader's payload), the no-op-rework gate must NOT fall back to
    reading ``pr_state["decision"]`` directly -- that is exactly the stale
    state.json divergence AC1 exists to eliminate (a crash between the file
    write and the state write leaves state.json holding a stale verdict).
    ``review_decision=None`` must therefore leave the gate inactive even
    though ``pr_state`` alone would have triggered it under the pre-#1362
    behavior."""
    pr = _green_pr(headRefOid="abc123")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        review_decision=None,
    )

    assert verdict.ok is True
    assert verdict.is_no_op_rework is False
    assert not any(
        "PR head unchanged since request_changes verdict" in f for f in verdict.failures
    )


def test_run_janitor_stale_ci_skip_fails_closed_on_red_required_check() -> None:
    """A required check still failing must never suppress the no-op check,
    even though the decision has the exact stale-CI shape: is_stale_ci_verdict
    is False when ``summary.ready`` is False, so the skip must not fire."""
    pr = _green_pr(headRefOid="abc123")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }
    red_checks = [
        {"name": "Tests passed", "state": "FAILURE"},
        {"name": "Lint & Format", "bucket": "pass"},
    ]
    decision = _stale_decision(["Tests passed: .github:18 — Process completed with exit code 1."])

    verdict = run_janitor(
        pr,
        red_checks,
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        review_decision=decision,
    )

    assert verdict.no_op_check_skipped_stale_ci is False
    assert verdict.is_no_op_rework is True
    assert any("PR head unchanged since request_changes verdict" in f for f in verdict.failures)
    assert not any("No-op rework check skipped" in w for w in verdict.warnings)


def test_run_janitor_stale_ci_skip_fails_closed_on_escalated_decision() -> None:
    """An escalated request_changes verdict is never treated as stale-CI
    (required_check_citation_names returns None for escalated=True), so the
    no-op check must still run."""
    pr = _green_pr(headRefOid="abc123")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }
    decision = {
        "decision": "request_changes",
        "escalated": True,
        "required_changes": ["Tests passed: .github:18 — Process completed with exit code 1."],
    }

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        review_decision=decision,
    )

    assert verdict.no_op_check_skipped_stale_ci is False
    assert verdict.is_no_op_rework is True
    assert any("PR head unchanged since request_changes verdict" in f for f in verdict.failures)


def test_run_janitor_stale_ci_skip_fails_closed_on_prose_finding() -> None:
    """A request_changes verdict citing a real code finding (not a required-
    check status observation) is not stale-CI-shaped, so the no-op check
    must still run even though every other condition (same head, green
    checks) matches the skip scenario."""
    pr = _green_pr(headRefOid="abc123")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }
    decision = _stale_decision(["src/foo.py:42 — off-by-one error in the loop bound."])

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        review_decision=decision,
    )

    assert verdict.no_op_check_skipped_stale_ci is False
    assert verdict.is_no_op_rework is True
    assert any("PR head unchanged since request_changes verdict" in f for f in verdict.failures)


def test_run_janitor_stale_ci_skip_fails_closed_when_summary_unavailable() -> None:
    """No required checks configured means ``summary`` is None -- even a
    perfectly stale-CI-shaped decision must not suppress the no-op check,
    since there is nothing live to confirm the citation is actually green."""
    pr = _green_pr(headRefOid="abc123")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }
    decision = _stale_decision(["Tests passed: .github:18 — Process completed with exit code 1."])

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(required_checks=()),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        review_decision=decision,
    )

    assert verdict.no_op_check_skipped_stale_ci is False
    assert verdict.is_no_op_rework is True
    assert any("PR head unchanged since request_changes verdict" in f for f in verdict.failures)
    assert not any("No-op rework check skipped" in w for w in verdict.warnings)
