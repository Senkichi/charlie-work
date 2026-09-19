"""Required-check gate and citation-name extraction tests for the janitor.

Split out of ``tests/test_janitor.py`` (issue #1558, Track 1): the
``_check_required_checks`` verdict surface (failure / pending / missing /
unavailable / infra-failed handling, missing-checks-only and check-failure
block classification, rerun run-id reporting) and the
``required_check_citation_names`` review-decision parser.
"""

from __future__ import annotations

from pathlib import Path

from _janitor_fixtures import (
    _STALE_REQUIRED,
    _config,
    _green_pr,
    _stale_decision,
)

from charlie_work.janitor import (
    required_check_citation_names,
    run_janitor,
)


def test_required_check_failure_blocks() -> None:
    checks = [
        {"name": "Tests passed", "state": "FAILURE"},
        {"name": "Lint & Format", "bucket": "pass"},
    ]

    verdict = run_janitor(_green_pr(), checks, _config(), repo_root=Path.cwd())

    assert verdict.ok is False
    assert any("Tests passed" in f for f in verdict.failures)


def test_required_check_missing_blocks() -> None:
    checks = [{"name": "Lint & Format", "bucket": "pass"}]

    verdict = run_janitor(_green_pr(), checks, _config(), repo_root=Path.cwd())

    assert verdict.ok is False
    assert any("missing" in f.lower() and "Tests passed" in f for f in verdict.failures)


def test_required_check_missing_only_is_missing_checks_only_block() -> None:
    """Issue #1133: when "Required check(s) missing" is the SOLE janitor
    failure, the verdict exposes ``is_missing_checks_only_block`` so
    ``_is_dead_blocker`` can carve out the transient (not-yet-reported-CI)
    population from the durably-stuck one. Branch on the structured flag,
    never on the failure-message text.
    """
    checks = [{"name": "Lint & Format", "bucket": "pass"}]

    verdict = run_janitor(_green_pr(), checks, _config(), repo_root=Path.cwd())

    assert verdict.ok is False
    assert verdict.missing_required_checks == ("Tests passed",)
    assert verdict.is_missing_checks_only_block is True


def test_required_check_missing_with_other_blocker_is_not_missing_checks_only_block() -> None:
    """Issue #1133: a co-occurring durable failure (merge conflict) disqualifies
    the transient carve-out -- the PR is durably stuck, not waiting on CI.
    """
    checks = [{"name": "Lint & Format", "bucket": "pass"}]

    verdict = run_janitor(
        _green_pr(mergeable="CONFLICTING"), checks, _config(), repo_root=Path.cwd()
    )

    assert verdict.ok is False
    assert verdict.missing_required_checks == ("Tests passed",)
    assert verdict.is_missing_checks_only_block is False


def test_required_check_failed_is_not_missing_checks_only_block() -> None:
    """Issue #1133: a failed required check (CI ran and failed) is durable,
    not transient -- ``is_missing_checks_only_block`` must be False.
    """
    checks = [
        {"name": "Tests passed", "state": "FAILURE"},
        {"name": "Lint & Format", "bucket": "pass"},
    ]

    verdict = run_janitor(_green_pr(), checks, _config(), repo_root=Path.cwd())

    assert verdict.ok is False
    assert verdict.failed_required_checks == ("Tests passed",)
    assert verdict.missing_required_checks == ()
    assert verdict.is_missing_checks_only_block is False


def test_required_checks_unavailable_blocks() -> None:
    verdict = run_janitor(_green_pr(), None, _config(), repo_root=Path.cwd())

    assert verdict.ok is False
    assert any("Checks unavailable (gh failure)" in f for f in verdict.failures)


def test_required_check_pending_warns_not_fails() -> None:
    checks = [
        {"name": "Tests passed", "state": "PENDING"},
        {"name": "Lint & Format", "bucket": "pass"},
    ]

    verdict = run_janitor(_green_pr(), checks, _config(), repo_root=Path.cwd())

    assert verdict.ok is True
    assert verdict.failures == ()
    assert any("pending" in w.lower() and "Tests passed" in w for w in verdict.warnings)


def test_required_check_failure_exposes_failed_required_checks() -> None:
    checks = [
        {"name": "Tests passed", "state": "FAILURE"},
        {"name": "Lint & Format", "bucket": "pass"},
    ]

    verdict = run_janitor(_green_pr(), checks, _config(), repo_root=Path.cwd())

    assert verdict.ok is False
    assert verdict.failed_required_checks == ("Tests passed",)
    assert verdict.is_check_failure_block is True
    assert any("Tests passed" in f for f in verdict.failures)


def test_required_check_failure_with_other_blocker_is_not_check_failure_block() -> None:
    checks = [
        {"name": "Tests passed", "state": "FAILURE"},
        {"name": "Lint & Format", "bucket": "pass"},
    ]

    verdict = run_janitor(_green_pr(isDraft=True), checks, _config(), repo_root=Path.cwd())

    assert verdict.ok is False
    assert verdict.failed_required_checks == ("Tests passed",)
    assert verdict.is_check_failure_block is False
    # Issue #818: draft co-occurring with a real failing required check must
    # NOT be treated as "otherwise ready" -- a draft PR with a genuine
    # failure is not silently auto-readied.
    assert verdict.is_draft is True
    assert verdict.is_draft_only_block is False


def test_required_check_infra_failed_is_not_check_failure_block() -> None:
    checks = [{"name": "Tests passed", "state": "CANCELLED"}]

    verdict = run_janitor(_green_pr(), checks, _config(), repo_root=Path.cwd())

    assert verdict.ok is False
    assert verdict.failed_required_checks == ()
    assert verdict.is_check_failure_block is False
    assert any("infrastructure" in f.lower() for f in verdict.failures)


def test_infra_blocked_block_holds_when_non_required_check_also_fails() -> None:
    """Round-2 #1383 guard: a fleet-wide outage typically fails non-required
    jobs too (e.g. an optional docs/lint job that also never started). The
    janitor gate must still classify the PR as ``is_infra_blocked_block`` so
    the protection is not silently disqualified.

    This locks in the structural reason the disqualification does NOT happen:
    ``summarize_checks`` only iterates the ``required`` tuple, so a
    non-required check's FAILURE never enters the janitor ``failures`` list
    and therefore never inflates ``non_required_checks_failures`` (which
    counts failures NOT contributed by ``_check_required_checks`` -- draft,
    state, mergeable, linked-issue, body, no-op-rework -- not literally
    "non-required check failures"). The required check is shown here in its
    post-enrichment ``INFRA_BLOCKED`` marker state, exactly as
    ``_enrich_checks_infra_blocked`` would rewrite it before ``run_janitor``
    sees it.
    """
    checks = [
        {"name": "Tests passed", "state": "INFRA_BLOCKED"},
        {"name": "Lint & Format", "bucket": "pass"},
        # A non-required job that also failed with a zero-step / infra
        # signature during the same fleet-wide outage.
        {"name": "Optional Docs Build", "state": "FAILURE"},
    ]

    verdict = run_janitor(_green_pr(), checks, _config(), repo_root=Path.cwd())

    assert verdict.is_infra_blocked_block is True
    assert verdict.failed_required_checks == ()
    # The only recorded failure is the infra-blocked required check -- the
    # non-required FAILURE contributed nothing to ``failures``.
    assert len(verdict.failures) == 1
    assert "infra-blocked" in verdict.failures[0].lower()


def test_no_required_checks_configured_skips_check_gate() -> None:
    verdict = run_janitor(_green_pr(), [], _config(required_checks=()))

    assert verdict.ok is True


def test_required_check_first_failure_returns_rerun_run_id() -> None:
    checks = [
        {"name": "Tests passed", "state": "FAILURE", "runId": 100},
        {"name": "Lint & Format", "bucket": "pass"},
    ]
    verdict = run_janitor(_green_pr(), checks, _config(), repo_root=Path.cwd())

    assert verdict.ok is False
    assert verdict.is_check_failure_block is True
    assert verdict.rerun_run_ids == (100,)
    assert verdict.check_rerun_attempts == {"abc123": {"Tests passed": [100]}}


def test_required_check_second_failure_returns_no_rerun() -> None:
    checks = [
        {"name": "Tests passed", "state": "FAILURE", "runId": 100},
        {"name": "Lint & Format", "bucket": "pass"},
    ]
    pr_state = {"check_rerun_attempts": {"abc123": {"Tests passed": [100]}}}
    verdict = run_janitor(_green_pr(), checks, _config(), pr_state=pr_state, repo_root=Path.cwd())

    assert verdict.ok is False
    assert verdict.is_check_failure_block is True
    assert verdict.rerun_run_ids == ()
    assert verdict.failed_required_checks == ("Tests passed",)


def test_required_check_citation_names_matches_contaminated_shape() -> None:
    """The real #1111 shape: a check-status observation, not a code finding."""
    decision = _stale_decision(["Tests passed: .github:18 — Process completed with exit code 1."])
    assert required_check_citation_names(decision, _STALE_REQUIRED) == ("Tests passed",)


def test_required_check_citation_names_leading_whitespace_still_matches() -> None:
    decision = _stale_decision(
        ["  Tests passed: .github:18 — Process completed with exit code 1."]
    )
    assert required_check_citation_names(decision, _STALE_REQUIRED) == ("Tests passed",)


def test_required_check_citation_names_mixed_entries_returns_none() -> None:
    """One citation plus one real code finding must not be treated as stale."""
    decision = _stale_decision(
        [
            "Tests passed: .github:18 — Process completed with exit code 1.",
            "src/foo.py:42 — off-by-one error in the loop bound.",
        ]
    )
    assert required_check_citation_names(decision, _STALE_REQUIRED) is None


def test_required_check_citation_names_prose_only_entry_returns_none() -> None:
    decision = _stale_decision(["The implementation does not handle the empty-list case."])
    assert required_check_citation_names(decision, _STALE_REQUIRED) is None


def test_required_check_citation_names_empty_required_changes_returns_none() -> None:
    decision = _stale_decision([])
    assert required_check_citation_names(decision, _STALE_REQUIRED) is None


def test_required_check_citation_names_approved_decision_returns_none() -> None:
    decision = {
        "decision": "approve",
        "escalated": False,
        "required_changes": ["Tests passed: .github:18 — Process completed with exit code 1."],
    }
    assert required_check_citation_names(decision, _STALE_REQUIRED) is None


def test_required_check_citation_names_escalated_returns_none() -> None:
    decision = {
        "decision": "request_changes",
        "escalated": True,
        "required_changes": ["Tests passed: .github:18 — Process completed with exit code 1."],
    }
    assert required_check_citation_names(decision, _STALE_REQUIRED) is None


def test_required_check_citation_names_non_string_entry_returns_none() -> None:
    decision = _stale_decision(
        [{"check": "Tests passed"}]  # type: ignore[list-item]
    )
    assert required_check_citation_names(decision, _STALE_REQUIRED) is None


def test_required_check_citation_names_empty_required_tuple_returns_none() -> None:
    decision = _stale_decision(["Tests passed: .github:18 — Process completed with exit code 1."])
    assert required_check_citation_names(decision, ()) is None


def test_required_check_citation_names_none_decision_returns_none() -> None:
    assert required_check_citation_names(None, _STALE_REQUIRED) is None
