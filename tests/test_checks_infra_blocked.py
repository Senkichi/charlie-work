"""``is_infra_blocked_check`` tests for ``charlie_work.checks``.

Split out of ``tests/test_checks.py`` (issue #1565, Track-1 shoulder):
the zero-step / setup-only / budget-annotation / missing-steps
infra-blocked signals and the disabled-config / real-failure negatives
(issue #1383, including the round-2 missing-``steps``-key regression).
"""

from __future__ import annotations

from charlie_work.checks import is_infra_blocked_check
from charlie_work.config import InfraBlockedConfig


# ---------------------------------------------------------------------------
# Issue #1383: infra_blocked classification
# ---------------------------------------------------------------------------


def test_infra_blocked_zero_step_job_classified() -> None:
    """AC1: a FAILURE job with zero steps is infra_blocked."""
    job = {"conclusion": "FAILURE", "steps": []}
    assert is_infra_blocked_check(job, [], InfraBlockedConfig()) is True


def test_infra_blocked_setup_only_steps_classified() -> None:
    """AC1: a FAILURE job with only setup steps is infra_blocked."""
    job = {
        "conclusion": "FAILURE",
        "steps": [
            {"name": "Set up job", "conclusion": "SUCCESS"},
            {"name": "Checkout", "conclusion": "SUCCESS"},
        ],
    }
    assert is_infra_blocked_check(job, [], InfraBlockedConfig()) is True


def test_infra_blocked_budget_annotation_classified() -> None:
    """AC1: a FAILURE job with a budget annotation is infra_blocked."""
    job = {"conclusion": "FAILURE", "steps": [{"name": "Run tests", "conclusion": "FAILURE"}]}
    annotations = [
        {"message": "The job was not started because your spending limit needs to be increased."}
    ]
    assert is_infra_blocked_check(job, annotations, InfraBlockedConfig()) is True


def test_infra_blocked_instant_fail_no_steps_classified() -> None:
    """AC1: a FAILURE job with no steps array is infra_blocked (zero-step
    signal). The timestamps are incidental -- classification rests on the
    missing ``steps`` key, not on any timing threshold (round-2 #1383)."""
    job = {
        "conclusion": "FAILURE",
        "started_at": "2026-08-21T10:00:00Z",
        "completed_at": "2026-08-21T10:00:05Z",
    }
    assert is_infra_blocked_check(job, [], InfraBlockedConfig()) is True


def test_infra_blocked_missing_steps_key_no_timestamps_classified() -> None:
    """Round-2 #1383 regression: a FAILURE job with NO ``steps`` key at all
    and NO timestamps must classify as infra_blocked, restoring the
    pre-#1383 ``is_infrastructure_failure`` behavior. The prior code only
    handled an empty ``steps`` list under Signal 2 and gated the missing-key
    case on an instant-fail duration that required timestamps -- so this
    shape (a real Actions API omission with no timing data) returned False
    and routed the outage back to rework."""
    job = {"conclusion": "FAILURE"}
    assert is_infra_blocked_check(job, [], InfraBlockedConfig()) is True
    # Also covers the null-value shape (key present, value None).
    assert (
        is_infra_blocked_check({"conclusion": "FAILURE", "steps": None}, [], InfraBlockedConfig())
        is True
    )


def test_infra_blocked_real_test_failure_not_classified() -> None:
    """A FAILURE job with real test steps is NOT infra_blocked."""
    job = {
        "conclusion": "FAILURE",
        "steps": [
            {"name": "Set up job", "conclusion": "SUCCESS"},
            {"name": "Run tests", "conclusion": "FAILURE"},
        ],
    }
    assert is_infra_blocked_check(job, [], InfraBlockedConfig()) is False


def test_infra_blocked_non_failure_conclusion_not_classified() -> None:
    """A SUCCESS/CANCELLED job is never infra_blocked."""
    assert is_infra_blocked_check({"conclusion": "SUCCESS"}, [], InfraBlockedConfig()) is False
    assert is_infra_blocked_check({"conclusion": "CANCELLED"}, [], InfraBlockedConfig()) is False


def test_infra_blocked_disabled_config_not_classified() -> None:
    """When config.enabled is False, no classification happens."""
    cfg = InfraBlockedConfig(enabled=False)
    job = {"conclusion": "FAILURE", "steps": []}
    assert is_infra_blocked_check(job, [], cfg) is False


def test_infra_blocked_custom_annotation_pattern() -> None:
    """Config-listed annotation patterns are matched (not hardcoded in code)."""
    cfg = InfraBlockedConfig(annotation_patterns=("billing exhausted",))
    job = {"conclusion": "FAILURE", "steps": [{"name": "Run tests", "conclusion": "FAILURE"}]}
    annotations = [{"message": "Error: billing exhausted for this account"}]
    assert is_infra_blocked_check(job, annotations, cfg) is True


def test_infra_blocked_missing_steps_classifies_even_with_zero_instant_fail_threshold() -> None:
    """Round-2 #1383: ``instant_fail_seconds=0`` does NOT disable
    classification of a missing-``steps`` key. The zero-step signal is
    independent of the (now reserved) timing threshold -- a FAILURE job
    with no step data never started the work regardless of how long it
    ran, mirroring the pre-#1383 ``is_infrastructure_failure``. The prior
    code returned False here because the missing-key case was gated on
    ``instant_fail_seconds > 0`` under Signal 3."""
    cfg = InfraBlockedConfig(instant_fail_seconds=0)
    job = {
        "conclusion": "FAILURE",
        "started_at": "2026-08-21T10:00:00Z",
        "completed_at": "2026-08-21T10:00:03Z",
    }
    assert is_infra_blocked_check(job, [], cfg) is True
