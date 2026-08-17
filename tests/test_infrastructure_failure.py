"""Tests for is_infrastructure_failure, carved out of test_charlie_work.py (#1284)."""

from __future__ import annotations

from charlie_work.github import is_infrastructure_failure


def test_is_infrastructure_failure_zero_step_job() -> None:
    """Jobs with zero non-setup steps should be classified as infrastructure failure."""
    job = {
        "conclusion": "FAILURE",
        "steps": [
            {"name": "Set up job"},
            {"name": "Checkout"},
        ],
    }
    annotations = []

    assert is_infrastructure_failure(job, annotations) is True


def test_is_infrastructure_failure_with_test_steps() -> None:
    """Jobs with actual test steps should not be classified as infrastructure failure."""
    job = {
        "conclusion": "FAILURE",
        "steps": [
            {"name": "Set up job"},
            {"name": "Checkout"},
            {"name": "Run tests"},
        ],
    }
    annotations = []

    assert is_infrastructure_failure(job, annotations) is False


def test_is_infrastructure_failure_billing_annotation() -> None:
    """Jobs with billing annotation should be classified as infrastructure failure."""
    job = {
        "conclusion": "FAILURE",
        "steps": [{"name": "Run tests"}],
    }
    annotations = [
        {
            "message": "The job was not started because recent account payments have failed or your spending limit needs to be increased."
        }
    ]

    assert is_infrastructure_failure(job, annotations) is True


def test_is_infrastructure_failure_mixed_billing_annotation_text() -> None:
    """Billing annotation detection should be case-insensitive and match partial text."""
    job = {
        "conclusion": "FAILURE",
        "steps": [{"name": "Run tests"}],
    }
    annotations = [{"message": "The job WAS NOT STARTED due to billing issues"}]

    assert is_infrastructure_failure(job, annotations) is True


def test_is_infrastructure_failure_no_infrastructure_signals() -> None:
    """Jobs without infrastructure failure signals should not be classified as such."""
    job = {
        "conclusion": "FAILURE",
        "steps": [
            {"name": "Set up job"},
            {"name": "Checkout"},
            {"name": "Run tests"},
        ],
    }
    annotations = [{"message": "Test failed: assertion error"}]

    assert is_infrastructure_failure(job, annotations) is False


def test_is_infrastructure_failure_empty_steps() -> None:
    """Job with no steps at all should be classified as infrastructure failure (primary signal)."""
    job = {
        "conclusion": "FAILURE",
        "steps": [],
    }
    annotations = []

    assert is_infrastructure_failure(job, annotations) is True


def test_is_infrastructure_failure_non_failed_job() -> None:
    """Jobs that didn't fail should not trigger infrastructure failure detection."""
    job = {
        "conclusion": "SUCCESS",
        "steps": [],
    }
    annotations = []

    assert is_infrastructure_failure(job, annotations) is False
