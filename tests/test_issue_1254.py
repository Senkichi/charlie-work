"""Regression coverage for issue #1254.

Issue #1254: concurrent Tests jobs on one self-hosted host thrash past the
30m timeout, producing false reds. On 2026-08-15 four PRs' Tests jobs ran
concurrently on the shared self-hosted box alongside other repos' CI and the
dispatch tree's pytest sessions; CPU contention stretched a suite that
normally finishes in ~8m (-n 2) to 29:43, and the 30m job timeout killed each
at the finish line -- the suites PASSED but the jobs were cancelled during
cleanup, producing false reds. A second symptom (issue comment) showed a
mid-run xdist-worker death under the same CPU saturation.

The fix has two complementary parts, both in ``.github/workflows/ci.yml``:

1. A **job-level ``concurrency`` group** on the Tests job with a constant
   group key and ``cancel-in-progress: false``. This serializes all Tests
   jobs across every branch so at most one runs on the shared box at a time
   -- the structural concurrency bound the issue asks for instead of operator
   discipline. Excess jobs queue (not cancel) so PR check coverage is never
   silently dropped.

2. **``timeout-minutes`` raised from 30 to 45.** The job-level concurrency
   group eliminates the primary contention source (concurrent Tests jobs),
   but residual contention from sibling repos' CI and the dispatch tree
   persists, so the timeout must still survive it. 45 min is ~1.9x the
   uncontended max (23.72m) and ~1.5x the observed contended runtime (30m).

These tests guard both parts by parsing the workflow YAML statically.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CI_YML = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def _load_ci_workflow() -> dict:
    """Load and return the parsed ci.yml workflow, asserting it exists."""
    assert CI_YML.exists(), f"ci.yml not found at {CI_YML}"
    return yaml.safe_load(CI_YML.read_text(encoding="utf-8"))


def test_tests_job_has_concurrency_group() -> None:
    """The Tests job must have a job-level ``concurrency`` block with a
    constant (non-branch-specific) group key.

    A constant group key (not ``${{ github.ref }}``) is what makes this
    serialize *cross-branch* -- the dimension that causes oversubscription.
    The workflow-level concurrency block already serializes per-branch; the
    job-level block is the additional cross-branch bound.
    """
    workflow = _load_ci_workflow()
    tests_job = workflow.get("jobs", {}).get("Tests")
    assert tests_job is not None, "ci.yml has no 'Tests' job"

    concurrency = tests_job.get("concurrency")
    assert concurrency is not None, (
        "ci.yml Tests job has no 'concurrency' block -- without it, multiple "
        "Tests jobs run concurrently on the shared self-hosted box and thrash "
        "past the timeout (issue #1254)"
    )

    group = concurrency.get("group")
    assert group is not None, "ci.yml Tests job concurrency has no 'group' key"
    # The group must be a constant string, not a per-branch expression.
    # If it contained ${{ github.ref }} it would only serialize per-branch,
    # which the workflow-level block already does -- that is not the fix.
    assert "${{ github.ref }}" not in group, (
        f"ci.yml Tests job concurrency group is {group!r} -- it must NOT "
        "contain ${{ github.ref }}; a per-branch group only serializes "
        "within a branch (already done by the workflow-level block) and "
        "does not bound cross-branch concurrency (issue #1254)"
    )
    assert "${{ github.head_ref }}" not in group, (
        f"ci.yml Tests job concurrency group is {group!r} -- it must NOT "
        "contain ${{ github.head_ref }}; same reasoning as github.ref"
    )


def test_tests_concurrency_cancel_in_progress_is_false() -> None:
    """The Tests job concurrency must set ``cancel-in-progress: false``.

    With ``true``, a newer Tests job would cancel a queued (not yet started)
    one, silently dropping that PR's check coverage. With ``false``, excess
    jobs queue and run when the current one finishes -- the structural bound
    without losing coverage.
    """
    workflow = _load_ci_workflow()
    tests_job = workflow.get("jobs", {}).get("Tests")
    assert tests_job is not None

    concurrency = tests_job.get("concurrency")
    assert concurrency is not None, "Tests job has no concurrency block"

    cancel = concurrency.get("cancel-in-progress")
    assert cancel is False, (
        f"ci.yml Tests job concurrency cancel-in-progress is {cancel!r}, "
        "expected false -- true would cancel queued Tests jobs and silently "
        "drop PR check coverage (issue #1254)"
    )


def test_tests_timeout_minutes_raised_from_30() -> None:
    """The Tests job ``timeout-minutes`` must be at least 45.

    30 min was the value that produced the #1254 false reds: under contention
    the suite stretched to 29:43 and the 30m job timeout killed it at the
    finish line. The job-level concurrency group eliminates the primary
    contention source, but residual contention from sibling repos and the
    dispatch tree persists, so the timeout must still survive it. 45 min is
    ~1.9x the uncontended max (23.72m) and ~1.5x the observed contended
    runtime (30m).
    """
    workflow = _load_ci_workflow()
    tests_job = workflow.get("jobs", {}).get("Tests")
    assert tests_job is not None

    timeout = tests_job.get("timeout-minutes")
    assert timeout is not None, "ci.yml Tests job has no timeout-minutes"
    assert timeout >= 45, (
        f"ci.yml Tests job timeout-minutes is {timeout}, expected >= 45 -- "
        "30 produced false reds under contention (issue #1254); the "
        "concurrency group bounds the primary contention source but residual "
        "contention from sibling repos and the dispatch tree persists"
    )


@pytest.mark.parametrize("job_name", ["Tests"])
def test_concurrency_is_job_level_not_workflow_level(job_name: str) -> None:
    """The concurrency group must be on the Tests *job*, not just the
    workflow level.

    The workflow-level concurrency block (``${{ github.workflow }}-${{
    github.ref }}``) serializes per-branch only. The job-level block is the
    additional cross-branch bound that prevents oversubscription. This test
    confirms the block is present on the job itself.
    """
    workflow = _load_ci_workflow()
    job = workflow.get("jobs", {}).get(job_name)
    assert job is not None
    assert "concurrency" in job, (
        f"ci.yml {job_name} job has no job-level 'concurrency' -- the "
        "workflow-level block only serializes per-branch, not cross-branch "
        "(issue #1254)"
    )
