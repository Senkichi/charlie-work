"""Regression coverage for issue #1254 and its regression, issue #1399.

Issue #1254: concurrent Tests jobs on one self-hosted host thrash past the
30m timeout, producing false reds. PR #1388 tried to bound that with a
job-level ``concurrency`` group on the Tests job (constant group key,
``cancel-in-progress: false``) on the premise that excess jobs would queue.

Issue #1399: they do not queue. GitHub keeps at most ONE pending job per
concurrency group and CANCELS the older pending job whenever a newer one
enters the same group, regardless of ``cancel-in-progress`` (that flag only
protects the RUNNING job). With a constant group key shared by every branch,
each new CI run cancelled whichever Tests job was still queued -- ~10 PR runs
killed in two hours on 2026-08-22 with ``runner_name=""`` / ``steps=[]``, and
three approved PRs stranded unmergeable. job-cannon's ``ci.yml`` header
documents the same trap and the correct primitive (a runner LABEL as the
capacity semaphore, scheduler-enforced, queues instead of cancelling).

What survives from #1388 is the ``timeout-minutes`` raise. Issue #1434
raised it further from 45 to 75 (the ``suite`` runner-label semaphore
#1404 bounds cw Tests to 2 concurrent, but the host also carries
job-cannon's two ``suite`` legs plus dispatch-tree pytest sessions; a
suite that runs 7:38 solo at -n 2 took 43.7 min with three siblings, and
45m left only ~12% headroom -- overnight 2026-08-23/24, 9 of 16 completed
Tests runs were cap-killed at 45m). These tests guard both facts by
parsing the workflow YAML statically:

1. The Tests job has NO job-level ``concurrency`` block (any group key, any
   ``cancel-in-progress`` value -- the pending-cancel semantics apply to all
   of them).
2. ``timeout-minutes`` stays >= 75.
"""

from __future__ import annotations

from pathlib import Path

import yaml

CI_YML = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def _load_ci_workflow() -> dict:
    """Load and return the parsed ci.yml workflow, asserting it exists."""
    assert CI_YML.exists(), f"ci.yml not found at {CI_YML}"
    return yaml.safe_load(CI_YML.read_text(encoding="utf-8"))


def _tests_job() -> dict:
    workflow = _load_ci_workflow()
    tests_job = workflow.get("jobs", {}).get("Tests")
    assert tests_job is not None, "ci.yml has no 'Tests' job"
    return tests_job


def test_tests_job_has_no_job_level_concurrency_group() -> None:
    """The Tests job must NOT carry a job-level ``concurrency`` block.

    A job-level group with a constant key is not a queue: GitHub cancels the
    older pending job in the group when a newer one arrives, regardless of
    ``cancel-in-progress`` (issue #1399). A per-branch key would be
    redundant with the workflow-level block. Either form is wrong here; the
    cross-branch capacity bound #1254 wants belongs in runner labels.
    """
    tests_job = _tests_job()
    concurrency = tests_job.get("concurrency")
    assert concurrency is None, (
        f"ci.yml Tests job carries a job-level concurrency block {concurrency!r} "
        "-- GitHub cancels the older PENDING job in a concurrency group whenever "
        "a newer one enters it, regardless of cancel-in-progress, so a shared "
        "group kills queued Tests jobs cross-branch (issue #1399). Bound "
        "concurrent Tests jobs with a runner label, not a concurrency group."
    )


def test_workflow_level_concurrency_is_per_ref() -> None:
    """The workflow-level concurrency group must stay keyed on the ref.

    ``${{ github.workflow }}-${{ github.ref }}`` is per-PR (``refs/pull/N/
    merge``), so a newer run on the SAME branch supersedes the older one but
    runs on different branches never share a group. A constant key here would
    reintroduce #1399 one level up.
    """
    workflow = _load_ci_workflow()
    concurrency = workflow.get("concurrency")
    assert concurrency is not None, "ci.yml has no workflow-level concurrency block"
    group = concurrency.get("group", "")
    assert "${{ github.ref }}" in group, (
        f"ci.yml workflow-level concurrency group is {group!r}; it must include "
        "${{ github.ref }} so runs on different branches never share a group "
        "(issue #1399)"
    )


def test_tests_timeout_minutes_raised_from_30() -> None:
    """The Tests job ``timeout-minutes`` must be at least 75.

    30 min was the value that produced the #1254 false reds: under contention
    the suite stretched to 29:43 and the 30m job timeout killed it at the
    finish line. 45 min (#1254 raise) cleared the then-observed contended
    runtime (30m) by ~1.5x but was itself overtaken: the ``suite`` runner-label
    semaphore (#1404) bounds cw Tests to 2 concurrent, yet the host also
    carries job-cannon's two ``suite`` legs plus dispatch-tree pytest
    sessions, so per-repo semaphores cannot see each other and a suite that
    runs 7:38 solo at -n 2 took 43.7 min with three siblings. 45m left only
    ~12% headroom and overnight 2026-08-23/24, 9 of 16 completed Tests runs
    were cap-killed at 45m (#1434). 75 min clears the worst measured
    contended runtime (43.7m) by ~70% while still catching genuine hangs.
    """
    tests_job = _tests_job()
    timeout = tests_job.get("timeout-minutes")
    assert timeout is not None, "ci.yml Tests job has no timeout-minutes"
    assert timeout >= 75, (
        f"ci.yml Tests job timeout-minutes is {timeout}, expected >= 75 -- "
        "45 produced false reds under sustained cross-repo contention "
        "(issue #1434; 30 produced false reds under contention, issue #1254)"
    )
