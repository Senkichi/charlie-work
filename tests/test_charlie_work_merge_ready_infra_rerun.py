"""Merge-ready infra-failure auto-rerun + escalation (issue #1912).

The #841 remediation (``classify_infra_failures`` + ``gh run rerun`` +
cap-exhaustion escalation) lived only in ``review()`` -- fed by
``run_janitor`` -- but an approved PR whose verdict carries forward to the
live head never re-enters ``review()``: ``loop()``'s already_approved fast
path routes it straight to ``merge_ready()``. A CANCELLED/TIMED_OUT
required check on that head was therefore retried by nothing and escalated
to nobody: ``merge_ready`` polled, emitted a diagnostic
``merge_failed_attempt_alarm``, and looped forever (live instance: swole
PR #349 / issue #174).

These tests pin the merge-ready side of the fix against the same FakeGitHub
rerun-capture fixture the ``review()``-side tests (test_charlie_work_janitor_infra)
use. The verdict is written as an approved same-head decision -- identical
to the post-carry-forward state ``_update_approval_head`` leaves behind
(``reviewed_head_sha`` == live ``headRefOid``), which is the exact lane the
issue's stuck PR occupied.
"""

from __future__ import annotations

import json
from pathlib import Path

from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github import FakeGitHubWithRerunCapture
from _review_fixtures import _required_checks_config
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state
from charlie_work.workflow import OrchestratorApp

_RUN_LINK = "https://github.com/owner/repo/actions/runs/12345/job/67890"

_CANCELLED_CHECKS = [
    {"name": "Tests passed", "state": "CANCELLED", "link": _RUN_LINK},
    {"name": "Lint & Format", "bucket": "pass"},
    {"name": "Pre-commit", "state": "SUCCESS"},
]


def _merge_ready_app(
    tmp_path: Path,
    fake_gh: FakeGitHubWithRerunCapture,
):
    """Approved-at-live-head PR #456 + configured required checks.

    The recorded decision matches FakeGitHub's default PR head
    (``sha-abc123``), so ``merge_ready`` reaches its check-evaluation block
    through the same code path a carried-forward verdict leaves behind --
    head_moved False, approved True.
    """
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    decision_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )
    return app, config, paths


def _events(paths, kind: str) -> list[dict]:
    return [e for e in load_state(paths.state_file).get("events", []) if e["kind"] == kind]


def test_merge_ready_infra_cancelled_check_triggers_rerun(tmp_path: Path) -> None:
    """AC: a CANCELLED required check on the approved head is retried via
    ``gh run rerun`` (WITHOUT --failed -- the job never completed), the
    attempt is persisted per-head/check/run-id, and
    ``infra_rerun_triggered`` is emitted. Previously this pass only
    incremented the failed-attempt counter and emitted nothing infra."""
    fake_gh = FakeGitHubWithRerunCapture(checks=list(_CANCELLED_CHECKS))
    app, config, paths = _merge_ready_app(tmp_path, fake_gh)

    result = app.merge_ready(456)

    assert result.ok is True
    assert result.data["can_merge"] is False
    assert result.data["merged"] is False
    assert result.data.get("infra_rerun_run_ids") == [12345]
    assert fake_gh.rerun_calls == [["run", "rerun", "12345"]]
    assert "--failed" not in fake_gh.rerun_calls[0]
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["infra_rerun_attempts"] == {
        "sha-abc123": {"Tests passed": {"12345": 1}}
    }
    assert len(_events(paths, "infra_rerun_triggered")) == 1
    # Not escalated, not merged, no rework label -- remediation is in flight.
    assert (123, config.labels.operator_queue) not in fake_gh.labels_added
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added
    assert fake_gh.merged == []
    assert "123" not in state.get("issues", {})


def test_merge_ready_infra_cap_exhausted_escalates_then_stays_terminal(
    tmp_path: Path,
) -> None:
    """AC: after ``infra_rerun_attempt_cap`` (default 2) retries on the same
    head, the check escalates to the operator queue (mechanical) instead of
    looping -- and once escalated, subsequent passes do NOT re-fire the
    rerun or re-emit ``infra_rerun_escalated``."""
    fake_gh = FakeGitHubWithRerunCapture(checks=list(_CANCELLED_CHECKS))
    app, config, paths = _merge_ready_app(tmp_path, fake_gh)

    r1 = app.merge_ready(456)
    assert r1.data.get("infra_rerun_run_ids") == [12345]
    r2 = app.merge_ready(456)
    assert r2.data.get("infra_rerun_run_ids") == [12345]
    assert len(fake_gh.rerun_calls) == 2

    # Pass 3: still cancelled, cap exhausted -> escalate, no third rerun.
    r3 = app.merge_ready(456)
    assert r3.ok is True
    assert r3.data.get("infra_escalated") is True
    assert r3.data["can_merge"] is False
    assert len(fake_gh.rerun_calls) == 2
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "infra_rerun_cap_exceeded"
    assert state["issues"]["123"]["reason_class"] == "mechanical"
    assert state["prs"]["456"]["status"] == "escalated"
    assert (123, config.labels.operator_queue) in fake_gh.labels_added
    assert len(_events(paths, "infra_rerun_escalated")) == 1

    # Pass 4: escalation is terminal -- no re-rerun, no re-escalation event.
    app.merge_ready(456)
    assert len(fake_gh.rerun_calls) == 2
    assert len(_events(paths, "infra_rerun_escalated")) == 1


def test_merge_ready_infra_rerun_api_error_does_not_consume_attempt(tmp_path: Path) -> None:
    """AC: a ``gh run rerun`` API error is recorded as ``infra_rerun_failed``
    without consuming the bounded attempt and without escalating -- the next
    pass gets a fresh try."""
    fake_gh = FakeGitHubWithRerunCapture(checks=list(_CANCELLED_CHECKS), rerun_ok=False)
    app, config, paths = _merge_ready_app(tmp_path, fake_gh)

    result = app.merge_ready(456)

    assert result.ok is True
    assert result.data["can_merge"] is False
    assert len(fake_gh.rerun_calls) == 1
    state = load_state(paths.state_file)
    # The attempt was not persisted because the API call failed.
    assert "infra_rerun_attempts" not in state["prs"].get("456", {})
    assert len(_events(paths, "infra_rerun_failed")) == 1
    assert not _events(paths, "infra_rerun_escalated")
    assert state.get("issues", {}).get("123", {}).get("status") != "escalated"
    assert (123, config.labels.operator_queue) not in fake_gh.labels_added


def test_merge_ready_infra_unparseable_run_id_escalates_immediately(tmp_path: Path) -> None:
    """AC: an infra-failed required check with no parseable Actions run id
    cannot be auto-retried at all -- it is definitive on the first pass and
    escalates rather than looping."""
    checks = [
        {"name": "Tests passed", "state": "CANCELLED", "link": "https://external.ci/run"},
        {"name": "Lint & Format", "bucket": "pass"},
        {"name": "Pre-commit", "state": "SUCCESS"},
    ]
    fake_gh = FakeGitHubWithRerunCapture(checks=checks)
    app, config, paths = _merge_ready_app(tmp_path, fake_gh)

    result = app.merge_ready(456)

    assert result.ok is True
    assert result.data.get("infra_escalated") is True
    assert fake_gh.rerun_calls == []
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert len(_events(paths, "infra_rerun_escalated")) == 1


def test_merge_ready_infra_co_occurring_genuine_failure_does_not_rerun(
    tmp_path: Path,
) -> None:
    """A genuine code FAILURE co-occurring with an infra-failed check is owned
    by the check-failure rework lane, not the infra lane -- mirrors the
    janitor's ``is_infra_failure_block`` sole-blocker requirement (and the
    review()-side pin for the same combination)."""
    checks = [
        {"name": "Tests passed", "state": "FAILURE"},
        {"name": "Lint & Format", "state": "CANCELLED", "link": _RUN_LINK},
        {"name": "Pre-commit", "state": "SUCCESS"},
    ]
    fake_gh = FakeGitHubWithRerunCapture(checks=checks)
    app, _config, paths = _merge_ready_app(tmp_path, fake_gh)

    result = app.merge_ready(456)

    assert result.data["can_merge"] is False
    assert fake_gh.rerun_calls == []
    assert not _events(paths, "infra_rerun_triggered")
    assert not _events(paths, "infra_rerun_escalated")


def test_merge_ready_infra_failed_alongside_pending_check_still_reruns(
    tmp_path: Path,
) -> None:
    """Pending sibling checks do NOT disqualify the infra lane -- mirrors the
    janitor's ``is_infra_failure_block``, which excludes failed/missing/
    unavailable/infra_blocked co-occurrence but not pending."""
    checks = [
        {"name": "Tests passed", "state": "CANCELLED", "link": _RUN_LINK},
        {"name": "Lint & Format", "state": "PENDING"},
        {"name": "Pre-commit", "state": "SUCCESS"},
    ]
    fake_gh = FakeGitHubWithRerunCapture(checks=checks)
    app, _config, paths = _merge_ready_app(tmp_path, fake_gh)

    result = app.merge_ready(456)

    assert fake_gh.rerun_calls == [["run", "rerun", "12345"]]
    assert len(_events(paths, "infra_rerun_triggered")) == 1
    assert result.data.get("infra_rerun_run_ids") == [12345]


def test_merge_ready_infra_blocked_check_does_not_rerun_or_escalate(tmp_path: Path) -> None:
    """``infra_blocked`` (fleet-wide Actions budget/runner outage, #1383) is a
    distinct condition from per-PR ``infra_failed`` -- it must not enter the
    per-run rerun/escalation lane."""

    class _InfraBlockedWithRerun(FakeGitHubWithRerunCapture):
        def __init__(self, checks, jobs):
            super().__init__(checks=checks)
            self._jobs = jobs

        def actions_job(self, job_id):
            return self._jobs.get(job_id)

    fake_gh = _InfraBlockedWithRerun(
        checks=[
            {"name": "Tests passed", "state": "FAILURE", "databaseId": 9001},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ],
        jobs={9001: {"conclusion": "FAILURE", "steps": []}},
    )
    app, _config, paths = _merge_ready_app(tmp_path, fake_gh)

    result = app.merge_ready(456)

    assert result.data["can_merge"] is False
    assert fake_gh.rerun_calls == []
    assert not _events(paths, "infra_rerun_triggered")
    assert not _events(paths, "infra_rerun_failed")
    assert not _events(paths, "infra_rerun_escalated")
    state = load_state(paths.state_file)
    assert state.get("issues", {}).get("123", {}).get("status") != "escalated"


def test_merge_ready_infra_already_escalated_issue_skips_lane(tmp_path: Path) -> None:
    """An already-escalated PR/issue is operator-owned (review()'s own entry
    gate treats escalation as terminal for automated remediation): the lane
    must neither fire ``gh run rerun`` nor re-emit ``infra_rerun_escalated``."""
    fake_gh = FakeGitHubWithRerunCapture(checks=list(_CANCELLED_CHECKS))
    app, config, paths = _merge_ready_app(tmp_path, fake_gh)
    state = load_state(paths.state_file)
    state["issues"]["123"] = {
        "status": "escalated",
        "escalation_reason": "watchdog_redispatch_cap",
        "reason_class": "mechanical",
    }
    state["prs"]["456"] = {"status": "escalated"}
    save_state(paths.state_file, state)

    result = app.merge_ready(456)

    assert fake_gh.rerun_calls == []
    assert not _events(paths, "infra_rerun_triggered")
    assert not _events(paths, "infra_rerun_escalated")
    assert result.data["can_merge"] is False


def test_merge_ready_infra_failed_pass_does_not_double_count_failed_attempts(
    tmp_path: Path,
) -> None:
    """A pass that dispatches an infra rerun is not also a failed merge
    attempt -- remediation is in flight, so the failed-attempt counter stays
    at 0 rather than marching toward ``merge_failed_attempt_alarm`` while
    the rerun heals CI."""
    fake_gh = FakeGitHubWithRerunCapture(checks=list(_CANCELLED_CHECKS))
    app, _config, paths = _merge_ready_app(tmp_path, fake_gh)

    app.merge_ready(456)

    state = load_state(paths.state_file)
    assert state["prs"]["456"].get("consecutive_failed_merge_attempts", 0) == 0
