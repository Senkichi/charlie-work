"""Janitor gate infra-failure rerun mechanics: first-cancel rerun, attempt cap, API-error fallthrough, and sibling-running refusal.

Split out of ``tests/test_charlie_work.py`` (issue #1553,
Track-1 wave 7/8).
"""

from __future__ import annotations

import json
from pathlib import Path

from _review_fixtures import _required_checks_config
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github import FakeGitHubWithRerunCapture


def test_janitor_required_check_rerun_api_error_falls_through_to_rework(tmp_path: Path) -> None:
    """Issue #391: a rerun API error surfaces as an event and falls through to rework."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    link = "https://github.com/owner/repo/actions/runs/12345/job/67890"
    fake_gh = FakeGitHubWithRerunCapture(
        checks=[
            {"name": "Tests passed", "state": "FAILURE", "link": link},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ],
        rerun_ok=False,
    )
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["decision"] == "request_changes"
    # The rerun attempt was not persisted because the API call failed.
    assert "check_rerun_attempts" not in state["prs"]["456"]
    assert any(event["kind"] == "flake_rerun_failed" for event in state.get("events", []))


def test_janitor_required_check_rerun_refused_while_sibling_running_defers_to_next_pass(
    tmp_path: Path,
) -> None:
    """Issue #992: a flake rerun refused because the containing workflow run is
    still in progress must defer to the next pass, not fall through to a
    request_changes verdict. The rerun attempt must not be consumed so a later
    pass can retry once the run has completed."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    link = "https://github.com/owner/repo/actions/runs/12345/job/67890"
    fixture = Path(__file__).parent / "fixtures" / "gh_run_rerun_already_running.json"
    rerun_error = json.loads(fixture.read_text(encoding="utf-8"))["error"]
    fake_gh = FakeGitHubWithRerunCapture(
        checks=[
            {"name": "Tests passed", "state": "FAILURE", "link": link},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ],
        rerun_ok=False,
        rerun_error=rerun_error,
    )
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    assert result.data.get("already_running") is True
    assert result.data.get("rerun_run_ids") == [12345]
    assert fake_gh.rerun_calls == [["run", "rerun", "12345", "--failed"]]
    state = load_state(paths.state_file)
    # The rerun attempt was not persisted and the PR was not routed to rework.
    assert "check_rerun_attempts" not in state["prs"].get("456", {})
    assert "decision" not in state["prs"].get("456", {})
    assert "request_changes_count" not in state["prs"].get("456", {})
    assert state.get("issues", {}).get("123", {}).get("status") != "rework_requested"
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added
    assert any(event["kind"] == "flake_rerun_failed" for event in state.get("events", []))


# Infra-failed (CANCELLED/INFRA_FAILURE/TIMED_OUT) required-check auto-rerun +
# escalation (issue #841). Before this, classify_check_failures only ever
# iterated summary.failed (a code push can't fix an infra kill), so a PR
# blocked on infra_failed sat there forever behind only a diagnostic
# merge_failed_attempt_alarm event -- no rerun, no rework, no escalation.


def test_janitor_infra_failed_first_cancel_triggers_rerun_without_failed_flag(
    tmp_path: Path,
) -> None:
    """Criterion 1: a CANCELLED required check whose run is the current head
    becomes eligible for exactly one auto-rerun. Also asserts the rerun is
    dispatched WITHOUT --failed: the job never completed (cancelled, not
    failed), so --failed's "rerun the failed jobs in this run" semantics do
    not apply -- verified against classify_check_failures' own --failed call,
    which this must NOT copy verbatim."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    link = "https://github.com/owner/repo/actions/runs/12345/job/67890"
    fake_gh = FakeGitHubWithRerunCapture(
        checks=[
            {"name": "Tests passed", "state": "CANCELLED", "link": link},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    assert result.data.get("infra_rerun_run_ids") == [12345]
    assert len(fake_gh.rerun_calls) == 1
    assert fake_gh.rerun_calls[0] == ["run", "rerun", "12345"]
    assert "--failed" not in fake_gh.rerun_calls[0]
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["infra_rerun_attempts"] == {
        "sha-abc123": {"Tests passed": {"12345": 1}}
    }
    assert any(event["kind"] == "infra_rerun_triggered" for event in state.get("events", []))
    # Not escalated, not routed to rework -- this pass only triggered the rerun.
    assert (123, config.labels.human_needed) not in fake_gh.labels_added
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added
    assert "123" not in state.get("issues", {})


def test_janitor_infra_failed_cap_exhausted_escalates_to_operator_queue(tmp_path: Path) -> None:
    """Criterion 2: once the infra rerun attempt cap (default 2) is exhausted,
    there is no code-fix rework path -- the PR must escalate instead of
    looping forever. Issue #1266: infra_rerun_cap_exceeded is a mechanical
    reason (a process-attempt-cap limit, not a judgment call), so it lands
    agent:operator-queue, not agent:human-needed."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    link = "https://github.com/owner/repo/actions/runs/12345/job/67890"
    fake_gh = FakeGitHubWithRerunCapture(
        checks=[
            {"name": "Tests passed", "state": "CANCELLED", "link": link},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Pass 1: first cancel -> rerun attempt 1.
    result1 = app.review(456)
    assert result1.ok is False
    assert result1.data.get("infra_rerun_run_ids") == [12345]

    # Pass 2: STILL cancelled (the rerun itself timed out again, same run id,
    # per verified production behavior of `gh run rerun` reusing run ids) ->
    # rerun attempt 2 (at the default cap).
    result2 = app.review(456)
    assert result2.ok is False
    assert result2.data.get("infra_rerun_run_ids") == [12345]
    assert len(fake_gh.rerun_calls) == 2

    # Pass 3: STILL cancelled, cap (2) now exhausted -> escalate, not a third rerun.
    result3 = app.review(456)
    assert result3.ok is False
    assert result3.data.get("infra_escalated") is True
    assert len(fake_gh.rerun_calls) == 2  # no third rerun dispatched
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "infra_rerun_cap_exceeded"
    assert state["issues"]["123"]["reason_class"] == "mechanical"
    assert state["prs"]["456"]["status"] == "escalated"
    assert (123, config.labels.operator_queue) in fake_gh.labels_added
    assert any(event["kind"] == "infra_rerun_escalated" for event in state.get("events", []))


def test_janitor_infra_failed_rerun_api_error_does_not_consume_attempt_or_escalate(
    tmp_path: Path,
) -> None:
    """A `gh run rerun` API error must not silently consume the attempt (the
    next pass gets a fresh try) and must not escalate -- mirrors the genuine-
    failure flake-rerun error handling, minus the rework fallback (infra has
    none)."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    link = "https://github.com/owner/repo/actions/runs/12345/job/67890"
    fake_gh = FakeGitHubWithRerunCapture(
        checks=[
            {"name": "Tests passed", "state": "CANCELLED", "link": link},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ],
        rerun_ok=False,
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    state = load_state(paths.state_file)
    # The attempt was not persisted because the API call failed.
    assert "infra_rerun_attempts" not in state["prs"].get("456", {})
    assert any(event["kind"] == "infra_rerun_failed" for event in state.get("events", []))
    # Not escalated -- an API error is not a genuine cap exhaustion.
    assert "123" not in state.get("issues", {})
    assert (123, config.labels.human_needed) not in fake_gh.labels_added


def test_janitor_mixed_genuine_failure_and_infra_failure_routes_to_rework_not_infra_rerun(
    tmp_path: Path,
) -> None:
    """Criterion 3: a genuine code FAILURE co-occurring with an infra-failed
    check must still route to rework -- the infra-rerun/escalation wiring
    must not shadow or double-dispatch. is_check_failure_block and
    is_infra_failure_block are both False here (mirrors the pre-existing
    is_draft-co-occurring precedent for is_check_failure_block), so this pass
    triggers neither remediation and falls through to the plain
    janitor_blocked path reporting both failures."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithRerunCapture(
        checks=[
            {"name": "Tests passed", "state": "FAILURE"},
            {"name": "Lint & Format", "state": "CANCELLED"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    assert len(fake_gh.rerun_calls) == 0
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "janitor_blocked"
    failures = state["prs"]["456"]["janitor_failures"]
    assert any("Tests passed" in f for f in failures)
    assert any("infrastructure" in f.lower() and "Lint & Format" in f for f in failures)
    assert "123" not in state.get("issues", {})


# Issue #1936: an infra rerun refused "workflow already running" is a
# retry-later race, not a terminal failure -- the containing run is still
# in progress and will land a terminal state on its own. Previously the
# refusal fell through to the blocked bookkeeping every pass while the
# run stayed alive (API + event churn, merge_ready-side failed-attempt
# counting), and if the pass was parked before the run finished the
# follow-up never fired at all. The driver now persists the refused run
# id under ``infra_rerun_deferred``, probes the containing run's status
# instead of re-calling a guaranteed refusal, and fires exactly one
# follow-up rerun the moment the run reads terminal.

_CANCELLED_CHECKS = [
    {
        "name": "Tests passed",
        "state": "CANCELLED",
        "link": "https://github.com/owner/repo/actions/runs/12345/job/67890",
    },
    {"name": "Lint & Format", "bucket": "pass"},
    {"name": "Pre-commit", "state": "SUCCESS"},
]


def _already_running_error() -> str:
    fixture = Path(__file__).parent / "fixtures" / "gh_run_rerun_already_running.json"
    return json.loads(fixture.read_text(encoding="utf-8"))["error"]


def test_janitor_infra_rerun_refused_already_running_defers_then_follows_up_on_terminal(
    tmp_path: Path,
) -> None:
    """Issue #1936 AC: an "already running" refusal parks the run id in
    ``infra_rerun_deferred`` without consuming the attempt or parking the
    PR; a still-in-progress probe defers quietly (no doomed ``gh run
    rerun`` call, no event churn), and the first terminal probe fires
    exactly one follow-up rerun that consumes the bounded attempt."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithRerunCapture(
        checks=list(_CANCELLED_CHECKS),
        rerun_ok=False,
        rerun_error=_already_running_error(),
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Pass 1: the rerun collides with the in-progress run -> refused.
    result = app.review(456)

    assert result.ok is False
    assert result.data.get("already_running") is True
    assert result.data.get("infra_rerun_run_ids") == [12345]
    assert result.data.get("infra_rerun_deferred_run_ids") == [12345]
    assert fake_gh.rerun_calls == [["run", "rerun", "12345"]]
    # No status probe until a deferral exists -- the common path is unchanged.
    assert fake_gh.workflow_runs_calls == []
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["infra_rerun_deferred"] == {"sha-abc123": [12345]}
    assert "infra_rerun_attempts" not in state["prs"]["456"]
    infra_failed_events = [e for e in state.get("events", []) if e["kind"] == "infra_rerun_failed"]
    assert len(infra_failed_events) == 1
    assert infra_failed_events[0]["payload"]["deferred_run_ids"] == [12345]
    # A wait-on-CI pass is not a blocked park, rework, or escalation.
    assert "decision" not in state["prs"]["456"]
    assert state["prs"]["456"].get("status") != "janitor_blocked"
    assert state.get("issues", {}).get("123", {}).get("status") != "escalated"
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added

    # Pass 2: run still in progress -> the probe defers again without
    # re-calling `gh run rerun` (a guaranteed refusal) or re-emitting the
    # event.
    fake_gh.workflow_runs = [{"id": 12345, "status": "in_progress", "conclusion": None}]
    result = app.review(456)

    assert result.ok is False
    assert result.data.get("already_running") is True
    assert result.data.get("infra_rerun_deferred_run_ids") == [12345]
    assert len(fake_gh.rerun_calls) == 1
    assert fake_gh.workflow_runs_calls == ["sha-abc123"]
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["infra_rerun_deferred"] == {"sha-abc123": [12345]}
    assert len([e for e in state.get("events", []) if e["kind"] == "infra_rerun_failed"]) == 1

    # Pass 3: the run lands a terminal state -> exactly one follow-up rerun.
    fake_gh.workflow_runs = [{"id": 12345, "status": "completed", "conclusion": "cancelled"}]
    fake_gh.rerun_ok = True
    result = app.review(456)

    assert result.ok is False
    assert result.data.get("infra_rerun_run_ids") == [12345]
    assert fake_gh.rerun_calls == [
        ["run", "rerun", "12345"],
        ["run", "rerun", "12345"],
    ]
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["infra_rerun_attempts"] == {
        "sha-abc123": {"Tests passed": {"12345": 1}}
    }
    assert state["prs"]["456"]["infra_rerun_deferred"] == {}
    assert len([e for e in state.get("events", []) if e["kind"] == "infra_rerun_triggered"]) == 1


def test_janitor_infra_rerun_deferred_follow_up_stays_under_attempt_cap(
    tmp_path: Path,
) -> None:
    """Issue #1936: the deferred follow-up consumes the same bounded attempt
    budget as a fresh rerun -- the refusal does not hand out a free retry,
    and once ``infra_rerun_attempt_cap`` is spent the check still escalates
    to the operator queue."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithRerunCapture(
        checks=list(_CANCELLED_CHECKS),
        rerun_ok=False,
        rerun_error=_already_running_error(),
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Pass 1: refused -> deferred, attempt NOT consumed.
    app.review(456)
    # Pass 2: run terminal -> follow-up succeeds -> attempt 1 consumed.
    fake_gh.workflow_runs = [{"id": 12345, "status": "completed", "conclusion": "cancelled"}]
    fake_gh.rerun_ok = True
    app.review(456)
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["infra_rerun_attempts"] == {
        "sha-abc123": {"Tests passed": {"12345": 1}}
    }

    # Pass 3: check still cancelled -> second (capped) attempt, dispatched
    # directly -- the deferred marker is gone so no probe runs.
    app.review(456)
    assert fake_gh.rerun_calls == [["run", "rerun", "12345"]] * 3
    assert len(fake_gh.workflow_runs_calls) == 1
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["infra_rerun_attempts"] == {
        "sha-abc123": {"Tests passed": {"12345": 2}}
    }

    # Pass 4: cap exhausted -> escalate, not a fourth rerun.
    result = app.review(456)
    assert result.ok is False
    assert result.data.get("infra_escalated") is True
    assert len(fake_gh.rerun_calls) == 3
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "infra_rerun_cap_exceeded"
    assert (123, config.labels.operator_queue) in fake_gh.labels_added


def test_janitor_infra_rerun_deferred_genuine_error_clears_deferral(
    tmp_path: Path,
) -> None:
    """Once the containing run reads terminal the deferral is over -- a
    follow-up that then fails for a REAL reason takes the ordinary
    API-error path (attempt unconsumed, marker cleared, ``infra_rerun_
    failed`` event) and the next pass retries directly, no probe."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithRerunCapture(
        checks=list(_CANCELLED_CHECKS),
        rerun_ok=False,
        rerun_error=_already_running_error(),
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Pass 1: refused -> deferred.
    app.review(456)

    # Pass 2: run terminal, but the follow-up hits a genuine refusal.
    fake_gh.workflow_runs = [{"id": 12345, "status": "completed", "conclusion": "cancelled"}]
    fake_gh.rerun_error = "This workflow run cannot be retried"
    app.review(456)

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["infra_rerun_deferred"] == {}
    assert "infra_rerun_attempts" not in state["prs"]["456"]
    assert len([e for e in state.get("events", []) if e["kind"] == "infra_rerun_failed"]) == 2
    assert len(fake_gh.rerun_calls) == 2

    # Pass 3: back to the ordinary path -- no probe, direct rerun succeeds.
    fake_gh.rerun_ok = True
    app.review(456)
    assert len(fake_gh.workflow_runs_calls) == 1
    assert fake_gh.rerun_calls == [["run", "rerun", "12345"]] * 3
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["infra_rerun_attempts"] == {
        "sha-abc123": {"Tests passed": {"12345": 1}}
    }


def test_janitor_infra_rerun_deferred_probe_failure_fails_open_to_rerun(
    tmp_path: Path,
) -> None:
    """A failed ``workflow_runs_for_head`` probe (returns None) cannot leave
    the follow-up permanently deferred -- the driver fails open to the
    ``gh run rerun`` call itself, which re-defers on a live refusal."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithRerunCapture(
        checks=list(_CANCELLED_CHECKS),
        rerun_ok=False,
        rerun_error=_already_running_error(),
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Pass 1: refused -> deferred.
    app.review(456)
    # Pass 2: probe returns None (unconfigured) -> fail open -> the rerun
    # call is refused again -> stays deferred.
    app.review(456)

    assert len(fake_gh.rerun_calls) == 2
    assert len(fake_gh.workflow_runs_calls) == 1
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["infra_rerun_deferred"] == {"sha-abc123": [12345]}
