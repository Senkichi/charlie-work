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
