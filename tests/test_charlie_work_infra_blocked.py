"""Infra-blocked check routing: CI-infrastructure failures must not burn rework.

Split out of ``tests/test_charlie_work.py`` (issue #1547, Track-1 wave 1/8):
the ``test_infra_blocked_*`` seam -- budget/runner-outage check conclusions
route around the rework counter, the one-escalation-per-window persistence
rule, window reset semantics, the check_infra_blocked event, and config
parsing. Shared fakes and helpers in
``tests/_rework_dispatch_fixtures.py``.
"""

from __future__ import annotations

from datetime import (
    UTC,
    datetime,
)
from pathlib import Path
from _fakes_github import FakeGitHub
from _review_fixtures import _required_checks_config
from _rework_dispatch_fixtures import (
    _FakeGitHubWithInfraBlockedJob,
)
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp


def test_infra_blocked_budget_failure_no_rework(tmp_path: Path) -> None:
    """AC1: a simulated check run failing in under 10 seconds with zero steps
    and a budget annotation is classified ``infra_blocked`` and dispatches no
    rework."""
    from charlie_work.workflow import _infra_blocked_window

    _infra_blocked_window.clear()
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = _FakeGitHubWithInfraBlockedJob(
        checks=[
            {"name": "Tests passed", "state": "FAILURE", "databaseId": 9001},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ],
        annotations_by_check_run_id={
            9001: [
                {
                    "message": "The job was not started because your spending limit needs to be increased."
                }
            ],
        },
        jobs_by_check_run_id={
            9001: {"conclusion": "FAILURE", "steps": []},
        },
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    # AC1: no rework dispatched
    assert result.ok is False
    assert result.data.get("infra_blocked") is True
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added
    # The issue is NOT transitioned to in_progress (no rework)
    assert (123, config.labels.in_progress) not in fake_gh.labels_added


def test_infra_blocked_no_rework_counter_incremented(tmp_path: Path) -> None:
    """AC2: attempt counters remain unchanged after an infra_blocked pass."""
    from charlie_work.workflow import _infra_blocked_window

    _infra_blocked_window.clear()
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = _FakeGitHubWithInfraBlockedJob(
        checks=[
            {"name": "Tests passed", "state": "FAILURE", "databaseId": 9001},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ],
        jobs_by_check_run_id={
            9001: {"conclusion": "FAILURE", "steps": []},
        },
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.review(456)

    state = load_state(paths.state_file)
    pr_state = state.get("prs", {}).get("456", {})
    # No rework attempts recorded for the infra_blocked PR
    assert pr_state.get("no_op_rework_attempts", 0) == 0
    assert pr_state.get("check_rerun_attempts", {}).get("Tests passed", 0) == 0


def test_infra_blocked_then_genuine_failure_routes_to_rework(tmp_path: Path) -> None:
    """AC2: a later genuine test failure on the same PR still routes to rework normally."""
    from charlie_work.workflow import _infra_blocked_window

    _infra_blocked_window.clear()
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Pass 1: infra_blocked (zero-step FAILURE)
    fake_gh_infra = _FakeGitHubWithInfraBlockedJob(
        checks=[
            {"name": "Tests passed", "state": "FAILURE", "databaseId": 9001},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ],
        jobs_by_check_run_id={
            9001: {"conclusion": "FAILURE", "steps": []},
        },
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh_infra)
    result1 = app.review(456)
    assert result1.data.get("infra_blocked") is True

    # Pass 2: genuine test failure (real test step failed)
    fake_gh_real = _FakeGitHubWithInfraBlockedJob(
        checks=[
            {"name": "Tests passed", "state": "FAILURE", "databaseId": 9002},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ],
        jobs_by_check_run_id={
            9002: {
                "conclusion": "FAILURE",
                "steps": [
                    {"name": "Set up job", "conclusion": "SUCCESS"},
                    {"name": "Run tests", "conclusion": "FAILURE"},
                ],
            },
        },
    )
    fake_gh_real.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app2 = OrchestratorApp(tmp_path, paths, config, fake_gh_real)
    result2 = app2.review(456)

    # AC2: genuine failure routes to rework normally
    assert result2.ok is True  # review packet generated (rework dispatched)
    assert result2.data.get("infra_blocked") is not True


def test_infra_blocked_persistence_one_escalation_per_window(tmp_path: Path) -> None:
    """AC3: persistence across N passes emits exactly one operator escalation
    event per window, not one event per PR per pass."""
    from charlie_work.config import InfraBlockedConfig
    from charlie_work.workflow import _infra_blocked_window

    _infra_blocked_window.clear()  # reset cross-pass state from other tests

    config = _required_checks_config(
        infra_blocked=InfraBlockedConfig(persistence_passes=2, escalation_window_minutes=60),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    def make_infra_blocked_fake() -> _FakeGitHubWithInfraBlockedJob:
        return _FakeGitHubWithInfraBlockedJob(
            checks=[
                {"name": "Tests passed", "state": "FAILURE", "databaseId": 9001},
                {"name": "Lint & Format", "bucket": "pass"},
                {"name": "Pre-commit", "state": "SUCCESS"},
            ],
            jobs_by_check_run_id={
                9001: {"conclusion": "FAILURE", "steps": []},
            },
        )

    # Pass 1: infra_blocked, but persistence_passes=2 so no escalation yet
    app1 = OrchestratorApp(tmp_path, paths, config, make_infra_blocked_fake())
    app1.review(456)
    escalated_1 = query_events(paths.state_file, kind="infra_blocked_escalated")
    assert len(escalated_1) == 0

    # Pass 2: persistence threshold reached -> one escalation
    app2 = OrchestratorApp(tmp_path, paths, config, make_infra_blocked_fake())
    app2.review(456)
    escalated_2 = query_events(paths.state_file, kind="infra_blocked_escalated")
    assert len(escalated_2) == 1

    # Pass 3: same window -> no additional escalation
    app3 = OrchestratorApp(tmp_path, paths, config, make_infra_blocked_fake())
    app3.review(456)
    escalated_3 = query_events(paths.state_file, kind="infra_blocked_escalated")
    assert len(escalated_3) == 1  # still exactly one


def test_infra_blocked_multiple_prs_one_pass_no_premature_escalation(
    tmp_path: Path,
) -> None:
    """AC3 (rework finding): multiple infra-blocked PRs encountered within a
    single loop pass must increment ``consecutive_passes`` at most ONCE for
    that pass, not once per PR. Without the correlation_id gate on the
    increment, N concurrent infra-blocked PRs in one pass reach
    ``persistence_passes=N`` and fire ``infra_blocked_escalated`` within
    that single pass -- contradicting the documented "persistence across N
    passes, not per PR per pass" design.

    This test simulates a single pass (one ``correlation_context``) that
    encounters three infra-blocked PRs by calling ``review()`` three times
    within that context. With ``persistence_passes=3``, the bug would fire
    escalation on the third call within the same pass; the fix holds the
    counter at 1 for the whole pass, so no escalation fires until two more
    passes observe infra-blocked PRs.
    """
    from charlie_work.config import InfraBlockedConfig
    from charlie_work.instrumentation import correlation_context
    from charlie_work.workflow import _infra_blocked_window

    _infra_blocked_window.clear()

    config = _required_checks_config(
        infra_blocked=InfraBlockedConfig(persistence_passes=3, escalation_window_minutes=60),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    def make_infra_blocked_fake() -> _FakeGitHubWithInfraBlockedJob:
        return _FakeGitHubWithInfraBlockedJob(
            checks=[
                {"name": "Tests passed", "state": "FAILURE", "databaseId": 9001},
                {"name": "Lint & Format", "bucket": "pass"},
                {"name": "Pre-commit", "state": "SUCCESS"},
            ],
            jobs_by_check_run_id={
                9001: {"conclusion": "FAILURE", "steps": []},
            },
        )

    # A single pass: one correlation_context, three infra-blocked review()
    # calls (simulating three PRs encountered in the same pass). The
    # infra_blocked branch is idempotent (no state/label mutation, early
    # return), so repeated calls for the same PR exercise the same counter
    # path as distinct PRs would.
    app = OrchestratorApp(tmp_path, paths, config, make_infra_blocked_fake())
    with correlation_context() as cid:
        app.review(456)
        app.review(456)
        app.review(456)

    # The counter incremented exactly once for this pass, not three times.
    repo_key = str(tmp_path)
    assert _infra_blocked_window[repo_key]["consecutive_passes"] == 1, (
        f"expected one increment for pass {cid!r}, got "
        f"{_infra_blocked_window[repo_key]['consecutive_passes']!r}"
    )
    # persistence_passes=3 not reached within a single pass -> no escalation.
    escalated = query_events(paths.state_file, kind="infra_blocked_escalated")
    assert len(escalated) == 0, escalated

    # A second pass (new correlation_context) increments again -> counter=2,
    # still below threshold -> no escalation.
    app2 = OrchestratorApp(tmp_path, paths, config, make_infra_blocked_fake())
    with correlation_context():
        app2.review(456)
    assert _infra_blocked_window[repo_key]["consecutive_passes"] == 2
    escalated_2 = query_events(paths.state_file, kind="infra_blocked_escalated")
    assert len(escalated_2) == 0

    # A third pass reaches persistence_passes=3 -> exactly one escalation.
    app3 = OrchestratorApp(tmp_path, paths, config, make_infra_blocked_fake())
    with correlation_context():
        app3.review(456)
    assert _infra_blocked_window[repo_key]["consecutive_passes"] == 3
    escalated_3 = query_events(paths.state_file, kind="infra_blocked_escalated")
    assert len(escalated_3) == 1


def test_infra_blocked_window_resets_when_pass_reviews_clear_prs(tmp_path: Path) -> None:
    """Rework finding: the ``_loop_impl`` branch that resets
    ``_infra_blocked_window`` when a pass reviews PRs and finds none
    infra-blocked (the fleet-wide infra condition has cleared). A pass
    that reviews at least one clear PR must zero the consecutive-pass
    counter so the next outage starts fresh, rather than carrying stale
    persistence forward.

    ``FakeGitHub()`` ships a janitor-green PR 456 with passing required
    checks, so ``loop(limit=0)`` runs a full pass (intake, dispatch, the
    review lane, the post-pass reset sweep) that actually reviews PR 456
    and emits no ``check_infra_blocked`` events for this pass's
    correlation_id -- exactly the "we looked and the coast was clear"
    condition that triggers the reset.
    """
    from charlie_work.workflow import _infra_blocked_window

    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Pre-populate the window as if prior passes observed infra-blocked PRs.
    repo_key = str(tmp_path)
    _infra_blocked_window[repo_key] = {
        "consecutive_passes": 2,
        "last_escalation": datetime.now(UTC),
        "last_pass_cid": "stale-cid-from-prior-pass",
    }

    result = app.loop(limit=0)

    # The pass reviewed PR 456 (green) and found no infra-blocked PRs -> the
    # window was reset.
    assert len(result.data["reviews"]) >= 1
    window = _infra_blocked_window[repo_key]
    assert window["consecutive_passes"] == 0
    assert window["last_escalation"] is None
    assert window["last_pass_cid"] is None


def test_infra_blocked_window_not_reset_when_pass_reviews_zero_prs(tmp_path: Path) -> None:
    """Round-2 #1383 regression: a pass that reviews ZERO PRs during a live
    outage must NOT reset ``_infra_blocked_window``. The prior reset logic
    conflated "zero ``check_infra_blocked`` events emitted" with "we looked
    and the coast was clear" -- but a pass that reviews nothing also emits
    zero such events, so an idle pass during a live outage silently cleared
    ``consecutive_passes``/``last_escalation`` and could prevent the AC3
    persistence escalation from ever firing.

    Distinct from ``test_infra_blocked_window_resets_when_pass_reviews_clear_prs``
    (which reviews a green PR and DOES reset): here the PR/issue queue is
    empty so the review lane runs over zero PRs.
    """
    from charlie_work.workflow import _infra_blocked_window

    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Empty queue: no PRs and no issues, so the review lane reviews nothing.
    fake_gh.prs = []
    fake_gh.issues = []
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Pre-populate the window as if prior passes observed infra-blocked PRs
    # (a live outage is in progress).
    repo_key = str(tmp_path)
    prior_last_esc = datetime.now(UTC)
    _infra_blocked_window[repo_key] = {
        "consecutive_passes": 2,
        "last_escalation": prior_last_esc,
        "last_pass_cid": "stale-cid-from-prior-pass",
    }

    result = app.loop(limit=0)

    # The pass reviewed zero PRs -> the window must be preserved, not reset.
    assert len(result.data["reviews"]) == 0
    window = _infra_blocked_window[repo_key]
    assert window["consecutive_passes"] == 2
    assert window["last_escalation"] == prior_last_esc
    assert window["last_pass_cid"] == "stale-cid-from-prior-pass"


def test_infra_blocked_check_infra_blocked_event_emitted(tmp_path: Path) -> None:
    """AC4: the ``check_infra_blocked`` event is emitted and has a consumer
    (heartbeat_check.py's ``check_infra_blocked_events``)."""
    from charlie_work.instrumentation import _LEVEL_BY_KIND
    from charlie_work.workflow import _infra_blocked_window

    _infra_blocked_window.clear()

    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = _FakeGitHubWithInfraBlockedJob(
        checks=[
            {"name": "Tests passed", "state": "FAILURE", "databaseId": 9001},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ],
        jobs_by_check_run_id={
            9001: {"conclusion": "FAILURE", "steps": []},
        },
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.review(456)

    blocked_events = query_events(paths.state_file, kind="check_infra_blocked")
    assert len(blocked_events) == 1
    assert blocked_events[0]["payload"]["pr_number"] == 456
    assert "Tests passed" in blocked_events[0]["payload"]["checks"]
    # The event kind is registered at warning level
    assert _LEVEL_BY_KIND["check_infra_blocked"] == "warning"
    # The escalation kind is registered at error level
    assert _LEVEL_BY_KIND["infra_blocked_escalated"] == "error"


def test_infra_blocked_config_parses_from_yaml(tmp_path: Path) -> None:
    """The infra_blocked config section is parsed from YAML correctly."""
    from charlie_work.config import load_config

    path = tmp_path / "c.yaml"
    path.write_text(
        "auto_merge:\n"
        "  required_checks: [Tests passed]\n"
        "  infra_blocked:\n"
        "    enabled: true\n"
        "    instant_fail_seconds: 15\n"
        "    annotation_patterns: [billing exhausted, quota exceeded]\n"
        "    persistence_passes: 5\n"
        "    escalation_window_minutes: 120\n",
        encoding="utf-8",
    )
    config = load_config(path)
    cfg = config.auto_merge.infra_blocked
    assert cfg.enabled is True
    assert cfg.instant_fail_seconds == 15
    assert cfg.annotation_patterns == ("billing exhausted", "quota exceeded")
    assert cfg.persistence_passes == 5
    assert cfg.escalation_window_minutes == 120


def test_infra_blocked_does_not_shadow_cancelled_auto_rerun_path(tmp_path: Path) -> None:
    """Round-3 review finding: the #1383 ``_enrich_checks_infra_blocked``
    helper gates on ``state == "FAILURE"``, so it must NOT reclassify a
    ``CANCELLED`` (or ``INFRA_FAILURE``/``TIMED_OUT``) required check. The
    pre-existing #841 ``is_infra_failure_block`` auto-rerun+escalate path is
    fed by ``run_janitor`` over those conclusions and must still fire
    post-#1383. This test confirms a ``CANCELLED`` required check -- carrying
    a ``databaseId`` and a zero-step job that WOULD match
    ``is_infra_blocked_check`` if the gate admitted it -- still routes to
    ``infra_rerun`` (not ``infra_blocked``) with the #1383 classifier enabled
    (its default). The ``databaseId`` + zero-step job exercise the gate so a
    mutation that broadens it to ``CANCELLED`` is caught."""

    class _FakeGitHubWithInfraJobAndRerun(_FakeGitHubWithInfraBlockedJob):
        """Combines actions_job (for the enrichment gate) with rerun capture
        (for the #841 auto-rerun assertion)."""

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.rerun_calls: list[list[str]] = []

        def run(self, args: list[str], *, json_output: bool = False, allow_failure: bool = False):  # noqa: ANN202
            if len(args) >= 2 and args[0] == "run" and args[1] == "rerun":
                self.rerun_calls.append(list(args))
                return "DRY-RUN: gh run rerun " + " ".join(args[2:])
            return super().run(args, json_output=json_output, allow_failure=allow_failure)

    from charlie_work.workflow import _infra_blocked_window

    _infra_blocked_window.clear()
    config = _required_checks_config()  # InfraBlockedConfig.enabled defaults to True
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    link = "https://github.com/owner/repo/actions/runs/12345/job/67890"
    fake_gh = _FakeGitHubWithInfraJobAndRerun(
        checks=[
            {"name": "Tests passed", "state": "CANCELLED", "link": link, "databaseId": 9001},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ],
        jobs_by_check_run_id={
            # Zero-step FAILURE job that is_infra_blocked_check WOULD classify
            # -- but the outer gate excludes CANCELLED check-state, so this
            # job is never consulted and the check stays CANCELLED.
            9001: {"conclusion": "FAILURE", "steps": []},
        },
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    # The #841 auto-rerun path fired for the CANCELLED check.
    assert result.ok is False
    assert result.data.get("infra_rerun_run_ids") == [12345]
    assert len(fake_gh.rerun_calls) == 1
    # The #1383 infra_blocked hold path did NOT fire.
    assert result.data.get("infra_blocked") is not True
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added
    # No check_infra_blocked event was emitted -- the CANCELLED check is not
    # in the infra_blocked population.
    blocked_events = query_events(paths.state_file, kind="check_infra_blocked")
    assert len(blocked_events) == 0
