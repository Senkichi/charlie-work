"""Loop-pass mechanics: errors-bucket surfacing, merge tripwire, foreign-ref parking, intake failure, backpressure, cadence lanes, concurrency governor, sink metric, and preflight gating.

Split out of ``tests/test_charlie_work.py`` (issue #1553,
Track-1 wave 7/8).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from _fakes_github import FakeGitHub
from _merge_tripwire_fixtures import _arm_unauthorized_merge_tripwire
from _review_fixtures import _make_loop_app, _required_checks_config
from charlie_work.config import DevinConfig, DispatchConfig, OrchestratorConfig, WorkerRoleConfig
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state
from charlie_work.workflow import CommandResult, OrchestratorApp
from _dispatch_fixtures import _main_ci_reclaim_app
from _dispatch_fixtures import _reconcile_pass_app
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_loop_surfaces_unauthorized_merge_in_errors_bucket(tmp_path: Path) -> None:
    """loop() must wire the post-merge tripwire into the errors bucket even when dispatch() had no ready issues and returned an empty merged_prs list (issue #502)."""
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)

    class CountingFakeGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.merged_pr_list_calls = 0

        def merged_pr_list(self):
            self.merged_pr_list_calls += 1
            return super().merged_pr_list()

    fake_gh = CountingFakeGitHub()
    # No ready issues and no open PRs — only a merged worker PR the tripwire
    # must catch. dispatch() will return merged_prs=[] (no ready issues), so
    # the tripwire must fall back to fetching its own list to stay armed.
    fake_gh.issues = []
    fake_gh.prs = [
        {
            "number": 501,
            "title": "fix: worker self-merge",
            "url": "https://example.test/pull/501",
            "headRefName": "agent/issue-494-fix",
            "baseRefName": "main",
            "headRefOid": "sha-501",
            "state": "MERGED",
            "isCrossRepository": False,
            "body": "Closes #494",
            "labels": [],
        },
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.loop(merge=False)

    # dispatch() returned an empty merged_prs list because there were no ready
    # issues — the tripwire must NOT treat that as "no merged PRs to check".
    assert result.data["dispatch"].get("merged_prs") == []
    assert len(result.data["errors"]) == 1
    error = result.data["errors"][0]
    assert error["pr"] == 501
    assert error["issue"] == 494
    assert "MERGED" in error["error"]
    assert "possible worker self-merge" in error["error"]
    # The tripwire fetched its own list because the reused list was empty.
    assert fake_gh.merged_pr_list_calls >= 1


def test_loop_tripwire_silent_for_approved_matching_head(tmp_path: Path) -> None:
    """loop() must not flag a merged worker PR whose approved review decision covers the merged head (issue #502)."""
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    # Arm with an empty baseline so the silence asserted below is evidence that
    # the approval matched the merged head — an unarmed pass is silent about
    # everything, which would make this test pass for the wrong reason.
    _arm_unauthorized_merge_tripwire(paths)

    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = [
        {
            "number": 502,
            "title": "fix: approved merge",
            "url": "https://example.test/pull/502",
            "headRefName": "agent/issue-495-fix",
            "baseRefName": "main",
            "headRefOid": "sha-502",
            "state": "MERGED",
            "isCrossRepository": False,
            "body": "Closes #495",
            "labels": [],
        },
    ]

    pr_dir = paths.prs / "pr-502"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-502"}),
        encoding="utf-8",
    )

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.loop(merge=False)

    tripwire_errors = [
        e for e in result.data["errors"] if "possible worker self-merge" in e["error"]
    ]
    assert tripwire_errors == [], (
        f"approved matching-head merge must not be flagged, got {tripwire_errors}"
    )


def test_loop_parks_foreign_issue_ref_pr(monkeypatch, tmp_path: Path) -> None:
    """A PR whose branch-derived issue number does not exist in this repo
    (e.g. opened against the wrong fleet repo) is parked via
    ``foreign_issue_ref`` instead of failing the pass every 5 minutes
    forever. GitHubNotFoundError from issue_view is caught before the
    general GitHubError handler, so it never lands in result.data["errors"]
    and does not flip result.ok to False.

    Issue #1132: parking now requires ``confirm_passes`` (default 2)
    consecutive not-found passes before the marker is confirmed and the
    one-shot digest is emitted. A transient window (minutes) clears before
    two 5-minute passes complete."""
    from charlie_work.config import NotifyConfig
    from charlie_work.github import GitHubNotFoundError

    class ForeignIssueGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = []
            self.prs = [
                {
                    "number": 789,
                    "title": "Fix #4242: foreign",
                    "url": "https://example.test/pull/789",
                    "headRefName": "agent/issue-4242-x",
                    "baseRefName": "main",
                    "headRefOid": "sha-789",
                    "mergeStateStatus": "CLEAN",
                    "body": "Closes #4242",
                    "labels": [],
                    "isCrossRepository": False,
                    "state": "OPEN",
                }
            ]
            self.issue_view_calls = 0

        def issue_view(self, number: int):
            if number == 4242:
                self.issue_view_calls += 1
                raise GitHubNotFoundError("could not resolve to a Issue with the number 4242.")
            return super().issue_view(number)

    config = OrchestratorConfig(
        notify=NotifyConfig(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = ForeignIssueGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    captured: list[Any] = []
    monkeypatch.setattr(
        "charlie_work.workflow.emit_digest",
        lambda notify_config, digest: captured.append(digest),
    )

    # Pass 1: first not-found — marker written with confirmations=1, but
    # not yet confirmed (1 < 2), so no digest and the PR is still tracked.
    result = app.loop(limit=0)

    assert result.ok is True
    assert result.data["errors"] == []
    assert fake_gh.issue_view_calls == 1
    assert len(captured) == 0  # not yet confirmed

    state = load_state(app.paths.state_file)
    assert state["prs"]["789"]["foreign_issue_ref"]["issue"] == 4242
    assert state["prs"]["789"]["foreign_issue_ref"]["confirmations"] == 1

    # Pass 2: second not-found — confirmations reaches 2, marker confirmed,
    # one-shot digest emitted.
    result2 = app.loop(limit=0)

    assert result2.ok is True
    assert result2.data["errors"] == []
    assert fake_gh.issue_view_calls == 2
    assert len(captured) == 1
    assert captured[0].transitions[0].health == "FOREIGN_ISSUE_REF"
    assert captured[0].transitions[0].issue_number == 789

    state = load_state(app.paths.state_file)
    assert state["prs"]["789"]["foreign_issue_ref"]["confirmations"] == 2

    # Pass 3: the confirmed marker skips all per-PR work with zero GitHub
    # calls and no repeat digest.
    result3 = app.loop(limit=0)

    assert result3.ok is True
    assert result3.data["open_tracked_prs"] == 0
    assert fake_gh.issue_view_calls == 2
    assert len(captured) == 1
    # Issue #1132: parked PRs are now visible in the loop_completed payload.
    assert result3.data["parked_prs"] == [789]


def test_loop_honors_intake_failure_signal(tmp_path: Path) -> None:
    """loop() must propagate intake() failures into its ok flag and message so
    a partially failed intake is not silently reported as a clean loop."""
    from charlie_work.github import GitHubError as _GitHubError

    class FlakyIntakeGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = [
                {
                    "number": 123,
                    "title": "Good issue",
                    "url": "https://example.test/issues/123",
                    "body": "ok",
                    "labels": [{"name": "automated-ready"}],
                },
                {
                    "number": 124,
                    "title": "Broken issue",
                    "url": "https://example.test/issues/124",
                    "body": "broken",
                    "labels": [{"name": "automated-ready"}],
                },
            ]

        def issue_view(self, number: int):
            if number == 124:
                raise _GitHubError("transient gh issue view failure")
            return super().issue_view(number)

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FlakyIntakeGitHub())

    result = app.loop(limit=0)

    assert result.ok is False
    assert "intake failures" in result.message
    assert result.data["intake"]["failed"] == [
        {"issue": 124, "error": "transient gh issue view failure"}
    ]
    assert result.data["errors"] == []


def test_loop_surfaces_open_pr_backpressure_fields(tmp_path: Path) -> None:
    """Issue #1129 rework: loop() surfaces the dispatch-scoped open-PR fields.

    The clamp engages inside dispatch() (1 open PR, cap 1 -> dispatch_limit 0);
    _loop_body's 'prefer the dispatch-scoped governor values' copy loop must
    lift open_pr_count/open_pr_max to the top-level CommandResult.data exactly
    like the session-concurrency keys, so a loop() caller sees the backpressure
    that clamped this pass without digging into data["dispatch"].
    """

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_open_agent_prs=1, default_limit=5),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.loop(merge=False)

    assert result.ok is True
    # The clamp engaged inside dispatch (dispatch-scoped values present).
    assert result.data["dispatch"]["open_pr_count"] == 1
    assert result.data["dispatch"]["open_pr_max"] == 1
    # And the copy loop lifted them to the top level.
    assert result.data["open_pr_count"] == 1
    assert result.data["open_pr_max"] == 1


def test_loop_forwards_shared_now_to_cadence_gated_lanes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #828 (critical wiring test): a single frozen ``now`` passed to
    ``loop()`` must reach every cadence-gated lane's ``now`` keyword
    unchanged -- ``_maybe_probe_quota_recovery``, ``_maybe_reconcile_drift``,
    ``_maybe_reclaim_worktrees``, and the module-level
    ``_detect_and_handle_stalled_sessions`` reaper. Each of those functions
    is exercised directly (not via loop()) by its own dedicated test
    elsewhere in this file, so none of those tests would notice a future
    edit that deleted the ``now=now`` forwarding at loop()'s call sites in
    ``_loop_body`` -- this test exists specifically to catch that class of
    regression (L3 "wired", not merely L2 "exists and works standalone").
    Every lane is stubbed to a no-op recorder rather than asserted on
    cadence state, so the test is immune to each lane's own due/not-due
    gating and to the unrelated behavior each lane performs.
    """
    from charlie_work import workflow as workflow_module

    app = _reconcile_pass_app(tmp_path)
    frozen_now = datetime.now(UTC)
    received: dict[str, datetime | None] = {}

    def _record_probe(self: OrchestratorApp, *, now: datetime | None = None) -> None:
        received["probe_quota_recovery"] = now

    def _record_reconcile(self: OrchestratorApp, *, now: datetime | None = None) -> None:
        received["reconcile_drift"] = now

    def _record_reclaim(
        self: OrchestratorApp, *, now: datetime | None = None
    ) -> dict[str, Any] | None:
        received["reclaim_worktrees"] = now
        return None

    def _record_stalled_sessions(
        sessions_dir: Path,
        state_file: Path,
        config: OrchestratorConfig,
        *,
        write_gate: object,
        now: datetime | None = None,
    ) -> list[dict[str, int]]:
        received["stalled_sessions"] = now
        return []

    monkeypatch.setattr(OrchestratorApp, "_maybe_probe_quota_recovery", _record_probe)
    monkeypatch.setattr(OrchestratorApp, "_maybe_reconcile_drift", _record_reconcile)
    monkeypatch.setattr(OrchestratorApp, "_maybe_reclaim_worktrees", _record_reclaim)
    monkeypatch.setattr(
        workflow_module, "_detect_and_handle_stalled_sessions", _record_stalled_sessions
    )

    app.loop(limit=0, now=frozen_now)

    assert received.keys() == {
        "probe_quota_recovery",
        "reconcile_drift",
        "reclaim_worktrees",
        "stalled_sessions",
    }
    for lane, value in received.items():
        assert value is frozen_now, f"{lane} did not receive the pass's frozen now"


def test_loop_corrects_escalated_label_divergence_via_reconcile_pass(tmp_path: Path) -> None:
    """B-AC5 (critical wiring test): a state/label divergence of the
    escalated_labels_converged shape -- state says status "escalated", GitHub
    still carries the stale needs-rework label -- must be corrected by a
    single app.loop() pass. This proves _maybe_reconcile_drift is actually
    wired into _loop_body's production call path, not merely present and
    independently callable. Must FAIL if the wiring call site is removed;
    see the removal verification recorded in the PR description."""
    from charlie_work.config import ReconcilePassConfig

    class EscalatedDivergenceGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = [
                {
                    "number": 40,
                    "title": "issue 40",
                    "url": "https://example.test/issues/40",
                    "body": "",
                    "labels": [{"name": "agent:needs-rework"}],
                    "state": "OPEN",
                }
            ]
            self.prs = []

        def run(
            self, args: list[str], *, json_output: bool = False, allow_failure: bool = False
        ) -> Any:
            if args[:2] == ["issue", "list"]:
                return self.issues
            if args[:2] == ["pr", "list"]:
                return self.prs
            return super().run(args, json_output=json_output, allow_failure=allow_failure)

        def issue_list(self, labels: Any = None, state: Any = None) -> list[dict[str, Any]]:
            return self.issues

        def pr_list(self) -> list[dict[str, Any]]:
            return self.prs

        def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
            return {
                issue["number"]
                for issue in self.issues
                if issue["number"] in issue_numbers and issue.get("state") == "OPEN"
            }

        def add_issue_label(self, number: int, label: str) -> bool:
            super().add_issue_label(number, label)
            for issue in self.issues:
                if issue["number"] == number:
                    names = {entry.get("name") for entry in issue["labels"]}
                    if label not in names:
                        issue["labels"].append({"name": label})
            return True

        def remove_issue_label(self, number: int, label: str) -> bool:
            super().remove_issue_label(number, label)
            for issue in self.issues:
                if issue["number"] == number:
                    issue["labels"] = [
                        entry for entry in issue["labels"] if entry.get("name") != label
                    ]
            return True

    gh = EscalatedDivergenceGitHub()
    config = OrchestratorConfig(
        reconcile_pass=ReconcilePassConfig(enabled=True, interval_minutes=30)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    app = OrchestratorApp(tmp_path, paths, config, gh)

    state = load_state(app.paths.state_file)
    state = {
        **state,
        "issues": {**state.get("issues", {}), "40": {"number": 40, "status": "escalated"}},
    }
    save_state(app.paths.state_file, state)

    app.loop(limit=1)

    assert (40, "agent:human-needed") in gh.labels_added
    assert (40, "agent:needs-rework") in gh.labels_removed

    # B-AC7 (critical safety invariant): reconcile must never rewrite an
    # open escalated issue's status to match labels -- only `charlie unescalate`
    # re-enters the machine.
    final_state = load_state(app.paths.state_file)
    assert final_state["issues"]["40"]["status"] == "escalated"


def test_loop_calls_maybe_reclaim_superseded_main_ci(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L3 'wired' regression guard (#863/#815): proves _loop_body actually
    calls this lane, not just that the lane works standalone."""
    app = _main_ci_reclaim_app(tmp_path)
    calls = {"count": 0}

    def _record(self: OrchestratorApp) -> None:
        calls["count"] += 1

    monkeypatch.setattr(OrchestratorApp, "_maybe_reclaim_superseded_main_ci", _record)
    app.loop(limit=0)
    assert calls["count"] == 1


def test_loop_emits_concurrency_fields_when_governor_enabled(tmp_path: Path) -> None:
    """Regression test for issue #100: loop() must emit concurrency fields when governor is enabled,
    even when not clamped.

    On origin/main, loop() emitted concurrency_limit/live_session_count/available_slots whenever
    max_concurrent > 0 (governor enabled), regardless of whether it actually clamped anything.
    The initial PR implementation changed this to only emit when clamped, which was a silent behavior
    change. This test ensures the original semantics are preserved: fields appear when the governor
    is enabled, not only when it's actively throttling.
    """
    from charlie_work.config import DevinConfig, DispatchConfig

    # Configure with max_concurrent_sessions=5 (enabled but not clamping in this scenario)
    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=5),
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Run loop with no live sessions (governor enabled but not clamped)
    result = app.loop(limit=0)

    # Assert that concurrency fields are present even though governor is not clamped
    assert "concurrency_limit" in result.data
    assert result.data["concurrency_limit"] == 5
    assert "live_session_count" in result.data
    assert result.data["live_session_count"] == 0
    assert "available_slots" in result.data


# ---------------------------------------------------------------------------
# loop() additions: open_tracked_prs + same-head packet skip
# ---------------------------------------------------------------------------


def test_loop_open_tracked_prs_counted(tmp_path: Path) -> None:
    """loop() data includes open_tracked_prs = number of PRs with linked issues."""
    prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "headRefOid": "sha-abc",
            "body": "Closes #123",
            "labels": [],
            "isCrossRepository": False,
        },
        # This PR has no linked issue — should NOT count
        {
            "number": 999,
            "title": "Manual PR",
            "url": "https://example.test/pull/999",
            "headRefName": "manual-branch",
            "headRefOid": "sha-xyz",
            "body": "no issue link",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app, _ = _make_loop_app(tmp_path, prs=prs)
    result = app.loop(limit=0)
    assert "open_tracked_prs" in result.data
    assert result.data["open_tracked_prs"] == 1


def test_loop_open_tracked_prs_zero_when_no_prs(tmp_path: Path) -> None:
    """loop() returns open_tracked_prs=0 when there are no open PRs."""
    app, _ = _make_loop_app(tmp_path, prs=[])
    result = app.loop(limit=0)
    assert result.data["open_tracked_prs"] == 0


def test_loop_records_sink_metric_in_completed_event_and_pass_row(
    tmp_path: Path,
) -> None:
    """Issue #1083: a loop pass reports the sink metric alongside autonomy.

    Autonomy (merge_count/review_count) must never be reported without its
    drop rate. This test asserts the ``loop_completed`` event payload and the
    ``loop_passes`` row both carry ``sink_population``, ``sink_arrivals``,
    and ``sink_clears`` for the pass, with arrivals derived from a
    before/after census diff around ``_loop_body``.

    ``_loop_body`` is replaced with a stub that escalates one fresh issue
    mid-pass, so the pass observes one pre-existing parked issue (population
    before) plus one arrival (population after = 2, arrivals = 1). The stub
    emits no ``deescalation_cleared`` event, so ``sink_clears`` is 0.
    """
    from charlie_work.instrumentation import _get_db

    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # One issue already parked in the sink before the pass.
    state = load_state(paths.state_file)
    state["issues"]["100"] = {
        "number": 100,
        "status": "escalated",
        "reason_class": "judgment",
    }
    save_state(paths.state_file, state)

    # Stub _loop_body to escalate a second issue during the pass and return a
    # clean CommandResult without touching GitHub. This isolates the sink
    # census diff (the issue #1083 measurement) from the rest of the pass.
    def stub_body(limit: int | None, *, merge: bool | None, now=None) -> CommandResult:
        mid = load_state(paths.state_file)
        mid["issues"]["200"] = {
            "number": 200,
            "status": "blocked",
            "reason_class": "judgment",
        }
        save_state(paths.state_file, mid)
        return CommandResult(
            ok=True, message="stub", data={"errors": [], "merges": [], "reviews": []}
        )

    app._loop_body = stub_body  # type: ignore[assignment]
    app.loop(limit=0)

    completed = query_events(paths.state_file, kind="loop_completed")
    assert completed, "loop_completed event was not emitted"
    payload = completed[-1]["payload"]
    assert payload["sink_population"] == 2
    assert payload["sink_arrivals"] == 1
    assert payload["sink_clears"] == 0

    # The loop_passes row carries the same metric in queryable columns.
    conn = _get_db(paths.state_file)
    assert conn is not None
    row = conn.execute(
        "SELECT sink_population, sink_arrivals, sink_clears FROM loop_passes "
        "ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    assert row is not None
    assert row["sink_population"] == 2
    assert row["sink_arrivals"] == 1
    assert row["sink_clears"] == 0


# ---------------------------------------------------------------------------
# Issue #1363 PART 2: preflight gate wiring into OrchestratorApp.loop()
#
# Both tests below monkeypatch ``charlie_work.workflow.run_preflight``
# directly with a canned ``PreflightResult`` rather than driving the real
# disk/clock/venv/config probes through ``tmp_path``. That is the correct
# abstraction layer for a *wiring* test: run_preflight's own check logic
# (disk_floor math, venv_identity path matching, config_freshness
# once-per-change semantics, ...) is already exhaustively covered at the
# unit level in test_preflight.py. What is untested until now is whether
# OrchestratorApp._loop_impl reacts to a PreflightResult correctly -- and
# canning the result also makes these tests immune to the ambient
# sys.executable/orchestrator_root() of whatever environment happens to run
# them (a real venv-synced checkout under CI, a PYTHONPATH-overridden
# worktree locally, ...), which the real venv_identity check is otherwise
# sensitive to.
# ---------------------------------------------------------------------------


def test_loop_fatal_preflight_refusal_skips_loop_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC2: a fatal preflight failure must refuse the pass -- _loop_body must
    never run, a loop_refused_preflight event must be recorded, and loop()
    must return a non-ok CommandResult naming the failing check -- without
    a loop_completed event ever appearing."""
    from charlie_work.preflight import PreflightCheck, PreflightResult

    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    fatal_check = PreflightCheck(
        name="disk_floor", ok=False, detail="0.10 GB free (floor 5 GB)", fatal=True
    )
    fake_result = PreflightResult(checks=(fatal_check,))
    monkeypatch.setattr("charlie_work.workflow.run_preflight", lambda *args, **kwargs: fake_result)

    body_invoked = False

    def stub_body(limit: int | None, *, merge: bool | None, now=None) -> CommandResult:
        nonlocal body_invoked
        body_invoked = True
        return CommandResult(ok=True, message="stub", data={})

    app._loop_body = stub_body  # type: ignore[assignment]
    result = app.loop(limit=0)

    assert body_invoked is False, "_loop_body ran despite a fatal preflight refusal"
    assert result.ok is False
    assert result.data.get("pass_skipped") is True
    assert result.data.get("reason") == "preflight_refused"
    assert result.data.get("check") == "disk_floor"

    refused = query_events(paths.state_file, kind="loop_refused_preflight")
    assert refused, "loop_refused_preflight event was not emitted"
    assert refused[-1]["payload"]["check"] == "disk_floor"

    completed = query_events(paths.state_file, kind="loop_completed")
    assert not completed, "loop_completed must not fire when preflight refuses the pass"


def test_loop_healthy_preflight_proceeds_with_no_extra_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC3: a fully-passing preflight run must add nothing beyond the
    ordinary loop event sequence -- no loop_refused_preflight, no
    preflight_warning/preflight_config_stale noise -- and loop_completed
    must still fire normally. This is the "healthy host" regression control
    for AC2: proof that the gate's presence is invisible on a clean pass."""
    from charlie_work.preflight import PreflightCheck, PreflightResult

    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    all_ok_result = PreflightResult(
        checks=(
            PreflightCheck(name="disk_floor", ok=True, detail="ok", fatal=True),
            PreflightCheck(name="clock_sanity", ok=True, detail="ok", fatal=False),
            PreflightCheck(name="venv_identity", ok=True, detail="ok", fatal=True),
            PreflightCheck(name="config_freshness", ok=True, detail="ok", fatal=False),
        )
    )
    monkeypatch.setattr(
        "charlie_work.workflow.run_preflight", lambda *args, **kwargs: all_ok_result
    )

    def stub_body(limit: int | None, *, merge: bool | None, now=None) -> CommandResult:
        return CommandResult(
            ok=True, message="stub", data={"errors": [], "merges": [], "reviews": []}
        )

    app._loop_body = stub_body  # type: ignore[assignment]
    result = app.loop(limit=0)

    assert result.ok is True
    assert not query_events(paths.state_file, kind="loop_refused_preflight")
    assert not query_events(paths.state_file, kind="preflight_warning")
    assert not query_events(paths.state_file, kind="preflight_config_stale")
    completed = query_events(paths.state_file, kind="loop_completed")
    assert completed, "loop_completed event was not emitted on a healthy preflight pass"
