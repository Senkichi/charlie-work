"""Mid-lane in-pass deadline enforcement for fleet repo lanes (issue #1948).

``fleet_loop``'s cooperative deadline (``max_pass_runtime_seconds``, issue
#1832) was previously only consulted *between* repo lanes, so a single lane
whose GitHub calls all timed out could overrun the pass budget by tens of
minutes -- production ``fleet_pass_deadline_deferred`` events showed
whole-pass elapsed times of 2708-4878s against a 1800s budget. The fix
threads the same ``deadline_exceeded`` predicate three ways:

1. ``_run_fleet_repo_lane`` arms it on the lane's own ``GitHub`` client --
   ``GitHub.run()`` refuses a new call (or aborts a retry chain) the moment
   the budget is spent, raising ``PassDeadlineExceeded`` (a
   ``BaseException`` -- the ``asyncio.CancelledError`` precedent, so no
   ``except Exception`` / ``except GitHubError`` site can swallow it and
   re-enter the ordinary-failure vocabulary) instead of accumulating
   timeout+backoff cycles.
2. The lane checks it at its own yield points (before the first real
   phase, between the work-only dispatch and review-dispatch calls).
3. ``loop()`` -> ``_loop_impl`` -> ``_loop_body`` carries it into the
   per-repo pass, which checks it at every GitHub-touching sub-phase
   boundary and at the top of the per-PR review/merge scan; a lane cut
   short returns ``data["deadline_deferred"] = True``, which ``fleet_loop``
   collects into ``deadline_partial_repo_keys`` (distinct from the
   never-started ``deferred`` list).
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from _fakes_github import FakeGitHub
from _fleet_dispatch_fixtures import (
    _StepClock,
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
    _per_repo_runtime_paths,
)
from _fake_transport import FakeAdapter, failure, make_github, ok
from charlie_work import github as github_module
from charlie_work import layout
from charlie_work.config import (
    DeescalationConfig,
    DevinConfig,
    MainCiReclaimConfig,
    OrchestratorConfig,
    ReconcilePassConfig,
    ReviewDispatchConfig,
    WorkerRoleConfig,
    WorktreeReclamationConfig,
)
from charlie_work.fleet_dispatch import fleet_loop
from charlie_work.fleet_lanes import _run_fleet_repo_lane
from charlie_work.fleet_paths import fleet_dir
from charlie_work.github import GitHub
from charlie_work.github_transport import FailureKind
from charlie_work.instrumentation import query_events
from charlie_work.pass_deadline import (
    PassDeadlineExceeded,
    set_pass_deadline_exceeded,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import empty_state, save_state
from charlie_work.command_result import CommandResult
from charlie_work.workflow import OrchestratorApp


# ---------------------------------------------------------------------------
# GitHub.run() transport-level refusal (the per-call yield point)
# ---------------------------------------------------------------------------


def test_gh_run_refuses_call_when_pass_deadline_exceeded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An armed, tripped deadline refuses a gh call without spawning it.

    The refusal RAISES ``PassDeadlineExceeded`` under BOTH allow_failure
    modes: a refusal is control flow, not a transport failure, and every
    existing allow_failure consumer misreads a refusal-shaped result as a
    real gh failure (the review finding this contract answers).
    """
    gh, http, _ = make_github(tmp_path)
    set_pass_deadline_exceeded(gh, lambda: True)

    with pytest.raises(PassDeadlineExceeded, match="in-pass deadline"):
        gh.run(["api", "rate_limit"], allow_failure=True)
    assert http.calls == []

    with pytest.raises(PassDeadlineExceeded, match="in-pass deadline"):
        gh.run(["api", "rate_limit"], allow_failure=False)
    assert http.calls == []


def test_pass_deadline_exceeded_is_cancellation_not_an_exception() -> None:
    """``PassDeadlineExceeded`` must be uncatchable by ``except Exception``.

    ``asyncio.CancelledError`` precedent: a refusal is cooperative
    cancellation. The lane's call graph contains broad ``except Exception``
    handlers whose fallbacks are dangerous on a spent budget (stale-data
    rework routing, fail-open branch validation, merge_ready's comment
    handler falling through to the failed-attempt counter). BaseException
    makes the whole class of swallow bugs unreachable, and keeps the
    refusal out of every ``except GitHubError`` failure-value translator.
    """
    assert issubclass(PassDeadlineExceeded, BaseException)
    assert not issubclass(PassDeadlineExceeded, Exception)

    from charlie_work.github import GitHubError

    assert not issubclass(PassDeadlineExceeded, GitHubError)


def test_gh_run_unarmed_deadline_is_inert(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No armed hook (every non-fleet caller) means zero behavior change."""
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [ok("ok-out")]))

    assert gh.run(["api", "rate_limit"]) == "ok-out"
    assert len(http.calls) == 1


def test_gh_run_aborts_retry_chain_at_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A deadline reached mid-retry-chain aborts without burning a backoff.

    The predicate is False until the first attempt has been sent and True
    for the pre-sleep re-check, so the run must refuse rather than sleeping
    into a second spawn -- the whole point of #1948 is that the lane stops
    accumulating timeout+backoff cycles once the budget is spent.
    """
    http = FakeAdapter("http", [failure(FailureKind.TIMEOUT, "timed out after 30s")])
    gh, _http, _ = make_github(tmp_path, http=http)

    def _deadline() -> bool:
        # Open for the pre-attempt checks (including the one gating the
        # ``gh auth token`` mint), spent once the first request has failed.
        return len(http.api_requests) >= 1

    set_pass_deadline_exceeded(gh, _deadline)

    sleeps: list[float] = []
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    with pytest.raises(PassDeadlineExceeded, match="in-pass deadline"):
        gh.run(["api", "rate_limit"], allow_failure=True)
    assert len(http.calls) == 1, (
        "a retryable timeout followed by a tripped deadline must not send "
        "another request -- the refusal replaces the whole remaining retry chain"
    )
    assert sleeps == [], "the deadline refusal must precede the backoff sleep"


def test_gh_run_checks_deadline_before_transient_retry_backoff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The NON-timeout retry branch re-checks the deadline before backoff.

    ``test_gh_run_aborts_retry_chain_at_deadline`` covers the
    ``TimeoutExpired`` branch; this covers the sibling branch -- a
    transient non-timeout gh failure ("connection reset") that
    ``is_retryable`` classifies retryable for a read call. The predicate
    is False for the pre-attempt check and True for the pre-sleep check.
    """
    http = FakeAdapter("http", [failure(FailureKind.SENT_NO_RESPONSE, "connection reset by peer")])
    gh, _http, _ = make_github(tmp_path, http=http)

    def _deadline() -> bool:
        # Open for the pre-attempt checks (including the one gating the
        # ``gh auth token`` mint), spent once the first request has failed.
        return len(http.api_requests) >= 1

    set_pass_deadline_exceeded(gh, _deadline)

    sleeps: list[float] = []
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    with pytest.raises(PassDeadlineExceeded, match="in-pass deadline"):
        gh.run(["api", "rate_limit"], allow_failure=True)
    assert len(http.calls) == 1, (
        "a retryable transient failure followed by a tripped deadline must "
        "not send another request -- the refusal replaces the retry chain"
    )
    assert sleeps == [], "the deadline refusal must precede the backoff sleep"


# ---------------------------------------------------------------------------
# _run_fleet_repo_lane yield points + hook arming
# ---------------------------------------------------------------------------


def _lane_lock() -> MagicMock:
    return MagicMock(name="lane_lock")


def test_fleet_repo_lane_arms_deadline_on_real_github(tmp_path: Path) -> None:
    """A lane with a real GitHub client arms the run()-level deadline hook."""
    real_gh = GitHub(tmp_path)
    app = MagicMock()
    app.gh = real_gh
    app.loop.return_value = CommandResult(True, "ok", {})
    predicate = lambda: False  # noqa: E731 -- tiny predicate, not a named fn

    result = _run_fleet_repo_lane(
        "owner/repo1",
        app,
        OrchestratorConfig(),
        _lane_lock(),
        work_only=False,
        drain=False,
        limit=3,
        merge=True,
        ensure_labels=False,
        fleet_state_path=tmp_path / "state.json",
        deadline_exceeded=predicate,
    )

    assert result.ok is True
    assert real_gh._pass_deadline_exceeded is predicate
    # And the armed client actually refuses once the predicate trips.
    set_pass_deadline_exceeded(real_gh, lambda: True)
    with pytest.raises(PassDeadlineExceeded, match="in-pass deadline"):
        real_gh.run(["api", "rate_limit"], allow_failure=True)


def test_fleet_repo_lane_stops_between_dispatch_and_review_dispatch(tmp_path: Path) -> None:
    """work_only lane: deadline tripped after dispatch defers dispatch_reviews."""
    app = MagicMock()
    app.dispatch.return_value = CommandResult(True, "dispatched", {"selected_count": 1})
    config = OrchestratorConfig(review_dispatch=ReviewDispatchConfig(enabled=True))
    lock = _lane_lock()

    checks = iter((False, True))
    result = _run_fleet_repo_lane(
        "owner/repo1",
        app,
        config,
        lock,
        work_only=True,
        drain=False,
        limit=3,
        merge=None,
        ensure_labels=False,
        fleet_state_path=tmp_path / "state.json",
        deadline_exceeded=lambda: next(checks),
    )

    app.dispatch.assert_called_once_with(3)
    app.dispatch_reviews.assert_not_called()
    assert result.data["deadline_deferred"] is True
    assert result.data["dispatch_reviews"] == {"deadline_deferred": True}
    lock.release.assert_called_once()


def test_fleet_repo_lane_bails_before_first_phase(tmp_path: Path) -> None:
    """Deadline already spent at lane start: no dispatch/loop call at all."""
    app = MagicMock()
    lock = _lane_lock()

    result = _run_fleet_repo_lane(
        "owner/repo1",
        app,
        OrchestratorConfig(),
        lock,
        work_only=True,
        drain=False,
        limit=3,
        merge=None,
        ensure_labels=False,
        fleet_state_path=tmp_path / "state.json",
        deadline_exceeded=lambda: True,
    )

    app.dispatch.assert_not_called()
    app.loop.assert_not_called()
    assert result.ok is True
    assert result.data["deadline_deferred"] is True
    lock.release.assert_called_once()


def test_fleet_repo_lane_threads_deadline_into_loop(tmp_path: Path) -> None:
    """The full-loop lane forwards the SAME predicate object into app.loop()."""
    app = MagicMock()
    app.loop.return_value = CommandResult(True, "ok", {})
    predicate = lambda: False  # noqa: E731 -- tiny predicate, not a named fn

    _run_fleet_repo_lane(
        "owner/repo1",
        app,
        OrchestratorConfig(),
        _lane_lock(),
        work_only=False,
        drain=False,
        limit=3,
        merge=True,
        ensure_labels=False,
        fleet_state_path=tmp_path / "state.json",
        deadline_exceeded=predicate,
    )

    app.loop.assert_called_once_with(3, merge=True, deadline_exceeded=predicate)


def test_fleet_repo_lane_no_deadline_is_inert(tmp_path: Path) -> None:
    """deadline_exceeded=None keeps the pre-#1948 lane behavior byte-identical."""
    app = MagicMock()
    app.loop.return_value = CommandResult(True, "ok", {})

    _run_fleet_repo_lane(
        "owner/repo1",
        app,
        OrchestratorConfig(),
        _lane_lock(),
        work_only=False,
        drain=False,
        limit=3,
        merge=True,
        ensure_labels=False,
        fleet_state_path=tmp_path / "state.json",
    )

    app.loop.assert_called_once_with(3, merge=True, deadline_exceeded=None)


# ---------------------------------------------------------------------------
# _loop_body sub-phase boundary checks (real OrchestratorApp + FakeGitHub)
# ---------------------------------------------------------------------------


def _build_app(root: Path) -> tuple[OrchestratorApp, Any, FakeGitHub]:
    """Minimal real-app harness in the shape of test_write_gate_dry_run_loop's."""
    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        deescalation=DeescalationConfig(enabled=False),
        worktree_reclamation=WorktreeReclamationConfig(enabled=False),
        main_ci_reclaim=MainCiReclaimConfig(enabled=False),
        reconcile_pass=ReconcilePassConfig(enabled=False),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(root, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    save_state(paths.state_file, empty_state())

    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = []

    app = OrchestratorApp(root, paths, config, fake_gh)
    (root / ".var" / "charlie-work" / "dispatches" / "sessions").mkdir(parents=True, exist_ok=True)
    return app, paths, fake_gh


def test_loop_body_returns_deferred_when_deadline_already_hit(tmp_path: Path) -> None:
    """A lane submitted on a dead budget returns an empty deferred pass."""
    app, paths, fake_gh = _build_app(tmp_path / "repo")
    intake_calls: list = []
    real_issue_list = fake_gh.issue_list
    fake_gh.issue_list = lambda *a, **k: (intake_calls.append(1), real_issue_list(*a, **k))[1]

    result = app._loop_body(
        limit=0,
        merge=False,
        now=datetime.datetime.now(datetime.UTC),
        deadline_exceeded=lambda: True,
    )

    assert result.ok is True
    assert result.data == {"deadline_deferred": True}
    assert intake_calls == [], "with the budget already spent, the pass must not reach even intake"
    assert query_events(paths.state_file, kind="loop_pass_deadline_deferred") == []


def test_loop_body_defers_phases_after_deadline_trips_mid_pass(tmp_path: Path) -> None:
    """Deadline tripped during intake: every later gh sub-phase is deferred.

    The predicate trips inside the real ``intake()`` call, so every
    sub-phase boundary after it must read True: the dispatch lanes become
    ``deadline_deferred`` placeholders, the PR scan runs on an empty list,
    and the pass is marked partial rather than erroring.
    """
    app, paths, fake_gh = _build_app(tmp_path / "repo")
    pr_list_calls: list = []
    real_pr_list = fake_gh.pr_list
    fake_gh.pr_list = lambda *a, **k: (pr_list_calls.append(1), real_pr_list(*a, **k))[1]

    tripped = {"hit": False}
    real_intake = app.intake

    def _intake_and_trip():
        tripped["hit"] = True
        return real_intake()

    app.intake = _intake_and_trip

    result = app._loop_body(
        limit=0,
        merge=False,
        now=datetime.datetime.now(datetime.UTC),
        deadline_exceeded=lambda: tripped["hit"],
    )

    assert result.ok is True
    assert result.data["deadline_deferred"] is True
    assert "(partial: in-pass deadline reached)" in result.message
    # intake itself ran (the deadline was still open when it started)...
    assert "deadline_deferred" not in result.data["intake"]
    # ...but every GitHub-touching phase after it was deferred.
    for phase in ("dispatch_rework", "dispatch", "dispatch_reviews", "local_lane"):
        assert result.data[phase] == {"deadline_deferred": True}, phase
    assert result.data["reaped"] == []
    assert result.data["open_tracked_prs"] == 0
    assert pr_list_calls == [], "the PR scan's own fetch must not start"
    events = query_events(paths.state_file, kind="loop_pass_deadline_deferred")
    assert len(events) == 1


def test_loop_body_breaks_pr_scan_when_deadline_trips_mid_iteration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The per-PR review/merge scan stops at the first PR past the deadline.

    PR1 resolves to no linked issue (cheap continue); the predicate trips
    during PR1's ``linked_issue_number`` call, so iteration 2's boundary
    check breaks the scan before PR2 -- a linked PR that would otherwise
    reach the review machinery -- is touched.
    """
    app, paths, fake_gh = _build_app(tmp_path / "repo")
    unlinked_pr = {
        "number": 101,
        "title": "Unrelated work",
        "url": "https://example.test/pull/101",
        "headRefName": "feature/unrelated",
        "baseRefName": "main",
        "headRefOid": "sha-unlinked",
        "mergeStateStatus": "CLEAN",
        "body": "no linked issue here",
        "labels": [],
        "isCrossRepository": False,
        "state": "OPEN",
    }
    linked_pr = {
        "number": 456,
        "title": "Fix #123: search",
        "url": "https://example.test/pull/456",
        "headRefName": "agent/issue-123-fix-search",
        "baseRefName": "main",
        "headRefOid": "sha-abc123",
        "mergeStateStatus": "CLEAN",
        "body": "Closes #123",
        "labels": [],
        "isCrossRepository": False,
        "state": "OPEN",
    }
    fake_gh.prs = [unlinked_pr, linked_pr]

    import charlie_work.workflow as _wf

    tripped = {"hit": False}
    real_linked = _wf.linked_issue_number

    def _linked_and_trip(*args, **kwargs):
        tripped["hit"] = True
        return real_linked(*args, **kwargs)

    monkeypatch.setattr("charlie_work.workflow.linked_issue_number", _linked_and_trip)

    result = app._loop_body(
        limit=0,
        merge=False,
        now=datetime.datetime.now(datetime.UTC),
        deadline_exceeded=lambda: tripped["hit"],
    )

    assert result.ok is True
    assert result.data["deadline_deferred"] is True
    # PR1 was iterated (unlinked -> continue); PR2's linked PR never reached
    # the counter -- had the loop-top check not fired, this would be 1.
    assert result.data["open_tracked_prs"] == 0
    # PR2's review machinery never ran: no pr-456 packet directory.
    assert not (paths.prs / "pr-456").exists()
    events = query_events(paths.state_file, kind="loop_pass_deadline_deferred")
    assert len(events) == 1


def test_loop_body_unarmed_deadline_runs_full_pass(tmp_path: Path) -> None:
    """No predicate (None) or a never-tripping one: no deadline markers."""
    app, paths, fake_gh = _build_app(tmp_path / "repo")

    result = app._loop_body(
        limit=0,
        merge=False,
        now=datetime.datetime.now(datetime.UTC),
        deadline_exceeded=lambda: False,
    )

    assert "deadline_deferred" not in result.data
    assert "(partial:" not in result.message
    assert query_events(paths.state_file, kind="loop_pass_deadline_deferred") == []


# ---------------------------------------------------------------------------
# fleet_loop collection: mid-lane partials are not never-started deferrals
# ---------------------------------------------------------------------------


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_records_mid_lane_partial_repo(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """A lane that ran and returned ``deadline_deferred`` is a PARTIAL pass.

    It lands in ``data["repos"]`` (the lane really ran) AND in
    ``data["deadline_partial_repo_keys"]`` / the pass-level event --
    never in ``data["deferred"]``, which is reserved for lanes that never
    started.
    """
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "repo1"),
                "config_path": "orchestrator.config.yaml",
            },
        }
    }
    mock_load_registry.return_value = registry
    (tmp_path / "repo1").mkdir()
    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_runtime_paths.side_effect = _per_repo_runtime_paths

    mock_app = MagicMock()
    mock_app.loop.return_value = CommandResult(
        True, "loop complete (partial: in-pass deadline reached)", {"deadline_deferred": True}
    )
    mock_app_class.return_value = mock_app
    mock_gh_class.return_value = MagicMock()

    # Clock always under the fleet-level deadline -> the partial marking
    # came from the lane itself, not a between-lane deferral.
    clock = _StepClock(steps=[0.0], after=1.0)

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=("owner/repo1",),
        work_only=False,
        deadline_seconds=100,
        pass_clock=clock,
    )

    assert result.ok is True
    assert result.data["repos"].keys() == {"owner/repo1"}
    assert result.data["repos"]["owner/repo1"]["ok"] is True
    assert result.data["deferred"] == []
    assert result.data["deadline_partial_repo_keys"] == ["owner/repo1"]

    # The predicate threaded into the lane is the live pass budget check.
    _, kwargs = mock_app.loop.call_args
    assert callable(kwargs["deadline_exceeded"])
    assert kwargs["deadline_exceeded"]() is False

    fleet_state_path = layout.state_file_path(fleet_dir(override=str(tmp_path / "fleet")))
    events = query_events(fleet_state_path, kind="fleet_pass_deadline_deferred")
    assert len(events) == 1
    assert events[0]["payload"]["deadline_partial_repo_keys"] == ["owner/repo1"]
    assert events[0]["payload"]["deferred_repo_keys"] == []
