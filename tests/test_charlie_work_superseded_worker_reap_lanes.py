"""Launch-lane integration for the superseded-worker reap (issue #1494).

The unit-level coverage of ``_reap_superseded_workers`` and
``_reap_superseded_workers_for_launch`` lives in the sibling module
``test_charlie_work_superseded_worker_reap.py``; both share
``tests/_superseded_worker_reap_fixtures.py``. This module exercises the
two launch lanes end to end:

* ``OrchestratorApp.dispatch_rework`` (remote rework,
  ``_dispatch_rework_impl``) must reap the issue's recorded prior worker
  before calling ``dispatch_sessions``; a surviving or live-
  unfingerprinted prior worker produces a synthetic
  ``prior_worker_still_alive`` blocked result that flows through the
  blocked-environment accounting (``blocked_environment_at`` accrual, cap
  escalation) instead of launching a second writer.
* ``OrchestratorApp._local_dispatch_rework`` (local/no-remote lane) shares
  the same helper; a blocked launch releases the claim back to
  ``rework_requested`` with no blocked-environment counter.

The final guard test pins the single enforcement point: both lane modules
must delegate to ``_reap_superseded_workers_for_launch`` and neither may
spell the failure-kind literal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import (
    _blocked_env_timestamps,
    _seed_two_rework_issues,
    _TwoReworkIssuesGitHub,
)
from _superseded_worker_reap_fixtures import (
    PRIOR_PID,
    PRIOR_START,
    _ok_dispatch_factory,
    _patch_reap_internals,
    _rework_app,
    _worker_view,
)
from charlie_work.adapters import SessionDispatchResult
from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.host.fakes import FakeWorkerLauncher
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp


# ---------------------------------------------------------------------------
# _dispatch_rework_impl — launch-trigger integration
# ---------------------------------------------------------------------------


def test_dispatch_rework_reaps_live_prior_worker_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    """Issue #1494 acceptance: when rework redispatch fires while the issue's
    prior worker is still running, the replacement launch must kill the prior
    worker's process tree first — verified fingerprint and all.

    The seeded state is exactly what the janitor-gate/worktree-rescue stall
    route leaves behind: status ``rework_requested`` with the prior worker's
    ``worker_pid``/``worker_process_start_time`` still recorded (preserved
    deliberately for recovery probes, issues #165/#282/#295) and a live
    sidecar in sessions_dir.
    """
    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; sys.exit(1)")),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
    )
    app, _fake_gh, paths = _rework_app(tmp_path, config)

    # The still-recorded prior worker: state pid + live sidecar agree.
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"]["worker_pid"] = PRIOR_PID
        state["issues"]["123"]["worker_process_start_time"] = PRIOR_START
        save_state(paths.state_file, state)

    alive = {PRIOR_PID: True}
    order: list[str] = []
    kill_calls: list[tuple[int, float | None]] = []
    events: list[tuple[str, dict]] = []
    _patch_reap_internals(
        monkeypatch,
        workers=[_worker_view(123, PRIOR_PID)],
        alive=alive,
        kill_calls=kill_calls,
        events=events,
    )

    def _ordered_kill(pid: int, st: float | None = None) -> list[int]:
        order.append("kill_process_tree")
        kill_calls.append((pid, st))
        alive[pid] = False
        return [pid]

    monkeypatch.setattr("charlie_work.write_gate.kill_process_tree", _ordered_kill)
    fake_host(worker_launch=FakeWorkerLauncher([_ok_dispatch_factory(order)]))

    result = app.dispatch_rework()

    assert result.ok is True
    # The kill ran with the recorded fingerprint BEFORE any launch call.
    assert kill_calls == [(PRIOR_PID, PRIOR_START)]
    assert order == ["kill_process_tree", "dispatch_sessions"]
    assert [kind for kind, _p in events] == ["superseded_worker_reaped"]

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatched"
    # The replacement's own pid supersedes the stale record.
    assert state["issues"]["123"]["worker_pid"] == 99999


def test_dispatch_rework_janitor_gate_route_reaps_prior_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    """The exact #1494 route: a rework request produced by the janitor gate
    (merge-conflict routing through ``_route_janitor_gate_failure_to_rework``,
    driven here by merge_ready on a CONFLICTING approved PR) still carries a
    live prior worker — the redispatch must reap it before launching."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed",),
            update_open_prs="next",
            failed_attempt_alarm=1,
        ),
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; sys.exit(1)")),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "DIRTY",
            "mergeable": "CONFLICTING",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    # The prior worker is still alive mid-run: record its pid the way a
    # dispatch bookkeeping write leaves it.
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"]["worker_pid"] = PRIOR_PID
        state["issues"]["123"]["worker_process_start_time"] = PRIOR_START
        save_state(paths.state_file, state)

    # The janitor gate routes the conflict to rework without checking the
    # prior worker's liveness — the state #1494 covers.
    dispatch_result = app.merge_ready(456, merge=False)
    assert dispatch_result.ok is True
    assert dispatch_result.data["merge_conflict"] is True
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    # worker_pid survives the rework routing (recovery-probe data, #165/#282).
    assert state["issues"]["123"]["worker_pid"] == PRIOR_PID

    order: list[str] = []
    kill_calls: list[tuple[int, float | None]] = []
    events: list[tuple[str, dict]] = []
    alive = {PRIOR_PID: True}
    _patch_reap_internals(
        monkeypatch,
        workers=[_worker_view(123, PRIOR_PID)],
        alive=alive,
        kill_calls=kill_calls,
        events=events,
    )

    def _ordered_kill(pid: int, st: float | None = None) -> list[int]:
        order.append("kill_process_tree")
        kill_calls.append((pid, st))
        alive[pid] = False
        return [pid]

    monkeypatch.setattr("charlie_work.write_gate.kill_process_tree", _ordered_kill)
    fake_host(worker_launch=FakeWorkerLauncher([_ok_dispatch_factory(order)]))

    result = app.dispatch_rework()

    assert result.ok is True
    assert kill_calls == [(PRIOR_PID, PRIOR_START)]
    assert order == ["kill_process_tree", "dispatch_sessions"]
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatched"


def test_dispatch_rework_blocks_launch_when_prior_worker_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    """A prior worker that survives the reap attempt (kill refused by the
    fingerprint re-verification, or the pid still alive afterward) must NOT
    get a replacement launched next to it. The failed launch routes through
    blocked-environment accounting: ``blocked_environment_at`` grows,
    ``redispatch_at`` does not, and the claim releases back to
    ``rework_requested``."""
    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; sys.exit(1)")),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
    )
    app, _fake_gh, paths = _rework_app(tmp_path, config)

    # Prior worker live WITHOUT a fingerprint: cannot be killed safely.
    alive = {PRIOR_PID: True}
    kill_calls: list[tuple[int, float | None]] = []
    events: list[tuple[str, dict]] = []
    _patch_reap_internals(
        monkeypatch,
        workers=[_worker_view(123, PRIOR_PID, process_start_time=None)],
        alive=alive,
        kill_calls=kill_calls,
        events=events,
    )

    def _dispatch_must_not_run(_repo_root, _manifest, _results, _settings, _requests):
        raise AssertionError("dispatch_sessions must not run while a prior worker is still alive")

    fake_host(worker_launch=FakeWorkerLauncher([_dispatch_must_not_run]))

    result = app.dispatch_rework()

    assert result.ok is False
    assert kill_calls == []
    assert [kind for kind, _p in events] == ["superseded_worker_reap_failed"]

    state = load_state(paths.state_file)
    entry = state["issues"]["123"]
    assert entry["status"] == "rework_requested"
    assert entry.get("redispatch_at") is None
    assert len(entry.get("blocked_environment_at", [])) == 1
    assert any(
        e["kind"] == "rework_dispatch_blocked_environment"
        and e["payload"].get("failure_kind") == "prior_worker_still_alive"
        for e in state["events"]
    )


def test_dispatch_rework_prior_worker_block_escalates_at_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    """At the blocked-environment cap, a still-unreaped prior worker
    escalates with ``dispatch_blocked_environment`` — same lane as
    worktree_foreign_writer — instead of retrying forever."""
    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; sys.exit(1)")),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
    )
    app, fake_gh, paths = _rework_app(tmp_path, config)

    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"]["blocked_environment_at"] = _blocked_env_timestamps(2)
        save_state(paths.state_file, state)

    alive = {PRIOR_PID: True}
    kill_calls: list[tuple[int, float | None]] = []
    events: list[tuple[str, dict]] = []
    _patch_reap_internals(
        monkeypatch,
        workers=[_worker_view(123, PRIOR_PID, process_start_time=None)],
        alive=alive,
        kill_calls=kill_calls,
        events=events,
    )
    fake_host(
        worker_launch=FakeWorkerLauncher([lambda *a, **k: pytest.fail("dispatch must not run")])
    )

    app.dispatch_rework()

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "dispatch_blocked_environment"
    assert (123, config.labels.operator_queue) in fake_gh.labels_added


def test_dispatch_rework_mixed_batch_launches_clean_and_blocks_survivor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    """PR #1926 review, impl level: in one pass with two rework candidates,
    the issue whose prior worker survives its reap is refused BEFORE launch
    while the clean issue still reaches dispatch_sessions — the wrapper's
    blocked result must flow through the same blocked-environment
    accounting as any pre-launch failure (claim released to
    ``rework_requested``, ``blocked_environment_at`` accrued, redispatch_at
    untouched)."""
    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; sys.exit(1)")),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    _seed_two_rework_issues(paths, config, rescue_issue_numbers=set())
    app = OrchestratorApp(tmp_path, paths, config, _TwoReworkIssuesGitHub())

    # Issue 123's prior worker is live but carries no fingerprint — the
    # unfingerprinted-survivor shape. Issue 124 has no recorded worker.
    alive = {PRIOR_PID: True}
    kill_calls: list[tuple[int, float | None]] = []
    events: list[tuple[str, dict]] = []
    _patch_reap_internals(
        monkeypatch,
        workers=[_worker_view(123, PRIOR_PID, process_start_time=None)],
        alive=alive,
        kill_calls=kill_calls,
        events=events,
    )

    dispatch_calls: list[list[int]] = []

    def _recording_dispatch(_repo_root, _manifest, _results, settings, requests):
        dispatch_calls.append([r.issue_number for r in requests])
        return [
            SessionDispatchResult(
                issue_number=r.issue_number,
                issue_title=r.issue_title,
                prompt_path=str(r.prompt_path),
                branch_name=r.branch_name,
                adapter=settings.adapter,
                ok=True,
                pid=99999,
                process_start_time=2.0,
            )
            for r in requests
        ]

    fake_host(worker_launch=FakeWorkerLauncher([_recording_dispatch]))

    result = app.dispatch_rework()

    # Only the clean request reached the launch path.
    assert dispatch_calls == [[124]]
    assert result.ok is False  # issue 123's synthetic failure failed the pass

    state = load_state(paths.state_file)
    assert state["issues"]["124"]["status"] == "dispatched"
    entry123 = state["issues"]["123"]
    assert entry123["status"] == "rework_requested"
    assert len(entry123.get("blocked_environment_at", [])) == 1
    assert entry123.get("redispatch_at") is None
    assert any(
        e["kind"] == "rework_dispatch_blocked_environment"
        and e["payload"].get("issue_number") == 123
        and e["payload"].get("failure_kind") == "prior_worker_still_alive"
        for e in state["events"]
    )


# ---------------------------------------------------------------------------
# _local_dispatch_rework — the no-remote lane gets the same protection
# ---------------------------------------------------------------------------


def test_local_dispatch_rework_blocks_when_prior_worker_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    """The local/no-remote rework lane shares the launch-trigger reap: a
    live unfingerprinted prior worker blocks the replacement launch and the
    claim releases back to ``rework_requested``."""
    config = OrchestratorConfig(worker=WorkerRoleConfig(harness="command"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["7"] = {
            "number": 7,
            "title": "Local issue",
            "status": "rework_requested",
        }
        state["prs"]["7"] = {
            "local": True,
            "issue_number": 7,
            "branch": "agent/issue-7-x",
            "title": "Local rework",
        }
        save_state(paths.state_file, state)
    pr_dir = paths.prs / "pr-7"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "rework-prompt.md").write_text("Fix it", encoding="utf-8")

    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    alive = {PRIOR_PID: True}
    kill_calls: list[tuple[int, float | None]] = []
    events: list[tuple[str, dict]] = []
    _patch_reap_internals(
        monkeypatch,
        workers=[_worker_view(7, PRIOR_PID, process_start_time=None)],
        alive=alive,
        kill_calls=kill_calls,
        events=events,
    )
    fake_host(
        worker_launch=FakeWorkerLauncher([lambda *a, **k: pytest.fail("dispatch must not run")])
    )

    result = app._local_dispatch_rework()

    assert result["dispatched"] == []
    assert result["failed"] and result["failed"][0]["issue"] == 7
    assert "prior worker still alive" in result["failed"][0]["error"]
    state = load_state(paths.state_file)
    assert state["issues"]["7"]["status"] == "rework_requested"


# ---------------------------------------------------------------------------
# Single-enforcement-point guard — the two lanes must not drift apart
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parents[1]
_LANE_PATHS = (
    _REPO_ROOT / "src" / "charlie_work" / "orchestration" / "state_dispatch_rework.py",
    _REPO_ROOT / "src" / "charlie_work" / "orchestration" / "local_lanes.py",
)


def test_lanes_share_the_single_reap_helper_and_named_kind() -> None:
    """PR #1926 review: the reap-then-synthetic-failure block was duplicated
    across the remote and local rework lanes. Both lanes must delegate to
    ``_reap_superseded_workers_for_launch`` and neither may carry the
    failure-kind literal — the kind is spelled once, as
    ``PRIOR_WORKER_STILL_ALIVE_FAILURE_KIND`` in config.py."""
    from charlie_work.config import (
        PRE_LAUNCH_BLOCKED_ENVIRONMENT_FAILURE_KINDS,
        PRIOR_WORKER_STILL_ALIVE_FAILURE_KIND,
    )

    for lane_path in _LANE_PATHS:
        source = lane_path.read_text(encoding="utf-8")
        assert "prior_worker_still_alive" not in source, (
            f"{lane_path.name} spells the failure kind literally — use "
            "PRIOR_WORKER_STILL_ALIVE_FAILURE_KIND via the shared helper"
        )
        assert "_reap_superseded_workers_for_launch" in source, (
            f"{lane_path.name} does not call the shared reap helper — the "
            "two lanes must share the single enforcement point"
        )

    assert PRIOR_WORKER_STILL_ALIVE_FAILURE_KIND in PRE_LAUNCH_BLOCKED_ENVIRONMENT_FAILURE_KINDS
