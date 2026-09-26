"""Superseded-worker reap at the rework launch trigger (issue #1494).

Every producer of ``rework_requested`` — the janitor-gate/worktree-rescue
stall route this issue covers, the verdict reconcile, the stranded-commit
restore — flips the issue status without checking the prior worker's
liveness, and ``_check_worktree_writer_marker`` deliberately exempts a
marker owned by one of our own live sessions. A still-running prior worker
therefore admitted its replacement into the same worktree (#1337 ran two
workers against one worktree this way).

The fix enforces at the launch trigger: ``_reap_superseded_workers``
(superseded_worker_reap.py) kills every recorded prior-worker pid for the issue
via the fingerprinted ``write_gate.kill_process_tree`` before
``_dispatch_rework_impl`` / ``_local_dispatch_rework`` call
``dispatch_sessions``. A pid that survives (or is live with no
process-start fingerprint) blocks the launch as a
``prior_worker_still_alive`` blocked-environment failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import _blocked_env_timestamps
from charlie_work.adapters import SessionDispatchResult
from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.superseded_worker_reap import _reap_superseded_workers
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.worker import WorkerView
from charlie_work.workflow import OrchestratorApp
from charlie_work.write_gate import WriteGate


PRIOR_PID = 48840
PRIOR_START = 1234567890.0


def _worker_view(
    issue_number: int,
    pid: int | None,
    *,
    process_start_time: float | None = PRIOR_START,
    error: str | None = None,
    worktree_path: str = "",
    session_id: str | None = "prior-session",
) -> WorkerView:
    return WorkerView(
        adapter_kind="devin",
        issue_number=issue_number,
        repo_key="",
        pid=pid,
        started_at="2026-01-01T00:00:00Z",
        process_start_time=process_start_time,
        log_path="",
        worktree_path=worktree_path,
        error=error,
        failure_kind=None,
        reclaimed=None,
        session_id=session_id,
    )


def _patch_reap_internals(
    monkeypatch: pytest.MonkeyPatch,
    *,
    workers: list[WorkerView],
    alive: dict[int, bool],
    kill_calls: list[tuple[int, float | None]],
    events: list[tuple[str, dict]],
    orphans: list[dict] | None = None,
    orphan_kill_calls: list[int] | None = None,
) -> None:
    """Patch every seam ``_reap_superseded_workers`` touches.

    ``alive`` maps pid -> liveness; the fake ``kill_process_tree`` flips the
    pid to dead so the post-kill recheck observes the reap. Patching the
    module-level names ``write_gate`` delegates to keeps a real ``WriteGate``
    under test (dry-run short-circuit included).

    The stall detector is no-oped: ``_dispatch_rework_impl`` runs
    ``_detect_and_handle_stalled_sessions`` before candidate selection, and
    a live seeded worker would otherwise be reaped by that lane first —
    leaving the launch-trigger helper nothing to do. Issue #1494 is exactly
    the case the stall lane does NOT cover, so these tests isolate it.
    """

    monkeypatch.setattr(
        "charlie_work.workflow._detect_and_handle_stalled_sessions",
        lambda *a, **k: [],
    )
    monkeypatch.setattr("charlie_work.worker.iter_workers", lambda _sessions_dir: list(workers))
    monkeypatch.setattr(
        "charlie_work.superseded_worker_reap.is_pid_alive",
        lambda pid, _st=None: alive.get(pid, False),
    )

    def _fake_kill_process_tree(pid: int, st: float | None = None) -> list[int]:
        kill_calls.append((pid, st))
        if alive.get(pid, False):
            alive[pid] = False
            return [pid]
        return []

    monkeypatch.setattr("charlie_work.write_gate.kill_process_tree", _fake_kill_process_tree)
    monkeypatch.setattr(
        "charlie_work.write_gate.log_event",
        lambda _state_path, kind, payload, **_kw: events.append((kind, payload)),
    )
    monkeypatch.setattr(
        "charlie_work.superseded_worker_reap.sweep_orphan_processes",
        lambda _wt: list(orphans or []),
    )
    monkeypatch.setattr(
        "charlie_work.write_gate.kill_orphan_pid",
        lambda pid: orphan_kill_calls.append(pid) if orphan_kill_calls is not None else None,
    )


def _gate(tmp_path: Path, *, dry_run: bool = False) -> WriteGate:
    return WriteGate(dry_run=dry_run, state_path=tmp_path / "state.json", repo="test")


# ---------------------------------------------------------------------------
# _reap_superseded_workers — unit coverage
# ---------------------------------------------------------------------------


def test_reap_superseded_workers_kills_live_sidecar_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live sidecar-recorded worker for the issue is killed via the
    fingerprinted process-tree kill, then confirmed dead on recheck."""
    alive = {PRIOR_PID: True}
    kill_calls: list[tuple[int, float | None]] = []
    events: list[tuple[str, dict]] = []
    _patch_reap_internals(
        monkeypatch,
        workers=[_worker_view(123, PRIOR_PID, worktree_path="C:/wt/agent-123")],
        alive=alive,
        kill_calls=kill_calls,
        events=events,
    )

    survivors = _reap_superseded_workers(
        123, {}, tmp_path / "sessions", write_gate=_gate(tmp_path)
    )

    assert survivors == []
    assert kill_calls == [(PRIOR_PID, PRIOR_START)]
    assert [kind for kind, _payload in events] == ["superseded_worker_reaped"]
    assert events[0][1]["issue_number"] == 123
    assert events[0][1]["pid"] == PRIOR_PID


def test_reap_superseded_workers_dead_pid_no_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorded-but-dead (or fingerprint-recycled) pid is already gone —
    no kill, no event, launch proceeds."""
    kill_calls: list[tuple[int, float | None]] = []
    events: list[tuple[str, dict]] = []
    _patch_reap_internals(
        monkeypatch,
        workers=[_worker_view(123, PRIOR_PID)],
        alive={},  # is_pid_alive -> False
        kill_calls=kill_calls,
        events=events,
    )

    survivors = _reap_superseded_workers(
        123, {}, tmp_path / "sessions", write_gate=_gate(tmp_path)
    )

    assert survivors == []
    assert kill_calls == []
    assert events == []


def test_reap_superseded_workers_state_pid_without_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The state-entry fallback covers a prior worker whose sidecar was
    already reaped: ``worker_pid``/``worker_process_start_time`` drive the
    fingerprinted kill."""
    alive = {PRIOR_PID: True}
    kill_calls: list[tuple[int, float | None]] = []
    events: list[tuple[str, dict]] = []
    _patch_reap_internals(
        monkeypatch,
        workers=[],
        alive=alive,
        kill_calls=kill_calls,
        events=events,
    )

    survivors = _reap_superseded_workers(
        123,
        {"worker_pid": PRIOR_PID, "worker_process_start_time": PRIOR_START},
        tmp_path / "sessions",
        write_gate=_gate(tmp_path),
    )

    assert survivors == []
    assert kill_calls == [(PRIOR_PID, PRIOR_START)]
    assert events[0][0] == "superseded_worker_reaped"
    assert events[0][1]["source"] == "state"


def test_reap_superseded_workers_last_known_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``last_known_worker_*`` is the fallback when ``worker_pid`` is absent —
    the same precedence ``_probe_recovery_liveness`` uses."""
    alive = {PRIOR_PID: True}
    kill_calls: list[tuple[int, float | None]] = []
    events: list[tuple[str, dict]] = []
    _patch_reap_internals(
        monkeypatch,
        workers=[],
        alive=alive,
        kill_calls=kill_calls,
        events=events,
    )

    survivors = _reap_superseded_workers(
        123,
        {
            "last_known_worker_pid": PRIOR_PID,
            "last_known_worker_process_start_time": PRIOR_START,
        },
        tmp_path / "sessions",
        write_gate=_gate(tmp_path),
    )

    assert survivors == []
    assert kill_calls == [(PRIOR_PID, PRIOR_START)]


def test_reap_superseded_workers_live_unfingerprinted_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live pid with no process_start_time fingerprint cannot be killed
    safely — kill_process_tree would have nothing to re-verify against —
    so it blocks the launch instead of killing on bare pid liveness."""
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

    survivors = _reap_superseded_workers(
        123, {}, tmp_path / "sessions", write_gate=_gate(tmp_path)
    )

    assert survivors == [PRIOR_PID]
    assert kill_calls == []
    assert [kind for kind, _p in events] == ["superseded_worker_reap_failed"]
    assert "no process_start_time" in events[0][1]["reason"]


def test_reap_superseded_workers_surviving_kill_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fingerprinted pid still alive after the guarded kill attempt is a
    survivor: launch must be refused."""
    alive = {PRIOR_PID: True}
    kill_calls: list[tuple[int, float | None]] = []
    events: list[tuple[str, dict]] = []
    _patch_reap_internals(
        monkeypatch,
        workers=[_worker_view(123, PRIOR_PID)],
        alive=alive,
        kill_calls=kill_calls,
        events=events,
    )
    # Kill attempt runs but the process survives it (refused identity
    # re-verification, or the pid genuinely outlived the signal).
    monkeypatch.setattr(
        "charlie_work.write_gate.kill_process_tree",
        lambda pid, st=None: kill_calls.append((pid, st)) or [],
    )

    survivors = _reap_superseded_workers(
        123, {}, tmp_path / "sessions", write_gate=_gate(tmp_path)
    )

    assert survivors == [PRIOR_PID]
    assert kill_calls == [(PRIOR_PID, PRIOR_START)]
    assert [kind for kind, _p in events] == ["superseded_worker_reap_failed"]


def test_reap_superseded_workers_skips_other_issues_and_errored_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only this issue's non-errored sidecars are candidates: an errored
    sidecar's pid field can carry a diagnostic pid (the foreign/live worker
    a failed launch reported), not this issue's own worker."""
    alive = {PRIOR_PID: True, 999: True, 777: True}
    kill_calls: list[tuple[int, float | None]] = []
    events: list[tuple[str, dict]] = []
    _patch_reap_internals(
        monkeypatch,
        workers=[
            _worker_view(456, 999),  # a different issue's worker — untouched
            _worker_view(123, 777, error="launch failed"),  # diagnostic pid
        ],
        alive=alive,
        kill_calls=kill_calls,
        events=events,
    )

    survivors = _reap_superseded_workers(
        123,
        {"worker_pid": PRIOR_PID, "worker_process_start_time": PRIOR_START},
        tmp_path / "sessions",
        write_gate=_gate(tmp_path),
    )

    assert survivors == []
    assert kill_calls == [(PRIOR_PID, PRIOR_START)]


def test_reap_superseded_workers_dry_run_never_kills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under a dry-run gate the underlying kill primitive is never invoked;
    a live prior worker is reported as a survivor so a real run would block."""
    kill_calls: list[tuple[int, float | None]] = []
    events: list[tuple[str, dict]] = []
    _patch_reap_internals(
        monkeypatch,
        workers=[_worker_view(123, PRIOR_PID)],
        alive={PRIOR_PID: True},
        kill_calls=kill_calls,
        events=events,
    )

    survivors = _reap_superseded_workers(
        123, {}, tmp_path / "sessions", write_gate=_gate(tmp_path, dry_run=True)
    )

    assert survivors == [PRIOR_PID]
    assert kill_calls == []
    # The dry-run gate suppresses event emission too.
    assert events == []


def test_reap_superseded_workers_sweeps_orphans_after_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful tree kill is followed by an orphan sweep of the worker's
    recorded worktree — detached/daemonized children the tree kill leaves
    behind, same sweep the stall lane runs."""
    orphan_kills: list[int] = []
    kill_calls: list[tuple[int, float | None]] = []
    events: list[tuple[str, dict]] = []
    _patch_reap_internals(
        monkeypatch,
        workers=[_worker_view(123, PRIOR_PID, worktree_path="C:/wt/agent-123")],
        alive={PRIOR_PID: True},
        kill_calls=kill_calls,
        events=events,
        orphans=[{"pid": 555, "name": "python.exe", "command_line": "x"}],
        orphan_kill_calls=orphan_kills,
    )

    survivors = _reap_superseded_workers(
        123, {}, tmp_path / "sessions", write_gate=_gate(tmp_path)
    )

    assert survivors == []
    assert orphan_kills == [555]
    assert events[0][1]["orphan_pids"] == [555]


def test_reap_superseded_workers_requires_gate(tmp_path: Path) -> None:
    """The helper is WriteGate-migrated: a missing gate is a TypeError, not
    a silent ungated kill."""
    with pytest.raises(TypeError):
        _reap_superseded_workers(123, {}, tmp_path / "sessions", write_gate=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _dispatch_rework_impl — launch-trigger integration
# ---------------------------------------------------------------------------


def _rework_app(tmp_path: Path, config: OrchestratorConfig):
    """Issue 123 in rework_requested + open PR 456 + rework prompt on disk —
    the standard dispatch_rework candidate fixture."""
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
        }
        save_state(paths.state_file, state)

    fake_gh = ReworkGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    (pr_dir / "rework-prompt.md").write_text("Fix the issues", encoding="utf-8")
    return app, fake_gh, paths


def _ok_dispatch_factory(order: list[str]):
    def _fake(_repo_root, _manifest, _results, _settings, requests):
        order.append("dispatch_sessions")
        return [
            SessionDispatchResult(
                issue_number=request.issue_number,
                issue_title=request.issue_title,
                prompt_path=str(request.prompt_path),
                branch_name=request.branch_name,
                adapter="command",
                ok=True,
                pid=99999,
                process_start_time=2.0,
            )
            for request in requests
        ]

    return _fake


def test_dispatch_rework_reaps_live_prior_worker_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", _ok_dispatch_factory(order))

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", _ok_dispatch_factory(order))

    result = app.dispatch_rework()

    assert result.ok is True
    assert kill_calls == [(PRIOR_PID, PRIOR_START)]
    assert order == ["kill_process_tree", "dispatch_sessions"]
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatched"


def test_dispatch_rework_blocks_launch_when_prior_worker_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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

    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", _dispatch_must_not_run)

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    monkeypatch.setattr(
        "charlie_work.workflow.dispatch_sessions",
        lambda *a, **k: pytest.fail("dispatch must not run"),
    )

    app.dispatch_rework()

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "dispatch_blocked_environment"
    assert (123, config.labels.operator_queue) in fake_gh.labels_added


# ---------------------------------------------------------------------------
# _local_dispatch_rework — the no-remote lane gets the same protection
# ---------------------------------------------------------------------------


def test_local_dispatch_rework_blocks_when_prior_worker_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    monkeypatch.setattr(
        "charlie_work.workflow.dispatch_sessions",
        lambda *a, **k: pytest.fail("dispatch must not run"),
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
