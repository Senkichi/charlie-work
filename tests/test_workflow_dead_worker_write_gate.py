"""New WriteGate-conversion coverage for the dead-worker cluster in
workflow.py (issue #1264, W6 PR3) and the issue #1311 regression it fixes.

Three per-function tests prove each of the three ``_loop_body`` call sites
named by issue #1311 -- ``_sweep_orphan_processes_for_dead_sessions``,
``_detect_and_handle_orphaned_workers``, and
``_classify_dead_sessions_and_update_throttle_state`` -- actually gates the
write primitive(s) it wraps under ``dry_run=True``: not merely that the
function's return value looks unchanged, but that the underlying
``charlie_work.write_gate`` primitives (and, for the orphan sweep, the real
``subprocess.run`` call the R6a ``kill_process`` design ultimately reaches)
are never invoked, and that on-disk ``state.json`` is byte-identical before
and after. Each pairs a ``dry_run=False`` positive control -- run in a
separate directory, BEFORE any monkeypatch touches the gated primitive --
proving the corresponding ``events.db`` assertion is decisive, not silently
broken (per this repo's "control must be positive by evidence" discipline;
see ``tests/test_stalled_review_reap_write_gate.py``, whose exact idiom this
file mirrors for W6 PR2's stalled-review cluster).

The orphan-sweep test is the one that must assert *zero kills*: it patches
``subprocess.run`` itself (not just ``write_gate.kill_orphan_pid``), so the
assertion proves the whole chain -- including the real, unswapped
``process_utils.kill_orphan_pid`` primitive R6a wraps -- never reaches the
OS kill call. That is R6a's entire payoff and half of #1311's stated
requirement (the other half being the state/event writes the other two
tests cover).

A fourth, loop-level test is the #1311 regression test proper: before this
PR, ``_loop_body``'s three call sites to these functions were unconditional
-- they held no ``write_gate`` parameter at all, so nothing could gate them
regardless of ``self.dry_run``. This test drives the REAL entry point,
``OrchestratorApp.loop()``, with ``dry_run=True``, using a fixture that
reaches a real write in the classify-and-throttle lane (mirroring
``test_charlie_work.py``'s ``test_loop_classifies_dead_sessions_and_sets_
throttle_state``, which is not vacuous -- it is the loop-path acceptance
test for this exact lane) and asserts on-disk state is untouched, then pairs
it with a wiring-level assertion that all three call sites receive the same
``app.write_gate`` instance (proving the fix threads ONE real gate through
all three sites named in #1311, not merely that this one reachable branch
happens not to write).
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from _fakes_github import FakeGitHub
from _sessions_db_fixtures import make_sessions_db
from charlie_work.config import (
    AutoMergeConfig,
    DeescalationConfig,
    DevinConfig,
    OrchestratorConfig,
    PostMortemConfig,
    WorktreeReclamationConfig,
)
from charlie_work.devin_shell import SessionRecord
from charlie_work.instrumentation import event_counts_by_kind
from charlie_work.paths import runtime_paths
from charlie_work.state import empty_state, load_state, save_state
from charlie_work.workflow import (
    OrchestratorApp,
    _classify_dead_sessions_and_update_throttle_state,
    _detect_and_handle_orphaned_workers,
    _sweep_orphan_processes_for_dead_sessions,
)
from charlie_work.write_gate import WriteGate


def _wg(state_file: Path, *, dry_run: bool = False) -> WriteGate:
    return WriteGate(dry_run=dry_run, state_path=state_file, repo="charlie-work")


# ---------------------------------------------------------------------------
# 1. _sweep_orphan_processes_for_dead_sessions -- R6a's whole payoff: an
#    orphan PID that WOULD be killed for real must not be, under dry_run=True.
# ---------------------------------------------------------------------------


def _seed_orphan_sweep_scenario(root: Path) -> tuple[Path, Path, str]:
    sessions_dir = root / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    state_file = root / "state.json"
    save_state(state_file, empty_state())
    worktree_path = str(root / "dead-worktree")
    # A live pid (54321) is required here, NOT None: read_session_records's
    # "is this session dead" check for this function's purposes is
    # `record.pid is None or record.error is not None: continue` -- a
    # session with no pid at all is skipped entirely, not treated as dead.
    # Liveness is instead determined by mocking devin_shell.is_session_alive
    # to return False for this pid (matching test_charlie_work.py's
    # test_sweep_orphan_processes_for_dead_sessions_unit fixture).
    record = SessionRecord(
        issue_number=900,
        branch="agent/issue-900",
        worktree_path=worktree_path,
        prompt_path="/fake/prompt.md",
        command=("devin", "--print"),
        pid=54321,
        started_at=datetime.now(UTC).isoformat(),
        log_path="/fake/log",
        error=None,
        process_start_time=1.0,
    )
    (sessions_dir / "issue-900.json").write_text(json.dumps(record.to_dict()), encoding="utf-8")
    return sessions_dir, state_file, worktree_path


def _run_recorder(real_run):
    calls: list[list[str]] = []

    def _runner(*args, **kwargs):
        if args and args[0] and args[0][0] == "taskkill":
            calls.append(list(args[0]))
            return MagicMock(returncode=0, stdout="", stderr="")
        return real_run(*args, **kwargs)

    return _runner, calls


def test_sweep_orphan_processes_for_dead_sessions_dry_run_true_kills_nothing(
    tmp_path: Path,
) -> None:
    """R6a / issue #1311: an orphan PID found in a dead session's worktree
    must not actually be killed -- and the ``orphan_processes_killed`` event
    must not be recorded -- when ``write_gate.dry_run`` is True.

    Patches ``subprocess.run`` itself (not just
    ``charlie_work.write_gate.kill_orphan_pid``), so the assertion proves the
    whole chain, including the real ``process_utils.kill_orphan_pid``
    primitive, never reaches the OS kill call. test_write_gate.py's own unit
    tests already prove WriteGate.kill_process's isolated short-circuit; this
    test proves R6a's design actually protects a live orphan-sweep call
    site end-to-end.
    """
    config = OrchestratorConfig(devin=DevinConfig(adapter="devin-shell"))
    real_run = subprocess.run

    # Positive control: dry_run=False, separate directory, before any
    # monkeypatch touches the gated primitive.
    control_root = tmp_path / "control"
    control_sessions_dir, control_state_file, control_worktree = _seed_orphan_sweep_scenario(
        control_root
    )

    def _control_sweep(worktree_path: str) -> list[dict[str, Any]]:
        if worktree_path == control_worktree:
            return [{"pid": 77777, "name": "python.exe", "command_line": "orphan"}]
        return []

    control_runner, control_calls = _run_recorder(real_run)
    with (
        patch("charlie_work.dead_worker_reap.sweep_orphan_processes", side_effect=_control_sweep),
        patch("charlie_work.workflow.os.name", "nt"),
        patch("charlie_work.devin_shell.is_session_alive", return_value=False),
        patch("subprocess.run", side_effect=control_runner),
    ):
        _sweep_orphan_processes_for_dead_sessions(
            control_sessions_dir,
            control_state_file,
            config,
            write_gate=_wg(control_state_file, dry_run=False),
        )
    assert control_calls, "positive control: dry_run=False must really invoke taskkill"
    control_counts = event_counts_by_kind(control_state_file)
    assert control_counts.get("orphan_processes_killed", 0) >= 1, (
        "positive control: the events.db query itself must be decisive"
    )

    # dry_run=True: identical scenario, poison-pill subprocess.run.
    sessions_dir, state_file, worktree_path = _seed_orphan_sweep_scenario(tmp_path)
    before_bytes = state_file.read_bytes()

    def _dry_sweep(wt: str) -> list[dict[str, Any]]:
        if wt == worktree_path:
            return [{"pid": 88888, "name": "python.exe", "command_line": "orphan"}]
        return []

    dry_runner, dry_calls = _run_recorder(real_run)
    with (
        patch("charlie_work.dead_worker_reap.sweep_orphan_processes", side_effect=_dry_sweep),
        patch("charlie_work.workflow.os.name", "nt"),
        patch("charlie_work.devin_shell.is_session_alive", return_value=False),
        patch("subprocess.run", side_effect=dry_runner),
    ):
        _sweep_orphan_processes_for_dead_sessions(
            sessions_dir, state_file, config, write_gate=_wg(state_file, dry_run=True)
        )

    assert dry_calls == [], "dry_run=True must never reach taskkill for a detected orphan pid"
    assert state_file.read_bytes() == before_bytes
    assert event_counts_by_kind(state_file).get("orphan_processes_killed", 0) == 0, (
        "the orphan-kill event must not reach events.db under dry_run=True"
    )


# ---------------------------------------------------------------------------
# 2. _detect_and_handle_orphaned_workers -- the request_changes/unchanged-
#    head reset branch (issue #207), which flushes through the R5-completed
#    _append_sweep_events batcher and write_gate.save_state.
# ---------------------------------------------------------------------------


def test_detect_and_handle_orphaned_workers_dry_run_true_writes_nothing(tmp_path: Path) -> None:
    """C1.2: a dead worker with a ``request_changes`` review decision and an
    unchanged head normally resets the issue to ``rework_requested`` (issue
    #207) via the R5-completed sweep-events batcher plus a final
    ``write_gate.save_state``. Under ``dry_run=True`` neither must fire, and
    state.json must stay byte-identical -- proving both R5 completion and
    this function's own conversion are real, not merely that the classify
    lane (test 3 below) is gated."""
    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
    )

    def _seed(root: Path) -> Path:
        paths = runtime_paths(root, config.runtime.state_dir)
        state = load_state(paths.state_file)
        state["issues"]["207"] = {
            "status": "dispatched",
            "worker_pid": 99999,  # dead
            "worker_process_start_time": 1234567890.0,
            "dispatched_at": "2024-01-01T00:00:00Z",
        }
        state["prs"]["100"] = {
            "decision": "request_changes",
            "reviewed_head_sha": "abc123",
        }
        save_state(paths.state_file, state)
        # Issue #1362 Stage 1: control-flow reads of the review decision go
        # through the file-first reader now, not state.json's decision
        # field -- the flat file must exist and agree with the fixture above.
        pr_dir = paths.prs / "pr-100"
        pr_dir.mkdir(parents=True, exist_ok=True)
        (pr_dir / "review-decision.json").write_text(
            json.dumps({"decision": "request_changes", "reviewed_head_sha": "abc123"}),
            encoding="utf-8",
        )
        return paths.state_file

    class FakeGitHubForOrphan(FakeGitHub):
        def pr_list(self):
            return [
                {
                    "number": 100,
                    "headRefOid": "abc123",  # unchanged since request_changes
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-207",
                }
            ]

    # Positive control: dry_run=False, separate directory.
    control_root = tmp_path / "control"
    control_state_file = _seed(control_root)
    control_sessions_dir = control_root / "sessions"
    control_sessions_dir.mkdir(parents=True, exist_ok=True)
    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        _detect_and_handle_orphaned_workers(
            control_sessions_dir,
            control_state_file,
            config,
            FakeGitHubForOrphan(),
            write_gate=_wg(control_state_file, dry_run=False),
        )
    control_state = load_state(control_state_file)
    assert control_state["issues"]["207"]["status"] == "rework_requested", (
        "positive control: dry_run=False must really apply the recovery reset"
    )
    control_counts = event_counts_by_kind(control_state_file)
    assert control_counts.get("orphaned_worker_recovered", 0) >= 1

    # dry_run=True: identical scenario.
    state_file = _seed(tmp_path)
    before_bytes = state_file.read_bytes()
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    save_calls: list[tuple[tuple, dict]] = []
    append_calls: list[tuple[tuple, dict]] = []
    with (
        patch("charlie_work.workflow._worker_pid_alive", return_value=False),
        patch(
            "charlie_work.write_gate.save_state",
            lambda *a, **k: save_calls.append((a, k)) or {"BUG": "raw save_state was called"},
        ),
        patch(
            "charlie_work.write_gate.append_event",
            lambda *a, **k: append_calls.append((a, k)) or {"BUG": "raw append_event was called"},
        ),
    ):
        _detect_and_handle_orphaned_workers(
            sessions_dir,
            state_file,
            config,
            FakeGitHubForOrphan(),
            write_gate=_wg(state_file, dry_run=True),
        )

    assert save_calls == []
    assert append_calls == []
    assert state_file.read_bytes() == before_bytes
    assert event_counts_by_kind(state_file).get("orphaned_worker_recovered", 0) == 0


# ---------------------------------------------------------------------------
# 3. _classify_dead_sessions_and_update_throttle_state -- the worker_blocked
#    escalation branch, which is the one reachable path that in a single
#    pass calls all three of write_gate.save_state, .transition, and
#    .append_event (workflow.py ~4381-4398).
# ---------------------------------------------------------------------------


def test_classify_dead_sessions_worker_blocked_escalation_dry_run_true_writes_nothing(
    tmp_path: Path,
) -> None:
    """C1.2, cluster-payoff test: a dead session whose post-mortem shows
    worker_blocked normally escalates the issue -- state.json's status flips
    to "escalated" (write_gate.save_state), the operator-queue/in-progress
    labels swap on GitHub (write_gate.transition), and a
    session_failed_escalated event is recorded (write_gate.append_event) --
    all three inside one state_lock block. Under dry_run=True none of the
    three primitives may fire, GitHub must see zero label writes (proving
    write_gate.transition's own gate, not just the state-side effects), and
    state.json must stay byte-identical.

    Mirrors test_charlie_work.py's
    test_classify_dead_sessions_worker_blocked_escalates_and_suppresses_
    redispatch fixture (no real git worktree needed -- the command adapter
    plus a sessions.db post-mortem row is enough to reach this branch)."""

    def _build(root: Path) -> tuple[OrchestratorConfig, Any, Path, Path]:
        now = datetime.now(UTC)
        worktree_path = str(root / "worktree")
        db_path = root / "sessions.db"
        make_sessions_db(
            db_path,
            session_id="sess-1",
            working_directory=worktree_path,
            created_at=now.isoformat(),
            rows=[
                {
                    "role": "tool",
                    "content": (
                        'Tool blocked: {"decision": "block", "reason": "push-gate hook rejected"}'
                    ),
                    "created_at": now.isoformat(),
                }
            ],
        )
        config = OrchestratorConfig(
            auto_merge=AutoMergeConfig(
                required_checks=("Tests passed", "Lint & Format", "Pre-commit")
            ),
            devin=DevinConfig(
                adapter="command",
                dispatch_command=(sys.executable, "-c", "import sys; print('ok')"),
            ),
            post_mortem=PostMortemConfig(db_path=str(db_path)),
        )
        paths = runtime_paths(root, config.runtime.state_dir)
        paths.state_file.parent.mkdir(parents=True, exist_ok=True)
        # Seed a real (pre-existing) state.json so the dry_run=True run below
        # has a byte-identity baseline to compare against -- otherwise this
        # function's own write_gate.save_state would be the very first
        # attempt to CREATE the file, and a fully-gated pass would correctly
        # leave it never created at all, which read_bytes() cannot compare.
        save_state(paths.state_file, empty_state())
        fake_gh = FakeGitHub()
        fake_gh.issues = [
            {
                "number": 42,
                "title": "Fix search",
                "url": "https://example.test/issues/42",
                "body": "Search is broken",
                "labels": [{"name": config.labels.in_progress}],
            }
        ]
        fake_gh.prs = []

        sessions_dir = root / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        log_path = sessions_dir / "issue-42.log"
        log_path.write_text("some work then silence\n", encoding="utf-8")
        record = SessionRecord(
            issue_number=42,
            branch="agent/issue-42-x",
            worktree_path=worktree_path,
            prompt_path="/tmp/prompt.md",
            command=("devin", "--prompt-file", "/tmp/prompt.md"),
            pid=None,
            started_at=now.isoformat().replace("+00:00", "Z"),
            log_path=str(log_path),
            error=None,
        )
        (sessions_dir / "issue-42.json").write_text(json.dumps(record.to_dict()), encoding="utf-8")
        return config, fake_gh, sessions_dir, paths.state_file

    # Positive control: dry_run=False, separate directory.
    control_root = tmp_path / "control"
    config, control_gh, control_sessions_dir, control_state_file = _build(control_root)
    _classify_dead_sessions_and_update_throttle_state(
        control_sessions_dir,
        control_state_file,
        control_gh,
        config,
        write_gate=_wg(control_state_file, dry_run=False),
    )
    assert (42, config.labels.operator_queue) in control_gh.labels_added, (
        "positive control: dry_run=False must really escalate via write_gate.transition"
    )
    control_state = load_state(control_state_file)
    assert control_state["issues"]["42"]["status"] == "escalated"
    control_counts = event_counts_by_kind(control_state_file)
    assert control_counts.get("session_failed_escalated", 0) >= 1

    # dry_run=True: identical scenario. Deliberately re-binds config from
    # THIS call, not the control call above -- config.post_mortem.db_path is
    # per-root (control_root vs tmp_path), so reusing the control's config
    # here would point post-mortem lookup at the wrong sqlite file and this
    # session's worker_blocked verdict would silently never be found.
    config, fake_gh, sessions_dir, state_file = _build(tmp_path)
    before_bytes = state_file.read_bytes()

    save_calls: list[tuple[tuple, dict]] = []
    append_calls: list[tuple[tuple, dict]] = []
    with (
        patch(
            "charlie_work.write_gate.save_state",
            lambda *a, **k: save_calls.append((a, k)) or {"BUG": "raw save_state was called"},
        ),
        patch(
            "charlie_work.write_gate.append_event",
            lambda *a, **k: append_calls.append((a, k)) or {"BUG": "raw append_event was called"},
        ),
    ):
        _classify_dead_sessions_and_update_throttle_state(
            sessions_dir,
            state_file,
            fake_gh,
            config,
            write_gate=_wg(state_file, dry_run=True),
        )

    assert not fake_gh.labels_added, "write_gate.transition must not touch GitHub under dry_run"
    assert not fake_gh.labels_removed
    assert save_calls == []
    assert append_calls == []
    assert state_file.read_bytes() == before_bytes
    assert event_counts_by_kind(state_file).get("session_failed_escalated", 0) == 0


# ---------------------------------------------------------------------------
# 4. Issue #1311 regression proper: the real _loop_body entry point,
#    app.loop(), with dry_run=True, must leave state.json untouched, and
#    must thread the SAME app.write_gate instance into all three named call
#    sites -- not merely that one reachable branch happens not to write.
# ---------------------------------------------------------------------------


def test_loop_dry_run_true_leaks_no_writes_through_the_three_1311_call_sites(
    tmp_path: Path,
) -> None:
    """Issue #1311: before this PR, _loop_body's calls to
    _classify_dead_sessions_and_update_throttle_state,
    _sweep_orphan_processes_for_dead_sessions, and
    _detect_and_handle_orphaned_workers were unconditional -- none of the
    three took a write_gate parameter at all, so a --dry-run pass still
    classified dead sessions, set throttled_until, killed orphan processes,
    and reset issue state for real.

    This test drives the REAL entry point, OrchestratorApp.loop(), with
    dry_run=True, using the same non-vacuous dead-session-with-rate-limit-
    log fixture as test_charlie_work.py's
    test_loop_classifies_dead_sessions_and_sets_throttle_state (the loop-
    path acceptance test for this exact lane -- NOT an early-return/empty-
    queue fixture, which would let the assertions below pass without ever
    reaching the converted code). It asserts state.json is byte-identical
    across the pass, then separately proves all three call sites receive
    the identical app.write_gate instance -- the wiring #1311 actually
    fixed -- via wraps=real_fn spies, since call-args identity is what
    distinguishes "threaded through" from "coincidentally didn't write this
    particular pass."

    A dry_run=False positive control (same fixture, separate app/tmp_path)
    reruns test_loop_classifies_dead_sessions_and_sets_throttle_state's own
    throttled_until assertion, proving the scenario really does mutate
    state when not gated."""
    from charlie_work.workflow import (
        _classify_dead_sessions_and_update_throttle_state as real_classify,
    )
    from charlie_work.workflow import (
        _detect_and_handle_orphaned_workers as real_detect_orphaned,
    )
    from charlie_work.workflow import (
        _sweep_orphan_processes_for_dead_sessions as real_sweep_orphan,
    )

    def _build_app(root: Path, *, dry_run: bool) -> tuple[OrchestratorApp, Any]:
        config = OrchestratorConfig(
            auto_merge=AutoMergeConfig(
                required_checks=("Tests passed", "Lint & Format", "Pre-commit")
            ),
            devin=DevinConfig(
                adapter="command",
                dispatch_command=(sys.executable, "-c", "import sys; print('ok')"),
            ),
            # Issue #1311 scoping: _maybe_deescalate_mechanical and
            # _maybe_reclaim_worktrees (both called later in _loop_body,
            # both unrelated to the three named call sites this test
            # targets) each persist their own cadence/summary state via a
            # RAW save_state/append_event with no dry_run check at all --
            # further, separate instances of the same leak class, found
            # while writing this test (alongside _maybe_reconcile_drift and
            # _maybe_reclaim_superseded_main_ci below, which have no config
            # toggle and are no-op'd directly). Disabling both here keeps
            # this test scoped to the three functions issue #1311 and this
            # PR actually named; see this PR's commit body / handoff report
            # for the disclosure covering all four.
            deescalation=DeescalationConfig(enabled=False),
            worktree_reclamation=WorktreeReclamationConfig(enabled=False),
        )
        paths = runtime_paths(root, config.runtime.state_dir)
        paths.state_file.parent.mkdir(parents=True, exist_ok=True)
        # Seed a real state.json baseline (see the analogous comment in
        # test 3's _build) so the dry_run=True pass has something to prove
        # byte-identical against.
        save_state(paths.state_file, empty_state())
        fake_gh = FakeGitHub()
        app = OrchestratorApp(root, paths, config, fake_gh, dry_run=dry_run)

        sessions_dir = root / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        log_path = sessions_dir / "issue-42.log"
        log_path.write_text(
            "Some work done...\n"
            "Error: Reached overall message rate limit. Please try again later. "
            "Your limit will reset in 10 minutes.\n",
            encoding="utf-8",
        )
        record = SessionRecord(
            issue_number=42,
            branch="agent/issue-42-x",
            worktree_path="/tmp/worktree",
            prompt_path="/tmp/prompt.md",
            command=("devin", "--prompt-file", "/tmp/prompt.md"),
            pid=None,
            started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            log_path=str(log_path),
            error=None,
        )
        (sessions_dir / "issue-42.json").write_text(json.dumps(record.to_dict()), encoding="utf-8")
        return app, paths

    # Positive control: dry_run=False, separate directory -- reruns
    # test_charlie_work.py's own acceptance assertion for this lane.
    control_root = tmp_path / "control"
    control_app, control_paths = _build_app(control_root, dry_run=False)
    frozen_now = datetime.now(UTC) + timedelta(hours=1)
    control_app.loop(limit=0, now=frozen_now)
    control_state = load_state(control_paths.state_file)
    assert control_state.get("throttled_until") is not None, (
        "positive control: dry_run=False must really set throttled_until"
    )

    # dry_run=True: identical scenario through the real entry point.
    app, paths = _build_app(tmp_path, dry_run=True)
    before_bytes = paths.state_file.read_bytes()

    with (
        patch(
            "charlie_work.workflow._classify_dead_sessions_and_update_throttle_state",
            wraps=real_classify,
        ) as mock_classify,
        patch(
            "charlie_work.workflow._sweep_orphan_processes_for_dead_sessions",
            wraps=real_sweep_orphan,
        ) as mock_sweep,
        patch(
            "charlie_work.workflow._detect_and_handle_orphaned_workers",
            wraps=real_detect_orphaned,
        ) as mock_detect,
        # Issue #1311 scoping (see the deescalation comment in _build_app
        # above): _maybe_reconcile_drift and _maybe_reclaim_superseded_
        # main_ci -- also called unconditionally from _loop_body, also
        # unrelated to the three call sites this PR converted -- were found
        # via this test to carry the SAME unguarded-write defect (raw
        # append_event/save_state calls with no dry_run check at all:
        # reconcile_pass_completed, main_ci_reclaim_failed,
        # worktrees_reclaimed). Neither has a config toggle like
        # deescalation.enabled, so they are no-op'd directly here to keep
        # this test's byte-identity assertion scoped to the three functions
        # #1311 and this PR actually name -- not because their own leak is
        # addressed. See this PR's commit body / handoff report for the
        # disclosure covering them.
        patch.object(app, "_maybe_reconcile_drift", lambda **_k: None),
        patch.object(app, "_maybe_reclaim_superseded_main_ci", lambda: None),
    ):
        app.loop(limit=0, now=frozen_now)

    # Wiring proof: #1311's actual bug was these three call sites carrying no
    # write_gate at all. Assert each received the SAME gate instance the app
    # constructed for itself, with dry_run=True -- not a coincidentally-
    # unreached call, and not a second, independently-constructed gate.
    for mock_fn, label in (
        (mock_classify, "_classify_dead_sessions_and_update_throttle_state"),
        (mock_sweep, "_sweep_orphan_processes_for_dead_sessions"),
        (mock_detect, "_detect_and_handle_orphaned_workers"),
    ):
        mock_fn.assert_called_once()
        gate = mock_fn.call_args.kwargs.get("write_gate")
        assert gate is app.write_gate, f"{label} must receive app.write_gate itself"
        assert gate is not None and gate.dry_run is True, (
            f"{label}'s write_gate must carry dry_run=True from the app"
        )

    # Behavioral proof for the one lane this fixture actually reaches: no
    # write landed on disk.
    assert paths.state_file.read_bytes() == before_bytes
    state = load_state(paths.state_file)
    assert state.get("throttled_until") is None, (
        "issue #1311: dry_run=True must not leak a real throttled_until write"
    )
