"""Loop-pass dead-session classification, launch-failure reap, and stalled-session sweep.

Split out of ``tests/test_charlie_work.py`` (issue #1553,
Track-1 wave 7/8).
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from _fakes_github import FakeGitHub
from charlie_work.config import (
    DevinConfig,
    NotifyConfig,
    OrchestratorConfig,
    ReviewDispatchConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_loop_dead_session_notifies_when_watchdog_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #706: when ``watchdog.enabled=False``, ``_detect_stalled_sessions``
    returns ``[]`` (it is gated on watchdog), so the stalled-session notify path
    never fires. But dead workers ARE still reaped by
    ``_classify_dead_sessions_and_update_throttle_state`` (which is NOT
    watchdog-gated). The loop must feed those reaped dead-session transitions
    into the notify digest so an operator monitoring
    ``notify/digest.jsonl`` gets signal instead of a permanently empty file.
    """
    from charlie_work.devin_shell import SessionRecord, _sidecar_path as devin_sidecar_path

    issue_number = 706
    config = OrchestratorConfig(
        watchdog=WatchdogConfig(
            enabled=False,
            stall_minutes=20,
            # Pin to 0 so the dead-session lane reaps immediately rather than
            # deferring for max_inconclusive_probe_deferrals passes (a bare
            # test environment has no real sessions.db, so the probe is always
            # inconclusive).
            max_inconclusive_probe_deferrals=0,
        ),
        notify=NotifyConfig(enabled=True, sink="file", file_path=""),
        review_dispatch=ReviewDispatchConfig(enabled=False),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    (paths.root).mkdir(parents=True, exist_ok=True)

    # Seed state so the dead-session lane has a dispatched issue to reap.
    with state_lock(paths.state_file):
        save_state(
            paths.state_file,
            {
                "version": 1,
                "issues": {
                    str(issue_number): {
                        "status": "dispatched",
                        "branch_name": f"agent/issue-{issue_number}-test",
                        "worker_pid": 99999,
                        "worker_process_start_time": 1234567890.0,
                    }
                },
                "prs": {},
                "events": [],
            },
        )

    # Create a dead session sidecar (non-existent PID).
    sessions_dir = paths.root / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / f"issue-{issue_number}.log"
    log_path.write_text("Session log\n", encoding="utf-8")
    sidecar_path = devin_sidecar_path(sessions_dir, issue_number)
    record = SessionRecord(
        issue_number=issue_number,
        branch=f"agent/issue-{issue_number}-test",
        worktree_path="/tmp/worktree-706",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=99999,  # Non-existent PID → dead
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # FakeGitHub with the issue present (so the dead-session lane can relabel).
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": issue_number,
            "title": "Test issue 706",
            "url": f"https://example.test/issues/{issue_number}",
            "body": "Test",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    captured: list[Any] = []
    monkeypatch.setattr(
        "charlie_work.workflow.emit_digest",
        lambda notify_config, digest: captured.append(digest),
    )

    result = app.loop(limit=0)

    assert result.ok is True
    # The dead session was reaped and a DEAD transition was emitted to notify.
    dead_digests = [d for d in captured if any(t.health == "DEAD" for t in d.transitions)]
    assert len(dead_digests) == 1, (
        f"Expected exactly one DEAD notify digest, got {len(dead_digests)} (captured: {captured})"
    )
    transition = dead_digests[0].transitions[0]
    assert transition.issue_number == issue_number
    assert transition.health == "DEAD"
    # The sidecar was reaped (proving the dead-session lane ran).
    assert not sidecar_path.exists()


def test_loop_classifies_dead_sessions_and_sets_throttle_state(tmp_path: Path) -> None:
    """Test that loop() classifies dead sessions and sets throttled_until in state.

    This is a loop-path integration test: it constructs the app with a fake adapter,
    simulates a session that died with the rate-limit signature, runs a loop pass,
    then asserts (a) throttled_until is persisted in state and (b) a subsequent
    dispatch() defers launches until it expires.

    The test MUST fail when _classify_dead_sessions_and_update_throttle_state is
    removed from loop() — this is the acceptance test for the exact regression class.
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime, timedelta

    # Use command adapter to avoid needing real devin binary
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit")
        ),
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Ensure state directory exists
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    # Create a sessions directory with a dead session that has a rate-limit log
    # Use the config's sessions_dir path
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Write a session log with rate-limit signature
    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        "Some work done...\n"
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    # Write a session record for a dead session (pid=None to simulate dead)
    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,  # Dead session
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,  # No launch error - exited normally
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # Run a loop pass with limit=0 (no actual dispatch, just the classification logic).
    # `now` is frozen and injected (issue #822's clock seam -- see
    # workflow._classify_dead_sessions_and_update_throttle_state) so the throttle
    # timestamp assertion below is exact instead of racing wall-clock time under
    # CI runner contention. We don't assert result.ok because dispatch may fail
    # with no issues to process -- the key is that the classification logic runs
    # regardless.
    # frozen_now is offset 1 hour into the future (rather than the real instant)
    # so the throttle window below stays open for the dispatch-deferral check
    # further down regardless of how long the process stalls between here and
    # there -- the offset is arbitrary and only needs to exceed any plausible
    # CI stall; it does not affect the exact-equality assertion since both sides
    # derive from this same captured value.
    frozen_now = datetime.now(UTC) + timedelta(hours=1)
    app.loop(limit=0, now=frozen_now)

    # Verify throttled_until was set in state by the loop's classification pass
    state = load_state(paths.state_file)
    assert state.get("throttled_until") is not None

    # Verify the cooldown reflects the parsed 10 minutes plus the resume margin,
    # computed from the same frozen `now` the loop pass was given -- exact
    # equality, no wall-clock tolerance window.
    throttle_time = datetime.fromisoformat(state["throttled_until"].replace("Z", "+00:00"))
    expected_time = (
        frozen_now + timedelta(minutes=10, seconds=config.runtime.throttle_resume_margin_s)
    ).replace(microsecond=0)
    assert throttle_time == expected_time

    # Verify that a subsequent dispatch() defers launches while throttled
    # Add a dispatchable issue
    fake_gh.issues = [
        {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "body": "Search is broken",
            "labels": [{"name": "automated-ready"}],
        }
    ]

    app.gh.prs[0]["state"] = "CLOSED"
    dispatch_result = app.dispatch(limit=1)
    # Dispatch should be deferred due to throttle (ok=False is expected for deferral)
    assert dispatch_result.ok is False
    assert "deferred" in dispatch_result.message.lower()
    # Should defer launch due to throttle
    assert dispatch_result.data["selected_count"] == 0


def test_loop_wires_persist_inconclusive_probe_counter_false_to_dead_lane(
    tmp_path: Path,
) -> None:
    """Issue #343 Finding 2 wiring test: loop() must call the dead lane with
    persist_inconclusive_probe_counter=False.

    The stall lane runs unconditionally at the top of loop() (line ~4100) and is
    the sole writer of the not-alive-worker inconclusive-probe deferral counter
    for that pass. If loop()'s call to _classify_dead_sessions_and_update_throttle_state
    (line ~4119) ever drops the persist_inconclusive_probe_counter=False keyword
    (e.g. reverted to the default True during a rebase), the dead lane would
    silently double-write that counter on top of the stall lane's write within a
    single loop() pass.

    This MUST fail if that call site's persist_inconclusive_probe_counter=False
    keyword is removed or flipped to True -- verified by temporarily reverting the
    call site during development of this test (not asserted here, since a
    mutation test would require editing production code from within a test).

    Deliberately does not assert on the inconclusive-probe counter's persisted
    value: the stall lane itself runs 3x per loop() pass (direct, plus via
    dispatch_rework() and dispatch()'s own internal calls -- a separate,
    pre-existing redundancy tracked outside this issue), which would make an
    end-to-end counter-value assertion through loop() fragile. Pinning the
    call-args of the dead lane directly is the precise, stable way to gate this
    specific wiring.
    """
    from charlie_work.workflow import (
        _classify_dead_sessions_and_update_throttle_state as real_classify_dead_sessions,
    )

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    with patch(
        "charlie_work.workflow._classify_dead_sessions_and_update_throttle_state",
        wraps=real_classify_dead_sessions,
    ) as mock_classify:
        app.loop(limit=0)

    mock_classify.assert_called_once()
    assert mock_classify.call_args.kwargs["persist_inconclusive_probe_counter"] is False


def test_loop_reaps_launch_failure_sidecar_and_reports_reaped(
    tmp_path: Path,
) -> None:
    """Issue #266: loop() reaps launch-failure sidecars (pid=None, error set)
    and reports them in the ``reaped`` section of the pass result.
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit")
        ),
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(sessions_dir / "issue-42.log"),
        error="devin binary not found",
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    result = app.loop(limit=0)

    assert not sidecar_path.exists()
    reaped = result.data.get("reaped", [])
    assert len(reaped) == 1
    assert reaped[0]["issue_number"] == 42
    assert reaped[0]["failure_kind"] == "launch_failed"
    assert reaped[0]["error"] == "devin binary not found"


def test_loop_launch_failure_with_throttle_signature_persists_throttled_until(
    tmp_path: Path,
) -> None:
    """Issue #266 + cross-family finding: a throttle-caused LAUNCH failure must
    persist throttled_until exactly like the dead-session lane.

    A launch-failure sidecar (pid=None, error set) whose log carries the
    rate-limit signature is classified through the same failure classifier;
    discarding its throttled_until would relaunch straight into the throttled
    provider. This test MUST fail if the launch-failure branch drops the
    classifier's throttle window.
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime, timedelta

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit")
        ),
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error="Reached overall message rate limit. Your limit will reset in 10 minutes.",
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # Issue #828: freeze the clock and thread it through loop() so the
    # assertion below can compare against an exact expected instant instead
    # of a wall-clock-tolerance window (a stall between loop() and the
    # assertion previously had ~5s to blow the tolerance and flake CI).
    frozen_now = datetime.now(UTC)
    result = app.loop(limit=0, now=frozen_now)

    # The launch-failure sidecar is reaped and reported
    reaped = result.data.get("reaped", [])
    matching = [entry for entry in reaped if entry["issue_number"] == 42]
    assert len(matching) == 1
    assert not sidecar_path.exists()

    # The throttle window from the classifier is persisted, same as the
    # dead-session lane, including the resume margin.
    state = load_state(paths.state_file)
    assert state.get("throttled_until") is not None
    throttle_time = datetime.fromisoformat(state["throttled_until"].replace("Z", "+00:00"))
    expected_time = (
        frozen_now + timedelta(minutes=10, seconds=config.runtime.throttle_resume_margin_s)
    ).replace(microsecond=0)
    assert throttle_time == expected_time


def test_loop_pid_none_no_error_not_classified_as_launch_failed(
    tmp_path: Path,
) -> None:
    """Issue #266: a pid=None + error=None sidecar is a dead session, not a launch failure.

    The launch-failure branch must not fire here; the dead-session branch handles
    it and does not tag it with failure_kind="launch_failed".
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit")
        ),
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(sessions_dir / "issue-42.log"),
        error=None,
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    result = app.loop(limit=0)

    reaped = result.data.get("reaped", [])
    # The record must actually be reaped (dead-session lane owns it) — a bare
    # loop over a possibly-empty list would pass vacuously if the sidecar were
    # skipped entirely, which is the old pin-the-loop-open behavior.
    matching = [entry for entry in reaped if entry["issue_number"] == 42]
    assert len(matching) == 1
    assert matching[0]["failure_kind"] != "launch_failed"


def test_loop_reaps_stalled_session_with_no_candidates(tmp_path: Path) -> None:
    """Test that loop() reaps stalled sessions even with zero ready/rework candidates (issue #165)."""
    from datetime import UTC, datetime, timedelta
    from charlie_work.devin_shell import SessionRecord

    config = OrchestratorConfig(
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Issue #1325: dry_run=True suppresses sidecar writes (failure_kind stays
    # None), so this test must use dry_run=False to verify the sidecar is
    # actually classified. The stalled PID (99999) does not exist, so
    # kill_process_tree is a no-op — no real process is harmed.
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=False)

    # Create a session record for issue 123 with a live PID and stale log
    sessions_dir = app._layout.sessions_dir
    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_file = sessions_dir / "issue-123.json"
    log_file = sessions_dir / "issue-123.log"

    # Write a log file with old mtime (stalled)
    log_file.write_text("working on issue\nmaking progress\n", encoding="utf-8")
    old_time = datetime.now(UTC) - timedelta(minutes=25)
    timestamp = old_time.timestamp()
    os.utime(log_file, (timestamp, timestamp))

    # Create a session record with a fake PID
    session_record = SessionRecord(
        issue_number=123,
        branch="agent/issue-123-fix-search",
        worktree_path=str(tmp_path / "worktrees" / "agent-123"),
        prompt_path=str(
            tmp_path / ".var" / "charlie-work" / "issues" / "issue-123" / "worker-prompt.md"
        ),
        command=("devin", "--prompt-file", "{prompt_path}"),
        pid=99999,
        started_at=datetime.now(UTC).isoformat(),
        log_path=str(log_file),
        process_start_time=time.time(),
    )
    session_file.write_text(json.dumps(session_record.to_dict()), encoding="utf-8")

    # Ensure zero ready issues and zero rework candidates
    fake_gh.issues = []
    fake_gh.prs = []

    # Mock the liveness check to return True (simulating a live but stalled process)
    with patch("charlie_work.worker.is_session_alive", return_value=True):
        result = app.loop()

    # The loop should complete and the stalled session should be reaped
    assert result.ok is True
    # Verify the session file was marked with failure_kind: stalled
    updated_session = json.loads(session_file.read_text(encoding="utf-8"))
    assert updated_session.get("failure_kind") == "stalled"


def test_loop_advances_inconclusive_probe_deferral_counter_once_per_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One loop() pass advances the Signal-1 deferral counter at most once per worker.

    Regression test for the issue #343 Finding 2 follow-up: loop() runs the
    stall lane (_detect_and_handle_stalled_sessions) itself, and dispatch()/
    dispatch_rework() each used to re-run it internally — three sweeps per
    pass, each independently incrementing inconclusive_probe_deferred_count
    for a not-alive worker with an inconclusive real-activity probe. That
    collapsed max_inconclusive_probe_deferrals' "N passes of grace" into a
    single pass. loop() now hands its sweep result down to both dispatch
    lanes so the counter is written exactly once per pass.

    The dead-session lane is neutralized here: pre-PR-#352 it reaps any
    not-alive worker outright (deleting the sidecar mid-pass), and post-#352
    it defers with its own counter suppression — either way it is covered by
    its own tests, and this test pins the stall-lane/dispatch-lane interplay
    in isolation so it holds on both sides of that merge.
    """
    from datetime import UTC, datetime
    from charlie_work import workflow as workflow_module
    from charlie_work.devin_shell import SessionRecord
    from charlie_work.post_mortem import ActivitySource, RealActivityProbe

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(
            enabled=True, stall_minutes=20, max_inconclusive_probe_deferrals=10
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = []
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=False)

    sessions_dir = app._layout.sessions_dir
    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_file = sessions_dir / "issue-343.json"
    log_file = sessions_dir / "issue-343.log"
    # Fresh log so Signal 3 (progress staleness) never fires; the counter is
    # driven purely by Signal 1 (not alive) + inconclusive probe.
    log_file.write_text("working on issue\n", encoding="utf-8")

    session_record = SessionRecord(
        issue_number=343,
        branch="agent/issue-343-fix",
        worktree_path=str(tmp_path / "worktrees" / "agent-343"),
        prompt_path=str(
            tmp_path / ".var" / "charlie-work" / "issues" / "issue-343" / "worker-prompt.md"
        ),
        command=("devin", "--prompt-file", "{prompt_path}"),
        pid=99999,
        started_at=datetime.now(UTC).isoformat(),
        log_path=str(log_file),
        process_start_time=time.time(),
    )
    session_file.write_text(json.dumps(session_record.to_dict()), encoding="utf-8")

    def _inconclusive_probe(view: Any, cfg: Any, now: Any) -> RealActivityProbe:
        return RealActivityProbe(
            sources=(
                ActivitySource(
                    name="sessions.db",
                    timestamp=None,
                    staleness_seconds=None,
                    error="message_nodes query failed",
                ),
            )
        )

    # Worker process is gone; the probe cannot corroborate either way.
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda record: False)
    monkeypatch.setattr("charlie_work.worker.real_activity_probe_for", _inconclusive_probe)
    # Neutralize the sibling lanes (see docstring) so only the stall-lane
    # sweeps driven by loop()/dispatch()/dispatch_rework() touch the sidecar.
    monkeypatch.setattr(
        workflow_module,
        "_classify_dead_sessions_and_update_throttle_state",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        workflow_module,
        "_sweep_orphan_processes_for_dead_sessions",
        lambda *args, **kwargs: None,
    )

    result = app.loop(limit=0)
    assert result.ok is True

    sidecar = json.loads(session_file.read_text(encoding="utf-8"))
    assert sidecar.get("inconclusive_probe_deferred_count") == 1
    assert sidecar.get("failure_kind") is None

    # Cross-pass accumulation still works: a second pass advances it once more.
    result = app.loop(limit=0)
    assert result.ok is True
    sidecar = json.loads(session_file.read_text(encoding="utf-8"))
    assert sidecar.get("inconclusive_probe_deferred_count") == 2
    assert sidecar.get("failure_kind") is None
