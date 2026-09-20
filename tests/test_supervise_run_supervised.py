"""``run_supervised`` loop mechanics: cadence, locking, pass summary, exit edges.

Split out of ``tests/test_supervise.py`` (issue #1562, Track 1) --
bodies are verbatim relocations; shared helpers live in
``tests/_supervise_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from _supervise_fixtures import FakeApp, FakeClock, _active_result, _drained_result
from charlie_work.config import SupervisorConfig
from charlie_work.instrumentation import query_events
from charlie_work.supervise import run_supervised, try_acquire_supervisor_lock
from charlie_work.workflow import CommandResult


def test_run_supervised_exits_with_launch_failure_sidecar(tmp_path: Path) -> None:
    """Issue #266: loop exits when only a launch-failure sidecar is present."""
    import json
    from charlie_work.devin_shell import SessionRecord
    from charlie_work.config import SupervisorConfig

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    sidecar = sessions / "issue-1.json"
    sidecar.write_text(
        json.dumps(
            SessionRecord(
                issue_number=1,
                branch="agent/issue-1-x",
                worktree_path="/tmp/worktree",
                prompt_path="/tmp/prompt.md",
                command=("devin",),
                pid=None,
                started_at="2024-01-01T00:00:00Z",
                log_path="/tmp/issue-1.log",
                error="devin binary not found",
            ).to_dict()
        ),
        encoding="utf-8",
    )

    cfg = SupervisorConfig(
        full_pass_interval_seconds=999,
        active_cooldown_seconds=0,
        max_runtime_minutes=999,
    )
    app = FakeApp(tmp_path, results=[], supervisor_cfg=cfg)
    app._sessions_dir = sessions

    def _remove_sidecar_and_drain(limit: Any = None, *, merge: Any = None) -> CommandResult:
        sidecar.unlink(missing_ok=True)
        return _drained_result()

    app.loop = _remove_sidecar_and_drain

    result = run_supervised(
        app,
        sleep=FakeClock().sleep,
        clock=FakeClock().now,
        max_passes=2,
    )
    assert result.ok is True
    assert not sidecar.exists()


def test_pass_summary_reports_zero_merged_for_all_failed_attempts(
    tmp_path: Path, capsys: Any
) -> None:
    """Regression: today's production run printed "merged 3" for a pass where
    all 3 merge attempts had can_merge=False and zero PRs actually merged.
    The summary line must report the real (successful) count -- with the
    attempt count surfaced (0/3) since it diverges from successes.
    """
    app = FakeApp(tmp_path, [_active_result(merge_failed=3, open_prs=3)])
    fc = FakeClock(auto_advance=1.0)
    run_supervised(app, clock=fc.now, sleep=fc.sleep, max_passes=1)

    out = capsys.readouterr().out
    assert "merged 0/3" in out
    assert "merged 3" not in out.replace("merged 0/3", "")


def test_pass_summary_reports_plain_count_when_all_attempts_succeed(
    tmp_path: Path, capsys: Any
) -> None:
    """When attempts == successes, the line stays in the compact plain form
    ("merged N"), not "N/N" -- keeps the common case stable for callers/tests
    that grep the plain form.
    """
    app = FakeApp(tmp_path, [_active_result(merged=2)])
    fc = FakeClock(auto_advance=1.0)
    run_supervised(app, clock=fc.now, sleep=fc.sleep, max_passes=1)

    out = capsys.readouterr().out
    assert "merged 2" in out
    assert "merged 2/2" not in out


def test_pass_summary_reports_warnings_count(tmp_path: Path, capsys: Any) -> None:
    """Issue #254: the summary line counts pass warnings (e.g. merge alarms)."""
    app = FakeApp(
        tmp_path,
        [_active_result(warnings=["PR #456 approved but unmergeable for 3 passes"])],
    )
    fc = FakeClock(auto_advance=1.0)
    run_supervised(app, clock=fc.now, sleep=fc.sleep, max_passes=1)

    out = capsys.readouterr().out
    assert "warnings 1" in out


def test_run_supervised_ensures_labels_once_at_startup(tmp_path: Path) -> None:
    """Issue #1339: run_supervised runs the LabelConfig-derived label ensure
    exactly once at startup (not per pass), so a new LabelConfig field
    converges to its label with no operator action.
    """
    app = FakeApp(tmp_path, [_drained_result(), _drained_result()])
    fc = FakeClock()
    run_supervised(app, clock=fc.now, sleep=fc.sleep, max_passes=2)

    assert app.ensure_labels_calls == 1


def test_run_supervised_ensure_labels_failure_does_not_block(tmp_path: Path) -> None:
    """Issue #1339 AC #2: a label-ensure failure must not block the supervisor."""
    app = FakeApp(tmp_path, [_drained_result()])
    ensure_calls: list[int] = []

    def _raising_ensure() -> CommandResult:
        ensure_calls.append(1)
        raise RuntimeError("boom")

    app.ensure_labels = _raising_ensure  # type: ignore[assignment]
    fc = FakeClock()

    # Must not raise; the supervisor proceeds to its loop.
    result = run_supervised(app, clock=fc.now, sleep=fc.sleep, max_passes=1)
    assert result.ok is True
    # The ensure was actually invoked (not skipped), and its failure was caught.
    assert ensure_calls == [1], ensure_calls


def test_run_supervised_exits_when_drained_first_pass(tmp_path: Path) -> None:
    """First pass drains everything → loop exits immediately."""
    app = FakeApp(tmp_path, [_drained_result()])
    fc = FakeClock()
    result = run_supervised(
        app,
        clock=fc.now,
        sleep=fc.sleep,
        max_passes=5,
    )
    assert result.ok is True
    assert app._call_count == 1


def test_run_supervised_records_ci_fleet_provenance(tmp_path: Path) -> None:
    """Issue #954: run_supervised records ci_fleet provenance to events.db.

    Mirrors ``test_run_fleet_supervise_records_ci_fleet_provenance``: the
    per-repo supervisor stamps ``ci_fleet.__file__`` plus the sibling repo's
    HEAD/branch/dirty-state into its ``events.db`` at every start, so the
    editable-working-tree coupling is attributable rather than silent. The
    event is recorded before the supervisor lock and loop, so it lands even
    on a drained single-pass run.

    Review finding for #954: this event-write had no regression test -- only
    a fixture attribute (``_FakePaths.state_file``) was patched to stop
    existing tests from breaking. This test asserts the event is actually
    written with the expected payload via ``query_events``.
    """
    app = FakeApp(tmp_path, [_drained_result()])
    fc = FakeClock()
    result = run_supervised(app, clock=fc.now, sleep=fc.sleep, max_passes=1)
    assert result.ok is True

    rows = query_events(app.paths.state_file, kind="ci_fleet_provenance")
    assert rows is not None, "no events.db reader -- the event was not recorded"
    assert len(rows) == 1, f"expected exactly one ci_fleet_provenance event, got {len(rows)}"
    payload = rows[0]["payload"]
    # ci_fleet is importable in this venv, so __file__ is always set.
    assert payload["ci_fleet_file"] is not None
    # All six fields from the shared payload helper must be present (None is a
    # valid value for the sibling fields when declared_ci_fleet_sibling_root
    # abstains -- e.g. the published-wheel deployment has no sibling checkout).
    for key in (
        "ci_fleet_file",
        "sibling_root",
        "sibling_head",
        "sibling_branch",
        "sibling_dirty",
        "error",
    ):
        assert key in payload, f"missing field {key!r} in ci_fleet_provenance payload"


def test_run_supervised_infill_freed_slot_triggers_prompt_pass(tmp_path: Path) -> None:
    """A sidecar disappearing (worker exited) triggers a delta → prompt pass
    which dispatches.

    Bounded proof (finding #12): ``full_pass_interval_seconds`` is set far out
    of reach and ``max_passes=2`` caps the run, so pass 2 can ONLY fire because
    ``has_delta()`` detected the vanished sidecar -- not because the fallback
    timer eventually fires (the old assertion, ``call_count >= 2``, would have
    passed even with ``has_delta`` hard-coded to ``False``, since the fallback
    would eventually force a pass).
    """
    # Plant a sidecar that will vanish between first poll and first pass
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    sidecar = sessions_dir / "issue-1.json"
    sidecar.write_text("{}", encoding="utf-8")

    cfg = SupervisorConfig(
        poll_interval_seconds=5,
        full_pass_interval_seconds=100_000,
        active_cooldown_seconds=9,
    )
    # Pass 1: still-live (open_prs=1 keeps loop alive), Pass 2: dispatches 1
    # (dispatched > 0 keeps the loop alive one more cooldown before
    # max_passes=2 stops it -- that trailing sleep is expected and doesn't
    # weaken the bound on how pass 2 itself was triggered).
    results = [
        _active_result(open_prs=1),
        _active_result(dispatched=1),
    ]
    app = FakeApp(tmp_path, results, supervisor_cfg=cfg)
    app._sessions_dir = sessions_dir

    fc = FakeClock(auto_advance=1.0)

    def sleeping(seconds: float) -> None:
        fc.sleep(seconds)
        # After first sleep, remove the sidecar to simulate worker exit
        if len(fc.sleep_calls) == 1 and sidecar.exists():
            sidecar.unlink()
        assert len(fc.sleep_calls) <= 2, (
            "pass 2 should have fired after exactly one poll interval via "
            "has_delta(); the fallback timer is set far out of reach so "
            "further sleeps mean delta detection did not fire it"
        )

    result = run_supervised(
        app,
        clock=fc.now,
        sleep=sleeping,
        max_passes=2,
    )
    assert result.ok is True
    assert app._call_count == 2
    # Bound the proof: pass 2 fires after exactly ONE poll-interval sleep
    # (5.0s, not the 100_000s fallback); the trailing active-cooldown sleep
    # (9.0s) follows pass 2 itself, per should_exit keeping the loop alive
    # while dispatched > 0.
    assert fc.sleep_calls == [5.0, 9.0]


def test_run_supervised_verdict_file_triggers_pass_while_live_zero(tmp_path: Path) -> None:
    """A verdict file appearing while live=0 keeps the loop alive and triggers
    a merge pass.

    Bounded proof (finding #12): ``full_pass_interval_seconds`` is set far out
    of reach and ``max_passes=2`` caps the run, so pass 2 can ONLY be
    explained by ``has_delta()`` picking up the new verdict file -- not by the
    fallback timer (the old assertion, ``call_count >= 2``, would have passed
    even with ``has_delta`` hard-coded to ``False``).
    """
    prs_dir = tmp_path / ".var" / "charlie-work" / "prs"
    prs_dir.mkdir(parents=True, exist_ok=True)
    pr_dir = prs_dir / "pr-456"
    pr_dir.mkdir()

    cfg = SupervisorConfig(
        poll_interval_seconds=5,
        full_pass_interval_seconds=100_000,
        active_cooldown_seconds=9,
    )
    # Pass 1: no workers, no dispatched, but open_prs=1 → stay alive
    # Pass 2: merges=1 after verdict written
    results = [
        _active_result(open_prs=1),
        _active_result(merged=1),
    ]
    app = FakeApp(tmp_path, results, supervisor_cfg=cfg)

    fc = FakeClock(auto_advance=1.0)

    def sleeping(seconds: float) -> None:
        fc.sleep(seconds)
        # Write verdict after first sleep
        if len(fc.sleep_calls) == 1:
            verdict = pr_dir / "review-decision.json"
            verdict.write_text('{"decision": "approved"}', encoding="utf-8")
        assert len(fc.sleep_calls) <= 2, (
            "pass 2 should have fired after exactly one poll interval via "
            "has_delta(); the fallback timer is set far out of reach so "
            "further sleeps mean delta detection did not fire it"
        )

    result = run_supervised(
        app,
        clock=fc.now,
        sleep=sleeping,
        max_passes=2,
    )
    assert result.ok is True
    assert app._call_count == 2
    # Bound the proof: pass 2 fires after exactly ONE poll-interval sleep
    # (5.0s, not the 100_000s fallback); the trailing active-cooldown sleep
    # (9.0s) follows pass 2 itself, per should_exit keeping the loop alive
    # while merged > 0.
    assert fc.sleep_calls == [5.0, 9.0]


def test_run_supervised_fallback_timer_fires_with_no_delta(tmp_path: Path) -> None:
    """After a pass with no local-signal delta, the fallback timer still
    forces a subsequent pass once ``full_pass_interval_seconds`` genuinely
    elapses.

    This is distinct from first-pass priming (finding #12): the OLD test only
    proved pass 1 fires, which is trivially true on iteration 1 regardless of
    whether the fallback timer's elapsed-time math works at all (priming sets
    ``last_full_pass_at`` behind "now" specifically to force iteration 1).
    Here the clock is advanced PAST the interval only after pass 1 completes,
    with nothing else changing in sessions/prs, so pass 2 can only be
    explained by the fallback timer noticing real elapsed time.
    """
    cfg = SupervisorConfig(
        full_pass_interval_seconds=10,
        poll_interval_seconds=5,
        active_cooldown_seconds=5,
    )
    results = [_active_result(open_prs=1), _drained_result()]
    app = FakeApp(tmp_path, results, supervisor_cfg=cfg)

    fc = FakeClock(start=0.0, auto_advance=0.0)

    def sleeping(seconds: float) -> None:
        fc.sleep(seconds)
        if len(fc.sleep_calls) == 1:
            # Push the clock past the fallback threshold only AFTER pass 1
            # has completed with no file changes.
            fc.advance(cfg.full_pass_interval_seconds)

    result = run_supervised(
        app,
        clock=fc.now,
        sleep=sleeping,
        max_passes=5,
    )
    assert result.ok is True
    # Pass 1 fires from priming; pass 2 fires from the fallback timer once
    # real elapsed time (not priming) crosses full_pass_interval_seconds.
    assert app._call_count == 2


def test_run_supervised_active_cooldown_sleep_after_dispatch(tmp_path: Path) -> None:
    """After a pass that dispatches, sleep time equals active_cooldown_seconds."""
    cfg = SupervisorConfig(
        poll_interval_seconds=20,
        active_cooldown_seconds=7,
        full_pass_interval_seconds=300,
    )
    # Pass 1: dispatches 1 (stays alive), Pass 2: drained
    results = [_active_result(dispatched=1), _drained_result()]
    app = FakeApp(tmp_path, results, supervisor_cfg=cfg)

    fc = FakeClock(auto_advance=1.0)
    result = run_supervised(
        app,
        clock=fc.now,
        sleep=fc.sleep,
        max_passes=5,
    )
    assert result.ok is True
    # First sleep after dispatching pass should be active_cooldown_seconds (7)
    assert fc.sleep_calls[0] == 7.0


def test_run_supervised_poll_interval_sleep_when_idle(tmp_path: Path) -> None:
    """After a pass with no dispatch/merge, sleep time equals poll_interval_seconds."""
    cfg = SupervisorConfig(
        poll_interval_seconds=15,
        active_cooldown_seconds=7,
        full_pass_interval_seconds=300,
    )
    # Pass 1: open_prs=1 (stay alive), Pass 2: drained
    results = [_active_result(open_prs=1), _drained_result()]
    app = FakeApp(tmp_path, results, supervisor_cfg=cfg)

    fc = FakeClock(auto_advance=1.0)
    result = run_supervised(
        app,
        clock=fc.now,
        sleep=fc.sleep,
        max_passes=5,
    )
    assert result.ok is True
    # First sleep after idle pass (open_prs=1 but no dispatch/merge)
    assert fc.sleep_calls[0] == 15.0


def test_run_supervised_max_passes_exits(tmp_path: Path) -> None:
    """max_passes cap causes exit before draining."""
    # Supply 10 non-draining results
    results = [_active_result(open_prs=1)] * 10
    app = FakeApp(tmp_path, results)
    fc = FakeClock(auto_advance=1.0)
    result = run_supervised(
        app,
        clock=fc.now,
        sleep=fc.sleep,
        max_passes=3,
    )
    assert result.ok is True
    assert app._call_count == 3


def test_run_supervised_max_runtime_exits(tmp_path: Path) -> None:
    """max_runtime_override cap stops the loop after the wall-clock expires."""
    results = [_active_result(open_prs=1)] * 100
    app = FakeApp(tmp_path, results)

    # Clock advances 70 seconds per sleep call (= >1 minute)
    fc = FakeClock(start=0.0, auto_advance=70.0)
    result = run_supervised(
        app,
        clock=fc.now,
        sleep=fc.sleep,
        max_runtime_override=1,  # 1 minute
        max_passes=100,
    )
    assert result.ok is True
    # Should have run fewer than 100 passes
    assert app._call_count < 100


def test_run_supervised_keyboard_interrupt_returns_ok(tmp_path: Path) -> None:
    """KeyboardInterrupt is caught; result is ok=True with summary."""
    call_count = [0]

    class InterruptApp(FakeApp):
        def loop(self, limit: Any = None, *, merge: Any = None) -> CommandResult:
            call_count[0] += 1
            if call_count[0] >= 2:
                raise KeyboardInterrupt
            return _active_result(open_prs=1)

    app = InterruptApp(tmp_path, [])
    fc = FakeClock(auto_advance=1.0)
    result = run_supervised(
        app,
        clock=fc.now,
        sleep=fc.sleep,
        max_passes=10,
    )
    assert result.ok is True
    assert "supervised loop complete" in result.message


def test_run_supervised_exception_returns_ok_false_and_releases_lock(tmp_path: Path) -> None:
    """Regression for finding #3: a raw exception from app.loop() must not
    propagate past run_supervised (errors-as-values invariant) -- it comes
    back as CommandResult(ok=False, ...) with the pass number in the
    message, and the supervisor lock is still released afterward.
    """
    call_count = [0]

    class RaisingApp(FakeApp):
        def loop(self, limit: Any = None, *, merge: Any = None) -> CommandResult:
            call_count[0] += 1
            if call_count[0] >= 2:
                raise RuntimeError("boom")
            return _active_result(open_prs=1)

    app = RaisingApp(tmp_path, [])
    fc = FakeClock(auto_advance=1.0)
    result = run_supervised(
        app,
        clock=fc.now,
        sleep=fc.sleep,
        max_passes=10,
    )
    assert result.ok is False
    assert "pass 2" in result.message
    assert "boom" in result.message

    # Lock must be released even though the loop aborted via exception --
    # a fresh acquire must succeed.
    lock_path = app.paths.root / "supervisor.lock"
    lock = try_acquire_supervisor_lock(lock_path)
    assert lock is not None, "lock should be released after an aborted pass"
    lock.release()


def test_try_acquire_supervisor_lock_zero_byte_existing_file_succeeds(tmp_path: Path) -> None:
    """Regression for finding #8: a pre-existing 0-byte lock file (e.g. left
    over from an older touch()-based implementation) must remain acquirable.

    On the deployed runtime (Python 3.13.5, Windows 11), ``msvcrt.locking``
    with ``LK_NBLCK`` succeeds on a genuine 0-byte file, so the lock helper
    does not need to pad the file before locking.
    """
    lock_path = tmp_path / "supervisor.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"")  # simulate the old touch()-created 0-byte file
    assert lock_path.stat().st_size == 0

    lock = try_acquire_supervisor_lock(lock_path)
    assert lock is not None, "0-byte pre-existing lock file should still be acquirable"
    lock.release()


def test_run_supervised_second_instance_lock_returns_false(tmp_path: Path) -> None:
    """Second invocation while lock is held returns ok=False."""
    app = FakeApp(tmp_path, [_drained_result()])

    # Acquire the lock externally before calling run_supervised
    lock_path = app.paths.root / "supervisor.lock"
    lock = try_acquire_supervisor_lock(lock_path)
    assert lock is not None, "pre-requisite: should acquire lock in test setup"

    try:
        result = run_supervised(app, max_passes=1)
        assert result.ok is False
        assert "supervisor already running" in result.message
    finally:
        lock.release()


def test_run_supervised_lock_released_after_run(tmp_path: Path) -> None:
    """After run_supervised finishes, the lock is released (second call succeeds)."""
    app = FakeApp(tmp_path, [_drained_result()])
    fc = FakeClock(auto_advance=0.0)
    result1 = run_supervised(app, clock=fc.now, sleep=fc.sleep, max_passes=5)
    assert result1.ok is True

    # Should be able to acquire again after first run
    lock_path = app.paths.root / "supervisor.lock"
    lock = try_acquire_supervisor_lock(lock_path)
    assert lock is not None, "lock should be free after run_supervised exits"
    lock.release()


def test_run_supervised_summary_uses_fleet_live_count(tmp_path: Path, capfd: Any) -> None:
    """The 'live ~N' summary line uses the dispatch-scoped fleet-wide count, not the local snapshot."""
    result = CommandResult(
        True,
        "loop complete",
        {
            "dispatch": {
                "selected_count": 0,
                "fleet_live_session_count": 2,
                "live_session_count": 1,
            },
            "dispatch_rework": {"selected_count": 0},
            "merges": [],
            "reviews": [],
            "errors": [],
            "open_tracked_prs": 0,
            "skipped_reviews": 0,
        },
    )
    app = FakeApp(tmp_path, [result])
    fc = FakeClock(auto_advance=0.0)
    run_supervised(app, clock=fc.now, sleep=fc.sleep, max_passes=1)

    out = capfd.readouterr().out
    assert "live ~2" in out, "summary should report fleet-wide live count"
    assert "live ~1" not in out, (
        "summary should not report local snapshot count when fleet count is available"
    )
