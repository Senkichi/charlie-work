from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

import pytest

from charlie_work import host
from charlie_work.host.clock import RealClock, format_utc
from charlie_work.host.fakes import FakeClock, FakeSessionCounter

_UTC_NOW_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def test_format_utc_matches_state_utc_now_shape() -> None:
    assert _UTC_NOW_RE.match(format_utc(RealClock().now()))


def test_format_utc_drops_microseconds_and_uses_z() -> None:
    moment = datetime(2026, 1, 2, 3, 4, 5, 999999, tzinfo=UTC)
    assert format_utc(moment) == "2026-01-02T03:04:05Z"


def test_fake_clock_advance_moves_now_and_monotonic() -> None:
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC), mono=10.0)
    clock.advance(30)
    assert clock.now() == datetime(2026, 1, 1, 0, 0, 30, tzinfo=UTC)
    assert clock.monotonic() == 40.0


def test_fake_clock_default_start_is_epoch_for_monotonic_only_users() -> None:
    clock = FakeClock()
    assert clock.now() == datetime(1970, 1, 1, tzinfo=UTC)
    assert clock.monotonic() == 0.0


def test_fake_clock_sleep_records_calls_and_advances_by_slept_seconds() -> None:
    clock = FakeClock()
    clock.sleep(2.5)
    clock.sleep(1.0)
    assert clock.sleep_calls == [2.5, 1.0]
    assert clock.monotonic() == 3.5
    assert clock.now() == datetime(1970, 1, 1, 0, 0, 3, 500000, tzinfo=UTC)


def test_fake_clock_sleep_auto_advance_overrides_slept_seconds() -> None:
    clock = FakeClock(auto_advance=1.0)
    clock.sleep(30.0)
    clock.sleep(9.0)
    assert clock.sleep_calls == [30.0, 9.0]
    assert clock.monotonic() == 2.0


def test_fake_host_swaps_current_and_restores(fake_host) -> None:
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
    fake = fake_host(clock=clock)
    assert host.current() is fake
    assert host.current().clock is clock


def test_fake_host_restored_after_previous_test(fake_host, monkeypatch) -> None:
    fake_host(clock=FakeClock(datetime(2026, 1, 1, tzinfo=UTC)))
    assert host.current() is not host.REAL
    monkeypatch.undo()
    assert host.current() is host.REAL


def test_fake_host_composes_with_previous_fake(fake_host) -> None:
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
    fake_host(clock=clock)
    fake_host(sessions=FakeSessionCounter(workers=1))
    assert host.current().clock is clock


def test_host_ports_is_frozen() -> None:
    with pytest.raises(AttributeError):
        host.REAL.clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))  # type: ignore[misc]


def test_fake_host_clock_freezes_state_and_workflow_utc_now(fake_host) -> None:
    from charlie_work import state, workflow

    fake_host(clock=FakeClock(datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC)))
    assert state.utc_now() == "2026-03-04T05:06:07Z"
    assert workflow.utc_now() == "2026-03-04T05:06:07Z"


def test_fake_process_probe_mirrors_primitive_semantics() -> None:
    from charlie_work.host.fakes import FakeProcessProbe

    probe = FakeProcessProbe({10: 5.0, 11: None})
    assert probe.is_alive(10, 5.0) is True
    assert probe.is_alive(10, 6.0) is False  # recycled pid
    assert probe.is_alive(10, None) is True  # indeterminate -> fail-open
    assert probe.is_alive(11, 7.0) is True
    assert probe.is_alive(99) is False
    assert probe.is_alive(None) is False
    assert probe.is_alive(0) is False
    assert probe.start_time(10) == 5.0


def test_fake_host_probe_reaches_worker_fate_and_sweeps(fake_host) -> None:
    from charlie_work import worker_fate
    from charlie_work.dead_worker_sweep.effects_sessions import _worker_pid_alive
    from charlie_work.dispatch_selection import _reviewer_pid_alive
    from charlie_work.host.fakes import FakeProcessProbe

    fake_host(probe=FakeProcessProbe({4242: 1.0}))
    assert worker_fate.is_alive(4242, 1.0) is True
    assert worker_fate.is_alive(4242, 2.0) is False
    assert worker_fate.is_alive(None, None) is False
    assert _worker_pid_alive({"worker_pid": 4242, "worker_process_start_time": 1.0}) is True
    assert _reviewer_pid_alive({"reviewer_pid": 4242, "reviewer_process_start_time": 9.0}) is False


def test_real_probe_late_binds_to_process_utils(monkeypatch) -> None:
    from charlie_work.host import REAL

    monkeypatch.setattr("charlie_work.process_utils.is_pid_alive", lambda pid, st=None: pid == 7)
    assert REAL.probe.is_alive(7, None) is True
    assert REAL.probe.is_alive(8, None) is False
    assert REAL.probe.is_alive(-1, None) is False


def test_fake_session_counter_reaches_review_fleet_gate(fake_host) -> None:
    from pathlib import Path

    from charlie_work.host.fakes import FakeSessionCounter

    counter = FakeSessionCounter(
        workers=2, reviews=3, fleet_workers=(5, ["x"]), fleet_reviews=(4, [])
    )
    ports = fake_host(sessions=counter)
    s = ports.sessions
    assert s.live_workers(Path("w"), None) == 2
    assert s.live_reviews(Path("r"), None) == 3
    assert s.fleet_live_workers(None) == (5, ["x"])
    assert s.fleet_live_reviews("d") == (4, [])
    assert [c[0] for c in counter.calls] == [
        "live_workers",
        "live_reviews",
        "fleet_live_workers",
        "fleet_live_reviews",
    ]


def test_real_session_counter_late_binds_to_existing_patch_targets(monkeypatch) -> None:
    from pathlib import Path

    from charlie_work.host import REAL

    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", lambda d, s=None: 11)
    monkeypatch.setattr("charlie_work.workflow.count_fleet_live_sessions", lambda o: (12, []))
    monkeypatch.setattr("charlie_work.workflow.count_fleet_live_reviews", lambda o: (13, []))
    monkeypatch.setattr(
        "charlie_work.dispatch_selection._count_live_reviews", lambda d, s=None: 14
    )
    assert REAL.sessions.live_workers(Path("."), None) == 11
    assert REAL.sessions.fleet_live_workers(None) == (12, [])
    assert REAL.sessions.fleet_live_reviews(None) == (13, [])
    assert REAL.sessions.live_reviews(Path("."), None) == 14


def test_fake_session_counter_scripts_issue_numbers_and_session_pids(fake_host) -> None:
    from pathlib import Path

    counter = FakeSessionCounter(issue_numbers={7, 9}, session_pids={"sess-1": 4321})
    ports = fake_host(sessions=counter)
    s = ports.sessions
    assert s.live_issue_numbers(Path("w")) == {7, 9}
    assert s.live_session_pids(Path("w")) == {"sess-1": 4321}
    assert [c[0] for c in counter.calls] == ["live_issue_numbers", "live_session_pids"]


def test_real_session_counter_issue_numbers_filters_dead_workers(monkeypatch) -> None:
    from pathlib import Path
    from types import SimpleNamespace

    from charlie_work.host import REAL

    workers = [
        SimpleNamespace(issue_number=1, is_alive=lambda: True),
        SimpleNamespace(issue_number=2, is_alive=lambda: False),
        SimpleNamespace(issue_number=3, is_alive=lambda: True),
    ]
    monkeypatch.setattr("charlie_work.worker.iter_workers", lambda _d: workers)
    assert REAL.sessions.live_issue_numbers(Path(".")) == {1, 3}


def test_real_session_counter_session_pids_late_binds(monkeypatch) -> None:
    from pathlib import Path

    from charlie_work.host import REAL

    monkeypatch.setattr("charlie_work.worktree._own_live_session_pids", lambda _d: {"s": 11})
    assert REAL.sessions.live_session_pids(Path(".")) == {"s": 11}


def test_fleet_live_workers_reaches_fleet_registry_patch(monkeypatch) -> None:
    """Issue #2230 rework: ``workflow.count_fleet_live_sessions`` is a re-export
    of ``host/sessions.py``'s late-binding facade, not a frozen ``from
    ... import`` binding -- so a patch against
    ``fleet_registry.count_fleet_live_sessions`` must still reach callers
    going through the port (the supervise self-deploy deferral).
    """
    from charlie_work.host import REAL

    monkeypatch.setattr(
        "charlie_work.fleet_registry.count_fleet_live_sessions", lambda o: (21, ["x"])
    )
    assert REAL.sessions.fleet_live_workers(None) == (21, ["x"])


def test_command_result_reexport_is_identity() -> None:
    from charlie_work import cli, command_result, workflow

    assert cli.CommandResult is command_result.CommandResult
    assert not hasattr(workflow, "CommandResult")


def _app(tmp_path, **kwargs):
    from _review_fixtures import _dispatch_reviews_app

    return _dispatch_reviews_app(tmp_path, **kwargs)


def test_app_default_host_reads_current_at_access_time(fake_host, tmp_path) -> None:
    from charlie_work.host.fakes import FakeProcessProbe

    app = _app(tmp_path)
    assert app.host is host.REAL
    probe = FakeProcessProbe({4242: 1.0})
    fake_host(probe=probe)  # installed AFTER the app was built
    assert app.host.probe is probe


def test_app_without_host_shares_the_fixture_fake_with_non_app_code(fake_host, tmp_path) -> None:
    from charlie_work import worker_fate
    from charlie_work.host.fakes import FakeProcessProbe

    fake_host(probe=FakeProcessProbe({4242: 1.0}))
    app = _app(tmp_path)
    assert app.host.probe.is_alive(4242, 1.0) is True
    assert worker_fate.is_alive(4242, 1.0) is True


def test_explicit_host_wins_over_fixture_and_does_not_mutate_global(fake_host, tmp_path) -> None:
    import dataclasses

    from charlie_work.host.fakes import FakeReviewLauncher

    explicit = dataclasses.replace(host.REAL, launch=FakeReviewLauncher())
    app = _app(tmp_path)
    app_explicit = type(app)(app.repo_root, app.paths, app.config, app.gh, host=explicit)
    fixture_ports = fake_host(launch=FakeReviewLauncher())
    assert app_explicit.host is explicit
    assert app.host is fixture_ports
    assert host.current() is fixture_ports


def test_fake_review_launcher_scripts_outcomes_and_records_requests() -> None:
    from charlie_work.host.fakes import FakeReviewLauncher

    fake = FakeReviewLauncher(["boom"])
    record = fake.launch("claude-code", pr_number=7, branch="b")
    assert record.error == "boom" and record.pid is None
    assert fake.requests == [("claude-code", {"pr_number": 7, "branch": "b"})]
    assert FakeReviewLauncher().launch("devin-shell", pr_number=1).pid == 1


def test_real_review_launcher_returns_errors_as_values(monkeypatch) -> None:
    def _boom(**_kw):
        raise OSError("no such binary")

    monkeypatch.setitem(
        __import__("charlie_work.workflow", fromlist=["x"])._REVIEW_LAUNCHERS, "api", _boom
    )
    record = host.REAL.launch.launch("api", pr_number=3, branch="b")
    assert record.pid is None and record.error == "OSError: no such binary"
    unknown = host.REAL.launch.launch("nope", pr_number=3, branch="b")
    assert unknown.error == "unsupported reviewer harness: 'nope'"


def test_fake_worker_launcher_scripts_outcomes_and_records_calls(tmp_path) -> None:
    """Issue #2229: a callable outcome takes the ``dispatch_sessions``
    signature so an existing dispatch fake drops in unchanged."""
    from charlie_work.adapters import SessionDispatchResult, SessionRequest
    from charlie_work.host.fakes import FakeWorkerLauncher

    request = SessionRequest(
        issue_number=1,
        issue_title="t",
        prompt_path=tmp_path / "p.md",
        branch_name="agent/issue-1",
    )
    ok = SessionDispatchResult(
        issue_number=1,
        issue_title="t",
        prompt_path=str(request.prompt_path),
        branch_name="agent/issue-1",
        adapter="command",
        ok=True,
        pid=42,
    )
    fake = FakeWorkerLauncher([lambda *a: [ok]])
    from charlie_work.adapters import AdapterSettings

    settings = AdapterSettings(adapter="command")
    results = fake.launch(tmp_path, tmp_path / "m.json", tmp_path / "r.json", settings, [request])
    assert results == [ok]
    assert fake.calls == [
        (tmp_path, tmp_path / "m.json", tmp_path / "r.json", settings, [request])
    ]
    # A ``str`` outcome fails every request through the worker error seam.
    fake = FakeWorkerLauncher(["launch failed: boom"])
    results = fake.launch(tmp_path, tmp_path / "m.json", tmp_path / "r.json", settings, [request])
    assert results[0].ok is False and results[0].error == "launch failed: boom"


def test_real_worker_launcher_late_binds_and_returns_errors_as_values(
    monkeypatch, tmp_path
) -> None:
    """Issue #2229: ``RealWorkerLauncher`` resolves ``workflow.dispatch_sessions``
    at call time (so existing patches keep intercepting) and converts a raise
    into per-request failure values plus ``launch_failed`` events."""
    from charlie_work.adapters import AdapterSettings, SessionRequest
    from charlie_work.instrumentation import query_events
    from charlie_work.paths import runtime_paths
    from charlie_work.config import OrchestratorConfig

    def _boom(*_a, **_k):
        raise RuntimeError("port exploded")

    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", _boom)

    request = SessionRequest(
        issue_number=2,
        issue_title="t",
        prompt_path=tmp_path / "p.md",
        branch_name="agent/issue-2",
    )
    settings = AdapterSettings(adapter="claude-code")
    results = host.REAL.worker_launch.launch(
        tmp_path, tmp_path / "m.json", tmp_path / "r.json", settings, [request]
    )

    (result,) = results
    assert result.ok is False and "port exploded" in (result.error or "")
    events = query_events(
        runtime_paths(tmp_path, OrchestratorConfig().runtime.state_dir).state_file,
        kind="launch_failed",
    )
    assert len(events) == 1
    assert events[0]["payload"]["issue_number"] == 2
    assert events[0]["payload"]["role"] == "worker"

    # And a patched success path is intercepted through the same late binding.
    monkeypatch.setattr(
        "charlie_work.workflow.dispatch_sessions",
        lambda *a, **_k: ["sentinel"],
    )
    assert host.REAL.worker_launch.launch(
        tmp_path, tmp_path / "m.json", tmp_path / "r.json", settings, [request]
    ) == ["sentinel"]


def test_fake_host_clock_reaches_state_predicates(fake_host) -> None:
    """Issue #2233: ``state.py``'s wall-clock reads must resolve through
    ``host.current().clock`` so ``fake_host(clock=...)`` can freeze them.

    The frozen instant sits in the wall-clock future, so every predicate
    discriminates between the two clocks: under the unfixed code the
    ``past`` timestamps still read as upcoming and each assertion flips.
    """
    from charlie_work import state

    frozen = datetime(2031, 3, 4, 5, 6, 7, tzinfo=UTC)
    fake_host(clock=FakeClock(frozen))
    past = (frozen - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    future = (frozen + timedelta(hours=1)).isoformat().replace("+00:00", "Z")

    assert state.is_throttled({"throttled_until": past}) is False
    assert state.is_throttled({"throttled_until": future}) is True
    assert (
        state.is_reviewer_quota_exhausted({"reviewer_quota": {"throttled_until": past}}) is False
    )
    assert state.is_claim_stale(past) is True
    assert state.is_claim_stale(future) is False
    assert state.stale_operator_claims({"issues": {"7": {"operator_claimed_at": past}}}) == {7}
    assert state.is_quota_probe_due({"quota_probe": {"next_probe_at": past}}) is True
    assert (
        state.is_operator_queue_review_due(
            {"deescalation_pass": {"next_operator_queue_review_at": past}}
        )
        is True
    )
    assert state.age_days_since(past) == 0.04


def test_fake_host_clock_reaches_due_schedulers(fake_host) -> None:
    """Issue #2233: the periodic-pass due-checks extracted from ``state.py``
    into ``periodic_pass_schedule.py`` are the same wall-clock family --
    they must resolve through the port or a frozen clock cannot gate the
    matching ``_maybe_*`` schedulers (which already read ``self.host``).
    """
    from charlie_work import state

    frozen = datetime(2031, 3, 4, 5, 6, 7, tzinfo=UTC)
    fake_host(clock=FakeClock(frozen))
    past = (frozen - timedelta(minutes=1)).isoformat().replace("+00:00", "Z")

    assert state.is_deescalation_due({"deescalation_pass": {"next_deescalation_at": past}}) is True
    assert state.is_reconcile_due({"reconcile_pass": {"next_reconcile_at": past}}) is True
    assert (
        state.is_worktree_reclamation_due({"worktree_reclamation": {"next_run_at": past}}) is True
    )


def test_fake_host_clock_stamps_reconcile_schedule(fake_host, tmp_path) -> None:
    """Issue #2233: ``_maybe_reconcile_drift``'s cadence stamp must be a
    port-clock reading, not the wall clock -- the armed ``next_reconcile_at``
    must be exactly ``frozen + interval_minutes``.
    """
    from _dispatch_fixtures import _reconcile_pass_app
    from charlie_work.state import load_state

    frozen = datetime(2031, 3, 4, 5, 6, 7, tzinfo=UTC)
    fake_host(clock=FakeClock(frozen))
    app = _reconcile_pass_app(tmp_path, interval_minutes=30)

    app._maybe_reconcile_drift()

    assert load_state(app.paths.state_file)["reconcile_pass"]["next_reconcile_at"] == (
        "2031-03-04T05:36:07Z"
    )
