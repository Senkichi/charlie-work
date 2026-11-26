"""Issue #3098: what both dispatch lanes record once a launch has returned.

Characterisation suite for the launch-outcome bookkeeping shared by fresh
``dispatch`` and remote ``dispatch_rework``: the ``label_error`` record left
behind when a label transition is not applied (every site in both lanes), and
the blocked-environment foreign-writer reap (fresh failure, rework failure,
and rework's pre-claim safety net). Every test drives the public entry points
(``app.dispatch()`` / ``app.dispatch_rework()``) with a scripted
``FakeWorkerLauncher`` and asserts the persisted record, the result data and the
exact Event kinds and payloads, so the suite pins the behaviour across the move
into :mod:`charlie_work.worker_launch_outcome`.

The rework pre-claim safety net writes its reap Event raw, with no ``repo``
(events.db column NULL). Since issue #3099 every Event of the failure branches,
rework's included, goes through the write gate and carries the repo name; the
failure branches themselves are pinned in ``tests/test_worker_launch_failures.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from _dispatch_fixtures import _cross_repo_issue_body
from _dispatch_fixtures import _write_fleet_registry, _write_sibling_repo_file
from _rework_dispatch_fixtures import _blocked_env_timestamps
from charlie_work.config import (
    OrchestratorConfig,
)
from charlie_work.host.fakes import FakeProcessProbe
from charlie_work.labels import TransitionOutcome
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp

from _launch_outcome_fixtures import (
    FRESH,
    LANES,
    _app,
    _assert_label_error,
    _config,
    _events,
    _fresh_app,
    _launch,
    _marker,
    _LabelFailGitHub,
    _rework_app,
    _run,
)

# --- label_error: every site in both lanes -------------------------------------


@LANES
def test_label_error_on_the_launched_edge(tmp_path: Path, fake_host, lane: str) -> None:
    """Fresh ``dispatched`` / rework ``rework_dispatched``: launch ok, label add fails."""
    app = _app(lane, tmp_path, config=_config(max_auto_redispatch=5), label_fail=True)
    _launch(fake_host, ok=True, pid=4242)

    result = _run(app, lane)

    edge = "dispatched" if lane == FRESH else "rework_dispatched"
    _assert_label_error(app, result, edge)
    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatched"


def test_label_error_on_rework_redispatch_cap_escalation(tmp_path: Path, fake_host) -> None:
    """Rework success past ``max_auto_redispatch`` escalates; its label write fails."""
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    app = _rework_app(
        tmp_path,
        config=_config(max_auto_redispatch=2),
        entry={"redispatch_at": [now_iso, now_iso]},
        label_fail=True,
    )
    _launch(fake_host, ok=True, pid=4242)

    result = app.dispatch_rework()

    _assert_label_error(app, result, "redispatch_operator_queued")
    entry = load_state(app.paths.state_file)["issues"]["123"]
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "redispatch_cap_exceeded"


@LANES
def test_label_error_on_deterministic_failure_escalation(
    tmp_path: Path, fake_host, lane: str
) -> None:
    """A deterministic launch failure escalates at once; its label write fails."""
    app = _app(lane, tmp_path, config=_config(max_auto_redispatch=5), label_fail=True)
    _launch(fake_host, ok=False, failure_kind="worker_blocked")

    result = _run(app, lane)

    _assert_label_error(app, result, "redispatch_operator_queued")
    entry = load_state(app.paths.state_file)["issues"]["123"]
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "worker_blocked"


def test_label_error_on_fresh_cross_repo_escalation(tmp_path: Path) -> None:
    """The cross-repo gate escalates the issue; its label write fails."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = _LabelFailGitHub()
    gh.prs[0]["state"] = "CLOSED"
    gh.issues[0]["body"] = _cross_repo_issue_body()
    sibling_root = tmp_path.parent / "sibling-ci-runners-3098"
    _write_sibling_repo_file(sibling_root, "src/ci_fleet/suite_coverage.py")
    _write_fleet_registry(
        tmp_path / "fleet", {"owner/ci-runners": {"repo_root": str(sibling_root)}}
    )
    app = OrchestratorApp(tmp_path, paths, config, gh)

    result = app.dispatch(limit=1)

    assert 123 in result.data["cross_repo_escalated_issue_numbers"]
    assert result.data["label_errors"] == [123]
    reason = result.data["failures"][123]
    assert reason.startswith("label transition 'redispatch_operator_queued' partial_failure")
    entry = load_state(paths.state_file)["issues"]["123"]
    assert entry["status"] == "escalated"
    assert entry["label_error"]["edge"] == "redispatch_operator_queued"
    assert entry["label_error"]["outcome"] == TransitionOutcome.PARTIAL_FAILURE.value
    assert entry["label_error"]["add_failures"]


def test_label_error_on_fresh_blocked_environment_escalation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    """Fresh blocked-environment cap with the reap cap reached: escalate, label fails."""
    reap_ts = _blocked_env_timestamps(2)
    app = _fresh_app(
        tmp_path,
        config=_config(max_auto_redispatch=2, max_foreign_writer_reaps=2),
        entry={
            "status": "dispatch_failed",
            "blocked_environment_at": _blocked_env_timestamps(2),
            "foreign_writer_reaps": reap_ts,
        },
        label_fail=True,
    )
    _launch(fake_host, ok=False, failure_kind="worktree_foreign_writer", pid=1234)

    result = app.dispatch(limit=1)

    _assert_label_error(app, result, "redispatch_operator_queued")
    entry = load_state(app.paths.state_file)["issues"]["123"]
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "dispatch_blocked_environment"
    assert entry["foreign_writer_reaps"] == reap_ts


# --- foreign-writer reap: the three copies -------------------------------------


def test_fresh_failure_reaps_blocked_foreign_writer_at_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    app = _fresh_app(
        tmp_path,
        config=_config(max_auto_redispatch=2, max_foreign_writer_reaps=2),
        entry={"status": "dispatch_failed", "blocked_environment_at": _blocked_env_timestamps(2)},
    )
    _launch(
        fake_host,
        ok=False,
        failure_kind="worktree_foreign_writer",
        pid=1234,
        worktree_path=str(tmp_path / "wt-foreign"),
    )
    reap_calls: list[tuple[Any, ...]] = []

    def _fake_reap(failed_result, config, state_file, issue_number, sessions_dir=None):
        reap_calls.append((failed_result.pid, config, state_file, issue_number))
        return True

    monkeypatch.setattr("charlie_work.workflow._try_reap_blocked_foreign_writer", _fake_reap)

    result = app.dispatch(limit=1)

    assert result.ok is False
    assert reap_calls == [(1234, app.config, app.paths.state_file, 123)]
    entry = load_state(app.paths.state_file)["issues"]["123"]
    assert entry["status"] == "dispatch_failed"
    assert entry["blocked_environment_at"] == []
    assert len(entry["foreign_writer_reaps"]) == 1
    assert "escalation_reason" not in entry
    (event,) = _events(app, "dispatch_blocked_environment_reaped")
    assert event["payload"] == {
        "issue_number": 123,
        "failure_kind": "worktree_foreign_writer",
        "pid": 1234,
        "blocked_environment_count": 3,
        "foreign_writer_reap_count": 1,
    }
    assert event["repo"] == tmp_path.name
    assert _events(app, "session_failed_escalated") == []


def test_fresh_failure_with_reap_cap_reached_does_not_reap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    app = _fresh_app(
        tmp_path,
        config=_config(max_auto_redispatch=2, max_foreign_writer_reaps=2),
        entry={
            "status": "dispatch_failed",
            "blocked_environment_at": _blocked_env_timestamps(2),
            "foreign_writer_reaps": _blocked_env_timestamps(2),
        },
    )
    _launch(fake_host, ok=False, failure_kind="worktree_foreign_writer", pid=1234)

    def _must_not_run(*_a: Any, **_k: Any) -> bool:
        raise AssertionError("reap must not be attempted once the reap cap is reached")

    monkeypatch.setattr("charlie_work.workflow._try_reap_blocked_foreign_writer", _must_not_run)

    app.dispatch(limit=1)

    entry = load_state(app.paths.state_file)["issues"]["123"]
    assert entry["status"] == "escalated"
    assert _events(app, "dispatch_blocked_environment_reaped") == []
    (escalated,) = _events(app, "session_failed_escalated")
    assert escalated["payload"]["reason"] == "dispatch_blocked_environment"


def test_fresh_failure_when_reap_declines_escalates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    app = _fresh_app(
        tmp_path,
        config=_config(max_auto_redispatch=2, max_foreign_writer_reaps=2),
        entry={"status": "dispatch_failed", "blocked_environment_at": _blocked_env_timestamps(2)},
    )
    _launch(fake_host, ok=False, failure_kind="worktree_foreign_writer", pid=1234)
    monkeypatch.setattr(
        "charlie_work.workflow._try_reap_blocked_foreign_writer", lambda *_a, **_k: False
    )

    app.dispatch(limit=1)

    entry = load_state(app.paths.state_file)["issues"]["123"]
    assert entry["status"] == "escalated"
    assert "foreign_writer_reaps" not in entry
    assert _events(app, "dispatch_blocked_environment_reaped") == []


def _rework_reap_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_host, *, label_fail: bool = False
) -> tuple[OrchestratorApp, list[tuple[Any, ...]]]:
    """Rework at ``max_auto_redispatch=0`` with a marker: the pre-claim safety net
    and the failure-path reap are both reachable in one pass."""
    app = _rework_app(
        tmp_path,
        config=_config(max_auto_redispatch=0, max_foreign_writer_reaps=2),
        label_fail=label_fail,
    )
    wt_path = _marker(app, tmp_path)
    fake_host(probe=FakeProcessProbe({1234: None}))
    _launch(
        fake_host,
        ok=False,
        failure_kind="worktree_foreign_writer",
        pid=1234,
        worktree_path=str(wt_path),
    )
    monkeypatch.setattr(
        "charlie_work.workflow._reap_idle_foreign_writer",
        lambda worktree_path, marker, _config, _sessions_dir=None, **_kw: True,
    )
    reap_calls: list[tuple[Any, ...]] = []
    return app, reap_calls


def test_rework_safety_net_and_failure_reap_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    """Both rework reaps in one pass. The pre-claim safety net emits the FRESH-lane
    kind (no ``failure_kind``); the failure-path reap emits the rework kind."""
    app, reap_calls = _rework_reap_app(tmp_path, monkeypatch, fake_host)

    def _fake_reap(failed_result, config, state_file, issue_number, sessions_dir=None):
        reap_calls.append((failed_result.pid, config, state_file, issue_number))
        return True

    monkeypatch.setattr("charlie_work.workflow._try_reap_blocked_foreign_writer", _fake_reap)

    result = app.dispatch_rework()

    assert result.ok is False
    assert reap_calls == [(1234, app.config, app.paths.state_file, 123)]
    entry = load_state(app.paths.state_file)["issues"]["123"]
    assert entry["status"] == "rework_requested"
    assert entry["dispatched_at"] is None
    assert entry["blocked_environment_at"] == []
    assert len(entry["foreign_writer_reaps"]) == 2
    (safety_net,) = _events(app, "dispatch_blocked_environment_reaped")
    assert safety_net["payload"] == {
        "issue_number": 123,
        "pid": 1234,
        "blocked_environment_count": 0,
        "foreign_writer_reap_count": 1,
    }
    (failure,) = _events(app, "rework_dispatch_blocked_environment_reaped")
    assert failure["payload"] == {
        "issue_number": 123,
        "failure_kind": "worktree_foreign_writer",
        "pid": 1234,
        "blocked_environment_count": 1,
        "foreign_writer_reap_count": 2,
    }
    # The pre-claim safety net still writes its Event raw (no repo); the failure
    # path's reap Event goes through the write gate (issue #3099), which binds repo.
    assert safety_net["repo"] is None
    assert failure["repo"] == tmp_path.name
    assert _events(app, "session_failed_escalated") == []


def test_rework_failure_when_reap_declines_escalates_with_label_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    app, _ = _rework_reap_app(tmp_path, monkeypatch, fake_host, label_fail=True)
    monkeypatch.setattr(
        "charlie_work.workflow._try_reap_blocked_foreign_writer", lambda *_a, **_k: False
    )

    result = app.dispatch_rework()

    _assert_label_error(app, result, "redispatch_operator_queued")
    entry = load_state(app.paths.state_file)["issues"]["123"]
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "dispatch_blocked_environment"
    assert len(entry["foreign_writer_reaps"]) == 1  # the safety net's, not the failure path's
    assert _events(app, "rework_dispatch_blocked_environment_reaped") == []
    (escalated,) = _events(app, "session_failed_escalated")
    assert escalated["payload"]["previous_status"] == "rework_requested"
    assert escalated["payload"]["failure_kind"] == "worktree_foreign_writer"


def test_rework_failure_with_reap_cap_reached_does_not_reap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    app = _rework_app(
        tmp_path,
        config=_config(max_auto_redispatch=0, max_foreign_writer_reaps=1),
        entry={"foreign_writer_reaps": _blocked_env_timestamps(1)},
    )
    _launch(fake_host, ok=False, failure_kind="worktree_foreign_writer", pid=1234)

    def _must_not_run(*_a: Any, **_k: Any) -> bool:
        raise AssertionError("reap must not be attempted once the reap cap is reached")

    monkeypatch.setattr("charlie_work.workflow._try_reap_blocked_foreign_writer", _must_not_run)

    app.dispatch_rework()

    entry = load_state(app.paths.state_file)["issues"]["123"]
    assert entry["status"] == "escalated"
    assert _events(app, "rework_dispatch_blocked_environment_reaped") == []


def test_rework_safety_net_reap_alone_lets_the_issue_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    """Pre-claim safety net with a successful launch afterwards: one reap, one Event."""
    app = _rework_app(
        tmp_path,
        config=_config(max_auto_redispatch=2, max_foreign_writer_reaps=2),
        entry={"blocked_environment_at": _blocked_env_timestamps(2)},
    )
    _marker(app, tmp_path)
    fake_host(probe=FakeProcessProbe({1234: None}))
    reaps: list[int] = []

    def _fake_reap(worktree_path, marker, _config, _sessions_dir=None, **_kw):
        reaps.append(marker["pid"])
        return True

    monkeypatch.setattr("charlie_work.workflow._reap_idle_foreign_writer", _fake_reap)
    _launch(fake_host, ok=True, pid=4242)

    app.dispatch_rework()

    assert reaps == [1234]
    entry = load_state(app.paths.state_file)["issues"]["123"]
    assert entry["status"] != "escalated"
    assert entry["blocked_environment_at"] == []
    assert len(entry["foreign_writer_reaps"]) == 1
    (event,) = _events(app, "dispatch_blocked_environment_reaped")
    assert event["payload"] == {
        "issue_number": 123,
        "pid": 1234,
        "blocked_environment_count": 2,
        "foreign_writer_reap_count": 1,
    }
    assert event["repo"] is None


def test_rework_safety_net_without_a_reap_escalates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    app = _rework_app(
        tmp_path,
        config=_config(max_auto_redispatch=2, max_foreign_writer_reaps=2),
        entry={"blocked_environment_at": _blocked_env_timestamps(2)},
    )
    _marker(app, tmp_path)
    fake_host(probe=FakeProcessProbe({1234: None}))
    monkeypatch.setattr("charlie_work.workflow._reap_idle_foreign_writer", lambda *_a, **_k: False)
    _launch(fake_host, ok=True, pid=4242)

    app.dispatch_rework()

    entry = load_state(app.paths.state_file)["issues"]["123"]
    assert entry["status"] == "escalated"
    assert "foreign_writer_reaps" not in entry
    assert _events(app, "dispatch_blocked_environment_reaped") == []
