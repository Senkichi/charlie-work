"""``throttle_window_set`` audit-event tests (issue #2006).

``state.set_throttled_until`` is the single writer API for the provider
throttle window, but event emission used to be left to each caller -- a
moved window was invisible in events.db (the audit gap the issue was filed
for). ``set_throttled_until`` now appends a ``throttle_window_set`` event
itself on every value change (extended or shortened), stays silent on a
no-op, and requires ``source`` -- a string literal naming the caller -- at
every call site.

Coverage here is layered:

* unit-level: emission shape, both directions, no-op silence, the three
  write channels (in-memory ring only, ``state_path`` dual-write,
  ``WriteGate``), and dry-run suppression;
* call-site level: one test per production call site (six emit-capable
  lanes: the four ``dead_worker_reap`` sites, ``reconcile.apply_fixes``,
  and the orphan sweep's ``dead_worker_classification`` classify-at-credit
  seam) proving a changed window produces exactly one
  ``throttle_window_set`` carrying the right
  ``previous``/``throttled_until``/``source``;
* structural: an AST scan that fails if a new ``set_throttled_until`` call
  site appears without a ``source=`` literal, and the signature checks that
  pin ``source`` as a required keyword-only argument with no default.
"""

from __future__ import annotations

import ast
import inspect
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _src_ast import parsed, source_files, source_text

from _fakes_github import FakeGitHub
from _orphan_sweep_fixtures import _dead_worker_rework_bed, _run_orphan_sweep
from _worker_fixtures import _make_stalled_devin_session, _stale_devin_probe, _wg
from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.instrumentation import _LEVEL_BY_KIND, close_db, read_event_log
from charlie_work.reconcile import DriftItem, apply_fixes
from charlie_work.state import empty_state, load_state, set_throttled_until
from charlie_work.write_gate import WriteGate


@pytest.fixture(autouse=True)
def _close_db_after_test(tmp_path: Path) -> None:
    """Close any events.db handles opened during the test (Windows can't
    delete a file an open SQLite connection still holds, so per-test
    tmp_path cleanup fails without this)."""
    yield
    close_db(tmp_path / "state.json")


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _future(*, minutes: int = 10) -> str:
    return _iso(datetime.now(UTC) + timedelta(minutes=minutes))


def _throttle_window_events(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for e in state.get("events", []) if e.get("kind") == "throttle_window_set"]


_RATE_LIMIT_LOG = (
    "Some work done...\n"
    "Error: Reached overall message rate limit. Please try again later. "
    "Your limit will reset in 10 minutes.\n"
)


# ---------------------------------------------------------------------------
# Unit-level: emission shape and no-op semantics
# ---------------------------------------------------------------------------


def test_set_throttled_until_emits_event_with_full_payload() -> None:
    """A changed window emits exactly one ``throttle_window_set`` whose
    payload is the audit record the issue specifies: previous value, new
    value, reason, adapter_kind, the stamped role ``(harness, model)`` that
    died (issue #2279), and the caller's ``source``."""
    new_value = _future()
    state = set_throttled_until(
        empty_state(),
        new_value,
        reason="rate_limited",
        adapter_kind="devin",
        harness="devin-shell",
        model="swe-2-high",
        source="test_source",
    )

    assert state["throttled_until"] == new_value
    assert state["throttle_harness"] == "devin-shell"
    assert state["throttle_model"] == "swe-2-high"
    events = _throttle_window_events(state)
    assert len(events) == 1
    assert events[0]["payload"] == {
        "previous": None,
        "throttled_until": new_value,
        "reason": "rate_limited",
        "adapter_kind": "devin",
        "harness": "devin-shell",
        "model": "swe-2-high",
        "source": "test_source",
    }


def test_set_throttled_until_emits_on_extension_and_not_on_kept_window() -> None:
    """The event fires when the window moves. Shortening a still-active
    window is refused by the #2042 monotonic guard, so the window is kept
    unchanged and -- nothing having changed -- no event is emitted."""
    earlier = _iso(datetime.now(UTC) + timedelta(minutes=5))
    later = _iso(datetime.now(UTC) + timedelta(minutes=20))

    extended = set_throttled_until(
        {**empty_state(), "throttled_until": earlier},
        later,
        source="test_source",
    )
    events = _throttle_window_events(extended)
    assert len(events) == 1
    assert events[0]["payload"]["previous"] == earlier
    assert events[0]["payload"]["throttled_until"] == later

    kept = set_throttled_until(
        {**empty_state(), "throttled_until": later},
        earlier,
        source="test_source",
    )
    assert kept["throttled_until"] == later
    assert _throttle_window_events(kept) == []


def test_set_throttled_until_noop_emits_nothing() -> None:
    """A no-op write (same value) emits nothing -- the audit record exists
    for changes, not for every call."""
    value = _future()
    state = set_throttled_until(
        {**empty_state(), "throttled_until": value},
        value,
        reason="rate_limited",
        adapter_kind="devin",
        source="test_source",
    )

    assert state["throttled_until"] == value
    assert _throttle_window_events(state) == []


# ---------------------------------------------------------------------------
# Write channels: in-memory ring, state_path dual-write, WriteGate
# ---------------------------------------------------------------------------


def test_set_throttled_until_state_path_dual_writes_to_events_db(tmp_path: Path) -> None:
    """``state_path`` opt-in dual-writes to events.db -- the store the issue
    found empty for a moved window."""
    state_path = tmp_path / "state.json"
    state = set_throttled_until(
        empty_state(),
        _future(),
        source="test_source",
        state_path=state_path,
    )

    assert len(_throttle_window_events(state)) == 1
    db_events = [e for e in read_event_log(state_path) if e["kind"] == "throttle_window_set"]
    assert len(db_events) == 1
    assert db_events[0]["level"] == "info"
    assert db_events[0]["payload"]["source"] == "test_source"


def test_set_throttled_until_write_gate_dual_writes_to_events_db(tmp_path: Path) -> None:
    """A non-dry-run ``WriteGate`` is the emission channel for gated lanes:
    the event lands in both the ring and events.db through the gate's own
    ``append_event``."""
    state_path = tmp_path / "state.json"
    gate = WriteGate(dry_run=False, state_path=state_path, repo="charlie-work")
    state = set_throttled_until(
        empty_state(),
        _future(),
        source="test_source",
        write_gate=gate,
    )

    assert len(_throttle_window_events(state)) == 1
    db_events = [e for e in read_event_log(state_path) if e["kind"] == "throttle_window_set"]
    assert len(db_events) == 1
    assert db_events[0]["repo"] == "charlie-work"


def test_set_throttled_until_write_gate_dry_run_suppresses_event(tmp_path: Path) -> None:
    """Under ``dry_run=True`` the event is suppressed entirely -- no ring
    append, no events.db row -- matching WriteGate's "same footprint as a
    caller that never ran" invariant. The state transform itself still
    applies (the caller's gated ``save_state`` decides whether it lands)."""
    state_path = tmp_path / "state.json"
    gate = WriteGate(dry_run=True, state_path=state_path, repo="charlie-work")
    new_value = _future()
    state = set_throttled_until(
        empty_state(),
        new_value,
        source="test_source",
        write_gate=gate,
    )

    assert state["throttled_until"] == new_value
    assert _throttle_window_events(state) == []
    assert not (state_path.parent / "events.db").exists()


# ---------------------------------------------------------------------------
# Structural enforcement: source is required, every src call site names itself
# ---------------------------------------------------------------------------


def test_source_is_a_required_keyword_only_parameter() -> None:
    """``source`` must be a required keyword-only argument with no default:
    a call site that forgets it gets a loud TypeError, never a silent
    untagged write."""
    params = inspect.signature(set_throttled_until).parameters
    assert "source" in params
    source = params["source"]
    assert source.kind is inspect.Parameter.KEYWORD_ONLY
    assert source.default is inspect.Parameter.empty

    with pytest.raises(TypeError):
        set_throttled_until(empty_state(), _future())


def test_every_src_set_throttled_until_call_site_passes_source_literal() -> None:
    """Every ``set_throttled_until`` call in src/ must pass ``source=`` as a
    non-empty string literal. ``**kwargs`` splats, a bare variable, or a
    missing keyword all fail here -- the audit trail only works if every
    writer names itself."""
    src_root = Path(__file__).parents[1] / "src" / "charlie_work"
    violations: list[str] = []
    for path in source_files(src_root):
        text = source_text(path)
        if "set_throttled_until" not in text and "persist_failure" not in text:
            continue
        tree = parsed(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            called = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else None
            )
            if called not in {"set_throttled_until", "persist_failure"}:
                continue
            keyword = next((kw for kw in node.keywords if kw.arg == "source"), None)
            rel = f"{path.name}:{node.lineno}"
            if keyword is None:
                violations.append(f"{rel}: missing source=")
            elif (
                path.name in {"worker_fate.py", "dead_worker_classification.py"}
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "source"
            ):
                # Forwarders of their own required ``source`` parameter; the
                # outermost callers are checked here as literal sites.
                continue
            elif not (
                isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
                and keyword.value.value
            ):
                violations.append(
                    f"{rel}: source= must be a non-empty string literal, "
                    f"got {ast.unparse(keyword.value)!r}"
                )
    assert not violations, "; ".join(violations)


def test_throttle_window_set_kind_is_registered_info() -> None:
    """The kind is registered in the event-level registry at ``info`` --
    routine audit bookkeeping, not a condition an operator must act on."""
    assert _LEVEL_BY_KIND["throttle_window_set"] == "info"


# ---------------------------------------------------------------------------
# Call-site coverage: one test per production writer
# ---------------------------------------------------------------------------


def _write_devin_sidecar(
    sessions_dir: Path,
    issue_number: int,
    log_text: str,
    *,
    error: str | None,
    pid: int | None = None,
) -> Path:
    """A devin sidecar with ``pid=None`` by default: ``error`` set is a
    launch failure (issue #266), ``error=None`` is a confirmed-dead session
    with no process to probe (``is_worker_confirmed_dead`` short-circuits on
    ``pid=None``). The orphan-sweep lane instead needs the sidecar pid to
    match the entry's recorded ``worker_pid`` (``_worker_view_for_entry``
    refuses a mismatched log), so ``pid`` is overridable."""
    from charlie_work import devin_shell

    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / f"issue-{issue_number}.log"
    log_path.write_text(log_text, encoding="utf-8")
    record = devin_shell.SessionRecord(
        issue_number=issue_number,
        branch=f"agent/issue-{issue_number}",
        worktree_path=str(sessions_dir.parent / "worktree"),
        prompt_path="prompt.md",
        command=("devin", "--prompt-file", "prompt.md"),
        pid=pid,
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=error,
    )
    sidecar_path = sessions_dir / f"issue-{issue_number}.json"
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    return sidecar_path


def _dead_lane_config() -> OrchestratorConfig:
    return OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )


def _seed_state_with_issue(state_file: Path, issue_number: int) -> None:
    state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "issues": {str(issue_number): {"status": "dispatched"}},
                "prs": {},
                "events": [],
            }
        ),
        encoding="utf-8",
    )


def test_stall_lane_rate_limit_defer_emits_throttle_window_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Call site 1: ``_detect_and_handle_stalled_sessions``'s rate-limit
    defer branch writes the window and emits ``throttle_window_set``."""
    from charlie_work import workflow
    from charlie_work.dead_worker_sweep import effects_sessions

    issue_number = 2006
    sessions_dir, state_file, _ = _make_stalled_devin_session(
        tmp_path, issue_number, _RATE_LIMIT_LOG
    )
    monkeypatch.setattr(
        "charlie_work.write_gate.kill_process_tree",
        lambda pid, start_time=None: [pid],
    )
    monkeypatch.setattr(effects_sessions, "sweep_orphan_processes", lambda worktree_path: [])
    monkeypatch.setattr("charlie_work.worker_fate.is_alive", lambda *_: True)
    monkeypatch.setattr("charlie_work.worker.real_activity_probe_for", _stale_devin_probe)

    config = OrchestratorConfig(
        watchdog=WatchdogConfig(
            rate_limit_defer_enabled=True,
            rate_limit_defer_slack_minutes=2,
        )
    )

    result = workflow._detect_and_handle_stalled_sessions(
        sessions_dir,
        state_file,
        config,
        write_gate=_wg(state_file),
        now=datetime.now(UTC),
    )
    assert result == []

    state = json.loads(state_file.read_text(encoding="utf-8"))
    events = _throttle_window_events(state)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["source"] == "stalled_sessions_rate_limit_defer"
    assert payload["previous"] is None
    assert payload["throttled_until"] == state["throttled_until"]
    assert payload["reason"] == "rate_limited"
    assert payload["adapter_kind"] == "devin"

    # The audit row lands in events.db too -- the store whose silence the
    # issue was filed against.
    db_events = [e for e in read_event_log(state_file) if e["kind"] == "throttle_window_set"]
    assert len(db_events) == 1
    assert db_events[0]["level"] == "info"


def test_stall_lane_reap_emits_throttle_window_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Call site 2: a deferred worker still silent past its deadline is
    killed, classified ``rate_limited`` from the log tail, and the write of
    the freshly computed window emits ``throttle_window_set``."""
    from charlie_work import workflow
    from charlie_work.dead_worker_sweep import effects_sessions

    issue_number = 2007
    frozen_now = datetime.now(UTC)
    past_defer = _iso(frozen_now - timedelta(minutes=5))
    sessions_dir, state_file, _ = _make_stalled_devin_session(
        tmp_path, issue_number, _RATE_LIMIT_LOG, rate_limit_defer_until=past_defer
    )
    monkeypatch.setattr(
        "charlie_work.write_gate.kill_process_tree",
        lambda pid, start_time=None: [pid],
    )
    monkeypatch.setattr(effects_sessions, "sweep_orphan_processes", lambda worktree_path: [])
    monkeypatch.setattr("charlie_work.worker_fate.is_alive", lambda *_: True)
    monkeypatch.setattr("charlie_work.worker.real_activity_probe_for", _stale_devin_probe)

    config = OrchestratorConfig(
        watchdog=WatchdogConfig(
            rate_limit_defer_enabled=True,
            rate_limit_defer_slack_minutes=2,
        )
    )

    result = workflow._detect_and_handle_stalled_sessions(
        sessions_dir,
        state_file,
        config,
        write_gate=_wg(state_file),
        now=frozen_now,
    )
    assert result == [{"issue": issue_number, "pid": 99999}]

    state = json.loads(state_file.read_text(encoding="utf-8"))
    events = _throttle_window_events(state)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["source"] == "stalled_sessions_reap"
    assert payload["previous"] is None
    assert payload["throttled_until"] == state["throttled_until"]
    assert payload["reason"] == "rate_limited"
    assert payload["adapter_kind"] == "devin"


def test_launch_failure_reap_emits_throttle_window_set(tmp_path: Path) -> None:
    """Call site 3: a launch-failure sidecar whose log tail carries a
    throttle signature persists the window via the dead-session lane's
    launch-failure branch, emitting ``throttle_window_set``."""
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    issue_number = 2008
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    _write_devin_sidecar(sessions_dir, issue_number, _RATE_LIMIT_LOG, error="launch died")
    state_file = tmp_path / "state.json"
    _seed_state_with_issue(state_file, issue_number)

    reaped = _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, state_file, FakeGitHub(), _dead_lane_config(), write_gate=_wg(state_file)
    )
    assert [r["issue_number"] for r in reaped] == [issue_number]

    state = load_state(state_file)
    events = _throttle_window_events(state)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["source"] == "dead_sessions_launch_failure"
    assert payload["previous"] is None
    assert payload["throttled_until"] == state["throttled_until"]
    assert payload["reason"] == "rate_limited"
    assert payload["adapter_kind"] == "devin"


def test_dead_session_reap_emits_throttle_window_set(tmp_path: Path) -> None:
    """Call site 4: a confirmed-dead session (no PID, no launch error)
    classified ``rate_limited`` from the log tail persists the window and
    emits ``throttle_window_set``."""
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    issue_number = 2009
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    _write_devin_sidecar(sessions_dir, issue_number, _RATE_LIMIT_LOG, error=None)
    state_file = tmp_path / "state.json"
    _seed_state_with_issue(state_file, issue_number)

    reaped = _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, state_file, FakeGitHub(), _dead_lane_config(), write_gate=_wg(state_file)
    )
    assert [r["issue_number"] for r in reaped] == [issue_number]

    state = load_state(state_file)
    events = _throttle_window_events(state)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["source"] == "dead_sessions_reap"
    assert payload["previous"] is None
    assert payload["throttled_until"] == state["throttled_until"]
    assert payload["reason"] == "rate_limited"
    assert payload["adapter_kind"] == "devin"


def test_reconcile_apply_fixes_emits_throttle_window_set(tmp_path: Path) -> None:
    """Call site 5: ``apply_fixes`` handling a ``provider_throttle_detected``
    drift item writes the window and emits ``throttle_window_set``, threaded
    through its ``state_path`` so the row lands in events.db."""
    state_path = tmp_path / "state.json"
    config = OrchestratorConfig()
    gh = FakeGitHub()

    throttled_until = _future(minutes=10)
    drift = [
        DriftItem(
            kind="provider_throttle_detected",
            issue_number=42,
            pr_number=None,
            detail="issue #42 session died with provider_auth",
            fix_actions=(f"set throttled_until={throttled_until}",),
            throttle_reason="provider_auth",
            throttle_adapter_kind="devin",
        )
    ]

    new_state = apply_fixes(gh, empty_state(), drift, config, state_path=state_path)

    assert new_state["throttled_until"] == throttled_until
    events = _throttle_window_events(new_state)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["source"] == "reconcile_apply_fixes"
    assert payload["previous"] is None
    assert payload["throttled_until"] == throttled_until
    assert payload["reason"] == "provider_auth"
    assert payload["adapter_kind"] == "devin"

    db_events = [e for e in read_event_log(state_path) if e["kind"] == "throttle_window_set"]
    assert len(db_events) == 1
    assert db_events[0]["level"] == "info"


def test_orphan_sweep_classification_emits_throttle_window_set(
    tmp_path: Path, monkeypatch
) -> None:
    """Call site 6: the state.json-keyed orphan sweep's classify-at-credit
    seam (``dead_worker_classification``, reached through
    ``orphaned_worker_sweep.handle_dead_worker_with_pr``) arms the cooldown
    for a throttle-classified death inside the sweep's ``state_lock``.
    Threaded through the sweep's ``WriteGate``, the audit row reaches
    events.db -- not only the in-memory ring, which is where this lane's
    event used to stop."""
    config, paths, fake_gh, _ = _dead_worker_rework_bed(tmp_path)
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    _write_devin_sidecar(
        sessions_dir,
        207,
        "applying rework\nReached free model rate limit\n",
        error=None,
        pid=99999,
    )

    _run_orphan_sweep(tmp_path, paths, config, fake_gh, monkeypatch=monkeypatch)

    state = load_state(paths.state_file)
    # The classification still lands: kind stamped, cooldown armed, the
    # provider-throttle death never credited.
    entry = state["issues"]["207"]
    assert entry["dead_worker_failure_kind"] == "rate_limited"
    assert state["throttled_until"]

    events = _throttle_window_events(state)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["source"] == "dead_worker_classification"
    assert payload["previous"] is None
    assert payload["throttled_until"] == state["throttled_until"]
    assert payload["reason"] == "rate_limited"
    assert payload["adapter_kind"] == "devin"

    db_events = [e for e in read_event_log(paths.state_file) if e["kind"] == "throttle_window_set"]
    assert len(db_events) == 1
    assert db_events[0]["level"] == "info"
    assert db_events[0]["payload"]["source"] == "dead_worker_classification"
    close_db(paths.state_file)


def test_orphan_sweep_classification_dry_run_suppresses_throttle_window_set(
    tmp_path: Path, monkeypatch
) -> None:
    """The threaded gate's dry-run suppression applies to this lane too:
    under ``dry_run=True`` no ``throttle_window_set`` is emitted and no
    events.db exists at all -- the same footprint as a pass that never ran
    (the WriteGate invariant), with state.json left untouched."""
    config, paths, fake_gh, _ = _dead_worker_rework_bed(tmp_path)
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    _write_devin_sidecar(
        sessions_dir,
        207,
        "applying rework\nReached free model rate limit\n",
        error=None,
        pid=99999,
    )

    _run_orphan_sweep(tmp_path, paths, config, fake_gh, dry_run=True, monkeypatch=monkeypatch)

    state = load_state(paths.state_file)
    assert not state.get("throttled_until")
    assert state["issues"]["207"]["status"] == "dispatched"
    assert "dead_worker_failure_kind" not in state["issues"]["207"]
    assert not (paths.state_file.parent / "events.db").exists()
