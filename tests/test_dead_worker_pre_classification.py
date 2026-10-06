"""Issue #2274: a rate-limit death reaches the role waterfall ledger through the real lanes.

Reconcile was the only lane that classified a dead worker's log. Every death
inside its ~31-minute interval reached the orphan sweep's no-PR arm, then the
redispatch overwrite and the phantom reap, and none of them classified it. The
tests here run the real ``_detect_and_handle_orphaned_workers`` sweep and the
real ``adapters.dispatch_sessions`` -> ``devin_shell.launch_devin_session``
overwrite. They never call ``profile.record_failure`` directly, which is how
``tests/test_role_waterfall_workers.py`` missed the gap.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import _wg
from _worktree_fixtures import _wt_scratch as _register_wt_scratch  # noqa: F401 -- registers the wt_scratch fixture (shallow tmp dir for real ``git worktree add``)
from _host_fixtures import host_probe
from charlie_work import role_quota_ledger, worker_fate
from charlie_work.adapters import (
    AdapterSettings,
    SessionDispatchResult,
    SessionRequest,
    dispatch_sessions,
)
from charlie_work.config import (
    DevinConfig,
    DispatchConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.role_chain import RoleEntry
from charlie_work.state import load_state, save_state
from charlie_work.worker import iter_workers
from charlie_work.workflow import OrchestratorApp, _detect_and_handle_orphaned_workers
from charlie_work.worktree import LiveWorkerRedispatchError

PRIMARY = RoleEntry("devin-shell", "swe-2-high")
FALLBACK = RoleEntry("devin-shell", "gemini-flash")
CHAIN = WorkerRoleConfig(harness=PRIMARY.harness, model=PRIMARY.model, fallbacks=(FALLBACK,))
ISSUE = 2269
DEAD_PID = 99999
RATE_LIMIT_LOG = (
    "Working on the issue...\n"
    "Error: Agent error: Reached free model rate limit. Your limit will reset in 14 minutes.\n"
)


def _config() -> OrchestratorConfig:
    return OrchestratorConfig(
        devin=DevinConfig(),
        worker=CHAIN,
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )


def _sessions_dir(root: Path) -> Path:
    sessions_dir = root / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir


def _seed_dead_primary(
    root: Path,
    config: OrchestratorConfig,
    *,
    entry: RoleEntry = PRIMARY,
    chain_index: int = 0,
) -> tuple[Any, Path, FakeGitHub]:
    """Repo A: a dispatched issue whose stamped-entry devin worker died with no PR."""
    paths = runtime_paths(root, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    state = load_state(paths.state_file)
    state["issues"][str(ISSUE)] = {
        "status": "dispatched",
        "dispatched_at": "2026-10-02T04:30:00Z",
        "worker_pid": DEAD_PID,
        "worker_process_start_time": 1784000000.0,
        "branch_name": f"agent/issue-{ISSUE}",
    }
    save_state(paths.state_file, state)

    sessions_dir = _sessions_dir(root)
    log_path = sessions_dir / f"issue-{ISSUE}.log"
    log_path.write_text(RATE_LIMIT_LOG, encoding="utf-8")
    sidecar = {
        "issue_number": ISSUE,
        "branch": f"agent/issue-{ISSUE}",
        "worktree_path": "",
        "prompt_path": "",
        "command": [],
        "pid": DEAD_PID,
        "started_at": "2026-10-02T04:30:00Z",
        "log_path": str(log_path),
        role_quota_ledger.SESSION_ROLE_KEY: role_quota_ledger.session_stamp(
            "worker", entry.harness, entry.model, chain_index
        ),
    }
    (sessions_dir / f"issue-{ISSUE}.json").write_text(json.dumps(sidecar), encoding="utf-8")
    # The terminal record: the worker exited 1 with no outcome -- not a completion,
    # so the #656 guard must not suppress the classification.
    (sessions_dir / f"issue-{ISSUE}.devin.terminal.json").write_text(
        json.dumps({"pid": DEAD_PID, "exit_code": 1, "ended_at": "2026-10-02T04:45:00Z"}),
        encoding="utf-8",
    )

    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": ISSUE,
            "title": "rate limited worker",
            "url": f"https://example.test/issues/{ISSUE}",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []
    return paths, sessions_dir, fake_gh


def _sweep(paths: Any, sessions_dir: Path, config: OrchestratorConfig, gh: FakeGitHub) -> None:
    with host_probe(alive=False):
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, gh, write_gate=_wg(paths.state_file)
        )


def _redispatch_averted(root: Path, sessions_dir: Path, config: OrchestratorConfig) -> None:
    """The real redispatch overwrite: ``launch_devin_session`` hits the (false)
    live-worker probe and writes its ``live_worker_redispatch_averted`` record over
    the dead session's sidecar."""

    def _live(*_args: Any, **_kwargs: Any) -> Any:
        raise LiveWorkerRedispatchError(
            issue_number=ISSUE,
            pid=DEAD_PID,
            process_start_time=None,
            probe_result="devin_per_pid_log_activity",
        )

    prompt = root / "prompt.md"
    prompt.write_text("prompt", encoding="utf-8")
    settings = AdapterSettings(
        adapter="devin-shell",
        sessions_dir=sessions_dir,
        worker_model=PRIMARY.model,
        config=config,
    )
    request = SessionRequest(ISSUE, "rate limited worker", prompt, f"agent/issue-{ISSUE}")
    with patch("charlie_work.devin_shell.create_worktree", side_effect=_live):
        [result] = dispatch_sessions(
            root, root / "manifest.json", root / "results.json", settings, [request]
        )
    assert result.failure_kind == "live_worker_redispatch_averted"
    sidecar = json.loads((sessions_dir / f"issue-{ISSUE}.json").read_text(encoding="utf-8"))
    assert sidecar["failure_kind"] == "live_worker_redispatch_averted"  # the overwrite happened


def _phantom_reap(sessions_dir: Path) -> None:
    """The phantom path's unclassified ``reap_sidecar`` (misc_worker_dispatch)."""
    for view in iter_workers(sessions_dir):
        if view.issue_number == ISSUE:
            view.reap_sidecar(sessions_dir)
    assert not (sessions_dir / f"issue-{ISSUE}.json").exists()


def _assert_primary_restricted() -> None:
    assert set(role_quota_ledger.load_restrictions()) == {PRIMARY.key}


def _assert_repo_stamped(paths: Any, entry: RoleEntry = PRIMARY) -> None:
    state = load_state(paths.state_file)
    assert worker_fate.persisted_failure(state["issues"][str(ISSUE)]).kind == "rate_limited"
    assert state.get("throttled_until")
    assert state.get("throttle_reason") == "rate_limited"
    # Issue #2279: the window carries the dead session's stamped role entry.
    assert state.get("throttle_harness") == entry.harness
    assert state.get("throttle_model") == entry.model
    windows = query_events(paths.state_file, kind="throttle_window_set")
    assert len(windows) == 1, windows
    assert windows[0]["payload"]["source"] == "dead_worker_classification"


def _fallback_app(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[OrchestratorApp, list]:
    """Repo B: a fresh app on the same chain with a fake launcher that records settings."""
    root.mkdir(parents=True, exist_ok=True)
    config = OrchestratorConfig(worker=CHAIN, dispatch=DispatchConfig(default_limit=5))
    paths = runtime_paths(root, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    fake_gh = FakeGitHub()
    fake_gh.prs[0]["state"] = "CLOSED"
    calls: list[AdapterSettings] = []

    def _fake(_repo_root, _manifest, _results, settings, requests):
        calls.append(settings)
        return [
            SessionDispatchResult(
                issue_number=r.issue_number,
                issue_title=r.issue_title,
                prompt_path=str(r.prompt_path),
                branch_name=r.branch_name,
                adapter=settings.adapter,
                ok=True,
                pid=424242,
                process_start_time=1.0,
            )
            for r in requests
        ]

    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", _fake)
    return OrchestratorApp(root, paths, config, fake_gh, fleet_dir_override=None), calls


def _assert_repo_b_selects_fallback(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, calls = _fallback_app(root, monkeypatch)
    result = app.dispatch()
    assert [s.worker_model for s in calls] == [FALLBACK.model], result.message
    [event] = query_events(app.paths.state_file, kind="role_fallback_selected")
    assert (event["payload"]["harness"], event["payload"]["model"]) == FALLBACK.key
    assert event["payload"]["chain_index"] == 1


def test_one_sweep_pass_classifies_the_death_and_feeds_the_waterfall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    paths, sessions_dir, gh = _seed_dead_primary(tmp_path / "a", config)

    _sweep(paths, sessions_dir, config, gh)

    _assert_primary_restricted()
    _assert_repo_stamped(paths)
    # The no-PR arm still ran (the classification did not short-circuit it).
    relabeled = query_events(paths.state_file, kind="session_failed_relabeled")
    assert [e["payload"]["reason"] for e in relabeled] == ["dead_worker_no_open_pr_orphan_sweep"]
    sidecar = json.loads((sessions_dir / f"issue-{ISSUE}.json").read_text(encoding="utf-8"))
    assert sidecar["failure_kind"] == "rate_limited"

    _assert_repo_b_selects_fallback(tmp_path / "b", monkeypatch)


def test_redispatch_and_phantom_in_the_same_pass_keep_the_ledger_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    paths, sessions_dir, gh = _seed_dead_primary(tmp_path / "a", config)

    _sweep(paths, sessions_dir, config, gh)
    _redispatch_averted(tmp_path / "a", sessions_dir, config)
    _phantom_reap(sessions_dir)

    _assert_primary_restricted()
    _assert_repo_stamped(paths)
    _assert_repo_b_selects_fallback(tmp_path / "b", monkeypatch)


def test_redispatch_before_any_sweep_classifies_the_dead_sidecar_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard 2: the overwrite path alone (no sweep reached the death) still records
    the restriction before the averted record replaces the evidence."""
    config = _config()
    _paths, sessions_dir, _gh = _seed_dead_primary(tmp_path / "a", config)

    _redispatch_averted(tmp_path / "a", sessions_dir, config)
    _phantom_reap(sessions_dir)

    _assert_primary_restricted()
    _assert_repo_b_selects_fallback(tmp_path / "b", monkeypatch)


def test_redispatch_guard_never_classifies_a_live_worker(tmp_path: Path) -> None:
    config = _config()
    _paths, sessions_dir, _gh = _seed_dead_primary(tmp_path / "a", config)

    with patch.object(worker_fate, "is_alive", return_value=True):
        _redispatch_averted(tmp_path / "a", sessions_dir, config)

    assert role_quota_ledger.load_restrictions() == {}


def test_completed_worker_is_not_classified_by_the_sweep(tmp_path: Path) -> None:
    """#656 guard: a terminal record proving this pid exited 0 with an outcome makes
    the log tail completion prose, so the sweep's classification stays off."""
    config = _config()
    paths, sessions_dir, gh = _seed_dead_primary(tmp_path / "a", config)
    (sessions_dir / f"issue-{ISSUE}.devin.terminal.json").write_text(
        json.dumps(
            {
                "pid": DEAD_PID,
                "exit_code": 0,
                "ended_at": "2026-10-02T04:45:00Z",
                "worker_outcome": {"status": "completed"},
            }
        ),
        encoding="utf-8",
    )

    _sweep(paths, sessions_dir, config, gh)

    assert role_quota_ledger.load_restrictions() == {}
    state = load_state(paths.state_file)
    assert worker_fate.persisted_failure(state["issues"][str(ISSUE)]).kind is None
    assert not state.get("throttled_until")


def test_dry_run_sweep_does_not_classify(tmp_path: Path) -> None:
    config = _config()
    paths, sessions_dir, gh = _seed_dead_primary(tmp_path / "a", config)

    with host_probe(alive=False):
        _detect_and_handle_orphaned_workers(
            sessions_dir,
            paths.state_file,
            config,
            gh,
            write_gate=_wg(paths.state_file, dry_run=True),
        )

    assert role_quota_ledger.load_restrictions() == {}
    sidecar = json.loads((sessions_dir / f"issue-{ISSUE}.json").read_text(encoding="utf-8"))
    assert "failure_kind" not in sidecar or sidecar["failure_kind"] is None


def test_fallback_quota_death_does_not_block_the_recovered_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #2279 acceptance: chain [A, B] on ONE adapter; B dies
    ``rate_limited`` while A's own restriction has already lapsed.

    Before the fix, B's death armed an adapter-wide per-repo window that
    pinned the recovered primary for the full backoff. Now the window is
    stamped with B's ``(harness, model)`` and covered by B's own ledger
    entry, so the same repo's next launch proceeds on A.
    """
    config = _config()
    paths, sessions_dir, gh = _seed_dead_primary(
        tmp_path / "a", config, entry=FALLBACK, chain_index=1
    )
    # A's earlier restriction already expired -- B's is the only live one.
    role_quota_ledger.record_restriction(
        PRIMARY.harness,
        PRIMARY.model,
        datetime.now(UTC) - timedelta(minutes=1),
        reason="rate_limited",
        source="test",
    )

    _sweep(paths, sessions_dir, config, gh)

    _assert_repo_stamped(paths, entry=FALLBACK)
    assert set(role_quota_ledger.load_restrictions()) == {PRIMARY.key, FALLBACK.key}

    # Same repo, same window: selection returns A with nothing skipped and
    # B's stamped window is covered by B's ledger restriction.
    app, calls = _fallback_app(tmp_path / "a", monkeypatch)
    app.dispatch()
    assert [s.worker_model for s in calls] == [PRIMARY.model]
    assert query_events(app.paths.state_file, kind="role_fallback_selected") == []


def test_fallback_quota_death_window_blocks_when_its_ledger_entry_lapses_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Counter-case: once B's own ledger restriction also expires, its window
    is unexplained and blocks again -- a stamped window is not a free pass."""
    config = _config()
    paths, sessions_dir, gh = _seed_dead_primary(
        tmp_path / "a", config, entry=FALLBACK, chain_index=1
    )
    _sweep(paths, sessions_dir, config, gh)
    _assert_repo_stamped(paths, entry=FALLBACK)

    # B's ledger restriction lapses while the per-repo window still runs.
    ledger_path = role_quota_ledger.ledger_path()
    raw = json.loads(ledger_path.read_text(encoding="utf-8"))
    past = _z_datetime(datetime.now(UTC) - timedelta(minutes=1))
    raw["restrictions"][f"{FALLBACK.harness}|{FALLBACK.model}"]["until"] = past
    ledger_path.write_text(json.dumps(raw), encoding="utf-8")

    app, calls = _fallback_app(tmp_path / "a", monkeypatch)
    result = app.dispatch()
    assert calls == []
    assert result.data.get("deferred_reason") == "provider_throttled", result.data


def _z_datetime(moment: datetime) -> str:
    return moment.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def test_orphan_sweep_death_event_carries_the_throttle_classification(tmp_path: Path) -> None:
    """Issue #2289: the no-PR orphan sweep's death event names the throttle kind.

    ``attempt_resume`` reads the failure kind off this event, so a
    ``session_failed_relabeled`` with ``failure_kind: None`` would hide every
    throttle death this lane handles.
    """
    config = _config()
    paths, sessions_dir, gh = _seed_dead_primary(tmp_path / "a", config)

    _sweep(paths, sessions_dir, config, gh)

    [event] = query_events(paths.state_file, kind="session_failed_relabeled")
    assert event["payload"]["reason"] == "dead_worker_no_open_pr_orphan_sweep"
    assert event["payload"]["failure_kind"] == "rate_limited"


def test_orphan_sweep_throttle_death_then_rescue_ref_seeds_the_redispatch(
    wt_scratch: Path,
) -> None:
    """End to end: the real sweep's event plus a rescue ref makes create_worktree seed."""
    from _time_shift import shifted_datetime
    from _worktree_fixtures import _init_repo
    from charlie_work import worktree as worktree_module
    from charlie_work.worktree import (
        _capture_worktree_work_to_rescue_ref,
        create_worktree,
        remove_worktree,
    )

    root = wt_scratch / "repo"
    _init_repo(root)
    config = _config()
    paths, sessions_dir, gh = _seed_dead_primary(root, config)
    branch = f"agent/issue-{ISSUE}"
    info = create_worktree(root, branch, base_ref="HEAD", issue_number=ISSUE)
    (info.path / "work.txt").write_text("603 lines of real work\n", encoding="utf-8")

    _sweep(paths, sessions_dir, config, gh)
    # The rescue ref is made at redispatch, after the death: stamp it 2 s later
    # instead of sleeping across the whole-second boundary.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(worktree_module, "datetime", shifted_datetime(2))
        capture = _capture_worktree_work_to_rescue_ref(root, info.path, ISSUE)
    assert capture.error is None
    assert remove_worktree(root, info.path, force=True, branch=branch)

    redispatched = create_worktree(
        root, branch, base_ref="HEAD", issue_number=ISSUE, config=config
    )

    assert redispatched.resumed_attempt is not None
    assert redispatched.resumed_attempt.ref == capture.ref_name
    assert (redispatched.path / "work.txt").read_text(encoding="utf-8") == (
        "603 lines of real work\n"
    )
