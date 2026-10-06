"""Issue #2086: worker role chain, app level, through the real launch gate.

Two repos share one fleet directory (conftest's ``CHARLIE_WORK_FLEET_DIR``).
A quota death classified on repo A's primary-entry session restricts that
``(harness, model)`` fleet-wide, so repo B's next launch -- fresh dispatch or
remote rework -- selects the next chain entry and carries the fallback fields;
when the restriction expires the next launch is back on the primary; with every
entry restricted the lane defers with no launch. The fallback session is then
classified with its own recorded harness and restricts its own key.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _fakes_github import FakeGitHub
from charlie_work import role_quota_ledger
from charlie_work.adapters import AdapterSettings, SessionDispatchResult, SessionRequest
from charlie_work.config import (
    DispatchConfig,
    OrchestratorConfig,
    WorkerRoleConfig,
)
from charlie_work.host.fakes import FakeWorkerLauncher
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.role_chain import RoleEntry
from charlie_work.state import load_state, save_state, set_throttled_until, state_lock
from charlie_work.worker import iter_workers
from charlie_work.worker_launch_gate import (
    REASON_PROVIDER_THROTTLED,
    WorkerLaunchDeferral,
    WorkerLaunchPermit,
    issue_worker_launch_permit,
)
from charlie_work.worker_fate import profile_for
from charlie_work.workflow import OrchestratorApp

FRESH = "fresh"
REWORK = "rework"
LANES = pytest.mark.parametrize("lane", [FRESH, REWORK])

PRIMARY = RoleEntry("claude-code", "claude-sonnet-5-5")
FALLBACK = RoleEntry("devin-shell", "swe-2")
CHAINED = WorkerRoleConfig(harness=PRIMARY.harness, model=PRIMARY.model, fallbacks=(FALLBACK,))
QUOTA_LOG = "Some work done...\nError: daily usage quota has been exhausted. Try tomorrow.\n"


def _app(root: Path, lane: str, worker: WorkerRoleConfig = CHAINED) -> OrchestratorApp:
    root.mkdir(parents=True, exist_ok=True)
    config = OrchestratorConfig(worker=worker, dispatch=DispatchConfig(default_limit=5))
    paths = runtime_paths(root, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    fake_gh = FakeGitHub()
    if lane == REWORK:
        with state_lock(paths.state_file):
            state = load_state(paths.state_file)
            state["issues"]["123"] = {"number": 123, "status": "rework_requested"}
            state["prs"]["456"] = {"number": 456, "issue_number": 123}
            save_state(paths.state_file, state)
    else:
        fake_gh.prs[0]["state"] = "CLOSED"
    app = OrchestratorApp(root, paths, config, fake_gh, fleet_dir_override=None)
    if lane == REWORK:
        pr_dir = paths.prs / "pr-456"
        pr_dir.mkdir(parents=True, exist_ok=True)
        (pr_dir / "rework-prompt.md").write_text("rework prompt", encoding="utf-8")
    return app


def _spy(fake_host) -> list[tuple[AdapterSettings, SessionRequest]]:
    """Fake ``dispatch_sessions`` that writes the sidecar a real launch would."""
    calls: list[tuple[AdapterSettings, SessionRequest]] = []

    def _fake(_repo_root, _manifest, _results, settings, requests):
        results = []
        for request in requests:
            calls.append((settings, request))
            sidecar = role_quota_ledger.sidecar_path_for(
                settings.sessions_dir, settings.adapter, request.issue_number
            )
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            log_path = sidecar.with_name(f"issue-{request.issue_number}.test.log")
            log_path.write_text(QUOTA_LOG, encoding="utf-8")
            sidecar.write_text(
                json.dumps(
                    {
                        "issue_number": request.issue_number,
                        "pid": 999999,
                        "log_path": str(log_path),
                        "branch": request.branch_name,
                        "started_at": "2026-09-30T00:00:00Z",
                    }
                ),
                encoding="utf-8",
            )
            results.append(
                SessionDispatchResult(
                    issue_number=request.issue_number,
                    issue_title=request.issue_title,
                    prompt_path=str(request.prompt_path),
                    branch_name=request.branch_name,
                    adapter=settings.adapter,
                    ok=True,
                    pid=999999,
                    process_start_time=1.0,
                )
            )
        return results

    fake_host(worker_launch=FakeWorkerLauncher([_fake]))
    return calls


def _run(app: OrchestratorApp, lane: str) -> Any:
    return app.dispatch() if lane == FRESH else app.dispatch_rework()


def _launched_entry(settings: AdapterSettings) -> tuple[str, str]:
    """The (harness, model) a launch actually pins, read from the adapter carriers."""
    if settings.adapter == "claude-code":
        return (settings.adapter, settings.config.worker.model)
    return (settings.adapter, settings.worker_model)


def _classify_dead(app: OrchestratorApp) -> list[tuple[str, str | None, str | None]]:
    """Run the real reap classification over every sidecar (sidecar-typed harness)."""
    out = []
    for view in iter_workers(app._layout.sessions_dir):
        profile = profile_for(view.adapter_kind)
        kind, until = profile.record_failure(
            app._layout.sessions_dir, view.issue_number, fallback_kind="stalled", config=app.config
        )
        out.append((view.adapter_kind, kind, until))
    return out


def _stamp(app: OrchestratorApp, harness: str) -> dict[str, Any]:
    sidecar = role_quota_ledger.sidecar_path_for(app._layout.sessions_dir, harness, 123)
    return json.loads(sidecar.read_text(encoding="utf-8"))[role_quota_ledger.SESSION_ROLE_KEY]


@LANES
def test_quota_death_on_primary_moves_other_repo_to_fallback_and_back(
    tmp_path: Path, fake_host, lane: str
) -> None:
    calls = _spy(fake_host)
    repo_a = _app(tmp_path / "a", lane)
    repo_b = _app(tmp_path / "b", lane)

    # Repo A launches on the primary; its session is stamped with entry 0.
    _run(repo_a, lane)
    assert [_launched_entry(s) for s, _ in calls] == [PRIMARY.key]
    assert _stamp(repo_a, PRIMARY.harness) == {
        "role": "worker",
        "harness": PRIMARY.harness,
        "model": PRIMARY.model,
        "chain_index": 0,
    }

    # Its death is classified quota_exhausted -> the primary is restricted fleet-wide.
    [(adapter_kind, kind, until)] = _classify_dead(repo_a)
    assert (adapter_kind, kind) == ("claude-code", "quota_exhausted")
    assert set(role_quota_ledger.load_restrictions()) == {PRIMARY.key}

    # Repo B (no per-repo throttle of its own) launches on the fallback.
    calls.clear()
    result = _run(repo_b, lane)
    assert [_launched_entry(s) for s, _ in calls] == [FALLBACK.key], result.message
    assert _stamp(repo_b, FALLBACK.harness)["chain_index"] == 1
    [event] = query_events(repo_b.paths.state_file, kind="role_fallback_selected")
    payload = event["payload"]
    assert payload["role"] == "worker"
    assert payload["chain_index"] == 1
    assert (payload["harness"], payload["model"]) == FALLBACK.key
    assert payload["skipped"] == [{**PRIMARY.to_payload(), "index": 0, "until": until}]

    # The fallback session dies too; it is classified with ITS OWN harness and
    # restricts its own key, not the primary's.
    classified = _classify_dead(repo_b)
    assert [(k, fk) for k, fk, _ in classified] == [("devin", "quota_exhausted")]
    assert set(role_quota_ledger.load_restrictions()) == {PRIMARY.key, FALLBACK.key}


@LANES
def test_all_entries_restricted_defers_with_no_launch(
    tmp_path: Path, fake_host, lane: str
) -> None:
    calls = _spy(fake_host)
    later = datetime.now(UTC) + timedelta(hours=2)
    for entry in (PRIMARY, FALLBACK):
        role_quota_ledger.record_restriction(
            entry.harness, entry.model, later, reason="quota_exhausted", source="test"
        )
    result = _run(_app(tmp_path / "b", lane), lane)
    assert calls == []
    assert result.data.get("deferred_reason") == "provider_throttled", result.data
    assert result.data.get("chain_length") == 2
    assert [s["index"] for s in result.data.get("skipped", [])] == [0, 1]


@LANES
def test_expired_restriction_returns_to_primary(tmp_path: Path, fake_host, lane: str) -> None:
    calls = _spy(fake_host)
    ledger_path = role_quota_ledger.ledger_path()
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    past = (datetime.now(UTC) - timedelta(minutes=1)).replace(microsecond=0)
    ledger_path.write_text(
        json.dumps(
            {
                "version": 1,
                "restrictions": {
                    "x": {**PRIMARY.to_payload(), "until": past.isoformat()},
                },
            }
        ),
        encoding="utf-8",
    )
    app = _app(tmp_path / "b", lane)
    _run(app, lane)
    assert [_launched_entry(s) for s, _ in calls] == [PRIMARY.key]
    assert query_events(app.paths.state_file, kind="role_fallback_selected") == []


@LANES
def test_per_repo_window_the_ledger_does_not_explain_still_blocks(
    tmp_path: Path, fake_host, lane: str
) -> None:
    """An operator hold / pre-ledger throttle outlasting every ledger entry blocks."""
    calls = _spy(fake_host)
    role_quota_ledger.record_restriction(
        PRIMARY.harness,
        PRIMARY.model,
        datetime.now(UTC) + timedelta(hours=1),
        reason="quota_exhausted",
        source="test",
    )
    app = _app(tmp_path / "b", lane)
    hold = (datetime.now(UTC) + timedelta(hours=5)).replace(microsecond=0)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state = set_throttled_until(state, hold.isoformat().replace("+00:00", "Z"), source="t")
        save_state(app.paths.state_file, state)
    _run(app, lane)
    assert calls == []


@LANES
def test_length_one_chain_is_unchanged_by_the_ledger(tmp_path: Path, fake_host, lane: str) -> None:
    """Existing behavior: no fallbacks -> the primary launches regardless of the
    ledger, and a per-repo throttle blocks exactly as before."""
    calls = _spy(fake_host)
    role_quota_ledger.record_restriction(
        PRIMARY.harness,
        PRIMARY.model,
        datetime.now(UTC) + timedelta(hours=1),
        reason="quota_exhausted",
        source="test",
    )
    single = WorkerRoleConfig(harness=PRIMARY.harness, model=PRIMARY.model)
    app = _app(tmp_path / "b", lane, worker=single)
    result = _run(app, lane)
    assert [_launched_entry(s) for s, _ in calls] == [PRIMARY.key]
    assert "chain_length" not in result.data
    assert query_events(app.paths.state_file, kind="role_fallback_selected") == []

    calls.clear()
    app2 = _app(tmp_path / "c", lane, worker=single)
    until = (datetime.now(UTC) + timedelta(hours=1)).replace(microsecond=0)
    with state_lock(app2.paths.state_file):
        state = load_state(app2.paths.state_file)
        state = set_throttled_until(state, until.isoformat().replace("+00:00", "Z"), source="t")
        save_state(app2.paths.state_file, state)
    result = _run(app2, lane)
    assert calls == []
    assert result.data.get("deferred_reason") == "provider_throttled"


@LANES
def test_same_repo_throttle_explained_by_the_ledger_is_covered(
    tmp_path: Path, fake_host, lane: str
) -> None:
    """The dying repo's own per-repo window (same ``until`` as the ledger entry,
    written by ``persist_failure``) does not stop it launching the fallback."""
    calls = _spy(fake_host)
    until = (datetime.now(UTC) + timedelta(hours=2)).replace(microsecond=0)
    until_z = until.isoformat().replace("+00:00", "Z")
    role_quota_ledger.record_restriction(
        PRIMARY.harness, PRIMARY.model, until, reason="quota_exhausted", source="test"
    )
    app = _app(tmp_path / "a", lane)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        stamped = set_throttled_until(
            state, until_z, source="t", reason="quota_exhausted", adapter_kind="claude-code"
        )
        save_state(app.paths.state_file, stamped)
    result = _run(app, lane)
    assert [_launched_entry(s) for s, _ in calls] == [FALLBACK.key], result.message


def _hold(app: OrchestratorApp, until: datetime, **provenance: str | None) -> None:
    """Write a per-repo throttle window; ``provenance`` is reason/adapter_kind."""
    until_z = until.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        save_state(
            app.paths.state_file, set_throttled_until(state, until_z, source="t", **provenance)
        )


def _restrict(entry: RoleEntry, until: datetime) -> None:
    role_quota_ledger.record_restriction(
        entry.harness, entry.model, until, reason="quota_exhausted", source="test"
    )


@LANES
def test_unstamped_hold_ending_before_a_skipped_entrys_restriction_still_blocks(
    tmp_path: Path, fake_host, lane: str
) -> None:
    """An operator hold (no provenance) that is SHORTER than the primary's ledger
    restriction used to be waved through as "explained"; it must still block."""
    calls = _spy(fake_host)
    _restrict(PRIMARY, datetime.now(UTC) + timedelta(hours=3))
    app = _app(tmp_path / "b", lane)
    _hold(app, datetime.now(UTC) + timedelta(hours=1))
    result = _run(app, lane)
    assert calls == []
    assert result.data.get("deferred_reason") == "provider_throttled", result.data


@LANES
def test_hold_is_not_explained_by_a_restriction_on_an_unselected_later_entry(
    tmp_path: Path, fake_host, lane: str
) -> None:
    """Primary free, only the FALLBACK restricted: selection skipped nothing, so
    even a fully stamped quota window on the primary's adapter blocks."""
    calls = _spy(fake_host)
    _restrict(FALLBACK, datetime.now(UTC) + timedelta(hours=3))
    app = _app(tmp_path / "b", lane)
    _hold(
        app,
        datetime.now(UTC) + timedelta(hours=1),
        reason="quota_exhausted",
        adapter_kind="claude-code",
    )
    result = _run(app, lane)
    assert calls == []
    assert result.data.get("deferred_reason") == "provider_throttled", result.data


@LANES
def test_stamped_window_on_an_adapter_no_skipped_entry_uses_still_blocks(
    tmp_path: Path, fake_host, lane: str
) -> None:
    calls = _spy(fake_host)
    _restrict(PRIMARY, datetime.now(UTC) + timedelta(hours=3))  # claude-code skipped
    app = _app(tmp_path / "b", lane)
    _hold(
        app,
        datetime.now(UTC) + timedelta(hours=1),
        reason="quota_exhausted",
        adapter_kind="devin",
    )
    _run(app, lane)
    assert calls == []


@pytest.mark.parametrize("stamped", [False, True], ids=["unstamped-blocks", "stamped-covered"])
def test_under_lock_throttle_read_applies_the_same_attribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stamped: bool
) -> None:
    """The window lands AFTER the lock-free pre-check (written from inside the
    governor call), so only the authoritative under-lock read can see it."""
    _restrict(PRIMARY, datetime.now(UTC) + timedelta(hours=3))
    app = _app(tmp_path / "b", FRESH)
    real_governor = app._apply_concurrency_governor

    def _throttle_then_govern(*args: Any, **kwargs: Any) -> Any:
        provenance = (
            {"reason": "quota_exhausted", "adapter_kind": "claude-code"} if stamped else {}
        )
        _hold(app, datetime.now(UTC) + timedelta(hours=1), **provenance)
        return real_governor(*args, **kwargs)

    monkeypatch.setattr(app, "_apply_concurrency_governor", _throttle_then_govern)
    outcome = issue_worker_launch_permit(app, 1)
    if stamped:
        assert isinstance(outcome, WorkerLaunchPermit), outcome
        assert outcome.role_selection.entry == FALLBACK
        outcome.release()
    else:
        assert isinstance(outcome, WorkerLaunchDeferral), outcome
        assert outcome.reason == REASON_PROVIDER_THROTTLED
        assert outcome.governor is not None  # came from the under-lock gate, not the pre-check


# ---------------------------------------------------------------------------
# Issue #2279: the per-repo window's own (harness, model) stamp
# ---------------------------------------------------------------------------


@LANES
def test_fallback_stamped_window_is_covered_once_the_primary_recovers(
    tmp_path: Path, fake_host, lane: str
) -> None:
    """B's quota death armed a window stamped (devin-shell, swe-2). The primary
    is free, so selection skipped nothing -- only the stamp attributes the
    window to the still-restricted fallback, and the launch proceeds on A."""
    calls = _spy(fake_host)
    _restrict(FALLBACK, datetime.now(UTC) + timedelta(hours=3))
    app = _app(tmp_path / "b", lane)
    _hold(
        app,
        datetime.now(UTC) + timedelta(hours=1),
        reason="quota_exhausted",
        adapter_kind="devin",
        harness=FALLBACK.harness,
        model=FALLBACK.model,
    )
    _run(app, lane)
    assert [_launched_entry(s) for s, _ in calls] == [PRIMARY.key]


@LANES
def test_window_stamped_with_the_selected_entry_itself_still_blocks(
    tmp_path: Path, fake_host, lane: str
) -> None:
    """An entry's own death window is never covered -- the stamp equal to the
    selected entry is the one attribution the rule must refuse."""
    calls = _spy(fake_host)
    app = _app(tmp_path / "b", lane)
    _hold(
        app,
        datetime.now(UTC) + timedelta(hours=1),
        reason="quota_exhausted",
        adapter_kind="claude-code",
        harness=PRIMARY.harness,
        model=PRIMARY.model,
    )
    result = _run(app, lane)
    assert calls == []
    assert result.data.get("deferred_reason") == "provider_throttled", result.data


@LANES
def test_fallback_stamped_window_still_blocks_when_its_ledger_entry_is_shorter(
    tmp_path: Path, fake_host, lane: str
) -> None:
    """The ledger explains the window only through the stamped entry's
    restriction; a window outliving it is part unexplained and blocks."""
    calls = _spy(fake_host)
    _restrict(FALLBACK, datetime.now(UTC) + timedelta(hours=1))
    app = _app(tmp_path / "b", lane)
    _hold(
        app,
        datetime.now(UTC) + timedelta(hours=3),
        reason="quota_exhausted",
        adapter_kind="devin",
        harness=FALLBACK.harness,
        model=FALLBACK.model,
    )
    result = _run(app, lane)
    assert calls == []
    assert result.data.get("deferred_reason") == "provider_throttled", result.data


def test_under_lock_throttle_read_applies_the_stamped_entry_attribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#2279 under-lock arm: the stamped window lands after the lock-free
    pre-check; the authoritative read applies the same coverage."""
    _restrict(FALLBACK, datetime.now(UTC) + timedelta(hours=3))
    app = _app(tmp_path / "b", FRESH)
    real_governor = app._apply_concurrency_governor

    def _throttle_then_govern(*args: Any, **kwargs: Any) -> Any:
        _hold(
            app,
            datetime.now(UTC) + timedelta(hours=1),
            reason="quota_exhausted",
            adapter_kind="devin",
            harness=FALLBACK.harness,
            model=FALLBACK.model,
        )
        return real_governor(*args, **kwargs)

    monkeypatch.setattr(app, "_apply_concurrency_governor", _throttle_then_govern)
    outcome = issue_worker_launch_permit(app, 1)
    assert isinstance(outcome, WorkerLaunchPermit), outcome
    assert outcome.role_selection.entry == PRIMARY
    outcome.release()
