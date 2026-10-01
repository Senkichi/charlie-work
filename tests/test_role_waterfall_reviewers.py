"""Issue #2086: reviewer role chain, app level, through ``dispatch_reviews``.

The reviewer launchers are replaced (``_REVIEW_LAUNCHERS``) by a recorder that
writes the sidecar a real launch would, so each test asserts which harness,
model, effort and config a launch was actually handed. The per-repo
``reviewer_quota`` gate keeps its meaning for a length-1 chain (existing
behavior, including the single-probe recovery); for a chained role a quota
window the fleet ledger explains is covered and the fallback launches.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _fakes_github import FakeGitHub
from _helpers import _init_git_repo
from _review_fixtures import _fake_claude_worker_record, _write_review_packet
from charlie_work import role_quota_ledger
from charlie_work.config import OrchestratorConfig, ReviewDispatchConfig, ReviewerRoleConfig
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.role_chain import RoleEntry
from charlie_work.stalled_review_reap import _detect_and_handle_stalled_reviews
from charlie_work.state import load_state, save_state, set_reviewer_quota_exhausted, state_lock
from charlie_work.workflow import OrchestratorApp
from charlie_work.write_gate import WriteGate


PRIMARY = RoleEntry("devin-shell", "swe-2")
FALLBACK = RoleEntry("claude-code", "claude-sonnet-5-5", "high")
CHAINED = ReviewerRoleConfig(harness=PRIMARY.harness, model=PRIMARY.model, fallbacks=(FALLBACK,))
PRIMARY_ADAPTER = "devin"  # WorkerView.adapter_kind of the devin-shell harness
SINGLE = ReviewerRoleConfig(harness=PRIMARY.harness, model=PRIMARY.model)


# --- stalled-reviewer seeding (inlined: tests/ forbids cross-test imports) ---

_SESSION_LIMIT_NOTICE = "You've hit your session limit · resets 4:40pm (America/Los_Angeles)"


def _wg(state_file: Path) -> WriteGate:
    return WriteGate(dry_run=False, state_path=state_file, repo="charlie-work")


def _hour_ago() -> str:
    return (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")


def _seed_stalled(tmp_path: Path, pr_number: int):
    """One dispatched reviewer whose recorded pid is dead."""
    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    config = OrchestratorConfig(review_dispatch=ReviewDispatchConfig(enabled=True))
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}), encoding="utf-8"
    )
    with state_lock(state_file):
        state = load_state(state_file)
        state["prs"][str(pr_number)] = {
            "number": pr_number,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": _hour_ago(),
            "reviewer_pid": 999999999,
            "reviewer_process_start_time": 1.0,
        }
        save_state(state_file, state)
    return repo_root, reviews_dir, config, state_file


def _write_session_limit_reviewer(reviews_dir: Path, pr_number: int, tmp_path: Path) -> Path:
    """A dead claude-code reviewer whose log shows the session-limit notice."""
    log_path = reviews_dir / f"issue-{pr_number}-review.claude.log"
    log_path.write_text(_SESSION_LIMIT_NOTICE + "\n", encoding="utf-8")
    sidecar = {
        "issue_number": pr_number,
        "branch": f"agent/issue-{pr_number}-fix",
        "worktree_path": str(tmp_path / "worktrees" / f"issue-{pr_number}"),
        "prompt_path": str(tmp_path / "prompt.md"),
        "command": ["claude", "-p"],
        "pid": 999999999,
        "started_at": _hour_ago(),
        "log_path": str(log_path),
        "error": None,
        "process_start_time": 1.0,
    }
    sidecar_path = reviews_dir / f"issue-{pr_number}.claude.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    return sidecar_path


def _pr(number: int) -> dict[str, Any]:
    return {
        "number": number,
        "title": f"Fix #{number}",
        "url": f"https://example.test/pull/{number}",
        "headRefName": f"agent/issue-{number}-fix",
        "baseRefName": "main",
        "headRefOid": f"sha-{number}",
        "mergeStateStatus": "CLEAN",
        "body": f"Closes #{number}",
        "labels": [],
        "isCrossRepository": False,
        "state": "OPEN",
    }


def _app(root: Path, reviewer: ReviewerRoleConfig = CHAINED, n_prs: int = 1) -> OrchestratorApp:
    root.mkdir(parents=True, exist_ok=True)
    config = OrchestratorConfig(
        reviewer=reviewer, review_dispatch=ReviewDispatchConfig(enabled=True)
    )
    paths = runtime_paths(root, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}), encoding="utf-8"
    )
    prs = [_pr(300 + i) for i in range(n_prs)]
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = prs
    for pr in prs:
        _write_review_packet(root, pr["number"], pr["headRefOid"])
    return OrchestratorApp(root, paths, config, fake_gh)


def _recorder(monkeypatch: pytest.MonkeyPatch, *, error: str | None = None) -> list[dict]:
    calls: list[dict] = []

    def _launch(harness: str, kwargs: dict[str, Any]):
        calls.append({"harness": harness, **kwargs})
        pr = kwargs["pr_number"]
        record = _fake_claude_worker_record(pr, kwargs["branch"])
        if error is not None:
            from dataclasses import replace

            return replace(record, error=error, pid=None)
        sidecar = role_quota_ledger.sidecar_path_for(kwargs["reviews_dir"], harness, pr)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps({"issue_number": pr, "pid": 12345}), encoding="utf-8")
        return record

    import dataclasses

    from charlie_work import host as host_pkg
    from charlie_work.host.fakes import FakeReviewLauncher

    monkeypatch.setattr(
        host_pkg,
        "_ACTIVE",
        dataclasses.replace(host_pkg.current(), launch=FakeReviewLauncher([_launch])),
    )
    return calls


def _restrict(entry: RoleEntry, until: datetime) -> None:
    role_quota_ledger.record_restriction(
        entry.harness, entry.model, until, reason="quota_exhausted", source="test"
    )


def _z(moment: datetime) -> str:
    return moment.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _launched(calls: list[dict]) -> list[tuple[str, str | None, str | None, str]]:
    return [
        (
            c["harness"],
            c["model_override"],
            c["resolved_review_effort"],
            c["config"].reviewer.harness,
        )
        for c in calls
    ]


def test_primary_restricted_elsewhere_launches_fallback_with_its_own_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _recorder(monkeypatch)
    until = datetime.now(UTC) + timedelta(hours=2)
    _restrict(PRIMARY, until)
    app = _app(tmp_path / "b")

    result = app.dispatch_reviews()

    assert _launched(calls) == [("claude-code", FALLBACK.model, "high", "claude-code")]
    assert result.data["chain_index"] == 1
    assert result.data["probe_mode"] is False
    sidecar = role_quota_ledger.sidecar_path_for(app._layout.reviews_dir, "claude-code", 300)
    assert json.loads(sidecar.read_text(encoding="utf-8"))["role_entry"] == {
        "role": "reviewer",
        "harness": FALLBACK.harness,
        "model": FALLBACK.model,
        "chain_index": 1,
    }
    [event] = query_events(app.paths.state_file, kind="role_fallback_selected")
    assert event["payload"]["role"] == "reviewer"
    assert event["payload"]["skipped"] == [
        {**PRIMARY.to_payload(), "index": 0, "until": _z(until)}
    ]


def test_covered_per_repo_quota_launches_fallback_not_a_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _recorder(monkeypatch)
    until = datetime.now(UTC) + timedelta(hours=2)
    _restrict(PRIMARY, until)
    app = _app(tmp_path / "a", n_prs=2)
    far_probe = _z(datetime.now(UTC) + timedelta(hours=1))
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state = set_reviewer_quota_exhausted(
            state, throttled_until=_z(until), probe_after=far_probe
        )
        state["reviewer_quota"] = {
            **state["reviewer_quota"],
            "reason": "quota_exhausted",
            "adapter_kind": PRIMARY_ADAPTER,
        }
        save_state(app.paths.state_file, state)

    result = app.dispatch_reviews()

    assert [c[0] for c in _launched(calls)] == ["claude-code", "claude-code"], result.message
    assert result.data["probe_mode"] is False


def test_all_entries_restricted_defers_with_no_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _recorder(monkeypatch)
    until = datetime.now(UTC) + timedelta(hours=2)
    _restrict(PRIMARY, until)
    _restrict(FALLBACK, until + timedelta(minutes=5))

    result = _app(tmp_path / "b").dispatch_reviews()

    assert calls == []
    assert result.data["launched_count"] == 0
    assert result.data["deferred_reason"] == "reviewer_quota_probe_backoff"
    assert result.data["chain_retry_at"] == _z(until)


def test_expired_restriction_returns_to_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _recorder(monkeypatch)
    _restrict(PRIMARY, datetime.now(UTC) - timedelta(seconds=5))
    app = _app(tmp_path / "b")

    result = app.dispatch_reviews()

    assert _launched(calls) == [("devin-shell", PRIMARY.model, "", "devin-shell")]
    assert result.data["chain_index"] == 0
    assert query_events(app.paths.state_file, kind="role_fallback_selected") == []


def test_length_one_chain_ignores_the_ledger_and_keeps_probe_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing behavior: no fallbacks -> the ledger is never consulted, a quota
    window defers until the probe window, and a ready probe launches one."""
    calls = _recorder(monkeypatch)
    _restrict(PRIMARY, datetime.now(UTC) + timedelta(hours=2))
    app = _app(tmp_path / "b", reviewer=SINGLE, n_prs=2)
    result = app.dispatch_reviews()
    assert [c[0] for c in _launched(calls)] == ["devin-shell", "devin-shell"]
    assert "chain_index" not in result.data

    calls.clear()
    app2 = _app(tmp_path / "c", reviewer=SINGLE, n_prs=2)
    until = _z(datetime.now(UTC) + timedelta(hours=2))
    with state_lock(app2.paths.state_file):
        state = load_state(app2.paths.state_file)
        future = _z(datetime.now(UTC) + timedelta(hours=1))
        save_state(
            app2.paths.state_file,
            set_reviewer_quota_exhausted(state, throttled_until=until, probe_after=future),
        )
    result = app2.dispatch_reviews()
    assert calls == []
    assert result.data["deferred_reason"] == "reviewer_quota_probe_backoff"
    assert "chain_length" not in result.data

    with state_lock(app2.paths.state_file):
        state = load_state(app2.paths.state_file)
        past = _z(datetime.now(UTC) - timedelta(minutes=1))
        save_state(
            app2.paths.state_file,
            set_reviewer_quota_exhausted(state, throttled_until=until, probe_after=past),
        )
    result = app2.dispatch_reviews()
    assert result.data["probe_mode"] is True
    assert len(calls) == 1


def test_launch_time_quota_hit_restricts_the_entry_it_launched_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _restrict(PRIMARY, datetime.now(UTC) + timedelta(hours=2))
    _recorder(monkeypatch, error="Error: daily usage quota has been exhausted.")
    result = _app(tmp_path / "b").dispatch_reviews()
    assert result.data["quota_hit"] is True
    assert set(role_quota_ledger.load_restrictions()) == {PRIMARY.key, FALLBACK.key}


def test_dead_fallback_reviewer_restricts_its_own_recorded_entry(tmp_path: Path) -> None:
    """The stalled-review sweep classifies a dead reviewer by its sidecar and
    restricts the (harness, model) stamped on it -- the fallback, not config."""
    repo_root, reviews_dir, config, state_file = _seed_stalled(tmp_path, 100)
    sidecar = _write_session_limit_reviewer(reviews_dir, 100, tmp_path)
    stamp = role_quota_ledger.session_stamp("reviewer", FALLBACK.harness, FALLBACK.model, 1)
    assert role_quota_ledger.stamp_session(sidecar, stamp)

    _detect_and_handle_stalled_reviews(
        reviews_dir, state_file, config, repo_root, write_gate=_wg(state_file)
    )

    quota_until = load_state(state_file)["reviewer_quota"]["throttled_until"]
    restrictions = role_quota_ledger.load_restrictions()
    assert set(restrictions) == {FALLBACK.key}
    assert _z(restrictions[FALLBACK.key]) == quota_until


def _seed_quota_window(app: OrchestratorApp, until: datetime, **provenance: str | None) -> None:
    far_probe = _z(datetime.now(UTC) + timedelta(hours=1))
    with state_lock(app.paths.state_file):
        state = set_reviewer_quota_exhausted(
            load_state(app.paths.state_file), throttled_until=_z(until), probe_after=far_probe
        )
        state["reviewer_quota"] = {**state["reviewer_quota"], **provenance}
        save_state(app.paths.state_file, state)


@pytest.mark.parametrize(
    "provenance",
    [
        pytest.param({}, id="unstamped-pre-ledger-window"),
        pytest.param({"reason": None, "adapter_kind": PRIMARY_ADAPTER}, id="no-reason"),
        pytest.param(
            {"reason": "quota_exhausted", "adapter_kind": "claude-code"},
            id="adapter-of-no-skipped-entry",
        ),
    ],
)
def test_reviewer_window_shorter_than_a_skipped_restriction_but_unexplained_still_defers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provenance: dict[str, str | None]
) -> None:
    """The primary's ledger restriction outlasts the per-repo window, which the
    old timestamp-only check waved through; without quota provenance attributable
    to the skipped primary it must still defer exactly as before #2086."""
    calls = _recorder(monkeypatch)
    _restrict(PRIMARY, datetime.now(UTC) + timedelta(hours=3))
    app = _app(tmp_path / "b", n_prs=2)
    _seed_quota_window(app, datetime.now(UTC) + timedelta(hours=1), **provenance)

    result = app.dispatch_reviews()

    assert calls == []
    assert result.data["launched_count"] == 0
    assert result.data["deferred_reason"] == "reviewer_quota_probe_backoff"


def test_reviewer_window_is_not_explained_by_a_later_unselected_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Primary free, only the fallback restricted: nothing was skipped, so even a
    stamped quota window on the primary's adapter keeps deferring."""
    calls = _recorder(monkeypatch)
    _restrict(FALLBACK, datetime.now(UTC) + timedelta(hours=3))
    app = _app(tmp_path / "b")
    _seed_quota_window(
        app,
        datetime.now(UTC) + timedelta(hours=1),
        reason="quota_exhausted",
        adapter_kind=PRIMARY_ADAPTER,
    )

    result = app.dispatch_reviews()

    assert calls == []
    assert result.data["deferred_reason"] == "reviewer_quota_probe_backoff"


def test_launch_time_quota_hit_stamps_the_window_with_the_launched_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _restrict(PRIMARY, datetime.now(UTC) + timedelta(hours=2))  # fallback (claude-code) launches
    _recorder(monkeypatch, error="Error: daily usage quota has been exhausted.")
    app = _app(tmp_path / "b")

    app.dispatch_reviews()

    quota = load_state(app.paths.state_file)["reviewer_quota"]
    assert (quota["reason"], quota["adapter_kind"]) == ("quota_exhausted", "claude-code")


def test_dead_reviewer_sweep_stamps_the_window_with_the_dead_sessions_adapter(
    tmp_path: Path,
) -> None:
    repo_root, reviews_dir, config, state_file = _seed_stalled(tmp_path, 100)
    _write_session_limit_reviewer(reviews_dir, 100, tmp_path)

    _detect_and_handle_stalled_reviews(
        reviews_dir, state_file, config, repo_root, write_gate=_wg(state_file)
    )

    quota = load_state(state_file)["reviewer_quota"]
    assert (quota["reason"], quota["adapter_kind"]) == ("quota_exhausted", "claude-code")
