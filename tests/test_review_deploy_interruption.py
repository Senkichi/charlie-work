"""Issue #2103: a reviewer killed by a code-only self_deploy is not a miss."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _review_fixtures import (
    _dispatch_reviews_app,
    _make_dead_review_sidecar,
    _set_review_dispatched_state,
    _write_review_events,
    _write_review_packet,
)
from charlie_work.instrumentation import log_event
from charlie_work.review_deploy_interruption import (
    SELF_DEPLOY_SUCCEEDED,
    deploy_interrupted_review,
)
from charlie_work.state import load_state, save_state, state_lock

_PR = {
    "number": 100,
    "title": "Fix #10",
    "url": "https://example.test/pull/100",
    "headRefName": "agent/issue-10-fix",
    "baseRefName": "main",
    "headRefOid": "sha-100",
    "mergeStateStatus": "CLEAN",
    "body": "Closes #10",
    "labels": [],
    "isCrossRepository": False,
    "state": "OPEN",
}


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _setup(
    monkeypatch,
    tmp_path: Path,
    *,
    deployed: bool,
    deploy_payload: dict | None = None,
):
    app = _dispatch_reviews_app(tmp_path, prs=[_PR])
    _write_review_packet(tmp_path, 100, "sha-100")
    reviews_dir = app._layout.reviews_dir
    started = datetime.now(UTC) - timedelta(minutes=10)
    _make_dead_review_sidecar(reviews_dir, 100, "no verdict", started_at=_iso(started))
    _write_review_events(reviews_dir, 100, turns=3, tool_calls=2)
    _set_review_dispatched_state(app, 100, 10, _iso(started))
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["100"]["review_dispatch_attempt_count"] = 2
        save_state(app.paths.state_file, state)

    deploy_state = tmp_path / "daemon" / "state.json"
    deploy_state.parent.mkdir()
    if deployed:
        log_event(
            deploy_state,
            SELF_DEPLOY_SUCCEEDED,
            deploy_payload or {"ok": True, "changed": True},
        )
    monkeypatch.setattr("charlie_work.worker_fate.is_alive", lambda *_: False)
    monkeypatch.setattr(
        "charlie_work.orchestration.misc_review_verdicts.self_deploy_state_path",
        lambda: deploy_state,
    )
    return app, reviews_dir


def _kinds(state: dict) -> list[str]:
    return [e["kind"] for e in state["events"]]


def test_deploy_interrupted_reviewer_is_rolled_back_not_missed(monkeypatch, tmp_path) -> None:
    app, reviews_dir = _setup(monkeypatch, tmp_path, deployed=True)

    result = app._reap_review_verdicts(reviews_dir)

    assert result["missed"] == []
    state = load_state(app.paths.state_file)
    kinds = _kinds(state)
    assert "review_verdict_missed" not in kinds
    assert "review_interrupted_by_deploy" in kinds
    pr = state["prs"]["100"]
    assert pr["review_dispatch_status"] is None
    assert pr["review_dispatch_attempt_count"] == 1
    assert not pr.get("review_turn_limit_miss_streak")


def test_dead_reviewer_without_deploy_is_still_a_miss(monkeypatch, tmp_path) -> None:
    app, reviews_dir = _setup(monkeypatch, tmp_path, deployed=False)

    result = app._reap_review_verdicts(reviews_dir)

    assert result["missed"][0]["reason"] == "died_mid_session"
    state = load_state(app.paths.state_file)
    assert "review_interrupted_by_deploy" not in _kinds(state)
    assert state["prs"]["100"]["review_dispatch_attempt_count"] == 2


def test_deploy_before_reviewer_start_does_not_count(tmp_path) -> None:
    deploy_state = tmp_path / "state.json"
    log_event(deploy_state, SELF_DEPLOY_SUCCEEDED, {"ok": True, "changed": True})
    later = _iso(datetime.now(UTC) + timedelta(minutes=5))
    assert deploy_interrupted_review(later, deploy_state) is None
    assert deploy_interrupted_review("garbage", deploy_state) is None
    assert deploy_interrupted_review(later, None) is None
    assert deploy_interrupted_review(later, tmp_path / "absent" / "state.json") is None


def test_venv_repair_only_deploy_is_not_a_witness(tmp_path) -> None:
    deploy_state = tmp_path / "state.json"
    log_event(
        deploy_state,
        SELF_DEPLOY_SUCCEEDED,
        {"ok": True, "changed": False, "venv_repaired": True},
    )
    earlier = _iso(datetime.now(UTC) - timedelta(minutes=5))
    assert deploy_interrupted_review(earlier, deploy_state) is None


def test_venv_repair_only_deploy_leaves_death_an_ordinary_miss(monkeypatch, tmp_path) -> None:
    app, reviews_dir = _setup(
        monkeypatch,
        tmp_path,
        deployed=True,
        deploy_payload={"ok": True, "changed": False, "venv_repaired": True},
    )

    result = app._reap_review_verdicts(reviews_dir)

    assert result["missed"][0]["reason"] == "died_mid_session"
    state = load_state(app.paths.state_file)
    assert "review_interrupted_by_deploy" not in _kinds(state)
    assert state["prs"]["100"]["review_dispatch_attempt_count"] == 2
