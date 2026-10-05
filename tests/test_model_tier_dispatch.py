"""TIS-CW-6: a ``model:<tier>`` label routes a worker launch through the real launch gate.

The chain is sonnet (primary) -> swe-2 (devin) -> opus. An issue labelled
``model:opus`` launches on the opus entry in both the fresh and the rework
lane; an unlabelled one launches on the primary (the fleet's default tier); a
tier the chain cannot serve launches on the normal chain and says why.
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _fakes_github import FakeGitHub
from charlie_work import role_quota_ledger
from charlie_work.adapters import AdapterSettings, SessionDispatchResult, SessionRequest
from charlie_work.config import DispatchConfig, LabelConfig, OrchestratorConfig, WorkerRoleConfig
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.role_chain import RoleEntry
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

FRESH = "fresh"
REWORK = "rework"
LANES = pytest.mark.parametrize("lane", [FRESH, REWORK])

PRIMARY = RoleEntry("claude-code", "claude-sonnet-5-5")
DEVIN = RoleEntry("devin-shell", "swe-2")
OPUS = RoleEntry("claude-code", "claude-opus-5-5")
CHAIN = WorkerRoleConfig(harness=PRIMARY.harness, model=PRIMARY.model, fallbacks=(DEVIN, OPUS))


def _app(
    root: Path,
    lane: str,
    labels: tuple[str, ...] = ("model:opus",),
    *,
    label_config: LabelConfig | None = None,
) -> OrchestratorApp:
    root.mkdir(parents=True, exist_ok=True)
    config = OrchestratorConfig(
        worker=CHAIN,
        dispatch=DispatchConfig(default_limit=5),
        labels=label_config or LabelConfig(),
    )
    paths = runtime_paths(root, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [{"name": "automated-ready"}] + [{"name": n} for n in labels]
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


def _spy(monkeypatch: pytest.MonkeyPatch) -> list[tuple[AdapterSettings, SessionRequest]]:
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
            sidecar.write_text(
                json.dumps({"issue_number": request.issue_number, "pid": 999999}),
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

    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", _fake)
    return calls


def _run(app: OrchestratorApp, lane: str) -> Any:
    return app.dispatch() if lane == FRESH else app.dispatch_rework()


def _launched(calls: list[tuple[AdapterSettings, SessionRequest]]) -> dict[int, tuple[str, str]]:
    """Issue number -> the (harness, model) its launch pins, read from the adapter carriers."""
    out = {}
    for settings, request in calls:
        model = (
            settings.config.worker.model
            if settings.adapter == "claude-code"
            else settings.worker_model
        )
        out[request.issue_number] = (settings.adapter, model)
    return out


def _events(app: OrchestratorApp, kind: str) -> list[dict[str, Any]]:
    return [event["payload"] for event in query_events(app.paths.state_file, kind=kind)]


@LANES
def test_model_opus_label_launches_on_the_opus_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str
) -> None:
    """The rollout step 5 canary, in miniature."""
    calls = _spy(monkeypatch)
    app = _app(tmp_path, lane)
    result = _run(app, lane)
    assert _launched(calls) == {123: OPUS.key}, result.message
    [payload] = _events(app, "worker_model_tier_selected")
    assert (payload["tier"], payload["numbers"], payload["chain_index"]) == ("opus", [123], 2)
    assert (payload["harness"], payload["model"]) == OPUS.key
    # Routing is not a quota fallback, and nothing fell back.
    assert _events(app, "role_fallback_selected") == []
    assert _events(app, "worker_model_tier_fallback") == []
    sidecar = role_quota_ledger.sidecar_path_for(app._layout.sessions_dir, OPUS.harness, 123)
    stamp = json.loads(sidecar.read_text(encoding="utf-8"))[role_quota_ledger.SESSION_ROLE_KEY]
    assert (stamp["model"], stamp["chain_index"]) == (OPUS.model, 2)


@LANES
def test_unlabelled_issue_launches_on_the_default_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str
) -> None:
    calls = _spy(monkeypatch)
    app = _app(tmp_path, lane, labels=())
    _run(app, lane)
    assert _launched(calls) == {123: PRIMARY.key}
    assert _events(app, "worker_model_tier_selected") == []
    assert _events(app, "worker_model_tier_fallback") == []


def test_a_primary_tier_label_is_honoured_at_index_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _spy(monkeypatch)
    app = _app(tmp_path, FRESH, labels=("model:sonnet",))
    _run(app, FRESH)
    assert _launched(calls) == {123: PRIMARY.key}
    [payload] = _events(app, "worker_model_tier_selected")
    assert (payload["tier"], payload["chain_index"]) == ("sonnet", 0)


@pytest.mark.parametrize(
    ("labels", "tier", "reason"),
    [
        (("model:haiku",), "haiku", "no_entry"),
        (("model:opus", "model:sonnet"), "opus,sonnet", "conflicting_labels"),
    ],
)
def test_an_unservable_tier_falls_back_to_the_normal_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    labels: tuple[str, ...],
    tier: str,
    reason: str,
) -> None:
    calls = _spy(monkeypatch)
    app = _app(tmp_path, FRESH, labels=labels)
    _run(app, FRESH)
    assert _launched(calls) == {123: PRIMARY.key}
    [payload] = _events(app, "worker_model_tier_fallback")
    assert (payload["tier"], payload["reason"], payload["numbers"]) == (tier, reason, [123])
    assert _events(app, "worker_model_tier_selected") == []


def test_a_restricted_tier_falls_back_to_the_normal_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _spy(monkeypatch)
    later = datetime.now(UTC) + timedelta(hours=2)
    role_quota_ledger.record_restriction(
        OPUS.harness, OPUS.model, later, reason="quota_exhausted", source="test"
    )
    app = _app(tmp_path, FRESH)
    _run(app, FRESH)
    assert _launched(calls) == {123: PRIMARY.key}
    [payload] = _events(app, "worker_model_tier_fallback")
    assert (payload["tier"], payload["reason"]) == ("opus", "restricted")


def test_a_mixed_batch_launches_each_tier_on_its_entry_and_keeps_one_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _spy(monkeypatch)
    app = _app(tmp_path, FRESH)
    plain = copy.deepcopy(app.gh.issues[0])
    plain.update(number=124, title="Fix sort", url="https://example.test/issues/124")
    plain["labels"] = [{"name": "automated-ready"}]
    app.gh.issues.append(plain)
    _run(app, FRESH)
    assert _launched(calls) == {123: OPUS.key, 124: PRIMARY.key}
    manifest = json.loads(app._layout.session_manifest.read_text(encoding="utf-8"))
    assert sorted(s["issue_number"] for s in manifest["sessions"]) == [123, 124]
    assert manifest["adapter"] == "claude-code"
    results = json.loads(app._layout.session_results.read_text(encoding="utf-8"))
    assert sorted(r["issue_number"] for r in results["results"]) == [123, 124]


def test_an_empty_prefix_turns_routing_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _spy(monkeypatch)
    app = _app(tmp_path, FRESH, label_config=LabelConfig(model_tier_prefix=""))
    _run(app, FRESH)
    assert _launched(calls) == {123: PRIMARY.key}
    assert _events(app, "worker_model_tier_selected") == []
    assert _events(app, "worker_model_tier_fallback") == []
