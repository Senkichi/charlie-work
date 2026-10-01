"""Issue #2086: reviewer role chain in the no-remote local review lane.

Two layers:

* ``record_error_quota_hit`` -- the local lane has no ``reviewer_quota`` gate,
  so a launch error matching the quota markers teaches the fleet ledger
  directly. Covered directly: provider-stated reset clock (+ resume margin),
  the ``quota_reset_hours`` fallback, and the no-selected-entry no-op.
* ``_local_dispatch_reviewers`` -- chain-exhausted deferral (and its payload
  key hygiene), launching the fallback entry with its own pins, and the
  launch-time quota-hit ledger write.

Fixture helpers are reinlined per file (the zero cross-test-import guard).
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _local_lane_fixtures import _git, _init_repo, _local_config, _make_branch, _parked_issue
from _review_fixtures import _fake_claude_worker_record
from charlie_work import role_quota_ledger, role_selection
from charlie_work.config import build_config_from_data
from charlie_work.instrumentation import query_events
from charlie_work.local_issues import LocalFileGitHub
from charlie_work.paths import runtime_paths
from charlie_work.role_chain import RoleEntry
from charlie_work.workflow import OrchestratorApp

import charlie_work.workflow as wf

PRIMARY = RoleEntry("devin-shell", "swe-2")
FALLBACK = RoleEntry("claude-code", "claude-sonnet-5-5", "high")
REVIEWER_SECTION = {
    "harness": PRIMARY.harness,
    "model": PRIMARY.model,
    "fallbacks": [{"harness": FALLBACK.harness, "model": FALLBACK.model, "effort": "high"}],
}
QUOTA_ERROR = "Error: daily usage quota has been exhausted."
MARGIN_S = 90
RESET_HOURS = 5


def _z(moment: datetime) -> str:
    return moment.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _restrict(entry: RoleEntry, until: datetime) -> None:
    role_quota_ledger.record_restriction(
        entry.harness, entry.model, until, reason="quota_exhausted", source="test"
    )


# --- record_error_quota_hit (direct) -------------------------------------------


def _chain_config() -> Any:
    return build_config_from_data(
        {
            "reviewer": REVIEWER_SECTION,
            "runtime": {"throttle_resume_margin_s": MARGIN_S},
            "review_dispatch": {"quota_reset_hours": RESET_HOURS},
        }
    )


def _selection(config: Any, ledger: dict | None = None) -> Any:
    return role_selection.build_selection(config.reviewer.chain, ledger or {}, datetime.now(UTC))


def test_error_quota_hit_uses_the_provider_reset_clock_plus_the_resume_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _chain_config()
    reset_at = (datetime.now(UTC) + timedelta(hours=3)).replace(microsecond=0)
    seen: list[tuple[str, datetime]] = []

    def _parse(text: str, now: datetime) -> datetime:
        seen.append((text, now))
        return reset_at

    monkeypatch.setattr("charlie_work.throttle_signatures.parse_reset_clock_time", _parse)

    changed = role_selection.record_error_quota_hit(
        _selection(config), "limit reached, resets 4:40pm", config, source="t"
    )

    assert changed is True
    assert [text for text, _ in seen] == ["limit reached, resets 4:40pm"]
    assert role_quota_ledger.load_restrictions() == {
        PRIMARY.key: reset_at + timedelta(seconds=MARGIN_S)
    }


@pytest.mark.parametrize("error_text", [QUOTA_ERROR, None], ids=["no-clock-in-text", "no-text"])
def test_error_quota_hit_falls_back_to_quota_reset_hours(error_text: str | None) -> None:
    config = _chain_config()
    before = datetime.now(UTC)

    assert role_selection.record_error_quota_hit(
        _selection(config), error_text, config, source="t"
    )

    after = datetime.now(UTC)
    until = role_quota_ledger.load_restrictions()[PRIMARY.key]
    window = timedelta(hours=RESET_HOURS)
    # The ledger stores whole seconds (truncated), hence the 1s slack below.
    assert before + window - timedelta(seconds=1) <= until <= after + window


def test_error_quota_hit_restricts_the_entry_it_launched_on_not_the_primary() -> None:
    config = _chain_config()
    selection = _selection(config, {PRIMARY.key: datetime.now(UTC) + timedelta(hours=1)})
    assert selection.entry == FALLBACK

    assert role_selection.record_error_quota_hit(selection, QUOTA_ERROR, config, source="t")

    # Only the fallback it launched on; the primary's restriction was pre-existing
    # in the selection's view and is not re-written by this call.
    assert set(role_quota_ledger.load_restrictions()) == {FALLBACK.key}


def test_error_quota_hit_is_a_noop_when_no_entry_was_selected() -> None:
    config = _chain_config()
    later = datetime.now(UTC) + timedelta(hours=1)
    exhausted = _selection(config, {PRIMARY.key: later, FALLBACK.key: later})
    assert exhausted.entry is None
    before = role_quota_ledger.load_restrictions()

    assert (
        role_selection.record_error_quota_hit(exhausted, QUOTA_ERROR, config, source="t") is False
    )

    assert role_quota_ledger.load_restrictions() == before


# --- _local_dispatch_reviewers (app level) --------------------------------------


def _local_app(tmp_path: Path) -> OrchestratorApp:
    repo = tmp_path / "localrepo"
    repo.mkdir()
    _init_repo(repo)
    issues_dir = repo / "docs" / "issues"
    config = _local_config(
        repo,
        issues_dir,
        reviewer=REVIEWER_SECTION,
        review_dispatch={"max_local_review_processes": 0},
    )
    paths = runtime_paths(repo, config.runtime.state_dir)
    app = OrchestratorApp(
        repo,
        paths,
        config,
        LocalFileGitHub(repo_root=repo, issues_dir=issues_dir),
        fleet_dir_override=str(tmp_path / "fleet"),
    )
    for number in (7, 8):
        _make_branch(repo, f"agent/issue-{number}-x", f"f{number}.py", f"x = {number}\n")
        _git(repo, "checkout", "main")
        _parked_issue(app, issues_dir, number, f"agent/issue-{number}-x")
    app._local_review_packets()
    return app


def _recorder(monkeypatch: pytest.MonkeyPatch, *, error: str | None = None) -> list[dict]:
    """Replace every review launcher; successful launches write the sidecar a real one would."""
    calls: list[dict] = []

    def _make(harness: str):
        def _launch(**kwargs: Any):
            calls.append({"harness": harness, **kwargs})
            pr = kwargs["pr_number"]
            record = _fake_claude_worker_record(pr, kwargs["branch"])
            if error is not None:
                return replace(record, error=error, pid=None)
            sidecar = role_quota_ledger.sidecar_path_for(kwargs["reviews_dir"], harness, pr)
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(json.dumps({"issue_number": pr, "pid": 12345}), encoding="utf-8")
            return record

        return _launch

    for harness in list(wf._REVIEW_LAUNCHERS):
        monkeypatch.setitem(wf._REVIEW_LAUNCHERS, harness, _make(harness))
    return calls


def test_chain_exhausted_defers_without_clobbering_skipped_pr_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _recorder(monkeypatch)
    soon = datetime.now(UTC) + timedelta(hours=1)
    later = soon + timedelta(hours=1)
    _restrict(PRIMARY, soon)
    _restrict(FALLBACK, later)
    app = _local_app(tmp_path)

    result = app._local_dispatch_reviewers()

    assert calls == []
    assert result["deferred_reason"] == "reviewer_chain_exhausted"
    assert result["launched"] == []
    # ``skipped`` keeps its meaning (PR numbers left undispatched this pass) ...
    assert len(result["skipped"]) == 2
    assert all(isinstance(number, int) for number in result["skipped"])
    # ... and the chain's per-entry payloads travel under their own key.
    assert [item["index"] for item in result["chain_skipped"]] == [0, 1]
    assert result["chain_retry_at"] == _z(soon)
    assert result["chain_length"] == 2


def test_primary_restricted_launches_the_fallback_with_its_own_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _recorder(monkeypatch)
    until = datetime.now(UTC) + timedelta(hours=2)
    _restrict(PRIMARY, until)
    app = _local_app(tmp_path)

    result = app._local_dispatch_reviewers()

    assert len(result["launched"]) == 2
    assert {
        (
            c["harness"],
            c["model_override"],
            c["resolved_review_effort"],
            c["config"].reviewer.harness,
        )
        for c in calls
    } == {("claude-code", FALLBACK.model, "high", "claude-code")}
    [event] = query_events(app.paths.state_file, kind="role_fallback_selected")
    assert event["payload"]["role"] == "reviewer"
    assert event["payload"]["skipped"] == [
        {**PRIMARY.to_payload(), "index": 0, "until": _z(until)}
    ]
    sidecar = role_quota_ledger.sidecar_path_for(
        app._layout.reviews_dir, "claude-code", result["launched"][0]["pr"]
    )
    assert json.loads(sidecar.read_text(encoding="utf-8"))["role_entry"]["chain_index"] == 1


def test_launch_quota_hit_writes_the_launched_entry_to_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _recorder(monkeypatch, error=QUOTA_ERROR)
    app = _local_app(tmp_path)
    assert role_quota_ledger.load_restrictions() == {}

    result = app._local_dispatch_reviewers()

    assert result["quota_hit"] is True
    assert result["launched"] == []
    restrictions = role_quota_ledger.load_restrictions()
    assert set(restrictions) == {PRIMARY.key}
    assert restrictions[PRIMARY.key] > datetime.now(UTC)
