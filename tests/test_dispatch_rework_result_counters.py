"""Issue #3088: every dispatch_rework return path carries the same counter keys."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _fakes_github import FakeGitHub
from charlie_work import fleet_provider_throttle
from charlie_work.config import DevinConfig, OrchestratorConfig, WorkerRoleConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, set_throttled_until, state_lock
from charlie_work.workflow import OrchestratorApp

COUNTER_KEYS = ("worker_death_escalated", "operator_claimed_skipped")
OPERATOR_CLAIMED_ISSUE = 124


class _ReworkGitHub(FakeGitHub):
    """Fake whose issue fetch can simulate a concurrent dispatch claim."""

    on_issue_view = None

    def issue_view(self, number):
        if self.on_issue_view is not None:
            self.on_issue_view(number)
        return super().issue_view(number)


def _app(
    tmp_path: Path, *, harness: str = "command", seed_issue: bool = True, prompt: bool = False
) -> tuple[OrchestratorApp, _ReworkGitHub, Path]:
    config = OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness=harness),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        # An operator-claimed rework issue is skipped on every pass that scans state.
        state["issues"][str(OPERATOR_CLAIMED_ISSUE)] = {
            "number": OPERATOR_CLAIMED_ISSUE,
            "status": "rework_requested",
            "operator_claimed_at": "2026-01-01T00:00:00Z",
        }
        if seed_issue:
            state["issues"]["123"] = {
                "number": 123,
                "title": "Fix search",
                "url": "https://example.test/issues/123",
                "status": "rework_requested",
            }
        save_state(paths.state_file, state)
    if prompt:
        pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
        pr_dir.mkdir(parents=True)
        (pr_dir / "rework-prompt.md").write_text("Fix the issues", encoding="utf-8")
    gh = _ReworkGitHub()
    gh.issues[0]["labels"] = [{"name": "agent:needs-rework"}]
    return OrchestratorApp(tmp_path, paths, config, gh), gh, paths.state_file


def _assert_counters(result, *, claimed: list[int]) -> None:
    for key in COUNTER_KEYS:
        assert key in result.data, key
    assert result.data["worker_death_escalated"] == []
    assert result.data["operator_claimed_skipped"] == claimed


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("no_candidates", "no rework candidates found"),
        ("all_dispatched", "all rework candidates already dispatched"),
        ("no_prompts", "no valid rework prompts found"),
        ("launched", "rework dispatch complete"),
    ],
)
def test_dispatch_rework_result_carries_counter_keys_on_scan_paths(
    tmp_path: Path, path: str, message: str
) -> None:
    app, gh, state_file = _app(
        tmp_path, seed_issue=path != "no_candidates", prompt=path == "launched"
    )
    if path == "all_dispatched":

        def claim(_number: int) -> None:
            with state_lock(state_file):
                state = load_state(state_file)
                state["issues"]["123"]["status"] = "dispatched"
                save_state(state_file, state)

        gh.on_issue_view = claim

    result = app.dispatch_rework()

    assert result.message == message
    _assert_counters(result, claimed=[OPERATOR_CLAIMED_ISSUE])


def test_dispatch_rework_manual_adapter_carries_counter_keys(tmp_path: Path) -> None:
    app, _gh, _state_file = _app(tmp_path, harness="manual")

    result = app.dispatch_rework()

    assert result.message == "rework dispatch skipped for manual adapter"
    _assert_counters(result, claimed=[])


def test_dispatch_rework_launch_permit_deferral_carries_counter_keys(tmp_path: Path) -> None:
    app, _gh, state_file = _app(tmp_path, harness="devin-shell")
    throttled_until = (datetime.now(UTC) + timedelta(hours=1)).replace(microsecond=0)
    with state_lock(state_file):
        state = set_throttled_until(
            load_state(state_file),
            throttled_until.isoformat().replace("+00:00", "Z"),
            source="test",
        )
        save_state(state_file, state)

    result = app.dispatch_rework()

    assert result.data["deferred_reason"] == "provider_throttled"
    _assert_counters(result, claimed=[OPERATOR_CLAIMED_ISSUE])


def test_dispatch_rework_fleet_throttle_deferral_carries_counter_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _gh, _state_file = _app(tmp_path)
    monkeypatch.setattr(
        fleet_provider_throttle,
        "decide_for_app",
        lambda *_args, **_kwargs: fleet_provider_throttle.ResumeDecision(action="defer_probe"),
    )

    result = app.dispatch_rework()

    assert result.data["deferred_reason"] == "provider_resume_staggered"
    _assert_counters(result, claimed=[OPERATOR_CLAIMED_ISSUE])


def test_dispatch_rework_dry_run_carries_counter_keys(tmp_path: Path) -> None:
    app, gh, _state_file = _app(tmp_path, prompt=True)
    app.dry_run = True

    result = app.dispatch_rework()

    assert result.message.startswith("dry-run:")
    _assert_counters(result, claimed=[OPERATOR_CLAIMED_ISSUE])
