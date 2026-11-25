"""Issue #3088: every dispatch_rework return path carries the same counter keys."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from _fakes_github import FakeGitHub
from charlie_work.config import DevinConfig, OrchestratorConfig, WorkerRoleConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

COUNTER_KEYS = ("worker_death_escalated", "operator_claimed_skipped")


class _ReworkGitHub(FakeGitHub):
    """Fake whose issue fetch can simulate a concurrent dispatch claim."""

    on_issue_view = None

    def issue_view(self, number):
        if self.on_issue_view is not None:
            self.on_issue_view(number)
        return super().issue_view(number)


def _app(
    tmp_path: Path, *, seed_issue: bool, prompt: bool
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
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    if seed_issue:
        with state_lock(paths.state_file):
            state = load_state(paths.state_file)
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


@pytest.mark.parametrize("path", ["no_candidates", "all_dispatched", "no_prompts", "launched"])
def test_dispatch_rework_result_carries_counter_keys_on_every_path(
    tmp_path: Path, path: str
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

    expected_message = {
        "no_candidates": "no rework candidates found",
        "all_dispatched": "all rework candidates already dispatched",
        "no_prompts": "no valid rework prompts found",
        "launched": "rework dispatch complete",
    }[path]
    assert result.message == expected_message
    for key in COUNTER_KEYS:
        assert result.data[key] == [], (path, key)
