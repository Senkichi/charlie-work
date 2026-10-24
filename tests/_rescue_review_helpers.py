"""Shared fixtures for _process_rescue_review tests, split out of
tests/test_feat_rescue_tier.py so the config-threading finding test (issue
#2795) can live in tests/test_rescue_review.py without pushing either file
over its size cap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from charlie_work.config import OrchestratorConfig, RescueConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

from _fakes_github import FakeGitHub


def seed_pr_state(paths, pr_number: int, issue_number: int, **fields) -> None:
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["prs"][str(pr_number)] = {
            "number": pr_number,
            "issue_number": issue_number,
            **fields,
        }
        save_state(paths.state_file, state)


def _rescue_marked_app(tmp_path: Path, **config_kwargs: Any) -> OrchestratorApp:
    config = OrchestratorConfig(rescue=RescueConfig(enabled=True), **config_kwargs)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    seed_pr_state(paths, 456, 123, rescue_attempted=True, rescue_cause="rework_cycle_cap")
    return app
