"""Shared fixtures for the janitor merge-conflict / no-op-rework routing tests.

Hoisted out of ``test_fix_janitor_routing.py`` (issue #1284): a
review-decision.json writer and a FakeGitHub-backed app builder pre-seeded
with a CONFLICTING/DIRTY PR, both imported by other test modules that
exercise the same conflict-routing paths.
"""

from __future__ import annotations

import json
from pathlib import Path

from _fakes_github import FakeGitHub
from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp


def _set_decision(app: OrchestratorApp, pr_number: int, decision: str) -> None:
    pr_dir = app.paths.prs / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": decision}), encoding="utf-8"
    )


def _conflicting_app(tmp_path: Path, **config_kwargs) -> OrchestratorApp:
    config = OrchestratorConfig(**config_kwargs)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0]["mergeable"] = "CONFLICTING"
    fake_gh.prs[0]["mergeStateStatus"] = "DIRTY"
    return OrchestratorApp(tmp_path, paths, config, fake_gh)
