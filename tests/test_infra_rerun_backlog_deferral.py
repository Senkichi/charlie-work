"""Infra-rerun deferral while the repo's CI queue is backlogged (DD-13).

A ``gh run rerun`` adds a full run to a lane that is already queuing; when
the freshest ``runner_allocation`` event says the repo's oldest queued job has
waited longer than ``auto_merge.infra_rerun_backlog_defer_seconds``, the
driver defers the rerun to a later pass without consuming an attempt.

``tests/conftest.py``'s autouse fixture points ``CHARLIE_WORK_FLEET_DIR`` at a
per-test directory, so ``fleet_dir()`` is isolated here.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github_rerun import FakeGitHubWithRerunCapture
from _review_fixtures import _required_checks_config
from charlie_work import layout
from charlie_work.ci_backlog import infra_rerun_backlog_seconds
from charlie_work.ci_headroom import ALLOCATION_EVENT_KIND
from charlie_work.fleet_paths import fleet_dir
from charlie_work.instrumentation import log_event
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp

REPO_SLUG = "test-owner/test-repo"  # FakeGitHub.name_with_owner()
_RUN_LINK = "https://github.com/owner/repo/actions/runs/12345/job/67890"
_CANCELLED_CHECKS = [
    {"name": "Tests passed", "state": "CANCELLED", "link": _RUN_LINK},
    {"name": "Lint & Format", "bucket": "pass"},
    {"name": "Pre-commit", "state": "SUCCESS"},
]


def _state_path() -> Path:
    return layout.state_file_path(fleet_dir())


def _log_allocation(oldest: Any, repo: str = REPO_SLUG) -> None:
    log_event(
        _state_path(),
        ALLOCATION_EVENT_KIND,
        {
            "budget": 8,
            "budget_reason": "test",
            "targets": [
                {
                    "repo": repo,
                    "capacity": 4,
                    "demand": 9,
                    "running": 4,
                    "target": 4,
                    "pinned": False,
                    "oldest_queued_seconds": oldest,
                }
            ],
            "changes": [],
            "notes": [],
        },
    )


# --------------------------------------------------------------------------- ci_backlog


def test_backlog_above_threshold_returns_seconds() -> None:
    _log_allocation(1200)
    assert (
        infra_rerun_backlog_seconds(
            REPO_SLUG, fleet_state_path=_state_path(), threshold_seconds=900
        )
        == 1200
    )


def test_backlog_below_threshold_is_none() -> None:
    _log_allocation(300)
    assert (
        infra_rerun_backlog_seconds(
            REPO_SLUG, fleet_state_path=_state_path(), threshold_seconds=900
        )
        is None
    )


def test_backlog_fails_open() -> None:
    path = _state_path()
    assert (
        infra_rerun_backlog_seconds(REPO_SLUG, fleet_state_path=path, threshold_seconds=900)
        is None
    )
    _log_allocation(True)  # bool is not a reading
    assert (
        infra_rerun_backlog_seconds(REPO_SLUG, fleet_state_path=path, threshold_seconds=900)
        is None
    )
    _log_allocation(5000, repo="other/repo")
    assert (
        infra_rerun_backlog_seconds(REPO_SLUG, fleet_state_path=path, threshold_seconds=900)
        is None
    )
    assert infra_rerun_backlog_seconds("?", fleet_state_path=path, threshold_seconds=900) is None


def test_backlog_disabled_and_stale() -> None:
    _log_allocation(5000)
    path = _state_path()
    assert (
        infra_rerun_backlog_seconds(REPO_SLUG, fleet_state_path=path, threshold_seconds=0) is None
    )
    later = datetime.now(UTC) + timedelta(minutes=31)
    assert (
        infra_rerun_backlog_seconds(
            REPO_SLUG, fleet_state_path=path, threshold_seconds=900, now=later
        )
        is None
    )


# --------------------------------------------------------------------------- driver wiring


def _app(tmp_path: Path, fake_gh: FakeGitHubWithRerunCapture, *, defer_seconds: int | None = None):
    config = _required_checks_config()
    if defer_seconds is not None:
        config = dataclasses.replace(
            config,
            auto_merge=dataclasses.replace(
                config.auto_merge, infra_rerun_backlog_defer_seconds=defer_seconds
            ),
        )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    decision_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )
    return app, paths


def _events(paths, kind: str) -> list[dict]:
    return [e for e in load_state(paths.state_file).get("events", []) if e["kind"] == kind]


def test_backlog_defers_rerun_without_consuming_attempt(tmp_path: Path) -> None:
    _log_allocation(1800)
    fake_gh = FakeGitHubWithRerunCapture(checks=list(_CANCELLED_CHECKS))
    app, paths = _app(tmp_path, fake_gh)

    first = app.merge_ready(456)
    second = app.merge_ready(456)

    assert fake_gh.rerun_calls == []
    for result in (first, second):
        assert result.data["infra_rerun_backlog_deferred"] is True
        assert result.data["oldest_queued_seconds"] == 1800
    pr_state = load_state(paths.state_file)["prs"]["456"]
    assert pr_state.get("infra_rerun_attempts", {}) == {}
    assert pr_state["infra_rerun_backlog_deferred"] == {"sha-abc123": True}
    assert len(_events(paths, "infra_rerun_backlog_deferred")) == 1  # transition only


def test_rerun_fires_once_backlog_clears_and_resets_marker(tmp_path: Path) -> None:
    _log_allocation(1800)
    fake_gh = FakeGitHubWithRerunCapture(checks=list(_CANCELLED_CHECKS))
    app, paths = _app(tmp_path, fake_gh)
    app.merge_ready(456)
    _log_allocation(10)
    result = app.merge_ready(456)
    assert fake_gh.rerun_calls == [["run", "rerun", "12345"]]
    assert result.data["infra_rerun_run_ids"] == [12345]
    pr_state = load_state(paths.state_file)["prs"]["456"]
    assert pr_state["infra_rerun_attempts"] == {"sha-abc123": {"Tests passed": {"12345": 1}}}
    assert pr_state["infra_rerun_backlog_deferred"] == {}


def test_kill_switch_zero_reruns_despite_backlog(tmp_path: Path) -> None:
    _log_allocation(99999)
    fake_gh = FakeGitHubWithRerunCapture(checks=list(_CANCELLED_CHECKS))
    app, _ = _app(tmp_path, fake_gh, defer_seconds=0)
    app.merge_ready(456)
    assert fake_gh.rerun_calls == [["run", "rerun", "12345"]]


def test_no_allocation_data_reruns(tmp_path: Path) -> None:
    fake_gh = FakeGitHubWithRerunCapture(checks=list(_CANCELLED_CHECKS))
    app, _ = _app(tmp_path, fake_gh)
    app.merge_ready(456)
    assert fake_gh.rerun_calls == [["run", "rerun", "12345"]]
