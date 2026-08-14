from __future__ import annotations

import json as _json
import logging
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from charlie_work import layout
from charlie_work.doctor import _check_runner_allocation
from charlie_work.config import (
    ConfigError,
    OrchestratorConfig,
    RunnerAllocationConfig,
    RunnerScalingConfig,
    RuntimeConfig,
    SupervisorConfig,
)
from charlie_work.fleet_dispatch import (
    ApiWorkerFleetReport,
    FleetLocalSnapshot,
    _CiFleetDirtyCheck,
    _build_fleet_attention_digest,
    _emit_fleet_transition,
    _extract_attention_events,
    _fleet_has_configured_repos,
    _has_fleet_delta,
    _is_fleet_pass_active,
    _lane_failure_state_path,
    _run_fleet_allocation_prologue,
    _run_fleet_autoscale_prologue,
    _select_repos,
    _ci_fleet_worktree_dirty as _real_ci_fleet_worktree_dirty,
    _take_fleet_snapshot,
    compute_api_worker_fleet_report,
    fleet_loop,
    run_fleet_supervise,
    run_fleet_supervise_loop,
)
from charlie_work.subprocess_runner import RunResult
from charlie_work.notify import AttentionEntry
from charlie_work.fleet_registry import count_fleet_runners
from charlie_work.instrumentation import query_events
from ci_fleet.charlie_work_adapter import (
    ALLOCATION_STATE_FILENAME,
    ScaleAction,
    load_allocation_stamp,
)
from ci_fleet.runners import ScaleDecision
from ci_fleet.runner_allocation import (
    AllocationPlan,
    SlotAction,
    SlotChange,
    SlotChangeResult,
)
from ci_fleet.runner_allocation_pass import AllocationPassResult
from charlie_work.supervise import SelfDeployResult
from charlie_work.supervise_loop import EXIT_RESTART_REQUESTED
from charlie_work.github import GitHubError
from charlie_work.workflow import CommandResult


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


_supervisor_started = datetime.now(UTC).replace(microsecond=0)
SUPERVISOR_STARTED_AT = _iso(_supervisor_started)
SUPERVISOR_BEAT_AT = _iso(_supervisor_started + timedelta(seconds=2609))


class _FakeClock:
    """Monotonically advancing fake clock/sleep for supervisor tests."""

    def __init__(self, start: float = 0.0, auto_advance: float = 0.0) -> None:
        self._now = start
        self._auto_advance = auto_advance
        self.sleep_calls: list[float] = []

    def now(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self._now += self._auto_advance if self._auto_advance else seconds


@pytest.fixture(autouse=True)
def _patch_self_deploy_for_fleet_tests(monkeypatch: Any) -> dict[str, MagicMock]:
    """Self-deploy hits the real git/uv CLI; keep fleet supervisor unit tests hermetic.

    Also no-op the supervisor lifecycle instrumentation (issue #627) so existing
    supervisor tests do not write heartbeat/events to the real fleet dir. The
    lifecycle functions are replaced with MagicMocks keyed by name in the
    returned dict; dedicated wiring tests request this fixture to assert the
    calls. ``detect_prior_abnormal_exit`` defaults to ``None`` (no prior exit)
    and ``is_exit_alertable`` defaults to ``False`` so existing tests do not
    trip the prior-exit or alert branches.
    """
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.self_deploy",
        lambda _repo_root, **_kwargs: SelfDeployResult(
            ok=True,
            pulled=False,
            changed=False,
            synced=False,
            message="test no-op",
        ),
    )
    mocks: dict[str, MagicMock] = {}
    for name in (
        "detect_prior_abnormal_exit",
        "record_prior_abnormal_exit",
        "record_supervisor_started",
        "update_supervisor_heartbeat",
        "record_supervisor_exit",
        "is_exit_alertable",
    ):
        m = MagicMock(name=name)
        monkeypatch.setattr("charlie_work.fleet_dispatch." + name, m)
        mocks[name] = m
    mocks["detect_prior_abnormal_exit"].return_value = None
    mocks["is_exit_alertable"].return_value = False
    return mocks


@pytest.fixture(autouse=True)
def _patch_ci_fleet_dirty_for_hermetic_tests(monkeypatch: Any) -> None:
    """Fleet dispatch tests must not fail because the real ci_fleet tree is dirty.

    The editable path dependency lives in a sibling checkout whose porcelain
    state is outside these tests' control. A dirty upstream tree would force
    every allocation-prologue test into dry-run mode and break assertions on
    the ``dry_run`` flag. This fixture makes the guard inert; tests that need
    to exercise the dirty path monkeypatch it explicitly.
    """
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch._ci_fleet_worktree_dirty",
        lambda _module_file=None: _CiFleetDirtyCheck(is_dirty=False),
    )


def test_select_repos_all_sorted_by_last_seen() -> None:
    """_select_repos returns all repos sorted by oldest last_seen first."""
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": "/path/to/repo1",
                "last_seen": "2026-07-07T10:00:00Z",
            },
            "owner/repo2": {
                "repo_root": "/path/to/repo2",
                "last_seen": "2026-07-07T09:00:00Z",  # Oldest
            },
            "owner/repo3": {
                "repo_root": "/path/to/repo3",
                "last_seen": "2026-07-07T11:00:00Z",
            },
        }
    }

    selected = _select_repos(registry, None)

    assert len(selected) == 3
    assert selected[0][0] == "owner/repo2"  # Oldest first
    assert selected[1][0] == "owner/repo1"
    assert selected[2][0] == "owner/repo3"


def test_select_repos_without_last_seen_goes_last() -> None:
    """Repos without last_seen are sorted last (treated as newest)."""
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": "/path/to/repo1",
                "last_seen": "2026-07-07T10:00:00Z",
            },
            "owner/repo2": {
                "repo_root": "/path/to/repo2",
                # No last_seen
            },
            "owner/repo3": {
                "repo_root": "/path/to/repo3",
                "last_seen": "2026-07-07T09:00:00Z",
            },
        }
    }

    selected = _select_repos(registry, None)

    assert len(selected) == 3
    assert selected[0][0] == "owner/repo3"  # Oldest with last_seen
    assert selected[1][0] == "owner/repo1"
    assert selected[2][0] == "owner/repo2"  # No last_seen, goes last


def test_select_repos_explicit_subset() -> None:
    """_select_repos with explicit repos returns exactly that subset in given order."""
    registry = {
        "repos": {
            "owner/repo1": {"repo_root": "/path/to/repo1", "last_seen": "2026-07-07T10:00:00Z"},
            "owner/repo2": {"repo_root": "/path/to/repo2", "last_seen": "2026-07-07T09:00:00Z"},
            "owner/repo3": {"repo_root": "/path/to/repo3", "last_seen": "2026-07-07T11:00:00Z"},
        }
    }

    selected = _select_repos(registry, ("owner/repo3", "owner/repo1"))

    assert len(selected) == 2
    assert selected[0][0] == "owner/repo3"  # Explicit order
    assert selected[1][0] == "owner/repo1"


def test_select_repos_explicit_subset_skips_missing() -> None:
    """_select_repos skips keys that don't exist in the registry."""
    registry = {
        "repos": {
            "owner/repo1": {"repo_root": "/path/to/repo1", "last_seen": "2026-07-07T10:00:00Z"},
            "owner/repo2": {"repo_root": "/path/to/repo2", "last_seen": "2026-07-07T09:00:00Z"},
        }
    }

    selected = _select_repos(registry, ("owner/repo3", "owner/repo1"))

    assert len(selected) == 1
    assert selected[0][0] == "owner/repo1"  # owner/repo3 doesn't exist, skipped


def test_select_repos_empty_registry() -> None:
    """_select_repos with empty registry returns empty list."""
    registry = {"repos": {}}

    selected = _select_repos(registry, None)

    assert selected == []


def test_extract_attention_events_stalled() -> None:
    """_extract_attention_events extracts stalled sessions."""
    result = CommandResult(
        True,
        "loop complete",
        {
            "stalled": [
                {"session_id": "sess1", "issue_number": 123, "reason": "timeout"},
                {"session_id": "sess2", "issue_number": 456, "reason": "crash"},
            ],
            "errors": [],
        },
    )

    events = _extract_attention_events("owner/repo1", result)

    assert len(events) == 2
    assert events[0]["repo_key"] == "owner/repo1"
    assert events[0]["type"] == "stalled"
    assert events[0]["session_id"] == "sess1"
    assert events[0]["issue_number"] == 123
    assert events[0]["reason"] == "timeout"
    assert events[1]["repo_key"] == "owner/repo1"
    assert events[1]["type"] == "stalled"
    assert events[1]["session_id"] == "sess2"


def test_extract_attention_events_errors() -> None:
    """_extract_attention_events extracts PR errors."""
    result = CommandResult(
        True,
        "loop complete",
        {
            "stalled": [],
            "errors": [
                {"pr": 789, "error": "merge conflict"},
                {"pr": 101, "error": "network error"},
            ],
        },
    )

    events = _extract_attention_events("owner/repo2", result)

    assert len(events) == 2
    assert events[0]["repo_key"] == "owner/repo2"
    assert events[0]["type"] == "error"
    assert events[0]["pr"] == 789
    assert events[0]["error"] == "merge conflict"
    assert events[1]["repo_key"] == "owner/repo2"
    assert events[1]["type"] == "error"
    assert events[1]["pr"] == 101


def test_extract_attention_events_errors_prefer_issue_number() -> None:
    """_extract_attention_events surfaces the linked issue number for errors that carry one (issue #502)."""
    result = CommandResult(
        True,
        "loop complete",
        {
            "stalled": [],
            "errors": [
                {"pr": 501, "issue": 494, "error": "possible worker self-merge"},
            ],
        },
    )

    events = _extract_attention_events("owner/repo2", result)

    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["issue_number"] == 494
    assert events[0]["pr"] == 501


def test_extract_attention_events_health_transitions() -> None:
    """_extract_attention_events extracts health transitions."""
    result = CommandResult(
        True,
        "loop complete",
        {
            "stalled": [],
            "errors": [],
            "health_transitions": [
                {"session_id": "sess1", "from_state": "running", "to_state": "stalled"},
                {"session_id": "sess2", "from_state": "stalled", "to_state": "running"},
            ],
        },
    )

    events = _extract_attention_events("owner/repo3", result)

    assert len(events) == 2
    assert events[0]["repo_key"] == "owner/repo3"
    assert events[0]["type"] == "health_transition"
    assert events[0]["session_id"] == "sess1"
    assert events[0]["from_state"] == "running"
    assert events[0]["to_state"] == "stalled"
    assert events[1]["repo_key"] == "owner/repo3"
    assert events[1]["type"] == "health_transition"
    assert events[1]["session_id"] == "sess2"


def test_extract_attention_events_empty() -> None:
    """_extract_attention_events returns empty list for result with no events."""
    result = CommandResult(True, "loop complete", {"stalled": [], "errors": []})

    events = _extract_attention_events("owner/repo1", result)

    assert events == []


def test_extract_attention_events_nested_loop_skip() -> None:
    """Nested loop() sub-results (intake/dispatch/rework/reviews) carry their
    own skip signals, so extraction must recurse one level and surface them.
    """
    result = CommandResult(
        True,
        "loop complete",
        {
            "stalled": [],
            "errors": [],
            "intake": {"skipped": True, "reason": "state_lock_busy"},
            "dispatch": {"state_lock_busy": True, "reason": "state_lock_busy"},
            "dispatch_rework": {"deferred_reason": "graphql_rate_limit"},
            "dispatch_reviews": {"skipped": True, "reason": "state_lock_busy"},
        },
    )

    events = _extract_attention_events("owner/repo1", result)

    assert all(event["repo_key"] == "owner/repo1" for event in events)
    assert all(event["type"] == "skipped" for event in events)
    assert {event["reason"] for event in events} == {"state_lock_busy", "graphql_rate_limit"}
    assert len(events) == 2


def test_extract_attention_events_nested_dispatch_failures() -> None:
    """Issue #497: worker/rework/reviewer launch failures in nested dispatch
    sub-results are surfaced as attention events with the actual error text
    and a non-sentinel issue/PR identifier. Concurrency deferrals are not
    treated as launch failures.
    """
    review_error = (
        "failed to launch claude: [WinError 2] The system cannot find the file specified"
    )
    result = CommandResult(
        True,
        "loop complete",
        {
            "stalled": [],
            "errors": [],
            "intake": {"failed": []},
            "dispatch": {
                "selected_count": 0,
                "failures": {11: "failed to launch claude: OSError"},
            },
            "dispatch_rework": {
                "selected_count": 0,
                "deferred_by_concurrency": [13],
                "failures": {
                    12: "failed to launch claude: timeout",
                    13: "deferred by concurrency cap (limit: 1)",
                },
            },
            "dispatch_reviews": {
                "selected_count": 1,
                "failed_count": 1,
                "failed": [{"pr": 100, "error": review_error}],
            },
        },
    )

    events = _extract_attention_events("owner/repo1", result)

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 3
    by_issue = {e.get("issue_number", e.get("pr")): e for e in error_events}
    assert by_issue[11]["error"] == "failed to launch claude: OSError"
    assert by_issue[12]["error"] == "failed to launch claude: timeout"
    assert by_issue[100]["error"] == review_error
    assert 13 not in by_issue

    digest = _build_fleet_attention_digest(events)
    entries = [e for e in digest.transitions if e.health == "ERROR"]
    assert len(entries) == 3
    assert all(e.issue_number != -1 for e in entries)
    assert any(review_error in (e.last_log_line or "") for e in entries)


def test_extract_attention_events_deferred_by_concurrency_truncation_desync() -> None:
    """Issue #1005 review: ``deferred_by_concurrency`` is truncated to
    ``_MAX_DEFERRED_CONCURRENCY_EXAMPLES`` (5) in the persisted payload, but
    the ``failures`` map it feeds is not. A set-membership exclusion check
    against the truncated list would silently re-report the 6th+ deferred
    issue as a genuine launch failure -- a diagnostic regression in exactly
    the dimension #1005 is about. Exclusion is matched by reason-string
    prefix instead (``DEFERRED_BY_CONCURRENCY_REASON_PREFIX``), which is
    immune to truncation of the list.
    """
    deferred_issue_numbers = [101, 102, 103, 104, 105, 106, 107]
    result = CommandResult(
        True,
        "loop complete",
        {
            "stalled": [],
            "errors": [],
            "intake": {"failed": []},
            "dispatch": {
                "selected_count": 0,
                # Truncated to 5, mirroring the real persisted payload shape.
                "deferred_by_concurrency": deferred_issue_numbers[:5],
                "deferred_by_concurrency_count": len(deferred_issue_numbers),
                "failures": {
                    n: "deferred by concurrency cap (limit: 0)" for n in deferred_issue_numbers
                },
            },
        },
    )

    events = _extract_attention_events("owner/repo1", result)

    error_events = [e for e in events if e["type"] == "error"]
    assert error_events == []


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_calls_loop_per_repo(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """fleet_loop calls app.loop() exactly once per repo with that repo's config."""
    # Setup registry
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "repo1"),
                "config_path": "orchestrator.config.yaml",
            },
            "owner/repo2": {
                "repo_root": str(tmp_path / "repo2"),
                "config_path": "orchestrator.config.yaml",
            },
        }
    }
    mock_load_registry.return_value = registry

    # Create temp repo dirs
    (tmp_path / "repo1").mkdir()
    (tmp_path / "repo2").mkdir()

    # Mock config and paths
    mock_config = OrchestratorConfig()
    mock_load_layered_config.return_value = mock_config
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths

    # Mock OrchestratorApp instances
    mock_app1 = MagicMock()
    mock_app2 = MagicMock()
    mock_app1.loop.return_value = CommandResult(True, "repo1 loop complete", {})
    mock_app2.loop.return_value = CommandResult(True, "repo2 loop complete", {})
    mock_app_class.side_effect = [mock_app1, mock_app2]

    # Mock GitHub
    mock_gh = MagicMock()
    mock_gh_class.return_value = mock_gh

    # Run fleet_loop
    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
    )

    # Verify loop() was called exactly once per repo
    assert mock_app1.loop.call_count == 1
    assert mock_app2.loop.call_count == 1

    # Verify loop() was called with correct args
    mock_app1.loop.assert_called_once_with(3, merge=True)
    mock_app2.loop.assert_called_once_with(3, merge=True)

    # Verify result includes both repos
    assert "repos" in result.data
    assert "owner/repo1" in result.data["repos"]
    assert "owner/repo2" in result.data["repos"]


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_work_only_calls_dispatch(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """fleet_loop with work_only=True calls app.dispatch() instead of loop()."""
    # Setup registry
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "repo1"),
                "config_path": "orchestrator.config.yaml",
            }
        }
    }
    mock_load_registry.return_value = registry

    # Create temp repo dir
    (tmp_path / "repo1").mkdir()

    # Mock config and paths
    mock_config = OrchestratorConfig()
    mock_load_layered_config.return_value = mock_config
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths

    # Mock OrchestratorApp
    mock_app = MagicMock()
    mock_app.dispatch.return_value = CommandResult(True, "repo1 dispatch complete", {})
    mock_app_class.return_value = mock_app

    # Mock GitHub
    mock_gh = MagicMock()
    mock_gh_class.return_value = mock_gh

    # Run fleet_loop with work_only=True
    fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=None,
        dry_run=False,
        work_only=True,
    )

    # Verify dispatch() was called instead of loop()
    assert mock_app.dispatch.call_count == 1
    assert mock_app.loop.call_count == 0

    # Verify dispatch() was called with correct args
    mock_app.dispatch.assert_called_once_with(3)


@patch("charlie_work.fleet_dispatch._load_registry")
def test_fleet_loop_missing_repo_root_skipped(
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """fleet_loop skips repos with missing repo_root and records failure."""
    # Setup registry with one missing repo
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "nonexistent"),
                "config_path": "orchestrator.config.yaml",
            }
        }
    }
    mock_load_registry.return_value = registry

    # Run fleet_loop
    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
    )

    # Verify result includes the failed repo
    assert "repos" in result.data
    assert "owner/repo1" in result.data["repos"]
    assert result.data["repos"]["owner/repo1"]["ok"] is False
    assert "missing, skipped" in result.data["repos"]["owner/repo1"]["message"]


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_missing_repo_root_records_lane_failure(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """#749: a repo whose repo_root no longer exists is recorded to events.db
    and the fleet digest, not only per_repo_results."""
    repo1_state_dir = tmp_path / "repo1-state"
    repo1_state_dir.mkdir(parents=True)
    repo2 = tmp_path / "repo2"
    repo2.mkdir()

    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "nonexistent"),
                "state_dir": str(repo1_state_dir),
            },
            "owner/repo2": {
                "repo_root": str(repo2),
            },
        }
    }
    mock_load_registry.return_value = registry

    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths

    mock_app2 = MagicMock()
    mock_app2.loop.return_value = CommandResult(True, "repo2 loop complete", {})
    mock_app_class.return_value = mock_app2
    mock_gh_class.return_value = MagicMock()

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
    )

    # repo1 is reported as a failed repo, repo2 still runs.
    assert result.data["repos"]["owner/repo1"]["ok"] is False
    assert "repo_root missing, skipped" in result.data["repos"]["owner/repo1"]["message"]
    assert result.data["repos"]["owner/repo2"]["ok"] is True
    assert mock_app_class.call_count == 1
    assert mock_app2.loop.call_count == 1
    assert mock_load_layered_config.call_count == 1

    # The failure is durably recorded to repo1's own events.db.
    state_path = layout.state_file_path(repo1_state_dir)
    recorded = query_events(state_path, kind="fleet_pass_config_error")
    assert len(recorded) == 1
    assert recorded[0]["level"] == "error"
    assert recorded[0]["payload"]["repo_key"] == "owner/repo1"
    assert "repo_root missing, skipped" in recorded[0]["payload"]["error"]

    # The fleet digest carries a matching ERROR entry.
    digest_events = result.data["digest"]["events"]
    error_events = [e for e in digest_events if e.get("repo_key") == "owner/repo1"]
    assert len(error_events) == 1
    assert error_events[0]["type"] == "error"
    assert "repo_root missing, skipped" in error_events[0]["error"]

    attention_digest = _build_fleet_attention_digest(digest_events)
    matching = [e for e in attention_digest.transitions if e.adapter_kind == "owner/repo1"]
    assert len(matching) == 1
    assert matching[0].health == "ERROR"


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_github_error_isolated(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """fleet_loop isolates GitHubError from one repo and continues to others."""
    # Setup registry with two repos
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "repo1"),
                "config_path": "orchestrator.config.yaml",
            },
            "owner/repo2": {
                "repo_root": str(tmp_path / "repo2"),
                "config_path": "orchestrator.config.yaml",
            },
        }
    }
    mock_load_registry.return_value = registry

    # Create temp repo dirs
    (tmp_path / "repo1").mkdir()
    (tmp_path / "repo2").mkdir()

    # Mock config and paths
    mock_config = OrchestratorConfig()
    mock_load_layered_config.return_value = mock_config
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths

    # Mock OrchestratorApp instances - first one raises GitHubError
    mock_app1 = MagicMock()
    mock_app2 = MagicMock()
    mock_app1.loop.side_effect = GitHubError("API rate limit exceeded")
    mock_app2.loop.return_value = CommandResult(True, "repo2 loop complete", {})
    mock_app_class.side_effect = [mock_app1, mock_app2]

    # Mock GitHub
    mock_gh = MagicMock()
    mock_gh_class.return_value = mock_gh

    # Run fleet_loop
    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
    )

    # Verify result includes both repos
    assert "repos" in result.data
    assert "owner/repo1" in result.data["repos"]
    assert "owner/repo2" in result.data["repos"]

    # Verify first repo failed but second succeeded
    assert result.data["repos"]["owner/repo1"]["ok"] is False
    assert result.data["repos"]["owner/repo2"]["ok"] is True

    # Verify overall result is False (one repo failed)
    assert result.ok is False


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_unclassified_exception_isolated(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """fleet_loop isolates an unclassified exception from one repo and continues to others."""
    # Setup registry with two repos
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "repo1"),
                "config_path": "orchestrator.config.yaml",
            },
            "owner/repo2": {
                "repo_root": str(tmp_path / "repo2"),
                "config_path": "orchestrator.config.yaml",
            },
        }
    }
    mock_load_registry.return_value = registry

    # Create temp repo dirs
    (tmp_path / "repo1").mkdir()
    (tmp_path / "repo2").mkdir()

    # Mock config and paths
    mock_config = OrchestratorConfig()
    mock_load_layered_config.return_value = mock_config
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths

    # Mock OrchestratorApp instances - first one raises an unclassified RuntimeError
    mock_app1 = MagicMock()
    mock_app2 = MagicMock()
    mock_app1.loop.side_effect = RuntimeError("provider response malformed")
    mock_app2.loop.return_value = CommandResult(True, "repo2 loop complete", {})
    mock_app_class.side_effect = [mock_app1, mock_app2]

    # Mock GitHub
    mock_gh = MagicMock()
    mock_gh_class.return_value = mock_gh

    # Run fleet_loop - should not propagate the RuntimeError
    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
    )

    # Verify both repos are present in the result
    assert "repos" in result.data
    assert "owner/repo1" in result.data["repos"]
    assert "owner/repo2" in result.data["repos"]

    # Verify repo1 failed, repo2 succeeded and was processed
    assert result.data["repos"]["owner/repo1"]["ok"] is False
    assert result.data["repos"]["owner/repo2"]["ok"] is True
    assert mock_app2.loop.call_count == 1

    # Verify the failing repo's message is recorded
    assert "fleet pass error" in result.data["repos"]["owner/repo1"].get("message", "")
    # The exception type must be part of the surfaced message (diagnosability).
    assert "RuntimeError" in result.data["repos"]["owner/repo1"]["message"]

    # Issue #738: the genuine lane-crash path (this test's RuntimeError raised
    # inside app.loop()) must set ``errored: True`` on the per-repo result data
    # at its point of origin in fleet_loop's ``except Exception`` handler, so
    # the supervisor headline can split "errored" from "completed with
    # conditions". The downstream headline-split tests plant this flag via a
    # synthetic fixture; this assertion verifies the flag is actually set by
    # the real exception path, not just honored when present.
    assert result.data["repos"]["owner/repo1"].get("errored") is True
    # The successful repo must not carry the marker.
    assert "errored" not in result.data["repos"]["owner/repo2"]

    # Verify overall result is False (one repo failed)
    assert result.ok is False


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_config_load_error_isolated(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """#6-G / G-AC4 (most important): a repo whose lane fails during startup —
    i.e. inside ``load_layered_config`` itself, before ``OrchestratorApp`` is
    ever constructed — must not prevent another repo's lane from running.

    This is distinct from ``test_fleet_loop_unclassified_exception_isolated``
    above, which raises inside ``app.loop()`` (config load succeeds for both
    repos there). The real 2026-07-29 incident (``ConfigError: unknown
    key(s) in config section 'cross_family': auto_verdict``) failed at
    config-load time, before any per-repo app object existed — this test
    pins isolation at that exact point. D-4 requires the per-repo ``except``
    to keep catching this; this test would fail loudly (as a fleet-wide
    exception) if a future change narrowed or removed it.

    G-AC6: the injected failure happens inside ``load_layered_config``,
    strictly before ``paths = runtime_paths(...)`` executes in the same try
    block, so ``paths`` is genuinely unbound in repo1's except handler (not
    merely untested) -- see the ``mock_runtime_paths.call_count`` assertion
    below.
    """
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "repo1"),
                "config_path": "orchestrator.config.yaml",
            },
            "owner/repo2": {
                "repo_root": str(tmp_path / "repo2"),
                "config_path": "orchestrator.config.yaml",
            },
        }
    }
    mock_load_registry.return_value = registry

    (tmp_path / "repo1").mkdir()
    (tmp_path / "repo2").mkdir()

    # repo1's config load raises during startup; repo2's succeeds. Only one
    # OrchestratorApp is ever constructed (for repo2) because repo1 never
    # reaches that line — mock_app_class.return_value (not side_effect list)
    # pins that.
    #
    # Keyed by repo_root rather than a fixed-length call-order list: the
    # fleet pass also calls load_layered_config a second time for repo1 from
    # compute_api_worker_fleet_report (it re-loads any repo missing from
    # preloaded_configs, which repo1 is, since its first load failed). A
    # positional side_effect list of length 2 would exhaust after the two
    # per-repo-loop calls and raise a spurious StopIteration on that third
    # call. Retrying the same broken config deterministically re-raises the
    # same ConfigError, matching real load_layered_config behavior.
    repo1_root = tmp_path / "repo1"

    def _load_layered_config_side_effect(
        repo_root: Path, *args: Any, **kwargs: Any
    ) -> OrchestratorConfig:
        if Path(repo_root) == repo1_root:
            raise ConfigError(
                "unknown key(s) in config section 'cross_family': auto_verdict "
                "(valid: enabled, model, command)"
            )
        return OrchestratorConfig()

    mock_load_layered_config.side_effect = _load_layered_config_side_effect
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths

    mock_app2 = MagicMock()
    mock_app2.loop.return_value = CommandResult(True, "repo2 loop complete", {})
    mock_app_class.return_value = mock_app2

    mock_gh = MagicMock()
    mock_gh_class.return_value = mock_gh

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
    )

    # repo1 failed at startup; repo2's lane actually ran. This is the
    # isolation proof: only one OrchestratorApp was ever built, and its
    # loop() was called exactly once, for the surviving repo.
    assert result.data["repos"]["owner/repo1"]["ok"] is False
    assert result.data["repos"]["owner/repo2"]["ok"] is True
    assert mock_app_class.call_count == 1
    assert mock_app2.loop.call_count == 1

    # G-AC6: repo1's ConfigError is raised inside load_layered_config,
    # strictly before `paths = runtime_paths(...)` is reached in that same
    # try block. runtime_paths is therefore called exactly once (for repo2
    # only) -- proving `paths` is genuinely unbound in repo1's except
    # handler, not just untested. The handler itself never references
    # `paths` (it uses `repo_root`/`entry`, both bound before the try); if a
    # future change added a `paths.state_file` reference there, this would
    # raise UnboundLocalError *inside* the except block, which escapes the
    # per-repo isolation boundary entirely (D-4) instead of being caught by
    # it -- this assertion is what pins that it never happens.
    assert mock_runtime_paths.call_count == 1

    message = result.data["repos"]["owner/repo1"]["message"]
    assert "fleet pass error" in message
    assert "ConfigError" in message
    assert "cross_family" in message

    # G-AC2: the raw digest feed carries the failure even though app.loop()
    # never ran for repo1 — _extract_attention_events() (which only runs
    # after a successful loop()) never fires for repo1, so this event must
    # come from the except block itself.
    digest_events = result.data["digest"]["events"]
    error_events = [e for e in digest_events if e.get("repo_key") == "owner/repo1"]
    assert len(error_events) == 1
    assert error_events[0]["type"] == "error"

    # Confirm the reused "error" branch actually maps this to a real
    # AttentionEntry (health=ERROR, already desktop-toast-eligible via
    # _DESKTOP_SEVERITIES) rather than silently falling through.
    attention_digest = _build_fleet_attention_digest(digest_events)
    matching = [e for e in attention_digest.transitions if e.adapter_kind == "owner/repo1"]
    assert len(matching) == 1
    assert matching[0].health == "ERROR"
    assert "cross_family" in (matching[0].last_log_line or "")

    assert result.ok is False


@patch("charlie_work.fleet_dispatch.emit_digest")
@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_lane_failure_reaches_real_emit_digest(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    mock_emit_digest: MagicMock,
    tmp_path: Path,
) -> None:
    """#6-G / G-AC2: the lane-failure entry must reach the real ``emit_digest``
    sink, not just the raw ``digest["events"]`` list.

    ``test_fleet_loop_config_load_error_isolated`` proves the raw event dict
    is correct and that ``_build_fleet_attention_digest`` maps it to
    ``health=ERROR`` -- but it calls ``_build_fleet_attention_digest`` itself
    (out of band) and passes ``global_config=None`` to ``fleet_loop``, so the
    real ``if notify_config is not None and notify_config.enabled`` /
    ``if attention_digest.transitions`` gates that guard the actual
    ``emit_digest(...)`` call at the end of ``fleet_loop`` are never entered.
    That leaves open exactly the failure mode this AC exists to close: a gate
    keyed on something only the loop()-succeeded path populates would still
    pass the other test while leaving the desktop/file sink silent. This test
    turns notify on for real and asserts ``emit_digest`` fires with an ERROR
    entry for the failed repo, driving ``fleet_loop`` itself on the exact
    pass where ``app.loop()`` never ran for repo1 -- rather than re-deriving
    the mapping out of band.
    """
    from charlie_work.config import NotifyConfig

    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "repo1"),
                "config_path": "orchestrator.config.yaml",
            },
            "owner/repo2": {
                "repo_root": str(tmp_path / "repo2"),
                "config_path": "orchestrator.config.yaml",
            },
        }
    }
    mock_load_registry.return_value = registry
    (tmp_path / "repo1").mkdir()
    (tmp_path / "repo2").mkdir()

    repo1_root = tmp_path / "repo1"

    def _load_layered_config_side_effect(
        repo_root: Path, *args: Any, **kwargs: Any
    ) -> OrchestratorConfig:
        if Path(repo_root) == repo1_root:
            raise ConfigError(
                "unknown key(s) in config section 'cross_family': auto_verdict "
                "(valid: enabled, model, command)"
            )
        return OrchestratorConfig()

    mock_load_layered_config.side_effect = _load_layered_config_side_effect
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths

    mock_app2 = MagicMock()
    mock_app2.loop.return_value = CommandResult(True, "repo2 loop complete", {})
    mock_app_class.return_value = mock_app2

    mock_gh = MagicMock()
    mock_gh_class.return_value = mock_gh

    # The real gate: notify_config comes from the *outer* global_config
    # parameter (not from a per-repo loaded config), so this alone drives
    # whether the digest-build-and-emit block at the end of fleet_loop runs.
    global_config = OrchestratorConfig(
        notify=NotifyConfig(
            enabled=True,
            sink="file",
            file_path=str(tmp_path / "digest.jsonl"),
        )
    )

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=global_config,
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
    )

    assert result.data["repos"]["owner/repo1"]["ok"] is False

    # The discriminating assertion: the real sink actually fired, on the pass
    # where repo1's app.loop() never ran (only repo2's did).
    assert mock_emit_digest.called is True
    emitted_digest = mock_emit_digest.call_args[0][1]
    matching = [e for e in emitted_digest.transitions if e.adapter_kind == "owner/repo1"]
    assert len(matching) == 1
    assert matching[0].health == "ERROR"
    assert "cross_family" in (matching[0].last_log_line or "")


def test_lane_failure_state_path_prefers_registry_state_dir(tmp_path: Path) -> None:
    """_lane_failure_state_path uses the registry's recorded state_dir when
    present — the common case for a repo that previously registered
    successfully and only later started failing (e.g. self-deploy version
    skew, the actual 2026-07-29 shape)."""
    repo_root = tmp_path / "repo"
    recorded_state_dir = tmp_path / "custom-state"
    entry = {"repo_root": str(repo_root), "state_dir": str(recorded_state_dir)}

    result = _lane_failure_state_path(repo_root, entry)

    assert result == layout.state_file_path(recorded_state_dir)


def test_lane_failure_state_path_falls_back_to_default_without_registry_entry(
    tmp_path: Path,
) -> None:
    """Without a recorded state_dir (a repo that has never registered
    successfully), _lane_failure_state_path falls back to the conventional
    default location so the failure is still recorded somewhere findable."""
    repo_root = tmp_path / "repo"
    entry: dict[str, Any] = {"repo_root": str(repo_root)}

    result = _lane_failure_state_path(repo_root, entry)

    assert result == layout.state_file_path(layout.default_state_root(repo_root))


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_real_unknown_config_key_reproduces_incident(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """#6-G / G-AC5: full reproduction of the 2026-07-29 incident.

    An unknown key in one repo's real config file (``cross_family:
    totally_unknown_key``, mirroring the actual ``cross_family: auto_verdict``
    version-skew incident) drives the real, unmocked ``load_layered_config``
    to raise ``ConfigError``. This proves: (a) an events.db row is recorded
    for the failing repo (queryable via query_events(kind=
    "fleet_pass_config_error")), (b) the fleet digest carries a matching
    AttentionEntry, and (c) a second, healthy repo's lane still completes —
    while a doctor check run against the failing repo's own state directory
    surfaces the same event as a finding (see
    test_check_recent_lane_failures_surfaces_past_event in test_doctor.py,
    which covers the doctor half of this chain with the same event shape).

    Only load_layered_config is left unmocked; runtime_paths/GitHub/
    OrchestratorApp stay mocked exactly as in the other fleet_loop tests —
    this isolates "does the real config parser really raise ConfigError for
    an unknown key, and does fleet_loop's except really catch it" from the
    rest of the per-repo machinery.
    """
    repo1 = tmp_path / "repo1"
    repo1.mkdir()
    (repo1 / "orchestrator.config.yaml").write_text(
        "labels:\n"
        "  ready: automated-ready\n"
        "runtime:\n"
        "  state_dir: .var/charlie-work\n"
        "cross_family:\n"
        "  totally_unknown_key: true\n",
        encoding="utf-8",
    )
    repo1_state_dir = repo1 / ".var" / "charlie-work"
    repo1_state_dir.mkdir(parents=True)

    repo2 = tmp_path / "repo2"
    repo2.mkdir()

    mock_load_registry.return_value = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(repo1),
                "state_dir": str(repo1_state_dir),
                # No config_path override: load_layered_config resolves the
                # real file above via find_config_path(repo_root, None).
            },
            "owner/repo2": {
                "repo_root": str(repo2),
            },
        }
    }

    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths

    mock_app2 = MagicMock()
    mock_app2.loop.return_value = CommandResult(True, "repo2 loop complete", {})
    mock_app_class.return_value = mock_app2
    mock_gh_class.return_value = MagicMock()

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
    )

    # (c) repo2's lane proceeded despite repo1's real ConfigError.
    assert result.data["repos"]["owner/repo1"]["ok"] is False
    assert result.data["repos"]["owner/repo2"]["ok"] is True
    assert mock_app_class.call_count == 1
    assert mock_app2.loop.call_count == 1

    # G-AC6: same pre-`paths`-binding failure point as
    # test_fleet_loop_config_load_error_isolated, this time via the real,
    # unmocked load_layered_config raising the real ConfigError rather than
    # a mock side_effect. runtime_paths is called exactly once (repo2 only).
    assert mock_runtime_paths.call_count == 1

    message = result.data["repos"]["owner/repo1"]["message"]
    assert "ConfigError" in message
    assert "cross_family" in message
    assert "totally_unknown_key" in message

    # (a) the failure is durably recorded to repo1's own events.db.
    state_path = layout.state_file_path(repo1_state_dir)
    recorded = query_events(state_path, kind="fleet_pass_config_error")
    assert len(recorded) == 1
    assert recorded[0]["level"] == "error"
    assert recorded[0]["payload"]["repo_key"] == "owner/repo1"
    assert "totally_unknown_key" in recorded[0]["payload"]["error"]

    # (b) the fleet digest carries a matching entry.
    attention_digest = _build_fleet_attention_digest(result.data["digest"]["events"])
    matching = [e for e in attention_digest.transitions if e.adapter_kind == "owner/repo1"]
    assert len(matching) == 1
    assert matching[0].health == "ERROR"


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_dry_run_propagates(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """fleet_loop with dry_run=True propagates to every GitHub and OrchestratorApp."""
    # Setup registry
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "repo1"),
                "config_path": "orchestrator.config.yaml",
            }
        }
    }
    mock_load_registry.return_value = registry

    # Create temp repo dir
    (tmp_path / "repo1").mkdir()

    # Mock config and paths
    mock_config = OrchestratorConfig()
    mock_load_layered_config.return_value = mock_config
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths

    # Mock OrchestratorApp
    mock_app = MagicMock()
    mock_app.loop.return_value = CommandResult(True, "repo1 loop complete", {})
    mock_app_class.return_value = mock_app

    # Mock GitHub
    mock_gh = MagicMock()
    mock_gh_class.return_value = mock_gh

    # Run fleet_loop with dry_run=True
    fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=True,
        dry_run=True,
        work_only=False,
    )

    # Verify GitHub was constructed with dry_run=True and the runtime config
    mock_gh_class.assert_called_once_with(
        repo_root=tmp_path / "repo1",
        runtime=mock_config.runtime,
        dry_run=True,
    )

    # Verify OrchestratorApp was constructed with dry_run=True
    mock_app_class.assert_called_once()
    call_kwargs = mock_app_class.call_args[1]
    assert call_kwargs["dry_run"] is True


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_digest_aggregation(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """fleet_loop aggregates attention events from all repos into one digest."""
    # Setup registry
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "repo1"),
                "config_path": "orchestrator.config.yaml",
            },
            "owner/repo2": {
                "repo_root": str(tmp_path / "repo2"),
                "config_path": "orchestrator.config.yaml",
            },
        }
    }
    mock_load_registry.return_value = registry

    # Create temp repo dirs
    (tmp_path / "repo1").mkdir()
    (tmp_path / "repo2").mkdir()

    # Mock config and paths
    mock_config = OrchestratorConfig()
    mock_load_layered_config.return_value = mock_config
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths

    # Mock OrchestratorApp instances with attention events
    mock_app1 = MagicMock()
    mock_app2 = MagicMock()
    mock_app1.loop.return_value = CommandResult(
        True,
        "repo1 loop complete",
        {
            "stalled": [{"session_id": "sess1", "issue_number": 123, "reason": "timeout"}],
            "errors": [],
        },
    )
    mock_app2.loop.return_value = CommandResult(
        True,
        "repo2 loop complete",
        {
            "stalled": [],
            "errors": [{"pr": 456, "error": "merge conflict"}],
        },
    )
    mock_app_class.side_effect = [mock_app1, mock_app2]

    # Mock GitHub
    mock_gh = MagicMock()
    mock_gh_class.return_value = mock_gh

    # Run fleet_loop
    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
    )

    # Verify digest includes events from both repos
    assert "digest" in result.data
    digest = result.data["digest"]
    assert "events" in digest
    assert len(digest["events"]) == 2

    # Verify events are from different repos
    event_repo_keys = {e["repo_key"] for e in digest["events"]}
    assert event_repo_keys == {"owner/repo1", "owner/repo2"}

    # Verify orphan_sweep_calls metric is present
    assert "orphan_sweep_calls" in digest
    assert digest["orphan_sweep_calls"] == 2  # One per repo


@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch._load_registry")
def test_fleet_loop_skips_repo_when_supervisor_lock_held(
    mock_load_registry: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_gh_class: MagicMock,
    mock_app_class: MagicMock,
    mock_try_acquire: MagicMock,
    tmp_path: Path,
) -> None:
    """If the supervisor lock is held, the repo is skipped and others are still processed."""
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "repo1"),
                "config_path": "orchestrator.config.yaml",
            },
            "owner/repo2": {
                "repo_root": str(tmp_path / "repo2"),
                "config_path": "orchestrator.config.yaml",
            },
        }
    }
    mock_load_registry.return_value = registry

    (tmp_path / "repo1").mkdir()
    (tmp_path / "repo2").mkdir()

    mock_config = OrchestratorConfig()
    mock_load_layered_config.return_value = mock_config
    mock_paths = MagicMock()
    mock_runtime_paths.return_value = mock_paths

    mock_app2 = MagicMock()
    mock_app2.loop.return_value = CommandResult(True, "repo2 loop complete", {})
    mock_app_class.return_value = mock_app2

    mock_gh = MagicMock()
    mock_gh_class.return_value = mock_gh

    lock = MagicMock()
    # repo1 is held (external supervised loop), repo2 is free
    mock_try_acquire.side_effect = [None, lock]

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
    )

    # repo1 was skipped because the lock was held
    assert "owner/repo1" in result.data["repos"]
    repo1_data = result.data["repos"]["owner/repo1"]
    assert repo1_data["ok"] is True
    assert repo1_data["skipped"] is True
    assert repo1_data["reason"] == "supervisor_lock_held"

    # repo2 still ran
    assert "owner/repo2" in result.data["repos"]
    assert result.data["repos"]["owner/repo2"]["ok"] is True
    assert mock_app2.loop.call_count == 1

    # The lock taken for repo2 was released
    lock.release.assert_called_once()

    # The fleet digest surfaces the skipped repo
    skipped_events = [e for e in result.data["digest"]["events"] if e["type"] == "skipped"]
    assert len(skipped_events) == 1
    assert skipped_events[0]["repo_key"] == "owner/repo1"
    assert skipped_events[0]["reason"] == "supervisor_lock_held"


@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch._load_registry")
def test_fleet_loop_releases_lock_after_each_repo(
    mock_load_registry: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_gh_class: MagicMock,
    mock_app_class: MagicMock,
    mock_try_acquire: MagicMock,
    tmp_path: Path,
) -> None:
    """The per-repo supervisor lock is released after each repo, and the next repo can acquire it."""
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "repo1"),
                "config_path": "orchestrator.config.yaml",
            },
            "owner/repo2": {
                "repo_root": str(tmp_path / "repo2"),
                "config_path": "orchestrator.config.yaml",
            },
        }
    }
    mock_load_registry.return_value = registry

    (tmp_path / "repo1").mkdir()
    (tmp_path / "repo2").mkdir()

    mock_config = OrchestratorConfig()
    mock_load_layered_config.return_value = mock_config
    mock_paths = MagicMock()
    mock_runtime_paths.return_value = mock_paths

    mock_app1 = MagicMock()
    mock_app2 = MagicMock()
    mock_app1.loop.return_value = CommandResult(True, "repo1 loop complete", {})
    mock_app2.loop.return_value = CommandResult(True, "repo2 loop complete", {})
    mock_app_class.side_effect = [mock_app1, mock_app2]

    mock_gh = MagicMock()
    mock_gh_class.return_value = mock_gh

    lock1 = MagicMock()
    lock2 = MagicMock()
    mock_try_acquire.side_effect = [lock1, lock2]

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
    )

    assert result.ok is True
    assert mock_app1.loop.call_count == 1
    assert mock_app2.loop.call_count == 1
    assert lock1.release.called
    assert lock2.release.called


@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch._load_registry")
def test_fleet_loop_work_only_skips_locked_repo(
    mock_load_registry: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_gh_class: MagicMock,
    mock_app_class: MagicMock,
    mock_try_acquire: MagicMock,
    tmp_path: Path,
) -> None:
    """The dispatch-only (work_only) path also respects the supervisor lock."""
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "repo1"),
                "config_path": "orchestrator.config.yaml",
            }
        }
    }
    mock_load_registry.return_value = registry

    (tmp_path / "repo1").mkdir()

    mock_config = OrchestratorConfig()
    mock_load_layered_config.return_value = mock_config
    mock_paths = MagicMock()
    mock_runtime_paths.return_value = mock_paths

    mock_app = MagicMock()
    mock_app_class.return_value = mock_app

    mock_gh = MagicMock()
    mock_gh_class.return_value = mock_gh

    mock_try_acquire.return_value = None

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=None,
        dry_run=False,
        work_only=True,
    )

    assert mock_app.dispatch.call_count == 0
    repo_data = result.data["repos"]["owner/repo1"]
    assert repo_data["ok"] is True
    assert repo_data["skipped"] is True
    assert repo_data["reason"] == "supervisor_lock_held"


@patch("charlie_work.fleet_registry._load_registry")
@patch("charlie_work.fleet_registry.GitHub")
def test_count_fleet_runners_propagates_runtime_config(
    mock_gh_class: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """count_fleet_runners passes the caller's RuntimeConfig to every GitHub client."""
    repo_root = tmp_path / "repo1"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()

    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(repo_root),
                "config_path": str(repo_root / "orchestrator.config.yaml"),
            }
        }
    }
    mock_load_registry.return_value = registry

    mock_gh = MagicMock()
    mock_gh.run.return_value = {"runners": [{"busy": False}, {"busy": True}]}
    mock_gh_class.return_value = mock_gh

    runtime = RuntimeConfig(gh_max_retries=7, gh_retry_base_seconds=0.5)
    total, busy, skipped = count_fleet_runners(str(tmp_path / "fleet"), runtime=runtime)

    assert total == 2
    assert busy == 1
    assert skipped == []
    mock_gh_class.assert_called_once_with(repo_root=repo_root, runtime=runtime)


@patch("charlie_work.fleet_registry._load_registry")
@patch("charlie_work.fleet_registry.GitHub")
def test_count_fleet_runners_skips_repo_on_unreadable_response(
    mock_gh_class: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """Issue #756: a repo whose runner query is unreadable (gh.run raises
    GitHubError, e.g. because github.py's own boundary fix now raises on
    empty-stdout-success) must land in skipped_repos, not silently contribute
    0 to total_runners/total_busy_runners.

    Before the #756 fix, ``GitHub.run()`` could return a bare ``None`` for
    this case, which ``runners_data.get(...) if runners_data else []``
    coerced to "this repo has zero runners" -- feeding decide_autoscale() a
    false reading that looks identical to a genuinely idle repo. The
    surrounding ``except (GitHubError, Exception)`` here already routes any
    raised exception to skipped_repos; this test proves that contract holds
    end to end.
    """
    repo_root = tmp_path / "repo1"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()

    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(repo_root),
                "config_path": str(repo_root / "orchestrator.config.yaml"),
            }
        }
    }
    mock_load_registry.return_value = registry

    mock_gh = MagicMock()
    mock_gh.run.side_effect = GitHubError(
        "gh exited 0 with empty stdout for command: gh api ...; "
        "cannot distinguish an empty result from an unreadable one"
    )
    mock_gh_class.return_value = mock_gh

    total, busy, skipped = count_fleet_runners(str(tmp_path / "fleet"))

    assert total == 0
    assert busy == 0
    assert skipped == ["owner/repo1"]


def _drained_fleet_result() -> CommandResult:
    return CommandResult(
        True,
        "fleet pass complete",
        {
            "repos": {"owner/repo": {"ok": True}},
            "digest": {"count": 0, "events": []},
        },
    )


def _active_fleet_result(dispatched: int = 1) -> CommandResult:
    return CommandResult(
        True,
        "fleet pass complete",
        {
            "repos": {
                "owner/repo": {
                    "ok": True,
                    "dispatch": {"selected_count": dispatched},
                }
            },
            "digest": {"count": 0, "events": []},
        },
    )


def test_is_fleet_pass_active_true_on_dispatch() -> None:
    assert _is_fleet_pass_active(_active_fleet_result()) is True


def test_is_fleet_pass_active_false_when_drained() -> None:
    assert _is_fleet_pass_active(_drained_fleet_result()) is False


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_loops_until_max_passes(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
) -> None:
    """run_fleet_supervise runs fleet_loop repeatedly until max_passes is reached."""
    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=3, clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    assert result.data["passes"] == 3
    assert mock_fleet_loop.call_count == 3
    assert fc.sleep_calls == [5.0, 5.0, 5.0]
    # #862 AC4: exhausting the pass budget is a deliberate stop. It shares
    # ok=True with the restart-requesting exits, so the launcher distinguishes
    # them on this field alone -- a regression here would relaunch forever.
    assert result.data["exit_reason"] == "max_passes"
    assert result.data["restart_requested"] is False


def _failed_fleet_result(
    repo_messages: dict[str, str | None],
) -> CommandResult:
    """A fleet pass with one or more failing repos, each carrying its own message.

    Mirrors the shape ``fleet_loop`` actually returns (fleet_dispatch.py:1559-1565):
    every repo entry gets an ``ok`` bool and a ``message`` string alongside its
    ordinary data, regardless of pass outcome.
    """
    repos = {key: {"ok": False, "message": message} for key, message in repo_messages.items()}
    return CommandResult(
        False,
        "fleet pass complete",
        {"repos": repos, "digest": {"count": 0, "events": []}},
    )


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_pass_summary_includes_failure_reason(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Issue #893: the per-pass summary line must surface *why* repos failed.

    ``repos_data[key]["message"]`` (fleet_dispatch.py:1564) already carries the
    per-repo failure reason -- e.g. "loop completed with N PR error(s)" -- all
    the way to the print site, but the summary line only ever counted ``ok``
    and dropped the message. A repeating "0 ok, N failed" with no reason reads
    identically to a real outage as it does to a known, acked-releasable
    control (e.g. the unauthorized-merge tripwire) firing every pass.
    """
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_fleet_loop.return_value = _failed_fleet_result(
        {"owner/repo1": "loop completed with 2 PR error(s)"}
    )

    fc = _FakeClock(auto_advance=1.0)
    run_fleet_supervise(max_passes=1, clock=fc.now, sleep=fc.sleep)

    captured = capsys.readouterr()
    summary_lines = [line for line in captured.out.splitlines() if "fleet pass 1:" in line]
    assert summary_lines, f"no per-pass summary line found in output: {captured.out!r}"
    assert "loop completed with 2 PR error(s)" in summary_lines[0], (
        f"the failure reason must be on the summary line, not just counted: {summary_lines[0]!r}"
    )


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_pass_summary_dedupes_repeated_reasons(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """N repos failing for one shared cause must not repeat the string N times."""
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    same_reason = "unauthorized-merge tripwire: unacked finding #502"
    mock_fleet_loop.return_value = _failed_fleet_result(
        {"owner/repo1": same_reason, "owner/repo2": same_reason}
    )

    fc = _FakeClock(auto_advance=1.0)
    run_fleet_supervise(max_passes=1, clock=fc.now, sleep=fc.sleep)

    captured = capsys.readouterr()
    summary_lines = [line for line in captured.out.splitlines() if "fleet pass 1:" in line]
    assert summary_lines
    assert summary_lines[0].count(same_reason) == 1, (
        f"a shared failure reason must be deduped, not repeated per repo: {summary_lines[0]!r}"
    )


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_pass_summary_bounds_long_reason(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A single pathological message must not let the log line grow unbounded."""
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    huge_reason = "x" * 5000
    mock_fleet_loop.return_value = _failed_fleet_result({"owner/repo1": huge_reason})

    fc = _FakeClock(auto_advance=1.0)
    run_fleet_supervise(max_passes=1, clock=fc.now, sleep=fc.sleep)

    captured = capsys.readouterr()
    summary_lines = [line for line in captured.out.splitlines() if "fleet pass 1:" in line]
    assert summary_lines
    assert len(summary_lines[0]) < 2000, (
        f"an unbounded per-repo message must not dominate the summary line "
        f"(got {len(summary_lines[0])} chars)"
    )


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_pass_summary_guards_missing_message(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A failed repo with no/empty/whitespace-only message must not crash or append junk.

    Whitespace-only is the case that a naive ``if r.get("message")`` guard
    (truthy on unstripped text) lets through as a dangling ``" []"`` -- the
    filter must run *after* stripping, not before.
    """
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_fleet_loop.return_value = _failed_fleet_result(
        {"owner/repo1": None, "owner/repo2": "", "owner/repo3": "   "}
    )

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=1, clock=fc.now, sleep=fc.sleep)

    assert result.ok is True  # the supervisor loop itself must not crash
    captured = capsys.readouterr()
    summary_lines = [line for line in captured.out.splitlines() if "fleet pass 1:" in line]
    assert summary_lines
    # Issue #738: non-fatal ok=False conditions are "with conditions", not
    # "failed" -- the three repos here have no ``errored`` flag, so they land
    # in the conditions bucket and the errored count stays at zero.
    assert "3 with conditions" in summary_lines[0]
    assert "0 errored" in summary_lines[0]
    # No dangling empty reason marker when every message is absent -- check
    # only the text after "fleet pass N:" so the leading "[HH:MM:SS]"
    # timestamp bracket (unrelated to the reason suffix) is not confused for it.
    after_prefix = summary_lines[0].split("fleet pass 1:", 1)[1]
    assert "[]" not in after_prefix
    assert not after_prefix.rstrip().endswith("[")


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_pass_summary_silent_when_all_ok(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A fully healthy pass must not grow a reason suffix at all."""
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_fleet_loop.return_value = _drained_fleet_result()

    fc = _FakeClock(auto_advance=1.0)
    run_fleet_supervise(max_passes=1, clock=fc.now, sleep=fc.sleep)

    captured = capsys.readouterr()
    summary_lines = [line for line in captured.out.splitlines() if "fleet pass 1:" in line]
    assert summary_lines
    # Issue #738: a fully healthy pass reports zero errored and zero
    # conditions, not a single "0 failed" blob.
    assert "0 errored" in summary_lines[0]
    assert "0 with conditions" in summary_lines[0]
    # No reason suffix at all when nothing failed -- check only the text after
    # "fleet pass N:" so the leading "[HH:MM:SS]" timestamp bracket is not
    # confused for a (nonexistent) reason marker.
    after_prefix = summary_lines[0].split("fleet pass 1:", 1)[1]
    assert "[" not in after_prefix


def _mixed_fleet_result(
    conditions: dict[str, str | None] | None = None,
    errored: dict[str, str | None] | None = None,
    ok: dict[str, dict[str, Any]] | None = None,
) -> CommandResult:
    """A fleet pass with explicit errored / with-conditions / ok buckets.

    Mirrors the shape ``fleet_loop`` returns after issue #738: the exception
    path sets ``errored: True`` on the result data, while non-fatal
    ``ok=False`` conditions from ``app.loop()`` do not. This helper lets a
    test plant each bucket independently so the headline split is exercised
    in isolation.
    """
    repos: dict[str, dict[str, Any]] = {}
    for key, msg in (conditions or {}).items():
        repos[key] = {"ok": False, "message": msg}
    for key, msg in (errored or {}).items():
        repos[key] = {"ok": False, "message": msg, "errored": True}
    for key, data in (ok or {}).items():
        entry = {"ok": True}
        entry.update(data)
        repos[key] = entry
    return CommandResult(
        False,
        "fleet pass complete",
        {"repos": repos, "digest": {"count": 0, "events": []}},
    )


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_pass_summary_splits_errored_from_conditions(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Issue #738: the headline must separate lane crashes from non-fatal conditions.

    A pass with one crashed repo (``errored: True``) and one repo that
    completed with a non-fatal condition (``ok=False``, no ``errored``) must
    produce ``1 errored, 1 with conditions`` -- not the old ``2 failed`` that
    painted both red and gated nothing.
    """
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_fleet_loop.return_value = _mixed_fleet_result(
        conditions={"owner/repo1": "loop completed with 2 PR error(s)"},
        errored={"owner/repo2": "fleet pass error: RuntimeError: boom"},
    )

    fc = _FakeClock(auto_advance=1.0)
    run_fleet_supervise(max_passes=1, clock=fc.now, sleep=fc.sleep)

    captured = capsys.readouterr()
    summary_lines = [line for line in captured.out.splitlines() if "fleet pass 1:" in line]
    assert summary_lines
    line = summary_lines[0]
    assert "1 errored" in line, f"crashed repo must count as errored: {line!r}"
    assert "1 with conditions" in line, f"non-fatal repo must count as conditions: {line!r}"
    # The old undifferentiated "failed" count must not appear.
    assert "2 failed" not in line, f"old red-everywhere count must be gone: {line!r}"


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_pass_summary_all_errored(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Issue #738: a pass where every repo crashed reports N errored, 0 conditions.

    This is the genuine-outage case the old gauge could not distinguish from
    a routine pass -- two repos both crashing is now unambiguously ``2 errored``.
    """
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_fleet_loop.return_value = _mixed_fleet_result(
        errored={
            "owner/repo1": "fleet pass error: RuntimeError: boom",
            "owner/repo2": "fleet pass error: ConfigError: bad",
        }
    )

    fc = _FakeClock(auto_advance=1.0)
    run_fleet_supervise(max_passes=1, clock=fc.now, sleep=fc.sleep)

    captured = capsys.readouterr()
    summary_lines = [line for line in captured.out.splitlines() if "fleet pass 1:" in line]
    assert summary_lines
    line = summary_lines[0]
    assert "2 errored" in line
    assert "0 with conditions" in line


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_pass_summary_logs_non_ok_reason_at_warning(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Issue #738: every non-ok repo's reason must be emitted at WARNING.

    The exception path already logs via ``logger.exception`` inside
    ``fleet_loop``; this covers the non-fatal ``ok=False`` conditions from
    ``app.loop()`` that were previously silent in the supervisor log. The
    reason must be recoverable from the log after the fact, not just from the
    deduped/truncated summary suffix.
    """
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_fleet_loop.return_value = _mixed_fleet_result(
        conditions={"owner/repo1": "loop completed with 2 PR error(s)"},
    )

    fc = _FakeClock(auto_advance=1.0)
    with caplog.at_level(logging.WARNING, logger="charlie_work.fleet_dispatch"):
        run_fleet_supervise(max_passes=1, clock=fc.now, sleep=fc.sleep)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "owner/repo1" in r.getMessage() and "loop completed with 2 PR error(s)" in r.getMessage()
        for r in warnings
    ), f"non-ok repo reason must be logged at WARNING, got: {[r.getMessage() for r in warnings]!r}"


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_final_summary_splits_errored_from_conditions(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
) -> None:
    """Issue #738: the final supervisor summary line also splits the counts.

    The aggregate ``fleet supervisor complete`` line used the same
    ``total_failed_repos`` counter as the per-pass headline and had the same
    defect. It must now report ``N errored, N with conditions`` instead of
    ``N failed repo(s)``.
    """
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_fleet_loop.return_value = _mixed_fleet_result(
        conditions={"owner/repo1": "loop completed with 1 PR error(s)"},
        errored={"owner/repo2": "fleet pass error: RuntimeError: boom"},
    )

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=1, clock=fc.now, sleep=fc.sleep)

    assert "1 errored" in result.message
    assert "1 with conditions" in result.message
    assert "failed repo(s)" not in result.message
    # The return data carries the split counters alongside the legacy sum.
    assert result.data["total_errored_repos"] == 1
    assert result.data["total_conditions_repos"] == 1
    assert result.data["total_failed_repos"] == 2


def test_run_fleet_supervise_loop_reports_ok_on_a_clean_child_exit() -> None:
    """The wrapper is transparent when the supervisor stops deliberately."""
    result = run_fleet_supervise_loop(spawn=lambda _n: 0, max_relaunches=3)

    assert result.ok is True
    assert result.data["launches"] == 1
    assert result.data["cap_reached"] is False


def test_run_fleet_supervise_loop_reports_ok_when_the_cap_is_hit() -> None:
    """Hitting the cap is a clean handoff, not a failure.

    Stopping at the bound is the wrapper doing its job: it returns restart
    authority to the 5-minute trigger instead of spinning. Reporting it as
    ok=False would exit 1, which `except Exception` in `run_fleet_supervise`
    already uses -- collapsing "self-deploy is not converging" and "supervisor
    crashed" into one indistinguishable code. That is #862's own defect shape
    one layer up, so the cap is signalled by the event and log instead.
    """
    recorded: list[object] = []
    result = run_fleet_supervise_loop(
        spawn=lambda _n: EXIT_RESTART_REQUESTED,
        max_relaunches=2,
        on_cap_reached=recorded.append,
    )

    assert result.ok is True
    assert result.data["cap_reached"] is True
    assert result.data["launches"] == 3
    assert result.data["relaunches"] == 2
    # Never exit 3 upward: the wrapper is the thing that consumed the restart
    # request, so re-signalling it would ask the launcher to relaunch too.
    assert "restart_requested" not in result.data
    # ok=True is only defensible because the cap still announces itself.
    assert len(recorded) == 1


def test_run_fleet_supervise_loop_distinguishes_a_cap_from_an_abort() -> None:
    """The paired control for the test above -- ok=True must not mask a crash.

    Both conditions stop the wrapper, and the whole argument for ok=True on cap
    is that a crash keeps exit 1 to itself. If that ever stopped being true the
    cap's ok=True would be hiding real failures rather than disambiguating them.
    """
    capped_events: list[object] = []
    aborted_events: list[object] = []
    capped = run_fleet_supervise_loop(
        spawn=lambda _n: EXIT_RESTART_REQUESTED,
        max_relaunches=1,
        on_cap_reached=capped_events.append,
    )
    aborted = run_fleet_supervise_loop(
        spawn=lambda _n: 1, max_relaunches=1, on_cap_reached=aborted_events.append
    )

    assert (capped.ok, capped.data["cap_reached"]) == (True, True)
    assert (aborted.ok, aborted.data["cap_reached"]) == (False, False)
    assert (len(capped_events), len(aborted_events)) == (1, 0)


def test_run_fleet_supervise_loop_does_not_touch_the_real_state_file() -> None:
    """The cap callback must be injectable, not resolved from the live repo.

    Its default writes through ``orchestrator_root()`` to the real ``events.db``
    and ``state.json`` -- the ones the running supervisor owns. A cap test using
    defaults therefore injects fake events into production and contends for the
    live state lock. Asserting the parameter exists is what keeps the next cap
    test from quietly reaching production again.
    """
    import inspect

    signature = inspect.signature(run_fleet_supervise_loop)
    assert "on_cap_reached" in signature.parameters


def test_run_fleet_supervise_loop_propagates_a_child_failure() -> None:
    """An aborted supervisor stays non-ok rather than being masked by the wrapper."""
    result = run_fleet_supervise_loop(
        spawn=lambda _n: 1, max_relaunches=3, on_cap_reached=lambda _r: None
    )

    assert result.ok is False
    assert result.data["last_exit_code"] == 1
    assert result.data["cap_reached"] is False


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_logs_global_config_provenance(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The supervisor must record whether its global config layer was readable.

    This is the half of the #590 diagnostic that has to survive at the default
    log level: the loader's equivalent line is DEBUG, so on a real host this is
    the only place the fact appears. A successfully-loaded config that reports
    ``absent`` here is the silent-{} path in load_layered_config; one that
    reports ``present`` means the section was lost downstream of the read. The
    two demand opposite fixes, so neither reading may be missing.
    """
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_fleet_loop.return_value = _drained_fleet_result()

    def provenance_lines() -> str:
        return "\n".join(
            r.getMessage()
            for r in caplog.records
            if "Fleet supervisor global config" in r.getMessage()
        )

    # No global config on this fleet dir: reported as absent, at INFO.
    fc = _FakeClock(auto_advance=1.0)
    with caplog.at_level(logging.INFO, logger="charlie_work.fleet_dispatch"):
        run_fleet_supervise(
            max_passes=1, clock=fc.now, sleep=fc.sleep, fleet_dir_override=str(tmp_path)
        )
    absent = provenance_lines()
    assert absent, "the supervisor logged no global-config provenance at all"
    assert str(tmp_path / "config.yaml") in absent, "the path must be named"
    assert "absent" in absent, f"an absent global layer was not reported: {absent!r}"

    # Same call with the layer in place: distinguishable, with its size.
    caplog.clear()
    (tmp_path / "config.yaml").write_text("dispatch: {}\n", encoding="utf-8")
    fc = _FakeClock(auto_advance=1.0)
    with caplog.at_level(logging.INFO, logger="charlie_work.fleet_dispatch"):
        run_fleet_supervise(
            max_passes=1, clock=fc.now, sleep=fc.sleep, fleet_dir_override=str(tmp_path)
        )
    present = provenance_lines()
    assert "present" in present, f"a present global layer was not reported: {present!r}"
    assert "absent" not in present, "a present layer must not read as absent"
    assert "bytes=" in present, "the size distinguishes an empty layer from a populated one"


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_loud_on_absent_global_layer(
    mock_lock: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An absent global layer must be loud, not silent, in the supervisor.

    Before #623, ``load_layered_config`` treated an unreachable global fleet
    config as an empty mapping with no diagnostic, so a fleet supervisor whose
    global layer was missing silently ran on pristine dataclass defaults --
    ``runner_allocation`` off, ``notify`` off, ``labels`` back to built-ins --
    while passes kept reporting success. ``run_fleet_supervise`` now loads with
    ``require_global=True``, so the absent case raises ``ConfigError`` and the
    supervisor's existing handler catches it, warns, and prints -- the same
    loud path a malformed config already took.

    This test drives the *real* ``load_layered_config`` (not a mock) against an
    empty fleet dir so the ``require_global=True`` wiring is actually exercised
    end-to-end. The provenance line below the handler still fires because it
    uses ``describe_config_file`` directly, so the operator sees both the
    warning and the cause.
    """
    mock_fleet_loop.return_value = _drained_fleet_result()

    fc = _FakeClock(auto_advance=1.0)
    with caplog.at_level(logging.WARNING, logger="charlie_work.fleet_dispatch"):
        result = run_fleet_supervise(
            max_passes=1,
            clock=fc.now,
            sleep=fc.sleep,
            fleet_dir_override=str(tmp_path),
        )

    # The supervisor continues (the daemon must not crash on a missing global
    # layer) but it does so loudly: a WARNING was emitted naming the failure.
    warning_lines = [r.getMessage() for r in caplog.records if "could not load" in r.getMessage()]
    assert warning_lines, "an absent global layer must trigger the loud handler, not silence"
    assert "per-repo config only" in warning_lines[0], (
        f"the warning must name the fallback: {warning_lines[0]!r}"
    )
    # The ConfigError raised by require_global carries the path and the
    # describe_config_file cause, and the handler interpolates it into the
    # warning -- so the operator sees *why* the layer was unreadable, not just
    # that it was.
    assert str(tmp_path / "config.yaml") in warning_lines[0], (
        "the expected global config path must appear in the warning"
    )
    assert "absent" in warning_lines[0], (
        f"an absent layer must read as absent in the warning: {warning_lines[0]!r}"
    )

    # The handler also prints, so the failure is visible on stdout, not only in
    # the log.
    captured = capsys.readouterr()
    assert "config load failed" in captured.out, (
        "the absent-global failure must be printed, not only logged"
    )

    # The supervisor fell back to the per-repo config (NOT discarded to None or
    # pristine defaults) and still ran the pass -- the daemon stays alive, but
    # the operator has been told exactly why. Discarding the per-repo config
    # with the global layer would regress the #623 silent-disable failure.
    assert result.ok is True
    assert mock_fleet_loop.call_count == 1
    assert mock_fleet_loop.call_args.kwargs.get("global_config") is not None, (
        "fleet_loop must NOT receive global_config=None when the global layer "
        "is absent -- the per-repo config must survive the fallback, not be "
        "discarded with the global layer (#623 silent-disable regression)"
    )


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_respects_max_runtime(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
) -> None:
    """max_runtime_minutes stops the loop after the wall-clock cap expires."""
    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
            max_runtime_minutes=1,
        )
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _active_fleet_result()

    fc = _FakeClock(auto_advance=70.0)
    result = run_fleet_supervise(clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    assert result.data["passes"] == 1
    assert mock_fleet_loop.call_count == 1
    assert fc.sleep_calls == [7.0]
    # A runtime budget expiring is a deliberate stop, not a request to be
    # replaced -- it shares ok=True with the restarting exits, so this field is
    # the only thing keeping the wrapper from relaunching forever.
    assert result.data["exit_reason"] == "max_runtime"
    assert result.data["restart_requested"] is False


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_uses_active_cooldown_after_activity(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
) -> None:
    """After an active pass, sleep equals active_cooldown_seconds; idle equals poll_interval."""
    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.side_effect = [_active_fleet_result(), _drained_fleet_result()]

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=2, clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    assert result.data["passes"] == 2
    assert fc.sleep_calls == [7.0, 5.0]


@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_returns_false_when_lock_held(
    mock_lock: MagicMock,
) -> None:
    """A second concurrent invocation is rejected by the fleet supervisor lock."""
    mock_lock.return_value = None

    result = run_fleet_supervise()

    assert result.ok is False
    assert "fleet supervisor already running" in result.message


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_releases_lock_after_exception(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
) -> None:
    """The fleet supervisor lock is released even if fleet_loop raises."""
    lock = MagicMock()
    mock_lock.return_value = lock
    mock_load_config.return_value = OrchestratorConfig()
    mock_fleet_loop.side_effect = RuntimeError("boom")

    result = run_fleet_supervise(max_passes=3)

    assert result.ok is False
    assert "boom" in result.message
    lock.release.assert_called_once()


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_keyboard_interrupt_returns_ok(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
) -> None:
    """Ctrl+C is caught and reported as a clean completion."""
    lock = MagicMock()
    mock_lock.return_value = lock
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
        )
    )
    mock_fleet_loop.side_effect = [_drained_fleet_result(), KeyboardInterrupt]

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=5, clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    assert "fleet supervisor complete" in result.message
    assert result.data["passes"] >= 1


# ---------------------------------------------------------------------------
# Supervisor lifecycle wiring (issue #627)
# ---------------------------------------------------------------------------


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_supervisor_lifecycle_records_started_and_clean_exit(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
    _patch_self_deploy_for_fleet_tests: dict[str, MagicMock],
) -> None:
    """A run stopped by max_passes emits supervisor_started once and supervisor_exited with exit_code=0."""
    mocks = _patch_self_deploy_for_fleet_tests
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(poll_interval_seconds=5, full_pass_interval_seconds=1)
    )
    mock_fleet_loop.return_value = _drained_fleet_result()

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=2, clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    mocks["record_supervisor_started"].assert_called_once()
    start_kwargs = mocks["record_supervisor_started"].call_args.kwargs
    assert start_kwargs["max_pass_runtime_seconds"] == 1800
    mocks["record_supervisor_exit"].assert_called_once()
    exit_kwargs = mocks["record_supervisor_exit"].call_args.kwargs
    assert exit_kwargs["exit_code"] == 0
    assert exit_kwargs["reason"] == "max_passes"
    # Heartbeat refreshed once per loop iteration.
    assert mocks["update_supervisor_heartbeat"].call_count >= 2


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_supervisor_lifecycle_records_nonzero_exit_on_exception(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
    _patch_self_deploy_for_fleet_tests: dict[str, MagicMock],
) -> None:
    """An uncaught exception records supervisor_exited with exit_code=1."""
    mocks = _patch_self_deploy_for_fleet_tests
    mock_load_config.return_value = OrchestratorConfig()
    mock_fleet_loop.side_effect = RuntimeError("boom")

    result = run_fleet_supervise(max_passes=3)

    assert result.ok is False
    mocks["record_supervisor_exit"].assert_called_once()
    exit_kwargs = mocks["record_supervisor_exit"].call_args.kwargs
    assert exit_kwargs["exit_code"] == 1
    assert exit_kwargs["reason"] == "exception"


@patch("charlie_work.fleet_dispatch.probe_fleet_watchdog")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_supervisor_lifecycle_head_drift_exit_reason(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_probe: MagicMock,
    tmp_path: Path,
    _patch_self_deploy_for_fleet_tests: dict[str, MagicMock],
) -> None:
    """A HEAD-drift restart records reason=head_drift_restart with exit_code=0."""
    from charlie_work.fleet_dispatch import WatchdogProbe

    mocks = _patch_self_deploy_for_fleet_tests
    # Issue #604: the head-drift restart exit now probes the watchdog
    # scheduled task. Mock it to ``armed=None`` (unknown) so the test stays
    # hermetic -- no real ``schtasks`` subprocess call, no coupling to the
    # live state of the ``charlie-fleet-pass`` task -- and the alert path
    # (which fires only on a confirmed ``armed=False``) is not exercised
    # here. The dedicated watchdog-alert tests cover that path.
    mock_probe.return_value = WatchdogProbe(armed=None, detail="not probed (mocked)")
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(poll_interval_seconds=5, full_pass_interval_seconds=1)
    )
    mock_fleet_loop.return_value = _drained_fleet_result()
    # Force the external HEAD-drift branch by making read_head_sha diverge.
    with patch("charlie_work.fleet_dispatch.read_head_sha") as mock_head:
        mock_head.side_effect = ["aaa", "bbb"]  # startup_head, then current_head
        fc = _FakeClock(auto_advance=1.0)
        result = run_fleet_supervise(max_passes=5, clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    mocks["record_supervisor_exit"].assert_called_once()
    exit_kwargs = mocks["record_supervisor_exit"].call_args.kwargs
    assert exit_kwargs["exit_code"] == 0
    assert exit_kwargs["reason"] == "head_drift_restart"


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_supervisor_lifecycle_detects_and_records_prior_abnormal_exit(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
    _patch_self_deploy_for_fleet_tests: dict[str, MagicMock],
) -> None:
    """A prior abnormal exit is detected and recorded before the new supervisor starts."""
    mocks = _patch_self_deploy_for_fleet_tests
    mocks["detect_prior_abnormal_exit"].return_value = {
        "prior_pid": 4242,
        "prior_started_at": SUPERVISOR_STARTED_AT,
        "prior_last_beat_at": SUPERVISOR_BEAT_AT,
        "prior_pass_number": 9,
        "uptime_seconds": 2609.0,
    }
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(poll_interval_seconds=5, full_pass_interval_seconds=1)
    )
    mock_fleet_loop.return_value = _drained_fleet_result()

    fc = _FakeClock(auto_advance=1.0)
    run_fleet_supervise(max_passes=1, clock=fc.now, sleep=fc.sleep)

    mocks["detect_prior_abnormal_exit"].assert_called_once()
    mocks["record_prior_abnormal_exit"].assert_called_once()
    prior_arg = mocks["record_prior_abnormal_exit"].call_args.args[1]
    assert prior_arg["prior_pid"] == 4242
    # The new supervisor's own start is still recorded after the retroactive exit.
    mocks["record_supervisor_started"].assert_called_once()


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_supervisor_lifecycle_abnormal_exit_alerts_when_notify_enabled(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
    _patch_self_deploy_for_fleet_tests: dict[str, MagicMock],
) -> None:
    """An abnormal (non-zero) exit routes to the attention digest when notify is on."""
    mocks = _patch_self_deploy_for_fleet_tests
    mocks["is_exit_alertable"].return_value = True
    notify_config = MagicMock()
    notify_config.enabled = True
    mock_load_config.return_value = OrchestratorConfig(notify=notify_config)
    mock_fleet_loop.side_effect = RuntimeError("boom")

    with patch("charlie_work.fleet_dispatch._emit_fleet_transition") as mock_emit:
        run_fleet_supervise(max_passes=3)

    # The autouse fixture's self-deploy no-op also emits its own OK transition
    # unconditionally on every successful self-deploy (issue #817 fix, main-side
    # and unrelated to supervisor lifecycle), so two calls are expected here --
    # find the fleet-supervisor one specifically.
    fleet_supervisor_calls = [
        call
        for call in mock_emit.call_args_list
        if call.args[1].adapter_kind == "fleet-supervisor"
    ]
    assert len(fleet_supervisor_calls) == 1
    entry = fleet_supervisor_calls[0].args[1]
    assert entry.adapter_kind == "fleet-supervisor"
    assert entry.health == "ERROR"
    assert fleet_supervisor_calls[0].kwargs.get("persistent") is False


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_supervisor_lifecycle_clean_exit_does_not_alert(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
    _patch_self_deploy_for_fleet_tests: dict[str, MagicMock],
) -> None:
    """A clean (exit_code=0) exit never reaches the attention digest."""
    # is_exit_alertable defaults to False in the fixture.
    notify_config = MagicMock()
    notify_config.enabled = True
    mock_load_config.return_value = OrchestratorConfig(notify=notify_config)
    mock_fleet_loop.return_value = _drained_fleet_result()

    with patch("charlie_work.fleet_dispatch._emit_fleet_transition") as mock_emit:
        fc = _FakeClock(auto_advance=1.0)
        run_fleet_supervise(max_passes=1, clock=fc.now, sleep=fc.sleep)

    # The autouse fixture's self-deploy no-op emits its own OK transition
    # unconditionally on every successful self-deploy (issue #817 fix, main-side
    # and unrelated to supervisor lifecycle) -- that is expected. What this test
    # actually guards is that the clean supervisor exit itself does not
    # additionally alert.
    fleet_supervisor_calls = [
        call
        for call in mock_emit.call_args_list
        if call.args[1].adapter_kind == "fleet-supervisor"
    ]
    assert fleet_supervisor_calls == []


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_supervisor_lifecycle_does_not_record_when_lock_held(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    _patch_self_deploy_for_fleet_tests: dict[str, MagicMock],
) -> None:
    """A second invocation rejected by the lock records no lifecycle events."""
    mocks = _patch_self_deploy_for_fleet_tests
    mock_lock.return_value = None
    mock_load_config.return_value = OrchestratorConfig()

    result = run_fleet_supervise()

    assert result.ok is False
    mocks["record_supervisor_started"].assert_not_called()
    mocks["record_supervisor_exit"].assert_not_called()


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_supervisor_lifecycle_keyboard_interrupt_reason(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
    _patch_self_deploy_for_fleet_tests: dict[str, MagicMock],
) -> None:
    """Ctrl+C records reason=keyboard_interrupt with exit_code=0."""
    mocks = _patch_self_deploy_for_fleet_tests
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(poll_interval_seconds=5, full_pass_interval_seconds=1)
    )
    mock_fleet_loop.side_effect = [_drained_fleet_result(), KeyboardInterrupt]

    fc = _FakeClock(auto_advance=1.0)
    run_fleet_supervise(max_passes=5, clock=fc.now, sleep=fc.sleep)

    mocks["record_supervisor_exit"].assert_called_once()
    exit_kwargs = mocks["record_supervisor_exit"].call_args.kwargs
    assert exit_kwargs["exit_code"] == 0
    assert exit_kwargs["reason"] == "keyboard_interrupt"


@patch("charlie_work.fleet_dispatch.probe_fleet_watchdog")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_supervisor_lifecycle_self_deploy_head_move_reason(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_probe: MagicMock,
    tmp_path: Path,
    _patch_self_deploy_for_fleet_tests: dict[str, MagicMock],
) -> None:
    """A self-deploy HEAD move records reason=self_deploy_head_moved with exit_code=0."""
    from charlie_work.fleet_dispatch import WatchdogProbe

    mocks = _patch_self_deploy_for_fleet_tests
    # Issue #604: the self-deploy restart exit now probes the watchdog
    # scheduled task. Mock it to ``armed=None`` (unknown) so the test stays
    # hermetic -- no real ``schtasks`` subprocess call, no coupling to the
    # live state of the ``charlie-fleet-pass`` task -- and the alert path
    # (which fires only on a confirmed ``armed=False``) is not exercised
    # here. The dedicated watchdog-alert tests cover that path.
    mock_probe.return_value = WatchdogProbe(armed=None, detail="not probed (mocked)")
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(poll_interval_seconds=5, full_pass_interval_seconds=1)
    )
    mock_fleet_loop.return_value = _drained_fleet_result()
    # self_deploy is patched to a no-op by the autouse fixture; override it to
    # report a successful pull that moved HEAD.
    with patch(
        "charlie_work.fleet_dispatch.self_deploy",
        return_value=SelfDeployResult(
            ok=True,
            pulled=True,
            changed=True,
            synced=False,
            head_changed=True,
            from_sha="aaa",
            to_sha="bbb",
            message="moved",
        ),
    ):
        fc = _FakeClock(auto_advance=1.0)
        result = run_fleet_supervise(max_passes=5, clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    mocks["record_supervisor_exit"].assert_called_once()
    exit_kwargs = mocks["record_supervisor_exit"].call_args.kwargs
    assert exit_kwargs["exit_code"] == 0
    assert exit_kwargs["reason"] == "self_deploy_head_moved"


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_supervisor_lifecycle_prior_exit_alerts_when_notify_enabled(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
    _patch_self_deploy_for_fleet_tests: dict[str, MagicMock],
) -> None:
    """A detected prior abnormal exit routes to the attention digest when notify is on."""
    mocks = _patch_self_deploy_for_fleet_tests
    mocks["detect_prior_abnormal_exit"].return_value = {
        "prior_pid": 4242,
        "prior_started_at": SUPERVISOR_STARTED_AT,
        "prior_last_beat_at": SUPERVISOR_BEAT_AT,
        "prior_pass_number": 9,
        "uptime_seconds": 2609.0,
    }
    notify_config = MagicMock()
    notify_config.enabled = True
    mock_load_config.return_value = OrchestratorConfig(notify=notify_config)
    mock_fleet_loop.return_value = _drained_fleet_result()

    with patch("charlie_work.fleet_dispatch._emit_fleet_transition") as mock_emit:
        fc = _FakeClock(auto_advance=1.0)
        run_fleet_supervise(max_passes=1, clock=fc.now, sleep=fc.sleep)

    # The prior-exit ERROR transition is emitted.
    prior_emit = [
        call
        for call in mock_emit.call_args_list
        if call.args[1].last_log_line and "prior supervisor" in call.args[1].last_log_line
    ]
    assert prior_emit, "prior abnormal exit did not reach the attention digest"
    assert prior_emit[0].args[1].health == "ERROR"
    assert prior_emit[0].kwargs.get("persistent") is False


def test_supervisor_lifecycle_repeated_errors_are_not_deduped(tmp_path: Path) -> None:
    """Supervisor-kill alerts are occurrence-style: repeated kills must all fire."""
    notify_config = MagicMock()
    notify_config.enabled = True
    entry = AttentionEntry(
        issue_number=-1,
        adapter_kind="fleet-supervisor",
        health="ERROR",
        previous_health=None,
        last_log_line="prior supervisor terminated without an exit event",
        pid=4242,
    )

    fleet_dir = str(tmp_path / "fleet")
    with patch("charlie_work.fleet_dispatch.emit_digest") as mock_emit:
        _emit_fleet_transition(notify_config, entry, fleet_dir, persistent=False)
        _emit_fleet_transition(notify_config, entry, fleet_dir, persistent=False)

    assert mock_emit.call_count == 2


def test_supervisor_lifecycle_persistent_errors_are_deduped(tmp_path: Path) -> None:
    """Persistent health transitions still dedup by default."""
    notify_config = MagicMock()
    notify_config.enabled = True
    entry = AttentionEntry(
        issue_number=-1,
        adapter_kind="self-deploy",
        health="ERROR",
        previous_health=None,
        last_log_line="self-deploy failed",
        pid=None,
    )

    fleet_dir = str(tmp_path / "fleet-persist")
    with patch("charlie_work.fleet_dispatch.emit_digest") as mock_emit:
        _emit_fleet_transition(notify_config, entry, fleet_dir)
        _emit_fleet_transition(notify_config, entry, fleet_dir)

    assert mock_emit.call_count == 1


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_throttles_idle_passes(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Without a local delta, fleet_loop is skipped until the full-pass fallback expires."""
    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=100,
            max_runtime_minutes=1,
        )
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()

    snapshot = MagicMock()
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch._take_fleet_snapshot",
        lambda **_kwargs: snapshot,
    )
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch._has_fleet_delta",
        lambda _before, _after: False,
    )

    fc = _FakeClock(start=0.0, auto_advance=2.0)
    result = run_fleet_supervise(clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    assert mock_fleet_loop.call_count == 1  # only the initial fallback pass
    assert all(s == 5.0 for s in fc.sleep_calls)
    assert len(fc.sleep_calls) > 1


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_local_delta_triggers_pass_before_fallback(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A local signal delta triggers the next fleet pass before the fallback interval."""
    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=100,
        )
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()

    monkeypatch.setattr(
        "charlie_work.fleet_dispatch._take_fleet_snapshot",
        MagicMock(),
    )
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch._has_fleet_delta",
        MagicMock(side_effect=[False, True]),
    )

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=2, clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    assert result.data["passes"] == 2
    assert mock_fleet_loop.call_count == 2
    assert fc.sleep_calls == [5.0, 5.0]


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_full_pass_interval_fallback_triggers_pass(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """When no local delta, the full_pass_interval fallback still drives a pass."""
    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=10,
        )
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()

    monkeypatch.setattr(
        "charlie_work.fleet_dispatch._take_fleet_snapshot",
        MagicMock(),
    )
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch._has_fleet_delta",
        lambda _before, _after: False,
    )

    fc = _FakeClock(auto_advance=15.0)
    result = run_fleet_supervise(max_passes=2, clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    assert result.data["passes"] == 2
    assert mock_fleet_loop.call_count == 2
    assert fc.sleep_calls == [5.0, 5.0]


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_self_deploys_before_each_pass(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """The fleet supervisor calls self_deploy before every fleet_loop pass.

    ``from_sha == to_sha`` here deliberately means HEAD did not move on this
    pull (e.g. a pending dependency-sync marker with no new commit) --
    otherwise the supervisor's restart-for-fresh-code exit (see
    test_run_fleet_supervise_restarts_when_self_deploy_moves_head below)
    would legitimately break the loop after pass 1, since a running process
    never picks up newly-pulled source on its own.
    """
    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()

    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=True,
            pulled=True,
            changed=True,
            synced=False,
            from_sha="abc123",
            to_sha="abc123",
            message="already up to date",
        )
    )
    monkeypatch.setattr("charlie_work.fleet_dispatch.self_deploy", deploy_mock)

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=3, clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    assert result.data["passes"] == 3
    assert mock_fleet_loop.call_count == 3
    assert deploy_mock.call_count == 3


@patch("charlie_work.fleet_dispatch.probe_fleet_watchdog")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_restarts_when_self_deploy_moves_head(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_probe: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A pull that actually moves HEAD exits the loop instead of continuing.

    This process already imported every charlie_work module at startup;
    git changing files on disk underneath it does not hot-reload those
    modules. Left looping, the supervisor would keep running whatever code
    was live at process start for its entire max-runtime-0 lifetime,
    silently ignoring every fix merged to main afterward (observed
    2026-07-22: the daemon ran ~40 minutes on stale code after several
    fixes had already landed on main, because self_deploy's git pull only
    updates files on disk -- it never made the already-running process
    pick them up). Exiting here hands control back to the scheduled-task
    watchdog, which relaunches a fresh process with the new commit
    actually imported.
    """
    from charlie_work.fleet_dispatch import WatchdogProbe

    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()
    # Issue #604: the self-deploy restart exit now probes the watchdog
    # scheduled task. Mock it to ``armed=None`` (unknown) so the test stays
    # hermetic -- no real ``schtasks`` subprocess call, no coupling to the
    # live state of the ``charlie-fleet-pass`` task -- and the alert path
    # (which fires only on a confirmed ``armed=False``) is not exercised
    # here. The dedicated watchdog-alert tests cover that path.
    mock_probe.return_value = WatchdogProbe(armed=None, detail="not probed (mocked)")

    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=True,
            pulled=True,
            changed=True,
            synced=True,
            head_changed=True,
            from_sha="abc123",
            to_sha="def456",
            message="updated and synced: def456",
        )
    )
    monkeypatch.setattr("charlie_work.fleet_dispatch.self_deploy", deploy_mock)

    fc = _FakeClock(auto_advance=1.0)
    # max_passes=5 proves the exit is driven by the head-change detection,
    # not by exhausting the pass budget.
    result = run_fleet_supervise(max_passes=5, clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    assert result.data["passes"] == 1
    assert deploy_mock.call_count == 1
    # fleet_loop must never run this pass's (now-stale) code path.
    assert mock_fleet_loop.call_count == 0
    # #862: the exit must say *why*, so the launcher can relaunch immediately
    # instead of leaving the fleet unsupervised for a full watchdog interval.
    # ok=True alone is what made this indistinguishable from a clean timeout.
    assert result.data["exit_reason"] == "self_deploy"
    assert result.data["restart_requested"] is True


@patch("charlie_work.fleet_dispatch.probe_fleet_watchdog")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_zero_pass_bookkeeping_failure_cannot_cancel_a_self_deploy_restart(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_probe: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A failure in post-loop bookkeeping must not suppress the restart signal.

    ``record_zero_pass_streak`` runs after the loop and does real file I/O
    (mkdir, state_lock, log_event); its own docstring says it can raise. It used
    to sit bare inside the outer ``try``, whose handler rewrote ``exit_reason``
    to ``aborted`` and ``restart_requested`` to False unconditionally. So a
    self-deploy that pulled new code, followed by a counter write failing on a
    locked state file, produced an exit the wrapper read as "do not relaunch" --
    the #862 outage, reachable through a secondary failure that has nothing to
    do with whether new code is on disk.

    The important assertion is ``restart_requested``, not ``ok``: the run really
    did fail, so ok=False is correct. What must survive is the instruction to
    replace this process.
    """
    from charlie_work.fleet_dispatch import WatchdogProbe

    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()
    # Issue #604: mock the watchdog probe so this test does not perform a real
    # ``schtasks`` subprocess call. ``armed=None`` (unknown) does not trigger
    # the alert path, keeping the test focused on the bookkeeping-failure
    # invariant it exists to guard.
    mock_probe.return_value = WatchdogProbe(armed=None, detail="not probed (mocked)")

    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=True,
            pulled=True,
            changed=True,
            synced=True,
            head_changed=True,
            from_sha="abc123",
            to_sha="def456",
            message="updated and synced: def456",
        )
    )
    monkeypatch.setattr("charlie_work.fleet_dispatch.self_deploy", deploy_mock)

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("state file locked by another process")

    monkeypatch.setattr("charlie_work.fleet_dispatch.record_zero_pass_streak", _boom)

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=5, clock=fc.now, sleep=fc.sleep)

    assert result.data["exit_reason"] == "self_deploy"
    assert result.data["restart_requested"] is True


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_an_operator_interrupt_never_asks_to_be_relaunched(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Ctrl-C means stop, and a wrapper that relaunched would defeat that.

    ``interrupted`` is a named reason rather than an unset default precisely so
    this intent is stated and testable. Nothing verified it when the vocabulary
    was introduced, which left the one exit an operator triggers by hand relying
    on ``None`` happening to fall outside ``RESTART_EXIT_REASONS``.
    """
    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.side_effect = KeyboardInterrupt()

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=5, clock=fc.now, sleep=fc.sleep)

    assert result.data["exit_reason"] == "interrupted"
    assert result.data["restart_requested"] is False


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_a_mid_loop_crash_reports_aborted_and_does_not_relaunch(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A crash with no restarting reason already set stays non-restarting.

    The control for
    ``test_zero_pass_bookkeeping_failure_cannot_cancel_a_self_deploy_restart``:
    that test proves an already-set reason survives the handler, and this one
    proves the handler did not simply start relaunching on every exception.
    """
    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.side_effect = RuntimeError("boom")

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=5, clock=fc.now, sleep=fc.sleep)

    assert result.ok is False
    assert result.data["exit_reason"] == "aborted"
    assert result.data["restart_requested"] is False


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_does_not_restart_when_already_up_to_date(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """No new commits (from_sha == to_sha) must not trigger a restart-exit."""
    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()

    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=True,
            pulled=True,
            changed=False,
            synced=False,
            from_sha="abc123",
            to_sha="abc123",
            message="already up to date",
        )
    )
    monkeypatch.setattr("charlie_work.fleet_dispatch.self_deploy", deploy_mock)

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=3, clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    assert result.data["passes"] == 3
    assert mock_fleet_loop.call_count == 3


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_does_not_restart_on_deferred_sync_with_unmoved_head(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Regression guard for the total-fleet-outage bug (issue root cause).

    self_deploy reports a differing ``from_sha``/``to_sha`` pair even though
    HEAD did not move on *this* attempt, because those shas are carried
    forward from an earlier deferred-sync marker (see
    ``test_self_deploy_loud_warning_on_repeated_deferral`` in
    test_supervise.py for the producer side of this exact scenario). Gating
    the restart-exit on ``from_sha != to_sha`` instead of ``head_changed``
    made the supervisor exit and relaunch every single pass without ever
    reaching zero live workers to complete the deferred sync -- a total
    fleet outage. ``head_changed=False`` here must keep the loop running.
    """
    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()

    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=True,
            pulled=True,
            changed=True,
            synced=False,
            head_changed=False,
            from_sha="abc123",
            to_sha="def456",
            message="sync deferred: 2 runners active",
        )
    )
    monkeypatch.setattr("charlie_work.fleet_dispatch.self_deploy", deploy_mock)

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=3, clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    assert result.data["passes"] == 3
    assert deploy_mock.call_count == 3
    # The pending-sync marker's from_sha != to_sha must not trigger a
    # restart-exit when head_changed is False -- the loop must keep running
    # so live-worker draining can eventually reach zero and complete the
    # deferred sync.
    assert mock_fleet_loop.call_count == 3


@patch("charlie_work.fleet_dispatch.probe_fleet_watchdog")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_restarts_on_external_head_drift(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_probe: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """HEAD moved externally (operator pull, another process) triggers restart.

    self_deploy reports "already up to date" because HEAD was already at the
    new commit when the daemon's own git pull ran. Without an independent
    startup-vs-current HEAD comparison, the daemon would run stale code
    forever (observed 2026-07-23: ~90 minutes of ConfigError crashes).
    """
    from charlie_work.fleet_dispatch import WatchdogProbe

    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()
    # Issue #604: the head-drift restart exit now probes the watchdog
    # scheduled task. Mock it to ``armed=None`` (unknown) so the test does
    # not perform a real ``schtasks`` subprocess call and stays hermetic --
    # the same pattern used by the self_deploy/read_head_sha/fleet_loop
    # mocks already applied in this test. The alert path (armed=False) is
    # covered by the dedicated watchdog-alert tests below.
    mock_probe.return_value = WatchdogProbe(armed=None, detail="not probed (mocked)")

    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=True,
            pulled=True,
            changed=False,
            synced=False,
            from_sha="def456",
            to_sha="def456",
            message="already up to date",
        )
    )
    monkeypatch.setattr("charlie_work.fleet_dispatch.self_deploy", deploy_mock)

    # Simulate: startup HEAD is "abc123", then an external actor moved HEAD
    # to "def456" before the first pass. self_deploy sees "already up to date"
    # because its own pull didn't move anything, but the drift check catches it.
    sha_sequence = iter(["abc123", "def456", "def456"])
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.read_head_sha",
        lambda _root: next(sha_sequence),
    )

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=5, clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    assert result.data["passes"] == 1
    assert deploy_mock.call_count == 1
    # fleet_loop must never run with stale code.
    assert mock_fleet_loop.call_count == 0
    # head_drift is the OTHER half of the restart contract (RESTART_EXIT_REASONS
    # holds exactly self_deploy and head_drift). Only self_deploy was asserted
    # when the field was introduced, so an edit dropping the reason here would
    # have left drift silently non-restarting with every test still green.
    assert result.data["exit_reason"] == "head_drift"
    assert result.data["restart_requested"] is True


# ---------------------------------------------------------------------------
# Issue #604: a restart-requesting exit must verify the watchdog scheduled task
# is armed, and alert through a non-log channel when it is not. A disabled
# watchdog turns a clean, well-reported drift/self-deploy exit into a silent
# indefinite fleet outage (the 2026-07-25 incident).
# ---------------------------------------------------------------------------


def test_probe_fleet_watchdog_parses_enabled_state() -> None:
    """probe_fleet_watchdog maps the 'Scheduled Task State' field to armed."""
    from charlie_work.fleet_dispatch import probe_fleet_watchdog

    enabled_output = (
        "TaskName:                             \\charlie-fleet-pass\n"
        "Status:                               Ready\n"
        "Scheduled Task State:                 Enabled\n"
        "Last Result:                          0\n"
    )
    disabled_output = (
        "TaskName:                             \\charlie-fleet-pass\n"
        "Status:                               Disabled\n"
        "Scheduled Task State:                 Disabled\n"
    )

    calls = iter([enabled_output, disabled_output])

    def _runner_seq(command, *, cwd, timeout_seconds):
        return RunResult(returncode=0, stdout=next(calls), stderr="")

    with patch("charlie_work.fleet_dispatch.sys") as mock_sys:
        mock_sys.platform = "win32"
        probe = probe_fleet_watchdog(run_command=_runner_seq)
    assert probe.armed is True
    assert "Enabled" in probe.detail

    with patch("charlie_work.fleet_dispatch.sys") as mock_sys:
        mock_sys.platform = "win32"
        probe = probe_fleet_watchdog(run_command=_runner_seq)
    assert probe.armed is False
    assert "Disabled" in probe.detail


def test_probe_fleet_watchdog_unknown_on_missing_field_and_non_windows() -> None:
    """An unparseable or non-Windows probe degrades to armed=None, never False."""
    from charlie_work.fleet_dispatch import probe_fleet_watchdog

    # No 'Scheduled Task State' line -> cannot determine.
    with patch("charlie_work.fleet_dispatch.sys") as mock_sys:
        mock_sys.platform = "win32"
        probe = probe_fleet_watchdog(
            run_command=lambda command, *, cwd, timeout_seconds: RunResult(
                returncode=0, stdout="TaskName: \\charlie-fleet-pass\n", stderr=""
            )
        )
    assert probe.armed is None

    # schtasks query failure (task missing) -> unknown, not False.
    with patch("charlie_work.fleet_dispatch.sys") as mock_sys:
        mock_sys.platform = "win32"
        probe = probe_fleet_watchdog(
            run_command=lambda command, *, cwd, timeout_seconds: RunResult(
                returncode=1, stdout="", stderr="task not found", error="command exited 1"
            )
        )
    assert probe.armed is None

    # Non-Windows -> not probed, unknown.
    with patch("charlie_work.fleet_dispatch.sys") as mock_sys:
        mock_sys.platform = "linux"
        probe = probe_fleet_watchdog()
    assert probe.armed is None


@patch("charlie_work.fleet_dispatch.probe_fleet_watchdog")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_alerts_when_watchdog_disabled_on_head_drift(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_probe: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A HEAD-drift restart with the watchdog disabled alerts + records an event.

    This is the #604 regression: on 2026-07-25 the drift exit fired correctly
    but the ``charlie-fleet-pass`` task was ``Enabled=false``, so no relaunch
    came and the fleet went dark silently. The exit must now surface the
    disarmed watchdog through a channel that is not the launcher log.
    """
    from charlie_work.config import NotifyConfig
    from charlie_work.fleet_dispatch import WatchdogProbe

    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5, full_pass_interval_seconds=1, active_cooldown_seconds=7
        ),
        notify=NotifyConfig(enabled=True, sink="file"),
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.self_deploy",
        lambda _repo_root, **_kwargs: SelfDeployResult(
            ok=True,
            pulled=True,
            changed=False,
            synced=False,
            from_sha="def456",
            to_sha="def456",
            message="already up to date",
        ),
    )
    sha_sequence = iter(["abc123", "def456", "def456"])
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.read_head_sha", lambda _root: next(sha_sequence)
    )
    mock_probe.return_value = WatchdogProbe(
        armed=False, detail="task 'charlie-fleet-pass' is Disabled"
    )

    fleet_dir = tmp_path / "fleet"
    with patch("charlie_work.fleet_dispatch._emit_fleet_transition") as mock_emit:
        fc = _FakeClock(auto_advance=1.0)
        result = run_fleet_supervise(
            max_passes=5, clock=fc.now, sleep=fc.sleep, fleet_dir_override=str(fleet_dir)
        )

    assert result.data["exit_reason"] == "head_drift"
    assert result.data["restart_requested"] is True
    mock_probe.assert_called_once()

    # The alert reached the attention digest (a non-log channel).
    watchdog_calls = [
        call for call in mock_emit.call_args_list if call.args[1].adapter_kind == "fleet-watchdog"
    ]
    assert len(watchdog_calls) == 1
    entry = watchdog_calls[0].args[1]
    assert entry.health == "ERROR"
    assert watchdog_calls[0].kwargs.get("persistent") is False
    assert "disabled" in entry.last_log_line

    # And it was durably recorded to the fleet events.db.
    from charlie_work.supervisor_lifecycle import supervisor_heartbeat_path

    events = query_events(
        supervisor_heartbeat_path(str(fleet_dir)), kind="supervisor_restart_watchdog_disabled"
    )
    assert len(events) == 1
    assert events[0]["payload"]["exit_reason"] == "head_drift"


@patch("charlie_work.fleet_dispatch.probe_fleet_watchdog")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_no_alert_when_watchdog_armed_or_unknown(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_probe: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """An armed or unknown watchdog must not false-alarm on a restart exit.

    ``armed=None`` (non-Windows, schtasks missing, task not found) is not
    proof the watchdog is disarmed, so it must stay quiet -- otherwise every
    non-Windows restart would page.
    """
    from charlie_work.config import NotifyConfig
    from charlie_work.fleet_dispatch import WatchdogProbe

    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5, full_pass_interval_seconds=1, active_cooldown_seconds=7
        ),
        notify=NotifyConfig(enabled=True, sink="file"),
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.self_deploy",
        lambda _repo_root, **_kwargs: SelfDeployResult(
            ok=True, pulled=True, changed=False, synced=False, message="already up to date"
        ),
    )

    for armed, label in ((True, "armed"), (None, "unknown")):
        sha_sequence = iter(["abc123", "def456", "def456"])
        monkeypatch.setattr(
            "charlie_work.fleet_dispatch.read_head_sha", lambda _root: next(sha_sequence)
        )
        mock_probe.return_value = WatchdogProbe(armed=armed, detail=f"task state {label}")
        with patch("charlie_work.fleet_dispatch._emit_fleet_transition") as mock_emit:
            fc = _FakeClock(auto_advance=1.0)
            run_fleet_supervise(
                max_passes=5,
                clock=fc.now,
                sleep=fc.sleep,
                fleet_dir_override=str(tmp_path / "fleet"),
            )
        watchdog_calls = [
            call
            for call in mock_emit.call_args_list
            if call.args[1].adapter_kind == "fleet-watchdog"
        ]
        assert watchdog_calls == [], f"unexpected watchdog alert when {label}"


@patch("charlie_work.fleet_dispatch.probe_fleet_watchdog")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_alerts_when_watchdog_disabled_on_self_deploy(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_probe: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """The self-deploy head-moved restart site also alerts when the watchdog is off.

    Both members of RESTART_EXIT_REASONS (self_deploy, head_drift) carry the
    same watchdog dependency; the verification must not be wired at only one
    of the two break sites.
    """
    from charlie_work.config import NotifyConfig
    from charlie_work.fleet_dispatch import WatchdogProbe

    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5, full_pass_interval_seconds=1, active_cooldown_seconds=7
        ),
        notify=NotifyConfig(enabled=True, sink="file"),
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.self_deploy",
        lambda _repo_root, **_kwargs: SelfDeployResult(
            ok=True,
            pulled=True,
            changed=True,
            synced=False,
            head_changed=True,
            from_sha="aaa111",
            to_sha="bbb222",
            message="fast-forwarded",
        ),
    )
    # startup_head == current_head so the drift branch does not also fire.
    monkeypatch.setattr("charlie_work.fleet_dispatch.read_head_sha", lambda _root: "bbb222")
    mock_probe.return_value = WatchdogProbe(
        armed=False, detail="task 'charlie-fleet-pass' is Disabled"
    )

    fleet_dir = tmp_path / "fleet"
    with patch("charlie_work.fleet_dispatch._emit_fleet_transition") as mock_emit:
        fc = _FakeClock(auto_advance=1.0)
        result = run_fleet_supervise(
            max_passes=5, clock=fc.now, sleep=fc.sleep, fleet_dir_override=str(fleet_dir)
        )

    assert result.data["exit_reason"] == "self_deploy"
    assert result.data["restart_requested"] is True
    mock_probe.assert_called_once()
    watchdog_calls = [
        call for call in mock_emit.call_args_list if call.args[1].adapter_kind == "fleet-watchdog"
    ]
    assert len(watchdog_calls) == 1
    assert watchdog_calls[0].args[1].health == "ERROR"
    assert watchdog_calls[0].kwargs.get("persistent") is False
    assert "disabled" in watchdog_calls[0].args[1].last_log_line
    # Durably recorded to the fleet events.db with the self_deploy reason.
    from charlie_work.supervisor_lifecycle import supervisor_heartbeat_path

    sd_events = query_events(
        supervisor_heartbeat_path(str(fleet_dir)), kind="supervisor_restart_watchdog_disabled"
    )
    assert len(sd_events) == 1
    assert sd_events[0]["payload"]["exit_reason"] == "self_deploy"


@patch("charlie_work.fleet_dispatch.emit_digest")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_emits_attention_digest_on_venv_repaired(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_emit_digest: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A successful self_deploy venv repair emits an attention digest so it is never silent."""
    from charlie_work.config import NotifyConfig

    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        ),
        notify=NotifyConfig(
            enabled=True,
            sink="file",
            file_path=str(tmp_path / "digest.jsonl"),
        ),
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()
    lock = MagicMock()
    mock_lock.return_value = lock

    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=True,
            pulled=False,
            changed=False,
            synced=False,
            venv_repaired=True,
            message="venv editable target repaired: shared venv editable .pth points to main checkout src",
        )
    )
    monkeypatch.setattr("charlie_work.fleet_dispatch.self_deploy", deploy_mock)

    result = run_fleet_supervise(max_passes=1, fleet_dir_override=str(tmp_path / "fleet"))

    assert result.ok is True
    assert mock_fleet_loop.call_count == 1
    assert deploy_mock.call_count == 1
    assert mock_emit_digest.called is True
    digest = mock_emit_digest.call_args[0][1]
    assert digest.repo == "fleet"
    assert len(digest.transitions) == 1
    assert digest.transitions[0].issue_number == -1
    assert digest.transitions[0].adapter_kind == "self-deploy"
    assert digest.transitions[0].health == "REPAIRED"
    assert "venv editable target repaired" in digest.transitions[0].last_log_line


@patch("charlie_work.fleet_dispatch.emit_digest")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_emits_attention_digest_on_repair_failure(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_emit_digest: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A failed self_deploy repair emits an attention digest so it is never silent."""
    from charlie_work.config import NotifyConfig

    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        ),
        notify=NotifyConfig(
            enabled=True,
            sink="file",
            file_path=str(tmp_path / "digest.jsonl"),
        ),
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()
    lock = MagicMock()
    mock_lock.return_value = lock

    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=False,
            pulled=False,
            changed=False,
            synced=False,
            error="venv pth repair failed: Access is denied",
        )
    )
    monkeypatch.setattr("charlie_work.fleet_dispatch.self_deploy", deploy_mock)

    result = run_fleet_supervise(max_passes=1, fleet_dir_override=str(tmp_path / "fleet"))

    assert result.ok is True
    assert mock_fleet_loop.call_count == 1
    assert deploy_mock.call_count == 1
    assert mock_emit_digest.called is True
    digest = mock_emit_digest.call_args[0][1]
    assert digest.repo == "fleet"
    assert len(digest.transitions) == 1
    assert digest.transitions[0].issue_number == -1
    assert digest.transitions[0].adapter_kind == "self-deploy"
    assert digest.transitions[0].health == "ERROR"
    assert "Access is denied" in digest.transitions[0].last_log_line


def test_extract_attention_events_includes_live_worker_redispatch_averted() -> None:
    """Issue #506: fleet attention digest surfaces live-worker redispatch averted outcomes."""
    from charlie_work.fleet_dispatch import _build_fleet_attention_digest

    result = CommandResult(
        True,
        "dispatch complete",
        {
            "dispatch": {
                "live_worker_redispatch_averted": [
                    {
                        "issue_number": 1317,
                        "branch_name": "agent/issue-1317-fix-search",
                        "pid": 12345,
                        "probe_result": "pid_alive",
                        "adapter_kind": "devin-shell",
                    }
                ]
            }
        },
    )

    events = _extract_attention_events("owner/repo", result)
    assert len(events) == 1
    assert events[0]["type"] == "live_worker_redispatch_averted"
    assert events[0]["issue_number"] == 1317
    assert events[0]["reason"] == "pid_alive"
    assert events[0]["adapter_kind"] == "devin-shell"

    digest = _build_fleet_attention_digest(events)
    assert len(digest.transitions) == 1
    entry = digest.transitions[0]
    assert entry.issue_number == 1317
    assert entry.health == "DISPATCH_AVERTED"
    assert entry.last_log_line == "pid_alive"
    assert entry.adapter_kind == "devin-shell"


def test_extract_attention_events_review_verdicts() -> None:
    """Issue #507: recorded/missed review verdicts surface in fleet attention events."""
    result = CommandResult(
        True,
        "review dispatch: 0 launched, 0 failed; 1 verdict(s) recorded, 1 missed",
        {
            "stalled": [],
            "errors": [],
            "recorded_verdicts": [{"pr": 100, "issue": 10, "decision": "approved"}],
            "missed_verdicts": [{"pr": 101, "issue": 11, "reason": "no parseable verdict"}],
        },
    )

    events = _extract_attention_events("owner/repo1", result)

    recorded = [e for e in events if e["type"] == "review_verdict_recorded"]
    missed = [e for e in events if e["type"] == "review_verdict_missed"]
    assert len(recorded) == 1
    assert recorded[0]["pr"] == 100
    assert recorded[0]["issue_number"] == 10
    assert recorded[0]["decision"] == "approved"
    assert len(missed) == 1
    assert missed[0]["pr"] == 101
    assert missed[0]["reason"] == "no parseable verdict"


def test_extract_attention_events_nested_review_verdicts() -> None:
    """Issue #507: review verdict events in nested dispatch_reviews sub-results are extracted."""
    result = CommandResult(
        True,
        "loop complete",
        {
            "stalled": [],
            "errors": [],
            "dispatch_reviews": {
                "recorded_verdicts": [{"pr": 200, "issue": 20, "decision": "request_changes"}],
                "missed_verdicts": [],
            },
        },
    )

    events = _extract_attention_events("owner/repo1", result)

    assert len(events) == 1
    assert events[0]["type"] == "review_verdict_recorded"
    assert events[0]["pr"] == 200
    assert events[0]["issue_number"] == 20


def test_build_fleet_attention_digest_maps_review_verdict_events() -> None:
    """Issue #507: review verdict events map to OK/ERROR attention entries."""
    events = [
        {
            "repo_key": "owner/repo1",
            "type": "review_verdict_recorded",
            "issue_number": 10,
            "pr": 100,
            "decision": "approved",
        },
        {
            "repo_key": "owner/repo1",
            "type": "review_verdict_missed",
            "issue_number": 11,
            "pr": 101,
            "reason": "no parseable verdict",
        },
    ]

    digest = _build_fleet_attention_digest(events)

    by_health = {e.health: e for e in digest.transitions}
    assert by_health["OK"].last_log_line == "approved recorded for PR 100"
    assert by_health["OK"].issue_number == 10
    assert by_health["ERROR"].last_log_line == "no parseable verdict"
    assert by_health["ERROR"].issue_number == 11


def _repair_payload(
    issue_numbers: list[int] | None = None,
    failures: list[int] | None = None,
    errored: list[int] | None = None,
    deferred: int = 0,
) -> dict[str, Any]:
    """Build a payload shaped like ``OrchestratorApp._repair_escalated_labels()``.

    Mirrors the real return shape (``workflow.py``'s ``_repair_escalated_labels``):
    ``issue_numbers`` is every subject whose ``transition()`` ran (successes and
    failures both), ``failures`` is the subset that did not fully apply, ``errored``
    is disjoint from both (nothing was written for those), and ``deferred`` is a
    plain count. Kept realistic rather than minimal so these tests exercise the
    same key combinations production actually emits.
    """
    return {
        "issue_numbers": issue_numbers or [],
        "failures": failures or [],
        "errored": errored or [],
        "deferred": deferred,
    }


def test_extract_attention_events_escalated_label_repair_errored() -> None:
    """Issue #1088: a subject whose GitHub call raised must surface in the digest.

    ``errored`` is the only durable record that an escalated-label repair was
    attempted and failed to reach GitHub -- state.json gets nothing written for
    it, so events.db and this digest are the sole places an operator could ever
    learn about it. This also serves as the positive control for the
    success/deferred/steady-state tests below: it proves the collector CAN
    produce an event from this payload shape before those tests assert it does
    not.
    """
    result = CommandResult(
        True,
        "review dispatch disabled",
        {"escalated_labels_repaired": _repair_payload(errored=[501, 502])},
    )

    events = _extract_attention_events("owner/repo", result)

    repair_events = [e for e in events if e["type"] == "escalated_label_repair_error"]
    assert len(repair_events) == 1
    assert repair_events[0]["issue_number"] == 501
    assert repair_events[0]["repo_key"] == "owner/repo"


def test_extract_attention_events_escalated_label_repair_failures() -> None:
    """Issue #1088: a subject whose transition() ran but did not fully apply.

    ``failures`` (label add/remove partially rejected by GitHub) is a distinct
    operational state from ``errored`` (nothing written) -- both are things a
    human eventually has to look at, so both must produce an entry. ``errored``
    is empty here to isolate that ``failures`` alone is sufficient.
    """
    result = CommandResult(
        True,
        "review dispatch disabled",
        {"escalated_labels_repaired": _repair_payload(issue_numbers=[601], failures=[601])},
    )

    events = _extract_attention_events("owner/repo", result)

    repair_events = [e for e in events if e["type"] == "escalated_label_repair_error"]
    assert len(repair_events) == 1
    assert repair_events[0]["issue_number"] == 601


def test_extract_attention_events_escalated_label_repair_mixed_prefers_errored_anchor() -> None:
    """Issue #1088: with both `errored` and `failures` populated, `errored` anchors.

    A single real pass can produce both: one subject's ``issue_view``/``transition()``
    call raises (-> ``errored``) while a *different* subject's ``transition()``
    completes but doesn't fully apply (-> ``failures``). Every other test in this
    file exercises a payload where one of the two lists is empty, so the collector's
    ``(errored or failures)[0]`` choice is never actually exercised elsewhere --
    it degrades to "return the only non-empty list" and would pass just as well
    under a reversed `(failures or errored)[0]`. This pins the real tie-break: the
    unreachable subject (nothing durable in state.json, so this digest is its only
    record) anchors the entry over the diagnosable one (already recorded via
    `label_error` in state.json).
    """
    result = CommandResult(
        True,
        "review dispatch disabled",
        {
            "escalated_labels_repaired": _repair_payload(
                issue_numbers=[602], failures=[602], errored=[501]
            )
        },
    )

    events = _extract_attention_events("owner/repo", result)

    repair_events = [e for e in events if e["type"] == "escalated_label_repair_error"]
    assert len(repair_events) == 1
    assert repair_events[0]["issue_number"] == 501
    assert "1 unreachable" in repair_events[0]["error"]
    assert "1 not applied" in repair_events[0]["error"]


def test_extract_attention_events_escalated_label_repair_success_silent() -> None:
    """Issue #1088: a fully successful repair must NOT produce an attention event.

    Deliberate, per the collector's docstring: a self-healed success is not
    something needing attention, and it is already durable in events.db via
    the ``escalated_label_repaired`` state event. Flooding the digest with a
    healthy sweep's output would bury the ``errored``/``failures`` signal this
    whole feature exists to surface.
    """
    # Positive control: the same call shape, but with `errored` populated,
    # must produce an event -- otherwise the "no event" assertion below is
    # equally consistent with a broken test harness as with correct behavior.
    control_result = CommandResult(
        True,
        "review dispatch disabled",
        {"escalated_labels_repaired": _repair_payload(errored=[701])},
    )
    control_events = _extract_attention_events("owner/repo", control_result)
    assert len([e for e in control_events if e["type"] == "escalated_label_repair_error"]) == 1

    result = CommandResult(
        True,
        "review dispatch disabled",
        {"escalated_labels_repaired": _repair_payload(issue_numbers=[701])},
    )
    events = _extract_attention_events("owner/repo", result)

    repair_events = [e for e in events if e["type"] == "escalated_label_repair_error"]
    assert repair_events == []


def test_extract_attention_events_escalated_label_repair_deferred_silent() -> None:
    """Issue #1088: subjects held back by the per-pass cap must NOT produce an event.

    ``deferred`` counts subjects beyond ``escalated_label_repair_max_per_pass``;
    the sweep converges over subsequent passes by design, so this is normal
    steady-state progress, not a fault worth an operator's attention.
    """
    # Positive control -- same shape, `errored` populated, must fire.
    control_result = CommandResult(
        True,
        "review dispatch disabled",
        {"escalated_labels_repaired": _repair_payload(errored=[801])},
    )
    control_events = _extract_attention_events("owner/repo", control_result)
    assert len([e for e in control_events if e["type"] == "escalated_label_repair_error"]) == 1

    result = CommandResult(
        True,
        "review dispatch disabled",
        {"escalated_labels_repaired": _repair_payload(deferred=3)},
    )
    events = _extract_attention_events("owner/repo", result)

    repair_events = [e for e in events if e["type"] == "escalated_label_repair_error"]
    assert repair_events == []


def test_extract_attention_events_escalated_label_repair_steady_state_silent() -> None:
    """Issue #1088: the idle steady state (nothing to repair at all) is silent.

    This is the ``empty`` sentinel ``_repair_escalated_labels`` returns when
    there were no escalated subjects needing repair -- the common case on a
    healthy fleet. It must not manufacture a digest entry every single pass.
    """
    # Positive control -- same shape, `errored` populated, must fire.
    control_result = CommandResult(
        True,
        "review dispatch disabled",
        {"escalated_labels_repaired": _repair_payload(errored=[901])},
    )
    control_events = _extract_attention_events("owner/repo", control_result)
    assert len([e for e in control_events if e["type"] == "escalated_label_repair_error"]) == 1

    result = CommandResult(
        True,
        "review dispatch disabled",
        {"escalated_labels_repaired": _repair_payload()},
    )
    events = _extract_attention_events("owner/repo", result)

    repair_events = [e for e in events if e["type"] == "escalated_label_repair_error"]
    assert repair_events == []


def test_extract_attention_events_nested_escalated_label_repair() -> None:
    """Issue #1088: the key nested under ``dispatch_reviews`` (as loop() nests it).

    ``dispatch_reviews()``'s own CommandResult carries ``escalated_labels_repaired``
    at its top level; ``loop()`` nests that whole dict under a ``dispatch_reviews``
    key in its own result. Mirrors
    ``test_extract_attention_events_nested_review_verdicts`` -- if only the
    top-level form were checked, every deployed fleet pass (which goes through
    ``loop()``) would never surface this event at all.
    """
    result = CommandResult(
        True,
        "loop complete",
        {
            "stalled": [],
            "errors": [],
            "dispatch_reviews": {
                "escalated_labels_repaired": _repair_payload(errored=[801]),
            },
        },
    )

    events = _extract_attention_events("owner/repo1", result)

    repair_events = [e for e in events if e["type"] == "escalated_label_repair_error"]
    assert len(repair_events) == 1
    assert repair_events[0]["issue_number"] == 801


def test_build_fleet_attention_digest_maps_escalated_label_repair_error() -> None:
    """Issue #1088: the event survives the full digest-rendering pipeline as ERROR.

    This is the property that matters most: it is not enough for the collector
    to emit the right dict, since ``_build_fleet_attention_digest`` has an
    explicit branch per event type and a generic fallback for anything else.
    Issue #590 made that fallback render (instead of silently dropping)
    unbranched types, keyed off an ``_error``-suffixed type name mapping to
    ``health="ERROR"``. This test pins that ``escalated_label_repair_error``
    actually rides that fallback through to a real ``AttentionEntry`` end to
    end, rather than trusting the type-name convention by inspection.
    """
    result = CommandResult(
        True,
        "review dispatch disabled",
        {"escalated_labels_repaired": _repair_payload(errored=[901])},
    )
    events = _extract_attention_events("owner/repo", result)

    digest = _build_fleet_attention_digest(events)

    repair_entries = [e for e in digest.transitions if e.issue_number == 901]
    assert len(repair_entries) == 1
    entry = repair_entries[0]
    assert entry.health == "ERROR"
    assert entry.adapter_kind == "owner/repo"
    assert "901" in (entry.last_log_line or "")


def test_extract_attention_events_escalated_label_repair_malformed_input() -> None:
    """Issue #1088: malformed ``escalated_labels_repaired`` shapes must not raise.

    This payload comes from a per-repo ``CommandResult.data`` that ultimately
    traces back to another process's JSON. A shape drift there (e.g. a future
    refactor that changes ``errored`` from a list to a dict, or the whole key
    to a bool) must degrade to "no event" in the fleet digest builder, not
    crash the whole attention-extraction pass for every other repo in the
    fleet loop.
    """
    not_a_dict_result = CommandResult(
        True,
        "review dispatch disabled",
        {"escalated_labels_repaired": "not-a-dict"},
    )
    events_not_dict = _extract_attention_events("owner/repo", not_a_dict_result)
    assert [e for e in events_not_dict if e["type"] == "escalated_label_repair_error"] == []

    errored_not_list_result = CommandResult(
        True,
        "review dispatch disabled",
        {
            "escalated_labels_repaired": {
                "issue_numbers": [],
                "failures": [],
                "errored": "901",
                "deferred": 0,
            }
        },
    )
    events_bad_errored = _extract_attention_events("owner/repo", errored_not_list_result)
    assert [e for e in events_bad_errored if e["type"] == "escalated_label_repair_error"] == []


# ---------------------------------------------------------------------------
# api-worker fleet report (issue #483)
# ---------------------------------------------------------------------------

_API_WORKER_YAML = """\
api_worker:
  enabled: {enabled}
  provider: kimi-k3
  providers:
    kimi-k3:
      base_url: https://api.moonshot.ai/anthropic
      api_key_env: MOONSHOT_API_KEY
      model: kimi-k3
      input_usd_per_mtok: 3.0
      output_usd_per_mtok: 15.0
      cached_input_usd_per_mtok: 0.30
  budget:
    max_usd_per_day: 5.0
    lifetime_usd: 15.0
"""

_BASE_YAML = """\
labels:
  ready: automated-ready
  queued: agent:queued
  in_progress: agent:in-progress
runtime:
  state_dir: .var/charlie-work
"""


def _make_repo(tmp_path: Path, name: str, *, api_worker: str | None) -> Path:
    """Create a repo dir with a config file. api_worker is the YAML snippet or None."""
    repo = tmp_path / name
    repo.mkdir(parents=True)
    config = repo / "orchestrator.config.yaml"
    content = _BASE_YAML
    if api_worker is not None:
        content += "\n" + api_worker
    config.write_text(content, encoding="utf-8")
    (repo / ".var" / "charlie-work").mkdir(parents=True)
    return repo


def _make_fleet_json(tmp_path: Path, fleet_dir: Path, repos: dict[str, dict[str, Any]]) -> None:
    fleet_json = fleet_dir / "fleet.json"
    fleet_json.parent.mkdir(parents=True, exist_ok=True)
    registry = {"version": 1, "repos": repos}
    fleet_json.write_text(_json.dumps(registry, indent=2), encoding="utf-8")


def test_api_worker_fleet_report_no_repos_configured(tmp_path: Path) -> None:
    """0 repos configured → report is None (line omitted entirely)."""
    fleet_dir = tmp_path / "fleet"
    repos_map = {}
    for i in range(4):
        repo = _make_repo(tmp_path, f"repo{i}", api_worker=None)
        repos_map[f"owner/repo{i}"] = {
            "repo_root": str(repo),
            "config_path": str(repo / "orchestrator.config.yaml"),
            "state_dir": str(repo / ".var" / "charlie-work"),
        }
    _make_fleet_json(tmp_path, fleet_dir, repos_map)

    report = compute_api_worker_fleet_report(fleet_dir_override=str(fleet_dir))

    assert report is None


def test_api_worker_fleet_report_partial_enablement(tmp_path: Path) -> None:
    """1/4 enabled → report shows enabled 1/4 repos."""
    fleet_dir = tmp_path / "fleet"
    repos_map = {}
    for i in range(4):
        enabled = i == 0  # Only repo0 enabled
        repo = _make_repo(
            tmp_path,
            f"repo{i}",
            api_worker=_API_WORKER_YAML.format(enabled="true" if enabled else "false"),
        )
        repos_map[f"owner/repo{i}"] = {
            "repo_root": str(repo),
            "config_path": str(repo / "orchestrator.config.yaml"),
            "state_dir": str(repo / ".var" / "charlie-work"),
        }
    _make_fleet_json(tmp_path, fleet_dir, repos_map)

    report = compute_api_worker_fleet_report(fleet_dir_override=str(fleet_dir))

    assert report is not None
    assert report.enabled_k == 1
    assert report.enabled_m == 4
    assert report.provider == "kimi-k3"
    assert report.live == 0
    assert report.cap_usd == 15.0
    line = report.format_line()
    assert "enabled 1/4 repos" in line
    assert "kimi-k3" in line
    assert "$15.00" in line


def test_api_worker_fleet_report_all_enabled(tmp_path: Path) -> None:
    """4/4 enabled → report shows enabled 4/4 repos."""
    fleet_dir = tmp_path / "fleet"
    repos_map = {}
    for i in range(4):
        repo = _make_repo(tmp_path, f"repo{i}", api_worker=_API_WORKER_YAML.format(enabled="true"))
        repos_map[f"owner/repo{i}"] = {
            "repo_root": str(repo),
            "config_path": str(repo / "orchestrator.config.yaml"),
            "state_dir": str(repo / ".var" / "charlie-work"),
        }
    _make_fleet_json(tmp_path, fleet_dir, repos_map)

    report = compute_api_worker_fleet_report(fleet_dir_override=str(fleet_dir))

    assert report is not None
    assert report.enabled_k == 4
    assert report.enabled_m == 4
    line = report.format_line()
    assert "enabled 4/4 repos" in line


def test_api_worker_fleet_report_all_disabled_but_configured(tmp_path: Path) -> None:
    """All configured but none enabled → line still renders (rollout insurance)."""
    fleet_dir = tmp_path / "fleet"
    repos_map = {}
    for i in range(2):
        repo = _make_repo(
            tmp_path, f"repo{i}", api_worker=_API_WORKER_YAML.format(enabled="false")
        )
        repos_map[f"owner/repo{i}"] = {
            "repo_root": str(repo),
            "config_path": str(repo / "orchestrator.config.yaml"),
            "state_dir": str(repo / ".var" / "charlie-work"),
        }
    _make_fleet_json(tmp_path, fleet_dir, repos_map)

    report = compute_api_worker_fleet_report(fleet_dir_override=str(fleet_dir))

    assert report is not None
    assert report.enabled_k == 0
    assert report.enabled_m == 2
    line = report.format_line()
    assert "enabled 0/2 repos" in line


def test_api_worker_fleet_report_line_format() -> None:
    """The format_line method produces the exact required format."""
    report = ApiWorkerFleetReport(
        provider="kimi-k3",
        today_usd=1.50,
        lifetime_usd=7.25,
        cap_usd=15.00,
        live=2,
        enabled_k=1,
        enabled_m=4,
    )
    line = report.format_line()
    assert (
        line == "api-worker: kimi-k3, $1.50 today / $7.25 lifetime of $15.00, "
        "2 live, enabled 1/4 repos"
    )


def test_api_worker_fleet_report_to_dict() -> None:
    """to_dict includes all fields plus the formatted line."""
    report = ApiWorkerFleetReport(
        provider="kimi-k3",
        today_usd=0.0,
        lifetime_usd=0.0,
        cap_usd=15.0,
        live=0,
        enabled_k=1,
        enabled_m=4,
    )
    d = report.to_dict()
    assert d["provider"] == "kimi-k3"
    assert d["today_usd"] == 0.0
    assert d["lifetime_usd"] == 0.0
    assert d["cap_usd"] == 15.0
    assert d["live"] == 0
    assert d["enabled_k"] == 1
    assert d["enabled_m"] == 4
    assert "line" in d
    assert "api-worker:" in d["line"]


def test_api_worker_fleet_report_spend_from_ledger(tmp_path: Path) -> None:
    """The report reads spend from the representative (enabled) repo's ledger.

    Regression for issue #828 (originally #822's class): production derives
    its own `today = now.strftime("%Y-%m-%d")` ledger key independently of
    this test's fixture write. If the wall clock crosses UTC midnight between
    the write and `compute_api_worker_fleet_report`'s read, the lookup misses
    and the report shows $0.00 instead of the expected spend -- a real (if
    rare) production defect, not just a test flake. `now` is frozen and
    passed to both the fixture and the report call so the ledger key always
    matches regardless of any stall or midnight boundary in between.
    """
    from datetime import UTC, datetime

    fleet_dir = tmp_path / "fleet"
    repo0 = _make_repo(tmp_path, "repo0", api_worker=_API_WORKER_YAML.format(enabled="true"))
    repo1 = _make_repo(tmp_path, "repo1", api_worker=_API_WORKER_YAML.format(enabled="false"))
    state_dir0 = repo0 / ".var" / "charlie-work"

    # Write a ledger with today's spend.
    frozen_now = datetime(2026, 7, 29, 12, 0, 0, tzinfo=UTC)
    today = frozen_now.strftime("%Y-%m-%d")
    ledger_data = {
        "days": {today: {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "usd": 2.25}},
        "lifetime_usd": 8.75,
        "sessions": [],
    }
    (state_dir0 / "api-budget.json").write_text(_json.dumps(ledger_data), encoding="utf-8")

    repos_map = {
        "owner/repo0": {
            "repo_root": str(repo0),
            "config_path": str(repo0 / "orchestrator.config.yaml"),
            "state_dir": str(state_dir0),
        },
        "owner/repo1": {
            "repo_root": str(repo1),
            "config_path": str(repo1 / "orchestrator.config.yaml"),
            "state_dir": str(repo1 / ".var" / "charlie-work"),
        },
    }
    _make_fleet_json(tmp_path, fleet_dir, repos_map)

    report = compute_api_worker_fleet_report(fleet_dir_override=str(fleet_dir), now=frozen_now)

    assert report is not None
    assert report.today_usd == 2.25
    assert report.lifetime_usd == 8.75
    line = report.format_line()
    assert "$2.25 today" in line
    assert "$8.75 lifetime" in line


def test_api_worker_fleet_report_no_hardcoded_lists() -> None:
    """The report line must not contain any hardcoded repo or provider names
    beyond what is derived from the actual fleet config. This is a sanity
    check that the format string uses only the report's own fields."""
    report = ApiWorkerFleetReport(
        provider="custom-provider",
        today_usd=0.0,
        lifetime_usd=0.0,
        cap_usd=100.0,
        live=0,
        enabled_k=3,
        enabled_m=7,
    )
    line = report.format_line()
    # The provider name comes from the report field, not a hardcoded list.
    assert "custom-provider" in line
    assert "enabled 3/7 repos" in line
    # No hardcoded provider names like "kimi-k3" or "moonshot" in the format.
    assert "moonshot" not in line


def test_compute_api_worker_fleet_report_uses_preloaded_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """preloaded_configs skips load_layered_config for those repos (no redundant reload).

    Review finding: fleet_loop reloaded every repo's config each pass even
    though it had already loaded the selected repos' configs for dispatch.
    This test verifies the optimization: a repo present in preloaded_configs
    reuses that config and load_layered_config is NOT called for it, while a
    repo absent from the map still falls back to load_layered_config.
    """
    from charlie_work.global_config import load_layered_config as real_load

    fleet_dir = tmp_path / "fleet"
    repo0 = _make_repo(tmp_path, "repo0", api_worker=_API_WORKER_YAML.format(enabled="true"))
    repo1 = _make_repo(tmp_path, "repo1", api_worker=_API_WORKER_YAML.format(enabled="true"))
    repos_map = {
        "owner/repo0": {
            "repo_root": str(repo0),
            "config_path": str(repo0 / "orchestrator.config.yaml"),
            "state_dir": str(repo0 / ".var" / "charlie-work"),
        },
        "owner/repo1": {
            "repo_root": str(repo1),
            "config_path": str(repo1 / "orchestrator.config.yaml"),
            "state_dir": str(repo1 / ".var" / "charlie-work"),
        },
    }
    _make_fleet_json(tmp_path, fleet_dir, repos_map)

    # Preload repo0's config exactly as fleet_loop would (raw layered config).
    preloaded = {
        "owner/repo0": real_load(repo0, repo0 / "orchestrator.config.yaml"),
    }

    calls: list[str] = []

    def _spy(repo_root: Path, explicit: Path | None, *, fleet_dir_override: str | None = None):
        calls.append(str(repo_root))
        return real_load(repo_root, explicit, fleet_dir_override=fleet_dir_override)

    monkeypatch.setattr("charlie_work.fleet_dispatch.load_layered_config", _spy)

    report = compute_api_worker_fleet_report(
        fleet_dir_override=str(fleet_dir), preloaded_configs=preloaded
    )

    assert report is not None
    assert report.enabled_m == 2
    assert report.enabled_k == 2
    # load_layered_config called only for repo1 (repo0 was preloaded).
    assert calls == [str(repo1)]


def test_compute_api_worker_fleet_report_preloaded_overrides_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preloaded config wins over what disk would load for that repo_key.

    This pins the contract: preloaded_configs is an override, not a hint. If
    the caller passes a default (unconfigured) config for a repo whose disk
    config has api_worker enabled, the report uses the preloaded view.
    """
    from charlie_work.config import ApiWorkerConfig, OrchestratorConfig as _OC
    from charlie_work.global_config import load_layered_config as real_load

    fleet_dir = tmp_path / "fleet"
    repo0 = _make_repo(tmp_path, "repo0", api_worker=_API_WORKER_YAML.format(enabled="true"))
    repos_map = {
        "owner/repo0": {
            "repo_root": str(repo0),
            "config_path": str(repo0 / "orchestrator.config.yaml"),
            "state_dir": str(repo0 / ".var" / "charlie-work"),
        },
    }
    _make_fleet_json(tmp_path, fleet_dir, repos_map)

    # Sanity: disk config has api_worker configured.
    disk_config = real_load(repo0, repo0 / "orchestrator.config.yaml")
    assert disk_config.api_worker != ApiWorkerConfig()

    # Preload a default (unconfigured) config to prove override semantics.
    preloaded = {"owner/repo0": _OC()}

    report = compute_api_worker_fleet_report(
        fleet_dir_override=str(fleet_dir), preloaded_configs=preloaded
    )

    # The preloaded default (unconfigured) wins → no repo configures the section.
    assert report is None


@patch("charlie_work.fleet_dispatch.compute_api_worker_fleet_report")
@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_threads_api_worker_report_into_data(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    mock_compute_report: MagicMock,
    tmp_path: Path,
) -> None:
    """fleet_loop places compute_api_worker_fleet_report's dict into CommandResult.data.

    The standalone compute function is covered by the tests above; this
    verifies the fleet_loop wiring (the api_worker_report key in the returned
    CommandResult.data) so a silent breakage in the key-lookup path can't ship
    undetected. An empty registry means no per-repo work runs.
    """
    mock_load_registry.return_value = {"repos": {}}
    report = ApiWorkerFleetReport(
        provider="kimi-k3",
        today_usd=1.50,
        lifetime_usd=7.25,
        cap_usd=15.00,
        live=2,
        enabled_k=1,
        enabled_m=4,
    )
    mock_compute_report.return_value = report

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=1,
        merge=None,
        dry_run=True,
        work_only=True,
    )

    assert result.ok is True
    assert result.data["api_worker_report"] == report.to_dict()
    assert result.data["api_worker_report"]["provider"] == "kimi-k3"
    assert "line" in result.data["api_worker_report"]
    # The compute function is called with the fleet_dir override and the
    # configs fleet_loop already loaded this pass (no redundant reload).
    mock_compute_report.assert_called_once_with(
        fleet_dir_override=str(tmp_path / "fleet"),
        preloaded_configs={},
    )


@patch("charlie_work.fleet_dispatch.compute_api_worker_fleet_report")
@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_api_worker_report_none_when_unconfigured(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    mock_compute_report: MagicMock,
    tmp_path: Path,
) -> None:
    """fleet_loop sets api_worker_report to None when no repo configures the section."""
    mock_load_registry.return_value = {"repos": {}}
    mock_compute_report.return_value = None

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=1,
        merge=None,
        dry_run=True,
        work_only=True,
    )

    assert result.ok is True
    assert result.data["api_worker_report"] is None


def test_compute_api_worker_fleet_report_respects_global_devin_sessions_dir_override(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """The live-api count resolves sessions_dir from the layered config, not state_dir default.

    Regression for the review of issue #707: the live-api loop in
    compute_api_worker_fleet_report used layout.sessions_dir_default directly,
    so a devin.sessions_dir override from the global fleet layer was ignored.
    """
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir(parents=True, exist_ok=True)
    repo = _make_repo(tmp_path, "repo", api_worker=_API_WORKER_YAML.format(enabled="true"))

    # Global fleet layer sets the sessions dir; per-repo config only declares api_worker.
    (fleet_dir / "config.yaml").write_text(
        "devin:\n  sessions_dir: custom-sessions\n",
        encoding="utf-8",
    )

    # The default sessions dir is empty; the live api sidecar is in the override.
    custom_sessions = repo / "custom-sessions"
    custom_sessions.mkdir(parents=True)
    (custom_sessions / "issue-1.api.json").write_text(
        _json.dumps(
            {
                "issue_number": 1,
                "branch": "main",
                "worktree_path": str(repo / "worktrees" / "issue-1"),
                "prompt_path": str(repo / "prompt.md"),
                "command": ["claude"],
                "pid": 1234,
                "started_at": "2026-08-05T00:00:00Z",
                "log_path": str(repo / "log.txt"),
                "adapter_kind": "api",
                "provider": "kimi-k3",
            }
        ),
        encoding="utf-8",
    )

    repos_map = {
        "owner/repo": {
            "repo_root": str(repo),
            "config_path": str(repo / "orchestrator.config.yaml"),
            "state_dir": str(repo / ".var" / "charlie-work"),
        }
    }
    _make_fleet_json(tmp_path, fleet_dir, repos_map)

    monkeypatch.setattr("charlie_work.worker.is_worker_alive", lambda _record: True)

    report = compute_api_worker_fleet_report(fleet_dir_override=str(fleet_dir))

    assert report is not None
    assert report.live == 1


def test_take_fleet_snapshot_detects_delta_with_devin_sessions_dir_override(
    tmp_path: Path,
) -> None:
    """_take_fleet_snapshot uses the resolved sessions_dir, not the default.

    Regression for the review of issue #707: _repo_state_dirs built the
    sessions dir from layout.sessions_dir_default, so a devin.sessions_dir
    override produced no snapshot signal and no fleet delta.
    """
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir(parents=True, exist_ok=True)
    repo = _make_repo(tmp_path, "repo", api_worker=None)

    # Global fleet layer overrides the sessions dir.
    (fleet_dir / "config.yaml").write_text(
        "devin:\n  sessions_dir: custom-sessions\n",
        encoding="utf-8",
    )

    custom_sessions = repo / "custom-sessions"
    custom_sessions.mkdir(parents=True)
    (custom_sessions / "issue-1.json").write_text(
        _json.dumps({"dummy": "sidecar"}), encoding="utf-8"
    )

    repos_map = {
        "owner/repo": {
            "repo_root": str(repo),
            "config_path": str(repo / "orchestrator.config.yaml"),
            "state_dir": str(repo / ".var" / "charlie-work"),
        }
    }
    _make_fleet_json(tmp_path, fleet_dir, repos_map)

    before = _take_fleet_snapshot(fleet_dir_override=str(fleet_dir))

    (custom_sessions / "issue-2.json").write_text(
        _json.dumps({"dummy": "sidecar"}), encoding="utf-8"
    )

    after = _take_fleet_snapshot(fleet_dir_override=str(fleet_dir))

    assert _has_fleet_delta(before, after) is True


def test_take_fleet_snapshot_skips_repo_with_malformed_config(
    tmp_path: Path,
) -> None:
    """A repo with an unparseable per-repo config does not crash _take_fleet_snapshot.

    Regression for the review of issue #707: _take_fleet_snapshot's new
    load_layered_config call caught only ConfigError and OSError, so a
    malformed orchestrator.config.yaml (which raises yaml.YAMLError) crashed
    the fleet supervisor at startup.
    """
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir(parents=True, exist_ok=True)
    repo = _make_repo(tmp_path, "repo", api_worker=None)

    # Plant a malformed YAML file that yaml.safe_load cannot parse.
    (repo / "orchestrator.config.yaml").write_text(
        "devin:\n  sessions_dir: [unclosed\n",
        encoding="utf-8",
    )

    repos_map = {
        "owner/repo": {
            "repo_root": str(repo),
            "config_path": str(repo / "orchestrator.config.yaml"),
            "state_dir": str(repo / ".var" / "charlie-work"),
        }
    }
    _make_fleet_json(tmp_path, fleet_dir, repos_map)

    result = _take_fleet_snapshot(fleet_dir_override=str(fleet_dir))

    assert isinstance(result, FleetLocalSnapshot)


def test_compute_api_worker_fleet_report_skips_repo_with_malformed_config(
    tmp_path: Path,
) -> None:
    """A repo with an unparseable per-repo config does not crash compute_api_worker_fleet_report.

    Regression for the review of issue #707: compute_api_worker_fleet_report's
    first loop caught only (ConfigError, GitHubError, OSError), so a malformed
    orchestrator.config.yaml (which raises yaml.YAMLError) crashed the fleet
    pass and ``charlie fleet status`` instead of skipping the repo.
    """
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir(parents=True, exist_ok=True)
    repo = _make_repo(tmp_path, "repo", api_worker=_API_WORKER_YAML.format(enabled="true"))

    # Plant a malformed YAML file that yaml.safe_load cannot parse.
    (repo / "orchestrator.config.yaml").write_text(
        "devin:\n  sessions_dir: [unclosed\n",
        encoding="utf-8",
    )

    repos_map = {
        "owner/repo": {
            "repo_root": str(repo),
            "config_path": str(repo / "orchestrator.config.yaml"),
            "state_dir": str(repo / ".var" / "charlie-work"),
        }
    }
    _make_fleet_json(tmp_path, fleet_dir, repos_map)

    # Must return None (no repo configured a usable api_worker section) rather
    # than raising yaml.YAMLError.
    report = compute_api_worker_fleet_report(fleet_dir_override=str(fleet_dir))

    assert report is None


def test_take_fleet_snapshot_skips_repo_with_null_repo_root(
    tmp_path: Path,
) -> None:
    """A corrupted registry entry with repo_root: null does not crash _take_fleet_snapshot.

    Regression for the review of issue #707: ``Path(entry.get("repo_root", ""))``
    returns ``Path(None)`` when the key is present with a null value (``.get``'s
    default only applies when the key is *absent*), raising TypeError. The same
    bug class existed in compute_api_worker_fleet_report and the autoscale
    prologue; all three now use ``entry.get("repo_root") or ""``.
    """
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir(parents=True, exist_ok=True)
    repo = _make_repo(tmp_path, "repo", api_worker=None)

    repos_map = {
        "owner/repo": {
            "repo_root": None,
            "config_path": str(repo / "orchestrator.config.yaml"),
            "state_dir": str(repo / ".var" / "charlie-work"),
        }
    }
    _make_fleet_json(tmp_path, fleet_dir, repos_map)

    result = _take_fleet_snapshot(fleet_dir_override=str(fleet_dir))

    assert isinstance(result, FleetLocalSnapshot)


def test_compute_api_worker_fleet_report_skips_repo_with_null_repo_root(
    tmp_path: Path,
) -> None:
    """A corrupted registry entry with repo_root: null does not crash compute_api_worker_fleet_report.

    ``entry.get("repo_root") or ""`` makes a null value behave like a missing
    key (fall back to cwd), matching the pre-existing behavior — the fix is
    about preventing the TypeError crash, not changing the missing-key path.
    """
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir(parents=True, exist_ok=True)
    repo = _make_repo(tmp_path, "repo", api_worker=_API_WORKER_YAML.format(enabled="true"))

    repos_map = {
        "owner/repo": {
            "repo_root": None,
            "config_path": str(repo / "orchestrator.config.yaml"),
            "state_dir": str(repo / ".var" / "charlie-work"),
        }
    }
    _make_fleet_json(tmp_path, fleet_dir, repos_map)

    # Must not raise TypeError; the call completes and returns a value.
    report = compute_api_worker_fleet_report(fleet_dir_override=str(fleet_dir))

    assert report is None or isinstance(report, ApiWorkerFleetReport)


# --------------------------------------------------------------------------
# Runner-allocation prologue (the path the 5-minute fleet pass actually takes)
# --------------------------------------------------------------------------


def _allocation_config(**overrides: Any) -> OrchestratorConfig:
    """A global config with the allocation section populated."""
    return OrchestratorConfig(
        runner_allocation=RunnerAllocationConfig(**overrides),
        runner_scaling=RunnerScalingConfig(managed_root="C:/fallback-root"),
    )


def test_allocation_prologue_is_inert_when_disabled(tmp_path: Path) -> None:
    """Default-off must mean off: no registry read, no gh client, no events."""
    with patch("charlie_work.fleet_dispatch.run_allocation_pass") as pass_mock:
        events = _run_fleet_allocation_prologue(
            str(tmp_path / "fleet"),
            _allocation_config(enabled=False),
            dry_run=False,
        )

    assert events == []
    pass_mock.assert_not_called()


def test_allocation_prologue_skips_when_no_repo_root_is_usable(tmp_path: Path) -> None:
    """Without a real directory to anchor the gh client, skip rather than guess."""
    fleet_dir = tmp_path / "fleet"
    _make_fleet_json(
        tmp_path,
        fleet_dir,
        {"owner/gone": {"repo_root": str(tmp_path / "vanished"), "state_dir": ""}},
    )

    with patch("charlie_work.fleet_dispatch.run_allocation_pass") as pass_mock:
        events = _run_fleet_allocation_prologue(
            str(fleet_dir), _allocation_config(enabled=True), dry_run=False
        )

    # Visible, not silent: an unusable registry is nobody's deliberate choice,
    # so it reaches the digest rather than only the log (issue #590).
    assert [event["type"] for event in events] == ["runner_allocation_skipped"]
    assert "registry" in events[0]["reason"]
    pass_mock.assert_not_called()

    # The skip also leaves state-file evidence so the doctor probe does not
    # attribute a fresh unattended decline to "never ran" (issue #606).
    state_file = fleet_dir / ALLOCATION_STATE_FILENAME
    assert state_file.exists()
    stamp = load_allocation_stamp(fleet_dir)
    assert stamp.source == "prologue"
    assert stamp.full_pass_interval_seconds == 300
    assert stamp.skip_reason is not None
    assert "no usable repo root" in stamp.skip_reason


def test_allocation_prologue_skip_no_repo_root_is_usable_in_dry_run(
    tmp_path: Path,
) -> None:
    """A dry-run preview must not bump the allocation state file."""
    fleet_dir = tmp_path / "fleet"
    _make_fleet_json(
        tmp_path,
        fleet_dir,
        {"owner/gone": {"repo_root": str(tmp_path / "vanished"), "state_dir": ""}},
    )

    with patch("charlie_work.fleet_dispatch.run_allocation_pass") as pass_mock:
        events = _run_fleet_allocation_prologue(
            str(fleet_dir), _allocation_config(enabled=True), dry_run=True
        )

    assert [event["type"] for event in events] == ["runner_allocation_skipped"]
    pass_mock.assert_not_called()
    assert not (fleet_dir / ALLOCATION_STATE_FILENAME).exists()


def test_allocation_prologue_skip_no_repo_root_is_seen_by_doctor(
    tmp_path: Path,
) -> None:
    """A prologue skip with allocation enabled must reach the doctor probe."""
    fleet_dir = tmp_path / "fleet"
    _make_fleet_json(
        tmp_path,
        fleet_dir,
        {"owner/gone": {"repo_root": str(tmp_path / "vanished"), "state_dir": ""}},
    )

    with patch("charlie_work.fleet_dispatch.run_allocation_pass"):
        _run_fleet_allocation_prologue(
            str(fleet_dir), _allocation_config(enabled=True), dry_run=False
        )

    collected: list[tuple[str, bool, str, str]] = []

    def add(name: str, ok: bool, detail: str, *, severity: str = "error") -> None:
        collected.append((name, ok, detail, severity))

    _check_runner_allocation(
        add,
        _allocation_config(enabled=True),
        fleet_dir_override=str(fleet_dir),
    )

    assert len(collected) == 1
    name, ok, detail, severity = collected[0]
    assert name == "runner allocation"
    assert ok is False
    assert "declined to act" in detail
    assert "no usable repo root" in detail
    # A fresh unattended skip names its own cause; it must not be misread as
    # the "never reached allocation" shape of issue #590.
    assert "#590" not in detail
    assert severity == "warning"


def test_allocation_prologue_anchors_on_a_live_repo_and_passes_config_through(
    tmp_path: Path,
) -> None:
    """The prologue's whole job: find an anchor, hand the pass its wiring."""
    fleet_dir = tmp_path / "fleet"
    repo = _make_repo(tmp_path, "anchor", api_worker=None)
    _make_fleet_json(
        tmp_path,
        fleet_dir,
        {
            "owner/anchor": {
                "repo_root": str(repo),
                "state_dir": str(repo / ".var" / "charlie-work"),
            }
        },
    )

    plan = AllocationPlan(budget=8, budget_reason="configured", targets=(), changes=())
    result = AllocationPassResult(ok=True, plan=plan, notes=("cw: pinned",))

    with (
        patch("charlie_work.fleet_dispatch.run_allocation_pass", return_value=result) as pass_mock,
        patch("charlie_work.fleet_dispatch.GitHub") as gh_mock,
    ):
        events = _run_fleet_allocation_prologue(
            str(fleet_dir),
            _allocation_config(enabled=True, managed_root="C:/actions-runners"),
            dry_run=True,
        )

    assert gh_mock.call_args.kwargs["repo_root"] == repo
    kwargs = pass_mock.call_args.kwargs
    assert kwargs["managed_root_fallback"] == "C:/fallback-root"
    assert kwargs["fleet_dir_override"] == str(fleet_dir)
    # Issue #603: the state_path is the fleet-level path, not the anchor
    # repo's per-repo state.json. Host-wide allocation events go to the
    # fleet-level events.db, not whichever repo sorted first in the registry.
    assert kwargs["state_path"] == fleet_dir / "state.json"
    assert kwargs["dry_run"] is True
    # The driving interval is threaded from the caller's resolved config so the
    # state file records the cadence the daemon actually used (issue #606).
    assert kwargs["full_pass_interval_seconds"] == 300

    # A note alone is enough to surface an event; a balanced host stays quiet.
    assert [event["type"] for event in events] == ["runner_allocation"]
    assert events[0]["budget"] == 8
    assert events[0]["dry_run"] is True


def test_allocation_prologue_stays_quiet_when_nothing_moves(tmp_path: Path) -> None:
    """A balanced host must not add a line to every 5-minute digest."""
    fleet_dir = tmp_path / "fleet"
    repo = _make_repo(tmp_path, "anchor", api_worker=None)
    _make_fleet_json(
        tmp_path,
        fleet_dir,
        {"owner/anchor": {"repo_root": str(repo), "state_dir": str(repo / ".var")}},
    )
    balanced = AllocationPassResult(
        ok=True,
        plan=AllocationPlan(budget=8, budget_reason="configured", targets=(), changes=()),
    )

    with (
        patch("charlie_work.fleet_dispatch.run_allocation_pass", return_value=balanced),
        patch("charlie_work.fleet_dispatch.GitHub"),
    ):
        events = _run_fleet_allocation_prologue(
            str(fleet_dir), _allocation_config(enabled=True), dry_run=False
        )

    assert events == []


def test_allocation_prologue_surfaces_errors_and_failed_slots(tmp_path: Path) -> None:
    """Config typos and refused parks both have to reach the operator."""
    fleet_dir = tmp_path / "fleet"
    repo = _make_repo(tmp_path, "anchor", api_worker=None)
    _make_fleet_json(
        tmp_path,
        fleet_dir,
        {"owner/anchor": {"repo_root": str(repo), "state_dir": str(repo / ".var")}},
    )

    failed = AllocationPassResult(ok=False, error="managed_root does not exist: C:/nope")
    with (
        patch("charlie_work.fleet_dispatch.run_allocation_pass", return_value=failed),
        patch("charlie_work.fleet_dispatch.GitHub"),
    ):
        events = _run_fleet_allocation_prologue(
            str(fleet_dir), _allocation_config(enabled=True), dry_run=False
        )
    assert [event["type"] for event in events] == ["runner_allocation_error"]
    assert "does not exist" in events[0]["error"]

    change = SlotChange(
        repo="owner/anchor",
        runner_name="cw-2",
        path=tmp_path / "cw-2",
        action=SlotAction.PARK,
        reason="idle 3 passes",
    )
    with (
        patch(
            "charlie_work.fleet_dispatch.run_allocation_pass",
            return_value=AllocationPassResult(
                ok=True,
                plan=AllocationPlan(
                    budget=8, budget_reason="configured", targets=(), changes=(change,)
                ),
                results=(SlotChangeResult(change=change, ok=False, message="job in flight"),),
            ),
        ),
        patch("charlie_work.fleet_dispatch.GitHub"),
    ):
        events = _run_fleet_allocation_prologue(
            str(fleet_dir), _allocation_config(enabled=True), dry_run=False
        )

    assert [event["type"] for event in events] == ["runner_allocation_slot_error"]
    assert events[0]["runner"] == "cw-2"
    assert events[0]["action"] == "park"


def test_allocation_prologue_records_a_delegated_skip(tmp_path: Path) -> None:
    """Issue #958: ``run_allocation_pass`` declining must reach the digest
    and events.db.

    Pre-fix, this branch did not exist: control fell through to the
    started/parked/notes check below. Both current ci_fleet skip branches
    populate ``notes``, so a clean decline (``skipped=True``, ``error=None``)
    was misfiled as a healthy no-op ``runner_allocation`` event -- which the
    digest deliberately drops as noise -- so it reached neither the notify
    digest nor events.db. It must now surface as a durable
    ``runner_allocation_skipped`` event in both places instead.
    """
    fleet_dir = tmp_path / "fleet"
    repo = _make_repo(tmp_path, "anchor", api_worker=None)
    state_dir = repo / ".var"
    _make_fleet_json(
        tmp_path,
        fleet_dir,
        {"owner/anchor": {"repo_root": str(repo), "state_dir": str(state_dir)}},
    )

    declined = AllocationPassResult(
        ok=True,
        skipped=True,
        notes=("no configured runners found under C:/actions-runners",),
    )
    with (
        patch("charlie_work.fleet_dispatch.run_allocation_pass", return_value=declined),
        patch("charlie_work.fleet_dispatch.GitHub"),
    ):
        events = _run_fleet_allocation_prologue(
            str(fleet_dir), _allocation_config(enabled=True), dry_run=False
        )

    assert events == [
        {
            "repo_key": "fleet",
            "type": "runner_allocation_skipped",
            "reason": "no configured runners found under C:/actions-runners",
        }
    ]

    # And a genuine events.db row, not just the in-memory digest -- this is
    # the actual durable record the issue's evidence section was about.
    # Issue #603: the row lands in the fleet-level events.db, not the anchor
    # repo's per-repo database.
    fleet_state_path = fleet_dir / "state.json"
    rows = query_events(fleet_state_path, kind="runner_allocation_skipped")
    assert len(rows) == 1
    assert rows[0]["payload"]["reason"] == "no configured runners found under C:/actions-runners"
    assert rows[0]["payload"]["dry_run"] is False
    assert rows[0]["level"] == "warning"


def test_allocation_prologue_records_a_delegated_skip_with_no_notes(tmp_path: Path) -> None:
    """A delegated skip with no notes at all must still leave a trace.

    ``AllocationPassResult`` carries no dedicated reason field, so if a future
    ci_fleet skip branch ever returns ``skipped=True`` with empty notes, the
    digest must still record that the pass declined rather than silently
    dropping it the way the pre-fix code did for every delegated skip.
    """
    fleet_dir = tmp_path / "fleet"
    repo = _make_repo(tmp_path, "anchor", api_worker=None)
    _make_fleet_json(
        tmp_path,
        fleet_dir,
        {"owner/anchor": {"repo_root": str(repo), "state_dir": str(repo / ".var")}},
    )

    declined = AllocationPassResult(ok=True, skipped=True)
    with (
        patch("charlie_work.fleet_dispatch.run_allocation_pass", return_value=declined),
        patch("charlie_work.fleet_dispatch.GitHub"),
    ):
        events = _run_fleet_allocation_prologue(
            str(fleet_dir), _allocation_config(enabled=True), dry_run=False
        )

    assert [event["type"] for event in events] == ["runner_allocation_skipped"]
    assert "not exposed" in events[0]["reason"]


def test_allocation_prologue_delegated_skip_tolerates_no_anchor_state(tmp_path: Path) -> None:
    """A registry entry with no recorded ``state_dir`` must not crash.

    Pre-#603, ``anchor_state`` was ``None`` whenever the anchor repo's registry
    entry had no ``state_dir`` on file yet (e.g. its very first pass), so the
    durable events.db row was skipped and only the in-memory digest survived.
    Post-#603, the event-store path is derived from ``fleet_dir()``, not from
    the anchor's ``state_dir``, so the decline is always durably recorded in
    the fleet-level events.db regardless of the registry entry's state_dir.
    """
    fleet_dir = tmp_path / "fleet"
    repo = _make_repo(tmp_path, "anchor", api_worker=None)
    _make_fleet_json(
        tmp_path,
        fleet_dir,
        {"owner/anchor": {"repo_root": str(repo)}},  # no state_dir
    )

    declined = AllocationPassResult(ok=True, skipped=True, notes=("declined",))
    with (
        patch("charlie_work.fleet_dispatch.run_allocation_pass", return_value=declined),
        patch("charlie_work.fleet_dispatch.GitHub"),
    ):
        events = _run_fleet_allocation_prologue(
            str(fleet_dir), _allocation_config(enabled=True), dry_run=False
        )

    assert [event["type"] for event in events] == ["runner_allocation_skipped"]
    assert events[0]["reason"] == "declined"

    # Issue #603: the durable record lands in the fleet-level events.db even
    # though the anchor entry has no state_dir — the path no longer depends on
    # the registry entry that supplied the gh anchor.
    fleet_state_path = fleet_dir / "state.json"
    rows = query_events(fleet_state_path, kind="runner_allocation_skipped")
    assert len(rows) == 1
    assert rows[0]["payload"]["reason"] == "declined"


def test_allocation_prologue_routes_events_to_fleet_store_not_anchor_repo(
    tmp_path: Path,
) -> None:
    """Issue #603: host-wide allocation events land in the fleet-level events.db.

    Pre-fix, ``_run_fleet_allocation_prologue`` derived ``state_path`` from the
    same registry entry that supplied the gh anchor — the first entry with a
    valid ``repo_root``. The anchor and the event-store path are independent
    concerns: the anchor only needs auth and a valid directory (the pass
    addresses every repo by explicit slug), whereas ``state_path`` decides
    which repo's ``events.db`` records the host-wide allocation event. So the
    audit trail for a host-wide action landed in whichever repo happened to
    sort first in the registry, and moved if the registry order changed.

    Post-fix, the event-store path is ``fleet_dir() / "state.json"``, so
    ``log_event`` writes to ``fleet_dir() / "events.db"`` — the same
    fleet-level store ``supervisor_lifecycle`` already writes host-wide events
    to. The per-repo databases are disjoint from it by event scope.

    This test registers two repos with distinct ``state_dir`` paths, runs the
    prologue, and verifies:
    1. The allocation event is in the fleet-level events.db.
    2. Neither per-repo events.db contains the allocation event.
    3. The gh anchor is still derived from the first valid repo root.
    """
    fleet_dir = tmp_path / "fleet"
    repo_a = _make_repo(tmp_path, "alpha", api_worker=None)
    repo_b = _make_repo(tmp_path, "beta", api_worker=None)
    state_dir_a = repo_a / ".var" / "charlie-work"
    state_dir_b = repo_b / ".var" / "charlie-work"
    _make_fleet_json(
        tmp_path,
        fleet_dir,
        {
            "owner/alpha": {
                "repo_root": str(repo_a),
                "state_dir": str(state_dir_a),
            },
            "owner/beta": {
                "repo_root": str(repo_b),
                "state_dir": str(state_dir_b),
            },
        },
    )

    # Use a skipped result so the prologue's own log_event call writes a
    # durable ``runner_allocation_skipped`` row — ``run_allocation_pass`` is
    # mocked, so the ``runner_allocation`` event it would normally write
    # never reaches the DB. The skipped path exercises the same state_path
    # routing the fix changes.
    result = AllocationPassResult(ok=True, skipped=True, notes=("no configured runners",))
    with (
        patch("charlie_work.fleet_dispatch.run_allocation_pass", return_value=result) as pass_mock,
        patch("charlie_work.fleet_dispatch.GitHub") as gh_mock,
    ):
        _run_fleet_allocation_prologue(
            str(fleet_dir), _allocation_config(enabled=True), dry_run=False
        )

    # The gh anchor is still the first valid repo root — that concern is
    # unchanged. Only the event-store path was decoupled.
    assert gh_mock.call_args.kwargs["repo_root"] == repo_a

    # The state_path passed to run_allocation_pass is the fleet-level path,
    # not repo_a's per-repo state.json.
    assert pass_mock.call_args.kwargs["state_path"] == fleet_dir / "state.json"

    # The allocation skip event is in the fleet-level events.db, not in
    # either per-repo database. This is the core assertion of #603: the
    # audit trail no longer lands in whichever repo sorted first.
    fleet_state_path = fleet_dir / "state.json"
    fleet_rows = query_events(fleet_state_path, kind="runner_allocation_skipped")
    assert len(fleet_rows) == 1
    assert fleet_rows[0]["payload"]["reason"] == "no configured runners"

    for per_repo_state in (
        layout.state_file_path(state_dir_a),
        layout.state_file_path(state_dir_b),
    ):
        per_repo_rows = query_events(per_repo_state, kind="runner_allocation_skipped")
        assert per_repo_rows == [], (
            f"host-wide allocation event leaked into per-repo events.db at {per_repo_state}"
        )


def test_allocation_prologue_warns_when_the_config_lacks_the_section(
    tmp_path: Path, caplog: Any
) -> None:
    """A config object without the section means code/config disagree.

    That is a different failure from "the operator left it off" — it happens
    when a load failure already fell back to defaults, or when the process is
    holding a config built by other code — and it must not look like a
    deliberate opt-out.
    """
    import logging

    class _NoSection:
        pass

    with caplog.at_level(logging.WARNING, logger="charlie_work.fleet_dispatch"):
        events = _run_fleet_allocation_prologue(str(tmp_path / "fleet"), _NoSection(), False)

    assert [event["type"] for event in events] == ["runner_allocation_skipped"]
    assert "NoSection" in events[0]["reason"]
    assert any("no runner_allocation" in record.message for record in caplog.records)


def test_disabled_allocation_prologue_is_visible_at_info(tmp_path, caplog) -> None:
    """The disabled branch must log at INFO, not DEBUG.

    Regression guard for issue #590: the daemon runs at INFO, so a DEBUG line here
    is never written at all, which made a host where allocation never ran
    indistinguishable from a converged one. The message must also name the fleet
    directory, since that identifies which config.yaml governs the decision.
    """
    import logging
    from dataclasses import replace

    from charlie_work.config import OrchestratorConfig
    from charlie_work.fleet_dispatch import _run_fleet_allocation_prologue

    base = OrchestratorConfig()
    config = replace(base, runner_allocation=replace(base.runner_allocation, enabled=False))

    with caplog.at_level(logging.INFO, logger="charlie_work.fleet_dispatch"):
        events = _run_fleet_allocation_prologue(str(tmp_path), config, True)

    assert events == []
    assert "runner_allocation.enabled is false" in caplog.text
    assert str(tmp_path) in caplog.text


def test_allocation_prologue_logs_its_inputs_before_any_branch(tmp_path: Any, caplog: Any) -> None:
    """The entry line must be unconditional.

    Regression guard for the evidence gap that kept #590 unisolated: every skip
    path used to log from *inside* a branch, so an absent log line was equally
    consistent with "reached and declined" and "never reached". One line before
    the first branch separates those two readings, which is the only thing that
    distinguishes a misconfigured host from an unreached call site.
    """
    import logging

    class _NoSection:
        pass

    with caplog.at_level(logging.INFO, logger="charlie_work.fleet_dispatch"):
        _run_fleet_allocation_prologue(str(tmp_path / "fleet"), _NoSection(), False)

    # Logged even in the most degenerate case, where there is no section to read.
    assert "prologue: entered" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="charlie_work.fleet_dispatch"):
        _run_fleet_allocation_prologue(
            str(tmp_path / "fleet"),
            _allocation_config(enabled=False, managed_root="C:/actions-runners", **{}),
            False,
        )

    assert "prologue: entered" in caplog.text
    assert "enabled=False" in caplog.text
    assert "C:/actions-runners" in caplog.text


def _make_ci_fleet_git_repo(tmp_path: Path) -> Path:
    """Create a minimal editable-style ci_fleet repo with a clean ``src/`` tree."""
    repo = tmp_path / "ci_fleet_repo"
    repo.mkdir()
    pkg = repo / "src" / "ci_fleet"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("# ci_fleet", encoding="utf-8")

    # Use a per-test gitconfig so the commit does not depend on global config.
    gitconfig = tmp_path / "gitconfig"
    gitconfig.write_text("[user]\n\tname = Test\n\temail = test@test\n", encoding="utf-8")
    env = dict(os.environ, GIT_CONFIG_GLOBAL=str(gitconfig))

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
        env=env,
    )
    return repo


def test_ci_fleet_guard_detects_dirty_src_tree(tmp_path: Path) -> None:
    """Issue #927: the guard must fire when src/ has uncommitted changes.

    This is the positive-control test: a guard whose tests only exercise the
    clean path can pass forever while the dirty path is broken.
    """
    repo = _make_ci_fleet_git_repo(tmp_path)
    module_file = repo / "src" / "ci_fleet" / "__init__.py"
    # Uncommitted addition under src/ -- the same shape as an agent editing
    # planner.py in the live ci_fleet tree.
    (repo / "src" / "ci_fleet" / "planner.py").write_text("x = 1", encoding="utf-8")

    check = _real_ci_fleet_worktree_dirty(module_file)

    assert check.is_dirty is True
    assert check.repo_root == repo
    assert check.dirty_paths
    assert any("planner.py" in p for p in check.dirty_paths)


def test_ci_fleet_guard_clean_src_tree_is_inert(tmp_path: Path) -> None:
    """A clean src/ tree must let allocation proceed normally."""
    repo = _make_ci_fleet_git_repo(tmp_path)
    module_file = repo / "src" / "ci_fleet" / "__init__.py"

    check = _real_ci_fleet_worktree_dirty(module_file)

    assert check.is_dirty is False
    assert check.repo_root == repo
    assert check.dirty_paths == ()


def test_ci_fleet_guard_no_git_is_inert(tmp_path: Path) -> None:
    """A wheel install or missing .git must not block allocation."""
    pkg = tmp_path / "pkg" / "ci_fleet"
    pkg.mkdir(parents=True)
    module_file = pkg / "__init__.py"
    module_file.write_text("# installed wheel", encoding="utf-8")

    check = _real_ci_fleet_worktree_dirty(module_file)

    assert check.is_dirty is False
    assert check.repo_root is None
    assert check.dirty_paths == ()


def test_ci_fleet_guard_git_failure_is_inert(tmp_path: Path, monkeypatch: Any) -> None:
    """A git error must make the guard a no-op, not a hard stop."""
    repo = _make_ci_fleet_git_repo(tmp_path)
    module_file = repo / "src" / "ci_fleet" / "__init__.py"

    def _failing_git(*, cwd, timeout_seconds):  # pragma: no cover
        return RunResult(
            returncode=1,
            stdout="",
            stderr="git exploded",
            error="git exploded",
        )

    # Accept the positional `command` argument and ignore it.
    def _failing_run_captured(
        command: list[str], *, cwd: Path | str, timeout_seconds: int
    ) -> RunResult:
        return _failing_git(cwd=cwd, timeout_seconds=timeout_seconds)

    monkeypatch.setattr("charlie_work.fleet_dispatch.run_captured", _failing_run_captured)

    check = _real_ci_fleet_worktree_dirty(module_file)

    assert check.is_dirty is False
    assert check.repo_root == repo
    assert "git status failed" in (check.reason or "")


def test_allocation_prologue_forces_dry_run_when_ci_fleet_is_dirty(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Issue #927: a dirty ci_fleet src/ forces a dry run and emits a guard event."""
    fleet_dir = tmp_path / "fleet"
    repo = _make_repo(tmp_path, "anchor", api_worker=None)
    state_dir = repo / ".var" / "charlie-work"
    _make_fleet_json(
        tmp_path,
        fleet_dir,
        {"owner/anchor": {"repo_root": str(repo), "state_dir": str(state_dir)}},
    )

    dirty_check = _CiFleetDirtyCheck(
        is_dirty=True,
        repo_root=tmp_path / "ci_fleet",
        dirty_paths=(" M src/planner.py",),
    )
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch._ci_fleet_worktree_dirty",
        lambda _module_file=None: dirty_check,
    )

    plan = AllocationPlan(budget=8, budget_reason="configured", targets=(), changes=())
    result = AllocationPassResult(ok=True, plan=plan, notes=("cw: pinned",))

    with (
        patch("charlie_work.fleet_dispatch.run_allocation_pass", return_value=result) as pass_mock,
        patch("charlie_work.fleet_dispatch.GitHub"),
    ):
        events = _run_fleet_allocation_prologue(
            str(fleet_dir), _allocation_config(enabled=True), dry_run=False
        )

    assert pass_mock.call_args.kwargs["dry_run"] is True
    assert any(e["type"] == "ci_fleet_worktree_dirty" for e in events)

    # Issue #603: the guard event lands in the fleet-level events.db, not the
    # anchor repo's per-repo database.
    fleet_state_path = fleet_dir / "state.json"
    rows = query_events(fleet_state_path, kind="ci_fleet_worktree_dirty")
    assert len(rows) == 1
    assert rows[0]["payload"]["dirty_paths"] == [" M src/planner.py"]
    assert rows[0]["payload"]["dry_run_forced"] is True
    assert rows[0]["level"] == "warning"


def test_allocation_prologue_keeps_original_dry_run_when_ci_fleet_is_clean(
    tmp_path: Path,
) -> None:
    """A clean ci_fleet tree must not force dry_run on an actuating pass."""
    fleet_dir = tmp_path / "fleet"
    repo = _make_repo(tmp_path, "anchor", api_worker=None)
    _make_fleet_json(
        tmp_path,
        fleet_dir,
        {"owner/anchor": {"repo_root": str(repo), "state_dir": str(repo / ".var")}},
    )
    balanced = AllocationPassResult(
        ok=True,
        plan=AllocationPlan(budget=8, budget_reason="configured", targets=(), changes=()),
    )

    with (
        patch(
            "charlie_work.fleet_dispatch.run_allocation_pass", return_value=balanced
        ) as pass_mock,
        patch("charlie_work.fleet_dispatch.GitHub"),
    ):
        _run_fleet_allocation_prologue(
            str(fleet_dir), _allocation_config(enabled=True), dry_run=False
        )

    assert pass_mock.call_args.kwargs["dry_run"] is False


def test_digest_renders_event_types_that_have_no_explicit_branch() -> None:
    """An unmapped event type must not vanish.

    The mapper was an if/elif chain with no else, so the prologue's own
    ``runner_allocation_error`` was emitted correctly and then dropped on the
    floor — the digest could not show the one signal that mattered for #590.
    """
    events = [
        {"repo_key": "fleet", "type": "runner_allocation_error", "error": "managed_root missing"},
        {
            "repo_key": "fleet",
            "type": "runner_allocation_skipped",
            "reason": "no usable repo root",
        },
    ]

    digest = _build_fleet_attention_digest(events)

    assert len(digest.transitions) == 2
    rendered = {entry.health: entry.last_log_line for entry in digest.transitions}
    assert rendered["ERROR"] == "managed_root missing"
    assert rendered["INFO"] == "no usable repo root"


@patch("charlie_work.fleet_dispatch.run_allocation_pass")
@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_actually_reaches_the_allocation_pass(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    mock_run_allocation_pass: MagicMock,
    tmp_path: Path,
) -> None:
    """Pin the wiring, not just the pieces.

    Every prologue test calls ``_run_fleet_allocation_prologue`` directly and
    every ``run_fleet_supervise`` test patches ``fleet_loop`` out, so nothing
    asserted that a real fleet pass reaches allocation at all. This drives
    ``fleet_loop`` unmocked and patches only one level below the prologue.

    Scope, so this does not misdirect the next triage: it covers ``fleet_loop``
    only. ``run_fleet_supervise``'s own ``load_layered_config(Path.cwd(), ...)``
    and the HEAD-drift and self-deploy steps that run *before* ``fleet_loop`` are
    still unexercised, and #590 could live in any of them.
    """
    repo = _make_repo(tmp_path, "anchor", api_worker=None)
    mock_load_registry.return_value = {
        "repos": {
            "owner/anchor": {
                "repo_root": str(repo),
                "config_path": "orchestrator.config.yaml",
                "state_dir": str(repo / ".var" / "charlie-work"),
            }
        }
    }
    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths
    mock_app = MagicMock()
    mock_app.loop.return_value = CommandResult(True, "ok", {})
    mock_app_class.return_value = mock_app
    mock_run_allocation_pass.return_value = AllocationPassResult(
        ok=True,
        plan=AllocationPlan(budget=8, budget_reason="configured", targets=(), changes=()),
        notes=(),
    )

    fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=_allocation_config(enabled=True, managed_root="C:/actions-runners"),
        repos=None,
        limit=1,
        merge=False,
        dry_run=False,
        work_only=False,
    )

    from ci_fleet.charlie_work_adapter import UNATTENDED_ALLOCATION_SOURCE

    mock_run_allocation_pass.assert_called_once()
    assert mock_run_allocation_pass.call_args.kwargs["dry_run"] is False
    # The daemon must identify itself as the unattended writer: the doctor probe
    # accepts only this value as evidence that allocation runs without an operator.
    assert mock_run_allocation_pass.call_args.kwargs["source"] == UNATTENDED_ALLOCATION_SOURCE


@patch("charlie_work.fleet_dispatch.provision_runner")
@patch("charlie_work.fleet_dispatch.decide_autoscale")
@patch("charlie_work.fleet_dispatch.is_pool_idle_for_minutes")
@patch("charlie_work.fleet_dispatch.is_in_cooldown")
@patch("charlie_work.fleet_dispatch.observe_runner_pool")
@patch("charlie_work.fleet_dispatch.count_fleet_runners")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch._load_registry")
def test_autoscale_prologue_up_forwards_affinity_knobs(
    mock_load_registry: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_gh_class: MagicMock,
    mock_count_fleet_runners: MagicMock,
    mock_observe_runner_pool: MagicMock,
    mock_is_in_cooldown: MagicMock,
    mock_is_pool_idle: MagicMock,
    mock_decide_autoscale: MagicMock,
    mock_provision_runner: MagicMock,
    tmp_path: Path,
) -> None:
    """The fleet-wide autoscale-up call site forwards runner_allocation's knobs.

    Companion to ci_runners #92: provision_runner grew keyword-only
    reserved_threads/threads_per_slot, but this call site (distinct from
    cli.py's ``runners autoscale``) was independently inert until it forwarded
    them too. Pins that the values come from the representative repo's
    config.runner_allocation section -- never hardcoded, never left at the
    off default -- and reach provision_runner unchanged.
    """
    repo = _make_repo(tmp_path, "anchor", api_worker=None)
    mock_load_registry.return_value = {
        "repos": {
            "owner/anchor": {
                "repo_root": str(repo),
                "config_path": "orchestrator.config.yaml",
                "state_dir": str(repo / ".var" / "charlie-work"),
            }
        }
    }
    config = OrchestratorConfig(
        runner_scaling=RunnerScalingConfig(enabled=True, managed_root=str(tmp_path)),
        runner_allocation=RunnerAllocationConfig(reserved_threads=4, threads_per_slot=6),
    )
    mock_load_layered_config.return_value = config
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths
    mock_count_fleet_runners.return_value = (1, 0, [])
    mock_is_in_cooldown.return_value = False
    mock_is_pool_idle.return_value = False
    mock_decide_autoscale.return_value = ScaleDecision(action=ScaleAction.UP, count=1, reason="t")
    mock_provision_runner.return_value = MagicMock(ok=True, runner_name="jc-1")

    global_config = MagicMock()
    global_config.runners.fleet_autoscale_prologue = True
    global_config.runner_scaling.enabled = True

    _run_fleet_autoscale_prologue(str(tmp_path / "fleet"), global_config, False)

    mock_provision_runner.assert_called_once()
    _, kwargs = mock_provision_runner.call_args
    assert kwargs["reserved_threads"] == 4
    assert kwargs["threads_per_slot"] == 6


def test_digest_stays_quiet_on_a_converged_allocation_pass() -> None:
    """A healthy rebalance must not put an entry in every 5-minute digest.

    The prologue emits `runner_allocation` when anything moved OR any note was
    produced, and the notes include standing advisory conditions that persist as long
    as the condition does. Verified against this host's events.db: every recorded pass
    carried such a note while moving zero slots. So routing this event through the
    generic fallback would render a near-identical attention entry on every pass.

    Precise about what this does and does not assert. It pins the *entry*: none is
    rendered for a converged pass. The emission-level guarantee -- that `emit_digest`
    is not called at all on such a pass -- is covered by
    ``test_fleet_loop_converged_pass_does_not_emit_digest`` (issue #610), which drives
    the full ``fleet_loop`` notify gate. This test stays at the builder level so a
    regression in the routing (re-introducing a fallback for ``runner_allocation``)
    is caught here even if the gate test's mocks mask it.
    """
    from charlie_work.fleet_dispatch import _build_fleet_attention_digest

    converged = [
        {
            "repo_key": "fleet",
            "type": "runner_allocation",
            "started": 0,
            "parked": 0,
            "budget": 8,
            "notes": ["Senkichi/job-cannon: holding 4 surplus slot(s) - slack for 0/3 pass(es)"],
            "dry_run": False,
        }
    ]
    digest = _build_fleet_attention_digest(converged)
    assert digest.transitions == ()


@patch("charlie_work.fleet_dispatch.emit_digest")
@patch("charlie_work.fleet_dispatch.run_allocation_pass")
@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_converged_pass_does_not_emit_digest(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    mock_run_allocation_pass: MagicMock,
    mock_emit_digest: MagicMock,
    tmp_path: Path,
) -> None:
    """Issue #610: a converged pass must not call emit_digest at all.

    A converged allocation pass emits a ``runner_allocation`` event (the
    prologue fires it whenever anything moved *or any note was produced*, and
    standing advisory notes persist for as long as the condition does). That
    event is routed to an explicit ``continue`` in
    ``_build_fleet_attention_digest``, so ``attention_events`` is non-empty
    while ``transitions`` is empty.

    Note: #669's inner ``if attention_digest.transitions:`` gate already
    prevents ``emit_digest`` being called with ``transitions=()`` on this
    scenario — so this test would pass identically without this PR's outer
    gate change. It pins the inner gate's behavior on the
    converged-allocation-note shape. This PR's actual behavior change (the
    outer ``and attention_events`` removal) is exercised by
    ``test_fleet_loop_empty_events_still_builds_digest_when_notify_on``.

    This drives ``fleet_loop`` unmocked (only the prologue's
    ``run_allocation_pass`` and the per-repo ``OrchestratorApp`` are patched)
    so the gate at the emission site is the thing under test. The per-repo
    loop returns an empty ``CommandResult.data`` so no per-repo attention
    events are produced -- the only event is the converged
    ``runner_allocation``.
    """
    from dataclasses import replace

    from charlie_work.config import NotifyConfig

    repo = _make_repo(tmp_path, "anchor", api_worker=None)
    mock_load_registry.return_value = {
        "repos": {
            "owner/anchor": {
                "repo_root": str(repo),
                "config_path": "orchestrator.config.yaml",
                "state_dir": str(repo / ".var" / "charlie-work"),
            }
        }
    }
    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths
    mock_app = MagicMock()
    mock_app.loop.return_value = CommandResult(True, "ok", {})
    mock_app_class.return_value = mock_app
    # Converged: nothing moved, but a standing advisory note is present -- the
    # exact shape every recorded pass on this host had (verified against
    # events.db). The prologue emits runner_allocation because notes is non-empty.
    mock_run_allocation_pass.return_value = AllocationPassResult(
        ok=True,
        plan=AllocationPlan(budget=8, budget_reason="configured", targets=(), changes=()),
        notes=("Senkichi/job-cannon: holding 4 surplus slot(s) - slack for 0/3 pass(es)",),
    )

    cfg = replace(
        _allocation_config(enabled=True, managed_root="C:/actions-runners"),
        notify=NotifyConfig(
            enabled=True,
            sink="file",
            file_path=str(tmp_path / "digest.jsonl"),
        ),
    )

    fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=cfg,
        repos=None,
        limit=1,
        merge=False,
        dry_run=False,
        work_only=False,
    )

    # The raw event list is non-empty (runner_allocation), but every event
    # hit an explicit continue, so transitions is empty and the inner
    # ``if attention_digest.transitions:`` gate (#669) blocks emit_digest.
    # This pins that inner gate on the converged-allocation-note shape; the
    # outer-gate removal this PR makes is covered by the empty-events test.
    mock_emit_digest.assert_not_called()


@patch("charlie_work.fleet_dispatch.emit_digest")
@patch("charlie_work.fleet_dispatch.run_allocation_pass")
@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_empty_events_still_builds_digest_when_notify_on(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    mock_run_allocation_pass: MagicMock,
    mock_emit_digest: MagicMock,
    tmp_path: Path,
) -> None:
    """Issue #610: the genuinely new path this PR opens -- empty attention_events.

    Pre-PR the outer gate was ``notify_config.enabled and attention_events``,
    so a pass with *zero* attention events (no prologue event, no per-repo
    events) skipped the whole digest-build block: ``_build_fleet_attention_digest``
    never ran and the fleet health-state sidecar was never written. This PR
    drops ``and attention_events`` so the digest is built whenever notify is on.

    #669's inner ``if attention_digest.transitions:`` gate already prevents
    ``emit_digest`` being called with ``transitions=()`` on a converged pass
    (the non-empty-events case) -- so this PR's value is *not* fixing a
    currently-reproducing empty-envelope emission. Its value is removing the
    redundant, contradictory outer raw-list test. This test exercises the one
    behavior the diff actually changes: with ``attention_events == []`` and
    notify on, the digest-build / health-state-write path now runs (the
    sidecar file appears on disk), ``emit_digest`` is still not called
    (transitions empty, #669's inner guard), and ``digest["emitted"]`` stays
    ``False``.

    Drives ``fleet_loop`` with ``run_allocation_pass`` returning a fully
    converged result with empty notes (so the prologue emits no event) and the
    per-repo ``OrchestratorApp.loop`` returning empty data (so no per-repo
    events). The only thing under test is the outer gate.
    """
    from dataclasses import replace

    from charlie_work.config import NotifyConfig
    from charlie_work.fleet_dispatch import _fleet_health_state_path

    repo = _make_repo(tmp_path, "anchor", api_worker=None)
    mock_load_registry.return_value = {
        "repos": {
            "owner/anchor": {
                "repo_root": str(repo),
                "config_path": "orchestrator.config.yaml",
                "state_dir": str(repo / ".var" / "charlie-work"),
            }
        }
    }
    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths
    mock_app = MagicMock()
    mock_app.loop.return_value = CommandResult(True, "ok", {})
    mock_app_class.return_value = mock_app
    # Fully converged and quiet: nothing moved, no notes -- the prologue
    # emits no runner_allocation event (started/parked/notes all falsy), so
    # attention_events stays empty.
    mock_run_allocation_pass.return_value = AllocationPassResult(
        ok=True,
        plan=AllocationPlan(budget=8, budget_reason="configured", targets=(), changes=()),
        notes=(),
    )

    cfg = replace(
        _allocation_config(enabled=True, managed_root="C:/actions-runners"),
        notify=NotifyConfig(
            enabled=True,
            sink="file",
            file_path=str(tmp_path / "digest.jsonl"),
        ),
    )

    fleet_dir = tmp_path / "fleet"
    result = fleet_loop(
        fleet_dir_override=str(fleet_dir),
        global_config=cfg,
        repos=None,
        limit=1,
        merge=False,
        dry_run=False,
        work_only=False,
    )

    # The new path: with attention_events empty and notify on, the digest is
    # still built -- the health-state sidecar appears on disk. Pre-PR the
    # outer ``and attention_events`` gate skipped this block entirely, so the
    # file would not exist. This is the discriminating assertion for the diff.
    health_state = _fleet_health_state_path(str(fleet_dir))
    assert health_state.exists(), "digest-build path did not run on empty events"

    # Emission is still gated on the built digest's transitions (#669's
    # inner guard), so no envelope is written and emitted stays False.
    mock_emit_digest.assert_not_called()
    assert result.data["digest"]["emitted"] is False


def test_digest_still_surfaces_allocation_failures() -> None:
    """Dropping the success event must not also drop the errors and skips."""
    from charlie_work.fleet_dispatch import _build_fleet_attention_digest

    failures = [
        {"repo_key": "fleet", "type": "runner_allocation_error", "error": "managed_root missing"},
        {
            "repo_key": "fleet",
            "type": "runner_allocation_slot_error",
            "runner": "cw-2",
            "action": "park",
            "message": "still busy",
        },
        {"repo_key": "fleet", "type": "runner_allocation_skipped", "reason": "no registry anchor"},
    ]
    digest = _build_fleet_attention_digest(failures)
    assert len(digest.transitions) == 3
    healths = {e.health for e in digest.transitions}
    # Both error types must reach the desktop-eligible severity set, not just the log.
    assert "ERROR" in healths
    reasons = " ".join(str(e.last_log_line) for e in digest.transitions)
    assert "managed_root missing" in reasons
    assert "no registry anchor" in reasons


def test_filter_fleet_health_transitions_dedups_repeated_error(tmp_path: Path) -> None:
    """Issue #554: a persistent ERROR must not re-fire with previous_health:null every pass."""
    from dataclasses import replace

    from charlie_work.fleet_dispatch import (
        _fleet_health_state_path,
        _filter_fleet_health_transitions,
    )
    from charlie_work.notify import AttentionEntry

    state_file = _fleet_health_state_path(str(tmp_path / "fleet"))
    entry = AttentionEntry(
        issue_number=42,
        adapter_kind="owner/repo",
        health="ERROR",
        previous_health=None,
        last_log_line="failed to launch claude: OSError",
        pid=None,
    )

    # Pass 1: null -> ERROR is a real transition; emits with previous_health=None.
    first = _filter_fleet_health_transitions([entry], state_file)
    assert len(first) == 1
    assert first[0].health == "ERROR"
    assert first[0].previous_health is None

    # Pass 2: ERROR -> ERROR is not a transition; nothing emits.
    second = _filter_fleet_health_transitions([entry], state_file)
    assert second == []

    # Pass 3: ERROR -> STALLED is a real transition; emits with previous_health=ERROR.
    stalled = replace(entry, health="STALLED")
    third = _filter_fleet_health_transitions([stalled], state_file)
    assert len(third) == 1
    assert third[0].health == "STALLED"
    assert third[0].previous_health == "ERROR"


def test_build_fleet_attention_digest_stateful_skips_repeats(tmp_path: Path) -> None:
    """Issue #554: _build_fleet_attention_digest with state_file emits only on real transitions."""
    from charlie_work.fleet_dispatch import _build_fleet_attention_digest, _fleet_health_state_path

    state_file = _fleet_health_state_path(str(tmp_path / "fleet"))
    events = [
        {
            "repo_key": "owner/repo",
            "type": "error",
            "issue_number": 100,
            "error": "PR #100 review failed: timeout",
        }
    ]

    # First pass: null -> ERROR transition emits.
    digest1 = _build_fleet_attention_digest(events, state_file=state_file)
    assert len(digest1.transitions) == 1
    assert digest1.transitions[0].health == "ERROR"
    assert digest1.transitions[0].previous_health is None

    # Second pass: same ERROR, no transition — digest is empty.
    digest2 = _build_fleet_attention_digest(events, state_file=state_file)
    assert digest2.transitions == ()


@patch("charlie_work.fleet_dispatch.emit_digest")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_self_deploy_error_dedups_across_passes(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_emit_digest: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Issue #554: a persistent self-deploy ERROR emits once, not every supervisor pass."""
    from charlie_work.config import NotifyConfig

    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        ),
        notify=NotifyConfig(
            enabled=True,
            sink="file",
            file_path=str(tmp_path / "digest.jsonl"),
        ),
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()
    mock_lock.return_value = MagicMock()

    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=False,
            pulled=False,
            changed=False,
            synced=False,
            error="venv pth repair failed: Access is denied",
        )
    )
    monkeypatch.setattr("charlie_work.fleet_dispatch.self_deploy", deploy_mock)

    fc = _FakeClock(auto_advance=1.0)
    run_fleet_supervise(
        max_passes=2,
        clock=fc.now,
        sleep=fc.sleep,
        fleet_dir_override=str(tmp_path / "fleet"),
    )

    # Two passes, same persistent ERROR — emit_digest fires once (first pass).
    assert mock_emit_digest.call_count == 1
    digest = mock_emit_digest.call_args[0][1]
    assert len(digest.transitions) == 1
    assert digest.transitions[0].health == "ERROR"
    assert digest.transitions[0].previous_health is None


def test_build_fleet_attention_digest_stateful_keeps_review_verdict_heartbeat(
    tmp_path: Path,
) -> None:
    """PR #669 review: occurrence-style events must not be deduped by the
    stateful filter. ``review_verdict_recorded`` carries a constant ``OK``
    health string by construction; if the cross-pass dedup applied to it, the
    recorded-verdict heartbeat would collapse after the first occurrence per
    issue/repo and a silent 0%-recording-rate regression would go invisible.
    The fleet path in production goes through ``_build_fleet_attention_digest``
    with ``state_file`` set, so this locks in the filtered path (not just the
    unfiltered one covered by ``test_build_fleet_attention_digest_maps_review_verdict_events``).
    """
    from charlie_work.fleet_dispatch import _build_fleet_attention_digest, _fleet_health_state_path

    state_file = _fleet_health_state_path(str(tmp_path / "fleet"))
    events = [
        {
            "repo_key": "owner/repo1",
            "type": "review_verdict_recorded",
            "issue_number": 10,
            "pr": 100,
            "decision": "approved",
        }
    ]

    # Two passes, identical recorded-verdict event — both must emit. The
    # heartbeat stays visible; the baseline sidecar is not consulted for
    # occurrence-style entries.
    digest1 = _build_fleet_attention_digest(events, state_file=state_file)
    assert len(digest1.transitions) == 1
    assert digest1.transitions[0].health == "OK"
    assert digest1.transitions[0].last_log_line == "approved recorded for PR 100"

    digest2 = _build_fleet_attention_digest(events, state_file=state_file)
    assert len(digest2.transitions) == 1
    assert digest2.transitions[0].health == "OK"


def test_build_fleet_attention_digest_stateful_keeps_review_verdict_missed_every_pass(
    tmp_path: Path,
) -> None:
    """PR #669 review: ``review_verdict_missed`` has health ``ERROR`` (the same
    string as a persistent worker error) but is an occurrence-style event. The
    dedup must be scoped by entry type, not by health string, or repeat
    missed-verdict signals would be silently dropped after the first one per
    issue/repo. Locks in the exception for the ERROR-health occurrence case.
    """
    from charlie_work.fleet_dispatch import _build_fleet_attention_digest, _fleet_health_state_path

    state_file = _fleet_health_state_path(str(tmp_path / "fleet"))
    events = [
        {
            "repo_key": "owner/repo1",
            "type": "review_verdict_missed",
            "issue_number": 11,
            "pr": 101,
            "reason": "no parseable verdict",
        }
    ]

    digest1 = _build_fleet_attention_digest(events, state_file=state_file)
    assert len(digest1.transitions) == 1
    assert digest1.transitions[0].health == "ERROR"
    assert digest1.transitions[0].last_log_line == "no parseable verdict"

    # Same missed verdict next pass — still emits despite health == "ERROR",
    # because the event type is occurrence-style and bypasses the baseline.
    digest2 = _build_fleet_attention_digest(events, state_file=state_file)
    assert len(digest2.transitions) == 1
    assert digest2.transitions[0].health == "ERROR"


def test_build_fleet_attention_digest_stateful_mixed_persistent_and_occurrence(
    tmp_path: Path,
) -> None:
    """PR #669 review: a persistent ``error`` and an occurrence
    ``review_verdict_recorded`` for the same issue/repo share the dedup key
    ``adapter_kind:issue_number``. The persistent entry must still be deduped
    across passes while the occurrence entry keeps emitting, and the occurrence
    entry must not poison the persistent baseline (so a later persistent
    transition is not masked).
    """
    from charlie_work.fleet_dispatch import _build_fleet_attention_digest, _fleet_health_state_path

    state_file = _fleet_health_state_path(str(tmp_path / "fleet"))
    persistent_event = {
        "repo_key": "owner/repo",
        "type": "error",
        "issue_number": 42,
        "error": "launch failed: OSError",
    }
    occurrence_event = {
        "repo_key": "owner/repo",
        "type": "review_verdict_recorded",
        "issue_number": 42,
        "pr": 420,
        "decision": "approved",
    }

    # Pass 1: persistent null->ERROR emits; occurrence OK emits alongside.
    digest1 = _build_fleet_attention_digest(
        [persistent_event, occurrence_event], state_file=state_file
    )
    healths = [t.health for t in digest1.transitions]
    assert healths == ["ERROR", "OK"]

    # Pass 2: persistent ERROR->ERROR is deduped away; occurrence OK still emits.
    digest2 = _build_fleet_attention_digest(
        [persistent_event, occurrence_event], state_file=state_file
    )
    assert [t.health for t in digest2.transitions] == ["OK"]

    # Pass 3: persistent ERROR->STALLED is a real transition and must emit even
    # though the occurrence event sat between them every pass — the occurrence
    # entry never wrote the shared baseline key.
    stalled_event = {
        "repo_key": "owner/repo",
        "type": "stalled",
        "issue_number": 42,
        "reason": "no progress for 30m",
    }
    digest3 = _build_fleet_attention_digest(
        [stalled_event, occurrence_event], state_file=state_file
    )
    healths3 = [t.health for t in digest3.transitions]
    assert healths3 == ["STALLED", "OK"]
    assert digest3.transitions[0].previous_health == "ERROR"


# ---------------------------------------------------------------------------
# Issue #817: fleet health latch (producer never fed a recovery observation)
# ---------------------------------------------------------------------------


@patch("charlie_work.fleet_dispatch.emit_digest")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_self_deploy_failure_success_failure_emits_three_transitions(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_emit_digest: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Issue #817 AC1: failure -> success -> failure must emit three digest
    entries, not one.

    Before this fix, the producer only ever constructed an AttentionEntry
    for a *failed* self_deploy (item 1's defect): the recovery pass built no
    entry at all, so the baseline sidecar stayed latched at ERROR from the
    first failure onward. ``_filter_fleet_health_transitions`` itself was
    already a correct edge-detector -- the second failure would read
    ERROR -> ERROR against that latched baseline and be suppressed as a
    non-transition, even though a real recovery happened in between.
    """
    from charlie_work.config import NotifyConfig
    from charlie_work.fleet_dispatch import _fleet_health_state_path
    from charlie_work.fleet_dispatch import _load_fleet_health_state as _load_state

    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        ),
        notify=NotifyConfig(
            enabled=True,
            sink="file",
            file_path=str(tmp_path / "digest.jsonl"),
        ),
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()
    mock_lock.return_value = MagicMock()

    deploy_mock = MagicMock(
        side_effect=[
            SelfDeployResult(
                ok=False,
                pulled=False,
                changed=False,
                synced=False,
                error="fatal: Not possible to fast-forward, aborting.",
            ),
            SelfDeployResult(
                ok=True,
                pulled=True,
                changed=False,
                synced=False,
                from_sha="abc123",
                to_sha="abc123",
                message="already up to date",
            ),
            SelfDeployResult(
                ok=False,
                pulled=False,
                changed=False,
                synced=False,
                error="fatal: Not possible to fast-forward, aborting.",
            ),
        ]
    )
    monkeypatch.setattr("charlie_work.fleet_dispatch.self_deploy", deploy_mock)

    # from_sha == to_sha on the success pass deliberately: a real HEAD move
    # would trigger the supervisor's separate restart-for-fresh-code exit
    # (see test_run_fleet_supervise_restarts_when_self_deploy_moves_head),
    # which would end the loop after pass 2 and never reach the third
    # failure this test needs to observe.
    fc = _FakeClock(auto_advance=1.0)
    run_fleet_supervise(
        max_passes=3,
        clock=fc.now,
        sleep=fc.sleep,
        fleet_dir_override=str(tmp_path / "fleet"),
    )

    assert mock_emit_digest.call_count == 3
    healths = [call.args[1].transitions[0].health for call in mock_emit_digest.call_args_list]
    assert healths == ["ERROR", "OK", "ERROR"]
    previous = [
        call.args[1].transitions[0].previous_health for call in mock_emit_digest.call_args_list
    ]
    assert previous == [None, "ERROR", "OK"]

    # Final persisted baseline reflects the third (failed) pass.
    state_file = _fleet_health_state_path(str(tmp_path / "fleet"))
    assert _load_state(state_file) == {"self-deploy:-1": "ERROR"}


@patch("charlie_work.fleet_dispatch.emit_digest")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_self_deploy_success_clears_error_baseline(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_emit_digest: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Issue #817 AC2: after a failure -> success sequence, the *persisted*
    baseline sidecar itself reads back the healthy value -- not just the
    digest object returned in-process for that pass -- proving state
    genuinely moved off the ERROR latch. This is the fact AC1's third
    (failure) emission depends on: if the sidecar file did not actually
    change, the in-memory digest assertion alone would not distinguish a
    real fix from one that merely happens to return the right object once.
    """
    from charlie_work.config import NotifyConfig
    from charlie_work.fleet_dispatch import _fleet_health_state_path
    from charlie_work.fleet_dispatch import _load_fleet_health_state as _load_state

    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        ),
        notify=NotifyConfig(
            enabled=True,
            sink="file",
            file_path=str(tmp_path / "digest.jsonl"),
        ),
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()
    mock_lock.return_value = MagicMock()

    deploy_mock = MagicMock(
        side_effect=[
            SelfDeployResult(
                ok=False, pulled=False, changed=False, synced=False, error="pull failed"
            ),
            SelfDeployResult(
                ok=True,
                pulled=True,
                changed=False,
                synced=False,
                from_sha="abc123",
                to_sha="abc123",
                message="already up to date",
            ),
        ]
    )
    monkeypatch.setattr("charlie_work.fleet_dispatch.self_deploy", deploy_mock)

    fc = _FakeClock(auto_advance=1.0)
    run_fleet_supervise(
        max_passes=2,
        clock=fc.now,
        sleep=fc.sleep,
        fleet_dir_override=str(tmp_path / "fleet"),
    )

    state_file = _fleet_health_state_path(str(tmp_path / "fleet"))
    assert _load_state(state_file) == {"self-deploy:-1": "OK"}


# ---------------------------------------------------------------------------
# Zero-repo-pass streak (issue #855, the general shape behind #851)
# ---------------------------------------------------------------------------


def test_fleet_has_configured_repos_true_with_registered_repo(tmp_path: Path) -> None:
    fleet_dir = tmp_path / "fleet"
    _make_fleet_json(tmp_path, fleet_dir, {"owner/repo": {"repo_root": str(tmp_path / "repo")}})
    assert _fleet_has_configured_repos(str(fleet_dir), None) is True


def test_fleet_has_configured_repos_false_with_empty_registry(tmp_path: Path) -> None:
    fleet_dir = tmp_path / "fleet"
    _make_fleet_json(tmp_path, fleet_dir, {})
    assert _fleet_has_configured_repos(str(fleet_dir), None) is False


def test_fleet_has_configured_repos_false_with_no_registry_file(tmp_path: Path) -> None:
    """A fleet.json that was never written (fresh host, nothing registered
    yet) must read the same as an explicitly empty registry."""
    assert _fleet_has_configured_repos(str(tmp_path / "never-written"), None) is False


@patch("charlie_work.fleet_dispatch.probe_fleet_watchdog")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_zero_pass_streak_replays_851_outage_shape(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_probe: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Issue #855 acceptance criterion 7: replay the #851 outage shape.

    N consecutive supervisor *process* restarts -- each modeled as a
    separate ``run_fleet_supervise()`` call, exactly like the Task Scheduler
    watchdog relaunching the process every cycle -- every one exiting via
    the self-deploy HEAD-moved break before ``fleet_loop`` ever runs: exit
    code 0 every cycle, ``repo_passes == 0`` every cycle (the log line the
    issue's evidence quotes: "1 pass(es) ... 0 repo pass(es)"), despite a
    repo being registered in the fleet. Exactly one
    ``supervisor_zero_pass_alarm`` must fire, at the cycle the persisted
    streak reaches the configured threshold (3) -- not one per restart.
    """
    from charlie_work.fleet_dispatch import WatchdogProbe

    mock_lock.return_value = MagicMock()
    # Issue #604: every cycle exits via the self-deploy HEAD-moved break,
    # which now probes the watchdog scheduled task. Mock it to ``armed=None``
    # (unknown) so every cycle stays hermetic -- no real ``schtasks``
    # subprocess call, no coupling to the live state of the
    # ``charlie-fleet-pass`` task -- and the alert path (which fires only on
    # a confirmed ``armed=False``) is not exercised here. The dedicated
    # watchdog-alert tests cover that path.
    mock_probe.return_value = WatchdogProbe(armed=None, detail="not probed (mocked)")

    fleet_dir = tmp_path / "fleet"
    isolated_root = tmp_path / "orchestrator-root"
    isolated_root.mkdir()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _make_fleet_json(tmp_path, fleet_dir, {"owner/repo": {"repo_root": str(repo_root)}})

    # Isolate orchestrator_root() so the streak counter and alarm event
    # land under an ephemeral tmp_path state dir, never the real checkout
    # this test suite runs from.
    monkeypatch.setattr("charlie_work.fleet_dispatch.orchestrator_root", lambda: isolated_root)

    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
            zero_pass_alarm=3,
        )
    )
    mock_load_config.return_value = cfg

    # Every self_deploy call reports a HEAD move -> run_fleet_supervise
    # exits (break) right after pass 1, before ever reaching fleet_loop this
    # process's lifetime.
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.self_deploy",
        lambda _repo_root, **_kwargs: SelfDeployResult(
            ok=True,
            pulled=True,
            changed=True,
            synced=False,
            # run_fleet_supervise's restart gate reads head_changed, NOT
            # from_sha != to_sha (#853). Without this the simulated HEAD move
            # is a no-op, the supervisor never exits for a watchdog restart,
            # and this test stops exercising the #851 shape it is named for.
            head_changed=True,
            from_sha="a" * 12,
            to_sha="b" * 12,
            message="updated and synced: " + "b" * 12,
        ),
    )

    state_path = layout.state_file_path(layout.default_state_root(isolated_root))

    for cycle in range(1, 4):
        fc = _FakeClock(auto_advance=1.0)
        result = run_fleet_supervise(
            fleet_dir_override=str(fleet_dir),
            max_passes=5,
            clock=fc.now,
            sleep=fc.sleep,
        )
        assert result.ok is True
        assert result.data["passes"] == 1
        assert result.data["total_repo_passes"] == 0
        # fleet_loop must never run -- every cycle exits before reaching it.
        assert mock_fleet_loop.call_count == 0

        alarms = query_events(state_path, kind="supervisor_zero_pass_alarm")
        if cycle < 3:
            assert alarms == [], f"alarm fired early at cycle {cycle}"
        else:
            assert len(alarms) == 1, f"expected exactly one alarm by cycle {cycle}"
            assert alarms[0]["payload"]["consecutive_zero_pass_cycles"] == 3


@patch("charlie_work.fleet_dispatch.probe_fleet_watchdog")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_zero_pass_streak_never_fires_with_empty_registry(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_probe: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Issue #855 acceptance criterion 4, exercised end to end: a fleet with
    zero registered repos never fires the alarm, no matter how many
    consecutive zero-repo-pass cycles it runs -- that is a configuration
    state, not an incident.
    """
    from charlie_work.fleet_dispatch import WatchdogProbe

    mock_lock.return_value = MagicMock()
    # Issue #604: every cycle exits via the self-deploy HEAD-moved break,
    # which now probes the watchdog scheduled task. Mock it to ``armed=None``
    # (unknown) so every cycle stays hermetic -- no real ``schtasks``
    # subprocess call, no coupling to the live state of the
    # ``charlie-fleet-pass`` task -- and the alert path (which fires only on
    # a confirmed ``armed=False``) is not exercised here. The dedicated
    # watchdog-alert tests cover that path.
    mock_probe.return_value = WatchdogProbe(armed=None, detail="not probed (mocked)")

    fleet_dir = tmp_path / "fleet"
    isolated_root = tmp_path / "orchestrator-root"
    isolated_root.mkdir()
    # Deliberately no _make_fleet_json call: the registry is empty.

    monkeypatch.setattr("charlie_work.fleet_dispatch.orchestrator_root", lambda: isolated_root)
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.self_deploy",
        lambda _repo_root, **_kwargs: SelfDeployResult(
            ok=True,
            pulled=True,
            changed=True,
            synced=False,
            # run_fleet_supervise's restart gate reads head_changed, NOT
            # from_sha != to_sha (#853). Without this the simulated HEAD move
            # is a no-op, the supervisor never exits for a watchdog restart,
            # and this test stops exercising the #851 shape it is named for.
            head_changed=True,
            from_sha="a" * 12,
            to_sha="b" * 12,
            message="updated and synced: " + "b" * 12,
        ),
    )

    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
            zero_pass_alarm=3,
        )
    )
    mock_load_config.return_value = cfg

    state_path = layout.state_file_path(layout.default_state_root(isolated_root))

    for _ in range(6):
        fc = _FakeClock(auto_advance=1.0)
        result = run_fleet_supervise(
            fleet_dir_override=str(fleet_dir),
            max_passes=5,
            clock=fc.now,
            sleep=fc.sleep,
        )
        assert result.ok is True
        assert result.data["total_repo_passes"] == 0

    assert mock_fleet_loop.call_count == 0
    assert query_events(state_path, kind="supervisor_zero_pass_alarm") == []


@patch("charlie_work.fleet_dispatch.probe_fleet_watchdog")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_zero_pass_streak_resets_after_repo_work(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_probe: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A cycle that actually reaches fleet_loop and performs repo work resets
    the streak to 0, so a later zero-pass streak has to build back up to the
    threshold instead of alarming immediately off carried-over count.
    """
    from charlie_work.fleet_dispatch import WatchdogProbe

    mock_lock.return_value = MagicMock()
    # Issue #604: the zero-repo-pass cycles exit via the self-deploy
    # HEAD-moved break, which now probes the watchdog scheduled task. Mock
    # it to ``armed=None`` (unknown) so every cycle stays hermetic -- no
    # real ``schtasks`` subprocess call, no coupling to the live state of
    # the ``charlie-fleet-pass`` task -- and the alert path (which fires
    # only on a confirmed ``armed=False``) is not exercised here. The
    # dedicated watchdog-alert tests cover that path.
    mock_probe.return_value = WatchdogProbe(armed=None, detail="not probed (mocked)")

    fleet_dir = tmp_path / "fleet"
    isolated_root = tmp_path / "orchestrator-root"
    isolated_root.mkdir()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _make_fleet_json(tmp_path, fleet_dir, {"owner/repo": {"repo_root": str(repo_root)}})

    monkeypatch.setattr("charlie_work.fleet_dispatch.orchestrator_root", lambda: isolated_root)

    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
            zero_pass_alarm=3,
        )
    )
    mock_load_config.return_value = cfg
    state_path = layout.state_file_path(layout.default_state_root(isolated_root))

    head_moved = SelfDeployResult(
        ok=True,
        pulled=True,
        changed=True,
        synced=False,
        # run_fleet_supervise's restart gate reads head_changed, NOT
        # from_sha != to_sha (#853). Without this the simulated HEAD move
        # is a no-op, the supervisor never exits for a watchdog restart,
        # and this test stops exercising the #851 shape it is named for.
        head_changed=True,
        from_sha="a" * 12,
        to_sha="b" * 12,
        message="updated and synced: " + "b" * 12,
    )
    no_op = SelfDeployResult(
        ok=True, pulled=True, changed=False, synced=False, message="already up to date"
    )

    # Two zero-repo-pass cycles (streak -> 2, below threshold 3).
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.self_deploy", lambda _repo_root, **_kwargs: head_moved
    )
    for _ in range(2):
        fc = _FakeClock(auto_advance=1.0)
        run_fleet_supervise(
            fleet_dir_override=str(fleet_dir), max_passes=5, clock=fc.now, sleep=fc.sleep
        )
    assert query_events(state_path, kind="supervisor_zero_pass_alarm") == []

    # A cycle that actually performs repo work: self_deploy is a no-op, so
    # the loop proceeds to fleet_loop, which reports one repo processed.
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.self_deploy", lambda _repo_root, **_kwargs: no_op
    )
    mock_fleet_loop.return_value = _drained_fleet_result()
    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(
        fleet_dir_override=str(fleet_dir), max_passes=1, clock=fc.now, sleep=fc.sleep
    )
    assert result.data["total_repo_passes"] == 1

    # Two more zero-repo-pass cycles: if the streak had not reset, this
    # would already be 4 (past threshold 3) and would have alarmed already;
    # since it reset to 0, two more cycles land at 2 -- still below 3.
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.self_deploy", lambda _repo_root, **_kwargs: head_moved
    )
    for _ in range(2):
        fc = _FakeClock(auto_advance=1.0)
        run_fleet_supervise(
            fleet_dir_override=str(fleet_dir), max_passes=5, clock=fc.now, sleep=fc.sleep
        )
    assert query_events(state_path, kind="supervisor_zero_pass_alarm") == []


def test_filter_fleet_health_transitions_reconciles_stale_key_when_repo_observed(
    tmp_path: Path,
) -> None:
    """Issue #817 item 2/AC5: a stale ERROR baseline for an issue that is
    healthy again (produces no unhealthy event this pass) is cleared once
    its repo's lane is confirmed observed, instead of latching forever.

    This is also the drain mechanism for the pre-existing 34 latched keys
    (issue #817's diagnosis): each key's repo needs exactly one observed
    pass with no matching unhealthy entry to clear it, after which the next
    real failure emits with ``previous_health: null`` again instead of
    staying permanently suppressed by a baseline that could never move
    except deeper into an unhealthy value.
    """
    from charlie_work.fleet_dispatch import (
        _fleet_health_state_path,
        _filter_fleet_health_transitions,
        _load_fleet_health_state,
    )
    from charlie_work.notify import AttentionEntry

    state_file = _fleet_health_state_path(str(tmp_path / "fleet"))
    entry = AttentionEntry(
        issue_number=42,
        adapter_kind="owner/repo",
        health="ERROR",
        previous_health=None,
        last_log_line="failed to launch claude: OSError",
        pid=None,
    )

    # Pass 1: latch ERROR.
    first = _filter_fleet_health_transitions([entry], state_file)
    assert len(first) == 1
    assert _load_fleet_health_state(state_file) == {"owner/repo:42": "ERROR"}

    # Pass 2: issue #42 recovered -- no unhealthy entry for it this pass, but
    # its repo's lane still ran to completion (observed_repo_keys includes
    # "owner/repo"). The stale key must be cleared, not re-emitted as a
    # synthetic recovery entry.
    second = _filter_fleet_health_transitions(
        [], state_file, observed_repo_keys=frozenset({"owner/repo"})
    )
    assert second == []
    assert _load_fleet_health_state(state_file) == {}

    # Pass 3: the same issue fails again. Because the baseline was cleared,
    # this is a fresh null -> ERROR transition, not a suppressed repeat.
    third = _filter_fleet_health_transitions([entry], state_file)
    assert len(third) == 1
    assert third[0].previous_health is None


def test_filter_fleet_health_transitions_leaves_unobserved_repo_keys_untouched(
    tmp_path: Path,
) -> None:
    """A repo whose lane did NOT run this pass (missing repo_root, lock held,
    unhandled exception) must not have its stale keys reconciled away --
    absence of a check is not evidence of health. Only keys under repos
    present in ``observed_repo_keys`` are eligible for clearing.
    """
    from charlie_work.fleet_dispatch import (
        _fleet_health_state_path,
        _filter_fleet_health_transitions,
        _load_fleet_health_state,
    )
    from charlie_work.notify import AttentionEntry

    state_file = _fleet_health_state_path(str(tmp_path / "fleet"))
    entry = AttentionEntry(
        issue_number=7,
        adapter_kind="owner/skipped-repo",
        health="ERROR",
        previous_health=None,
        last_log_line="stalled",
        pid=None,
    )
    first = _filter_fleet_health_transitions([entry], state_file)
    assert len(first) == 1

    # A different repo's lane ran this pass; "owner/skipped-repo" did not.
    second = _filter_fleet_health_transitions(
        [], state_file, observed_repo_keys=frozenset({"owner/other-repo"})
    )
    assert second == []
    assert _load_fleet_health_state(state_file) == {"owner/skipped-repo:7": "ERROR"}


def test_filter_fleet_health_transitions_self_deploy_key_survives_repo_reconciliation(
    tmp_path: Path,
) -> None:
    """Item 1's ``self-deploy:-1`` baseline key and item 2's per-repo
    reconciliation share the same sidecar file within one supervisor pass
    (self_deploy emits first, then fleet_loop's digest reconciles). The
    ``self-deploy`` adapter_kind is a fixed literal, never a real repo's
    ``name_with_owner``, so it can never appear in ``observed_repo_keys`` and
    must never be cleared by issue-health reconciliation -- confirmed here by
    a test rather than left as an unverified by-construction claim.
    """
    from charlie_work.fleet_dispatch import (
        _fleet_health_state_path,
        _filter_fleet_health_transitions,
        _load_fleet_health_state,
    )
    from charlie_work.notify import AttentionEntry

    state_file = _fleet_health_state_path(str(tmp_path / "fleet"))
    self_deploy_entry = AttentionEntry(
        issue_number=-1,
        adapter_kind="self-deploy",
        health="ERROR",
        previous_health=None,
        last_log_line="pull failed",
        pid=None,
    )
    _filter_fleet_health_transitions([self_deploy_entry], state_file)
    assert _load_fleet_health_state(state_file) == {"self-deploy:-1": "ERROR"}

    # A real repo's lane runs and reconciles this pass; self-deploy's key
    # must not be touched even though no self-deploy entry was emitted.
    result = _filter_fleet_health_transitions(
        [], state_file, observed_repo_keys=frozenset({"owner/repo"})
    )
    assert result == []
    assert _load_fleet_health_state(state_file) == {"self-deploy:-1": "ERROR"}


def test_build_fleet_attention_digest_observed_repo_keys_reconciles_stale_error(
    tmp_path: Path,
) -> None:
    """Issue #817 item 2: ``_build_fleet_attention_digest`` forwards
    ``observed_repo_keys`` through to ``_filter_fleet_health_transitions``,
    so ``fleet_loop``'s per-pass reconciliation actually reaches the
    baseline sidecar rather than being silently dropped somewhere in
    between.
    """
    from charlie_work.fleet_dispatch import (
        _build_fleet_attention_digest,
        _fleet_health_state_path,
        _load_fleet_health_state,
    )

    state_file = _fleet_health_state_path(str(tmp_path / "fleet"))
    events = [
        {
            "repo_key": "owner/repo",
            "type": "error",
            "issue_number": 100,
            "error": "PR #100 review failed: timeout",
        }
    ]
    digest1 = _build_fleet_attention_digest(events, state_file=state_file)
    assert len(digest1.transitions) == 1
    assert _load_fleet_health_state(state_file) == {"owner/repo:100": "ERROR"}

    # Next pass: issue #100 is healthy again (no error event for it), but
    # the repo's lane ran to completion -- observed_repo_keys reconciles the
    # stale key away.
    digest2 = _build_fleet_attention_digest(
        [], state_file=state_file, observed_repo_keys=frozenset({"owner/repo"})
    )
    assert digest2.transitions == ()
    assert _load_fleet_health_state(state_file) == {}
