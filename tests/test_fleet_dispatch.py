from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from charlie_work.config import (
    OrchestratorConfig,
    RunnerAllocationConfig,
    RunnerScalingConfig,
    RuntimeConfig,
    SupervisorConfig,
)
from charlie_work.fleet_dispatch import (
    ApiWorkerFleetReport,
    _build_fleet_attention_digest,
    _extract_attention_events,
    _is_fleet_pass_active,
    _run_fleet_allocation_prologue,
    _select_repos,
    compute_api_worker_fleet_report,
    fleet_loop,
    run_fleet_supervise,
)
from charlie_work.fleet_registry import count_fleet_runners
from charlie_work.runner_allocation import (
    AllocationPlan,
    SlotAction,
    SlotChange,
    SlotChangeResult,
)
from charlie_work.runner_allocation_pass import AllocationPassResult
from charlie_work.supervise import SelfDeployResult
from charlie_work.github import GitHubError
from charlie_work.workflow import CommandResult


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
def _patch_self_deploy_for_fleet_tests(monkeypatch: Any) -> None:
    """Self-deploy hits the real git/uv CLI; keep fleet supervisor unit tests hermetic."""
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

    # Verify overall result is False (one repo failed)
    assert result.ok is False


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


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_restarts_when_self_deploy_moves_head(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
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
            synced=True,
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
def test_run_fleet_supervise_restarts_on_external_head_drift(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """HEAD moved externally (operator pull, another process) triggers restart.

    self_deploy reports "already up to date" because HEAD was already at the
    new commit when the daemon's own git pull ran. Without an independent
    startup-vs-current HEAD comparison, the daemon would run stale code
    forever (observed 2026-07-23: ~90 minutes of ConfigError crashes).
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

    result = run_fleet_supervise(max_passes=1)

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

    result = run_fleet_supervise(max_passes=1)

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
    """The report reads spend from the representative (enabled) repo's ledger."""
    from datetime import UTC, datetime

    fleet_dir = tmp_path / "fleet"
    repo0 = _make_repo(tmp_path, "repo0", api_worker=_API_WORKER_YAML.format(enabled="true"))
    repo1 = _make_repo(tmp_path, "repo1", api_worker=_API_WORKER_YAML.format(enabled="false"))
    state_dir0 = repo0 / ".var" / "charlie-work"

    # Write a ledger with today's spend.
    today = datetime.now(UTC).strftime("%Y-%m-%d")
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

    report = compute_api_worker_fleet_report(fleet_dir_override=str(fleet_dir))

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

    assert events == []
    pass_mock.assert_not_called()


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
    assert kwargs["state_path"] == repo / ".var" / "charlie-work" / "state.json"
    assert kwargs["dry_run"] is True

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

    assert events == []
    assert any("no runner_allocation" in record.message for record in caplog.records)
