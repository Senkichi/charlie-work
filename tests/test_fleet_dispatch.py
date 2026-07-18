from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from charlie_work.config import OrchestratorConfig, RuntimeConfig, SupervisorConfig
from charlie_work.fleet_dispatch import (
    _extract_attention_events,
    _is_fleet_pass_active,
    _select_repos,
    fleet_loop,
    run_fleet_supervise,
)
from charlie_work.fleet_registry import count_fleet_runners
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
        supervisor=SupervisorConfig(poll_interval_seconds=5, active_cooldown_seconds=7)
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
        supervisor=SupervisorConfig(poll_interval_seconds=5, active_cooldown_seconds=7)
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
        supervisor=SupervisorConfig(poll_interval_seconds=5)
    )
    mock_fleet_loop.side_effect = [_drained_fleet_result(), KeyboardInterrupt]

    result = run_fleet_supervise(max_passes=5, clock=_FakeClock().now, sleep=_FakeClock().sleep)

    assert result.ok is True
    assert "fleet supervisor complete" in result.message
    assert result.data["passes"] >= 1
