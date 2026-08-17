"""Tests for fleet_registry (touch_repo and count_fleet_live_sessions), carved out of test_charlie_work.py (#1284)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

from charlie_work.paths import runtime_paths


def test_touch_repo_dry_run_does_not_write_fleet_registry(tmp_path: Path) -> None:
    """Issue #618-B: ``touch_repo`` with ``dry_run=True`` must not create or
    update the fleet registry. Running a ``--dry-run`` command from a worktree
    would otherwise repoint the fleet's registry entry at the worktree path.
    """
    from charlie_work.fleet_registry import touch_repo
    from charlie_work.github import GitHub

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    paths = runtime_paths(repo_root, ".var/charlie-work")

    class FakeGitHub(GitHub):
        def name_with_owner(self) -> str:
            return "owner/repo"

    gh = FakeGitHub(repo_root=repo_root)
    fleet_dir = tmp_path / "fleet"

    registry = touch_repo(str(fleet_dir), repo_root, paths, gh, dry_run=True)

    # The registry should be empty (no write occurred)
    assert registry == {"version": 1, "repos": {}}
    # The fleet.json file must NOT exist
    assert not (fleet_dir / "fleet.json").exists()


def test_touch_repo_dry_run_preserves_existing_registry(tmp_path: Path) -> None:
    """Issue #618-B: ``touch_repo`` with ``dry_run=True`` must not bump
    ``last_seen`` or repoint ``repo_root`` for an already-registered repo.
    """
    from charlie_work.fleet_registry import touch_repo
    from charlie_work.github import GitHub

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    paths = runtime_paths(repo_root, ".var/charlie-work")

    class FakeGitHub(GitHub):
        def name_with_owner(self) -> str:
            return "owner/repo"

    gh = FakeGitHub(repo_root=repo_root)
    fleet_dir = tmp_path / "fleet"

    # First call (real) registers the repo
    touch_repo(str(fleet_dir), repo_root, paths, gh)
    fleet_json = fleet_dir / "fleet.json"
    assert fleet_json.exists()
    original = json.loads(fleet_json.read_text(encoding="utf-8"))
    original_last_seen = original["repos"]["owner/repo"]["last_seen"]

    # Second call from a DIFFERENT path (e.g. a worktree) with dry_run=True
    worktree_root = tmp_path / "worktree"
    worktree_root.mkdir(parents=True, exist_ok=True)
    worktree_paths = runtime_paths(worktree_root, ".var/charlie-work")
    gh_wt = FakeGitHub(repo_root=worktree_root)

    touch_repo(str(fleet_dir), worktree_root, worktree_paths, gh_wt, dry_run=True)

    # The registry must be unchanged — repo_root not repointed, last_seen not bumped
    after = json.loads(fleet_json.read_text(encoding="utf-8"))
    assert after == original
    assert after["repos"]["owner/repo"]["repo_root"] == str(repo_root)
    assert after["repos"]["owner/repo"]["last_seen"] == original_last_seen


def test_count_fleet_live_sessions_skips_vanished_repos(tmp_path: Path, monkeypatch) -> None:
    """count_fleet_live_sessions should skip repos that no longer exist and report them."""
    from charlie_work.fleet_registry import count_fleet_live_sessions

    # Create a fake fleet registry with 3 repos
    fleet_dir = tmp_path / ".fleet"
    fleet_dir.mkdir(parents=True)
    fleet_json = fleet_dir / "fleet.json"

    # Create two real repos and one vanished repo
    repo1 = tmp_path / "repo1"
    repo2 = tmp_path / "repo2"
    repo1.mkdir()
    repo2.mkdir()
    (repo1 / ".git").mkdir()
    (repo2 / ".git").mkdir()

    # Create state dirs for the real repos
    state1 = repo1 / ".var" / "charlie-work"
    state2 = repo2 / ".var" / "charlie-work"
    state1.mkdir(parents=True)
    state2.mkdir(parents=True)

    # Create sessions dirs (empty, so no live sessions)
    sessions1 = state1 / "dispatches" / "sessions"
    sessions2 = state2 / "dispatches" / "sessions"
    sessions1.mkdir(parents=True)
    sessions2.mkdir(parents=True)

    # Write the registry
    registry_data = {
        "version": 1,
        "repos": {
            "owner/repo1": {
                "repo_root": str(repo1),
                "name_with_owner": "owner/repo1",
                "config_path": str(repo1 / "orchestrator.config.yaml"),
                "state_dir": str(state1),
                "first_seen": "2024-01-01T00:00:00Z",
                "last_seen": "2024-01-01T00:00:00Z",
            },
            "owner/repo2": {
                "repo_root": str(repo2),
                "name_with_owner": "owner/repo2",
                "config_path": str(repo2 / "orchestrator.config.yaml"),
                "state_dir": str(state2),
                "first_seen": "2024-01-01T00:00:00Z",
                "last_seen": "2024-01-01T00:00:00Z",
            },
            "owner/vanished": {
                "repo_root": str(tmp_path / "vanished"),
                "name_with_owner": "owner/vanished",
                "config_path": str(tmp_path / "vanished" / "orchestrator.config.yaml"),
                "state_dir": str(tmp_path / "vanished" / ".var" / "charlie-work"),
                "first_seen": "2024-01-01T00:00:00Z",
                "last_seen": "2024-01-01T00:00:00Z",
            },
        },
    }
    fleet_json.write_text(json.dumps(registry_data), encoding="utf-8")

    # Redirect fleet_dir resolution to our test fleet dir via the env var
    # fleet_paths.fleet_dir() itself supports (checked before the platform
    # default). Patching the module-level name directly no longer works since
    # fleet_registry composes fleet paths through layout.py, which binds its
    # own reference to fleet_paths.fleet_dir at import time.
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))

    # Count fleet live sessions
    live_count, skipped_repos = count_fleet_live_sessions(None)

    # Should count 0 live sessions (both real repos have empty sessions dirs)
    assert live_count == 0
    # Should report the vanished repo
    assert "owner/vanished" in skipped_repos
    assert len(skipped_repos) == 1


def test_count_fleet_live_sessions_reports_missing_sessions_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A repo whose state_dir exists but whose sessions_dir is missing is reported."""
    from charlie_work.fleet_registry import count_fleet_live_sessions

    fleet_dir = tmp_path / ".fleet"
    fleet_dir.mkdir(parents=True)
    fleet_json = fleet_dir / "fleet.json"

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    # state_dir exists, but the default sessions subdir is deliberately absent.
    state = repo / ".var" / "charlie-work"
    state.mkdir(parents=True)

    registry_data = {
        "version": 1,
        "repos": {
            "owner/repo": {
                "repo_root": str(repo),
                "name_with_owner": "owner/repo",
                "config_path": str(repo / "orchestrator.config.yaml"),
                "state_dir": str(state),
                "first_seen": "2024-01-01T00:00:00Z",
                "last_seen": "2024-01-01T00:00:00Z",
            },
        },
    }
    fleet_json.write_text(json.dumps(registry_data), encoding="utf-8")
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))

    live_count, skipped_repos = count_fleet_live_sessions(None)

    assert live_count == 0
    assert "owner/repo" in skipped_repos
    assert len(skipped_repos) == 1


def test_count_fleet_live_sessions_respects_devin_sessions_dir_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A repo that overrides devin.sessions_dir is counted from that path."""
    from charlie_work.fleet_registry import count_fleet_live_sessions

    fleet_dir = tmp_path / ".fleet"
    fleet_dir.mkdir(parents=True)
    fleet_json = fleet_dir / "fleet.json"

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    # Default sessions dir is missing; the override is the live one.
    state = repo / ".var" / "charlie-work"
    state.mkdir(parents=True)

    custom_sessions = repo / "custom-sessions"
    custom_sessions.mkdir(parents=True)
    (custom_sessions / "issue-1.json").write_text(
        json.dumps(
            {
                "issue_number": 1,
                "branch": "main",
                "worktree_path": str(repo / "worktrees" / "issue-1"),
                "prompt_path": str(repo / "prompt.md"),
                "command": ["devin"],
                "pid": 1234,
                "started_at": "2026-08-05T00:00:00Z",
                "log_path": str(repo / "log.txt"),
            }
        ),
        encoding="utf-8",
    )

    (repo / "orchestrator.config.yaml").write_text(
        "devin:\n  sessions_dir: custom-sessions\n",
        encoding="utf-8",
    )

    registry_data = {
        "version": 1,
        "repos": {
            "owner/repo": {
                "repo_root": str(repo),
                "name_with_owner": "owner/repo",
                "config_path": str(repo / "orchestrator.config.yaml"),
                "state_dir": str(state),
                "first_seen": "2024-01-01T00:00:00Z",
                "last_seen": "2024-01-01T00:00:00Z",
            },
        },
    }
    fleet_json.write_text(json.dumps(registry_data), encoding="utf-8")
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda _record: True)

    live_count, skipped_repos = count_fleet_live_sessions(None)

    assert live_count == 1
    assert skipped_repos == []


def test_count_fleet_live_sessions_respects_global_devin_sessions_dir_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A repo with no per-repo override is still counted when fleet config sets devin.sessions_dir.

    Regression for the review of issue #707: count_fleet_live_sessions loaded the
    per-repo orchestrator.config.yaml only, so a global <fleet_dir>/config.yaml
    devin.sessions_dir override was silently ignored.
    """
    from charlie_work.fleet_registry import count_fleet_live_sessions

    fleet_dir = tmp_path / ".fleet"
    fleet_dir.mkdir(parents=True)
    fleet_json = fleet_dir / "fleet.json"

    # No per-repo config; all config comes from the fleet-wide layer.
    (fleet_dir / "config.yaml").write_text(
        "devin:\n  sessions_dir: custom-sessions\n",
        encoding="utf-8",
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    # Default sessions dir is missing; the global override is the live one.
    state = repo / ".var" / "charlie-work"
    state.mkdir(parents=True)

    custom_sessions = repo / "custom-sessions"
    custom_sessions.mkdir(parents=True)
    (custom_sessions / "issue-1.json").write_text(
        json.dumps(
            {
                "issue_number": 1,
                "branch": "main",
                "worktree_path": str(repo / "worktrees" / "issue-1"),
                "prompt_path": str(repo / "prompt.md"),
                "command": ["devin"],
                "pid": 1234,
                "started_at": "2026-08-05T00:00:00Z",
                "log_path": str(repo / "log.txt"),
            }
        ),
        encoding="utf-8",
    )

    registry_data = {
        "version": 1,
        "repos": {
            "owner/repo": {
                "repo_root": str(repo),
                "name_with_owner": "owner/repo",
                "config_path": str(repo / "orchestrator.config.yaml"),
                "state_dir": str(state),
                "first_seen": "2024-01-01T00:00:00Z",
                "last_seen": "2024-01-01T00:00:00Z",
            },
        },
    }
    fleet_json.write_text(json.dumps(registry_data), encoding="utf-8")
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda _record: True)

    live_count, skipped_repos = count_fleet_live_sessions(None)

    assert live_count == 1
    assert skipped_repos == []


def test_count_fleet_live_sessions_skips_repo_with_null_state_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A corrupted registry entry with state_dir: null does not crash count_fleet_live_sessions.

    Regression for the review of issue #707 (rework round 3):
    ``Path(entry.get("state_dir", ""))`` returns ``Path(None)`` when the key is
    present with a null value (``.get``'s default only applies when the key is
    *absent*), raising TypeError. The same bug class was already fixed in this
    PR for ``repo_root`` across fleet_dispatch.py but missed for ``state_dir``
    in this exact function. The fix ``entry.get("state_dir") or ""`` makes null
    behave identically to a missing key (fall back to cwd), and the repo is
    then reported in skipped_repos because the resolved sessions_dir under cwd
    does not exist.
    """
    from charlie_work.fleet_registry import count_fleet_live_sessions

    fleet_dir = tmp_path / ".fleet"
    fleet_dir.mkdir(parents=True)
    fleet_json = fleet_dir / "fleet.json"

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    # Pin cwd to tmp_path so the cwd-fallback (Path("") == Path(".")) resolves
    # deterministically: sessions_dir_default(tmp_path) = tmp_path/dispatches/
    # sessions, which does not exist, so the repo is skipped + reported.
    monkeypatch.chdir(tmp_path)

    registry_data = {
        "version": 1,
        "repos": {
            "owner/repo": {
                "repo_root": str(repo),
                "name_with_owner": "owner/repo",
                "config_path": str(repo / "orchestrator.config.yaml"),
                "state_dir": None,
                "first_seen": "2024-01-01T00:00:00Z",
                "last_seen": "2024-01-01T00:00:00Z",
            },
        },
    }
    fleet_json.write_text(json.dumps(registry_data), encoding="utf-8")
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))

    # Must not raise TypeError; the repo is skipped + reported.
    live_count, skipped_repos = count_fleet_live_sessions(None)

    assert live_count == 0
    assert "owner/repo" in skipped_repos
    assert len(skipped_repos) == 1


def test_count_fleet_live_sessions_skips_repo_with_malformed_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A repo with an unparseable per-repo config does not crash count_fleet_live_sessions.

    Regression for the review of issue #707 (rework round 3): the
    ``load_layered_config`` call in ``count_fleet_live_sessions`` is wrapped in
    a broad ``except Exception`` containment, but no test exercised a malformed
    ``orchestrator.config.yaml`` (which raises ``yaml.YAMLError``) against it.
    Mirrors the equivalent malformed-config tests already added for
    fleet_dispatch.py. The repo must land in skipped_repos instead of crashing.
    """
    from charlie_work.fleet_registry import count_fleet_live_sessions

    fleet_dir = tmp_path / ".fleet"
    fleet_dir.mkdir(parents=True)
    fleet_json = fleet_dir / "fleet.json"

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    state = repo / ".var" / "charlie-work"
    state.mkdir(parents=True)

    # Plant a malformed YAML file that yaml.safe_load cannot parse.
    (repo / "orchestrator.config.yaml").write_text(
        "devin:\n  sessions_dir: [unclosed\n",
        encoding="utf-8",
    )

    registry_data = {
        "version": 1,
        "repos": {
            "owner/repo": {
                "repo_root": str(repo),
                "name_with_owner": "owner/repo",
                "config_path": str(repo / "orchestrator.config.yaml"),
                "state_dir": str(state),
                "first_seen": "2024-01-01T00:00:00Z",
                "last_seen": "2024-01-01T00:00:00Z",
            },
        },
    }
    fleet_json.write_text(json.dumps(registry_data), encoding="utf-8")
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))

    # Must not raise yaml.YAMLError; the repo is skipped + reported.
    live_count, skipped_repos = count_fleet_live_sessions(None)

    assert live_count == 0
    assert "owner/repo" in skipped_repos
    assert len(skipped_repos) == 1


def test_fleet_registry_touch_repo_first_call(tmp_path: Path) -> None:
    """Test that touch_repo sets first_seen and last_seen on first registration."""
    from charlie_work.fleet_registry import touch_repo
    from charlie_work.github import GitHub

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    paths = runtime_paths(repo_root, ".var/charlie-work")

    # Mock GitHub that returns a nameWithOwner
    class FakeGitHub(GitHub):
        def name_with_owner(self) -> str:
            return "owner/repo"

    gh = FakeGitHub(repo_root=repo_root)

    # Touch repo with isolated fleet dir
    registry = touch_repo(str(tmp_path / "fleet"), repo_root, paths, gh)

    assert "repos" in registry
    assert "owner/repo" in registry["repos"]
    entry = registry["repos"]["owner/repo"]
    assert entry["repo_root"] == str(repo_root)
    assert entry["name_with_owner"] == "owner/repo"
    assert entry["config_path"] == str(repo_root / "orchestrator.config.yaml")
    assert entry["state_dir"] == str(paths.root)
    assert entry["first_seen"] == entry["last_seen"]  # First call: both equal


def test_fleet_registry_touch_repo_second_call(tmp_path: Path) -> None:
    """Test that touch_repo preserves first_seen and bumps last_seen on subsequent calls."""
    from charlie_work.fleet_registry import touch_repo
    from charlie_work.github import GitHub

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    paths = runtime_paths(repo_root, ".var/charlie-work")

    class FakeGitHub(GitHub):
        def name_with_owner(self) -> str:
            return "owner/repo"

    gh = FakeGitHub(repo_root=repo_root)

    # First call
    registry = touch_repo(str(tmp_path / "fleet"), repo_root, paths, gh)
    first_first_seen = registry["repos"]["owner/repo"]["first_seen"]
    first_last_seen = registry["repos"]["owner/repo"]["last_seen"]

    # Small delay to ensure timestamp difference (need >1s due to second resolution)
    time.sleep(2.0)

    # Second call
    registry = touch_repo(str(tmp_path / "fleet"), repo_root, paths, gh)
    second_first_seen = registry["repos"]["owner/repo"]["first_seen"]
    second_last_seen = registry["repos"]["owner/repo"]["last_seen"]

    assert second_first_seen == first_first_seen  # first_seen preserved
    assert second_last_seen != first_last_seen  # last_seen bumped
    assert second_last_seen > first_last_seen  # last_seen increased


def test_fleet_registry_touch_repo_moved_repo(tmp_path: Path) -> None:
    """Test that touch_repo updates repo_root when repo is moved."""
    from charlie_work.fleet_registry import touch_repo
    from charlie_work.github import GitHub

    repo_root_old = tmp_path / "repo_old"
    repo_root_old.mkdir(parents=True, exist_ok=True)
    paths_old = runtime_paths(repo_root_old, ".var/charlie-work")

    class FakeGitHub(GitHub):
        def name_with_owner(self) -> str:
            return "owner/repo"

    gh_old = FakeGitHub(repo_root=repo_root_old)

    # First registration
    registry = touch_repo(str(tmp_path / "fleet"), repo_root_old, paths_old, gh_old)
    first_first_seen = registry["repos"]["owner/repo"]["first_seen"]

    # Move repo
    repo_root_new = tmp_path / "repo_new"
    repo_root_new.mkdir(parents=True, exist_ok=True)
    paths_new = runtime_paths(repo_root_new, ".var/charlie-work")
    gh_new = FakeGitHub(repo_root=repo_root_new)

    # Re-register with new path
    registry = touch_repo(str(tmp_path / "fleet"), repo_root_new, paths_new, gh_new)

    # Should update repo_root but preserve first_seen (same nameWithOwner)
    entry = registry["repos"]["owner/repo"]
    assert entry["repo_root"] == str(repo_root_new)
    assert entry["first_seen"] == first_first_seen  # Preserved on move


def test_fleet_registry_touch_repo_gh_error(tmp_path: Path) -> None:
    """Test that touch_repo silently skips registration on gh error."""
    from charlie_work.fleet_registry import touch_repo
    from charlie_work.github import GitHub, GitHubError

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    paths = runtime_paths(repo_root, ".var/charlie-work")

    class FakeGitHub(GitHub):
        def name_with_owner(self) -> str:
            raise GitHubError("gh not available")

    gh = FakeGitHub(repo_root=repo_root)

    # Should not raise, should return empty registry
    registry = touch_repo(str(tmp_path / "fleet"), repo_root, paths, gh)
    assert registry == {"version": 1, "repos": {}}


def test_fleet_registry_uses_state_lock(tmp_path: Path) -> None:
    """Test that fleet_registry writes go through state.save_state."""
    from charlie_work.fleet_registry import touch_repo
    from charlie_work.github import GitHub

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    paths = runtime_paths(repo_root, ".var/charlie-work")

    class FakeGitHub(GitHub):
        def name_with_owner(self) -> str:
            return "owner/repo"

    gh = FakeGitHub(repo_root=repo_root)

    # Spy on save_state
    from charlie_work.state import save_state

    original_save_state = save_state
    calls = []

    def spy_save_state(path: Path, data: dict) -> dict:
        calls.append(path)
        return original_save_state(path, data)

    with patch("charlie_work.fleet_registry.save_state", side_effect=spy_save_state):
        touch_repo(str(tmp_path / "fleet"), repo_root, paths, gh)

    # Verify save_state was called with fleet.json path
    assert len(calls) == 1
    assert calls[0] == tmp_path / "fleet" / "fleet.json"
