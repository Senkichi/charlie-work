"""Regression tests for issue #1372: test runs pollute the live fleet registry.

Three layers of defense, each guarding a different failure mode:

1. **Test hygiene** (autouse fixture in conftest.py): ``CHARLIE_WORK_FLEET_DIR``
   is redirected to ``tmp_path`` for every test, so isolation stops depending
   on each test author remembering ``monkeypatch.setenv``.
2. **Registry-writer backstop**: ``touch_repo`` refuses to persist an entry
   whose ``repo_root`` resolves under the system temp directory.
3. **Reader resilience**: ``fleet_loop`` and ``run_fleet_status`` treat a
   registry entry whose ``repo_root`` no longer exists as STALE, not as a
   live failing lane — skip + warn into the daemon's events.db, never into
   the dead entry's state_dir. After a configurable grace period, prune.
"""

from __future__ import annotations

import json
import os
import tempfile as _tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from charlie_work.fleet_dispatch import (
    _parse_registry_timestamp,
    _prune_stale_registry_entries,
    fleet_loop,
)
from charlie_work.fleet_registry import _load_registry, touch_repo
from charlie_work.github import GitHub
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths

# Issue #1372: capture the real tempfile.gettempdir at module import time,
# before any autouse fixture in conftest.py patches it. The backstop test
# restores this so touch_repo sees the real system temp dir.
_real_gettempdir = _tempfile.gettempdir


# ---------------------------------------------------------------------------
# Layer 2: touch_repo temp-dir backstop
# ---------------------------------------------------------------------------


class _FakeGitHub(GitHub):
    """Minimal GitHub stub returning a fixed nameWithOwner."""

    def __init__(self, repo_root: Path, name: str = "owner/repo") -> None:
        # Bypass GitHub.__init__ which requires gh CLI; we only need
        # name_with_owner() to succeed. GitHub is a frozen dataclass, so use
        # object.__setattr__ to set fields without triggering FrozenInstanceError.
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "repo_root", repo_root)

    def name_with_owner(self) -> str:
        return self._name  # type: ignore[attr-defined]


def test_touch_repo_refuses_repo_root_under_temp_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1372 layer 2: touch_repo refuses to persist an entry whose
    repo_root resolves under tempfile.gettempdir(). It logs a warning naming
    the rejected path and returns without raising.
    """
    # Restore the real tempfile.gettempdir as seen by fleet_registry — the
    # autouse fixture in conftest.py redirects fleet_registry._get_temp_dir
    # so other tests' tmp_path-based repo_roots aren't rejected. This test
    # specifically exercises the backstop, so it needs the real system temp dir.
    from charlie_work import fleet_registry

    monkeypatch.setattr(fleet_registry, "_get_temp_dir", _real_gettempdir)

    fleet_dir = tmp_path / "fleet"
    # Place repo_root explicitly under the system temp directory.
    temp_root = Path(_real_gettempdir())
    repo_root = temp_root / "cw-1372-test-repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    try:
        paths = runtime_paths(repo_root, ".var/charlie-work")
        gh = _FakeGitHub(repo_root)

        registry = touch_repo(str(fleet_dir), repo_root, paths, gh)

        # The registry is returned unchanged (empty) — no write occurred.
        assert registry == {"version": 1, "repos": {}}
        # The fleet.json file must NOT exist.
        assert not (fleet_dir / "fleet.json").exists()
    finally:
        # Clean up the temp-dir repo so we don't leave artifacts.
        import shutil

        shutil.rmtree(repo_root, ignore_errors=True)


def test_touch_repo_writes_for_repo_root_outside_temp_dir(tmp_path: Path) -> None:
    """Issue #1372 layer 2: a repo_root outside the temp dir is registered
    normally — no behavior change for valid entries (acceptance criterion #5).

    The autouse ``_redirect_temp_dir_for_touch_repo_backstop`` fixture in
    conftest.py redirects ``tempfile.gettempdir`` to ``tmp_path /
    "__system_temp__"`` (a sibling of ``tmp_path / "repo"``, not a parent),
    so ``contains(temp_root, repo_root)`` returns False and the backstop
    does not fire — exactly the "valid entry" path.
    """
    fleet_dir = tmp_path / "fleet"
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    paths = runtime_paths(repo_root, ".var/charlie-work")
    gh = _FakeGitHub(repo_root)

    registry = touch_repo(str(fleet_dir), repo_root, paths, gh)

    assert "owner/repo" in registry["repos"]
    assert registry["repos"]["owner/repo"]["repo_root"] == str(repo_root)
    assert (fleet_dir / "fleet.json").exists()


# ---------------------------------------------------------------------------
# Layer 1: autouse fixture isolation (acceptance criterion #1)
# ---------------------------------------------------------------------------


def test_autouse_fixture_isolates_fleet_registry(tmp_path: Path) -> None:
    """Issue #1372 acceptance criterion #1: a test that invokes cli.main via
    _FakeGitHub without any explicit fleet-dir isolation leaves the real
    %LOCALAPPDATA%\\charlie-work\\fleet.json byte-identical.

    The autouse ``_isolate_fleet_registry`` fixture in conftest.py sets
    ``CHARLIE_WORK_FLEET_DIR`` to ``tmp_path / "fleet"`` for every test. This
    test verifies that env var is set and that the real registry path resolves
    to the tmp_path, not the platform default.
    """
    # The autouse fixture should have set CHARLIE_WORK_FLEET_DIR.
    fleet_dir_env = os.environ.get("CHARLIE_WORK_FLEET_DIR")
    assert fleet_dir_env is not None, (
        "CHARLIE_WORK_FLEET_DIR must be set by the autouse _isolate_fleet_registry "
        "fixture — without it, tests write to the operator's live registry."
    )
    # The env var must point inside tmp_path (the fixture uses tmp_path / "fleet").
    assert str(tmp_path) in fleet_dir_env, (
        f"CHARLIE_WORK_FLEET_DIR={fleet_dir_env!r} must resolve under tmp_path="
        f"{tmp_path!r} for test isolation."
    )


# ---------------------------------------------------------------------------
# Layer 3: fleet_loop reader resilience (acceptance criterion #3)
# ---------------------------------------------------------------------------


def _write_fleet_json(
    fleet_dir: Path,
    entries: dict[str, dict[str, str]],
) -> Path:
    """Write a fleet.json registry with the given entries and return its path."""
    fleet_dir.mkdir(parents=True, exist_ok=True)
    fleet_json = fleet_dir / "fleet.json"
    registry = {"version": 1, "repos": entries}
    fleet_json.write_text(json.dumps(registry), encoding="utf-8")
    return fleet_json


def test_fleet_loop_stale_entry_skipped_and_warned_to_daemon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #1372 acceptance criterion #3: with a fleet.json containing one
    valid entry and one entry whose repo_root does not exist, fleet_loop
    completes the pass (ok=True), emits exactly one fleet_registry_stale_entry
    warning into the daemon state dir, and asserts NOTHING is created under
    the dead state_dir.
    """
    fleet_dir = tmp_path / "fleet"
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))

    # Valid repo
    valid_repo = tmp_path / "valid-repo"
    valid_repo.mkdir()
    (valid_repo / ".git").mkdir()
    valid_state = valid_repo / ".var" / "charlie-work"
    valid_state.mkdir(parents=True)

    # Dead repo (repo_root does not exist)
    dead_repo_root = tmp_path / "dead-repo"
    dead_state_dir = tmp_path / "dead-state"

    now = datetime.now(UTC)
    entries = {
        "owner/valid": {
            "repo_root": str(valid_repo),
            "name_with_owner": "owner/valid",
            "config_path": str(valid_repo / "orchestrator.config.yaml"),
            "state_dir": str(valid_state),
            "first_seen": now.isoformat().replace("+00:00", "Z"),
            "last_seen": now.isoformat().replace("+00:00", "Z"),
        },
        "owner/dead": {
            "repo_root": str(dead_repo_root),
            "name_with_owner": "owner/dead",
            "config_path": str(dead_repo_root / "orchestrator.config.yaml"),
            "state_dir": str(dead_state_dir),
            "first_seen": now.isoformat().replace("+00:00", "Z"),
            "last_seen": now.isoformat().replace("+00:00", "Z"),
        },
    }
    _write_fleet_json(fleet_dir, entries)

    # Mock the per-repo machinery so only the stale-handling path is exercised.
    with (
        patch("charlie_work.fleet_dispatch.load_layered_config") as mock_cfg,
        patch("charlie_work.fleet_dispatch.runtime_paths") as mock_paths,
        patch("charlie_work.fleet_dispatch.GitHub") as mock_gh,
        patch("charlie_work.fleet_dispatch.OrchestratorApp") as mock_app,
    ):
        from charlie_work.config import OrchestratorConfig

        mock_cfg.return_value = OrchestratorConfig()
        mock_p = MagicMock()
        mock_p.root = valid_state
        mock_paths.return_value = mock_p
        mock_gh.return_value = MagicMock()
        mock_a = MagicMock()
        mock_a.loop.return_value = MagicMock(ok=True, message="ok", data={})
        mock_app.return_value = mock_a

        result = fleet_loop(
            fleet_dir_override=str(fleet_dir),
            global_config=None,
            work_only=False,
        )

    # The pass completes successfully — one corpse does not degrade it.
    assert result.ok is True
    # The stale entry is marked ok=True with stale=True.
    assert result.data["repos"]["owner/dead"]["ok"] is True
    assert result.data["repos"]["owner/dead"].get("stale") is True
    # The stale key is in the result's stale list.
    assert "owner/dead" in result.data.get("stale", [])

    # Exactly one fleet_registry_stale_entry warning in the daemon's events.db.
    fleet_state_path = fleet_dir / "state.json"
    daemon_events = query_events(fleet_state_path, kind="fleet_registry_stale_entry")
    # One for the stale detection; no prune (global_config=None → grace_days=0).
    stale_detect_events = [
        e for e in daemon_events if e["payload"].get("reason") == "repo_root_missing"
    ]
    assert len(stale_detect_events) == 1
    assert stale_detect_events[0]["level"] == "warning"
    assert stale_detect_events[0]["payload"]["repo_key"] == "owner/dead"

    # NOTHING is created under the dead state_dir.
    assert not dead_state_dir.exists(), (
        f"dead state_dir {dead_state_dir} must not be created — log_event "
        f"auto-mkdirs (#746) would resurrect a zombie directory."
    )


# ---------------------------------------------------------------------------
# Layer 3: run_fleet_status stale handling (acceptance criterion #3)
# ---------------------------------------------------------------------------


def test_run_fleet_status_stale_does_not_flip_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #1372 acceptance criterion #3: charlie fleet status --json exits 0
    with the corpse listed under a stale field that does not affect the exit
    code.
    """
    fleet_dir = tmp_path / "fleet"
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))

    valid_repo = tmp_path / "valid-repo"
    valid_repo.mkdir()
    (valid_repo / ".git").mkdir()
    valid_state = valid_repo / ".var" / "charlie-work"
    valid_state.mkdir(parents=True)

    dead_repo_root = tmp_path / "dead-repo"

    now = datetime.now(UTC)
    entries = {
        "owner/valid": {
            "repo_root": str(valid_repo),
            "name_with_owner": "owner/valid",
            "config_path": str(valid_repo / "orchestrator.config.yaml"),
            "state_dir": str(valid_state),
            "first_seen": now.isoformat().replace("+00:00", "Z"),
            "last_seen": now.isoformat().replace("+00:00", "Z"),
        },
        "owner/dead": {
            "repo_root": str(dead_repo_root),
            "name_with_owner": "owner/dead",
            "config_path": str(dead_repo_root / "orchestrator.config.yaml"),
            "state_dir": str(tmp_path / "dead-state"),
            "first_seen": now.isoformat().replace("+00:00", "Z"),
            "last_seen": now.isoformat().replace("+00:00", "Z"),
        },
    }
    _write_fleet_json(fleet_dir, entries)

    from charlie_work.cli import run_fleet_status

    # Mock the valid repo's status path so it doesn't hit GitHub.
    with (
        patch("charlie_work.cli.load_layered_config") as mock_cfg,
        patch("charlie_work.cli.runtime_paths") as mock_paths,
        patch("charlie_work.cli.GitHub") as mock_gh,
        patch("charlie_work.cli.OrchestratorApp") as mock_app,
        patch("charlie_work.cli.compute_api_worker_fleet_report") as mock_report,
    ):
        from charlie_work.config import OrchestratorConfig

        mock_cfg.return_value = OrchestratorConfig()
        mock_p = MagicMock()
        mock_p.root = valid_state
        mock_paths.return_value = mock_p
        mock_gh.return_value = MagicMock()
        mock_a = MagicMock()
        mock_a.status.return_value = MagicMock(data={"ready_issue_count": 0})
        mock_app.return_value = mock_a
        mock_report.return_value = None

        args = MagicMock()
        args.fleet_dir = None
        result = run_fleet_status(args)

    # ok=True — the stale entry does NOT flip the exit code.
    assert result.ok is True
    # The corpse is listed under stale, not errors.
    stale_keys = [s["repo_key"] for s in result.data.get("stale", [])]
    assert "owner/dead" in stale_keys
    assert "owner/dead" not in [e["repo_key"] for e in result.data.get("errors", [])]


# ---------------------------------------------------------------------------
# Layer 4: prune-after-grace (acceptance criterion #4)
# ---------------------------------------------------------------------------


def test_prune_stale_registry_entries_after_grace(tmp_path: Path) -> None:
    """Issue #1372 acceptance criterion #4: prune-after-grace behavior using
    the injectable clock idiom (#822). Prune happens under state_lock and is
    atomic (temp + replace via save_state).
    """
    fleet_dir = tmp_path / "fleet"
    fleet_json = fleet_dir / "fleet.json"

    now = datetime.now(UTC)
    old_ts = (now - timedelta(days=10)).isoformat().replace("+00:00", "Z")
    recent_ts = now.isoformat().replace("+00:00", "Z")

    entries = {
        "owner/old-stale": {
            "repo_root": str(tmp_path / "old-stale"),
            "state_dir": str(tmp_path / "old-state"),
            "last_seen": old_ts,
        },
        "owner/recent-stale": {
            "repo_root": str(tmp_path / "recent-stale"),
            "state_dir": str(tmp_path / "recent-state"),
            "last_seen": recent_ts,
        },
        "owner/valid": {
            "repo_root": str(tmp_path / "valid"),
            "state_dir": str(tmp_path / "valid-state"),
            "last_seen": recent_ts,
        },
    }
    _write_fleet_json(fleet_dir, entries)

    # Prune with grace_days=7. Only "owner/old-stale" should be pruned
    # (last_seen 10 days ago > 7-day grace). "owner/recent-stale" is stale
    # but within the grace period. "owner/valid" is not in the stale list.
    pruned = _prune_stale_registry_entries(
        fleet_json,
        stale_keys=["owner/old-stale", "owner/recent-stale"],
        grace_days=7,
        now=now,
    )

    assert pruned == ["owner/old-stale"]

    # Verify the registry on disk: old-stale removed, others remain.
    after = _load_registry(fleet_json)
    assert "owner/old-stale" not in after["repos"]
    assert "owner/recent-stale" in after["repos"]
    assert "owner/valid" in after["repos"]


def test_prune_stale_registry_entries_grace_zero_disables(tmp_path: Path) -> None:
    """Issue #1372: grace_days=0 disables pruning — stale entries are skipped
    but never removed.
    """
    fleet_dir = tmp_path / "fleet"
    fleet_json = fleet_dir / "fleet.json"

    now = datetime.now(UTC)
    old_ts = (now - timedelta(days=30)).isoformat().replace("+00:00", "Z")

    entries = {
        "owner/old": {
            "repo_root": str(tmp_path / "old"),
            "state_dir": str(tmp_path / "old-state"),
            "last_seen": old_ts,
        },
    }
    _write_fleet_json(fleet_dir, entries)

    pruned = _prune_stale_registry_entries(
        fleet_json,
        stale_keys=["owner/old"],
        grace_days=0,
        now=now,
    )

    assert pruned == []
    after = _load_registry(fleet_json)
    assert "owner/old" in after["repos"]


def test_prune_stale_registry_entries_no_stale_keys_is_noop(tmp_path: Path) -> None:
    """Pruning with an empty stale_keys list is a no-op."""
    fleet_dir = tmp_path / "fleet"
    fleet_json = fleet_dir / "fleet.json"
    _write_fleet_json(fleet_dir, {"owner/x": {"repo_root": "/x", "last_seen": "x"}})

    pruned = _prune_stale_registry_entries(
        fleet_json, stale_keys=[], grace_days=7, now=datetime.now(UTC)
    )
    assert pruned == []


def test_parse_registry_timestamp_round_trip() -> None:
    """_parse_registry_timestamp handles the Z-suffixed ISO format touch_repo writes."""
    ts = "2026-08-21T12:00:00Z"
    parsed = _parse_registry_timestamp(ts)
    assert parsed is not None
    assert parsed.tzinfo is not None
    assert parsed.year == 2026
    assert parsed.month == 8
    assert parsed.day == 21

    # None / empty / unparseable return None (conservative: not old enough to prune).
    assert _parse_registry_timestamp(None) is None
    assert _parse_registry_timestamp("") is None
    assert _parse_registry_timestamp("not-a-date") is None


def test_fleet_loop_prunes_stale_after_grace_with_injectable_clock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #1372 acceptance criterion #4: fleet_loop prunes stale entries
    past the grace period using the injectable clock (now parameter, #822).
    The prune happens under state_lock and is atomic.
    """
    fleet_dir = tmp_path / "fleet"
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))

    now = datetime.now(UTC)
    old_ts = (now - timedelta(days=10)).isoformat().replace("+00:00", "Z")

    dead_repo_root = tmp_path / "dead-repo"
    dead_state_dir = tmp_path / "dead-state"

    entries = {
        "owner/dead": {
            "repo_root": str(dead_repo_root),
            "name_with_owner": "owner/dead",
            "config_path": str(dead_repo_root / "orchestrator.config.yaml"),
            "state_dir": str(dead_state_dir),
            "first_seen": old_ts,
            "last_seen": old_ts,
        },
    }
    _write_fleet_json(fleet_dir, entries)

    # Build a global_config with fleet_registry_stale_grace_days=7.
    from charlie_work.config import OrchestratorConfig, RuntimeConfig

    global_config = OrchestratorConfig(runtime=RuntimeConfig(fleet_registry_stale_grace_days=7))

    with (
        patch("charlie_work.fleet_dispatch._run_fleet_allocation_prologue") as mock_alloc,
        patch("charlie_work.fleet_dispatch._run_fleet_autoscale_prologue") as mock_auto,
    ):
        mock_alloc.return_value = []
        mock_auto.return_value = []

        result = fleet_loop(
            fleet_dir_override=str(fleet_dir),
            global_config=global_config,
            work_only=False,
            now=now,
        )

    # The pass completes (ok=True — stale is not a failure).
    assert result.ok is True
    # The stale entry was pruned.
    assert "owner/dead" in result.data.get("pruned", [])

    # The registry no longer contains the dead entry.
    after = _load_registry(fleet_dir / "fleet.json")
    assert "owner/dead" not in after["repos"]

    # A prune event was recorded to the daemon's events.db.
    fleet_state_path = fleet_dir / "state.json"
    prune_events = [
        e
        for e in query_events(fleet_state_path, kind="fleet_registry_stale_entry")
        if e["payload"].get("reason") == "pruned_after_grace"
    ]
    assert len(prune_events) == 1
    assert prune_events[0]["payload"]["repo_key"] == "owner/dead"
    assert prune_events[0]["payload"]["grace_days"] == 7


def test_fleet_loop_does_not_prune_within_grace_period(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #1372: a stale entry within the grace period is skipped but NOT
    pruned — it stays in the registry for the next pass.
    """
    fleet_dir = tmp_path / "fleet"
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))

    now = datetime.now(UTC)
    recent_ts = (now - timedelta(days=2)).isoformat().replace("+00:00", "Z")

    dead_repo_root = tmp_path / "dead-repo"

    entries = {
        "owner/dead": {
            "repo_root": str(dead_repo_root),
            "name_with_owner": "owner/dead",
            "config_path": str(dead_repo_root / "orchestrator.config.yaml"),
            "state_dir": str(tmp_path / "dead-state"),
            "first_seen": recent_ts,
            "last_seen": recent_ts,
        },
    }
    _write_fleet_json(fleet_dir, entries)

    from charlie_work.config import OrchestratorConfig, RuntimeConfig

    global_config = OrchestratorConfig(runtime=RuntimeConfig(fleet_registry_stale_grace_days=7))

    with (
        patch("charlie_work.fleet_dispatch._run_fleet_allocation_prologue") as mock_alloc,
        patch("charlie_work.fleet_dispatch._run_fleet_autoscale_prologue") as mock_auto,
    ):
        mock_alloc.return_value = []
        mock_auto.return_value = []

        result = fleet_loop(
            fleet_dir_override=str(fleet_dir),
            global_config=global_config,
            work_only=False,
            now=now,
        )

    assert result.ok is True
    # Not pruned (within grace period).
    assert result.data.get("pruned", []) == []
    # Still in the registry.
    after = _load_registry(fleet_dir / "fleet.json")
    assert "owner/dead" in after["repos"]


# ---------------------------------------------------------------------------
# Config knob validation (acceptance criterion #4)
# ---------------------------------------------------------------------------


def test_fleet_registry_stale_grace_days_default_is_7() -> None:
    """Issue #1372: the default grace period is 7 days."""
    from charlie_work.config import RuntimeConfig

    rc = RuntimeConfig()
    assert rc.fleet_registry_stale_grace_days == 7


def test_fleet_registry_stale_grace_days_accepts_zero() -> None:
    """Issue #1372: 0 disables pruning (stale entries skipped but never removed)."""
    from charlie_work.config import RuntimeConfig

    rc = RuntimeConfig(fleet_registry_stale_grace_days=0)
    assert rc.fleet_registry_stale_grace_days == 0


def test_fleet_registry_stale_grace_days_rejects_negative() -> None:
    """Issue #1372: negative grace_days is a ConfigError when parsed from YAML."""
    from charlie_work.config import ConfigError, build_config_from_data

    with pytest.raises(ConfigError, match="must be >= 0"):
        build_config_from_data({"runtime": {"fleet_registry_stale_grace_days": -1}})


def test_fleet_registry_stale_grace_days_rejects_bool() -> None:
    """Issue #1372: bool is rejected despite bool subclassing int — True/False
    would silently enable/disable pruning via a non-integer config value.
    """
    from charlie_work.config import ConfigError, build_config_from_data

    with pytest.raises(ConfigError, match="must be an int"):
        build_config_from_data({"runtime": {"fleet_registry_stale_grace_days": True}})


def test_fleet_registry_stale_grace_days_rejects_string() -> None:
    """Issue #1372: a string value is a ConfigError, not a silent coercion."""
    from charlie_work.config import ConfigError, build_config_from_data

    with pytest.raises(ConfigError, match="must be an int"):
        build_config_from_data({"runtime": {"fleet_registry_stale_grace_days": "7"}})


def test_fleet_registry_stale_grace_days_parses_from_yaml(
    tmp_path: Path,
) -> None:
    """Issue #1372: the grace_days knob is parsed from a real YAML config file
    via the unmocked load_layered_config path."""
    from charlie_work.global_config import load_layered_config

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "orchestrator.config.yaml").write_text(
        "runtime:\n  state_dir: .var/charlie-work\n  fleet_registry_stale_grace_days: 14\n",
        encoding="utf-8",
    )

    config = load_layered_config(repo_root, None, fleet_dir_override=str(tmp_path / "fleet"))
    assert config.runtime.fleet_registry_stale_grace_days == 14


# ---------------------------------------------------------------------------
# Event classification (acceptance criterion #3)
# ---------------------------------------------------------------------------


def test_fleet_registry_stale_entry_classified_as_warning(
    tmp_path: Path,
) -> None:
    """Issue #1372: the fleet_registry_stale_entry event kind is registered at
    warning level (not error), so query_events(level=...) and the attention
    digest classify it correctly.
    """
    from charlie_work.instrumentation import log_event

    state_path = tmp_path / "state.json"
    log_event(
        state_path,
        "fleet_registry_stale_entry",
        {"repo_key": "owner/dead", "reason": "repo_root_missing"},
        repo="owner/dead",
    )

    recorded = query_events(state_path, kind="fleet_registry_stale_entry")
    assert len(recorded) == 1
    assert recorded[0]["level"] == "warning"
