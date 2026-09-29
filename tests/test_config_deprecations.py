"""Tests for the deprecated-config-key registry, read-time signal, and
fleet retirement sweep (issue #1976).

The registry (``config_deprecations.DEPRECATED_CONFIG_KEYS``) is the single
source of truth for deprecated config keys. Config loading emits
``config_key_deprecated_read`` per registered key per layer file read; the
fleet pass runs ``run_config_retirement_sweep`` once to track per-key quiet
windows and mark the removal issue Ready once a key has been absent from
every layer of every registered repo for ``runtime.config_retirement_quiet_days``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from charlie_work import layout
from charlie_work.config import OrchestratorConfig, load_config
from charlie_work.config_deprecations import (
    DEPRECATED_CONFIG_KEYS,
    DeprecatedConfigKey,
    deprecated_keys_in,
)
from charlie_work.config_retirement_sweep import run_config_retirement_sweep
from charlie_work.global_config import load_layered_config
from charlie_work.instrumentation import query_events
from _fakes_github import FakeGitHub


def _repo_state_path(repo_root: Path, state_dir: str = layout.DEFAULT_STATE_DIR) -> Path:
    return layout.state_file_path(repo_root / state_dir)


def _write_repo_config(repo_root: Path, content: str) -> Path:
    config_path = repo_root / "orchestrator.config.yaml"
    config_path.write_text(content, encoding="utf-8")
    return config_path


def _write_fleet_registry(fleet_dir: Path, entries: dict[str, dict]) -> Path:
    fleet_dir.mkdir(parents=True, exist_ok=True)
    registry_path = layout.fleet_registry_path(override=str(fleet_dir))
    payload = {"version": 1, "repos": entries}
    registry_path.write_text(json.dumps(payload), encoding="utf-8")
    return registry_path


def _registry_entry(repo_root: Path) -> dict:
    return {
        "repo_root": str(repo_root),
        "name_with_owner": "test/" + repo_root.name,
        "config_path": str(repo_root / "orchestrator.config.yaml"),
        "state_dir": ".var/charlie-work",
        "first_seen": "2026-01-01T00:00:00Z",
        "last_seen": "2026-01-01T00:00:00Z",
    }


def _fake_gh_with_issue(issue_number: int, *, state: str = "OPEN") -> FakeGitHub:
    gh = FakeGitHub()
    gh.issues.append(
        {
            "number": issue_number,
            "title": f"Remove deprecated key #{issue_number}",
            "url": f"https://example.test/issues/{issue_number}",
            "body": "",
            "labels": [],
            "state": state,
        }
    )
    return gh


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_registers_require_worker_github_token_for_issue_1977():
    match = [
        entry
        for entry in DEPRECATED_CONFIG_KEYS
        if entry.section == "dispatch" and entry.key == "require_worker_github_token"
    ]
    assert len(match) == 1
    entry = match[0]
    assert isinstance(entry, DeprecatedConfigKey)
    assert entry.removal_issue == 1977
    assert entry.replacement is None or isinstance(entry.replacement, str)


def test_deprecated_keys_in_finds_registered_key_in_section():
    data = {"dispatch": {"require_worker_github_token": True}}
    found = deprecated_keys_in(data)
    assert [entry.dotted for entry in found] == ["dispatch.require_worker_github_token"]


def test_deprecated_keys_in_ignores_unregistered_key():
    data = {"dispatch": {"default_limit": 2}, "labels": {"ready": "automated-ready"}}
    assert deprecated_keys_in(data) == []


# ---------------------------------------------------------------------------
# Read-time signal
# ---------------------------------------------------------------------------


def test_load_config_emits_deprecated_read_event(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo_config(
        repo,
        "dispatch:\n  require_worker_github_token: true\n",
    )
    load_config(repo / "orchestrator.config.yaml")

    rows = query_events(_repo_state_path(repo), kind="config_key_deprecated_read")
    assert len(rows) == 1
    payload = rows[0]["payload"]
    assert payload["section"] == "dispatch"
    assert payload["key"] == "require_worker_github_token"
    assert payload["source"] == str(repo / "orchestrator.config.yaml")
    assert payload["issue_number"] == 1977
    assert rows[0]["level"] == "warning"


def test_load_config_no_event_for_unregistered_key(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo_config(
        repo,
        "dispatch:\n  default_limit: 2\n",
    )
    load_config(repo / "orchestrator.config.yaml")

    assert query_events(_repo_state_path(repo), kind="config_key_deprecated_read") == []


def test_layered_config_emits_per_layer(tmp_path: Path):
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    (fleet_dir / layout.GLOBAL_CONFIG_FILENAME).write_text(
        "dispatch:\n  require_worker_github_token: true\n", encoding="utf-8"
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo_config(repo, "dispatch:\n  require_worker_github_token: false\n")

    load_layered_config(repo, fleet_dir_override=str(fleet_dir))

    fleet_rows = query_events(layout.state_file_path(fleet_dir), kind="config_key_deprecated_read")
    assert len(fleet_rows) == 1
    assert fleet_rows[0]["payload"]["source"] == str(fleet_dir / layout.GLOBAL_CONFIG_FILENAME)

    repo_rows = query_events(_repo_state_path(repo), kind="config_key_deprecated_read")
    assert len(repo_rows) == 1
    assert repo_rows[0]["payload"]["source"] == str(repo / "orchestrator.config.yaml")


# ---------------------------------------------------------------------------
# Retirement sweep
# ---------------------------------------------------------------------------

# Derived from the real clock (never a hardcoded calendar date) — the sweep
# only ever compares ``now`` against timestamps it recorded itself, but a
# literal date in a test seed can still rot if a future edit compares it
# against the wall clock.
_NOW = datetime.now(UTC).replace(microsecond=0)


def _run_sweep(
    tmp_path: Path,
    *,
    gh: FakeGitHub,
    now: datetime = _NOW,
    quiet_days: float = 7.0,
    dry_run: bool = False,
) -> dict:
    return run_config_retirement_sweep(
        fleet_dir_override=str(tmp_path / "fleet"),
        quiet_days=quiet_days,
        labels=OrchestratorConfig().labels,
        gh=gh,
        dry_run=dry_run,
        now=now,
    )


def _sidecar(tmp_path: Path) -> dict:
    path = tmp_path / "fleet" / "config_retirement_state.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def test_sweep_key_only_in_untracked_local_layer_keeps_issue_unarmed(tmp_path: Path):
    repo = tmp_path / "repo-a"
    repo.mkdir()
    _write_repo_config(
        repo,
        "dispatch:\n  require_worker_github_token: true\n",
    )
    _write_fleet_registry(tmp_path / "fleet", {"repo-a": _registry_entry(repo)})
    gh = _fake_gh_with_issue(1977)

    # Even long past the quiet window, a key still set somewhere never arms.
    result = _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=30))

    assert gh.labels_added == []
    assert result["keys"]["dispatch.require_worker_github_token"]["present_in"]
    finding = result["keys"]["dispatch.require_worker_github_token"]["present_in"][0]
    assert finding["layer"] == "repo-untracked-local"
    assert finding["repo"] == "repo-a"


def test_sweep_absent_everywhere_does_not_arm_before_quiet_window(tmp_path: Path):
    repo = tmp_path / "repo-a"
    repo.mkdir()
    _write_repo_config(repo, "labels:\n  ready: automated-ready\n")
    _write_fleet_registry(tmp_path / "fleet", {"repo-a": _registry_entry(repo)})
    gh = _fake_gh_with_issue(1977)

    _run_sweep(tmp_path, gh=gh, now=_NOW)
    # Second pass inside the window: still no arm.
    _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=6))

    assert gh.labels_added == []


def test_sweep_arms_after_quiet_window(tmp_path: Path):
    repo = tmp_path / "repo-a"
    repo.mkdir()
    _write_repo_config(repo, "labels:\n  ready: automated-ready\n")
    _write_fleet_registry(tmp_path / "fleet", {"repo-a": _registry_entry(repo)})
    gh = _fake_gh_with_issue(1977)

    _run_sweep(tmp_path, gh=gh, now=_NOW)
    assert gh.labels_added == []

    _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=7))

    assert (1977, "automated-ready") in gh.labels_added
    fleet_state = layout.state_file_path(tmp_path / "fleet")
    armed = query_events(fleet_state, kind="config_key_retirement_armed")
    assert len(armed) == 1
    assert armed[0]["payload"]["issue_number"] == 1977
    # One comment naming repos, layers, and the quiet-window start.
    comments = getattr(gh, "issue_comments_posted", [])
    assert len(comments) == 1
    number, body = comments[0]
    assert number == 1977
    assert "repo-a" in body
    assert "user-global" in body
    assert "repo-" in body  # repo layer classification is named
    assert str(_NOW.date()) in body  # quiet-window start


def test_sweep_arms_at_most_once_across_passes(tmp_path: Path):
    repo = tmp_path / "repo-a"
    repo.mkdir()
    _write_repo_config(repo, "labels:\n  ready: automated-ready\n")
    _write_fleet_registry(tmp_path / "fleet", {"repo-a": _registry_entry(repo)})
    gh = _fake_gh_with_issue(1977)

    _run_sweep(tmp_path, gh=gh, now=_NOW)
    _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=7))
    _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=14))

    assert len(gh.labels_added) == 1
    assert len(getattr(gh, "issue_comments_posted", [])) == 1


def test_sweep_reappearance_resets_quiet_clock(tmp_path: Path):
    repo = tmp_path / "repo-a"
    repo.mkdir()
    config_path = _write_repo_config(repo, "labels:\n  ready: automated-ready\n")
    _write_fleet_registry(tmp_path / "fleet", {"repo-a": _registry_entry(repo)})
    gh = _fake_gh_with_issue(1977)

    _run_sweep(tmp_path, gh=gh, now=_NOW)
    # Key reappears mid-window.
    config_path.write_text("dispatch:\n  require_worker_github_token: true\n", encoding="utf-8")
    _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=4))
    # Key removed again — the quiet clock restarts at the first pass that
    # observes the re-disappearance (+10d), not the original observation.
    config_path.write_text("labels:\n  ready: automated-ready\n", encoding="utf-8")
    _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=10))
    _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=16))
    assert gh.labels_added == []

    _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=17))
    assert (1977, "automated-ready") in gh.labels_added


def test_sweep_reappearance_after_arming_regresses_and_keeps_label(tmp_path: Path):
    repo = tmp_path / "repo-a"
    repo.mkdir()
    config_path = _write_repo_config(repo, "labels:\n  ready: automated-ready\n")
    _write_fleet_registry(tmp_path / "fleet", {"repo-a": _registry_entry(repo)})
    gh = _fake_gh_with_issue(1977)

    _run_sweep(tmp_path, gh=gh, now=_NOW)
    _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=7))
    assert (1977, "automated-ready") in gh.labels_added

    # Key reappears after the issue was marked Ready.
    config_path.write_text("dispatch:\n  require_worker_github_token: true\n", encoding="utf-8")
    _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=8))

    fleet_state = layout.state_file_path(tmp_path / "fleet")
    regressed = query_events(fleet_state, kind="config_key_retirement_regressed")
    assert len(regressed) == 1
    # Commented on the issue, but the Ready label was NOT removed: a worker may
    # already be running.
    assert (1977, "automated-ready") not in gh.labels_removed
    assert len(getattr(gh, "issue_comments_posted", [])) == 2


def test_sweep_closed_removal_issue_is_never_armed(tmp_path: Path):
    repo = tmp_path / "repo-a"
    repo.mkdir()
    _write_repo_config(repo, "labels:\n  ready: automated-ready\n")
    _write_fleet_registry(tmp_path / "fleet", {"repo-a": _registry_entry(repo)})
    gh = _fake_gh_with_issue(1977, state="CLOSED")

    _run_sweep(tmp_path, gh=gh, now=_NOW)
    _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=30))

    assert gh.labels_added == []
    assert getattr(gh, "issue_comments_posted", []) == []


def test_sweep_already_ready_issue_adopts_without_rearming(tmp_path: Path):
    repo = tmp_path / "repo-a"
    repo.mkdir()
    _write_repo_config(repo, "labels:\n  ready: automated-ready\n")
    _write_fleet_registry(tmp_path / "fleet", {"repo-a": _registry_entry(repo)})
    gh = _fake_gh_with_issue(1977)
    gh.issues[-1]["labels"] = [{"name": "automated-ready"}]

    _run_sweep(tmp_path, gh=gh, now=_NOW)
    _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=30))

    # The label was already there: no add, no comment.
    assert gh.labels_added == []
    assert getattr(gh, "issue_comments_posted", []) == []


def test_sweep_quiet_days_comes_from_runtime_config(tmp_path: Path):
    repo = tmp_path / "repo-a"
    repo.mkdir()
    _write_repo_config(repo, "labels:\n  ready: automated-ready\n")
    _write_fleet_registry(tmp_path / "fleet", {"repo-a": _registry_entry(repo)})
    gh = _fake_gh_with_issue(1977)

    _run_sweep(tmp_path, gh=gh, now=_NOW, quiet_days=2.0)
    _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=2, hours=1), quiet_days=2.0)

    assert (1977, "automated-ready") in gh.labels_added


def test_sweep_dry_run_makes_no_writes(tmp_path: Path):
    repo = tmp_path / "repo-a"
    repo.mkdir()
    _write_repo_config(repo, "labels:\n  ready: automated-ready\n")
    _write_fleet_registry(tmp_path / "fleet", {"repo-a": _registry_entry(repo)})
    gh = _fake_gh_with_issue(1977)

    _run_sweep(tmp_path, gh=gh, now=_NOW, dry_run=True)
    _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=30), dry_run=True)

    assert gh.labels_added == []
    assert _sidecar(tmp_path) == {}
    fleet_state = layout.state_file_path(tmp_path / "fleet")
    assert query_events(fleet_state, kind="config_retirement_sweep") == []


def test_sweep_reports_presence_per_pass(tmp_path: Path):
    """The sweep's per-pass event reports where a registered key is still set."""
    repo = tmp_path / "repo-a"
    repo.mkdir()
    _write_repo_config(repo, "dispatch:\n  require_worker_github_token: true\n")
    _write_fleet_registry(tmp_path / "fleet", {"repo-a": _registry_entry(repo)})
    gh = _fake_gh_with_issue(1977)

    _run_sweep(tmp_path, gh=gh, now=_NOW)

    fleet_state = layout.state_file_path(tmp_path / "fleet")
    rows = query_events(fleet_state, kind="config_retirement_sweep")
    assert len(rows) == 1
    keys = rows[0]["payload"]["keys"]
    present = keys["dispatch.require_worker_github_token"]["present_in"]
    assert present[0]["repo"] == "repo-a"
    assert present[0]["layer"] == "repo-untracked-local"
    assert keys["dispatch.require_worker_github_token"]["armed"] is False
    assert rows[0]["payload"]["checked"]  # every layer slot was visited


def test_runtime_config_parses_quiet_days(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    config_path = _write_repo_config(repo, "runtime:\n  config_retirement_quiet_days: 3\n")
    config = load_config(config_path)
    assert config.runtime.config_retirement_quiet_days == 3


def test_runtime_config_quiet_days_default_is_seven():
    assert OrchestratorConfig().runtime.config_retirement_quiet_days == 7
