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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from charlie_work import fleet_lanes, layout
from charlie_work.config import ConfigError, OrchestratorConfig, load_config
from charlie_work.config_deprecations import (
    DEPRECATED_CONFIG_KEYS,
    DeprecatedConfigKey,
    deprecated_keys_in,
)
from charlie_work.config_retirement_sweep import run_config_retirement_sweep
from charlie_work.fleet_dispatch import fleet_loop
from charlie_work.global_config import config_layer_paths, load_layered_config
from charlie_work.instrumentation import query_events
from charlie_work.labels import TransitionOutcome, transition
from charlie_work.subprocess_runner import run_captured
from _fakes_github import FakeGitHub
from _fleet_dispatch_fixtures import (
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
)


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
    run_git=run_captured,
) -> dict:
    return run_config_retirement_sweep(
        fleet_dir_override=str(tmp_path / "fleet"),
        quiet_days=quiet_days,
        labels=OrchestratorConfig().labels,
        gh=gh,
        dry_run=dry_run,
        now=now,
        run_git=run_git,
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


@pytest.mark.parametrize("value", ["true", "-1", "-0.5", '"soon"', "[1]"])
def test_runtime_config_quiet_days_rejects_non_number_and_negative(tmp_path: Path, value: str):
    """quiet_days validation: bool is not a number; negative and non-numeric
    values fail closed at load time (issue #1976)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    config_path = _write_repo_config(repo, f"runtime:\n  config_retirement_quiet_days: {value}\n")
    with pytest.raises(ConfigError, match="config_retirement_quiet_days"):
        load_config(config_path)


# ---------------------------------------------------------------------------
# Retirement sweep — fail-closed enumeration (issue #1976 rework)
# ---------------------------------------------------------------------------

_DOTTED = "dispatch.require_worker_github_token"


def test_sweep_missing_or_unreachable_repo_root_blocks_arming(tmp_path: Path):
    """A registered repo whose repo_root is gone — or never recorded — cannot
    prove the key absent, so it counts as an unreadable layer and holds the
    quiet window even when it would otherwise be long past."""
    gone = tmp_path / "gone"  # never created
    _write_fleet_registry(
        tmp_path / "fleet",
        {
            "repo-a": _registry_entry(gone),
            "repo-b": {
                # registry entry with no repo_root at all
                "name_with_owner": "test/repo-b",
                "config_path": str(tmp_path / "repo-b" / "orchestrator.config.yaml"),
            },
        },
    )
    gh = _fake_gh_with_issue(1977)

    # quiet_days=0 would otherwise arm on the first clean pass.
    result = _run_sweep(tmp_path, gh=gh, now=_NOW, quiet_days=0.0)
    _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=30), quiet_days=0.0)

    assert gh.labels_added == []
    per_key = result["keys"][_DOTTED]
    assert per_key["armed"] is False
    blocked_repos = {rec["repo"] for rec in per_key["unreadable_layers"]}
    assert blocked_repos == {"repo-a", "repo-b"}

    # Once the stale entries are pruned (the #1372 path), the same pass arms.
    _write_fleet_registry(tmp_path / "fleet", {})
    _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=31), quiet_days=0.0)
    assert (1977, "automated-ready") in gh.labels_added


def test_sweep_repo_root_that_is_a_file_blocks_arming(tmp_path: Path):
    not_a_dir = tmp_path / "not-a-dir"
    not_a_dir.write_text("i am a file", encoding="utf-8")
    _write_fleet_registry(tmp_path / "fleet", {"repo-a": _registry_entry(not_a_dir)})
    gh = _fake_gh_with_issue(1977)

    result = _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=30), quiet_days=0.0)

    assert gh.labels_added == []
    blocked = result["keys"][_DOTTED]["unreadable_layers"]
    assert blocked and blocked[0]["repo"] == "repo-a"
    assert "not a readable directory" in blocked[0]["error"]


def test_sweep_unparseable_layer_blocks_arming(tmp_path: Path):
    """A layer file that exists but cannot be parsed proves nothing — the key
    cannot arm while any registered layer is unreadable."""
    repo = tmp_path / "repo-a"
    repo.mkdir()
    _write_repo_config(repo, "dispatch:\n  require_worker_github_token: [\n")
    _write_fleet_registry(tmp_path / "fleet", {"repo-a": _registry_entry(repo)})
    gh = _fake_gh_with_issue(1977)

    result = _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=30), quiet_days=0.0)

    assert gh.labels_added == []
    per_key = result["keys"][_DOTTED]
    assert per_key["armed"] is False
    unreadable = per_key["unreadable_layers"]
    assert len(unreadable) == 1
    assert unreadable[0]["repo"] == "repo-a"
    assert unreadable[0]["error"]


def test_sweep_git_tracked_config_classified_repo_tracked(tmp_path: Path):
    """A repo config known to `git ls-files` is reported as repo-tracked; an
    untracked-local one is not (the removal comment tells operators which is
    which)."""
    tracked = tmp_path / "repo-tracked"
    untracked = tmp_path / "repo-untracked"
    tracked.mkdir()
    untracked.mkdir()
    _write_repo_config(tracked, "dispatch:\n  require_worker_github_token: true\n")
    _write_repo_config(untracked, "dispatch:\n  require_worker_github_token: true\n")
    _write_fleet_registry(
        tmp_path / "fleet",
        {
            "repo-tracked": _registry_entry(tracked),
            "repo-untracked": _registry_entry(untracked),
        },
    )
    gh = _fake_gh_with_issue(1977)

    def _run_git(argv, *, cwd=None, timeout_seconds=None):
        return SimpleNamespace(ok=Path(cwd).name == "repo-tracked")

    result = _run_sweep(tmp_path, gh=gh, run_git=_run_git)

    by_repo = {f["repo"]: f["layer"] for f in result["keys"][_DOTTED]["present_in"]}
    assert by_repo == {
        "repo-tracked": "repo-tracked",
        "repo-untracked": "repo-untracked-local",
    }


def test_sweep_partial_failure_retries_without_recording_armed(tmp_path: Path):
    """A partially-applied label transition must not be recorded as armed —
    the next pass retries the edge instead of believing it already fired."""

    class _FailingAddGitHub(FakeGitHub):
        fail_label_adds = False

        def add_issue_label(self, number: int, label: str) -> bool:
            if self.fail_label_adds:
                return False
            return super().add_issue_label(number, label)

    repo = tmp_path / "repo-a"
    repo.mkdir()
    _write_repo_config(repo, "labels:\n  ready: automated-ready\n")
    _write_fleet_registry(tmp_path / "fleet", {"repo-a": _registry_entry(repo)})
    gh = _FailingAddGitHub()
    gh.issues.append(
        {
            "number": 1977,
            "title": "Remove deprecated key",
            "url": "https://example.test/issues/1977",
            "body": "",
            "labels": [],
            "state": "OPEN",
        }
    )
    gh.fail_label_adds = True

    _run_sweep(tmp_path, gh=gh, now=_NOW)
    result = _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=30))

    # The transition failed: nothing was recorded as armed, no comment posted.
    assert result["keys"][_DOTTED]["armed"] is False
    assert "armed_at" not in _sidecar(tmp_path).get("keys", {}).get(_DOTTED, {})
    assert getattr(gh, "issue_comments_posted", []) == []

    # Next pass with a healthy client retries and arms for real.
    gh.fail_label_adds = False
    result = _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=31))
    assert result["keys"][_DOTTED]["armed"] is True
    assert (1977, "automated-ready") in gh.labels_added


def test_sweep_regression_comment_dedupes_per_episode(tmp_path: Path):
    """A reappearance after arming comments once per episode — a key that
    stays present does not re-comment every pass, but a new episode (absent,
    then present again) reports fresh."""
    repo = tmp_path / "repo-a"
    repo.mkdir()
    config_path = _write_repo_config(repo, "labels:\n  ready: automated-ready\n")
    _write_fleet_registry(tmp_path / "fleet", {"repo-a": _registry_entry(repo)})
    gh = _fake_gh_with_issue(1977)

    _run_sweep(tmp_path, gh=gh, now=_NOW)
    _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=7))
    assert (1977, "automated-ready") in gh.labels_added
    assert len(gh.issue_comments_posted) == 1  # arm comment

    config_path.write_text("dispatch:\n  require_worker_github_token: true\n", encoding="utf-8")
    _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=8))
    assert len(gh.issue_comments_posted) == 2  # regression comment

    # Key still present: no second regression comment for the same episode.
    _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=9))
    assert len(gh.issue_comments_posted) == 2

    # Absent again (clears the per-episode marker), then present again:
    # that is a *new* episode and comments fresh.
    config_path.write_text("labels:\n  ready: automated-ready\n", encoding="utf-8")
    _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=10))
    config_path.write_text("dispatch:\n  require_worker_github_token: true\n", encoding="utf-8")
    _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=11))
    assert len(gh.issue_comments_posted) == 3

    fleet_state = layout.state_file_path(tmp_path / "fleet")
    regressed = query_events(fleet_state, kind="config_key_retirement_regressed")
    assert len(regressed) == 2


def test_sweep_dry_run_reports_would_arm(tmp_path: Path):
    """A dry-run pass past the quiet window reports would_arm without
    touching the issue or the sidecar."""
    repo = tmp_path / "repo-a"
    repo.mkdir()
    _write_repo_config(repo, "labels:\n  ready: automated-ready\n")
    _write_fleet_registry(tmp_path / "fleet", {"repo-a": _registry_entry(repo)})
    gh = _fake_gh_with_issue(1977)

    _run_sweep(tmp_path, gh=gh, now=_NOW)
    result = _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=30), dry_run=True)

    assert result["keys"][_DOTTED]["would_arm"] is True
    assert gh.labels_added == []
    assert getattr(gh, "issue_comments_posted", []) == []
    assert "armed_at" not in _sidecar(tmp_path).get("keys", {}).get(_DOTTED, {})


def test_sweep_corrupt_sidecar_starts_fresh(tmp_path: Path):
    """A corrupt sidecar re-opens the quiet window instead of arming on stale
    or crashing the pass."""
    repo = tmp_path / "repo-a"
    repo.mkdir()
    _write_repo_config(repo, "labels:\n  ready: automated-ready\n")
    _write_fleet_registry(tmp_path / "fleet", {"repo-a": _registry_entry(repo)})
    sidecar_path = layout.config_retirement_state_path(override=str(tmp_path / "fleet"))
    sidecar_path.write_text('{"keys": {"dispatch.require_', encoding="utf-8")
    gh = _fake_gh_with_issue(1977)

    result = _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=30))

    assert "error" not in result
    assert result["keys"][_DOTTED]["armed"] is False
    # The pass rewrote the sidecar as valid JSON with a fresh quiet window.
    reloaded = _sidecar(tmp_path)
    assert reloaded["keys"][_DOTTED]["absent_since"]
    assert "armed_at" not in reloaded["keys"][_DOTTED]


def test_sweep_quiet_days_zero_arms_on_first_clean_pass(tmp_path: Path):
    """quiet_days=0 is the documented opt-out: the first pass that observes
    the key absent everywhere arms immediately — no second pass required."""
    repo = tmp_path / "repo-a"
    repo.mkdir()
    _write_repo_config(repo, "labels:\n  ready: automated-ready\n")
    _write_fleet_registry(tmp_path / "fleet", {"repo-a": _registry_entry(repo)})
    gh = _fake_gh_with_issue(1977)

    _run_sweep(tmp_path, gh=gh, now=_NOW, quiet_days=0.0)

    assert (1977, "automated-ready") in gh.labels_added


# ---------------------------------------------------------------------------
# Label edge
# ---------------------------------------------------------------------------


def test_config_retirement_ready_edge_marks_ready_and_clears_workflow():
    """The config_retirement_ready edge adds the ready label and strips every
    workflow label so a removal issue arrives at dispatch as a clean
    candidate — and it never strips the ready label it just added."""
    labels = OrchestratorConfig().labels
    gh = FakeGitHub()
    gh.issues[0]["labels"] = [
        {"name": "agent:in-progress"},
        {"name": "agent:needs-rework"},
    ]

    result = transition(gh, labels, 123, "config_retirement_ready")

    assert result.outcome is TransitionOutcome.APPLIED
    assert (123, "automated-ready") in gh.labels_added
    assert (123, "agent:in-progress") in gh.labels_removed
    assert (123, "agent:needs-rework") in gh.labels_removed
    assert all(label != "automated-ready" for _, label in gh.labels_removed)


# ---------------------------------------------------------------------------
# Layer-pairing contract (loader <-> sweep)
# ---------------------------------------------------------------------------


def test_load_layered_config_consumes_config_layer_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """The loader resolves its layer files through config_layer_paths — the
    sweep enumerates the same function, so repointing it repoints the loader's
    reads too (this is the pin that keeps the two from drifting)."""
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    global_layer = fleet_dir / "config.yaml"
    global_layer.write_text("labels:\n  ready: fleet-ready\n", encoding="utf-8")
    repo_layer = tmp_path / "elsewhere.yaml"
    repo_layer.write_text("dispatch:\n  default_limit: 9\n", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()

    def _layers(repo_root, explicit=None, *, fleet_dir_override=None):
        return (("user-global", global_layer), ("repo", repo_layer))

    monkeypatch.setattr("charlie_work.global_config.config_layer_paths", _layers)

    config = load_layered_config(repo, fleet_dir_override=str(fleet_dir))

    assert config.labels.ready == "fleet-ready"
    assert config.dispatch.default_limit == 9
    assert set(config.sources) == {str(global_layer), str(repo_layer)}


def test_config_layer_paths_match_what_the_loader_reads(tmp_path: Path):
    """The set of existing layer files the loader reports in ``sources`` is
    exactly the existing subset of config_layer_paths; absent layers are still
    enumerated as slots for the sweep."""
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    (fleet_dir / "config.yaml").write_text("labels:\n  ready: fleet-ready\n", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo_config(repo, "dispatch:\n  default_limit: 9\n")

    config = load_layered_config(repo, fleet_dir_override=str(fleet_dir))

    slot_paths = {
        str(path) for _, path in config_layer_paths(repo, fleet_dir_override=str(fleet_dir))
    }
    assert set(config.sources) == slot_paths

    # An absent repo layer is still a slot (the sweep counts it "checked").
    empty_repo = tmp_path / "empty"
    empty_repo.mkdir()
    slots = dict(config_layer_paths(empty_repo, fleet_dir_override=str(fleet_dir)))
    assert slots["repo"] == empty_repo / "orchestrator.config.yaml"
    assert slots["user-global"] == fleet_dir / "config.yaml"


# ---------------------------------------------------------------------------
# fleet_loop wiring
# ---------------------------------------------------------------------------


@patch("charlie_work.fleet_dispatch.compute_api_worker_fleet_report")
@patch("charlie_work.fleet_dispatch._run_fleet_config_retirement_sweep")
@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_runs_retirement_sweep_and_surfaces_attention(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    mock_sweep: MagicMock,
    mock_compute_report: MagicMock,
    tmp_path: Path,
):
    """fleet_loop invokes the retirement sweep once per pass and its
    attention entries reach the consolidated digest (issue #1976)."""
    mock_load_registry.return_value = {"repos": {}}
    mock_compute_report.return_value = None
    mock_sweep.return_value = [
        {
            "repo_key": "fleet",
            "type": "config_retirement_armed",
            "key": _DOTTED,
            "issue_number": 1977,
            "reason": f"{_DOTTED} absent everywhere for 7.0d; removal issue marked Ready",
        }
    ]

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=1,
        merge=None,
        dry_run=True,
        work_only=True,
    )

    mock_sweep.assert_called_once()
    sweep_args = mock_sweep.call_args.args
    assert sweep_args[0] == str(tmp_path / "fleet")  # fleet_dir_override
    digest = result.data["digest"]
    assert digest["count"] >= 1
    assert any(e.get("type") == "config_retirement_armed" for e in digest["events"])


def test_retirement_sweep_wrapper_survives_client_construction_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """If the GitHub client cannot even be constructed, the fleet-loop helper
    degrades to a config_retirement_error attention entry instead of raising
    into the pass (issue #1976 — 'the sweep never raises' at the call site)."""

    def _boom(**_kwargs):
        raise RuntimeError("gh ctor boom")

    monkeypatch.setattr(fleet_lanes, "GitHub", _boom)

    entries = fleet_lanes._run_fleet_config_retirement_sweep(
        str(tmp_path / "fleet"), OrchestratorConfig(), True, _NOW
    )

    assert [e["type"] for e in entries] == ["config_retirement_error"]
