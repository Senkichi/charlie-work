"""Tests for the deprecated-config-key registry, read-time signal, and
fleet retirement sweep (issue #1976).

The registry (``config_deprecations.DEPRECATED_CONFIG_KEYS``) is the single
source of truth for deprecated config keys. Config loading emits
``config_key_deprecated_read`` per registered key per layer file read; the
fleet pass runs ``run_config_retirement_sweep`` once to track per-key quiet
windows and mark the removal issue Ready once a key has been absent from
every layer of every registered repo for ``runtime.config_retirement_quiet_days``.

The loader<->sweep layer-pairing contract and ``fleet_loop`` wiring tests live
in the sibling ``tests/test_config_deprecations_wiring.py``; shared helpers
are hoisted to ``tests/_config_deprecations_fixtures.py``.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from charlie_work import layout
from charlie_work.config import ConfigError, OrchestratorConfig, load_config
from charlie_work.config_deprecations import (
    DEPRECATED_CONFIG_KEYS,
    deprecated_keys_in,
)
from charlie_work.global_config import load_layered_config
from charlie_work.instrumentation import query_events
from charlie_work.labels import TransitionOutcome, transition
from _config_deprecations_fixtures import (
    _DOTTED,
    _NOW,
    _fake_gh_with_issue,
    _registry_entry,
    _repo_state_path,
    _run_sweep,
    _sidecar,
    _write_fleet_registry,
    _write_repo_config,
)
from _fakes_github import FakeGitHub


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_registers_require_worker_github_token_for_issue_1977(tmp_path: Path):
    """Issue #1977: the registration this leaf name describes was itself the
    removal target -- the registry must now carry NO entry for
    ``dispatch.require_worker_github_token``, a config file that still sets
    it fails with the normal unknown-key error, and no
    ``config_key_deprecated_read`` event fires (the key is gone, not
    deprecated).

    The leaf name predates the removal and is kept verbatim: the
    collect-only gate (issue #1538) fails a required check on any leaf-name
    removal, rename included, absent the operator-applied
    ``collect-gate-exempt`` label."""
    assert all(
        entry.dotted != "dispatch.require_worker_github_token" for entry in DEPRECATED_CONFIG_KEYS
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    config_path = _write_repo_config(repo, "dispatch:\n  require_worker_github_token: true\n")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(config_path)
    assert query_events(_repo_state_path(repo), kind="config_key_deprecated_read") == []


def test_deprecated_keys_in_finds_registered_key_in_section():
    data = {"supervisor": {"zero_pass_alarm": 5}}
    found = deprecated_keys_in(data)
    assert [entry.dotted for entry in found] == ["supervisor.zero_pass_alarm"]


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
        "supervisor:\n  zero_pass_alarm: 5\n",
    )
    load_config(repo / "orchestrator.config.yaml")

    rows = query_events(_repo_state_path(repo), kind="config_key_deprecated_read")
    assert len(rows) == 1
    payload = rows[0]["payload"]
    assert payload["section"] == "supervisor"
    assert payload["key"] == "zero_pass_alarm"
    assert payload["source"] == str(repo / "orchestrator.config.yaml")
    assert payload["issue_number"] == 1979
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
        "supervisor:\n  zero_pass_alarm: 9\n", encoding="utf-8"
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo_config(repo, "supervisor:\n  wedge_kill_loop_alarm: 2\n")

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


def test_sweep_key_only_in_untracked_local_layer_keeps_issue_unarmed(tmp_path: Path):
    repo = tmp_path / "repo-a"
    repo.mkdir()
    _write_repo_config(
        repo,
        "dispatch:\n  synthetic_retired_key: true\n",
    )
    _write_fleet_registry(tmp_path / "fleet", {"repo-a": _registry_entry(repo)})
    gh = _fake_gh_with_issue(1977)

    # Even long past the quiet window, a key still set somewhere never arms.
    result = _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=30))

    assert gh.labels_added == []
    assert result["keys"][_DOTTED]["present_in"]
    finding = result["keys"][_DOTTED]["present_in"][0]
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
    config_path.write_text("dispatch:\n  synthetic_retired_key: true\n", encoding="utf-8")
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
    config_path.write_text("dispatch:\n  synthetic_retired_key: true\n", encoding="utf-8")
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
    _write_repo_config(repo, "dispatch:\n  synthetic_retired_key: true\n")
    _write_fleet_registry(tmp_path / "fleet", {"repo-a": _registry_entry(repo)})
    gh = _fake_gh_with_issue(1977)

    _run_sweep(tmp_path, gh=gh, now=_NOW)

    fleet_state = layout.state_file_path(tmp_path / "fleet")
    rows = query_events(fleet_state, kind="config_retirement_sweep")
    assert len(rows) == 1
    keys = rows[0]["payload"]["keys"]
    present = keys[_DOTTED]["present_in"]
    assert present[0]["repo"] == "repo-a"
    assert present[0]["layer"] == "repo-untracked-local"
    assert keys[_DOTTED]["armed"] is False
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
    _write_repo_config(repo, "dispatch:\n  synthetic_retired_key: [\n")
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
    _write_repo_config(tracked, "dispatch:\n  synthetic_retired_key: true\n")
    _write_repo_config(untracked, "dispatch:\n  synthetic_retired_key: true\n")
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

    config_path.write_text("dispatch:\n  synthetic_retired_key: true\n", encoding="utf-8")
    _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=8))
    assert len(gh.issue_comments_posted) == 2  # regression comment

    # Key still present: no second regression comment for the same episode.
    _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=9))
    assert len(gh.issue_comments_posted) == 2

    # Absent again (clears the per-episode marker), then present again:
    # that is a *new* episode and comments fresh.
    config_path.write_text("labels:\n  ready: automated-ready\n", encoding="utf-8")
    _run_sweep(tmp_path, gh=gh, now=_NOW + timedelta(days=10))
    config_path.write_text("dispatch:\n  synthetic_retired_key: true\n", encoding="utf-8")
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
    sidecar_path.write_text('{"keys": {"dispatch.synthetic_', encoding="utf-8")
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

    result = transition(gh, labels, 123, "config_retirement_ready", state_path=None)

    assert result.outcome is TransitionOutcome.APPLIED
    assert (123, "automated-ready") in gh.labels_added
    assert (123, "agent:in-progress") in gh.labels_removed
    assert (123, "agent:needs-rework") in gh.labels_removed
    assert all(label != "automated-ready" for _, label in gh.labels_removed)
