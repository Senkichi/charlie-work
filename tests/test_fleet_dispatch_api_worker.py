"""API-worker fleet report: ``compute_api_worker_fleet_report`` / ``ApiWorkerFleetReport``.

Split out of ``tests/test_fleet_dispatch.py`` (issue #1557, Track 1) --
bodies are verbatim relocations; shared helpers and the autouse hermeticity
fixtures live in ``tests/_fleet_dispatch_fixtures.py``.
"""

from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any
import pytest
from _fleet_dispatch_fixtures import (
    _API_WORKER_YAML,
    _make_fleet_json,
    _make_repo,
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
)
from charlie_work.fleet_dispatch import (
    ApiWorkerFleetReport,
    compute_api_worker_fleet_report,
)


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
