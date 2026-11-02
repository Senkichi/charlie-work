"""Shared fakes/helpers for the config-deprecation test modules (issue #1976).

Hoisted verbatim out of ``tests/test_config_deprecations.py`` when that module
grew past the 800-line file-size-ratchet cap and was split into seam-named
siblings -- ``test_config_deprecations.py`` keeps the registry, read-time
signal, retirement-sweep, and label-edge tests, while
``test_config_deprecations_wiring.py`` carries the loader<->sweep
layer-pairing contract and ``fleet_loop`` wiring tests. The ``tests/_*.py``
hoisted-fixture convention is the sanctioned import target for shared test
helpers (see ``tests/test_zero_cross_test_import_guard.py``).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from charlie_work import layout
import charlie_work.config_deprecations as config_deprecations_module
from charlie_work.config import OrchestratorConfig
from charlie_work.config_deprecations import DeprecatedConfigKey
from charlie_work.config_retirement_sweep import run_config_retirement_sweep
from charlie_work.subprocess_runner import run_captured
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
    # ``registry`` alone is not enough: the sweep's layer scanner calls
    # ``deprecated_keys_in``, which reads the module-level
    # ``DEPRECATED_CONFIG_KEYS`` -- patch it for the duration of the call so
    # detection and processing see the same synthetic entry.
    with patch.object(config_deprecations_module, "DEPRECATED_CONFIG_KEYS", (_SWEEP_ENTRY,)):
        return run_config_retirement_sweep(
            fleet_dir_override=str(tmp_path / "fleet"),
            quiet_days=quiet_days,
            labels=OrchestratorConfig().labels,
            gh=gh,
            dry_run=dry_run,
            now=now,
            run_git=run_git,
            registry=(_SWEEP_ENTRY,),
        )


def _sidecar(tmp_path: Path) -> dict:
    path = tmp_path / "fleet" / "config_retirement_state.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# Synthetic registry entry injected into every sweep test through
# ``run_config_retirement_sweep(registry=...)`` -- deliberately NOT a live
# config key. Issue #1977 removed ``dispatch.require_worker_github_token``,
# the last non-``supervisor.*`` entry, and #1979 retires that family too, so
# pinning the sweep fixture to a real key just re-breaks the suite on the
# next removal. The issue number is a fixture value -- FakeGitHub fabricates
# the issue view -- not a claim about a real ticket.
_SWEEP_ENTRY = DeprecatedConfigKey(
    section="dispatch",
    key="synthetic_retired_key",
    replacement=None,
    removal_issue=1977,
)
_DOTTED = _SWEEP_ENTRY.dotted
