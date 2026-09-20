"""``fleet_loop`` health-baseline GC tests against a real on-disk fleet.json.

Issue #1755 round-2: the registry-membership GC in
``reconcile_fleet_health_baselines`` (fleet_health_baseline.py) is fed by
``fleet_loop`` with the pass-start registry minus this pass's pruned keys,
and must FAIL CLOSED -- ``_load_registry`` collapses a missing, corrupt, or
empty ``fleet.json`` into ``{"repos": {}}``, so an empty derived set is
indistinguishable from an unreadable registry and is passed as ``None``
(GC disabled) rather than ``frozenset()`` (delete every repo-shaped key).

Unlike ``test_fleet_dispatch_loop_pass.py``'s GC test, these do NOT mock
``fleet_dispatch._load_registry`` -- the pass-start registry, the prune's
own reload, and the derived membership set all go through the real loader,
so the collapse behavior under test is the production one.

Shared helpers and the autouse hermeticity fixtures live in
``tests/_fleet_dispatch_fixtures.py``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _fleet_dispatch_fixtures import (
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
)
from charlie_work.config import OrchestratorConfig
from charlie_work.fleet_dispatch import fleet_loop


def _write_health_baseline(fleet_dir: Path, issues: dict[str, str]) -> Path:
    from charlie_work.fleet_dispatch import _fleet_health_state_path

    health_state = _fleet_health_state_path(str(fleet_dir))
    health_state.write_text(json.dumps({"version": 1, "issues": issues}), encoding="utf-8")
    return health_state


def _load_health_baseline(health_state: Path) -> dict[str, str]:
    from charlie_work.fleet_dispatch import _load_fleet_health_state

    return _load_fleet_health_state(health_state)


@pytest.mark.parametrize("registry_state", ["missing", "corrupt", "empty"])
def test_fleet_loop_unreadable_or_empty_registry_preserves_health_baseline(
    registry_state: str,
    tmp_path: Path,
) -> None:
    """Issue #1755 round-2 (fail-closed GC): ``_load_registry`` collapses a
    missing, corrupt, or empty ``fleet.json`` into ``{"repos": {}}``
    (fleet_registry.py:71-89). ``fleet_loop`` must not hand that to the
    digest filter as ``frozenset()`` -- an empty registered-repo set read
    literally would delete every ``/``-shaped baseline key, inverting #817's
    "absence is not evidence" protection: a registry that cannot be
    positively read as non-empty is evidence of nothing. With zero positive
    registry reads the pass must leave every baseline key untouched.
    """
    from charlie_work.config import NotifyConfig

    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir(parents=True)
    if registry_state == "corrupt":
        (fleet_dir / "fleet.json").write_text("{not valid json", encoding="utf-8")
    elif registry_state == "empty":
        (fleet_dir / "fleet.json").write_text(
            json.dumps({"version": 1, "repos": {}}), encoding="utf-8"
        )
    # "missing" writes no fleet.json at all; all three states collapse
    # identically through _load_registry.

    baseline = {
        "owner/ghost:-1": "ERROR",
        "owner/missing:9": "ERROR",
        "self-deploy:-1": "OK",
    }
    health_state = _write_health_baseline(fleet_dir, baseline)

    global_config = OrchestratorConfig(
        notify=NotifyConfig(
            enabled=True,
            sink="file",
            file_path=str(tmp_path / "digest.jsonl"),
        )
    )

    fleet_loop(
        fleet_dir_override=str(fleet_dir),
        global_config=global_config,
        repos=None,
        limit=1,
        merge=False,
        dry_run=False,
        work_only=False,
    )

    assert _load_health_baseline(health_state) == baseline


def test_fleet_loop_gc_drops_baseline_key_for_repo_pruned_this_pass(
    tmp_path: Path,
) -> None:
    """Issue #1755 round-2: the registered-repo set handed to the digest
    filter is the pass-start registry minus keys pruned THIS pass (the prune
    is what removes a stale entry from the membership view). A repo pruned
    after its grace period has its baseline keys GC'd in the same pass; a
    repo stale-but-within-grace keeps its keys (registered but unobserved --
    the #817 protection); a never-registered key is residue and drops.
    """
    from charlie_work.config import NotifyConfig, RuntimeConfig

    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir(parents=True)
    now = datetime.now(UTC)
    old_ts = (now - timedelta(days=10)).isoformat().replace("+00:00", "Z")
    recent_ts = (now - timedelta(days=2)).isoformat().replace("+00:00", "Z")

    fleet_json = {
        "version": 1,
        "repos": {
            # repo_root gone AND last_seen past the 7-day grace -> pruned
            # this pass -> absent from the derived registered set.
            "owner/dead": {
                "repo_root": str(tmp_path / "dead"),
                "name_with_owner": "owner/dead",
                "config_path": str(tmp_path / "dead" / "orchestrator.config.yaml"),
                "state_dir": str(tmp_path / "dead-state"),
                "first_seen": old_ts,
                "last_seen": old_ts,
            },
            # repo_root gone but last_seen within grace -> skipped, NOT
            # pruned -> still a registered (unobserved) member.
            "owner/missing": {
                "repo_root": str(tmp_path / "missing"),
                "name_with_owner": "owner/missing",
                "config_path": str(tmp_path / "missing" / "orchestrator.config.yaml"),
                "state_dir": str(tmp_path / "missing-state"),
                "first_seen": recent_ts,
                "last_seen": recent_ts,
            },
        },
    }
    (fleet_dir / "fleet.json").write_text(json.dumps(fleet_json), encoding="utf-8")

    baseline = {
        "owner/dead:3": "ERROR",
        "owner/ghost:-1": "ERROR",
        "owner/missing:9": "ERROR",
        "self-deploy:-1": "OK",
    }
    health_state = _write_health_baseline(fleet_dir, baseline)

    global_config = OrchestratorConfig(
        runtime=RuntimeConfig(fleet_registry_stale_grace_days=7),
        notify=NotifyConfig(
            enabled=True,
            sink="file",
            file_path=str(tmp_path / "digest.jsonl"),
        ),
    )

    fleet_loop(
        fleet_dir_override=str(fleet_dir),
        global_config=global_config,
        repos=None,
        limit=1,
        merge=False,
        dry_run=False,
        work_only=False,
        now=now,
    )

    assert _load_health_baseline(health_state) == {
        "owner/missing:9": "ERROR",
        "self-deploy:-1": "OK",
    }
