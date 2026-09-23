"""Allocation-pass wiring and digest emission for ``fleet_loop``.

Split out of ``tests/test_fleet_dispatch.py`` (issue #1557, Track 1) --
bodies are verbatim relocations; shared helpers and the autouse hermeticity
fixtures live in ``tests/_fleet_dispatch_fixtures.py``.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from _fleet_dispatch_fixtures import (
    _allocation_config,
    _make_fleet_json,
    _make_repo,
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
)
from charlie_work import layout
from charlie_work.config import (
    ConfigError,
    OrchestratorConfig,
)
from charlie_work.fleet_dispatch import fleet_loop
from charlie_work.instrumentation import query_events
from ci_fleet.runner_allocation import AllocationPlan
from ci_fleet.runner_allocation_pass import AllocationPassResult
from charlie_work.workflow import CommandResult


@patch("charlie_work.fleet_dispatch.emit_digest")
@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_lane_failure_reaches_real_emit_digest(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    mock_emit_digest: MagicMock,
    tmp_path: Path,
) -> None:
    """#6-G / G-AC2: the lane-failure entry must reach the real ``emit_digest``
    sink, not just the raw ``digest["events"]`` list.

    ``test_fleet_loop_config_load_error_isolated`` proves the raw event dict
    is correct and that ``_build_fleet_attention_digest`` maps it to
    ``health=ERROR`` -- but it calls ``_build_fleet_attention_digest`` itself
    (out of band) and passes ``global_config=None`` to ``fleet_loop``, so the
    real ``if notify_config is not None and notify_config.enabled`` /
    ``if attention_digest.transitions`` gates that guard the actual
    ``emit_digest(...)`` call at the end of ``fleet_loop`` are never entered.
    That leaves open exactly the failure mode this AC exists to close: a gate
    keyed on something only the loop()-succeeded path populates would still
    pass the other test while leaving the desktop/file sink silent. This test
    turns notify on for real and asserts ``emit_digest`` fires with an ERROR
    entry for the failed repo, driving ``fleet_loop`` itself on the exact
    pass where ``app.loop()`` never ran for repo1 -- rather than re-deriving
    the mapping out of band.
    """
    from charlie_work.config import NotifyConfig

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

    repo1_root = tmp_path / "repo1"

    def _load_layered_config_side_effect(
        repo_root: Path, *args: Any, **kwargs: Any
    ) -> OrchestratorConfig:
        if Path(repo_root) == repo1_root:
            raise ConfigError(
                "unknown key(s) in config section 'cross_family': auto_verdict "
                "(valid: enabled, model, command)"
            )
        return OrchestratorConfig()

    mock_load_layered_config.side_effect = _load_layered_config_side_effect
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths

    mock_app2 = MagicMock()
    mock_app2.loop.return_value = CommandResult(True, "repo2 loop complete", {})
    mock_app_class.return_value = mock_app2

    mock_gh = MagicMock()
    mock_gh_class.return_value = mock_gh

    # The real gate: notify_config comes from the *outer* global_config
    # parameter (not from a per-repo loaded config), so this alone drives
    # whether the digest-build-and-emit block at the end of fleet_loop runs.
    global_config = OrchestratorConfig(
        notify=NotifyConfig(
            enabled=True,
            sink="file",
            file_path=str(tmp_path / "digest.jsonl"),
        )
    )

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=global_config,
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
    )

    assert result.data["repos"]["owner/repo1"]["ok"] is False

    # The discriminating assertion: the real sink actually fired, on the pass
    # where repo1's app.loop() never ran (only repo2's did).
    assert mock_emit_digest.called is True
    emitted_digest = mock_emit_digest.call_args[0][1]
    matching = [e for e in emitted_digest.transitions if e.adapter_kind == "owner/repo1"]
    assert len(matching) == 1
    assert matching[0].health == "ERROR"
    assert "cross_family" in (matching[0].last_log_line or "")


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


@patch("charlie_work.fleet_dispatch.run_allocation_pass")
@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_actually_reaches_the_allocation_pass(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    mock_run_allocation_pass: MagicMock,
    tmp_path: Path,
) -> None:
    """Pin the wiring, not just the pieces.

    Every prologue test calls ``_run_fleet_allocation_prologue`` directly and
    every ``run_fleet_supervise`` test patches ``fleet_loop`` out, so nothing
    asserted that a real fleet pass reaches allocation at all. This drives
    ``fleet_loop`` unmocked and patches only one level below the prologue.

    Scope, so this does not misdirect the next triage: it covers ``fleet_loop``
    only. ``run_fleet_supervise``'s own ``load_layered_config(Path.cwd(), ...)``
    and the HEAD-drift and self-deploy steps that run *before* ``fleet_loop`` are
    still unexercised, and #590 could live in any of them.
    """
    repo = _make_repo(tmp_path, "anchor", api_worker=None)
    mock_load_registry.return_value = {
        "repos": {
            "owner/anchor": {
                "repo_root": str(repo),
                "config_path": "orchestrator.config.yaml",
                "state_dir": str(repo / ".var" / "charlie-work"),
            }
        }
    }
    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths
    mock_app = MagicMock()
    mock_app.loop.return_value = CommandResult(True, "ok", {})
    mock_app_class.return_value = mock_app
    mock_run_allocation_pass.return_value = AllocationPassResult(
        ok=True,
        plan=AllocationPlan(budget=8, budget_reason="configured", targets=(), changes=()),
        notes=(),
    )

    fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=_allocation_config(enabled=True, managed_root="C:/actions-runners"),
        repos=None,
        limit=1,
        merge=False,
        dry_run=False,
        work_only=False,
    )

    from ci_fleet.charlie_work_adapter import UNATTENDED_ALLOCATION_SOURCE

    mock_run_allocation_pass.assert_called_once()
    assert mock_run_allocation_pass.call_args.kwargs["dry_run"] is False
    # The daemon must identify itself as the unattended writer: the doctor probe
    # accepts only this value as evidence that allocation runs without an operator.
    assert mock_run_allocation_pass.call_args.kwargs["source"] == UNATTENDED_ALLOCATION_SOURCE


@patch("charlie_work.fleet_dispatch.emit_digest")
@patch("charlie_work.fleet_dispatch.run_allocation_pass")
@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_converged_pass_does_not_emit_digest(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    mock_run_allocation_pass: MagicMock,
    mock_emit_digest: MagicMock,
    tmp_path: Path,
) -> None:
    """Issue #610: a converged pass must not call emit_digest at all.

    A converged allocation pass emits a ``runner_allocation`` event (the
    prologue fires it whenever anything moved *or any note was produced*, and
    standing advisory notes persist for as long as the condition does). That
    event is routed to an explicit ``continue`` in
    ``_build_fleet_attention_digest``, so ``attention_events`` is non-empty
    while ``transitions`` is empty.

    Note: #669's inner ``if attention_digest.transitions:`` gate already
    prevents ``emit_digest`` being called with ``transitions=()`` on this
    scenario — so this test would pass identically without this PR's outer
    gate change. It pins the inner gate's behavior on the
    converged-allocation-note shape. This PR's actual behavior change (the
    outer ``and attention_events`` removal) is exercised by
    ``test_fleet_loop_empty_events_still_builds_digest_when_notify_on``.

    This drives ``fleet_loop`` unmocked (only the prologue's
    ``run_allocation_pass`` and the per-repo ``OrchestratorApp`` are patched)
    so the gate at the emission site is the thing under test. The per-repo
    loop returns an empty ``CommandResult.data`` so no per-repo attention
    events are produced -- the only event is the converged
    ``runner_allocation``.
    """
    from dataclasses import replace

    from charlie_work.config import NotifyConfig

    repo = _make_repo(tmp_path, "anchor", api_worker=None)
    mock_load_registry.return_value = {
        "repos": {
            "owner/anchor": {
                "repo_root": str(repo),
                "config_path": "orchestrator.config.yaml",
                "state_dir": str(repo / ".var" / "charlie-work"),
            }
        }
    }
    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths
    mock_app = MagicMock()
    mock_app.loop.return_value = CommandResult(True, "ok", {})
    mock_app_class.return_value = mock_app
    # Converged: nothing moved, but a standing advisory note is present -- the
    # exact shape every recorded pass on this host had (verified against
    # events.db). The prologue emits runner_allocation because notes is non-empty.
    mock_run_allocation_pass.return_value = AllocationPassResult(
        ok=True,
        plan=AllocationPlan(budget=8, budget_reason="configured", targets=(), changes=()),
        notes=("Senkichi/job-cannon: holding 4 surplus slot(s) - slack for 0/3 pass(es)",),
    )

    cfg = replace(
        _allocation_config(enabled=True, managed_root="C:/actions-runners"),
        notify=NotifyConfig(
            enabled=True,
            sink="file",
            file_path=str(tmp_path / "digest.jsonl"),
        ),
    )

    fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=cfg,
        repos=None,
        limit=1,
        merge=False,
        dry_run=False,
        work_only=False,
    )

    # The raw event list is non-empty (runner_allocation), but every event
    # hit an explicit continue, so transitions is empty and the inner
    # ``if attention_digest.transitions:`` gate (#669) blocks emit_digest.
    # This pins that inner gate on the converged-allocation-note shape; the
    # outer-gate removal this PR makes is covered by the empty-events test.
    mock_emit_digest.assert_not_called()


@patch("charlie_work.fleet_dispatch.emit_digest")
@patch("charlie_work.fleet_dispatch.run_allocation_pass")
@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_empty_events_still_builds_digest_when_notify_on(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    mock_run_allocation_pass: MagicMock,
    mock_emit_digest: MagicMock,
    tmp_path: Path,
) -> None:
    """Issue #610: the genuinely new path this PR opens -- empty attention_events.

    Pre-PR the outer gate was ``notify_config.enabled and attention_events``,
    so a pass with *zero* attention events (no prologue event, no per-repo
    events) skipped the whole digest-build block: ``_build_fleet_attention_digest``
    never ran and the fleet health-state sidecar was never written. This PR
    drops ``and attention_events`` so the digest is built whenever notify is on.

    #669's inner ``if attention_digest.transitions:`` gate already prevents
    ``emit_digest`` being called with ``transitions=()`` on a converged pass
    (the non-empty-events case) -- so this PR's value is *not* fixing a
    currently-reproducing empty-envelope emission. Its value is removing the
    redundant, contradictory outer raw-list test. This test exercises the one
    behavior the diff actually changes: with ``attention_events == []`` and
    notify on, the digest-build / health-state-write path now runs (the
    sidecar file appears on disk), ``emit_digest`` is still not called
    (transitions empty, #669's inner guard), and ``digest["emitted"]`` stays
    ``False``.

    Drives ``fleet_loop`` with ``run_allocation_pass`` returning a fully
    converged result with empty notes (so the prologue emits no event) and the
    per-repo ``OrchestratorApp.loop`` returning empty data (so no per-repo
    events). The only thing under test is the outer gate.
    """
    from dataclasses import replace

    from charlie_work.config import NotifyConfig
    from charlie_work.fleet_dispatch import _fleet_health_state_path

    repo = _make_repo(tmp_path, "anchor", api_worker=None)
    mock_load_registry.return_value = {
        "repos": {
            "owner/anchor": {
                "repo_root": str(repo),
                "config_path": "orchestrator.config.yaml",
                "state_dir": str(repo / ".var" / "charlie-work"),
            }
        }
    }
    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths
    mock_app = MagicMock()
    mock_app.loop.return_value = CommandResult(True, "ok", {})
    mock_app_class.return_value = mock_app
    # Fully converged and quiet: nothing moved, no notes -- the prologue
    # emits no runner_allocation event (started/parked/notes all falsy), so
    # attention_events stays empty.
    mock_run_allocation_pass.return_value = AllocationPassResult(
        ok=True,
        plan=AllocationPlan(budget=8, budget_reason="configured", targets=(), changes=()),
        notes=(),
    )

    cfg = replace(
        _allocation_config(enabled=True, managed_root="C:/actions-runners"),
        notify=NotifyConfig(
            enabled=True,
            sink="file",
            file_path=str(tmp_path / "digest.jsonl"),
        ),
    )

    fleet_dir = tmp_path / "fleet"
    result = fleet_loop(
        fleet_dir_override=str(fleet_dir),
        global_config=cfg,
        repos=None,
        limit=1,
        merge=False,
        dry_run=False,
        work_only=False,
    )

    # The new path: with attention_events empty and notify on, the digest is
    # still built -- the health-state sidecar appears on disk. Pre-PR the
    # outer ``and attention_events`` gate skipped this block entirely, so the
    # file would not exist. This is the discriminating assertion for the diff.
    health_state = _fleet_health_state_path(str(fleet_dir))
    assert health_state.exists(), "digest-build path did not run on empty events"

    # Emission is still gated on the built digest's transitions (#669's
    # inner guard), so no envelope is written and emitted stays False.
    mock_emit_digest.assert_not_called()
    assert result.data["digest"]["emitted"] is False


@patch("charlie_work.fleet_dispatch.emit_digest")
@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_gc_health_baseline_for_unregistered_repo(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    mock_emit_digest: MagicMock,
    tmp_path: Path,
) -> None:
    """Issue #1755: ``fleet_loop`` supplies the live fleet registry to the
    digest filter, so a persisted baseline key for a repo no longer in
    ``fleet.json`` is garbage-collected on the next pass instead of latching
    forever (the live ``owner/repo:-1`` ERROR entry).

    Pins three outcomes at once: the unregistered ``owner/ghost`` key is
    dropped; a registered repo whose lane was skipped this pass
    (``owner/missing``, stale repo_root) keeps its key -- the #817 "absence
    of a check is not evidence of health" protection; and the
    ``self-deploy`` key survives because it is not a repo key at all.
    """
    from charlie_work.config import NotifyConfig
    from charlie_work.fleet_dispatch import (
        _fleet_health_state_path,
        _load_fleet_health_state,
    )

    repo = _make_repo(tmp_path, "anchor", api_worker=None)
    # owner/missing's repo_root does not exist: its lane is skipped as a
    # stale registry entry (never observed), and with no last_seen it is not
    # old enough to prune -- it stays a registered-but-unobserved member.
    mock_load_registry.return_value = {
        "repos": {
            "owner/anchor": {
                "repo_root": str(repo),
                "config_path": "orchestrator.config.yaml",
                "state_dir": str(repo / ".var" / "charlie-work"),
            },
            "owner/missing": {
                "repo_root": str(tmp_path / "missing"),
                "config_path": "orchestrator.config.yaml",
                "state_dir": str(tmp_path / "missing" / ".var" / "charlie-work"),
            },
        }
    }
    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths
    mock_app = MagicMock()
    mock_app.loop.return_value = CommandResult(True, "ok", {})
    mock_app_class.return_value = mock_app
    mock_gh_class.return_value = MagicMock()

    fleet_dir = tmp_path / "fleet"
    health_state = _fleet_health_state_path(str(fleet_dir))
    health_state.parent.mkdir(parents=True, exist_ok=True)
    health_state.write_text(
        json.dumps(
            {
                "version": 1,
                "issues": {
                    "owner/ghost:-1": "ERROR",
                    "owner/missing:9": "ERROR",
                    "self-deploy:-1": "OK",
                },
            }
        ),
        encoding="utf-8",
    )

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

    assert _load_fleet_health_state(health_state) == {
        "owner/missing:9": "ERROR",
        "self-deploy:-1": "OK",
    }


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_records_fleet_lane_completed_event(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """Issue #1078: ``fleet_loop`` records a ``fleet_lane_completed`` event to
    the fleet-level events.db for each repo after every pass, so an operator
    can observe per-repo lane liveness from one query without hand-querying
    each repo's individual events.db. This closes the diagnostic trap where
    silence in the shared fleet log was indistinguishable from a broken fleet.
    """
    from charlie_work.fleet_paths import fleet_dir

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

    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths

    mock_app1 = MagicMock()
    mock_app2 = MagicMock()
    mock_app1.loop.return_value = CommandResult(
        True, "repo1 loop complete", {"pass_skipped": False, "errored": False}
    )
    mock_app2.loop.return_value = CommandResult(
        True, "repo2 loop complete", {"pass_skipped": False, "errored": False}
    )
    mock_app_class.side_effect = [mock_app1, mock_app2]
    mock_gh_class.return_value = MagicMock()

    fleet_dir_override = str(tmp_path / "fleet")
    result = fleet_loop(
        fleet_dir_override=fleet_dir_override,
        global_config=None,
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
    )

    # Both repos ran successfully.
    assert result.data["repos"]["owner/repo1"]["ok"] is True
    assert result.data["repos"]["owner/repo2"]["ok"] is True

    # The fleet-level events.db must carry one fleet_lane_completed event per
    # repo, with the expected payload fields.
    fleet_state_path = layout.state_file_path(fleet_dir(override=fleet_dir_override))
    events = query_events(fleet_state_path, kind="fleet_lane_completed")
    assert len(events) == 2, f"expected 2 fleet_lane_completed events, got {len(events)}"

    by_repo = {e["payload"]["repo_key"]: e for e in events}
    assert set(by_repo) == {"owner/repo1", "owner/repo2"}

    for repo_key, event in by_repo.items():
        assert event["payload"]["ok"] is True
        assert event["payload"]["pass_skipped"] is False
        assert event["payload"]["errored"] is False
        assert "loop complete" in event["payload"]["message"]
        assert event["repo"] == repo_key


# ---------------------------------------------------------------------------
# In-pass deadline enforcement + last_seen rotation (issue #1832)
# ---------------------------------------------------------------------------


class _StepClock:
    """A deterministic fake monotonic clock for deadline tests.

    Returns ``steps`` in order, then repeats ``after`` forever once
    exhausted. Deliberately NOT tied to the exact number of ``pass_clock()``
    calls a given code path makes internally (e.g. a per-repo lane's own
    elapsed-time logging) -- only the calls a test cares about need an
    explicit, distinct step; everything past that reads a constant, so an
    unrelated extra/missing call elsewhere cannot flip the outcome.
    """

    def __init__(self, steps: list[float], after: float) -> None:
        self._steps = list(steps)
        self._after = after
        self._n = 0

    def __call__(self) -> float:
        value = self._steps[self._n] if self._n < len(self._steps) else self._after
        self._n += 1
        return value


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_deadline_defers_later_repos(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """Issue #1832: a pass over its in-pass deadline defers later repos cleanly.

    Three repos are selected explicitly (bypassing registry ordering). The
    fake clock lets repo1's own deadline check pass, then reports the
    deadline exceeded for every call after -- repo2 and repo3's lanes must
    never start (no app.dispatch() call, no per_repo_results entry for
    either), and both must show up in ``data["deferred"]`` instead of being
    counted as failed.
    """
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
            "owner/repo3": {
                "repo_root": str(tmp_path / "repo3"),
                "config_path": "orchestrator.config.yaml",
            },
        }
    }
    mock_load_registry.return_value = registry
    for name in ("repo1", "repo2", "repo3"):
        (tmp_path / name).mkdir()

    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths

    mock_app = MagicMock()
    mock_app.dispatch.return_value = CommandResult(True, "repo1 dispatch complete", {})
    mock_app_class.return_value = mock_app
    mock_gh_class.return_value = MagicMock()

    # steps[0]=pass_started_at, steps[1]=repo1's deadline check (1s elapsed,
    # under the 100s deadline -> repo1 proceeds); every call after reads
    # `after` (10000s elapsed -> over the deadline for repo2/repo3).
    clock = _StepClock(steps=[0.0, 1.0], after=10000.0)

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=("owner/repo1", "owner/repo2", "owner/repo3"),
        work_only=True,
        deadline_seconds=100,
        pass_clock=clock,
    )

    assert result.data["repos"].keys() == {"owner/repo1"}
    assert result.data["repos"]["owner/repo1"]["ok"] is True
    assert result.data["deferred"] == ["owner/repo2", "owner/repo3"]
    # A deferred repo is not a failure -- overall ok stays True.
    assert result.ok is True
    mock_app.dispatch.assert_called_once()

    from charlie_work.fleet_paths import fleet_dir

    fleet_state_path = layout.state_file_path(fleet_dir(override=str(tmp_path / "fleet")))
    deferred_events = query_events(fleet_state_path, kind="fleet_pass_deadline_deferred")
    assert len(deferred_events) == 1
    assert deferred_events[0]["payload"]["deferred_repo_keys"] == ["owner/repo2", "owner/repo3"]


@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_deadline_rotates_last_seen(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    tmp_path: Path,
) -> None:
    """Issue #1832: only repos actually attempted this pass get last_seen bumped.

    _select_repos orders an implicit (no explicit repos=) pass by oldest
    last_seen first. Without bumping last_seen for attempted repos, that
    order never changes pass to pass; a repo deferred every time would keep
    sorting identically to one that always runs. This test proves the fix:
    repo1 (attempted) gets a fresh last_seen; repo2/repo3 (deferred) keep
    their original last_seen, so _select_repos sorts them first next pass.
    """
    from charlie_work.fleet_dispatch import _select_repos
    from charlie_work.fleet_registry import _load_registry

    fleet_dir_path = tmp_path / "fleet"
    old_last_seen = "2020-01-01T00:00:00Z"
    registry_repos = {
        "owner/repo1": {
            "repo_root": str(tmp_path / "repo1"),
            "config_path": "orchestrator.config.yaml",
            "last_seen": old_last_seen,
        },
        "owner/repo2": {
            "repo_root": str(tmp_path / "repo2"),
            "config_path": "orchestrator.config.yaml",
            "last_seen": old_last_seen,
        },
        "owner/repo3": {
            "repo_root": str(tmp_path / "repo3"),
            "config_path": "orchestrator.config.yaml",
            "last_seen": old_last_seen,
        },
    }
    _make_fleet_json(tmp_path, fleet_dir_path, registry_repos)
    for name in ("repo1", "repo2", "repo3"):
        (tmp_path / name).mkdir()

    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths

    mock_app = MagicMock()
    mock_app.dispatch.return_value = CommandResult(True, "repo1 dispatch complete", {})
    mock_app_class.return_value = mock_app
    mock_gh_class.return_value = MagicMock()

    clock = _StepClock(steps=[0.0, 1.0], after=10000.0)
    pass_now = datetime.datetime(2026, 6, 1, tzinfo=datetime.UTC)

    fleet_loop(
        fleet_dir_override=str(fleet_dir_path),
        global_config=None,
        repos=None,
        work_only=True,
        deadline_seconds=100,
        pass_clock=clock,
        now=pass_now,
    )

    updated = _load_registry(fleet_dir_path / "fleet.json")
    expected_stamp = pass_now.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    assert updated["repos"]["owner/repo1"]["last_seen"] == expected_stamp
    assert updated["repos"]["owner/repo2"]["last_seen"] == old_last_seen
    assert updated["repos"]["owner/repo3"]["last_seen"] == old_last_seen

    # Rotation in practice: next pass's implicit ordering now starts with
    # the two repos that were deferred, not the one that just ran.
    next_order = [key for key, _ in _select_repos(updated, None)]
    assert next_order[0] in {"owner/repo2", "owner/repo3"}
    assert next_order[-1] == "owner/repo1"


@patch("charlie_work.fleet_dispatch._run_fleet_autoscale_prologue")
@patch("charlie_work.fleet_dispatch._run_fleet_allocation_prologue")
@patch("charlie_work.fleet_dispatch._load_registry")
def test_fleet_loop_deadline_defers_autoscale_prologue(
    mock_load_registry: MagicMock,
    mock_allocation_prologue: MagicMock,
    mock_autoscale_prologue: MagicMock,
    tmp_path: Path,
) -> None:
    """Issue #1832: the deadline is also checked between the two prologue "lanes".

    An empty registry means the per-repo loop body never runs, isolating
    this test to the allocation-vs-autoscale prologue boundary. The fake
    clock reports the deadline exceeded immediately after the allocation
    prologue returns, so autoscale must never be called.
    """
    mock_load_registry.return_value = {"repos": {}}
    mock_allocation_prologue.return_value = []

    # steps: pass_started_at=0, allocation_lane_start=0, elapsed-log call=5
    # (>= the 3s deadline) -> autoscale prologue must be skipped.
    clock = _StepClock(steps=[0.0, 0.0, 5.0], after=5.0)

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        work_only=False,
        deadline_seconds=3,
        pass_clock=clock,
    )

    mock_allocation_prologue.assert_called_once()
    mock_autoscale_prologue.assert_not_called()
    assert result.data["deferred_autoscale_prologue"] is True
