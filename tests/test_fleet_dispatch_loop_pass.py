"""Allocation-pass wiring and digest emission for ``fleet_loop``.

Split out of ``tests/test_fleet_dispatch.py`` (issue #1557, Track 1) --
bodies are verbatim relocations; shared helpers and the autouse hermeticity
fixtures live in ``tests/_fleet_dispatch_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from _fleet_dispatch_fixtures import (
    _allocation_config,
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
