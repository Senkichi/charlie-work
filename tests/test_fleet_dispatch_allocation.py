"""Runner-allocation and autoscale prologues for the fleet pass.

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
    _make_fleet_json,
    _make_repo,
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
)
from charlie_work import layout
from charlie_work.doctor import _check_runner_allocation
from charlie_work.config import (
    OrchestratorConfig,
    RunnerAllocationConfig,
    RunnerScalingConfig,
)
from charlie_work.fleet_dispatch import (
    _CiFleetDirtyCheck,
    _run_fleet_allocation_prologue,
    _run_fleet_autoscale_prologue,
)
from charlie_work.instrumentation import query_events
from ci_fleet.charlie_work_adapter import (
    ALLOCATION_STATE_FILENAME,
    ScaleAction,
    load_allocation_stamp,
)
from ci_fleet.runners import ScaleDecision
from ci_fleet.runner_allocation import (
    AllocationPlan,
    SlotAction,
    SlotChange,
    SlotChangeResult,
)
from ci_fleet.runner_allocation_pass import AllocationPassResult


def test_allocation_prologue_anchors_on_a_live_repo_and_passes_config_through(
    tmp_path: Path,
) -> None:
    """The prologue's whole job: find an anchor, hand the pass its wiring."""
    fleet_dir = tmp_path / "fleet"
    repo = _make_repo(tmp_path, "anchor", api_worker=None)
    _make_fleet_json(
        tmp_path,
        fleet_dir,
        {
            "owner/anchor": {
                "repo_root": str(repo),
                "state_dir": str(repo / ".var" / "charlie-work"),
            }
        },
    )

    plan = AllocationPlan(budget=8, budget_reason="configured", targets=(), changes=())
    result = AllocationPassResult(ok=True, plan=plan, notes=("cw: pinned",))

    with (
        patch("charlie_work.fleet_dispatch.run_allocation_pass", return_value=result) as pass_mock,
        patch("charlie_work.fleet_dispatch.GitHub") as gh_mock,
    ):
        events = _run_fleet_allocation_prologue(
            str(fleet_dir),
            _allocation_config(enabled=True, managed_root="C:/actions-runners"),
            dry_run=True,
        )

    assert gh_mock.call_args.kwargs["repo_root"] == repo
    kwargs = pass_mock.call_args.kwargs
    assert kwargs["managed_root_fallback"] == "C:/fallback-root"
    assert kwargs["fleet_dir_override"] == str(fleet_dir)
    # Issue #603: the state_path is the fleet-level path, not the anchor
    # repo's per-repo state.json. Host-wide allocation events go to the
    # fleet-level events.db, not whichever repo sorted first in the registry.
    assert kwargs["state_path"] == fleet_dir / "state.json"
    assert kwargs["dry_run"] is True
    # The driving interval is threaded from the caller's resolved config so the
    # state file records the cadence the daemon actually used (issue #606).
    assert kwargs["full_pass_interval_seconds"] == 300

    # A note alone is enough to surface an event; a balanced host stays quiet.
    assert [event["type"] for event in events] == ["runner_allocation"]
    assert events[0]["budget"] == 8
    assert events[0]["dry_run"] is True


def test_allocation_prologue_delegated_skip_tolerates_no_anchor_state(tmp_path: Path) -> None:
    """A registry entry with no recorded ``state_dir`` must not crash.

    Pre-#603, ``anchor_state`` was ``None`` whenever the anchor repo's registry
    entry had no ``state_dir`` on file yet (e.g. its very first pass), so the
    durable events.db row was skipped and only the in-memory digest survived.
    Post-#603, the event-store path is derived from ``fleet_dir()``, not from
    the anchor's ``state_dir``, so the decline is always durably recorded in
    the fleet-level events.db regardless of the registry entry's state_dir.
    """
    fleet_dir = tmp_path / "fleet"
    repo = _make_repo(tmp_path, "anchor", api_worker=None)
    _make_fleet_json(
        tmp_path,
        fleet_dir,
        {"owner/anchor": {"repo_root": str(repo)}},  # no state_dir
    )

    declined = AllocationPassResult(ok=True, skipped=True, notes=("declined",))
    with (
        patch("charlie_work.fleet_dispatch.run_allocation_pass", return_value=declined),
        patch("charlie_work.fleet_dispatch.GitHub"),
    ):
        events = _run_fleet_allocation_prologue(
            str(fleet_dir), _allocation_config(enabled=True), dry_run=False
        )

    assert [event["type"] for event in events] == ["runner_allocation_skipped"]
    assert events[0]["reason"] == "declined"

    # Issue #603: the durable record lands in the fleet-level events.db even
    # though the anchor entry has no state_dir — the path no longer depends on
    # the registry entry that supplied the gh anchor.
    fleet_state_path = fleet_dir / "state.json"
    rows = query_events(fleet_state_path, kind="runner_allocation_skipped")
    assert len(rows) == 1
    assert rows[0]["payload"]["reason"] == "declined"


def test_allocation_prologue_forces_dry_run_when_ci_fleet_is_dirty(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Issue #927: a dirty ci_fleet src/ forces a dry run and emits a guard event."""
    fleet_dir = tmp_path / "fleet"
    repo = _make_repo(tmp_path, "anchor", api_worker=None)
    state_dir = repo / ".var" / "charlie-work"
    _make_fleet_json(
        tmp_path,
        fleet_dir,
        {"owner/anchor": {"repo_root": str(repo), "state_dir": str(state_dir)}},
    )

    dirty_check = _CiFleetDirtyCheck(
        is_dirty=True,
        repo_root=tmp_path / "ci_fleet",
        dirty_paths=(" M src/planner.py",),
    )
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch._ci_fleet_worktree_dirty",
        lambda _module_file=None: dirty_check,
    )

    plan = AllocationPlan(budget=8, budget_reason="configured", targets=(), changes=())
    result = AllocationPassResult(ok=True, plan=plan, notes=("cw: pinned",))

    with (
        patch("charlie_work.fleet_dispatch.run_allocation_pass", return_value=result) as pass_mock,
        patch("charlie_work.fleet_dispatch.GitHub"),
    ):
        events = _run_fleet_allocation_prologue(
            str(fleet_dir), _allocation_config(enabled=True), dry_run=False
        )

    assert pass_mock.call_args.kwargs["dry_run"] is True
    assert any(e["type"] == "ci_fleet_worktree_dirty" for e in events)

    # Issue #603: the guard event lands in the fleet-level events.db, not the
    # anchor repo's per-repo database.
    fleet_state_path = fleet_dir / "state.json"
    rows = query_events(fleet_state_path, kind="ci_fleet_worktree_dirty")
    assert len(rows) == 1
    assert rows[0]["payload"]["dirty_paths"] == [" M src/planner.py"]
    assert rows[0]["payload"]["dry_run_forced"] is True
    assert rows[0]["level"] == "warning"


def test_allocation_prologue_is_inert_when_disabled(tmp_path: Path) -> None:
    """Default-off must mean off: no registry read, no gh client, no events."""
    with patch("charlie_work.fleet_dispatch.run_allocation_pass") as pass_mock:
        events = _run_fleet_allocation_prologue(
            str(tmp_path / "fleet"),
            _allocation_config(enabled=False),
            dry_run=False,
        )

    assert events == []
    pass_mock.assert_not_called()


def test_allocation_prologue_keeps_original_dry_run_when_ci_fleet_is_clean(
    tmp_path: Path,
) -> None:
    """A clean ci_fleet tree must not force dry_run on an actuating pass."""
    fleet_dir = tmp_path / "fleet"
    repo = _make_repo(tmp_path, "anchor", api_worker=None)
    _make_fleet_json(
        tmp_path,
        fleet_dir,
        {"owner/anchor": {"repo_root": str(repo), "state_dir": str(repo / ".var")}},
    )
    balanced = AllocationPassResult(
        ok=True,
        plan=AllocationPlan(budget=8, budget_reason="configured", targets=(), changes=()),
    )

    with (
        patch(
            "charlie_work.fleet_dispatch.run_allocation_pass", return_value=balanced
        ) as pass_mock,
        patch("charlie_work.fleet_dispatch.GitHub"),
    ):
        _run_fleet_allocation_prologue(
            str(fleet_dir), _allocation_config(enabled=True), dry_run=False
        )

    assert pass_mock.call_args.kwargs["dry_run"] is False


def test_allocation_prologue_logs_its_inputs_before_any_branch(tmp_path: Any, caplog: Any) -> None:
    """The entry line must be unconditional.

    Regression guard for the evidence gap that kept #590 unisolated: every skip
    path used to log from *inside* a branch, so an absent log line was equally
    consistent with "reached and declined" and "never reached". One line before
    the first branch separates those two readings, which is the only thing that
    distinguishes a misconfigured host from an unreached call site.
    """
    import logging

    class _NoSection:
        pass

    with caplog.at_level(logging.INFO, logger="charlie_work.fleet_dispatch"):
        _run_fleet_allocation_prologue(str(tmp_path / "fleet"), _NoSection(), False)

    # Logged even in the most degenerate case, where there is no section to read.
    assert "prologue: entered" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="charlie_work.fleet_dispatch"):
        _run_fleet_allocation_prologue(
            str(tmp_path / "fleet"),
            _allocation_config(enabled=False, managed_root="C:/actions-runners", **{}),
            False,
        )

    assert "prologue: entered" in caplog.text
    assert "enabled=False" in caplog.text
    assert "C:/actions-runners" in caplog.text


def test_allocation_prologue_records_a_delegated_skip(tmp_path: Path) -> None:
    """Issue #958: ``run_allocation_pass`` declining must reach the digest
    and events.db.

    Pre-fix, this branch did not exist: control fell through to the
    started/parked/notes check below. Both current ci_fleet skip branches
    populate ``notes``, so a clean decline (``skipped=True``, ``error=None``)
    was misfiled as a healthy no-op ``runner_allocation`` event -- which the
    digest deliberately drops as noise -- so it reached neither the notify
    digest nor events.db. It must now surface as a durable
    ``runner_allocation_skipped`` event in both places instead.
    """
    fleet_dir = tmp_path / "fleet"
    repo = _make_repo(tmp_path, "anchor", api_worker=None)
    state_dir = repo / ".var"
    _make_fleet_json(
        tmp_path,
        fleet_dir,
        {"owner/anchor": {"repo_root": str(repo), "state_dir": str(state_dir)}},
    )

    declined = AllocationPassResult(
        ok=True,
        skipped=True,
        notes=("no configured runners found under C:/actions-runners",),
    )
    with (
        patch("charlie_work.fleet_dispatch.run_allocation_pass", return_value=declined),
        patch("charlie_work.fleet_dispatch.GitHub"),
    ):
        events = _run_fleet_allocation_prologue(
            str(fleet_dir), _allocation_config(enabled=True), dry_run=False
        )

    assert events == [
        {
            "repo_key": "fleet",
            "type": "runner_allocation_skipped",
            "reason": "no configured runners found under C:/actions-runners",
        }
    ]

    # And a genuine events.db row, not just the in-memory digest -- this is
    # the actual durable record the issue's evidence section was about.
    # Issue #603: the row lands in the fleet-level events.db, not the anchor
    # repo's per-repo database.
    fleet_state_path = fleet_dir / "state.json"
    rows = query_events(fleet_state_path, kind="runner_allocation_skipped")
    assert len(rows) == 1
    assert rows[0]["payload"]["reason"] == "no configured runners found under C:/actions-runners"
    assert rows[0]["payload"]["dry_run"] is False
    assert rows[0]["level"] == "warning"


def test_allocation_prologue_records_a_delegated_skip_with_no_notes(tmp_path: Path) -> None:
    """A delegated skip with no notes at all must still leave a trace.

    ``AllocationPassResult`` carries no dedicated reason field, so if a future
    ci_fleet skip branch ever returns ``skipped=True`` with empty notes, the
    digest must still record that the pass declined rather than silently
    dropping it the way the pre-fix code did for every delegated skip.
    """
    fleet_dir = tmp_path / "fleet"
    repo = _make_repo(tmp_path, "anchor", api_worker=None)
    _make_fleet_json(
        tmp_path,
        fleet_dir,
        {"owner/anchor": {"repo_root": str(repo), "state_dir": str(repo / ".var")}},
    )

    declined = AllocationPassResult(ok=True, skipped=True)
    with (
        patch("charlie_work.fleet_dispatch.run_allocation_pass", return_value=declined),
        patch("charlie_work.fleet_dispatch.GitHub"),
    ):
        events = _run_fleet_allocation_prologue(
            str(fleet_dir), _allocation_config(enabled=True), dry_run=False
        )

    assert [event["type"] for event in events] == ["runner_allocation_skipped"]
    assert "not exposed" in events[0]["reason"]


def test_allocation_prologue_routes_events_to_fleet_store_not_anchor_repo(
    tmp_path: Path,
) -> None:
    """Issue #603: host-wide allocation events land in the fleet-level events.db.

    Pre-fix, ``_run_fleet_allocation_prologue`` derived ``state_path`` from the
    same registry entry that supplied the gh anchor — the first entry with a
    valid ``repo_root``. The anchor and the event-store path are independent
    concerns: the anchor only needs auth and a valid directory (the pass
    addresses every repo by explicit slug), whereas ``state_path`` decides
    which repo's ``events.db`` records the host-wide allocation event. So the
    audit trail for a host-wide action landed in whichever repo happened to
    sort first in the registry, and moved if the registry order changed.

    Post-fix, the event-store path is ``fleet_dir() / "state.json"``, so
    ``log_event`` writes to ``fleet_dir() / "events.db"`` — the same
    fleet-level store ``supervisor_lifecycle`` already writes host-wide events
    to. The per-repo databases are disjoint from it by event scope.

    This test registers two repos with distinct ``state_dir`` paths, runs the
    prologue, and verifies:
    1. The allocation event is in the fleet-level events.db.
    2. Neither per-repo events.db contains the allocation event.
    3. The gh anchor is still derived from the first valid repo root.
    """
    fleet_dir = tmp_path / "fleet"
    repo_a = _make_repo(tmp_path, "alpha", api_worker=None)
    repo_b = _make_repo(tmp_path, "beta", api_worker=None)
    state_dir_a = repo_a / ".var" / "charlie-work"
    state_dir_b = repo_b / ".var" / "charlie-work"
    _make_fleet_json(
        tmp_path,
        fleet_dir,
        {
            "owner/alpha": {
                "repo_root": str(repo_a),
                "state_dir": str(state_dir_a),
            },
            "owner/beta": {
                "repo_root": str(repo_b),
                "state_dir": str(state_dir_b),
            },
        },
    )

    # Use a skipped result so the prologue's own log_event call writes a
    # durable ``runner_allocation_skipped`` row — ``run_allocation_pass`` is
    # mocked, so the ``runner_allocation`` event it would normally write
    # never reaches the DB. The skipped path exercises the same state_path
    # routing the fix changes.
    result = AllocationPassResult(ok=True, skipped=True, notes=("no configured runners",))
    with (
        patch("charlie_work.fleet_dispatch.run_allocation_pass", return_value=result) as pass_mock,
        patch("charlie_work.fleet_dispatch.GitHub") as gh_mock,
    ):
        _run_fleet_allocation_prologue(
            str(fleet_dir), _allocation_config(enabled=True), dry_run=False
        )

    # The gh anchor is still the first valid repo root — that concern is
    # unchanged. Only the event-store path was decoupled.
    assert gh_mock.call_args.kwargs["repo_root"] == repo_a

    # The state_path passed to run_allocation_pass is the fleet-level path,
    # not repo_a's per-repo state.json.
    assert pass_mock.call_args.kwargs["state_path"] == fleet_dir / "state.json"

    # The allocation skip event is in the fleet-level events.db, not in
    # either per-repo database. This is the core assertion of #603: the
    # audit trail no longer lands in whichever repo sorted first.
    fleet_state_path = fleet_dir / "state.json"
    fleet_rows = query_events(fleet_state_path, kind="runner_allocation_skipped")
    assert len(fleet_rows) == 1
    assert fleet_rows[0]["payload"]["reason"] == "no configured runners"

    for per_repo_state in (
        layout.state_file_path(state_dir_a),
        layout.state_file_path(state_dir_b),
    ):
        per_repo_rows = query_events(per_repo_state, kind="runner_allocation_skipped")
        assert per_repo_rows == [], (
            f"host-wide allocation event leaked into per-repo events.db at {per_repo_state}"
        )


def test_allocation_prologue_skip_no_repo_root_is_seen_by_doctor(
    tmp_path: Path,
) -> None:
    """A prologue skip with allocation enabled must reach the doctor probe."""
    fleet_dir = tmp_path / "fleet"
    _make_fleet_json(
        tmp_path,
        fleet_dir,
        {"owner/gone": {"repo_root": str(tmp_path / "vanished"), "state_dir": ""}},
    )

    with patch("charlie_work.fleet_dispatch.run_allocation_pass"):
        _run_fleet_allocation_prologue(
            str(fleet_dir), _allocation_config(enabled=True), dry_run=False
        )

    collected: list[tuple[str, bool, str, str]] = []

    def add(name: str, ok: bool, detail: str, *, severity: str = "error") -> None:
        collected.append((name, ok, detail, severity))

    _check_runner_allocation(
        add,
        _allocation_config(enabled=True),
        fleet_dir_override=str(fleet_dir),
    )

    assert len(collected) == 1
    name, ok, detail, severity = collected[0]
    assert name == "runner allocation"
    assert ok is False
    assert "declined to act" in detail
    assert "no usable repo root" in detail
    # A fresh unattended skip names its own cause; it must not be misread as
    # the "never reached allocation" shape of issue #590.
    assert "#590" not in detail
    assert severity == "warning"


def test_allocation_prologue_skip_no_repo_root_is_usable_in_dry_run(
    tmp_path: Path,
) -> None:
    """A dry-run preview must not bump the allocation state file."""
    fleet_dir = tmp_path / "fleet"
    _make_fleet_json(
        tmp_path,
        fleet_dir,
        {"owner/gone": {"repo_root": str(tmp_path / "vanished"), "state_dir": ""}},
    )

    with patch("charlie_work.fleet_dispatch.run_allocation_pass") as pass_mock:
        events = _run_fleet_allocation_prologue(
            str(fleet_dir), _allocation_config(enabled=True), dry_run=True
        )

    assert [event["type"] for event in events] == ["runner_allocation_skipped"]
    pass_mock.assert_not_called()
    assert not (fleet_dir / ALLOCATION_STATE_FILENAME).exists()


def test_allocation_prologue_skips_when_no_repo_root_is_usable(tmp_path: Path) -> None:
    """Without a real directory to anchor the gh client, skip rather than guess."""
    fleet_dir = tmp_path / "fleet"
    _make_fleet_json(
        tmp_path,
        fleet_dir,
        {"owner/gone": {"repo_root": str(tmp_path / "vanished"), "state_dir": ""}},
    )

    with patch("charlie_work.fleet_dispatch.run_allocation_pass") as pass_mock:
        events = _run_fleet_allocation_prologue(
            str(fleet_dir), _allocation_config(enabled=True), dry_run=False
        )

    # Visible, not silent: an unusable registry is nobody's deliberate choice,
    # so it reaches the digest rather than only the log (issue #590).
    assert [event["type"] for event in events] == ["runner_allocation_skipped"]
    assert "registry" in events[0]["reason"]
    pass_mock.assert_not_called()

    # The skip also leaves state-file evidence so the doctor probe does not
    # attribute a fresh unattended decline to "never ran" (issue #606).
    state_file = fleet_dir / ALLOCATION_STATE_FILENAME
    assert state_file.exists()
    stamp = load_allocation_stamp(fleet_dir)
    assert stamp.source == "prologue"
    assert stamp.full_pass_interval_seconds == 300
    assert stamp.skip_reason is not None
    assert "no usable repo root" in stamp.skip_reason


def test_allocation_prologue_stays_quiet_when_nothing_moves(tmp_path: Path) -> None:
    """A balanced host must not add a line to every 5-minute digest."""
    fleet_dir = tmp_path / "fleet"
    repo = _make_repo(tmp_path, "anchor", api_worker=None)
    _make_fleet_json(
        tmp_path,
        fleet_dir,
        {"owner/anchor": {"repo_root": str(repo), "state_dir": str(repo / ".var")}},
    )
    balanced = AllocationPassResult(
        ok=True,
        plan=AllocationPlan(budget=8, budget_reason="configured", targets=(), changes=()),
    )

    with (
        patch("charlie_work.fleet_dispatch.run_allocation_pass", return_value=balanced),
        patch("charlie_work.fleet_dispatch.GitHub"),
    ):
        events = _run_fleet_allocation_prologue(
            str(fleet_dir), _allocation_config(enabled=True), dry_run=False
        )

    assert events == []


def test_allocation_prologue_surfaces_errors_and_failed_slots(tmp_path: Path) -> None:
    """Config typos and refused parks both have to reach the operator."""
    fleet_dir = tmp_path / "fleet"
    repo = _make_repo(tmp_path, "anchor", api_worker=None)
    _make_fleet_json(
        tmp_path,
        fleet_dir,
        {"owner/anchor": {"repo_root": str(repo), "state_dir": str(repo / ".var")}},
    )

    failed = AllocationPassResult(ok=False, error="managed_root does not exist: C:/nope")
    with (
        patch("charlie_work.fleet_dispatch.run_allocation_pass", return_value=failed),
        patch("charlie_work.fleet_dispatch.GitHub"),
    ):
        events = _run_fleet_allocation_prologue(
            str(fleet_dir), _allocation_config(enabled=True), dry_run=False
        )
    assert [event["type"] for event in events] == ["runner_allocation_error"]
    assert "does not exist" in events[0]["error"]

    change = SlotChange(
        repo="owner/anchor",
        runner_name="cw-2",
        path=tmp_path / "cw-2",
        action=SlotAction.PARK,
        reason="idle 3 passes",
    )
    with (
        patch(
            "charlie_work.fleet_dispatch.run_allocation_pass",
            return_value=AllocationPassResult(
                ok=True,
                plan=AllocationPlan(
                    budget=8, budget_reason="configured", targets=(), changes=(change,)
                ),
                results=(SlotChangeResult(change=change, ok=False, message="job in flight"),),
            ),
        ),
        patch("charlie_work.fleet_dispatch.GitHub"),
    ):
        events = _run_fleet_allocation_prologue(
            str(fleet_dir), _allocation_config(enabled=True), dry_run=False
        )

    assert [event["type"] for event in events] == ["runner_allocation_slot_error"]
    assert events[0]["runner"] == "cw-2"
    assert events[0]["action"] == "park"


def test_allocation_prologue_warns_when_the_config_lacks_the_section(
    tmp_path: Path, caplog: Any
) -> None:
    """A config object without the section means code/config disagree.

    That is a different failure from "the operator left it off" — it happens
    when a load failure already fell back to defaults, or when the process is
    holding a config built by other code — and it must not look like a
    deliberate opt-out.
    """
    import logging

    class _NoSection:
        pass

    with caplog.at_level(logging.WARNING, logger="charlie_work.fleet_dispatch"):
        events = _run_fleet_allocation_prologue(str(tmp_path / "fleet"), _NoSection(), False)

    assert [event["type"] for event in events] == ["runner_allocation_skipped"]
    assert "NoSection" in events[0]["reason"]
    assert any("no runner_allocation" in record.message for record in caplog.records)


@patch("charlie_work.fleet_dispatch.provision_runner")
@patch("charlie_work.fleet_dispatch.decide_autoscale")
@patch("charlie_work.fleet_dispatch.is_pool_idle_for_minutes")
@patch("charlie_work.fleet_dispatch.is_in_cooldown")
@patch("charlie_work.fleet_dispatch.observe_runner_pool")
@patch("charlie_work.fleet_dispatch.count_fleet_runners")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch._load_registry")
def test_autoscale_prologue_up_forwards_affinity_knobs(
    mock_load_registry: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_gh_class: MagicMock,
    mock_count_fleet_runners: MagicMock,
    mock_observe_runner_pool: MagicMock,
    mock_is_in_cooldown: MagicMock,
    mock_is_pool_idle: MagicMock,
    mock_decide_autoscale: MagicMock,
    mock_provision_runner: MagicMock,
    tmp_path: Path,
) -> None:
    """The fleet-wide autoscale-up call site forwards runner_allocation's knobs.

    Companion to ci_runners #92: provision_runner grew keyword-only
    reserved_threads/threads_per_slot, but this call site (distinct from
    cli.py's ``runners autoscale``) was independently inert until it forwarded
    them too. Pins that the values come from the representative repo's
    config.runner_allocation section -- never hardcoded, never left at the
    off default -- and reach provision_runner unchanged.
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
    config = OrchestratorConfig(
        runner_scaling=RunnerScalingConfig(enabled=True, managed_root=str(tmp_path)),
        runner_allocation=RunnerAllocationConfig(reserved_threads=4, threads_per_slot=6),
    )
    mock_load_layered_config.return_value = config
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths
    mock_count_fleet_runners.return_value = (1, 0, [])
    mock_is_in_cooldown.return_value = False
    mock_is_pool_idle.return_value = False
    mock_decide_autoscale.return_value = ScaleDecision(action=ScaleAction.UP, count=1, reason="t")
    mock_provision_runner.return_value = MagicMock(ok=True, runner_name="jc-1")

    global_config = MagicMock()
    global_config.runners.fleet_autoscale_prologue = True
    global_config.runner_scaling.enabled = True

    _run_fleet_autoscale_prologue(str(tmp_path / "fleet"), global_config, False)

    mock_provision_runner.assert_called_once()
    _, kwargs = mock_provision_runner.call_args
    assert kwargs["reserved_threads"] == 4
    assert kwargs["threads_per_slot"] == 6


def test_disabled_allocation_prologue_is_visible_at_info(tmp_path, caplog) -> None:
    """The disabled branch must log at INFO, not DEBUG.

    Regression guard for issue #590: the daemon runs at INFO, so a DEBUG line here
    is never written at all, which made a host where allocation never ran
    indistinguishable from a converged one. The message must also name the fleet
    directory, since that identifies which config.yaml governs the decision.
    """
    import logging
    from dataclasses import replace

    from charlie_work.config import OrchestratorConfig
    from charlie_work.fleet_dispatch import _run_fleet_allocation_prologue

    base = OrchestratorConfig()
    config = replace(base, runner_allocation=replace(base.runner_allocation, enabled=False))

    with caplog.at_level(logging.INFO, logger="charlie_work.fleet_dispatch"):
        events = _run_fleet_allocation_prologue(str(tmp_path), config, True)

    assert events == []
    assert "runner_allocation.enabled is false" in caplog.text
    assert str(tmp_path) in caplog.text
