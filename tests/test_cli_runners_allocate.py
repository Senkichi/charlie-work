"""``charlie runners allocate`` / ``autoscale-up`` / ``ensure-started`` command paths.

Split out of ``tests/test_cli.py`` (issue #1561, Track 1) -- bodies are
verbatim relocations; shared fakes live in ``tests/_cli_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from _cli_fixtures import _FakeGitHub
from charlie_work import (
    cli,
    fleet_dispatch,
)
from charlie_work.config import (
    OrchestratorConfig,
    RunnerAllocationConfig,
    RunnerScalingConfig,
)
from charlie_work.fleet_dispatch import _CiFleetDirtyCheck
from ci_fleet.charlie_work_adapter import ScaleAction
from ci_fleet.runner_allocation import AllocationPlan
from ci_fleet.runner_allocation_pass import AllocationPassResult
from ci_fleet.runners import ScaleDecision


def test_run_runners_allocate_loud_on_absent_global_layer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An absent global layer must fail loudly in `charlie runners allocate`.

    ``runner_allocation`` is a fleet-wide knob declared in the global fleet
    config layer. An unreachable global layer silently flips it to its
    dataclass default (``enabled=False``), and ``run_runners_allocate`` would
    then report "runner_allocation feature is not enabled in config" -- the
    exact #623 silent-disable failure: the operator sees an opt-out message
    when the real cause is an unready volume. The command now loads with
    ``require_global=True`` and fails loudly instead, so an unready volume is
    distinguishable from a fleet that genuinely opted out of runner allocation.

    Drives the REAL ``load_layered_config`` (not a mock) against an empty fleet
    dir so the ``require_global=True`` wiring is exercised end-to-end.
    ``find_repo_root`` is stubbed so the command does not require a real git
    work tree at cwd; the config load fails before any network/runner work.
    """
    monkeypatch.setattr(cli, "find_repo_root", lambda repo, explicit=False, **kw: tmp_path)
    # Deliberately NOT mocking cli.load_layered_config.

    args = cli.build_parser().parse_args(["--fleet-dir", str(tmp_path), "runners", "allocate"])
    result = cli.run_runners_allocate(args)

    # The command fails loudly with the config-load cause, NOT the silent
    # "not enabled" message that a defaulted runner_allocation would produce.
    assert result.ok is False, (
        "an absent required global layer must fail the command, not silently "
        "default runner_allocation to disabled"
    )
    assert "config load failed" in result.message, (
        f"the failure must name the config load, not 'not enabled': {result.message!r}"
    )
    assert "cannot decide runner_allocation" in result.message, (
        f"the failure must name runner_allocation: {result.message!r}"
    )
    assert str(tmp_path / "config.yaml") in result.message, (
        "the expected global config path must appear in the failure message"
    )
    assert "absent" in result.message, "an absent layer must read as absent in the failure message"
    assert "not enabled" not in result.message, (
        "the silent-disable 'not enabled' message must NOT appear when the real "
        "cause is an unreachable global layer"
    )


def test_run_runners_allocate_forces_dry_run_when_ci_fleet_is_dirty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Issue #927 rework (PR #1048): `charlie runners allocate` -- the operator
    path -- must be guarded against a dirty editable ci_fleet worktree exactly
    like the unattended supervisor prologue.

    The original guard only covered fleet_dispatch's prologue call site and
    left this CLI command -- which CLAUDE.md names as "the only thing allowed
    to decide which listeners run" -- completely unguarded. Both call sites
    now go through ``run_allocation_pass_with_ci_fleet_guard``, so patching
    the guard's underlying primitives (``_ci_fleet_worktree_dirty`` and
    ``run_allocation_pass``, both resolved from ``charlie_work.fleet_dispatch``)
    proves this CLI path is covered by that single enforcement point rather
    than a second, independently-written copy of the check.
    """
    monkeypatch.setattr(cli, "find_repo_root", lambda repo, explicit=False, **kw: tmp_path)
    config = OrchestratorConfig(
        runner_scaling=RunnerScalingConfig(managed_root=str(tmp_path)),
        runner_allocation=RunnerAllocationConfig(enabled=True),
    )
    monkeypatch.setattr(cli, "load_layered_config", lambda *a, **k: config)
    monkeypatch.setattr(cli, "GitHub", lambda *a, **k: _FakeGitHub())

    dirty_check = _CiFleetDirtyCheck(
        is_dirty=True,
        repo_root=tmp_path / "ci_fleet",
        dirty_paths=(" M src/runner_allocation.py",),
    )
    monkeypatch.setattr(
        fleet_dispatch,
        "_ci_fleet_worktree_dirty",
        lambda _module_file=None: dirty_check,
    )
    plan = AllocationPlan(budget=4, budget_reason="configured", targets=(), changes=())
    pass_result = AllocationPassResult(ok=True, plan=plan, notes=())
    pass_mock = MagicMock(return_value=pass_result)
    monkeypatch.setattr(fleet_dispatch, "run_allocation_pass", pass_mock)

    args = cli.build_parser().parse_args(["runners", "allocate"])
    outcome = cli.run_runners_allocate(args)

    assert pass_mock.call_args.kwargs["dry_run"] is True, (
        "a dirty ci_fleet worktree must force dry_run on the CLI allocate path, "
        "not only the unattended supervisor prologue"
    )
    assert outcome.data["dry_run"] is True
    assert outcome.data["ci_fleet_worktree_dirty"]["dirty_paths"] == [
        " M src/runner_allocation.py"
    ], "the forced-dry-run reason must surface in the CommandResult data"
    assert "forced dry-run" in outcome.message, (
        f"the CLI message must say why it refused to actuate: {outcome.message!r}"
    )


# --------------------------------------------------------------------------
# runners ensure-started: single-controller guard (issue #598)
# --------------------------------------------------------------------------


def test_run_runners_autoscale_up_forwards_affinity_knobs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The autoscale-up call site forwards runner_allocation's affinity knobs.

    Companion to ci_runners #92: provision_runner grew keyword-only
    reserved_threads/threads_per_slot, but the call site was inert until it
    forwarded them. This pins that the values are read from
    config.runner_allocation (never hardcoded, never defaulted away) and
    passed through unchanged.

    provision_runner is mocked at the charlie_work_adapter import boundary
    used by cli.py's local ``from ci_fleet.charlie_work_adapter import
    provision_runner`` -- this passes against whichever ci_fleet is
    currently installed, independent of whether #92 has merged yet.
    """
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    monkeypatch.setattr(cli, "find_repo_root", lambda repo, explicit=False, **kw: tmp_path)

    config = OrchestratorConfig(
        runner_scaling=RunnerScalingConfig(enabled=True, managed_root=str(tmp_path)),
        runner_allocation=RunnerAllocationConfig(reserved_threads=4, threads_per_slot=6),
    )
    monkeypatch.setattr(cli, "load_layered_config", lambda *a, **k: config)
    monkeypatch.setattr(cli, "observe_runner_pool", lambda *a, **k: MagicMock())
    monkeypatch.setattr(cli, "is_in_cooldown", lambda *a, **k: False)
    monkeypatch.setattr(cli, "is_pool_idle_for_minutes", lambda *a, **k: False)
    monkeypatch.setattr(
        cli,
        "decide_autoscale",
        lambda *a, **k: ScaleDecision(action=ScaleAction.UP, count=1, reason="test"),
    )

    provision_mock = MagicMock(return_value=MagicMock(ok=True, runner_name="jc-1"))
    monkeypatch.setattr(
        "ci_fleet.charlie_work_adapter.provision_runner",
        provision_mock,
    )

    args = cli.build_parser().parse_args(["runners", "autoscale"])
    result = cli.run_runners_autoscale(args)

    assert result.ok is True
    provision_mock.assert_called_once()
    _, kwargs = provision_mock.call_args
    assert kwargs["reserved_threads"] == 4
    assert kwargs["threads_per_slot"] == 6


def _ensure_started_config(
    *,
    scaling_enabled: bool = True,
    allocation_enabled: bool = False,
    managed_root: str = "",
) -> OrchestratorConfig:
    """Build an OrchestratorConfig with the runner_scaling/allocation knobs set.

    ``managed_root`` defaults to empty so the guard fires before any
    path-existence check; tests that need to reach ``ensure_runners_started``
    pass a real tmp_path.
    """
    return OrchestratorConfig(
        runner_scaling=RunnerScalingConfig(
            enabled=scaling_enabled,
            managed_root=managed_root,
        ),
        runner_allocation=RunnerAllocationConfig(enabled=allocation_enabled),
    )


def test_run_runners_ensure_started_refuses_when_allocation_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """ensure-started must refuse when runner_allocation is enabled (#598).

    ``ensure_runner_running`` relaunches any runner where
    ``not is_runner_launched(...)`` -- exactly the state a deliberately parked
    slot is in. Running ensure-started while allocation is enabled therefore
    restarts every parked listener and silently undoes ``runners allocate``,
    burning a full ``demand_idle_samples`` hysteresis window reconverging.
    The guard refuses and points at the single controller.
    """
    monkeypatch.setattr(cli, "find_repo_root", lambda repo, explicit=False, **kw: tmp_path)
    monkeypatch.setattr(
        cli,
        "load_layered_config",
        lambda *a, **k: _ensure_started_config(allocation_enabled=True),
    )
    ensure_mock = MagicMock(return_value=(0, []))
    monkeypatch.setattr(cli, "ensure_runners_started", ensure_mock)

    args = cli.build_parser().parse_args(["runners", "ensure-started"])
    result = cli.run_runners_ensure_started(args)

    assert result.ok is False, (
        "ensure-started must refuse when runner_allocation is enabled, not "
        "silently relaunch parked slots"
    )
    assert "runner_allocation is enabled" in result.message, (
        f"refusal must name runner_allocation: {result.message!r}"
    )
    assert "runners allocate" in result.message, (
        f"refusal must point at the single controller: {result.message!r}"
    )
    assert "--force" in result.message, (
        f"refusal must mention the --force escape hatch: {result.message!r}"
    )
    (
        ensure_mock.assert_not_called(),
        (
            "ensure_runners_started must NOT be called when the guard refuses -- "
            "calling it would relaunch parked slots, which is the exact bug"
        ),
    )


def test_run_runners_ensure_started_force_bypasses_allocation_guard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """--force is the explicit escape hatch past the allocation guard (#598).

    A deliberate manual recovery that ``runners allocate`` cannot do may need
    to relaunch every listener; --force records that intent explicitly so the
    operator owns the reconvergence cost rather than the command silently
    incurring it.
    """
    managed_root = tmp_path / "runners"
    managed_root.mkdir()
    monkeypatch.setattr(cli, "find_repo_root", lambda repo, explicit=False, **kw: tmp_path)
    monkeypatch.setattr(
        cli,
        "load_layered_config",
        lambda *a, **k: _ensure_started_config(
            allocation_enabled=True,
            managed_root=str(managed_root),
        ),
    )
    ensure_mock = MagicMock(return_value=(2, ["jc-1: launched", "jc-2: launched"]))
    monkeypatch.setattr(cli, "ensure_runners_started", ensure_mock)

    args = cli.build_parser().parse_args(["runners", "ensure-started", "--force"])
    result = cli.run_runners_ensure_started(args)

    assert result.ok is True, f"--force must bypass the guard and proceed: {result.message!r}"
    (
        ensure_mock.assert_called_once(),
        ("ensure_runners_started must be called when --force is passed"),
    )


def test_run_runners_ensure_started_proceeds_when_allocation_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No guard regression: allocation disabled means ensure-started runs.

    The single-controller guard only applies when allocation is enabled. A
    fleet that has not opted into runner_allocation must keep using
    ensure-started as the recovery path for managed runners that die on
    reboot/logoff -- the guard must not turn into a blanket refusal.
    """
    managed_root = tmp_path / "runners"
    managed_root.mkdir()
    monkeypatch.setattr(cli, "find_repo_root", lambda repo, explicit=False, **kw: tmp_path)
    monkeypatch.setattr(
        cli,
        "load_layered_config",
        lambda *a, **k: _ensure_started_config(
            allocation_enabled=False,
            managed_root=str(managed_root),
        ),
    )
    ensure_mock = MagicMock(return_value=(1, ["jc-1: launched"]))
    monkeypatch.setattr(cli, "ensure_runners_started", ensure_mock)

    args = cli.build_parser().parse_args(["runners", "ensure-started"])
    result = cli.run_runners_ensure_started(args)

    assert result.ok is True, (
        "ensure-started must proceed when runner_allocation is disabled -- "
        "the guard is not a blanket refusal"
    )
    (
        ensure_mock.assert_called_once(),
        ("ensure_runners_started must be called when allocation is disabled"),
    )
