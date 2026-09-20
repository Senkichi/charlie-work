"""``charlie runners provision`` manual scale-up trigger.

Split out of ``tests/test_cli.py`` (issue #1561, Track 1) -- bodies are
verbatim relocations; shared fakes live in ``tests/_cli_fixtures.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from charlie_work import cli
from charlie_work.config import (
    OrchestratorConfig,
    RunnerAllocationConfig,
    RunnerScalingConfig,
)
from charlie_work.workflow import CommandResult
from ci_fleet.charlie_work_adapter import ScaleAction
from ci_fleet.runners import ScaleDecision


# ---------------------------------------------------------------------------
# Issue #826: `charlie runners provision` — manual scale-up trigger
# ---------------------------------------------------------------------------


def _provision_config(
    *,
    scaling_enabled: bool = True,
    managed_root: str = "",
    max_runners: int = 10,
) -> OrchestratorConfig:
    """Build an OrchestratorConfig with runner_scaling knobs set for provision tests."""
    return OrchestratorConfig(
        runner_scaling=RunnerScalingConfig(
            enabled=scaling_enabled,
            managed_root=managed_root,
            max_runners=max_runners,
        ),
        runner_allocation=RunnerAllocationConfig(),
    )


def _provision_args(*, dry_run: bool = False, fleet_wide: bool = False) -> argparse.Namespace:
    """Parse ``runners provision`` args with the given flags."""
    cli_args = ["runners", "provision"]
    if dry_run:
        cli_args.append("--dry-run")
    if fleet_wide:
        cli_args.append("--fleet-wide")
    return cli.build_parser().parse_args(cli_args)


def test_run_runners_provision_refuses_when_scaling_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Feature disabled → hard refusal (issue #826 acceptance: inert under disabled).

    The operator ruling says ``enabled=false`` remains a hard refusal. The
    command must not observe the pool, run the decision, or call
    ``provision_runner`` — it must short-circuit immediately, exactly like
    ``runners status`` and ``runners autoscale`` do.
    """
    monkeypatch.setattr(cli, "find_repo_root", lambda repo, explicit=False, **kw: tmp_path)
    monkeypatch.setattr(
        cli, "load_layered_config", lambda *a, **k: _provision_config(scaling_enabled=False)
    )

    result = cli.run_runners_provision(_provision_args())

    assert result.ok is False
    assert "not enabled" in result.message


def test_run_runners_provision_inert_when_demand_within_capacity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Demand <= registered capacity → no provisioning (issue #826 acceptance: inert).

    ``decide_autoscale`` returns NONE when the pool is balanced (idle runners
    available or no queue). The provision command must report the decision
    and not call ``provision_runner``.
    """
    monkeypatch.setattr(cli, "find_repo_root", lambda repo, explicit=False, **kw: tmp_path)
    monkeypatch.setattr(
        cli, "load_layered_config", lambda *a, **k: _provision_config(managed_root=str(tmp_path))
    )
    monkeypatch.setattr(cli, "observe_runner_pool", lambda *a, **k: MagicMock())
    monkeypatch.setattr(cli, "is_in_cooldown", lambda *a, **k: False)
    monkeypatch.setattr(
        cli,
        "decide_autoscale",
        lambda *a, **k: ScaleDecision(action=ScaleAction.NONE, count=0, reason="Pool is balanced"),
    )

    provision_mock = MagicMock()
    monkeypatch.setattr("ci_fleet.charlie_work_adapter.provision_runner", provision_mock)

    result = cli.run_runners_provision(_provision_args())

    assert result.ok is True
    assert "no action" in result.message
    provision_mock.assert_not_called()


def test_run_runners_provision_inert_at_max_runners(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """max_runners guardrail → no provisioning (issue #826 acceptance: ceiling exercised).

    ``decide_autoscale`` returns NONE with a max_runners reason when the
    pool is at the cap. The provision command must respect that ceiling and
    not call ``provision_runner``. This test pins the guardrail so a future
    change cannot silently remove it.
    """
    monkeypatch.setattr(cli, "find_repo_root", lambda repo, explicit=False, **kw: tmp_path)
    monkeypatch.setattr(
        cli,
        "load_layered_config",
        lambda *a, **k: _provision_config(managed_root=str(tmp_path), max_runners=2),
    )
    monkeypatch.setattr(cli, "observe_runner_pool", lambda *a, **k: MagicMock())
    monkeypatch.setattr(cli, "is_in_cooldown", lambda *a, **k: False)
    monkeypatch.setattr(
        cli,
        "decide_autoscale",
        lambda *a, **k: ScaleDecision(
            action=ScaleAction.NONE, count=0, reason="At max_runners limit (2)"
        ),
    )

    provision_mock = MagicMock()
    monkeypatch.setattr("ci_fleet.charlie_work_adapter.provision_runner", provision_mock)

    result = cli.run_runners_provision(_provision_args())

    assert result.ok is True
    assert "no action" in result.message
    assert "max_runners" in result.data["decision"]["reason"]
    provision_mock.assert_not_called()


def test_run_runners_provision_invokes_provision_runner_on_scale_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Scale-up decision → provision_runner is actually invoked (issue #826).

    When ``decide_autoscale`` returns UP (queued_jobs > 0, idle_runners == 0,
    below max_runners, sufficient RAM, not in cooldown), the provision
    command must call ``provision_runner`` and record a scale event. This is
    the end-to-end actuator test — not just that the decision is UP, but
    that the provisioning engine is reached.
    """
    monkeypatch.setattr(cli, "find_repo_root", lambda repo, explicit=False, **kw: tmp_path)
    monkeypatch.setattr(
        cli, "load_layered_config", lambda *a, **k: _provision_config(managed_root=str(tmp_path))
    )
    monkeypatch.setattr(cli, "observe_runner_pool", lambda *a, **k: MagicMock(busy_runners=2))
    monkeypatch.setattr(cli, "is_in_cooldown", lambda *a, **k: False)
    monkeypatch.setattr(
        cli,
        "decide_autoscale",
        lambda *a, **k: ScaleDecision(
            action=ScaleAction.UP, count=1, reason="Queue has 5 waiting job(s)"
        ),
    )

    provision_mock = MagicMock(
        return_value=MagicMock(ok=True, runner_name="cw-selfhost-5", runner_dir=tmp_path / "cw-5")
    )
    monkeypatch.setattr("ci_fleet.charlie_work_adapter.provision_runner", provision_mock)
    record_mock = MagicMock()
    monkeypatch.setattr("ci_fleet.charlie_work_adapter.record_scale_event", record_mock)

    result = cli.run_runners_provision(_provision_args())

    assert result.ok is True
    assert "scaled up" in result.message
    provision_mock.assert_called_once()
    # Verify busy_runners is forwarded as the 3rd positional arg
    pos_args, _ = provision_mock.call_args
    assert pos_args[2] == 2
    # record_scale_event is called with ctx.paths.root (the state dir), not
    # the repo root — same convention as run_runners_autoscale.
    record_mock.assert_called_once()
    assert record_mock.call_args[0][1] == "up"


def test_run_runners_provision_forwards_affinity_knobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Provision forwards runner_allocation's affinity knobs (companion to autoscale test).

    Same as ``test_run_runners_autoscale_up_forwards_affinity_knobs`` but for
    the provision command. The knobs are sourced from
    ``config.runner_allocation``, never hardcoded.
    """
    monkeypatch.setattr(cli, "find_repo_root", lambda repo, explicit=False, **kw: tmp_path)

    config = OrchestratorConfig(
        runner_scaling=RunnerScalingConfig(enabled=True, managed_root=str(tmp_path)),
        runner_allocation=RunnerAllocationConfig(reserved_threads=4, threads_per_slot=6),
    )
    monkeypatch.setattr(cli, "load_layered_config", lambda *a, **k: config)
    monkeypatch.setattr(cli, "observe_runner_pool", lambda *a, **k: MagicMock(busy_runners=0))
    monkeypatch.setattr(cli, "is_in_cooldown", lambda *a, **k: False)
    monkeypatch.setattr(
        cli,
        "decide_autoscale",
        lambda *a, **k: ScaleDecision(action=ScaleAction.UP, count=1, reason="test"),
    )

    provision_mock = MagicMock(return_value=MagicMock(ok=True, runner_name="jc-1"))
    monkeypatch.setattr("ci_fleet.charlie_work_adapter.provision_runner", provision_mock)
    monkeypatch.setattr("ci_fleet.charlie_work_adapter.record_scale_event", MagicMock())

    result = cli.run_runners_provision(_provision_args())

    assert result.ok is True
    provision_mock.assert_called_once()
    _, kwargs = provision_mock.call_args
    assert kwargs["reserved_threads"] == 4
    assert kwargs["threads_per_slot"] == 6


def test_run_runners_provision_refuses_scale_down(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Scale-down decision → provision declines (provision is scale-up only).

    Even if ``decide_autoscale`` returns DOWN (e.g. pool idle), the provision
    command must NOT call ``scale_down_idle_runners`` or remove any runner.
    It reports the decision with ``declined: True`` and exits. This is the
    safety property that distinguishes ``provision`` from ``autoscale`` —
    provision is an "add capacity" button, never a second scale-down path.
    """
    monkeypatch.setattr(cli, "find_repo_root", lambda repo, explicit=False, **kw: tmp_path)
    monkeypatch.setattr(
        cli, "load_layered_config", lambda *a, **k: _provision_config(managed_root=str(tmp_path))
    )
    monkeypatch.setattr(cli, "observe_runner_pool", lambda *a, **k: MagicMock())
    monkeypatch.setattr(cli, "is_in_cooldown", lambda *a, **k: False)
    monkeypatch.setattr(
        cli,
        "decide_autoscale",
        lambda *a, **k: ScaleDecision(
            action=ScaleAction.DOWN, count=1, reason="Pool has been idle for 15 minutes"
        ),
    )

    provision_mock = MagicMock()
    monkeypatch.setattr("ci_fleet.charlie_work_adapter.provision_runner", provision_mock)
    scale_down_mock = MagicMock()
    monkeypatch.setattr(cli, "scale_down_idle_runners", scale_down_mock)

    result = cli.run_runners_provision(_provision_args())

    assert result.ok is True
    assert "declined" in result.message.lower()
    assert result.data["declined"] is True
    provision_mock.assert_not_called()
    scale_down_mock.assert_not_called()


def test_run_runners_provision_dry_run_does_not_execute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--dry-run returns the decision without calling provision_runner."""
    monkeypatch.setattr(cli, "find_repo_root", lambda repo, explicit=False, **kw: tmp_path)
    monkeypatch.setattr(
        cli, "load_layered_config", lambda *a, **k: _provision_config(managed_root=str(tmp_path))
    )
    monkeypatch.setattr(cli, "observe_runner_pool", lambda *a, **k: MagicMock(busy_runners=2))
    monkeypatch.setattr(cli, "is_in_cooldown", lambda *a, **k: False)
    monkeypatch.setattr(
        cli,
        "decide_autoscale",
        lambda *a, **k: ScaleDecision(
            action=ScaleAction.UP, count=1, reason="Queue has 5 waiting job(s)"
        ),
    )

    provision_mock = MagicMock()
    monkeypatch.setattr("ci_fleet.charlie_work_adapter.provision_runner", provision_mock)

    result = cli.run_runners_provision(_provision_args(dry_run=True))

    assert result.ok is True
    assert "no action" in result.message
    provision_mock.assert_not_called()


def test_main_dispatches_runners_provision(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() dispatches ``runners provision`` to run_runners_provision."""
    mock = MagicMock(return_value=CommandResult(True, "provision ok", {}))
    monkeypatch.setattr(cli, "run_runners_provision", mock)

    exit_code = cli.main(["runners", "provision"])

    mock.assert_called_once()
    assert exit_code == 0
