"""Issue #2810, L3: ``fleet_loop`` really runs the harness-auth episode step.

``tests/test_harness_auth_failure.py`` drives ``harness_auth_block.run_fleet_pass``
directly. This file proves the *consumer*: the fleet pass calls it on a full pass, so
an armed block is alerted without any operator action, and does not call it on a
dispatch-only (``work_only``) pass. Kept apart because importing the fleet fixtures
makes their autouse hermeticity patches apply to every test in the module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from _fleet_dispatch_fixtures import (
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
)
from _harness_auth_fixtures import EVIDENCE, HARNESS, arm, block, make_config
import pytest
from charlie_work import harness_auth_block, layout
from charlie_work.fleet_dispatch import fleet_loop
from charlie_work.instrumentation import query_events


@pytest.fixture
def fleet(tmp_path: Path) -> Path:
    return tmp_path / "fleet"


@pytest.fixture
def digests(monkeypatch: pytest.MonkeyPatch) -> list:
    sent: list = []
    monkeypatch.setattr(harness_auth_block, "emit_digest", lambda _cfg, d: sent.append(d))
    return sent


def _run(fleet: Path, *, work_only: bool, dry_run: bool = False) -> Any:
    with patch("charlie_work.fleet_dispatch._load_registry", return_value={"repos": {}}):
        return fleet_loop(
            fleet_dir_override=str(fleet),
            global_config=make_config(),
            repos=None,
            limit=1,
            merge=None,
            dry_run=dry_run,
            work_only=work_only,
        )


@pytest.mark.parametrize(
    ("work_only", "calls"),
    [
        pytest.param(False, 1, id="full-pass-runs-the-step"),
        pytest.param(True, 0, id="work-only-pass-skips-it"),
    ],
)
def test_fleet_loop_runs_the_episode_step_on_a_full_pass_only(
    fleet: Path, work_only: bool, calls: int
) -> None:
    spy = MagicMock(return_value=[])
    with patch.object(harness_auth_block, "run_fleet_pass", spy):
        result = _run(fleet, work_only=work_only)

    assert result.ok is True
    assert spy.call_count == calls
    if calls:
        kwargs = spy.call_args.kwargs
        assert kwargs["fleet_state_path"] == layout.state_file_path(fleet)
        assert kwargs["dry_run"] is False
        assert kwargs["fleet_dir_override"] == str(fleet)


def test_a_full_fleet_pass_alerts_an_armed_block_exactly_once(fleet: Path, digests: list) -> None:
    arm()
    state_path = layout.state_file_path(fleet)
    assert query_events(state_path, kind="harness_auth_blocked") == [], "control: nothing yet"

    _run(fleet, work_only=False)
    _run(fleet, work_only=False)

    (event,) = query_events(state_path, kind="harness_auth_blocked")
    assert event["payload"]["harness"] == HARNESS and event["payload"]["evidence"] == EVIDENCE
    assert len(digests) == 1
    assert block().alerted_at is not None


def test_a_work_only_or_dry_run_pass_leaves_the_block_unalerted(
    fleet: Path, digests: list
) -> None:
    arm()
    state_path = layout.state_file_path(fleet)

    _run(fleet, work_only=True)
    _run(fleet, work_only=False, dry_run=True)

    assert query_events(state_path, kind="harness_auth_blocked") == []
    assert digests == [] and block().alerted_at is None
    _run(fleet, work_only=False)
    assert len(query_events(state_path, kind="harness_auth_blocked")) == 1, (
        "control: the same block alerts on a full, non-dry pass"
    )
