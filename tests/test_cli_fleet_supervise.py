"""``charlie fleet supervise`` / ``supervise-loop`` parser and exit-code mapping.

Split out of ``tests/test_cli.py`` (issue #1561, Track 1) -- bodies are
verbatim relocations; shared fakes live in ``tests/_cli_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from charlie_work import cli
from charlie_work.workflow import CommandResult


def test_cli_fleet_supervise_parser() -> None:
    """``fleet supervise`` accepts the expected limit/repos/poll/max-runtime/merge flags."""
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "fleet",
            "supervise",
            "--limit",
            "2",
            "--repos",
            "a/b,c/d",
            "--poll-interval",
            "10",
            "--max-runtime",
            "60",
            "--merge",
        ]
    )
    assert args.command == "fleet"
    assert args.fleet_command == "supervise"
    assert args.limit == 2
    assert args.repos == "a/b,c/d"
    assert args.poll_interval == 10
    assert args.max_runtime == 60
    assert args.merge is True


def test_cli_fleet_supervise_command_runs_and_returns_ok(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``charlie fleet supervise`` is wired through ``main`` and returns the loop result."""
    captured: dict[str, Any] = {}

    def _fake_run_fleet_supervise(**kwargs: Any) -> CommandResult:
        captured["kwargs"] = kwargs
        return CommandResult(True, "fleet supervisor complete", {})

    monkeypatch.setattr(cli, "run_fleet_supervise", _fake_run_fleet_supervise)

    rc = cli.main(["fleet", "supervise", "--limit", "3", "--max-runtime", "120"])

    assert rc == 0
    assert captured["kwargs"]["limit"] == 3
    assert captured["kwargs"]["max_runtime_override"] == 120


def test_cli_maps_restart_requested_to_a_distinct_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#862 AC1: a restart-requesting exit is distinguishable from a clean one.

    Both are ``ok=True``, which is exactly why the fleet used to sit
    unsupervised for a full watchdog interval after every self-deploy without
    anything in Task Scheduler looking wrong.
    """

    def _fake_run_fleet_supervise(**_kwargs: Any) -> CommandResult:
        return CommandResult(
            True,
            "fleet supervisor complete",
            {"exit_reason": "self_deploy", "restart_requested": True},
        )

    monkeypatch.setattr(cli, "run_fleet_supervise", _fake_run_fleet_supervise)

    assert cli.main(["fleet", "supervise"]) == cli.EXIT_RESTART_REQUESTED


def test_cli_maps_a_deliberate_stop_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC4's control: same command, same ok=True, opposite exit code.

    Paired with the test above so the distinct code is attributable to
    ``restart_requested`` rather than to the command being run at all.
    """

    def _fake_run_fleet_supervise(**_kwargs: Any) -> CommandResult:
        return CommandResult(
            True,
            "fleet supervisor complete",
            {"exit_reason": "max_runtime", "restart_requested": False},
        )

    monkeypatch.setattr(cli, "run_fleet_supervise", _fake_run_fleet_supervise)

    assert cli.main(["fleet", "supervise"]) == 0


def test_cli_maps_restart_requested_even_on_a_failed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "Replace me" is orthogonal to "I succeeded".

    A supervisor that self-deployed and then hit an error still has new code on
    disk and still needs replacing -- relaunching is the recovery, not a reward
    for a clean run. This was gated on ``result.ok``, which made the preserved
    restart reason on an aborted run inert: exit 1, no relaunch, fleet
    unsupervised for the interval. That is #862 reached through the error path
    instead of the happy path.
    """

    def _fake_run_fleet_supervise(**_kwargs: Any) -> CommandResult:
        return CommandResult(
            False,
            "fleet supervisor aborted on pass 1: state file locked",
            {"exit_reason": "self_deploy", "restart_requested": True},
        )

    monkeypatch.setattr(cli, "run_fleet_supervise", _fake_run_fleet_supervise)

    assert cli.main(["fleet", "supervise"]) == cli.EXIT_RESTART_REQUESTED


def test_cli_maps_a_plain_failure_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """The control for the test above: ok=False without a restart reason stays 1.

    Ungating the restart code from ``ok`` must not turn every failure into a
    relaunch request -- only the ones that actually asked to be replaced.
    """

    def _fake_run_fleet_supervise(**_kwargs: Any) -> CommandResult:
        return CommandResult(
            False,
            "fleet supervisor aborted on pass 1: boom",
            {"exit_reason": "aborted", "restart_requested": False},
        )

    monkeypatch.setattr(cli, "run_fleet_supervise", _fake_run_fleet_supervise)

    assert cli.main(["fleet", "supervise"]) == 1


def test_cli_fleet_supervise_loop_forwards_args_after_the_separator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper passes supervise's flags through verbatim.

    Guards the passthrough contract the launcher depends on: `supervise-loop`
    deliberately does not re-declare supervise's flags, so a regression that
    swallowed them would start the supervisor with default settings instead of
    the launcher's `--max-runtime 0`.
    """
    captured: dict[str, Any] = {}

    def _fake_run_fleet_supervise_loop(**kwargs: Any) -> CommandResult:
        captured["kwargs"] = kwargs
        return CommandResult(True, "supervise-loop done", {})

    monkeypatch.setattr(cli, "run_fleet_supervise_loop", _fake_run_fleet_supervise_loop)

    rc = cli.main(["fleet", "supervise-loop", "--max-relaunches", "2", "--", "--max-runtime", "0"])

    assert rc == 0
    assert captured["kwargs"]["max_relaunches"] == 2
    assert captured["kwargs"]["supervise_args"] == ("--max-runtime", "0")
