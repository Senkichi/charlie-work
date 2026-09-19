"""CLI surface: parser construction, operator-queue command dispatch, roll-call JSON dependencies schema.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

from pathlib import Path
from _fakes_github import FakeGitHub
from charlie_work import cli
from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_roll_call_json_dependencies_schema(tmp_path: Path) -> None:
    """Test that roll-call --json includes dependencies payload with correct schema per issue #152."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create fake GitHub with dependency markers
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 100,
            "title": "issue-with-deps",
            "url": "https://github.com/test/repo/issues/100",
            "body": "Blocked by #200, #300",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-01T00:00:00Z",
            "updatedAt": "2026-07-01T00:00:00Z",
            "state": "OPEN",
        },
        {
            "number": 200,
            "title": "blocker-1",
            "url": "https://github.com/test/repo/issues/200",
            "body": "Blocker issue",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-02T00:00:00Z",
            "updatedAt": "2026-07-02T00:00:00Z",
            "state": "OPEN",
        },
        {
            "number": 300,
            "title": "blocker-2",
            "url": "https://github.com/test/repo/issues/300",
            "body": "Another blocker",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-03T00:00:00Z",
            "updatedAt": "2026-07-03T00:00:00Z",
            "state": "OPEN",
        },
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Run status with JSON output
    result = app.status()

    assert result.ok is True
    roll_call_data = result.data

    # Verify dependencies payload exists and has correct schema
    assert "issues" in roll_call_data
    issues_by_number = {issue["number"]: issue for issue in roll_call_data["issues"]}

    # Check issue 100 has dependencies
    issue_100 = issues_by_number[100]
    assert "dependencies" in issue_100
    deps = issue_100["dependencies"]
    assert "declared" in deps
    assert "open" in deps
    assert isinstance(deps["declared"], list)
    assert isinstance(deps["open"], list)
    # Issue 100 declares blockers 200 and 300
    assert set(deps["declared"]) == {200, 300}
    # Both blockers are open, so open blockers should match declared
    assert set(deps["open"]) == {200, 300}

    # Check blocker issues have empty dependencies
    issue_200 = issues_by_number[200]
    assert "dependencies" in issue_200
    assert issue_200["dependencies"]["declared"] == []
    assert issue_200["dependencies"]["open"] == []


def test_build_parser_fleet_subcommand() -> None:
    """Test that build_parser registers the fleet subcommand correctly."""
    parser = cli.build_parser()

    # Test fleet status parsing
    args = parser.parse_args(["fleet", "status"])
    assert args.command == "fleet"
    assert args.fleet_command == "status"

    # Test that existing subcommands still work
    args_roll_call = parser.parse_args(["roll-call"])
    assert args_roll_call.command == "roll-call"

    parser.parse_args(["doctor"])

    # Test fleet review-queue parsing
    args_review_queue = parser.parse_args(["fleet", "review-queue"])
    assert args_review_queue.command == "fleet"
    assert args_review_queue.fleet_command == "review-queue"

    # Test single-repo review-queue parsing
    args_single = parser.parse_args(["review-queue"])
    assert args_single.command == "review-queue"

    # Test fleet operator-queue parsing (issue #1314 item 1)
    args_fleet_operator_queue = parser.parse_args(["fleet", "operator-queue"])
    assert args_fleet_operator_queue.command == "fleet"
    assert args_fleet_operator_queue.fleet_command == "operator-queue"

    # Test single-repo operator-queue parsing (issue #1314 item 1)
    args_operator_queue = parser.parse_args(["operator-queue"])
    assert args_operator_queue.command == "operator-queue"


def test_run_command_dispatches_operator_queue(tmp_path: Path) -> None:
    """Issue #1314 item 1: ``run_command`` routes ``operator-queue`` to
    ``app.operator_queue()`` and returns its result verbatim.

    Mirrors the established convention that the CLI dispatch branch for a
    queue command is exercised end-to-end through ``run_command`` rather than
    only through the ``OrchestratorApp`` method in isolation — a broken
    dispatch branch (wrong ``args.command`` string, missing branch) would
    otherwise go undetected by the method-level tests.
    """
    args = cli.build_parser().parse_args(["operator-queue"])
    assert args.command == "operator-queue"

    dispatched: list[str] = []
    expected = cli.CommandResult(
        True, "operator queue: 0 issue(s) parked", {"queue": [], "depth": 0}
    )

    class _FakeApp:
        def operator_queue(self) -> cli.CommandResult:
            dispatched.append("operator_queue")
            return expected

    result = cli.run_command(_FakeApp(), args)  # type: ignore[arg-type]

    assert dispatched == ["operator_queue"]
    assert result is expected
