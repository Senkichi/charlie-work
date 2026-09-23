"""Fleet status and operator/review queue aggregation.

Split out of ``tests/test_charlie_work.py`` (issue #1548, Track-1 wave 2/8):
the fleet-status half of the ``test_fleet_*`` seam -- fleet status
aggregation, per-repo error isolation, JSON output shape, and operator/review
queue aggregation. Shared fakes and helpers in ``tests/_dispatch_fixtures.py``.
"""

from __future__ import annotations

import json
import sys
from datetime import (
    UTC,
    datetime,
    timedelta,
)
from pathlib import Path

import pytest

from charlie_work import (
    cli,
    github as github_module,
)
from charlie_work.config import OrchestratorConfig
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401

# Fleet status tests


def test_fleet_status_aggregates_multiple_repos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, autospec
) -> None:
    """Test that fleet status aggregates status from multiple repos."""
    # Set up fleet directory override
    fleet_override = str(tmp_path / "fleet")
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", fleet_override)

    # Create two repo directories with minimal setup
    repo1 = tmp_path / "repo1"
    repo2 = tmp_path / "repo2"
    repo1.mkdir()
    repo2.mkdir()

    # Create minimal configs
    config1 = repo1 / "orchestrator.config.yaml"
    config2 = repo2 / "orchestrator.config.yaml"
    config1.write_text(
        "labels:\n  ready: automated-ready\n  queued: agent:queued\n  in_progress: agent:in-progress\nruntime:\n  state_dir: .var/charlie-work\n"
    )
    config2.write_text(
        "labels:\n  ready: automated-ready\n  queued: agent:queued\n  in_progress: agent:in-progress\nruntime:\n  state_dir: .var/charlie-work\n"
    )

    # Create state directories
    (repo1 / ".var" / "charlie-work").mkdir(parents=True)
    (repo2 / ".var" / "charlie-work").mkdir(parents=True)

    # Create fleet.json with two repos
    fleet_json_path = Path(fleet_override) / "fleet.json"
    fleet_json_path.parent.mkdir(parents=True, exist_ok=True)
    registry_data = {
        "version": 1,
        "repos": {
            "owner/repo1": {
                "repo_root": str(repo1),
                "name_with_owner": "owner/repo1",
                "config_path": str(config1),
                "state_dir": str(repo1 / ".var" / "charlie-work"),
                "first_seen": "2026-07-06T12:00:00Z",
                "last_seen": "2026-07-06T12:00:00Z",
            },
            "owner/repo2": {
                "repo_root": str(repo2),
                "name_with_owner": "owner/repo2",
                "config_path": str(config2),
                "state_dir": str(repo2 / ".var" / "charlie-work"),
                "first_seen": "2026-07-06T12:00:00Z",
                "last_seen": "2026-07-06T12:00:00Z",
            },
        },
    }
    import json

    fleet_json_path.write_text(json.dumps(registry_data, indent=2))

    # Mock GitHub to return empty issue/PR lists
    from charlie_work.github import GitHub

    autospec(monkeypatch, GitHub, "issue_list", return_value=[])
    autospec(monkeypatch, GitHub, "pr_list", return_value=[])
    autospec(
        monkeypatch,
        github_module,
        "get_github_issue_dependencies",
        return_value=[],
    )
    # run_fleet_status creates a real GitHub instance; the field-list probe
    # needs a real ``gh`` CLI, so short-circuit it for these unit tests.
    autospec(monkeypatch, GitHub, "validate_field_lists", return_value=None)

    # Run fleet status
    args = cli.build_parser().parse_args(["fleet", "status"])
    result = cli.run_fleet_status(args)

    # Verify aggregation
    assert result.ok is True
    assert "2 repo(s)" in result.message
    assert len(result.data["repos"]) == 2
    assert "owner/repo1" in result.data["repos"]
    assert "owner/repo2" in result.data["repos"]
    assert result.data["errors"] == []


def test_fleet_status_isolates_broken_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, autospec
) -> None:
    """Test that fleet status isolates errors from broken repos."""
    # Set up fleet directory override
    fleet_override = str(tmp_path / "fleet")
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", fleet_override)

    # Create one valid repo
    repo_valid = tmp_path / "repo_valid"
    repo_valid.mkdir()
    config_valid = repo_valid / "orchestrator.config.yaml"
    config_valid.write_text(
        "labels:\n  ready: automated-ready\n  queued: agent:queued\n  in_progress: agent:in-progress\nruntime:\n  state_dir: .var/charlie-work\n"
    )
    (repo_valid / ".var" / "charlie-work").mkdir(parents=True)

    # Create fleet.json with one valid and one broken repo
    fleet_json_path = Path(fleet_override) / "fleet.json"
    fleet_json_path.parent.mkdir(parents=True, exist_ok=True)
    registry_data = {
        "version": 1,
        "repos": {
            "owner/repo_valid": {
                "repo_root": str(repo_valid),
                "name_with_owner": "owner/repo_valid",
                "config_path": str(config_valid),
                "state_dir": str(repo_valid / ".var" / "charlie-work"),
                "first_seen": "2026-07-06T12:00:00Z",
                "last_seen": "2026-07-06T12:00:00Z",
            },
            "owner/repo_broken": {
                "repo_root": str(tmp_path / "nonexistent"),
                "name_with_owner": "owner/repo_broken",
                "config_path": str(tmp_path / "nonexistent" / "orchestrator.config.yaml"),
                "state_dir": str(tmp_path / "nonexistent" / ".var" / "charlie-work"),
                "first_seen": "2026-07-06T12:00:00Z",
                "last_seen": "2026-07-06T12:00:00Z",
            },
        },
    }
    import json

    fleet_json_path.write_text(json.dumps(registry_data, indent=2))

    # Mock GitHub to return empty issue/PR lists
    from charlie_work.github import GitHub

    autospec(monkeypatch, GitHub, "issue_list", return_value=[])
    autospec(monkeypatch, GitHub, "pr_list", return_value=[])
    autospec(
        monkeypatch,
        github_module,
        "get_github_issue_dependencies",
        return_value=[],
    )
    # run_fleet_status creates a real GitHub instance; the field-list probe
    # needs a real ``gh`` CLI, so short-circuit it for these unit tests.
    autospec(monkeypatch, GitHub, "validate_field_lists", return_value=None)

    # Run fleet status
    args = cli.build_parser().parse_args(["fleet", "status"])
    result = cli.run_fleet_status(args)

    # Issue #1372: a repo whose repo_root does not exist is STALE, not a live
    # failing lane. It is reported in a separate "stale" list that does NOT
    # flip ok/exit-code, so one corpse cannot degrade fleet-wide tooling.
    assert result.ok is True  # Stale entries do not flip the exit code
    assert "1 repo(s)" in result.message
    assert "1 stale(s)" in result.message
    assert len(result.data["repos"]) == 1
    assert "owner/repo_valid" in result.data["repos"]
    # The broken repo is in stale, not errors.
    assert len(result.data["errors"]) == 0
    assert len(result.data["stale"]) == 1
    assert result.data["stale"][0]["repo_key"] == "owner/repo_broken"


def test_fleet_status_never_mutates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, autospec
) -> None:
    """Test that fleet status never mutates GitHub labels or state."""
    # Set up fleet directory override
    fleet_override = str(tmp_path / "fleet")
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", fleet_override)

    # Create a repo with a ready-labeled issue
    repo = tmp_path / "repo"
    repo.mkdir()
    config = repo / "orchestrator.config.yaml"
    config.write_text(
        "labels:\n  ready: automated-ready\n  queued: agent:queued\n  in_progress: agent:in-progress\nruntime:\n  state_dir: .var/charlie-work\n"
    )
    (repo / ".var" / "charlie-work").mkdir(parents=True)

    # Create state.json
    state_file = repo / ".var" / "charlie-work" / "state.json"
    import json

    initial_state = {
        "version": 1,
        "generated_at": "2026-07-06T12:00:00Z",
        "issues": {},
        "prs": {},
        "events": [],
    }
    state_file.write_text(json.dumps(initial_state, indent=2))

    # Create fleet.json
    fleet_json_path = Path(fleet_override) / "fleet.json"
    fleet_json_path.parent.mkdir(parents=True, exist_ok=True)
    registry_data = {
        "version": 1,
        "repos": {
            "owner/repo": {
                "repo_root": str(repo),
                "name_with_owner": "owner/repo",
                "config_path": str(config),
                "state_dir": str(repo / ".var" / "charlie-work"),
                "first_seen": "2026-07-06T12:00:00Z",
                "last_seen": "2026-07-06T12:00:00Z",
            },
        },
    }
    fleet_json_path.write_text(json.dumps(registry_data, indent=2))

    # Mock GitHub to return a ready issue and track mutating calls
    from charlie_work.github import GitHub

    mutating_calls = []

    def mock_run(self, args, json_output=False, allow_failure=False, long_call=False):
        mutating_calls.append(args)
        return ""

    def mock_issue_list(self, labels=None, state=None):
        # The unfiltered call must return a SUPERSET of the ready-filtered
        # one -- that relation is what the classifier cross-checks.
        name = labels[0] if isinstance(labels, (list, tuple)) and labels else labels
        if name is None:
            name = OrchestratorConfig().labels.ready
        return [{"number": 123, "title": "Test issue", "labels": [{"name": name}]}]

    autospec(monkeypatch, GitHub, "run", side_effect=mock_run)
    autospec(monkeypatch, GitHub, "issue_list", side_effect=mock_issue_list)
    autospec(monkeypatch, GitHub, "pr_list", return_value=[])
    autospec(
        monkeypatch,
        github_module,
        "get_github_issue_dependencies",
        return_value=[],
    )
    # run_fleet_status creates a real GitHub instance; the field-list probe
    # needs a real ``gh`` CLI, so short-circuit it for these unit tests.
    autospec(monkeypatch, GitHub, "validate_field_lists", return_value=None)

    # Run fleet status
    args = cli.build_parser().parse_args(["fleet", "status"])
    result = cli.run_fleet_status(args)

    # Verify no mutating calls were made
    assert result.ok is True
    # GitHub.run should not be called with mutating commands
    for call in mutating_calls:
        assert not any(
            mutating_cmd in call
            for mutating_cmd in ["issue edit", "label add", "label remove", "pr edit"]
        ), f"Mutating call detected: {call}"

    # Verify state.json was not modified
    final_state = json.loads(state_file.read_text())
    assert final_state["generated_at"] == initial_state["generated_at"]
    assert final_state == initial_state


def test_fleet_status_json_output_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, autospec
) -> None:
    """Test that fleet status --json produces the correct output shape."""
    from io import StringIO

    # Set up fleet directory override
    fleet_override = str(tmp_path / "fleet")
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", fleet_override)

    # Create a minimal repo
    repo = tmp_path / "repo"
    repo.mkdir()
    config = repo / "orchestrator.config.yaml"
    config.write_text(
        "labels:\n  ready: automated-ready\n  queued: agent:queued\n  in_progress: agent:in-progress\nruntime:\n  state_dir: .var/charlie-work\n"
    )
    (repo / ".var" / "charlie-work").mkdir(parents=True)

    # Create fleet.json
    fleet_json_path = Path(fleet_override) / "fleet.json"
    fleet_json_path.parent.mkdir(parents=True, exist_ok=True)
    registry_data = {
        "version": 1,
        "repos": {
            "owner/repo": {
                "repo_root": str(repo),
                "name_with_owner": "owner/repo",
                "config_path": str(config),
                "state_dir": str(repo / ".var" / "charlie-work"),
                "first_seen": "2026-07-06T12:00:00Z",
                "last_seen": "2026-07-06T12:00:00Z",
            },
        },
    }
    import json

    fleet_json_path.write_text(json.dumps(registry_data, indent=2))

    # Mock GitHub to return empty issue/PR lists
    from charlie_work.github import GitHub

    autospec(monkeypatch, GitHub, "issue_list", return_value=[])
    autospec(monkeypatch, GitHub, "pr_list", return_value=[])
    autospec(
        monkeypatch,
        github_module,
        "get_github_issue_dependencies",
        return_value=[],
    )
    # run_fleet_status creates a real GitHub instance; the field-list probe
    # needs a real ``gh`` CLI, so short-circuit it for these unit tests.
    autospec(monkeypatch, GitHub, "validate_field_lists", return_value=None)

    # Capture stdout
    fake_stdout = StringIO()
    monkeypatch.setattr(sys, "stdout", fake_stdout)

    # Run fleet status --json via main()
    try:
        cli.main(["fleet", "status", "--json"])
    except SystemExit:
        pass

    output = fake_stdout.getvalue()
    parsed = json.loads(output)

    # Verify JSON structure
    assert "ok" in parsed
    assert "message" in parsed
    assert "data" in parsed
    assert "repos" in parsed["data"]
    assert "errors" in parsed["data"]
    assert "owner/repo" in parsed["data"]["repos"]


def test_fleet_operator_queue_aggregates_and_isolates_errors(tmp_path: Path, monkeypatch) -> None:
    """Issue #1314 item 1: fleet operator-queue aggregates per repo and
    isolates errors, mirroring ``test_fleet_review_queue_aggregates_and_isolates_errors``.

    A broken repo (missing root) is isolated into ``errors`` while the good
    repo's ``operator_queue()`` result still populates ``per_repo``, and
    ``CommandResult.ok`` is False only because ``errors`` is non-empty.
    """
    fleet_override = str(tmp_path / "fleet")
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", fleet_override)

    repo_ok = tmp_path / "repo_ok"
    repo_ok.mkdir()
    config_ok = repo_ok / "orchestrator.config.yaml"
    config_ok.write_text(
        "labels:\n  ready: automated-ready\n  queued: agent:queued\n  in_progress: agent:in-progress\n  operator_queue: agent:operator-queue\nruntime:\n  state_dir: .var/charlie-work\n"
    )
    (repo_ok / ".var" / "charlie-work").mkdir(parents=True)

    # Good repo: one issue carrying the operator_queue label, plus a state.json
    # entry marking it a mechanical escalation with a terminal_since timestamp.
    now = datetime.now(UTC)
    terminal_since = (now - timedelta(days=2)).isoformat().replace("+00:00", "Z")
    (repo_ok / ".var" / "charlie-work" / "state.json").write_text(
        json.dumps(
            {
                "version": 1,
                "issues": {
                    "42": {
                        "number": 42,
                        "status": "escalated",
                        "reason_class": "mechanical",
                        "escalation_reason": "test escalation",
                        "terminal_since": terminal_since,
                    }
                },
                "prs": {},
                "events": [],
            }
        ),
        encoding="utf-8",
    )

    fleet_json_path = Path(fleet_override) / "fleet.json"
    fleet_json_path.parent.mkdir(parents=True, exist_ok=True)
    registry_data = {
        "version": 1,
        "repos": {
            "owner/repo_ok": {
                "repo_root": str(repo_ok),
                "name_with_owner": "owner/repo_ok",
                "config_path": str(config_ok),
                "state_dir": str(repo_ok / ".var" / "charlie-work"),
                "first_seen": "2026-07-06T12:00:00Z",
                "last_seen": "2026-07-06T12:00:00Z",
            },
            "owner/repo_broken": {
                "repo_root": str(tmp_path / "nonexistent"),
                "name_with_owner": "owner/repo_broken",
                "config_path": str(tmp_path / "nonexistent" / "orchestrator.config.yaml"),
                "state_dir": str(tmp_path / "nonexistent" / ".var" / "charlie-work"),
                "first_seen": "2026-07-06T12:00:00Z",
                "last_seen": "2026-07-06T12:00:00Z",
            },
        },
    }
    fleet_json_path.write_text(json.dumps(registry_data, indent=2))

    from charlie_work.github import GitHub

    def mock_issue_list(self, labels=None, state=None):
        return [
            {
                "number": 42,
                "title": "issue 42",
                "url": "https://example.test/issues/42",
                "body": "",
                "labels": [{"name": "agent:operator-queue"}],
                "state": "OPEN",
            }
        ]

    monkeypatch.setattr(GitHub, "issue_list", mock_issue_list)
    # run_fleet_operator_queue creates a real GitHub instance; short-circuit the
    # field-list probe, which would otherwise require an authenticated ``gh`` CLI.
    monkeypatch.setattr(GitHub, "validate_field_lists", lambda self: None)

    args = cli.build_parser().parse_args(["fleet", "operator-queue"])
    result = cli.run_fleet_operator_queue(args)

    assert result.ok is False
    assert "1 repo(s), 1 error(s)" in result.message
    # The good repo's queue still populated despite the broken repo's error.
    ok_queue = result.data["repos"]["owner/repo_ok"]["queue"]
    assert len(ok_queue) == 1
    assert ok_queue[0]["number"] == 42
    assert ok_queue[0]["reason_class"] == "mechanical"
    assert ok_queue[0]["terminal_since"] == terminal_since
    assert result.data["repos"]["owner/repo_ok"]["depth"] == 1
    # The broken repo is isolated into errors, not mixed into per_repo.
    assert len(result.data["errors"]) == 1
    assert result.data["errors"][0]["repo_key"] == "owner/repo_broken"
    assert "does not exist" in result.data["errors"][0]["error"]
    assert "owner/repo_broken" not in result.data["repos"]


def test_fleet_review_queue_aggregates_and_isolates_errors(tmp_path: Path, monkeypatch) -> None:
    """Issue #369: fleet review-queue aggregates per repo and isolates errors."""
    fleet_override = str(tmp_path / "fleet")
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", fleet_override)

    repo_ok = tmp_path / "repo_ok"
    repo_ok.mkdir()
    config_ok = repo_ok / "orchestrator.config.yaml"
    config_ok.write_text(
        "labels:\n  ready: automated-ready\n  queued: agent:queued\n  in_progress: agent:in-progress\nruntime:\n  state_dir: .var/charlie-work\n"
    )
    (repo_ok / ".var" / "charlie-work").mkdir(parents=True)

    # Good repo has one PR with a current packet and no decision
    prs_dir = repo_ok / ".var" / "charlie-work" / "prs" / "pr-7"
    prs_dir.mkdir(parents=True)
    (prs_dir / "pr.json").write_text(
        json.dumps({"number": 7, "headRefOid": "sha-7"}), encoding="utf-8"
    )
    (prs_dir / "review-prompt.md").write_text("packet for PR 7", encoding="utf-8")

    # Create a valid state.json so load_state doesn't fail
    (repo_ok / ".var" / "charlie-work" / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )

    fleet_json_path = Path(fleet_override) / "fleet.json"
    fleet_json_path.parent.mkdir(parents=True, exist_ok=True)
    registry_data = {
        "version": 1,
        "repos": {
            "owner/repo_ok": {
                "repo_root": str(repo_ok),
                "name_with_owner": "owner/repo_ok",
                "config_path": str(config_ok),
                "state_dir": str(repo_ok / ".var" / "charlie-work"),
                "first_seen": "2026-07-06T12:00:00Z",
                "last_seen": "2026-07-06T12:00:00Z",
            },
            "owner/repo_broken": {
                "repo_root": str(tmp_path / "nonexistent"),
                "name_with_owner": "owner/repo_broken",
                "config_path": str(tmp_path / "nonexistent" / "orchestrator.config.yaml"),
                "state_dir": str(tmp_path / "nonexistent" / ".var" / "charlie-work"),
                "first_seen": "2026-07-06T12:00:00Z",
                "last_seen": "2026-07-06T12:00:00Z",
            },
        },
    }
    fleet_json_path.write_text(json.dumps(registry_data, indent=2))

    from charlie_work.github import GitHub

    def mock_pr_list(self):
        return [
            {
                "number": 7,
                "title": "Fix #7: thing",
                "url": "https://example.test/pull/7",
                "headRefName": "agent/issue-7-fix",
                "baseRefName": "main",
                "headRefOid": "sha-7",
                "mergeStateStatus": "CLEAN",
                "body": "Closes #7",
                "labels": [],
                "isCrossRepository": False,
                "state": "OPEN",
            }
        ]

    monkeypatch.setattr(GitHub, "pr_list", mock_pr_list)
    # run_fleet_review_queue creates a real GitHub instance; short-circuit the
    # field-list probe, which would otherwise require an authenticated ``gh`` CLI.
    monkeypatch.setattr(GitHub, "validate_field_lists", lambda self: None)

    args = cli.build_parser().parse_args(["fleet", "review-queue"])
    result = cli.run_fleet_review_queue(args)

    assert result.ok is False
    assert "1 repo(s), 1 error(s)" in result.message
    assert result.data["repos"]["owner/repo_ok"]["queue"] == [
        {
            "pr": 7,
            "issue": 7,
            "packet_head_sha": "sha-7",
            "decision": "missing",
            "reviewed_head_sha": None,
            "mergeable": None,
            "mergeStateStatus": "CLEAN",
        }
    ]
    assert len(result.data["errors"]) == 1
    assert result.data["errors"][0]["repo_key"] == "owner/repo_broken"
    assert "does not exist" in result.data["errors"][0]["error"]
