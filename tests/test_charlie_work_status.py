"""``fleet status`` aggregation: counts, blocked section, batched-GraphQL blocker prefetch, and stalled section.

Split out of ``tests/test_charlie_work.py`` (issue #1553,
Track-1 wave 7/8).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from _fakes_github import FakeGitHub
from _fake_transport import (
    FakeAdapter,
    connection_page,
    graphql_failure,
    graphql_ok,
    graphql_variables,
    make_github,
    ok,
)
from charlie_work.config import DevinConfig, OrchestratorConfig, WatchdogConfig
from charlie_work.github_transport import GraphQLRequest, Request, Response, RestRequest
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_status_aggregates_counts(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    result = app.status()

    assert result.ok is True
    assert result.data["ready_issue_count"] == 1
    assert result.data["available_issue_count"] == 1
    assert result.data["open_linked_pr_count"] == 1


def test_status_includes_blocked_section(tmp_path: Path) -> None:
    """Issue #108: status (roll-call) should include blocked section."""
    config = OrchestratorConfig(
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create a fake GitHub with a blocked issue
    class FakeGitHubWithBlockers(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            # Override with only the test issues
            self.issues = [
                {
                    "number": 752,
                    "title": "Dependent issue",
                    "url": "https://example.test/issues/752",
                    "body": "Blocked by #743",
                    "labels": [{"name": "automated-ready"}],
                    "state": "OPEN",
                },
                {
                    "number": 743,
                    "title": "Blocker issue",
                    "url": "https://example.test/issues/743",
                    "body": "Foundation work",
                    "labels": [],  # Not ready
                    "state": "OPEN",
                },
            ]

        def issue_list(self, labels=None, state=None):
            # Support both old and new signature
            if isinstance(labels, str):
                ready_label = labels
                # Only return issue 752 (the dependent one)
                return [
                    issue
                    for issue in self.issues
                    if ready_label in [label["name"] for label in issue.get("labels", [])]
                ]
            elif labels:
                return [
                    issue
                    for issue in self.issues
                    if any(
                        label in [label_obj["name"] for label_obj in issue.get("labels", [])]
                        for label in labels
                    )
                ]
            return self.issues

        def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
            return {743}

    fake_gh = FakeGitHubWithBlockers()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.status()

    # Check that blocked section is present
    assert "blocked" in result.data
    assert len(result.data["blocked"]) == 1
    assert result.data["blocked"][0]["issue"] == 752
    assert result.data["blocked"][0]["blockers"] == [743]

    # available_issue_count should exclude blocked issues
    assert result.data["available_issue_count"] == 0


def test_status_prefetch_uses_batched_graphql_for_blocker_data(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #923: `fleet status --json` spawned 86 `gh` subprocesses, mostly
    one `gh api .../dependencies/blocked_by` per ready issue plus one `gh issue
    view` per unique blocker. That is replaced by a single batched GraphQL
    query per repo that fetches both the native blockedBy relationships and the
    blockers' open/closed states in one request.

    This uses a real GitHub (not the hand-rolled FakeGitHub used elsewhere in
    this file) over a scripted fake transport, so every request status()
    actually makes is visible and countable.

    Two ready issues (887, 888) both declare the same open blocker (886) via
    GitHub-native dependencies. Before this fix: 2 dependency-REST calls x2
    consumers = 4, plus 886's issue-view fetched twice per consumer = 4. After
    the batching fix: one GraphQL query fetches dependencies and blocker states
    for the whole set; no further requests are needed.
    """

    def make_issue(number: int) -> dict[str, Any]:
        # A GraphQL ``Issue`` node (the shape ``gh issue list --json`` is built from).
        return {
            "number": number,
            "title": f"Issue {number}",
            "url": f"https://example.test/issues/{number}",
            "body": "",
            "labels": {"nodes": [{"name": "automated-ready"}]},
            "author": {"__typename": "User", "login": "tester"},
            "createdAt": "2026-01-01T00:00:00Z",
            "updatedAt": "2026-01-01T00:00:00Z",
            "state": "OPEN",
        }

    ready_issues = [make_issue(887), make_issue(888)]
    blocked_by = {
        "nodes": [{"number": 886, "state": "OPEN"}],
        "pageInfo": {"hasNextPage": False},
    }

    def handler(request: Request) -> Response:
        if isinstance(request, RestRequest):
            # Issue #1337: status() now calls merged_pr_list() to compute the
            # merged-PR coverage exclusion set for the reachability classifier.
            # No merged PRs in this test -> empty page breaks pagination.
            # (The startup field-list probes also list labels and workflow runs.)
            if request.route.endswith("/actions/runs"):
                return ok({"workflow_runs": []})
            if request.route.endswith("/issues"):
                # Issue #2443: open issues are one REST read, filtered locally.
                return ok(
                    [
                        {
                            "number": n,
                            "title": f"Issue {n}",
                            "html_url": f"https://example.test/issues/{n}",
                            "body": "",
                            "state": "open",
                            "labels": [{"name": "automated-ready", "node_id": "L1"}],
                            "user": {"login": "tester", "type": "User"},
                            "created_at": "2026-01-01T00:00:00Z",
                            "updated_at": "2026-01-01T00:00:00Z",
                        }
                        for n in (887, 888)
                    ]
                )
            assert request.route.endswith(("/pulls", "/labels")), request
            return ok([])
        document = request.document
        if "blockedBy" in document:
            return graphql_ok(
                {
                    "repository": {
                        "i_887": {"number": 887, "blockedBy": blocked_by},
                        "i_888": {"number": 888, "blockedBy": blocked_by},
                    }
                }
            )
        if "issues(" in document:
            return connection_page("issues", ready_issues)
        if "pullRequests(" in document:
            return connection_page("pullRequests", [])
        # OrchestratorApp.__init__'s validate_field_lists() startup probe: the
        # schema accepts the selection and the probed object does not exist.
        return graphql_failure("Could not resolve to a node")

    transport = FakeAdapter("http", handler=handler)

    config = OrchestratorConfig(devin=DevinConfig())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh, http, _ = make_github(tmp_path, http=transport)
    app = OrchestratorApp(tmp_path, paths, config, gh)

    result = app.status()

    assert result.ok is True
    assert result.data["available_issue_count"] == 0
    blocked_by_issue = {b["issue"]: b["blockers"] for b in result.data["blocked"]}
    assert blocked_by_issue == {887: [886], 888: [886]}
    summaries = {s["number"]: s["dependencies"] for s in result.data["issues"]}
    assert summaries[887] == {"declared": [886], "open": [886]}
    assert summaries[888] == {"declared": [886], "open": [886]}

    requests = http.api_requests
    dependency_queries = [
        r for r in requests if isinstance(r, GraphQLRequest) and "blockedBy" in r.document
    ]
    rest_dependency_calls = [
        r for r in requests if isinstance(r, RestRequest) and "dependencies/blocked_by" in r.route
    ]
    # (The startup field-list probe views issue number 0; any other view is a fetch.)
    issue_view_calls = [
        r
        for r in requests
        if isinstance(r, GraphQLRequest)
        and "issue(number:$number)" in r.document.replace(" ", "")
        and graphql_variables(r)["number"] != 0
    ]

    # The whole dependency + blocker-state lookup is now one batched GraphQL
    # query for the two ready issues, not N per-issue REST calls + M issue views.
    assert len(dependency_queries) == 1
    assert len(rest_dependency_calls) == 0
    assert len(issue_view_calls) == 0


def test_status_includes_stalled_section(tmp_path: Path) -> None:
    """Issue #109: status (roll-call) should include stalled section."""
    from datetime import UTC, datetime, timedelta
    import os
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create a fake GitHub with a ready issue
    class FakeGitHubWithStalled(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = [
                {
                    "number": 109,
                    "title": "Test issue",
                    "url": "https://example.test/issues/109",
                    "body": "Test body",
                    "labels": [{"name": "automated-ready"}],
                    "state": "OPEN",
                },
            ]

        def issue_list(self, labels=None, state=None):
            # Support both old and new signature
            return self.issues

        def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
            return set()

    fake_gh = FakeGitHubWithStalled()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Create a fake stalled session sidecar
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Create a log file with old mtime (stalled by time)
    log_file = sessions_dir / "issue-109.log"
    log_file.write_text("working on issue\n", encoding="utf-8")
    old_time = datetime.now(UTC) - timedelta(minutes=25)
    timestamp = old_time.timestamp()
    os.utime(log_file, (timestamp, timestamp))

    # Create a sidecar with a fake PID
    sidecar = sessions_dir / "issue-109.json"
    sidecar.write_text(
        json.dumps(
            {
                "issue_number": 109,
                "branch": "agent/issue-109",
                "worktree_path": "/fake/path",
                "prompt_path": "/fake/prompt",
                "command": ["devin", "--print"],
                "pid": 99999,  # Fake PID that won't exist
                "started_at": datetime.now(UTC).isoformat(),
                "log_path": str(log_file),
                "error": None,
            }
        ),
        encoding="utf-8",
    )

    # Mock worker_fate.is_alive to return True for PID 99999 so detection runs
    with patch("charlie_work.worker_fate.is_alive", return_value=True):
        result = app.status()

    # Check that stalled section contains the issue number and pid
    assert "stalled" in result.data
    assert isinstance(result.data["stalled"], list)
    assert any(entry["issue"] == 109 for entry in result.data["stalled"])
    assert any(entry["pid"] == 99999 for entry in result.data["stalled"])


def test_status_stalled_section_unchanged(tmp_path: Path) -> None:
    """Issue #167: stalled section keeps its base {issue, pid} shape. Issue #261
    intentionally extends each entry with a "health" field (STALLED vs DEAD) so
    digest callers can surface dead-worker terminal cause instead of collapsing
    everything to "STALLED"; terminal_tool/terminal_reason are added only for
    DEAD entries with a matching post-mortem. This test now pins that extended
    shape rather than the original byte-for-byte one.
    """
    from datetime import UTC, datetime, timedelta
    import os
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create a fake GitHub with a ready issue
    class FakeGitHubWithStalled(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = [
                {
                    "number": 109,
                    "title": "Test issue",
                    "url": "https://example.test/issues/109",
                    "body": "Test body",
                    "labels": [{"name": "automated-ready"}],
                    "state": "OPEN",
                },
            ]

        def issue_list(self, labels=None, state=None):
            return self.issues

        def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
            return set()

    fake_gh = FakeGitHubWithStalled()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Create a fake stalled session sidecar
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Create a log file with old mtime (stalled by time)
    log_file = sessions_dir / "issue-109.log"
    log_file.write_text("working on issue\n", encoding="utf-8")
    old_time = datetime.now(UTC) - timedelta(minutes=25)
    timestamp = old_time.timestamp()
    os.utime(log_file, (timestamp, timestamp))

    # Create a sidecar with a fake PID
    sidecar = sessions_dir / "issue-109.json"
    sidecar.write_text(
        json.dumps(
            {
                "issue_number": 109,
                "branch": "agent/issue-109",
                "worktree_path": "/fake/path",
                "prompt_path": "/fake/prompt",
                "command": ["devin", "--print"],
                "pid": 99999,
                "started_at": datetime.now(UTC).isoformat(),
                "log_path": str(log_file),
                "error": None,
            }
        ),
        encoding="utf-8",
    )

    # Mock worker_fate.is_alive to return True for PID 99999 so detection runs
    with patch("charlie_work.worker_fate.is_alive", return_value=True):
        result = app.status()

    # Check that the stalled section keeps its base shape plus the issue #261
    # "health" field. This fixture is live (mocked) with a stale log, so it
    # classifies as STALLED (not DEAD), which means no terminal_tool/
    # terminal_reason keys are added (those are DEAD-only, per
    # _detect_stalled_sessions).
    assert "stalled" in result.data
    assert isinstance(result.data["stalled"], list)
    assert any(entry["issue"] == 109 for entry in result.data["stalled"])
    assert any(entry["pid"] == 99999 for entry in result.data["stalled"])
    for entry in result.data["stalled"]:
        assert set(entry.keys()) == {"issue", "pid", "health"}
        assert entry["health"] == "STALLED"
