"""Closed-unmerged issue finalization, finalize caps, and the merged-lookup circuit breaker.

Split out of ``tests/test_charlie_work.py`` (issue #1548, Track-1 wave 2/8):
the finalization seam of ``dispatch()`` -- closing out issues whose PRs are
gone or merged, per-pass finalize caps, and the circuit breaker that stops
externally-merged issue lookups after repeated failures. Shared fakes and
helpers in ``tests/_dispatch_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path

from _fakes_github import FakeGitHub
from charlie_work import github as github_module
from charlie_work.config import (
    DispatchConfig,
    OrchestratorConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
)
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_dispatch_finalizes_closed_issue_for_aviator_merge(tmp_path: Path) -> None:
    """Issue #427: an Aviator-mergequeue merged PR closes its linked issue via
    GitHub's 'Closes #N'. The issue may already be CLOSED while still carrying
    stale automated-ready and agent:pr-open labels. dispatch() must scan
    merged PRs regardless of the issue's open/closed state, run the merged
    label transition, and clean up state.json.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class FakeGitHubRespectingState(FakeGitHub):
        def issue_list(self, labels=None, state=None):
            issues = super().issue_list(labels=labels, state=state)
            if state is None:
                state = "OPEN"
            if state.upper() == "ALL":
                return issues
            return [i for i in issues if (i.get("state") or "OPEN").upper() == state.upper()]

    fake_gh = FakeGitHubRespectingState()

    # Seed state as if merge_ready handed the PR off to Aviator.
    seed = load_state(paths.state_file)
    seed["prs"]["456"] = {"status": "mergequeue", "issue_number": 123}
    seed["issues"]["123"] = {"status": "reviewing", "number": 123}
    save_state(paths.state_file, seed)

    # Aviator merged the PR; GitHub auto-closed the issue.
    fake_gh.prs[0]["state"] = "MERGED"
    fake_gh.prs[0]["labels"] = [{"name": "mergequeue"}]
    fake_gh.issues[0]["state"] = "CLOSED"
    fake_gh.issues[0]["labels"] = [
        {"name": config.labels.ready},
        {"name": config.labels.pr_open},
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["merged_pr_referenced_issue_numbers"] == [123]
    assert result.data["merged_pr_closed_issue_numbers"] == [123]
    assert result.data["merged_pr_flagged_issue_numbers"] == []
    assert (123, config.labels.done) in fake_gh.labels_added
    assert (123, config.labels.pr_open) in fake_gh.labels_removed
    assert (123, config.labels.ready) in fake_gh.labels_removed
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "merged"
    assert state["prs"]["456"]["merged"] is True
    assert state["issues"]["123"]["status"] == "closed"


def test_dispatch_finalizes_closed_issue_with_pr_outside_merged_pr_list_window(
    tmp_path: Path,
) -> None:
    """Issue #433: a closed ready-labeled issue whose merged PR is older than the
    most-recent-500 window of ``merged_pr_list()`` must still be finalized.

    This simulates a repo with 1250+ merged PRs where the global list cannot see
    the linked PR, but a per-issue ``gh pr list --search`` lookup finds it.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class FakeGitHubOutsideWindow(FakeGitHub):
        def issue_list(self, labels=None, state=None):
            issues = super().issue_list(labels=labels, state=state)
            if state is None:
                state = "OPEN"
            if state.upper() == "ALL":
                return issues
            return [i for i in issues if (i.get("state") or "OPEN").upper() == state.upper()]

        def merged_pr_list(self):
            # Simulate the 500-window truncation: the real merged PR is not visible.
            return []

    fake_gh = FakeGitHubOutsideWindow()

    seed = load_state(paths.state_file)
    seed["prs"]["456"] = {"status": "mergequeue", "issue_number": 123}
    seed["issues"]["123"] = {"status": "reviewing", "number": 123}
    save_state(paths.state_file, seed)

    fake_gh.prs[0]["state"] = "MERGED"
    fake_gh.prs[0]["labels"] = [{"name": "mergequeue"}]
    fake_gh.issues[0]["state"] = "CLOSED"
    fake_gh.issues[0]["labels"] = [
        {"name": config.labels.ready},
        {"name": config.labels.pr_open},
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["merged_pr_referenced_issue_numbers"] == [123]
    assert result.data["merged_pr_closed_issue_numbers"] == [123]
    assert result.data["merged_pr_flagged_issue_numbers"] == []
    assert (123, config.labels.done) in fake_gh.labels_added
    assert (123, config.labels.pr_open) in fake_gh.labels_removed
    assert (123, config.labels.ready) in fake_gh.labels_removed
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "merged"
    assert state["prs"]["456"]["merged"] is True
    assert state["issues"]["123"]["status"] == "closed"


def test_dispatch_caps_merge_finalization_per_pass(tmp_path: Path) -> None:
    """Issue #432: merge-finalization is capped at dispatch.finalize_limit per pass.

    A backlog of bound-and-closed issues carrying a stale ready marker drains
    oldest-first across passes; no single pass can monopolize the pass budget.
    """
    config = OrchestratorConfig(dispatch=DispatchConfig(finalize_limit=2))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    ready = config.labels.ready

    class LabelMutatingFakeGitHub(FakeGitHub):
        def add_issue_label(self, number: int, label: str) -> bool:
            super().add_issue_label(number, label)
            for issue in self.issues:
                if issue["number"] != number:
                    continue
                names = {item.get("name") for item in issue["labels"]}
                if label not in names:
                    issue["labels"].append({"name": label})
                break
            return True

        def remove_issue_label(self, number: int, label: str) -> bool:
            super().remove_issue_label(number, label)
            for issue in self.issues:
                if issue["number"] == number:
                    issue["labels"] = [
                        item for item in issue["labels"] if item.get("name") != label
                    ]
                    break
            return True

    fake_gh = LabelMutatingFakeGitHub()
    fake_gh.issues = [
        {
            "number": 101,
            "title": "Fix one",
            "url": "https://example.test/issues/101",
            "body": "body one",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-01-01T00:00:00Z",
        },
        {
            "number": 102,
            "title": "Fix two",
            "url": "https://example.test/issues/102",
            "body": "body two",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-01-02T00:00:00Z",
        },
        {
            "number": 103,
            "title": "Fix three",
            "url": "https://example.test/issues/103",
            "body": "body three",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-01-03T00:00:00Z",
        },
    ]
    fake_gh.prs = [
        {
            "number": 201,
            "title": "Fix #101",
            "url": "https://example.test/pull/201",
            "headRefName": "agent/issue-101-fix-one",
            "baseRefName": "main",
            "headRefOid": "sha-101",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #101",
            "labels": [],
            "isCrossRepository": False,
            "state": "MERGED",
        },
        {
            "number": 202,
            "title": "Fix #102",
            "url": "https://example.test/pull/202",
            "headRefName": "agent/issue-102-fix-two",
            "baseRefName": "main",
            "headRefOid": "sha-102",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #102",
            "labels": [],
            "isCrossRepository": False,
            "state": "MERGED",
        },
        {
            "number": 203,
            "title": "Fix #103",
            "url": "https://example.test/pull/203",
            "headRefName": "agent/issue-103-fix-three",
            "baseRefName": "main",
            "headRefOid": "sha-103",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #103",
            "labels": [],
            "isCrossRepository": False,
            "state": "MERGED",
        },
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result1 = app.dispatch()

    assert result1.ok is True
    assert result1.data["selected_count"] == 0
    assert result1.data["merged_pr_referenced_issue_numbers"] == [101, 102, 103]
    assert result1.data["merged_pr_closed_issue_numbers"] == [101, 102]
    assert 101 in fake_gh.closed_issues
    assert 102 in fake_gh.closed_issues
    assert 103 not in fake_gh.closed_issues
    assert (101, config.labels.done) in fake_gh.labels_added
    assert (102, config.labels.done) in fake_gh.labels_added
    assert (103, config.labels.done) not in fake_gh.labels_added
    state = load_state(paths.state_file)
    assert state["issues"]["101"]["status"] == "closed"
    assert state["issues"]["102"]["status"] == "closed"
    assert "103" not in state["issues"]

    # Second pass drains the remaining bound-and-closed issue.
    result2 = app.dispatch()
    assert result2.ok is True
    assert result2.data["selected_count"] == 0
    assert result2.data["merged_pr_closed_issue_numbers"] == [103]
    assert 103 in fake_gh.closed_issues
    assert (103, config.labels.done) in fake_gh.labels_added
    state = load_state(paths.state_file)
    assert state["issues"]["103"]["status"] == "closed"


def test_dispatch_caps_externally_merged_issue_lookups_at_finalize_limit(
    tmp_path: Path,
) -> None:
    """Issue #433 review: per-issue merged-PR lookups are capped at
    ``dispatch.finalize_limit`` and processed oldest-first.
    """
    config = OrchestratorConfig(dispatch=DispatchConfig(finalize_limit=2))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    ready = config.labels.ready

    class FakeGitHubCappedOutsideWindow(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.merged_prs_for_issue_calls: list[int] = []

        def merged_pr_list(self):
            # Outside the 500-window: the cheap list cannot see any merged PRs.
            return []

        def merged_prs_for_issue(self, issue_number: int, branch_prefix: str):
            self.merged_prs_for_issue_calls.append(issue_number)
            return github_module._MergedPRSearchResult(
                super().merged_prs_for_issue(issue_number, branch_prefix),
                ok=True,
            )

        def add_issue_label(self, number: int, label: str) -> bool:
            super().add_issue_label(number, label)
            for issue in self.issues:
                if issue["number"] != number:
                    continue
                names = {item.get("name") for item in issue["labels"]}
                if label not in names:
                    issue["labels"].append({"name": label})
                break
            return True

        def remove_issue_label(self, number: int, label: str) -> bool:
            super().remove_issue_label(number, label)
            for issue in self.issues:
                if issue["number"] == number:
                    issue["labels"] = [
                        item for item in issue["labels"] if item.get("name") != label
                    ]
                    break
            return True

    fake_gh = FakeGitHubCappedOutsideWindow()
    fake_gh.issues = [
        {
            "number": 101,
            "title": "Fix one",
            "url": "https://example.test/issues/101",
            "body": "body one",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-01-03T00:00:00Z",
        },
        {
            "number": 102,
            "title": "Fix two",
            "url": "https://example.test/issues/102",
            "body": "body two",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-01-01T00:00:00Z",
        },
        {
            "number": 103,
            "title": "Fix three",
            "url": "https://example.test/issues/103",
            "body": "body three",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-01-02T00:00:00Z",
        },
    ]
    fake_gh.prs = [
        {
            "number": 201,
            "title": "Fix #101",
            "url": "https://example.test/pull/201",
            "headRefName": "agent/issue-101-fix-one",
            "baseRefName": "main",
            "headRefOid": "sha-101",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #101",
            "labels": [],
            "isCrossRepository": False,
            "state": "MERGED",
        },
        {
            "number": 202,
            "title": "Fix #102",
            "url": "https://example.test/pull/202",
            "headRefName": "agent/issue-102-fix-two",
            "baseRefName": "main",
            "headRefOid": "sha-102",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #102",
            "labels": [],
            "isCrossRepository": False,
            "state": "MERGED",
        },
        {
            "number": 203,
            "title": "Fix #103",
            "url": "https://example.test/pull/203",
            "headRefName": "agent/issue-103-fix-three",
            "baseRefName": "main",
            "headRefOid": "sha-103",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #103",
            "labels": [],
            "isCrossRepository": False,
            "state": "MERGED",
        },
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result1 = app.dispatch()

    assert result1.ok is True
    assert result1.data["selected_count"] == 0
    # Oldest issues first: 102 (Jan 1), 103 (Jan 2); 101 (Jan 3) is deferred.
    assert result1.data["merged_pr_closed_issue_numbers"] == [102, 103]
    assert fake_gh.merged_prs_for_issue_calls == [102, 103]
    assert 101 not in fake_gh.closed_issues
    assert 102 in fake_gh.closed_issues
    assert 103 in fake_gh.closed_issues

    # Second pass drains the remaining issue.
    result2 = app.dispatch()
    assert result2.ok is True
    assert result2.data["selected_count"] == 0
    assert result2.data["merged_pr_closed_issue_numbers"] == [101]
    assert fake_gh.merged_prs_for_issue_calls == [102, 103, 101]
    assert 101 in fake_gh.closed_issues


def test_dispatch_circuit_breaker_stops_externally_merged_issue_lookups(
    tmp_path: Path,
) -> None:
    """Issue #433 review: 3 consecutive merged_prs_for_issue failures stop the
    pass; the next pass resumes from the remaining issues.
    """
    config = OrchestratorConfig(dispatch=DispatchConfig(finalize_limit=10))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    ready = config.labels.ready

    class FakeGitHubWithFailingSearch(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.merged_prs_for_issue_calls: list[int] = []

        def merged_pr_list(self):
            return []

        def merged_prs_for_issue(self, issue_number: int, branch_prefix: str):
            self.merged_prs_for_issue_calls.append(issue_number)
            if issue_number in {101, 102, 103}:
                return github_module._MergedPRSearchResult([], ok=False)
            return github_module._MergedPRSearchResult(
                super().merged_prs_for_issue(issue_number, branch_prefix),
                ok=True,
            )

    fake_gh = FakeGitHubWithFailingSearch()
    fake_gh.issues = [
        {
            "number": 101,
            "title": "Fix one",
            "url": "https://example.test/issues/101",
            "body": "body one",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-01-01T00:00:00Z",
        },
        {
            "number": 102,
            "title": "Fix two",
            "url": "https://example.test/issues/102",
            "body": "body two",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-01-02T00:00:00Z",
        },
        {
            "number": 103,
            "title": "Fix three",
            "url": "https://example.test/issues/103",
            "body": "body three",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-01-03T00:00:00Z",
        },
        {
            "number": 104,
            "title": "Fix four",
            "url": "https://example.test/issues/104",
            "body": "body four",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-01-04T00:00:00Z",
        },
    ]
    fake_gh.prs = [
        {
            "number": 204,
            "title": "Fix #104",
            "url": "https://example.test/pull/204",
            "headRefName": "agent/issue-104-fix-four",
            "baseRefName": "main",
            "headRefOid": "sha-104",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #104",
            "labels": [],
            "isCrossRepository": False,
            "state": "MERGED",
        },
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch()

    assert result.ok is True
    assert result.data["selected_count"] == 0
    # 101-103 all fail; the circuit breaker opens before 104 is searched.
    assert fake_gh.merged_prs_for_issue_calls == [101, 102, 103]
    assert result.data["merged_pr_closed_issue_numbers"] == []
    assert 104 not in fake_gh.closed_issues

    # Next pass resumes from the beginning; if failures continue it stops again.
    result2 = app.dispatch()
    assert fake_gh.merged_prs_for_issue_calls == [101, 102, 103, 101, 102, 103]
    assert result2.data["merged_pr_closed_issue_numbers"] == []


def test_dispatch_circuit_breaker_resets_on_interleaved_success(tmp_path: Path) -> None:
    """Issue #446: the consecutive-failure circuit breaker resets on success.

    Sequence fail, fail, success, fail, fail, fail must not trip at the third
    candidate (the success resets the counter) and must trip after the final
    three consecutive failures, leaving the next candidate untouched.
    """
    config = OrchestratorConfig(dispatch=DispatchConfig(finalize_limit=10))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    ready = config.labels.ready

    class FakeGitHubWithInterleavedSearch(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.merged_prs_for_issue_calls: list[int] = []

        def merged_pr_list(self):
            return []

        def merged_prs_for_issue(self, issue_number: int, branch_prefix: str):
            self.merged_prs_for_issue_calls.append(issue_number)
            if issue_number == 103:
                return github_module._MergedPRSearchResult(
                    [
                        {
                            "number": 203,
                            "title": "Fix #103",
                            "url": "https://example.test/pull/203",
                            "headRefName": "agent/issue-103-fix",
                            "baseRefName": "main",
                            "headRefOid": "sha-103",
                            "mergeStateStatus": "CLEAN",
                            "body": "Closes #103",
                            "labels": [],
                            "isCrossRepository": False,
                            "state": "MERGED",
                        }
                    ],
                    ok=True,
                )
            if issue_number in {101, 102, 104, 105, 106}:
                return github_module._MergedPRSearchResult([], ok=False)
            if issue_number == 107:
                return github_module._MergedPRSearchResult(
                    [
                        {
                            "number": 207,
                            "title": "Fix #107",
                            "url": "https://example.test/pull/207",
                            "headRefName": "agent/issue-107-fix",
                            "baseRefName": "main",
                            "headRefOid": "sha-107",
                            "mergeStateStatus": "CLEAN",
                            "body": "Closes #107",
                            "labels": [],
                            "isCrossRepository": False,
                            "state": "MERGED",
                        }
                    ],
                    ok=True,
                )
            return github_module._MergedPRSearchResult(
                super().merged_prs_for_issue(issue_number, branch_prefix), ok=True
            )

    fake_gh = FakeGitHubWithInterleavedSearch()
    fake_gh.issues = [
        {
            "number": n,
            "title": f"Issue {n}",
            "url": f"https://example.test/issues/{n}",
            "body": f"body {n}",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": f"2026-01-{i:02d}T00:00:00Z",
        }
        for i, n in enumerate(range(101, 108), start=1)
    ]
    fake_gh.prs = []

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch()

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert fake_gh.merged_prs_for_issue_calls == [101, 102, 103, 104, 105, 106]
    assert 107 not in fake_gh.merged_prs_for_issue_calls
    assert result.data["merged_pr_closed_issue_numbers"] == [103]
    assert 103 in fake_gh.closed_issues
    assert 107 not in fake_gh.closed_issues
    state = load_state(paths.state_file)
    assert state["issues"]["103"]["status"] == "closed"
    assert "107" not in state["issues"]


def test_dispatch_strips_ready_from_closed_unmerged_issue(tmp_path: Path) -> None:
    """Issue #429: a closed ready issue with no merged PR binding it is stripped.

    Uses fully self-contained fixtures so the test is hermetic and does not
    depend on sibling tests or module-level FakeGitHub defaults.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    ready = config.labels.ready

    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 123,
            "title": "Search is broken",
            "url": "https://example.test/issues/123",
            "body": "Search is broken",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-07-01T00:00:00Z",
        }
    ]
    fake_gh.prs = []

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["merged_pr_referenced_issue_numbers"] == []
    assert result.data["merged_pr_closed_issue_numbers"] == []
    assert result.data["merged_pr_flagged_issue_numbers"] == []
    assert 123 not in fake_gh.closed_issues
    assert (123, ready) in fake_gh.labels_removed
    assert fake_gh.labels_added == []
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "closed"
    stripped_events = [
        e
        for e in state.get("events", [])
        if e.get("kind") == "dispatch_closed_unmerged_ready_stripped"
    ]
    assert len(stripped_events) == 1
    assert stripped_events[0]["payload"]["issue_numbers"] == [123]


def test_dispatch_closed_unmerged_skips_candidate_on_lookup_failure(
    tmp_path: Path,
) -> None:
    """If merged_prs_for_issue fails for a closed-unmerged candidate, skip it."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    ready = config.labels.ready

    class FakeGitHubFailingSearch(FakeGitHub):
        def merged_pr_list(self):
            return []

        def merged_prs_for_issue(self, issue_number: int, branch_prefix: str):
            return github_module._MergedPRSearchResult([], ok=False)

    fake_gh = FakeGitHubFailingSearch()
    fake_gh.issues = [
        {
            "number": 123,
            "title": "Search is broken",
            "url": "https://example.test/issues/123",
            "body": "Search is broken",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-07-01T00:00:00Z",
        }
    ]
    fake_gh.prs = []

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert (123, ready) not in fake_gh.labels_removed
    assert fake_gh.labels_added == []
    state = load_state(paths.state_file)
    stripped_events = [
        e
        for e in state.get("events", [])
        if e.get("kind") == "dispatch_closed_unmerged_ready_stripped"
    ]
    assert stripped_events == []


def test_dispatch_closed_unmerged_capped_at_finalize_limit(
    tmp_path: Path,
) -> None:
    """Issue #432/#433: closed-unmerged stripping is capped at dispatch.finalize_limit
    and processed oldest-first.
    """
    config = OrchestratorConfig(dispatch=DispatchConfig(finalize_limit=2))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    ready = config.labels.ready

    class FakeGitHubCapped(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.merged_prs_for_issue_calls: list[int] = []

        def merged_pr_list(self):
            return []

        def merged_prs_for_issue(self, issue_number: int, branch_prefix: str):
            self.merged_prs_for_issue_calls.append(issue_number)
            return github_module._MergedPRSearchResult([], ok=True)

    fake_gh = FakeGitHubCapped()
    fake_gh.issues = [
        {
            "number": 101,
            "title": "Fix one",
            "url": "https://example.test/issues/101",
            "body": "body one",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-01-03T00:00:00Z",
        },
        {
            "number": 102,
            "title": "Fix two",
            "url": "https://example.test/issues/102",
            "body": "body two",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-01-01T00:00:00Z",
        },
        {
            "number": 103,
            "title": "Fix three",
            "url": "https://example.test/issues/103",
            "body": "body three",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-01-02T00:00:00Z",
        },
    ]
    fake_gh.prs = []

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch()

    assert result.ok is True
    assert result.data["selected_count"] == 0
    # Oldest issues first: 102 (Jan 1), 103 (Jan 2); 101 (Jan 3) is deferred.
    assert fake_gh.merged_prs_for_issue_calls == [102, 103]
    state = load_state(paths.state_file)
    stripped_events = [
        e
        for e in state.get("events", [])
        if e.get("kind") == "dispatch_closed_unmerged_ready_stripped"
    ]
    assert len(stripped_events) == 1
    assert stripped_events[0]["payload"]["issue_numbers"] == [102, 103]
