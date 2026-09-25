"""Dispatch blocker, dependency, and cross-repo gating.

Split out of ``tests/test_charlie_work.py`` (issue #1548, Track-1 wave 2/8):
the blocker-gate seam of ``dispatch()`` -- open-blocker skips, cyclic and
self-referential dependencies, prose-only dependency labels, batched
reachability lookups, and the cross-repo escalation gate. Shared fakes and
helpers in ``tests/_dispatch_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from _dispatch_fixtures import (
    _cross_repo_issue_body,
    _write_fleet_registry,
    _write_sibling_repo_file,
)
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github import FakeGitHub
from charlie_work.config import (
    DevinConfig,
    DispatchConfig,
    OrchestratorConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp


def test_dispatch_skips_issue_with_open_blocker(tmp_path: Path) -> None:
    """Issue #108: dispatch should skip issues with open blockers."""
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
                    "labels": [],  # Not ready, so won't be in dispatch list
                    "state": "OPEN",  # Still open, so should block
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
            # Return that #743 is open
            return {743}

    fake_gh = FakeGitHubWithBlockers()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(limit=10)

    # Issue 752 should be skipped due to open blocker #743
    assert result.data["selected_count"] == 0
    assert result.data["attempted_count"] == 0

    # Check that dispatch_skip_blocked event was logged
    state = load_state(paths.state_file)
    blocked_events = [
        e for e in state.get("events", []) if e.get("kind") == "dispatch_skip_blocked"
    ]
    assert len(blocked_events) == 1
    assert blocked_events[0]["payload"]["issue"] == 752
    assert blocked_events[0]["payload"]["blockers"] == [743]


def test_dispatch_skips_when_any_blocker_open(tmp_path: Path) -> None:
    """Issue #108: dispatch should skip when ANY blocker is open (logical AND)."""
    config = OrchestratorConfig(
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create a fake GitHub with multiple blockers, one still open
    class FakeGitHubWithMultipleBlockers(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            # Override with only the test issues
            self.issues = [
                {
                    "number": 752,
                    "title": "Dependent issue",
                    "url": "https://example.test/issues/752",
                    "body": "Blocked by #743, #744",
                    "labels": [{"name": "automated-ready"}],
                    "state": "OPEN",
                },
                {
                    "number": 743,
                    "title": "Blocker issue 1",
                    "url": "https://example.test/issues/743",
                    "body": "Foundation work",
                    "labels": [],  # Not ready
                    "state": "CLOSED",
                },
                {
                    "number": 744,
                    "title": "Blocker issue 2",
                    "url": "https://example.test/issues/744",
                    "body": "More foundation work",
                    "labels": [],  # Not ready
                    "state": "OPEN",  # Still open
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
            # Return that #744 is open
            return {744}

    fake_gh = FakeGitHubWithMultipleBlockers()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(limit=10)

    # Issue 752 should be skipped since #744 is still open
    assert result.data["selected_count"] == 0

    # Check that both declared blockers are listed in the event (AC3)
    state = load_state(paths.state_file)
    blocked_events = [
        e for e in state.get("events", []) if e.get("kind") == "dispatch_skip_blocked"
    ]
    assert len(blocked_events) == 1
    assert blocked_events[0]["payload"]["issue"] == 752
    assert set(blocked_events[0]["payload"]["blockers"]) == {743, 744}


def test_dispatch_proceeds_when_blocker_closed(tmp_path: Path) -> None:
    """Issue #108: dispatch should proceed when blocker is closed."""
    config = OrchestratorConfig(
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create a fake GitHub with a blocked issue but closed blocker
    class FakeGitHubWithClosedBlocker(FakeGitHub):
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
                    "labels": [],  # Not ready, so won't be in dispatch list
                    "state": "CLOSED",  # Closed, so should not block
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
            # Return that #743 is NOT open
            return set()

    fake_gh = FakeGitHubWithClosedBlocker()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(limit=10)

    # Issue 752 should be dispatched since blocker is closed
    assert result.data["selected_count"] == 1
    assert result.data["attempted_count"] == 1

    # Check that no dispatch_skip_blocked event was logged
    state = load_state(paths.state_file)
    blocked_events = [
        e for e in state.get("events", []) if e.get("kind") == "dispatch_skip_blocked"
    ]
    assert len(blocked_events) == 0


def test_dispatch_handles_cyclic_dependency_declaration(tmp_path: Path) -> None:
    """Test that dispatch handles cyclic dependency declarations without crashing per issue #152."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create fake GitHub with cyclic dependency:
    # - Issue A blocks B, B blocks A (both have open blockers on each other)
    # - Issue C is independent
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 100,
            "title": "issue-a",
            "url": "https://github.com/test/repo/issues/100",
            "body": "Blocked by #200",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-01T00:00:00Z",
            "updatedAt": "2026-07-01T00:00:00Z",
            "state": "OPEN",
        },
        {
            "number": 200,
            "title": "issue-b",
            "url": "https://github.com/test/repo/issues/200",
            "body": "Blocked by #100",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-02T00:00:00Z",
            "updatedAt": "2026-07-02T00:00:00Z",
            "state": "OPEN",
        },
        {
            "number": 300,
            "title": "issue-c",
            "url": "https://github.com/test/repo/issues/300",
            "body": "Independent issue",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-03T00:00:00Z",
            "updatedAt": "2026-07-03T00:00:00Z",
            "state": "OPEN",
        },
    ]

    # Mock issue_list to return all ready issues for out-degree computation
    def mock_issue_list(labels=None, state=None):
        if labels and "automated-ready" in labels:
            return fake_gh.issues
        return []

    fake_gh.issue_list = mock_issue_list

    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    # Dispatch should not crash on cyclic dependencies
    # Since A and B block each other, both should be filtered out
    # Only C should be dispatchable
    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(limit=2)

    assert result.ok is True
    # Only C should be selected (A and B are mutually blocked)
    assert result.data["selected_count"] == 1
    selected_numbers = [s["issue_number"] for s in result.data["sessions"]]
    assert selected_numbers == [300]


def test_dispatch_handles_missing_created_at(tmp_path: Path) -> None:
    """Test that dispatch handles missing createdAt field, sorting last per issue #152."""
    config = OrchestratorConfig(
        # Issue #1903: the default-on tree-headroom cap (cpu_count//2) would
        # clamp this 3-wide dispatch on small hosts -- pin both host-load
        # knobs to 0 so the createdAt sort order is the only thing tested.
        dispatch=DispatchConfig(
            host_load_max_pytest_processes=0,
            host_load_max_pytest_trees=0,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create fake GitHub with issues, some missing createdAt
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 100,
            "title": "old-issue",
            "url": "https://github.com/test/repo/issues/100",
            "body": "Old issue",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-01T00:00:00Z",
            "updatedAt": "2026-07-01T00:00:00Z",
            "state": "OPEN",
        },
        {
            "number": 200,
            "title": "new-issue",
            "url": "https://github.com/test/repo/issues/200",
            "body": "New issue",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-06T00:00:00Z",
            "updatedAt": "2026-07-06T00:00:00Z",
            "state": "OPEN",
        },
        {
            "number": 300,
            "title": "missing-date",
            "url": "https://github.com/test/repo/issues/300",
            "body": "Issue without createdAt",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            # No createdAt field
            "updatedAt": "2026-07-03T00:00:00Z",
            "state": "OPEN",
        },
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    # Dispatch all 3 - missing createdAt should sort last
    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(limit=3)

    assert result.ok is True
    assert result.data["selected_count"] == 3
    selected_numbers = [s["issue_number"] for s in result.data["sessions"]]
    # Order: 100 (oldest), 200 (newest), 300 (missing date, last)
    assert selected_numbers == [100, 200, 300]


def test_dispatch_handles_self_reference_blocker(tmp_path: Path) -> None:
    """Issue #108: self-referencing blockers should be filtered out with warning."""
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create a fake GitHub with a self-referencing blocker
    class FakeGitHubWithSelfRef(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            # Override with only the test issue
            self.issues = [
                {
                    "number": 123,
                    "title": "Self-referencing issue",
                    "url": "https://example.test/issues/123",
                    "body": "Blocked by #123",  # Self-reference
                    "labels": [{"name": "automated-ready"}],
                    "state": "OPEN",
                },
            ]

        def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
            # Return that the self-reference is "open" (it exists)
            return {123}

    fake_gh = FakeGitHubWithSelfRef()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Capture warning logs
    with patch("logging.Logger.warning") as mock_warning:
        app.gh.prs[0]["state"] = "CLOSED"
        result = app.dispatch(limit=10)

        # Issue should be dispatched (self-reference filtered out)
        assert result.data["selected_count"] == 1

        # Warning should have been logged for self-reference
        assert any(
            "self-referencing" in str(call.args[0]).lower() for call in mock_warning.call_args_list
        ), "Expected warning about self-referencing blocker"

    # No dispatch_skip_blocked event should be logged
    state = load_state(paths.state_file)
    blocked_events = [
        e for e in state.get("events", []) if e.get("kind") == "dispatch_skip_blocked"
    ]
    assert len(blocked_events) == 0


def test_dispatch_skips_prose_only_deps_labeled_issues(tmp_path: Path) -> None:
    """Issue #225: dispatch should skip issues labeled with prose-only-deps."""

    class ProseOnlyDepsGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = [
                {
                    "number": 908,
                    "title": "Issue with prose-only deps",
                    "url": "https://example.test/issues/908",
                    "body": "Do not dispatch before P2-T2/P2-T3 have landed.",
                    "labels": [{"name": "automated-ready"}, {"name": "agent:prose-only-deps"}],
                },
                {
                    "number": 910,
                    "title": "Normal issue",
                    "url": "https://example.test/issues/910",
                    "body": "Just a normal issue",
                    "labels": [{"name": "automated-ready"}],
                },
            ]

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = ProseOnlyDepsGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(limit=10)

    # Only issue 910 should be dispatched (908 is labeled with prose-only-deps)
    assert result.data["selected_count"] == 1
    assert result.data["attempted_count"] == 1
    # The dispatched issue should be 910, not 908
    assert result.data["sessions"][0]["issue_number"] == 910


def test_dispatch_reachability_blocker_lookups_are_batched(tmp_path: Path) -> None:
    """Issue #1110 rework: classify_backlog_reachability runs a per-issue
    blocker check inside _dispatch_impl. Without warming the pass-scoped
    cache first (the _prefetch_blocker_data warm-up issue #870 built for
    exactly this pattern), that check would issue N serial per-issue
    ``gh api .../dependencies/blocked_by`` calls -- one per ready issue --
    on every dispatch pass, on a superset of the issue population the
    dispatch candidate filter already blocker-checks.

    This test asserts the batched path is taken: the GitHub client's
    batched ``issue_dependencies`` method is called with every ready issue
    number, and zero per-issue dependency ``run`` calls escape (every
    lookup resolves from the warm cache). Against the unfixed code (no
    _prefetch_blocker_data call before reachability) the batch method is
    never invoked and N serial per-issue ``run`` calls are made instead.
    """
    config = OrchestratorConfig(devin=DevinConfig())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Three ready issues, each blocked by an open predecessor. The blocker
    # check inside classify_backlog_reachability runs on every one (they
    # all pass the label-only gates and reach the dependency-gate else
    # branch). The dispatch candidate filter also blocker-checks each as a
    # candidate, so both consumers must read the warm cache.
    ready_issues = [
        {
            "number": 11,
            "title": "t",
            "url": "https://example.test/issues/11",
            "body": "Blocked by #1",
            "labels": [{"name": "automated-ready"}],
            "state": "OPEN",
        },
        {
            "number": 12,
            "title": "t",
            "url": "https://example.test/issues/12",
            "body": "Blocked by #1",
            "labels": [{"name": "automated-ready"}],
            "state": "OPEN",
        },
        {
            "number": 13,
            "title": "t",
            "url": "https://example.test/issues/13",
            "body": "Blocked by #2",
            "labels": [{"name": "automated-ready"}],
            "state": "OPEN",
        },
    ]
    blocker_issues = [
        {
            "number": 1,
            "title": "b",
            "url": "https://example.test/issues/1",
            "body": "",
            "labels": [],
            "state": "OPEN",
        },
        {
            "number": 2,
            "title": "b",
            "url": "https://example.test/issues/2",
            "body": "",
            "labels": [],
            "state": "OPEN",
        },
    ]

    class FakeGitHubBatched(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = ready_issues + blocker_issues
            # Mirror the real GitHubClient's pass-scoped cache so
            # get_github_issue_dependencies reads/writes it.
            self._list_cache: dict[tuple[str, Any], Any] = {}
            self.issue_dependencies_calls: list[list[int]] = []
            self.dependency_run_calls = 0

        def issue_dependencies(self, issue_numbers: list[int]) -> dict[int, list[int]]:
            # The batched path _prefetch_blocker_data prefers. Mirrors the
            # real method's warm-cache contract: populate _list_cache so
            # subsequent per-issue get_github_issue_dependencies calls hit
            # the cache instead of issuing a live `gh run`.
            self.issue_dependencies_calls.append(list(issue_numbers))
            for n in issue_numbers:
                # No GitHub-native deps in this scenario (blockers are
                # body-declared); an empty list is a legitimate successful
                # resolution the real method caches on success.
                self._list_cache[("issue_dependencies", n)] = []
            return {n: [] for n in issue_numbers}

        def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
            return {n for n in issue_numbers if n in {1, 2}}

        def run(self, args, *, json_output=False, allow_failure=False):
            joined = " ".join(args)
            if "dependencies" in joined:
                # The per-issue serial path. With the cache warmed by
                # issue_dependencies, this must never be reached.
                self.dependency_run_calls += 1
                return [] if json_output else ""
            return super().run(args, json_output=json_output, allow_failure=allow_failure)

    fake_gh = FakeGitHubBatched()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)
    result = app.dispatch()

    assert result.ok
    # The batched dependency fetch ran -- _prefetch_blocker_data was
    # invoked before reachability's per-issue blocker check.
    assert len(fake_gh.issue_dependencies_calls) >= 1
    # Every ready issue number was handed to the batch call in one shot
    # (the warm-up covers the full ready set, not just candidates).
    fetched = set(fake_gh.issue_dependencies_calls[0])
    assert {11, 12, 13}.issubset(fetched)
    # Zero per-issue serial dependency `gh run` calls escaped: every
    # dependency lookup (reachability's per-issue check plus the dispatch
    # candidate filter's per-candidate check) resolved from the warm cache.
    # Against the unfixed code this is 3 (one serial call per ready issue
    # from reachability's uncached per-issue check).
    assert fake_gh.dependency_run_calls == 0


def test_dispatch_cross_repo_gate_escalates_when_all_paths_absent(tmp_path: Path) -> None:
    """Issue #1010 wiring: the real dispatch path escalates (not dispatches) an
    issue whose referenced file paths are all absent from the target repo.

    Drives ``OrchestratorApp.dispatch`` (the ``_dispatch_impl`` path that decides
    to escalate instead of dispatch), not ``cross_repo_gate`` in isolation, and
    asserts the three observable effects of the wiring: the label transition to
    ``agent:operator-queue`` (issue #1266: mechanical reason_class), the
    ``dispatch_cross_repo_escalated`` event, and
    exclusion from ``session_requests`` (``selected_count == 0``). A regression
    in the wiring that calls the gate would silently reintroduce the exact bug
    this PR fixes with every unit test green, so this test exercises the wiring.

    Issues #1756-#1758 (positive-evidence redesign): bare absence no longer
    escalates on its own -- a sibling repo must be registered in the fleet
    AND actually contain the missing path. This is also the call-site
    threading regression test for ``managed_repo_roots``/
    ``dispatching_repo_name``: dropping either argument at the real dispatch
    call site would make this fall back to abstaining (``selected_count``
    would go non-zero) with every other assertion in this file still green.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # The default fixture PR is open and linked to issue #123, which would
    # exclude the issue via the open-PR guard; close it so #123 is selectable.
    fake_gh.prs[0]["state"] = "CLOSED"
    # Every referenced path is absent from tmp_path (the repo root); one of
    # them is positively present under a registered sibling repo's root.
    fake_gh.issues[0]["body"] = _cross_repo_issue_body()
    sibling_root = tmp_path / "sibling-ci-runners"
    _write_sibling_repo_file(sibling_root, "src/ci_fleet/suite_coverage.py")
    _write_fleet_registry(
        tmp_path / "fleet",
        {"owner/ci-runners": {"repo_root": str(sibling_root)}},
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=1)

    # The issue was escalated, not dispatched: no session was requested.
    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert 123 in result.data["cross_repo_escalated_issue_numbers"]
    # No dispatch_results were produced for the escalated issue.
    assert result.data["dispatch_results"] == []

    # Label transition to operator-queue (the redispatch_escalated edge,
    # mechanical reason_class per issue #1266 -- a cross-repo target
    # mismatch is a structural/mechanical failure, not a judgment call).
    assert (123, config.labels.operator_queue) in fake_gh.labels_added

    # The dispatch_cross_repo_escalated event was recorded with the issue number
    # and a cross_repo_target reason.
    state = load_state(paths.state_file)
    escalated_events = [e for e in state["events"] if e["kind"] == "dispatch_cross_repo_escalated"]
    assert len(escalated_events) == 1
    assert escalated_events[0]["payload"]["issue_number"] == 123
    assert "cross_repo_target" in escalated_events[0]["payload"]["reason"]
    # Issue #1583: the event payload also carries ``neutral_paths`` and
    # ``missing_paths`` so the operator can see from the event alone which
    # citation tripped the gate. This body has no neutral candidates (every
    # referenced path is a plain missing dispatch target), so neutral_paths
    # is empty and missing_paths is the non-empty survivor set.
    payload = escalated_events[0]["payload"]
    assert isinstance(payload["neutral_paths"], list)
    assert isinstance(payload["missing_paths"], list)
    # escalated => every survivor is missing; this body has no neutral
    # candidates, so neutral_paths is empty and missing_paths is non-empty.
    assert payload["neutral_paths"] == []
    assert payload["missing_paths"]
    # Issues #1756-#1758: the event names the sibling the positive-evidence
    # check matched, turning a blind triage action into a one-glance fix.
    assert payload["found_in_repo"] == "ci-runners"

    # The issue is terminal in state, not left dispatch_pending.
    assert state["issues"]["123"]["status"] == "escalated"


def test_dispatch_cross_repo_override_label_skips_both_gates(tmp_path: Path) -> None:
    """Review finding 4: an issue carrying the cross-repo override label
    (``agent:cross-repo-override``) skips BOTH the cross-repo file-path
    gate and the cross-repo scope gate entirely and dispatches normally --
    the operator's one-step recovery for a residual false-positive
    escalation (issue #1758's own valve). This branch had zero test
    coverage before this test: a regression that made the override check
    always fail, or a copy-paste that only guarded one of the two gates,
    would go undetected with every other cross-repo test in this file
    green, since none of them carry the override label.

    Uses the SAME issue body and sibling registration as
    ``test_dispatch_cross_repo_gate_escalates_when_all_paths_absent`` --
    which escalates without the label -- so this test isolates the label
    as what changes the outcome, not some other fixture difference.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0]["state"] = "CLOSED"
    fake_gh.issues[0]["body"] = _cross_repo_issue_body()
    fake_gh.issues[0]["labels"].append({"name": config.labels.cross_repo_override})
    sibling_root = tmp_path / "sibling-ci-runners"
    _write_sibling_repo_file(sibling_root, "src/ci_fleet/suite_coverage.py")
    _write_fleet_registry(
        tmp_path / "fleet",
        {"owner/ci-runners": {"repo_root": str(sibling_root)}},
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=1)

    # The issue was dispatched normally -- the override short-circuited
    # both gates before either could escalate it.
    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert result.data["cross_repo_escalated_issue_numbers"] == []

    # No operator-queue escalation label was applied.
    assert (123, config.labels.operator_queue) not in fake_gh.labels_added

    # The override valve fired its own audit event; no escalation event
    # was recorded for the same issue.
    state = load_state(paths.state_file)
    overridden_events = [
        e for e in state["events"] if e["kind"] == "dispatch_cross_repo_gate_overridden"
    ]
    assert len(overridden_events) == 1
    assert overridden_events[0]["payload"]["issue_number"] == 123
    escalated_events = [e for e in state["events"] if e["kind"] == "dispatch_cross_repo_escalated"]
    assert escalated_events == []
