"""Dispatch-time merged-PR exclusion and merged-PR mention flags.

Split out of ``tests/test_charlie_work.py`` (issue #1548, Track-1 wave 2/8):
the merged-PR-list seam of ``dispatch()`` -- merged-PR list fetching and
caching, exclusion of issues referenced by merged PRs, and merged-PR mention
flag set/clear/rearm semantics. Sibling seam:
``test_charlie_work_dispatch_finalize.py`` (closed-unmerged finalization and
lookup caps). Shared fakes and helpers in ``tests/_dispatch_fixtures.py``.
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


def test_dispatch_merged_pr_list_called_once_per_pass(tmp_path: Path) -> None:
    """Issue #446: merged_pr_list() is listed once per dispatch pass even when
    both finalization and the dispatch binding step need it.
    """
    config = OrchestratorConfig(dispatch=DispatchConfig(default_limit=1))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    ready = config.labels.ready

    class CountingFakeGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.merged_pr_list_calls = 0

        def merged_pr_list(self):
            self.merged_pr_list_calls += 1
            return super().merged_pr_list()

    fake_gh = CountingFakeGitHub()
    fake_gh.issues = [
        {
            "number": 101,
            "title": "Closed issue",
            "url": "https://example.test/issues/101",
            "body": "body one",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-01-01T00:00:00Z",
        },
        {
            "number": 102,
            "title": "Open issue",
            "url": "https://example.test/issues/102",
            "body": "body two",
            "labels": [{"name": ready}],
            "state": "OPEN",
            "createdAt": "2026-01-02T00:00:00Z",
        },
    ]
    fake_gh.prs = [
        {
            "number": 201,
            "title": "Fix #101",
            "url": "https://example.test/pull/201",
            "headRefName": "agent/issue-101-fix",
            "baseRefName": "main",
            "headRefOid": "sha-101",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #101",
            "labels": [],
            "isCrossRepository": False,
            "state": "MERGED",
        },
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch()

    assert result.ok is True
    assert fake_gh.merged_pr_list_calls == 1
    assert result.data["merged_pr_closed_issue_numbers"] == [101]
    assert result.data["selected_count"] == 1


def test_dispatch_merged_pr_bound_exclusion_unaffected_by_rearm(tmp_path: Path) -> None:
    """Issue #1336 acceptance criterion 3: ``bound`` exclusions (branch-prefix
    / closing-verb linkage) stay scan-based and are never lifted by a re-arm.
    A bound PR genuinely addressed the issue, so even an issue carrying the
    durable ``mention_rearmed_at`` marker must remain excluded and be closed.
    A regression that subtracts rearmed issues from the bound set (instead of
    only the mention-only set) makes this test fail.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # PR #456 binds to issue #123 via the branch prefix -> bound, not mention.
    fake_gh.prs[0]["state"] = "MERGED"
    fake_gh.prs[0]["headRefName"] = "agent/issue-123-fix-search"
    fake_gh.prs[0]["title"] = "Fix #123: search"
    fake_gh.prs[0]["body"] = "Closes #123\n\nTests: regression coverage added."

    # Pre-seed state with both a mention flag timestamp AND a re-arm marker to
    # prove the bound exclusion ignores re-arm entirely.
    seed = load_state(paths.state_file)
    seed["issues"]["123"] = {
        "number": 123,
        "merged_pr_mention_flagged_at": "2026-01-01T00:00:00Z",
        "mention_rearmed_at": "2026-01-02T00:00:00Z",
    }
    save_state(paths.state_file, seed)

    result = app.dispatch(limit=1)
    assert result.ok is True
    # Bound issue stays excluded and is closed -- re-arm does not lift bound.
    assert 123 in result.data["merged_pr_referenced_issue_numbers"]
    assert 123 in fake_gh.closed_issues
    assert result.data["selected_count"] == 0


def test_dispatch_ignores_cross_repo_pr_mentioning_ready_issue(tmp_path: Path) -> None:
    """Regression for the isCrossRepository guard (workflow.py,
    _merged_pr_referenced_issue_numbers): a merged PR whose provenance is
    cross-repo (isCrossRepository=True) must never have its free-text "issue
    #N" mention counted, even though the text itself is indistinguishable
    from a same-repo mention. isCrossRepository describes head-branch
    provenance, not which repo the text refers to, but it is the only
    provenance signal available and must still gate the mention scan.

    Before this test, mutating the guard to unconditionally count mentions
    (e.g. `if True:` regardless of isCrossRepository) passed every existing
    test — this pins the behavior so that mutation is caught: issue #123
    must remain a normal, undisturbed dispatch candidate.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Merged PR #456 is cross-repo (e.g. a fork), has no branch/closing-keyword
    # binding, but its body text mentions "issue #123".
    fake_gh.prs[0]["state"] = "MERGED"
    fake_gh.prs[0]["headRefName"] = "some-fork-branch"
    fake_gh.prs[0]["title"] = "unrelated fork PR"
    fake_gh.prs[0]["body"] = "Unrelated change; see issue #123 in a different project."
    fake_gh.prs[0]["isCrossRepository"] = True

    result = app.dispatch(limit=1)

    assert result.ok is True
    # Issue #123 is dispatched normally — the cross-repo mention must not
    # exclude or flag it.
    assert result.data["selected_count"] == 1
    assert result.data["merged_pr_referenced_issue_numbers"] == []
    assert result.data["merged_pr_closed_issue_numbers"] == []
    assert result.data["merged_pr_flagged_issue_numbers"] == []
    assert 123 not in fake_gh.closed_issues
    assert (123, "agent:human-needed") not in fake_gh.labels_added


def test_dispatch_skips_merged_pr_list_query_when_no_ready_issues(tmp_path: Path) -> None:
    """Issue #361: when there are no ready-labeled issues to consider,
    merged_pr_list() must not be called at all.
    _merged_pr_referenced_issue_numbers() intersects its scan against the
    ready-issue-number set, so with zero ready issues it always returns empty
    sets regardless of what merged_pr_list() would have returned — the query
    is pure waste on every pass with an empty ready queue, and this is the
    dominant case the guard is meant to eliminate.
    """

    class FakeGitHubCountingMergedPrCalls(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.merged_pr_list_calls = 0
            self.issues = []  # no ready-labeled issues at all

        def merged_pr_list(self):
            self.merged_pr_list_calls += 1
            return super().merged_pr_list()

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubCountingMergedPrCalls()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert fake_gh.merged_pr_list_calls == 0


def test_dispatch_still_queries_merged_pr_list_when_ready_issues_exist(tmp_path: Path) -> None:
    """Sanity counterpart to the guard above: when a ready issue IS in the
    queue, merged_pr_list() must still be called — issue #203's guarantee
    (never re-dispatch an issue a merged PR already covers) depends on it.
    """

    class FakeGitHubCountingMergedPrCalls(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.merged_pr_list_calls = 0

        def merged_pr_list(self):
            self.merged_pr_list_calls += 1
            return super().merged_pr_list()

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubCountingMergedPrCalls()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert fake_gh.merged_pr_list_calls == 1


def test_dispatch_skips_ready_issue_with_merged_pr_reference(tmp_path: Path) -> None:
    """Issue #203: a ready issue whose merged PR is hijack-safely *bound* to it
    (here: the default fixture's ``agent/issue-123-...`` head branch matches the
    configured branch prefix) must not be dispatched, and — because binding is
    the same trust level issue #220 uses at merge time — the issue should be
    closed and labeled done as a belt-and-suspenders retry.

    Note: this fixture's PR title/body also happens to contain an "issue #123"
    text mention, but that is incidental to this test — the branch binding
    alone is sufficient to authorize the close. The mention-only path (no
    branch/closing-keyword binding) is isolated separately in
    test_dispatch_flags_but_does_not_close_ready_issue_with_bare_mention,
    which is the actual regression coverage for a bare-mention reference.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # PR #456 is merged; its headRefName ("agent/issue-123-fix-search", from
    # the default fixture) safely binds it to issue #123 via branch prefix.
    fake_gh.prs[0]["state"] = "MERGED"
    fake_gh.prs[0]["title"] = "fix(scope): reap sidecar files on session exit (issue #123)"
    fake_gh.prs[0]["body"] = "This PR addresses issue #123."
    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["merged_pr_referenced_issue_numbers"] == [123]
    assert result.data["merged_pr_closed_issue_numbers"] == [123]
    assert result.data["merged_pr_flagged_issue_numbers"] == []
    assert 123 in fake_gh.closed_issues
    assert (123, "agent:done") in fake_gh.labels_added
    assert (123, "agent:queued") not in fake_gh.labels_added
    assert (123, "agent:in-progress") not in fake_gh.labels_added
    prompt_path = tmp_path / ".var" / "charlie-work" / "issues" / "issue-123" / "worker-prompt.md"
    assert not prompt_path.exists()


def test_dispatch_flags_but_does_not_close_ready_issue_with_bare_mention(
    tmp_path: Path,
) -> None:
    """Issue #203 (review redesign): a merged PR that only *mentions* the issue
    in free text — no branch-prefix binding, no closing keyword — must never
    authorize an autonomous close. It must be excluded from dispatch and
    flagged with the human-needed label, and the issue must stay open, for
    the operator to decide.

    Isolates the mention-only path from the hijack-safe binding path: unlike
    test_dispatch_skips_ready_issue_with_merged_pr_reference (which reuses the
    default fixture's agent/issue-123-... head branch), this PR's head branch
    does not match the configured branch prefix and its body contains no
    closing keyword, so linked_issue_number() cannot bind it — only the loose
    "issue #N" text scan can find it.

    Mutation-verify: gutting issue_numbers_mentioned_by_pr() to return an
    empty set makes this test fail (selected_count would become 1 and no
    human-needed label would be added).
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # PR #456 is merged, same-repo, but its head branch does not match the
    # "agent/issue" prefix and neither title nor body has a closing keyword —
    # only a bare "issue #123" mention.
    fake_gh.prs[0]["state"] = "MERGED"
    fake_gh.prs[0]["headRefName"] = "cleanup-unrelated-branch"
    fake_gh.prs[0]["title"] = "chore: unrelated cleanup"
    fake_gh.prs[0]["body"] = "While in the area, this also happens to fix issue #123."
    assert fake_gh.prs[0]["isCrossRepository"] is False

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["merged_pr_referenced_issue_numbers"] == [123]
    assert result.data["merged_pr_closed_issue_numbers"] == []
    assert result.data["merged_pr_flagged_issue_numbers"] == [123]
    assert 123 not in fake_gh.closed_issues
    assert (123, "agent:human-needed") in fake_gh.labels_added
    assert (123, "agent:done") not in fake_gh.labels_added
    assert (123, "agent:queued") not in fake_gh.labels_added
    assert (123, "agent:in-progress") not in fake_gh.labels_added
    # The issue itself is left open — only the label lifecycle is touched.
    assert fake_gh.issue_view(123)["state"] == "OPEN"


def test_dispatch_merged_pr_mention_flag_is_one_shot(tmp_path: Path) -> None:
    """Issue #564: the merged-PR mention-only flag must fire exactly once per
    issue, not every pass. Before the fix, dispatch_merged_pr_mention_flagged
    re-fired on every pass for the same issue and the merged_pr_mention_flagged
    label transition re-applied agent:human-needed every pass — so an operator
    who removed the label could not win, and the event stream was spammed with
    one event per pass while the mention persisted.

    This pins the three required properties from issue #564:
      1. The flag fires once (first pass): one event, one label transition.
      2. A second pass with unchanged conditions emits no event and re-applies
         no label.
      3. An operator label removal survives subsequent passes: the label is
         never re-applied once merged_pr_mention_flagged_at is recorded.

    Re-flag semantics chosen (issue #564 point 3): keyed on the timestamp's
    absence — never re-flag once flagged, even if a new merged PR mentions the
    issue. Mutating the guard to re-fire on every pass makes this test fail.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    human_needed = config.labels.human_needed
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Merged PR #456 only *mentions* issue #123 in free text — no branch-prefix
    # binding, no closing keyword — so only the loose mention scan finds it.
    fake_gh.prs[0]["state"] = "MERGED"
    fake_gh.prs[0]["headRefName"] = "cleanup-unrelated-branch"
    fake_gh.prs[0]["title"] = "chore: unrelated cleanup"
    fake_gh.prs[0]["body"] = "While in the area, this also happens to fix issue #123."
    assert fake_gh.prs[0]["isCrossRepository"] is False

    # --- Pass 1: flag fires once -------------------------------------------
    result1 = app.dispatch(limit=1)
    assert result1.ok is True
    assert result1.data["merged_pr_flagged_issue_numbers"] == [123]
    assert (123, human_needed) in fake_gh.labels_added
    state1 = load_state(paths.state_file)
    assert state1["issues"]["123"].get("merged_pr_mention_flagged_at") is not None
    flagged_events_1 = [
        e
        for e in state1.get("events", [])
        if e.get("kind") == "dispatch_merged_pr_mention_flagged"
    ]
    assert len(flagged_events_1) == 1
    assert flagged_events_1[0]["payload"]["issue_numbers"] == [123]
    human_needed_adds_after_pass1 = fake_gh.labels_added.count((123, human_needed))

    # --- Pass 2: unchanged conditions -> no re-flag, no re-event ------------
    result2 = app.dispatch(limit=1)
    assert result2.ok is True
    # No newly-flagged issues this pass.
    assert result2.data["merged_pr_flagged_issue_numbers"] == []
    # The human-needed label was NOT re-applied on the second pass.
    assert fake_gh.labels_added.count((123, human_needed)) == human_needed_adds_after_pass1
    state2 = load_state(paths.state_file)
    flagged_events_2 = [
        e
        for e in state2.get("events", [])
        if e.get("kind") == "dispatch_merged_pr_mention_flagged"
    ]
    assert len(flagged_events_2) == 1  # still only the single pass-1 event

    # --- Pass 3: operator removed agent:human-needed -> removal survives ----
    # Simulate the operator's ruling: record a label removal and clear the
    # add-tracking so any re-application would be observable. The one-shot
    # guard keys off state, not the label, so the flag must not re-fire.
    fake_gh.labels_removed.append((123, human_needed))
    fake_gh.labels_added.clear()
    result3 = app.dispatch(limit=1)
    assert result3.ok is True
    assert result3.data["merged_pr_flagged_issue_numbers"] == []
    # The operator's removal survives: human-needed is never re-applied.
    assert (123, human_needed) not in fake_gh.labels_added
    state3 = load_state(paths.state_file)
    flagged_events_3 = [
        e
        for e in state3.get("events", [])
        if e.get("kind") == "dispatch_merged_pr_mention_flagged"
    ]
    assert len(flagged_events_3) == 1  # still only the single pass-1 event


def test_dispatch_merged_pr_mention_flag_skips_dedup_marker_on_partial_failure(
    tmp_path: Path,
) -> None:
    """A PARTIAL_FAILURE label transition for merged_pr_mention_flagged must
    NOT stamp merged_pr_mention_flagged_at. Before the fix, the dedup marker
    was written unconditionally regardless of transition()'s outcome, so a
    failed agent:human-needed label add still left the timestamp in state --
    and the one-shot guard exercised by
    test_dispatch_merged_pr_mention_flag_is_one_shot keys off exactly that
    timestamp's presence, so the issue would never be retried even though
    its label never actually changed.

    Pass 2 then clears the injected failure and re-dispatches, proving the
    withheld marker actually causes a retry -- not just that it is absent.
    """
    config = OrchestratorConfig()
    human_needed = config.labels.human_needed

    class LabelFailMentionGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.fail_add = True

        def add_issue_label(self, number: int, label: str) -> bool:
            if label == human_needed and self.fail_add:
                # Simulate a failed add (error-as-value) -> PARTIAL_FAILURE.
                return False
            return super().add_issue_label(number, label)

    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = LabelFailMentionGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Same mention-only setup as test_dispatch_merged_pr_mention_flag_is_one_shot:
    # merged PR #456 only *mentions* issue #123 in free text -- no branch-prefix
    # binding, no closing keyword -- so only the loose mention scan finds it.
    fake_gh.prs[0]["state"] = "MERGED"
    fake_gh.prs[0]["headRefName"] = "cleanup-unrelated-branch"
    fake_gh.prs[0]["title"] = "chore: unrelated cleanup"
    fake_gh.prs[0]["body"] = "While in the area, this also happens to fix issue #123."

    # --- Pass 1: label add fails -> marker withheld, no event ---------------
    result = app.dispatch(limit=1)
    assert result.ok is True

    # The add was attempted (proves the transition actually ran) but failed.
    assert (123, human_needed) not in fake_gh.labels_added

    state = load_state(paths.state_file)
    issue_entry = state["issues"].get("123", {})
    # The dedup marker must be ABSENT: a retry must occur on the next pass
    # instead of being permanently suppressed by a marker stamped despite
    # the label write having failed.
    assert issue_entry.get("merged_pr_mention_flagged_at") is None

    # The one-shot event is gated on the same success condition as the
    # marker, so it must not have fired either.
    flagged_events = [
        e for e in state.get("events", []) if e.get("kind") == "dispatch_merged_pr_mention_flagged"
    ]
    assert flagged_events == []

    # --- Pass 2: label add now succeeds -> the withheld marker must let the
    # mention scan retry, not permanently suppress the issue. This is the
    # behavior the marker's absence exists to enable -- proving it, not just
    # the marker's absence, is what distinguishes "will retry" from "silently
    # dropped."
    fake_gh.fail_add = False
    result2 = app.dispatch(limit=1)
    assert result2.ok is True
    assert result2.data["merged_pr_flagged_issue_numbers"] == [123]
    assert (123, human_needed) in fake_gh.labels_added

    state2 = load_state(paths.state_file)
    assert state2["issues"]["123"].get("merged_pr_mention_flagged_at") is not None
    flagged_events_2 = [
        e
        for e in state2.get("events", [])
        if e.get("kind") == "dispatch_merged_pr_mention_flagged"
    ]
    assert len(flagged_events_2) == 1
    assert flagged_events_2[0]["payload"]["issue_numbers"] == [123]


def test_dispatch_merged_pr_mention_rearmed_re_enters_dispatch(tmp_path: Path) -> None:
    """Issue #1336: an operator who reviews a mention-only flag and
    deliberately re-arms the issue (removes agent:human-needed) must get it
    dispatched again on the next pass. Before the fix, the mention-only
    dispatch exclusion keyed off the raw mention scan, which recomputed
    from the merged PR's immutable text on every pass and blocked the issue
    forever -- the #564 point-2 limitation.

    This pins acceptance criterion 1 (re-arm re-enters dispatch) and the
    one-shot flag preservation (no re-flag on the re-arm pass). Uses a
    label-mutating fake so the issue's labels genuinely reflect the
    operator's removal, exercising the re-arm detection path that reads
    the already-loaded issue labels.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    human_needed = config.labels.human_needed

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
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Merged PR #456 only *mentions* issue #123 in free text -- no
    # branch-prefix binding, no closing keyword -- so only the loose
    # mention scan finds it.
    fake_gh.prs[0]["state"] = "MERGED"
    fake_gh.prs[0]["headRefName"] = "cleanup-unrelated-branch"
    fake_gh.prs[0]["title"] = "chore: unrelated cleanup"
    fake_gh.prs[0]["body"] = "While in the area, this also happens to fix issue #123."
    assert fake_gh.prs[0]["isCrossRepository"] is False

    # --- Pass 1: never-flagged mention-only -> flagged, excluded, not dispatched
    result1 = app.dispatch(limit=1)
    assert result1.ok is True
    assert result1.data["merged_pr_flagged_issue_numbers"] == [123]
    assert result1.data["merged_pr_mention_rearmed_issue_numbers"] == []
    assert result1.data["selected_count"] == 0  # mention-only exclusion applies
    assert (123, human_needed) in fake_gh.labels_added
    state1 = load_state(paths.state_file)
    assert state1["issues"]["123"].get("merged_pr_mention_flagged_at") is not None
    assert state1["issues"]["123"].get("mention_rearmed_at") is None

    # --- Operator re-arm: remove agent:human-needed (deliberate ruling).
    assert fake_gh.remove_issue_label(123, human_needed)

    # --- Pass 2: re-arm detected -> exclusion lifts, issue dispatched, no re-flag
    result2 = app.dispatch(limit=1)
    assert result2.ok is True
    assert result2.data["merged_pr_flagged_issue_numbers"] == []  # one-shot preserved
    assert result2.data["merged_pr_mention_rearmed_issue_numbers"] == [123]
    assert result2.data["selected_count"] == 1  # re-entered dispatch candidates
    assert 123 in [s["issue_number"] for s in result2.data["sessions"]]
    state2 = load_state(paths.state_file)
    assert state2["issues"]["123"].get("mention_rearmed_at") is not None
    # The durable re-arm marker is recorded as an event.
    rearmed_events = [
        e
        for e in state2.get("events", [])
        if e.get("kind") == "dispatch_merged_pr_mention_rearmed"
    ]
    assert len(rearmed_events) == 1
    assert rearmed_events[0]["payload"]["issue_numbers"] == [123]

    # --- Pass 3: durable marker persists -> exclusion stays lifted. The issue
    # is now dispatched (active labels), so it is not re-selected; the point is
    # that the mention-only exclusion does NOT re-assert and block recovery.
    result3 = app.dispatch(limit=1)
    assert result3.ok is True
    state3 = load_state(paths.state_file)
    assert state3["issues"]["123"].get("mention_rearmed_at") is not None
    # No second re-arm event -- the durable marker suppresses re-detection.
    rearmed_events_3 = [
        e
        for e in state3.get("events", [])
        if e.get("kind") == "dispatch_merged_pr_mention_rearmed"
    ]
    assert len(rearmed_events_3) == 1


def test_dispatch_merged_pr_mention_still_carries_human_needed_stays_excluded(
    tmp_path: Path,
) -> None:
    """Issue #1336 acceptance criterion 2: a mention-only issue that was flagged
    and still carries agent:human-needed must remain excluded -- the operator
    has not re-armed, so the safe default from #564 is preserved. A regression
    that lifts the exclusion purely on the flag timestamp (without checking the
    label / durable re-arm marker) makes this test fail.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

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

    fake_gh = LabelMutatingFakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    fake_gh.prs[0]["state"] = "MERGED"
    fake_gh.prs[0]["headRefName"] = "cleanup-unrelated-branch"
    fake_gh.prs[0]["title"] = "chore: unrelated cleanup"
    fake_gh.prs[0]["body"] = "While in the area, this also happens to fix issue #123."

    # Pass 1: flagged, human_needed applied (and retained on the issue).
    result1 = app.dispatch(limit=1)
    assert result1.data["merged_pr_flagged_issue_numbers"] == [123]
    assert result1.data["selected_count"] == 0

    # Pass 2: operator did NOT remove human_needed -> still excluded.
    result2 = app.dispatch(limit=1)
    assert result2.ok is True
    assert result2.data["merged_pr_flagged_issue_numbers"] == []  # one-shot
    assert result2.data["merged_pr_mention_rearmed_issue_numbers"] == []
    assert result2.data["selected_count"] == 0  # still excluded
    state2 = load_state(paths.state_file)
    assert state2["issues"]["123"].get("mention_rearmed_at") is None


def test_dispatch_merged_pr_mention_never_flagged_stays_excluded(tmp_path: Path) -> None:
    """Issue #1336 acceptance criterion 2 (never-reviewed half): a
    mention-only issue that has never been flagged must remain excluded this
    pass while it is being flagged for the first time -- the re-arm lift only
    applies to issues flagged in a PRIOR pass. A regression that lifts the
    exclusion for any issue with the flag timestamp set this pass (before the
    operator has had a chance to rule) makes this test fail.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    fake_gh.prs[0]["state"] = "MERGED"
    fake_gh.prs[0]["headRefName"] = "cleanup-unrelated-branch"
    fake_gh.prs[0]["title"] = "chore: unrelated cleanup"
    fake_gh.prs[0]["body"] = "While in the area, this also happens to fix issue #123."

    result = app.dispatch(limit=1)
    assert result.ok is True
    # First-time flag: excluded this pass (flagged, not dispatched), no re-arm.
    assert result.data["merged_pr_flagged_issue_numbers"] == [123]
    assert result.data["merged_pr_mention_rearmed_issue_numbers"] == []
    assert result.data["selected_count"] == 0
    state = load_state(paths.state_file)
    assert state["issues"]["123"].get("mention_rearmed_at") is None


def test_dispatch_defers_when_merged_pr_list_raises_on_direct_fallback(
    tmp_path: Path,
) -> None:
    """dispatch() must defer — not crash — when merged_pr_list() raises on the
    _resolve_merged_prs direct-fallback path (#633 review finding).

    This is the COMMON-case path: open ready issues exist but no closed-ready
    issues this pass, so _finalize_externally_merged_issues skips the
    merged_pr_list() fetch entirely (returns an outcome with called=False).
    _resolve_merged_prs then calls merged_pr_list() directly (branch 1), and
    after #633 that call raises GitHubError on an unusable response instead of
    silently returning []. Without a handler the error propagates out of
    dispatch() and crashes the supervised loop daemon on a transient gh
    failure. dispatch()'s ``except GitHubError`` handler must catch it and
    return a deferred result so the next pass retries.
    """

    class FakeGitHubFailingMergedList(FakeGitHub):
        def merged_pr_list(self):
            raise github_module.GitHubError("merged_pr_list: unusable response")

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubFailingMergedList()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["deferred_reason"] == "github_error"
    assert "unusable response" in result.data["github_error"]


def test_dispatch_defers_when_merged_pr_list_raises_on_finalize_reraise(
    tmp_path: Path,
) -> None:
    """dispatch() must defer when _resolve_merged_prs re-raises a finalize error.

    When closed-ready issues exist, _finalize_externally_merged_issues calls
    merged_pr_list() itself and records a failure in _MergedPRListOutcome
    (called=True, error=GitHubError). _resolve_merged_prs branch 2 re-raises
    that stored error. dispatch()'s ``except GitHubError`` handler must catch
    the re-raise and defer — the same handler that covers the direct fallback
    (branch 1) — so a transient gh failure during finalization does not crash
    the supervised loop daemon.
    """

    class FakeGitHubFailingMergedList(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            # A closed ready-labeled issue forces _finalize_externally_merged_issues
            # to actually call merged_pr_list() (rather than skip with called=False),
            # so the error is recorded in the outcome. The OPEN ready issue stays
            # in the queue after finalization strips the closed one, so ``issues``
            # is non-empty in _dispatch_impl and _resolve_merged_prs branch 2
            # (``if outcome.error is not None and issues: raise``) fires.
            self.issues = [
                {
                    "number": 200,
                    "title": "Closed ready issue",
                    "url": "https://example.test/issues/200",
                    "body": "Already done",
                    "labels": [{"name": "automated-ready"}],
                    "state": "CLOSED",
                },
                {
                    "number": 201,
                    "title": "Open ready issue",
                    "url": "https://example.test/issues/201",
                    "body": "Needs work",
                    "labels": [{"name": "automated-ready"}],
                    "state": "OPEN",
                },
            ]

        def merged_pr_list(self):
            raise github_module.GitHubError("merged_pr_list: unusable response")

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubFailingMergedList()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["deferred_reason"] == "github_error"
    assert "unusable response" in result.data["github_error"]
