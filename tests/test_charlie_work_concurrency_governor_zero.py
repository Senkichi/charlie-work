"""Concurrency governor clamped-to-zero reporting: distinguishable from empty backlog.

Split out of ``tests/test_charlie_work.py`` (issue #1548, Track-1 wave 2/8):
the ``test_concurrency_governor_zero_*`` seam -- a pass clamped to zero
dispatches must be distinguishable, from CommandResult.data and events alone,
from a pass with no candidates. Shared fakes and helpers in
``tests/_dispatch_fixtures.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from _fakes_github import FakeGitHub
from charlie_work.config import (
    DevinConfig,
    DispatchConfig,
    FleetConfig,
    OrchestratorConfig,
    WorkerRoleConfig,
)
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    state_lock,
)
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_concurrency_governor_zero_rework_is_self_explaining(tmp_path: Path, monkeypatch) -> None:
    """Issue #1014: a dispatch_rework pass clamped by the concurrency governor
    must be distinguishable, from CommandResult.data and events.db alone, from
    a pass with an empty rework backlog.

    Before the fix: (a) the automatic (non-``--issues``) selection path in
    ``_dispatch_rework_impl`` unconditionally reported
    ``deferred_by_concurrency=[]`` even though candidates existed and were
    dropped by the clamp, and (b) the governor's own numbers
    (``report_fields()``) were merged into ``CommandResult.data`` but never
    into the persisted ``dispatch_rework`` event -- so ``events.db`` carried
    no capacity axis at all. Both defects are the same shape as #1005's
    fresh-dispatch fix, but in a structurally separate function that #1005
    did not touch.

    Four scenarios, each pinning a distinct acceptance criterion:

    1. Repo-cap clamp with 8 candidates, clamped to 0 -- proves the automatic
       path populates ``deferred_by_concurrency`` (not ``[]``), truncation
       (``_MAX_DEFERRED_CONCURRENCY_EXAMPLES=5``) engages, and the bounded
       ``failures`` map covers all 8 deferred issues (fed the FULL list, not
       the truncated one).
    2. The *same* clamped governor config against a genuinely empty backlog --
       proves two passes that are BOTH ``clamped=True`` are still
       distinguishable via ``deferred_by_concurrency_count``.
    3. Automatic path, partial clamp to 2 out of 8 -- 2 dispatched, 6 deferred;
       the persisted ``dispatch_rework`` event carries the
       ``concurrency_governor`` block and ``deferred_by_concurrency_count``
       (the early-return paths in scenarios 1/2 don't record an event, so this
       scenario is the one that exercises the event payload).
    4. The ``--issues`` path with 7 explicitly-requested issues, all deferred --
       proves ``_build_failure_map`` receives the FULL deferred list (all 7
       keys in ``failures``), not the truncated 5-item list.
    """

    def mock_count_live_one(sessions_dir, state_file=None):
        return 1

    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", mock_count_live_one)

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=1, default_limit=5),
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkSaturatedGitHub(FakeGitHub):
        """FakeGitHub with N rework issues, each with a matching open PR."""

        def __init__(self, count: int, base_number: int = 301) -> None:
            super().__init__()
            self.issues = [
                {
                    "number": base_number + i,
                    "title": f"Rework {i}",
                    "url": f"https://example.test/issues/{base_number + i}",
                    "body": "rework issue",
                    "labels": [{"name": "agent:needs-rework"}],
                    "state": "OPEN",
                }
                for i in range(count)
            ]
            self.prs = [
                {
                    "number": 500 + i,
                    "title": f"Fix #{base_number + i}",
                    "url": f"https://example.test/pull/{500 + i}",
                    "headRefName": f"agent/issue-{base_number + i}",
                    "baseRefName": "main",
                    "headRefOid": f"sha-{base_number + i}",
                    "mergeStateStatus": "CLEAN",
                    "body": f"Closes #{base_number + i}",
                    "labels": [],
                    "isCrossRepository": False,
                    "state": "OPEN",
                }
                for i in range(count)
            ]

    def _seed_rework_state(state_file, numbers):
        """Seed state.json with rework_requested status for the given issues."""
        from charlie_work.state import save_state

        save_state(
            state_file,
            {
                "issues": {str(n): {"status": "rework_requested"} for n in numbers},
                "prs": {},
                "events": [],
                "generated_at": "2024-01-01T00:00:00Z",
            },
        )

    def _create_rework_prompts(tmp, pr_numbers):
        """Create rework-prompt.md for each PR directory."""
        for pr_num in pr_numbers:
            pr_dir = tmp / ".var" / "charlie-work" / "prs" / f"pr-{pr_num}"
            pr_dir.mkdir(parents=True, exist_ok=True)
            (pr_dir / "rework-prompt.md").write_text("Fix the issues", encoding="utf-8")

    # --- Scenario 1: repo-cap clamp, 8 candidates, clamped to 0 ------------
    # Governor: max_concurrent_sessions=1, one session already live -> 0
    # available slots. dispatch_limit is clamped to 0 even though 8 issues
    # are rework-requested.
    numbers_8 = list(range(301, 309))
    pr_numbers_8 = list(range(500, 508))
    fake_gh = ReworkSaturatedGitHub(8)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _seed_rework_state(paths.state_file, numbers_8)

    result = app.dispatch_rework()

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["concurrency_limit"] == 1
    assert result.data["live_session_count"] == 1
    assert result.data["available_slots"] == 0

    # Defect (a): the automatic path must populate deferred_by_concurrency,
    # not report it as unconditionally empty. Truncated to the first 5 by
    # number for the persisted/data field, with the untruncated total carried
    # separately.
    assert result.data["deferred_by_concurrency"] == [301, 302, 303, 304, 305]
    assert result.data["deferred_by_concurrency_count"] == 8
    # The failures map must cover every deferred issue (all 8), not just the
    # 5 that made it into the truncated display field -- _build_failure_map
    # is fed the FULL deferred list.
    assert sorted(result.data["failures"].keys()) == [
        301,
        302,
        303,
        304,
        305,
        306,
        307,
        308,
    ]

    # The clamped-to-0 early return ("no rework candidates found") does not
    # record a dispatch_rework event -- only the dispatch-proceeding paths do.
    # So there is no event payload to check here; scenario 3 exercises the
    # event payload via a partial clamp that proceeds to dispatch.
    events = query_events(paths.state_file, kind="dispatch_rework")
    assert len(events) == 0

    # --- Scenario 2: same clamped config, genuinely empty backlog ----------
    # Machine-checkable differentiation: two passes that are BOTH clamped must
    # still be distinguishable from CommandResult.data alone.
    # deferred_by_concurrency_count is the field that separates "nothing to
    # dispatch" from "nothing COULD be dispatched".
    empty_gh = ReworkSaturatedGitHub(0)
    empty_app = OrchestratorApp(tmp_path, paths, config, empty_gh)
    _seed_rework_state(paths.state_file, [])
    empty_result = empty_app.dispatch_rework()
    assert empty_result.data["selected_count"] == 0
    assert empty_result.data["deferred_by_concurrency_count"] == 0
    assert empty_result.data["deferred_by_concurrency"] == []
    assert (
        empty_result.data["deferred_by_concurrency_count"]
        != result.data["deferred_by_concurrency_count"]
    )

    # --- Scenario 3: automatic path, partial clamp to 2 out of 8 ----------
    # 8 candidates, governor allows 2 slots (max_concurrent=2, 0 live). The
    # code proceeds past the early returns, dispatches 2 workers, and records
    # a dispatch_rework event -- the event payload is what this scenario
    # exercises (the early-return paths in scenarios 1/2 don't record events).
    partial_config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=2, default_limit=5),
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    partial_paths = runtime_paths(tmp_path / "partial", partial_config.runtime.state_dir)
    partial_gh = ReworkSaturatedGitHub(8)
    partial_app = OrchestratorApp(tmp_path / "partial", partial_paths, partial_config, partial_gh)
    _seed_rework_state(partial_paths.state_file, numbers_8)
    _create_rework_prompts(tmp_path / "partial", pr_numbers_8)

    monkeypatch.setattr(
        "charlie_work.workflow._count_live_sessions",
        lambda sessions_dir, state_file=None: 0,
    )
    partial_result = partial_app.dispatch_rework()

    assert partial_result.ok is True
    assert partial_result.data["selected_count"] == 2
    assert partial_result.data["concurrency_limit"] == 2
    assert partial_result.data["live_session_count"] == 0
    assert partial_result.data["available_slots"] == 2
    # 6 deferred (8 candidates - 2 selected), truncated to 5 for display.
    assert partial_result.data["deferred_by_concurrency"] == [303, 304, 305, 306, 307]
    assert partial_result.data["deferred_by_concurrency_count"] == 6
    # _build_failure_map sees the full 6-item deferred list.
    assert sorted(partial_result.data["failures"].keys()) == [
        303,
        304,
        305,
        306,
        307,
        308,
    ]

    # Defect (b): the persisted dispatch_rework event must carry the
    # governor's decision, not just the transient CommandResult.data.
    partial_events = query_events(partial_paths.state_file, kind="dispatch_rework")
    assert len(partial_events) == 1
    partial_payload = partial_events[0]["payload"]
    assert partial_payload["deferred_by_concurrency"] == [303, 304, 305, 306, 307]
    assert partial_payload["deferred_by_concurrency_count"] == 6
    assert partial_payload["concurrency_governor"] == {
        "clamped": True,
        "dispatch_limit": 2,
        "concurrency_limit": 2,
        "live_session_count": 0,
        "available_slots": 2,
    }

    # --- Scenario 4: --issues path, 7 deferred (full-vs-truncated pin) -----
    # _build_failure_map must see every deferred issue, not just the truncated
    # event examples. Mirrors scenario 4 of the #1005 fresh-dispatch test.
    issues_tmp_path = tmp_path / "issues_path"
    issues_paths = runtime_paths(issues_tmp_path, config.runtime.state_dir)
    issues_numbers_7 = list(range(401, 408))
    issues_pr_numbers_7 = list(range(600, 607))
    issues_gh = ReworkSaturatedGitHub(7, base_number=401)
    issues_app = OrchestratorApp(issues_tmp_path, issues_paths, config, issues_gh)
    _seed_rework_state(issues_paths.state_file, issues_numbers_7)
    _create_rework_prompts(issues_tmp_path, issues_pr_numbers_7)
    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", mock_count_live_one)

    issues_result = issues_app.dispatch_rework(only_issues="401,402,403,404,405,406,407")

    assert issues_result.ok is True
    assert issues_result.data["selected_count"] == 0
    assert issues_result.data["deferred_by_concurrency"] == [401, 402, 403, 404, 405]
    assert issues_result.data["deferred_by_concurrency_count"] == 7
    # All 7 deferred issues keep a failures entry -- not just the 5 that made
    # it into the truncated event/data field.
    assert sorted(issues_result.data["failures"].keys()) == [
        401,
        402,
        403,
        404,
        405,
        406,
        407,
    ]


def test_concurrency_governor_zero_dispatch_is_self_explaining_in_dispatch_event(
    tmp_path: Path, monkeypatch
) -> None:
    """Issue #1005: a dispatch pass clamped to zero by the concurrency governor
    must be distinguishable, from events.db alone, from a pass with an empty
    backlog.

    Before the fix: (a) the automatic (non-``--issues``) selection path
    unconditionally reported ``deferred_by_concurrency=[]`` even though
    candidates existed and were dropped by the clamp, and (b) the governor's
    own numbers (``report_fields()``) were merged into ``CommandResult.data``
    but never into the persisted ``dispatch`` event -- so ``events.db`` carried
    no capacity axis at all. Both defects are exercised here on the automatic
    path, which the existing ``--issues``-only governor tests do not cover.

    Four scenarios, each pinning a distinct acceptance criterion:

    1. Repo-cap clamp with 8 candidates -- proves truncation
       (``_MAX_DEFERRED_CONCURRENCY_EXAMPLES=5``) actually engages, and that
       the bounded ``failures`` map (built from the same truncated list) does
       not silently grow unbounded even though 8 issues were deferred.
    2. The *same* clamped governor config against a genuinely empty backlog --
       proves the forensic case that actually matters: two passes that are
       BOTH ``clamped=True`` are still distinguishable via
       ``deferred_by_concurrency_count``. (Clamped-vs-disabled would let
       ``clamped`` alone do the discriminating, which is a weaker claim.)
    3. A fleet-only-binding clamp -- repo governor recomputes
       ``available_slots=1`` (nonzero!) while the fleet cap independently
       zeroes ``dispatch_limit``. Confirms ``dispatch_limit`` -- not
       ``available_slots`` -- is the field that states the effective limit
       was zero, and that ``fleet_concurrency_limit``/``fleet_live_session_count``
       are present to identify which cap bound.
    4. The ``--issues`` path with 7 explicitly-requested issues, all
       deferred -- proves ``_build_failure_map`` receives the FULL deferred
       list (all 7 keys in ``failures``), not the truncated 5-item list that
       reaches the persisted event/``CommandResult.data`` fields. Pins a
       regression an earlier version of this fix introduced: truncating
       ``deferred_by_concurrency`` inside ``_select_dispatch_candidates``
       itself (rather than at each payload call site) silently dropped
       ``failures`` entries for the 6th+ deferred issue.
    """

    def mock_count_live_one(sessions_dir, state_file=None):
        return 1

    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", mock_count_live_one)

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=1, default_limit=5),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class SaturatedGitHub(FakeGitHub):
        def __init__(self, count: int) -> None:
            super().__init__()
            self.issues = [
                {
                    "number": 201 + i,
                    "title": f"Fix {i}",
                    "url": f"https://example.test/issues/{201 + i}",
                    "body": "backlog issue",
                    "labels": [{"name": "automated-ready"}],
                    "state": "OPEN",
                }
                for i in range(count)
            ]
            self.prs = []

    # --- Scenario 1: repo-cap clamp, 8 candidates, truncation engaged -----
    fake_gh = SaturatedGitHub(8)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Governor: max_concurrent_sessions=1, one session already live -> 0
    # available slots. dispatch_limit is clamped to 0 even though 8 issues
    # are dispatchable (mirrors the live incident: N dispatchable, 0
    # dispatched, fleet/repo at cap).
    result = app.dispatch()

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["concurrency_limit"] == 1
    assert result.data["live_session_count"] == 1
    assert result.data["available_slots"] == 0

    # Defect 2: the automatic path must populate deferred_by_concurrency, not
    # report it as unconditionally empty. Truncated to the first 5 by number
    # for the persisted/data field, with the untruncated total carried
    # separately.
    assert result.data["deferred_by_concurrency"] == [201, 202, 203, 204, 205]
    assert result.data["deferred_by_concurrency_count"] == 8
    # The failures map must cover every deferred issue (all 8), not just the
    # 5 that made it into the truncated display field -- _build_failure_map
    # is fed the FULL deferred list (a review-caught regression: an earlier
    # version of this fix truncated before _build_failure_map, silently
    # dropping failures entries for the 6th+ deferred issue). This is a
    # superset of _extract_attention_events' exclusion set (built from the
    # truncated deferred_by_concurrency event field), so no issue in the
    # truncated set is ever double-reported as a launch-failure "error" --
    # see scenario 4 for the dedicated full-vs-truncated pin.
    assert sorted(result.data["failures"].keys()) == [201, 202, 203, 204, 205, 206, 207, 208]

    # Defect 1: the persisted event -- not just the transient CommandResult.data
    # -- must carry the governor's decision.
    events = query_events(paths.state_file, kind="dispatch")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["deferred_by_concurrency"] == [201, 202, 203, 204, 205]
    assert payload["deferred_by_concurrency_count"] == 8
    assert payload["concurrency_governor"] == {
        "clamped": True,
        "dispatch_limit": 0,
        "concurrency_limit": 1,
        "live_session_count": 1,
        "available_slots": 0,
    }

    # --- Scenario 2: same clamped config, genuinely empty backlog ---------
    # Machine-checkable differentiation (acceptance criterion): two passes
    # that are BOTH clamped must still be distinguishable from events.db
    # alone. Reusing the identical governor config (not a disabled one) means
    # `clamped` is True in both payloads -- deferred_by_concurrency_count is
    # the field that actually separates "nothing to dispatch" from "nothing
    # COULD be dispatched", which is the realistic forensic question.
    empty_gh = SaturatedGitHub(0)
    empty_app = OrchestratorApp(tmp_path, paths, config, empty_gh)
    empty_result = empty_app.dispatch()
    assert empty_result.data["selected_count"] == 0
    empty_events = query_events(paths.state_file, kind="dispatch")
    empty_payload = empty_events[-1]["payload"]
    assert empty_payload["concurrency_governor"]["clamped"] is True
    assert empty_payload["deferred_by_concurrency_count"] == 0
    assert empty_payload["deferred_by_concurrency"] == []
    assert empty_payload["concurrency_governor"] == payload["concurrency_governor"]
    assert (
        empty_payload["deferred_by_concurrency_count"] != payload["deferred_by_concurrency_count"]
    )

    # --- Scenario 3: fleet-only-binding clamp ------------------------------
    # Repo governor recomputes available_slots=1 (nonzero: 2 - 1 live), but
    # the fleet cap independently saturates and drives dispatch_limit to 0.
    # available_slots alone would misleadingly suggest a slot was open.
    def mock_count_live_one_of_two(sessions_dir, state_file=None):
        return 1

    def mock_count_fleet_live_saturated(fleet_dir_override):
        return 3, []

    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", mock_count_live_one_of_two)
    monkeypatch.setattr(
        "charlie_work.workflow.count_fleet_live_sessions", mock_count_fleet_live_saturated
    )

    fleet_config = OrchestratorConfig(
        fleet=FleetConfig(global_max_concurrent_sessions=3),
        dispatch=DispatchConfig(max_concurrent_sessions=2, default_limit=5),
        devin=DevinConfig(),
    )
    fleet_gh = SaturatedGitHub(3)
    fleet_app = OrchestratorApp(
        tmp_path,
        paths,
        fleet_config,
        fleet_gh,
        fleet_dir_override=str(tmp_path / "fleet"),
    )
    fleet_result = fleet_app.dispatch()

    assert fleet_result.ok is True
    assert fleet_result.data["selected_count"] == 0
    assert fleet_result.data["available_slots"] == 1
    assert fleet_result.data["fleet_concurrency_limit"] == 3
    assert fleet_result.data["fleet_live_session_count"] == 3

    fleet_events = query_events(paths.state_file, kind="dispatch")
    fleet_payload = fleet_events[-1]["payload"]
    assert fleet_payload["concurrency_governor"] == {
        "clamped": True,
        "dispatch_limit": 0,
        "concurrency_limit": 2,
        "live_session_count": 1,
        "available_slots": 1,
        "fleet_concurrency_limit": 3,
        "fleet_live_session_count": 3,
    }

    # --- Scenario 4: --issues path, >5 deferred (review regression pin) ---
    # _build_failure_map must see every deferred issue, not just the
    # truncated event examples. An earlier version of this fix truncated
    # deferred_by_concurrency inside _select_dispatch_candidates before
    # returning it, so `failures` silently lost entries for the 6th+
    # deferred issue -- a regression specifically on the --issues path,
    # which passed the complete (untruncated) list on origin/main before
    # #1005 touched this function at all. Caught in review before merge.
    issues_tmp_path = tmp_path / "issues_path"
    issues_paths = runtime_paths(issues_tmp_path, config.runtime.state_dir)
    issues_gh = SaturatedGitHub(7)
    issues_app = OrchestratorApp(issues_tmp_path, issues_paths, config, issues_gh)
    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", mock_count_live_one)

    issues_result = issues_app.dispatch(only_issues="201,202,203,204,205,206,207")

    assert issues_result.ok is True
    assert issues_result.data["selected_count"] == 0
    assert issues_result.data["deferred_by_concurrency"] == [201, 202, 203, 204, 205]
    assert issues_result.data["deferred_by_concurrency_count"] == 7
    # All 7 deferred issues keep a failures entry -- not just the 5 that made
    # it into the truncated event/data field.
    assert sorted(issues_result.data["failures"].keys()) == [
        201,
        202,
        203,
        204,
        205,
        206,
        207,
    ]

    issues_events = query_events(issues_paths.state_file, kind="dispatch")
    issues_payload = issues_events[-1]["payload"]
    assert issues_payload["deferred_by_concurrency"] == [201, 202, 203, 204, 205]
    assert issues_payload["deferred_by_concurrency_count"] == 7


def test_concurrency_governor_zero_rework_dry_run_automatic_path(
    tmp_path: Path, monkeypatch
) -> None:
    """Issue #1014: the dry-run (``dry_run=True``) branch of
    ``_dispatch_rework_impl`` must populate ``deferred_by_concurrency`` and
    ``deferred_by_concurrency_count`` on the automatic (non-``--issues``)
    selection path under a concurrency-saturated governor.

    The live-path fix is covered by
    ``test_concurrency_governor_zero_rework_is_self_explaining``; this test
    pins the *same* defect in the structurally separate dry-run branch, which
    no prior test exercised with a saturated governor at all. Before the fix
    the dry-run automatic path unconditionally reported
    ``deferred_by_concurrency=[]`` even when candidates existed and were
    dropped by the clamp -- making a saturated governor indistinguishable
    from an empty backlog in the planning report.

    Two scenarios:

    1. Repo-cap clamp with 8 candidates, clamped to 0 -- the dry-run report
       must carry ``deferred_by_concurrency`` (truncated to 5), the full
       ``deferred_by_concurrency_count`` of 8, and a ``failures`` map covering
       all 8 deferred issues (fed the FULL list, not the truncated one). No
       ``dispatch_rework`` event is recorded (dry-run skips all state writes).
    2. The ``--issues`` dry-run path with 7 explicitly-requested issues, all
       deferred -- ``_build_failure_map`` must see every deferred issue (all 7
       keys), not just the truncated 5-item display list.
    """

    def mock_count_live_one(sessions_dir, state_file=None):
        return 1

    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", mock_count_live_one)

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=1, default_limit=5),
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkSaturatedGitHub(FakeGitHub):
        """FakeGitHub with N rework issues, each with a matching open PR."""

        def __init__(self, count: int, base_number: int = 301) -> None:
            super().__init__()
            self.issues = [
                {
                    "number": base_number + i,
                    "title": f"Rework {i}",
                    "url": f"https://example.test/issues/{base_number + i}",
                    "body": "rework issue",
                    "labels": [{"name": "agent:needs-rework"}],
                    "state": "OPEN",
                }
                for i in range(count)
            ]
            self.prs = [
                {
                    "number": 500 + i,
                    "title": f"Fix #{base_number + i}",
                    "url": f"https://example.test/pull/{500 + i}",
                    "headRefName": f"agent/issue-{base_number + i}",
                    "baseRefName": "main",
                    "headRefOid": f"sha-{base_number + i}",
                    "mergeStateStatus": "CLEAN",
                    "body": f"Closes #{base_number + i}",
                    "labels": [],
                    "isCrossRepository": False,
                    "state": "OPEN",
                }
                for i in range(count)
            ]

    def _seed_rework_state(state_file, numbers):
        from charlie_work.state import save_state

        save_state(
            state_file,
            {
                "issues": {str(n): {"status": "rework_requested"} for n in numbers},
                "prs": {},
                "events": [],
                "generated_at": "2024-01-01T00:00:00Z",
            },
        )

    # --- Scenario 1: automatic path, repo-cap clamp to 0, dry-run ---------
    # Governor: max_concurrent_sessions=1, one session already live -> 0
    # available slots. dispatch_limit is clamped to 0 even though 8 issues
    # are rework-requested. The dry-run report must still surface the 8
    # deferred issues rather than reporting [].
    numbers_8 = list(range(301, 309))
    fake_gh = ReworkSaturatedGitHub(8)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)
    _seed_rework_state(paths.state_file, numbers_8)

    result = app.dispatch_rework()

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["concurrency_limit"] == 1
    assert result.data["live_session_count"] == 1
    assert result.data["available_slots"] == 0
    # The automatic dry-run path must populate deferred_by_concurrency, not
    # report it as unconditionally empty. Truncated to the first 5 for the
    # data field, with the untruncated total carried separately.
    assert result.data["deferred_by_concurrency"] == [301, 302, 303, 304, 305]
    assert result.data["deferred_by_concurrency_count"] == 8
    # The failures map must cover every deferred issue (all 8), not just the
    # 5 that made it into the truncated display field -- _build_failure_map
    # is fed the FULL deferred list.
    assert sorted(result.data["failures"].keys()) == [
        301,
        302,
        303,
        304,
        305,
        306,
        307,
        308,
    ]
    # Dry-run skips all state writes, so no dispatch_rework event is recorded.
    events = query_events(paths.state_file, kind="dispatch_rework")
    assert len(events) == 0
    # State itself is untouched by the dry-run pass.
    with state_lock(paths.state_file):
        post_state = load_state(paths.state_file)
    assert all(
        entry.get("status") == "rework_requested"
        for entry in post_state.get("issues", {}).values()
        if isinstance(entry, dict)
    )

    # --- Scenario 2: --issues dry-run path, 7 deferred (full-vs-truncated) -
    # _build_failure_map must see every deferred issue, not just the truncated
    # event examples. Mirrors scenario 4 of the live-path test.
    issues_tmp_path = tmp_path / "issues_dry"
    issues_paths = runtime_paths(issues_tmp_path, config.runtime.state_dir)
    issues_numbers_7 = list(range(401, 408))
    issues_gh = ReworkSaturatedGitHub(7, base_number=401)
    issues_app = OrchestratorApp(issues_tmp_path, issues_paths, config, issues_gh, dry_run=True)
    _seed_rework_state(issues_paths.state_file, issues_numbers_7)

    issues_result = issues_app.dispatch_rework(only_issues="401,402,403,404,405,406,407")

    assert issues_result.ok is True
    assert issues_result.data["selected_count"] == 0
    assert issues_result.data["deferred_by_concurrency"] == [401, 402, 403, 404, 405]
    assert issues_result.data["deferred_by_concurrency_count"] == 7
    # All 7 deferred issues keep a failures entry -- not just the 5 that made
    # it into the truncated data field.
    assert sorted(issues_result.data["failures"].keys()) == [
        401,
        402,
        403,
        404,
        405,
        406,
        407,
    ]
    # Dry-run records no event.
    issues_events = query_events(issues_paths.state_file, kind="dispatch_rework")
    assert len(issues_events) == 0
