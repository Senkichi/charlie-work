"""Issue #2917: a ready ``priority:critical`` issue is not queued behind the rework backlog.

``loop()`` runs rework first and gives fresh dispatch what is left of the wave.
``critical_fresh_reserve`` is the one comparison between the lanes: the rework
lane is handed ``wave - reserve``. The first three tests drive the real
``loop()`` (the issue's acceptance cases); the rest pin each bound and fallback
of the reserve itself.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import _dispatch_rework_config
from charlie_work.config import DispatchConfig, LabelConfig, ReconcilePassConfig
from charlie_work.github import GitHubError
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.priority_reserve import critical_fresh_reserve
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

REWORK_ISSUES = (123, 124, 125)
FRESH_ISSUE = 200
EVENT_KIND = "dispatch_priority_reserve"


def _config(*, prefix: str = "priority:", limit: int = 2, max_open_prs: int = 0) -> Any:
    base = _dispatch_rework_config()
    return replace(
        base,
        dispatch=DispatchConfig(
            default_limit=limit,
            max_concurrent_sessions=limit,
            max_open_agent_prs=max_open_prs,
            # The host-load brake reads live pytest processes; keep it out of slot arithmetic.
            host_load_max_pytest_processes=0,
            host_load_max_pytest_trees=0,
        ),
        labels=LabelConfig(priority_prefix=prefix),
        reconcile_pass=ReconcilePassConfig(enabled=False),
    )


def _ready(number: int, *labels: str, body: str = "work") -> dict[str, Any]:
    return {
        "number": number,
        "title": f"issue {number}",
        "url": f"https://example.test/issues/{number}",
        "body": body,
        "labels": [{"name": "automated-ready"}, *({"name": label} for label in labels)],
        "state": "OPEN",
    }


class _Backlog(FakeGitHub):
    """Three issues awaiting rework (each with an open PR) plus the given ready issues."""

    def __init__(self, *ready: dict[str, Any]) -> None:
        super().__init__()
        rework = [
            {
                "number": number,
                "title": f"rework {number}",
                "url": f"https://example.test/issues/{number}",
                "body": "work",
                "labels": [{"name": "agent:needs-rework"}],
                "state": "OPEN",
            }
            for number in REWORK_ISSUES
        ]
        self.issues = [*rework, *ready]
        self.prs = [
            {
                "number": 400 + number,
                "title": f"PR for issue {number}",
                "url": f"https://example.test/pr/{400 + number}",
                "headRefOid": f"sha{number}",
                "isCrossRepository": False,
                "headRefName": f"agent/issue-{number}",
                "state": "OPEN",
            }
            for number in REWORK_ISSUES
        ]


def _app(tmp_path: Path, gh: FakeGitHub, config: Any, *, dry_run: bool = False) -> OrchestratorApp:
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        for number in REWORK_ISSUES:
            state["issues"][str(number)] = {
                "number": number,
                "title": f"rework {number}",
                "url": f"https://example.test/issues/{number}",
                "status": "rework_requested",
            }
        save_state(paths.state_file, state)
    for number in REWORK_ISSUES:
        prompt_dir = paths.prs / f"pr-{400 + number}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "rework-prompt.md").write_text("Fix the issues", encoding="utf-8")
    return OrchestratorApp(tmp_path, paths, config, gh, dry_run=dry_run)


def _launched(result: Any, lane: str) -> list[int]:
    return sorted(session["issue_number"] for session in result.data[lane]["sessions"])


def _reserve_events(app: OrchestratorApp) -> list[dict[str, Any]]:
    return query_events(app.paths.state_file, kind=EVENT_KIND)


# --- the issue's acceptance cases, through the real loop() --------------------


def test_a_critical_issue_gets_a_slot_ahead_of_the_rework_queue(tmp_path: Path) -> None:
    app = _app(tmp_path, _Backlog(_ready(FRESH_ISSUE, "priority:critical")), _config())
    result = app.loop(limit=2)
    assert result.ok, result.message
    assert _launched(result, "dispatch") == [FRESH_ISSUE]
    assert len(_launched(result, "dispatch_rework")) == 1


def test_a_normal_priority_issue_leaves_rework_first(tmp_path: Path) -> None:
    app = _app(tmp_path, _Backlog(_ready(FRESH_ISSUE, "priority:normal")), _config())
    result = app.loop(limit=2)
    assert result.ok, result.message
    assert len(_launched(result, "dispatch_rework")) == 2
    assert _launched(result, "dispatch") == []
    assert _reserve_events(app) == []


def test_an_empty_priority_prefix_matches_the_behaviour_before_the_reserve(
    tmp_path: Path,
) -> None:
    gh = _Backlog(_ready(FRESH_ISSUE, "priority:critical"))
    app = _app(tmp_path, gh, _config(prefix=""))
    result = app.loop(limit=2)
    assert result.ok, result.message
    assert len(_launched(result, "dispatch_rework")) == 2
    assert _launched(result, "dispatch") == []
    assert _reserve_events(app) == []


# --- the reserve ---------------------------------------------------------------


def test_the_reserve_is_one_slot_per_ready_critical_issue_and_is_recorded(tmp_path: Path) -> None:
    gh = _Backlog(_ready(FRESH_ISSUE, "priority:critical"), _ready(201, "priority:high"))
    app = _app(tmp_path, gh, _config())
    assert critical_fresh_reserve(app, 2) == 1
    (event,) = _reserve_events(app)
    assert event["payload"] == {
        "issue_numbers": [FRESH_ISSUE],
        "reserved": 1,
        "wave_limit": 2,
        "fresh_headroom": 2,
    }


def test_the_reserve_never_exceeds_the_wave(tmp_path: Path) -> None:
    gh = _Backlog(*(_ready(n, "priority:critical") for n in (200, 201, 202)))
    app = _app(tmp_path, gh, _config())
    assert critical_fresh_reserve(app, 2) == 2


def test_two_critical_issues_take_the_whole_wave_from_rework(tmp_path: Path) -> None:
    gh = _Backlog(_ready(200, "priority:critical"), _ready(201, "priority:critical"))
    app = _app(tmp_path, gh, _config())
    result = app.loop(limit=2)
    assert result.ok, result.message
    assert _launched(result, "dispatch") == [200, 201]
    # The rework lane still ran its pass; it launched nothing and reported why.
    rework = result.data["dispatch_rework"]
    assert rework["selected_count"] == 0
    assert rework["deferred_by_concurrency"] == list(REWORK_ISSUES)


@pytest.mark.parametrize(
    "limit",
    [0, -1],
)
def test_an_empty_wave_reserves_nothing_and_reads_nothing(tmp_path: Path, limit: int) -> None:
    class _NoReads(_Backlog):
        def issue_list(self, labels=None, state=None):  # noqa: ARG002
            raise AssertionError("an empty wave must not read issues")

    app = _app(tmp_path, _NoReads(_ready(FRESH_ISSUE, "priority:critical")), _config())
    assert critical_fresh_reserve(app, limit) == 0


def test_an_empty_prefix_reads_nothing(tmp_path: Path) -> None:
    class _NoReads(_Backlog):
        def issue_list(self, labels=None, state=None):  # noqa: ARG002
            raise AssertionError("an empty prefix must not read issues")

    app = _app(tmp_path, _NoReads(_ready(FRESH_ISSUE, "priority:critical")), _config(prefix=""))
    assert critical_fresh_reserve(app, 2) == 0


def test_a_blocked_critical_issue_reserves_nothing(tmp_path: Path) -> None:
    blocker = _ready(300)
    blocked = _ready(FRESH_ISSUE, "priority:critical", body="Blocked by #300")
    app = _app(tmp_path, _Backlog(blocker, blocked), _config())
    assert critical_fresh_reserve(app, 2) == 0
    assert _reserve_events(app) == []


def test_an_active_critical_issue_reserves_nothing(tmp_path: Path) -> None:
    issue = _ready(FRESH_ISSUE, "priority:critical", "agent:in-progress")
    app = _app(tmp_path, _Backlog(issue), _config())
    assert critical_fresh_reserve(app, 2) == 0


def test_an_operator_claimed_critical_issue_reserves_nothing(tmp_path: Path) -> None:
    app = _app(tmp_path, _Backlog(_ready(FRESH_ISSUE, "priority:critical")), _config())
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"][str(FRESH_ISSUE)] = {
            "number": FRESH_ISSUE,
            "operator_claimed_at": "claimed",
        }
        save_state(app.paths.state_file, state)
    assert critical_fresh_reserve(app, 2) == 0


def test_the_reserve_is_bounded_by_the_fresh_lanes_own_headroom(tmp_path: Path) -> None:
    """Open-PR backpressure binds fresh dispatch only; reserving past it would starve rework.

    The three rework PRs are open agent PRs, so ``max_open_agent_prs=3`` leaves
    fresh dispatch no headroom. Reserving a slot rework could never get back
    from fresh would pause the lane that drains those PRs.
    """
    gh = _Backlog(_ready(FRESH_ISSUE, "priority:critical"))
    app = _app(tmp_path, gh, _config(max_open_prs=3))
    assert critical_fresh_reserve(app, 2) == 0
    result = app.loop(limit=2)
    assert len(_launched(result, "dispatch_rework")) == 2


def test_a_github_failure_reserves_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    class _Unreadable(_Backlog):
        def issue_list(self, labels=None, state=None):  # noqa: ARG002
            raise GitHubError("boom")

    app = _app(tmp_path, _Unreadable(_ready(FRESH_ISSUE, "priority:critical")), _config())
    with caplog.at_level("WARNING", logger="charlie_work.priority_reserve"):
        assert critical_fresh_reserve(app, 2) == 0
    assert "priority reserve skipped" in caplog.text


def test_a_dry_run_reserves_the_same_slots_without_recording_an_event(tmp_path: Path) -> None:
    gh = _Backlog(_ready(FRESH_ISSUE, "priority:critical"))
    app = _app(tmp_path, gh, _config(), dry_run=True)
    assert critical_fresh_reserve(app, 2) == 1
    assert _reserve_events(app) == []
