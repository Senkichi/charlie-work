"""Tests for issue #1682: the #1110 dependency-blocked exemption must be
bounded by root-blocker progress.

When the entire ready backlog is dependency-blocked on a single root issue
and that root is making no progress, ``check_dispatch_staleness`` used to
return ``stale=False`` (``all_ready_blocked_by_dependencies``) unconditionally
-- a permanently-stuck root blocker was indistinguishable from a
legitimately-sequenced cohort tail, which is how 11 days of 0 merges / 0
closes produced no finding (root issue #1538 -> PR #1595 stuck in
``janitor_blocked``).

The exemption is now conditional: ``classify_backlog_reachability`` emits
``dependency_root_blockers`` (the distinct open blockers of every
``blocked_by_open_dependency`` issue that are not themselves
dependency-blocked, each carrying its GitHub ``updatedAt``), and
``check_dispatch_staleness`` fires ``dependency_root_blocker_idle`` when every
root's last recorded state/event change -- the newest of its GitHub
``updatedAt`` and any past-dated ``*_at``/``*_since`` timestamp on its
``state["issues"]``/linked-``state["prs"]`` entries -- is older than
``dispatch.dependency_stall_minutes``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github import FakeGitHub
from charlie_work.backlog_reachability import classify_backlog_reachability
from charlie_work.ci_findings import check_dispatch_staleness
from charlie_work.config import ConfigError, DispatchConfig, OrchestratorConfig, load_config
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.state import arm_dispatch_stale_alert, load_state, save_state
from charlie_work.workflow import OrchestratorApp


def _iso(ts: datetime) -> str:
    return ts.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _issue(
    number: int,
    names: set[str],
    *,
    body: str = "",
    updated_at: str | None = None,
) -> dict[str, Any]:
    return {
        "number": number,
        "title": f"Issue {number}",
        "url": f"https://example.test/issues/{number}",
        "labels": [{"name": n} for n in names],
        "body": body,
        "state": "OPEN",
        "updatedAt": updated_at,
    }


def _blocked_backlog(
    roots: list[dict[str, Any]] | None,
    *,
    open_total: int = 4,
    blocked: int = 3,
) -> dict[str, Any]:
    """A reachability reading where every ready issue is dependency-blocked."""
    backlog: dict[str, Any] = {
        "observed": True,
        "open_total": open_total,
        "dispatchable": 0,
        "blocked_by_open_dependency": blocked,
    }
    if roots is not None:
        backlog["dependency_root_blockers"] = roots
    return backlog


# ---------------------------------------------------------------------------
# classify_backlog_reachability: the new dependency_root_blockers field.
# ---------------------------------------------------------------------------


def test_classifier_emits_dependency_root_blockers() -> None:
    """The distinct open blockers of blocked-bin issues are emitted, each with
    the GitHub ``updatedAt`` the already-fetched open-issue list carried."""
    config = OrchestratorConfig()
    labels = config.labels
    root_updated = _iso(datetime.now(UTC) - timedelta(days=11))
    gh = FakeGitHub()
    gh.issues = [
        _issue(1538, {labels.ready, labels.pr_open}, updated_at=root_updated),
        _issue(1542, {labels.ready}, body="Blocked by #1538"),
        _issue(1543, {labels.ready}, body="Blocked by #1538"),
        _issue(1544, {labels.ready}, body="Blocked by #1538"),
        _issue(1600, {labels.ready}),  # genuinely dispatchable
    ]

    result = classify_backlog_reachability(gh, config)

    assert result["blocked_by_open_dependency"] == 3
    assert result["dependency_root_blockers"] == [{"number": 1538, "updated_at": root_updated}]


def test_classifier_root_excludes_blocker_that_is_itself_blocked() -> None:
    """A mid-chain blocker is not a root: 1542 blocked by 1543, which is
    itself blocked by 1538 -- only 1538 is the root the whole tail waits on."""
    config = OrchestratorConfig()
    labels = config.labels
    gh = FakeGitHub()
    gh.issues = [
        _issue(1538, {labels.ready, labels.pr_open}),
        _issue(1542, {labels.ready}, body="Blocked by #1543"),
        _issue(1543, {labels.ready}, body="Blocked by #1538"),
    ]

    result = classify_backlog_reachability(gh, config)

    assert result["blocked_by_open_dependency"] == 2
    assert [r["number"] for r in result["dependency_root_blockers"]] == [1538]


def test_classifier_emits_empty_roots_when_nothing_is_blocked() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub()
    gh.issues = [_issue(1, {config.labels.ready})]

    result = classify_backlog_reachability(gh, config)

    assert result["dependency_root_blockers"] == []


def test_classifier_emits_no_roots_when_unobserved() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub()
    gh.issues = []

    result = classify_backlog_reachability(gh, config)

    assert result["observed"] is False
    assert result["dependency_root_blockers"] == []


# ---------------------------------------------------------------------------
# check_dispatch_staleness: the bounded exemption.
# ---------------------------------------------------------------------------


def _idle_state(old: str) -> dict[str, Any]:
    """Root issue 1538 tracked as pr_open; its PR 1595 is janitor_blocked with
    a sole missing-required-checks failure -- the exact #1595/#1133 shape that
    kept ``dispatch_blocked_chain_dead`` silent."""
    return {
        "issues": {
            "1538": {
                "number": 1538,
                "status": "pr_open",
                "dispatched_at": old,
            }
        },
        "prs": {
            "1595": {
                "number": 1595,
                "issue_number": 1538,
                "status": "janitor_blocked",
                "is_missing_checks_only_block": True,
                "reviewed_at": old,
            }
        },
    }


def test_idle_root_blocker_fires_dependency_root_blocker_idle() -> None:
    """Acceptance criterion 1: dispatchable == 0, blocked > 0, every root
    blocker's last change older than dependency_stall_minutes -> loud finding
    naming the root issue and its blocking PR."""
    now = datetime.now(UTC).replace(microsecond=0)
    config = DispatchConfig(dispatch_staleness_minutes=240, dependency_stall_minutes=60)
    old = _iso(now - timedelta(minutes=180))
    state = _idle_state(old)
    backlog = _blocked_backlog([{"number": 1538, "updated_at": old}])

    result = check_dispatch_staleness(state, config, backlog, now=now)

    assert result["stale"] is True
    assert result["reason"] == "dependency_root_blocker_idle"
    assert result["should_emit"] is True
    roots = result["dependency_root_blockers"]
    assert [r["issue"] for r in roots] == [1538]
    assert roots[0]["last_change_at"] == old
    assert roots[0]["idle_seconds"] == 180 * 60
    assert roots[0]["blocking_prs"] == [
        {"number": 1595, "status": "janitor_blocked", "last_change_at": old}
    ]


def test_progressing_root_blocker_stays_quiet() -> None:
    """Acceptance criterion 2: same backlog shape, but the root's GitHub
    updatedAt moved inside the window -> the legitimate-sequencing case stays
    quiet."""
    now = datetime.now(UTC).replace(microsecond=0)
    config = DispatchConfig(dispatch_staleness_minutes=240, dependency_stall_minutes=60)
    old = _iso(now - timedelta(days=11))
    recent = _iso(now - timedelta(minutes=30))
    state = _idle_state(old)
    backlog = _blocked_backlog([{"number": 1538, "updated_at": recent}])

    result = check_dispatch_staleness(state, config, backlog, now=now)

    assert result["stale"] is False
    assert result["reason"] == "all_ready_blocked_by_dependencies"


def test_pr_side_progress_keeps_the_backlog_quiet() -> None:
    """A root whose *issue* has not moved but whose tracked PR has a recent
    state timestamp (review dispatch, janitor transition, retrigger) is
    progressing -- quiet."""
    now = datetime.now(UTC).replace(microsecond=0)
    config = DispatchConfig(dispatch_staleness_minutes=240, dependency_stall_minutes=60)
    old = _iso(now - timedelta(days=11))
    recent = _iso(now - timedelta(minutes=10))
    state = _idle_state(old)
    state["prs"]["1595"]["review_dispatched_at"] = recent
    backlog = _blocked_backlog([{"number": 1538, "updated_at": old}])

    result = check_dispatch_staleness(state, config, backlog, now=now)

    assert result["stale"] is False
    assert result["reason"] == "all_ready_blocked_by_dependencies"


def test_one_progressing_root_among_idle_roots_stays_quiet() -> None:
    """The alarm requires EVERY root to be idle: one root that moved inside
    the window means the blocking set is still alive."""
    now = datetime.now(UTC).replace(microsecond=0)
    config = DispatchConfig(dispatch_staleness_minutes=240, dependency_stall_minutes=60)
    old = _iso(now - timedelta(days=11))
    recent = _iso(now - timedelta(minutes=10))
    state = _idle_state(old)
    backlog = _blocked_backlog(
        [
            {"number": 1538, "updated_at": old},
            {"number": 1700, "updated_at": recent},
        ],
        open_total=5,
        blocked=3,
    )

    result = check_dispatch_staleness(state, config, backlog, now=now)

    assert result["stale"] is False
    assert result["reason"] == "all_ready_blocked_by_dependencies"
    # Even on the quiet path the diagnostic names what it evaluated.
    assert [r["issue"] for r in result["dependency_root_blockers"]] == [1538, 1700]


def test_root_with_no_progress_evidence_counts_as_idle() -> None:
    """A root with no updatedAt and no state entries has no recorded progress
    at all -- infinitely older than any window. Loud, with last_change_at
    honestly None."""
    now = datetime.now(UTC).replace(microsecond=0)
    config = DispatchConfig(dispatch_staleness_minutes=240, dependency_stall_minutes=60)
    backlog = _blocked_backlog([{"number": 1538, "updated_at": None}])

    result = check_dispatch_staleness({}, config, backlog, now=now)

    assert result["stale"] is True
    assert result["reason"] == "dependency_root_blocker_idle"
    roots = result["dependency_root_blockers"]
    assert roots[0]["issue"] == 1538
    assert roots[0]["last_change_at"] is None
    assert roots[0]["idle_seconds"] is None
    assert roots[0]["blocking_prs"] == []


def test_zero_dependency_stall_minutes_restores_unconditional_exemption() -> None:
    """``dependency_stall_minutes: 0`` disables the bound -- the #1110
    exemption becomes unconditional again, matching dispatch_staleness_minutes'
    own 0-disables convention."""
    now = datetime.now(UTC).replace(microsecond=0)
    config = DispatchConfig(dispatch_staleness_minutes=240, dependency_stall_minutes=0)
    old = _iso(now - timedelta(days=30))
    state = _idle_state(old)
    backlog = _blocked_backlog([{"number": 1538, "updated_at": old}])

    result = check_dispatch_staleness(state, config, backlog, now=now)

    assert result["stale"] is False
    assert result["reason"] == "all_ready_blocked_by_dependencies"


def test_missing_roots_field_stays_quiet() -> None:
    """A reachability dict without ``dependency_root_blockers`` (older caller,
    hand-built fixture) carries no root set to judge -- the unknown case stays
    quiet rather than firing an unbounded alarm, matching the
    backlog_not_observed / no_baseline precedent."""
    now = datetime.now(UTC).replace(microsecond=0)
    config = DispatchConfig(dispatch_staleness_minutes=240, dependency_stall_minutes=60)
    backlog = _blocked_backlog(None)

    result = check_dispatch_staleness({}, config, backlog, now=now)

    assert result["stale"] is False
    assert result["reason"] == "all_ready_blocked_by_dependencies"


def test_dependency_root_blocker_idle_obeys_reminder_cadence() -> None:
    """The new arm shares the dispatch_stale alert's edge-triggered +
    bounded-reminder bookkeeping: an already-armed alert inside the reminder
    window does not re-fire."""
    now = datetime.now(UTC).replace(microsecond=0)
    config = DispatchConfig(dispatch_staleness_minutes=240, dependency_stall_minutes=60)
    old = _iso(now - timedelta(minutes=180))
    state = _idle_state(old)
    state = arm_dispatch_stale_alert(state, _iso(now - timedelta(minutes=30)))
    backlog = _blocked_backlog([{"number": 1538, "updated_at": old}])

    result = check_dispatch_staleness(state, config, backlog, now=now)

    assert result["stale"] is True
    assert result["reason"] == "dependency_root_blocker_idle"
    assert result["should_emit"] is False


def test_future_dated_schedule_marker_does_not_count_as_progress() -> None:
    """A ``next_*_at``-style field records a scheduled FUTURE time, not a state
    change -- letting it count as "progress" would pin last_change in the
    future and silence the alarm forever."""
    now = datetime.now(UTC).replace(microsecond=0)
    config = DispatchConfig(dispatch_staleness_minutes=240, dependency_stall_minutes=60)
    old = _iso(now - timedelta(days=11))
    state = _idle_state(old)
    state["issues"]["1538"]["next_probe_at"] = _iso(now + timedelta(hours=1))
    backlog = _blocked_backlog([{"number": 1538, "updated_at": old}])

    result = check_dispatch_staleness(state, config, backlog, now=now)

    assert result["stale"] is True
    assert result["reason"] == "dependency_root_blocker_idle"


# ---------------------------------------------------------------------------
# Config plumbing.
# ---------------------------------------------------------------------------


def test_dependency_stall_minutes_defaults_and_loads(tmp_path: Path) -> None:
    assert DispatchConfig().dependency_stall_minutes == 1440

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text("dispatch:\n  dependency_stall_minutes: 720\n")
    assert load_config(config_file).dispatch.dependency_stall_minutes == 720


def test_dependency_stall_minutes_rejects_non_int(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text("dispatch:\n  dependency_stall_minutes: soon\n")
    with pytest.raises(ConfigError, match="must be an int"):
        load_config(config_file)


def test_dependency_stall_minutes_rejects_negative(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text("dispatch:\n  dependency_stall_minutes: -5\n")
    with pytest.raises(ConfigError, match="must be >= 0"):
        load_config(config_file)


# ---------------------------------------------------------------------------
# End-to-end: a real app.dispatch() pass emits the dispatch_stale event with
# the named root blocker -- the incident shape from the issue.
# ---------------------------------------------------------------------------


def test_dispatch_pass_emits_dispatch_stale_for_idle_root_blocker(tmp_path: Path) -> None:
    """Three ready issues all 'Blocked by #1538'; #1538's tracked PR #1595 has
    been janitor_blocked (missing-checks-only, the #1133 transient exemption
    shape) with no recorded progress longer than dependency_stall_minutes. The
    pass dispatches nothing AND must not stay silent: a dispatch_stale event
    fires with reason dependency_root_blocker_idle naming #1538 and PR #1595.
    """
    now = datetime.now(UTC).replace(microsecond=0)
    old = _iso(now - timedelta(minutes=180))
    config = OrchestratorConfig(
        dispatch=DispatchConfig(dispatch_staleness_minutes=240, dependency_stall_minutes=60)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeGitHub()
    gh.issues = [
        _issue(1538, {config.labels.ready, config.labels.pr_open}, updated_at=old),
        *[
            _issue(n, {config.labels.ready}, body="Blocked by #1538", updated_at=old)
            for n in (1542, 1543, 1544)
        ],
    ]
    gh.prs = [
        {
            "number": 1595,
            "title": "fix: root blocker work",
            "url": "https://example.test/pull/1595",
            "headRefName": "agent/issue-1538-root-blocker-work",
            "baseRefName": "main",
            "headRefOid": "sha-1595",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #1538",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
            "updatedAt": old,
        }
    ]
    app = OrchestratorApp(tmp_path, paths, config, gh)

    state = load_state(paths.state_file)
    state.update(_idle_state(old))
    save_state(paths.state_file, state)

    result = app.dispatch(limit=3)

    assert result.ok is True
    assert result.data["selected_count"] == 0
    stale_events = query_events(paths.state_file, kind="dispatch_stale")
    assert len(stale_events) == 1, stale_events
    payload = stale_events[0]["payload"]
    assert payload["reason"] == "dependency_root_blocker_idle"
    roots = payload["dependency_root_blockers"]
    assert [r["issue"] for r in roots] == [1538]
    assert roots[0]["blocking_prs"] == [
        {"number": 1595, "status": "janitor_blocked", "last_change_at": old}
    ]


def test_dispatch_pass_stays_quiet_when_root_blocker_progressed(tmp_path: Path) -> None:
    """Negative e2e control: identical shape, but #1538's issue updatedAt moved
    inside the window (someone/something touched it) -- no dispatch_stale."""
    now = datetime.now(UTC).replace(microsecond=0)
    old = _iso(now - timedelta(minutes=180))
    recent = _iso(now - timedelta(minutes=10))
    config = OrchestratorConfig(
        dispatch=DispatchConfig(dispatch_staleness_minutes=240, dependency_stall_minutes=60)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeGitHub()
    gh.issues = [
        _issue(1538, {config.labels.ready, config.labels.pr_open}, updated_at=recent),
        *[
            _issue(n, {config.labels.ready}, body="Blocked by #1538", updated_at=old)
            for n in (1542, 1543, 1544)
        ],
    ]
    gh.prs = [
        {
            "number": 1595,
            "title": "fix: root blocker work",
            "url": "https://example.test/pull/1595",
            "headRefName": "agent/issue-1538-root-blocker-work",
            "baseRefName": "main",
            "headRefOid": "sha-1595",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #1538",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
            "updatedAt": recent,
        }
    ]
    app = OrchestratorApp(tmp_path, paths, config, gh)

    state = load_state(paths.state_file)
    state.update(_idle_state(old))
    save_state(paths.state_file, state)

    result = app.dispatch(limit=3)

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert query_events(paths.state_file, kind="dispatch_stale") == []
