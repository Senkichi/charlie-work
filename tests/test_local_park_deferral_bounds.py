"""Issue #1971 rework: the bounded probe deferral and the park-lane guards.

The first cut of the dead-dispatched backstop park deferred forever on a
``probe_failed`` / ``park_failed`` verdict -- a deterministic git failure (a
missing base ref, no merge base) would wedge a ``dispatched`` entry with no
exit, a regression from the old reclaim/escalate. This file pins:

- the bound on BOTH lanes that defer -- the in-lock #654 backstop and the
  #1923 active-labeled reclaim lane -- including the probe error (git's own
  stderr) riding on the escalation / reclaim record;
- the #1923 reclaim lane's skip-on-``probe_failed`` (active label retained,
  no ready re-add, no ``session_failed_relabeled``) and its recovery;
- ``park_backstop_due_local_orphans``' skip guards, each of which fails if the
  guard is removed (a mock on the park call proves "not re-parked");
- ``park_labelless_dead_local_session``' guards;
- ``dead_dispatched_reap_due`` parity with the in-lock timer, including the
  provider-throttle exemption.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from _dws_facts import run_reap

from _local_park_fixtures import (
    _add_worktree_commit,
    _events,
    _git,
    _init_repo,
    _label_names,
    _local_config,
    _run_sweep,
    _seed_dead_dispatched,
    _sessions_dir,
    _wg,
    _write_issue,
)
from _local_park_fixtures import _shallow_wts as _register_shallow_wts  # noqa: F401
from charlie_work.config import LabelConfig
from charlie_work.dead_dispatched_timer import (
    LOCAL_PARK_DEFER_FIELDS,
    LOCAL_PARK_DEFER_MAX_PASSES,
    dead_dispatched_reap_due,
    defer_or_expire_local_park,
)
from charlie_work.local_issues import LocalFileGitHub
from charlie_work.local_lane import probe_branch_ref
from charlie_work.local_work_park import (
    LocalParkResult,
    park_backstop_due_local_orphans,
    park_labelless_dead_local_session,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state
from charlie_work.subprocess_runner import RunResult

_NOW = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)


_GIT_STDERR = "git diff main...agent/x: fatal: bad revision 'main...agent/x'"


def _probe_error():
    """Patch target: the park lane's branch-diff probe fails with git stderr."""
    return patch(
        "charlie_work.local_work_park.branch_diff_result",
        return_value=(None, _GIT_STDERR),
    )


def _drift(state_file: Path, issue_number: int) -> list[dict[str, Any]]:
    return [
        event
        for event in _events(state_file, "orphaned_worker_drift", issue_number)
        if event["payload"].get("reason") == "dead_dispatched_local_park_deferred"
    ]


def _worktree_gone_setup(tmp_path: Path, shallow_wts: Path, number: int, labels: tuple[str, ...]):
    """Repo with a committed branch whose worktree is gone (branch-ref probe)."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _local_config(repo_root, shallow_wts)
    branch = f"agent/issue-{number}-x"
    worktree_path = _add_worktree_commit(repo_root, config, branch)
    _git(repo_root, "worktree", "remove", "--force", str(worktree_path))
    issues_dir = repo_root / config.local_issues.issues_dir
    _write_issue(issues_dir, number, title="fix the flaky test", labels=labels)
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    return repo_root, config, gh, paths, branch


# ---------------------------------------------------------------------------
# #1923 reclaim lane: an ACTIVE-labeled orphan whose diff probe fails.
# ---------------------------------------------------------------------------


def test_reclaim_lane_probe_failed_skips_reclaim_then_parks(
    tmp_path: Path, shallow_wts: Path
) -> None:
    labels_cfg = LabelConfig()
    number = 1981
    _, config, gh, paths, branch = _worktree_gone_setup(
        tmp_path, shallow_wts, number, (labels_cfg.in_progress,)
    )
    # Drift armed 5 minutes ago: the #654 backstop is NOT due, so this is
    # purely the reclaim lane's decision.
    _seed_dead_dispatched(paths.state_file, number, branch, armed_drift_minutes_ago=5)
    sessions_dir = _sessions_dir(tmp_path)

    with _probe_error():
        _run_sweep(
            sessions_dir, paths.state_file, config, gh, _wg(paths.state_file), tmp_path / "fleet"
        )

    current = _label_names(gh, number)
    assert labels_cfg.in_progress in current  # active label retained
    assert labels_cfg.ready not in current  # no ready re-add
    assert labels_cfg.review_ready not in current
    assert _events(paths.state_file, "session_failed_relabeled", number) == []
    drift = _drift(paths.state_file, number)
    assert len(drift) == 1  # observable, and it names git's stderr
    assert "bad revision" in drift[0]["payload"]["detail"]
    entry = load_state(paths.state_file)["issues"][str(number)]
    assert entry["status"] == "dispatched"
    assert entry["local_park_defer_count"] == 1

    # Healthy next pass: the real diff is non-empty -> parked, counters gone.
    _run_sweep(
        sessions_dir, paths.state_file, config, gh, _wg(paths.state_file), tmp_path / "fleet"
    )
    current = _label_names(gh, number)
    assert labels_cfg.review_ready in current
    assert labels_cfg.in_progress not in current
    entry = load_state(paths.state_file)["issues"][str(number)]
    assert entry["status"] == "open_passive"
    assert "local_park_defer_count" not in entry


def test_reclaim_lane_probe_failure_budget_spent_reclaims_with_git_error(
    tmp_path: Path, shallow_wts: Path
) -> None:
    """The cap: a deterministic probe failure cannot wedge the entry."""
    labels_cfg = LabelConfig()
    number = 1982
    _, config, gh, paths, branch = _worktree_gone_setup(
        tmp_path, shallow_wts, number, (labels_cfg.in_progress,)
    )
    _seed_dead_dispatched(paths.state_file, number, branch, armed_drift_minutes_ago=5)
    sessions_dir = _sessions_dir(tmp_path)

    for _ in range(LOCAL_PARK_DEFER_MAX_PASSES - 1):
        with _probe_error():
            _run_sweep(
                sessions_dir,
                paths.state_file,
                config,
                gh,
                _wg(paths.state_file),
                tmp_path / "fleet",
            )
        assert labels_cfg.in_progress in _label_names(gh, number)
        assert _events(paths.state_file, "session_failed_relabeled", number) == []

    with _probe_error():
        _run_sweep(
            sessions_dir, paths.state_file, config, gh, _wg(paths.state_file), tmp_path / "fleet"
        )

    current = _label_names(gh, number)
    assert labels_cfg.in_progress not in current
    assert labels_cfg.ready in current
    relabeled = _events(paths.state_file, "session_failed_relabeled", number)
    assert len(relabeled) == 1
    payload = relabeled[0]["payload"]
    assert payload["salvage_failed"] is True
    assert "bad revision" in payload["salvage_error"]  # git stderr, not just argv
    assert str(LOCAL_PARK_DEFER_MAX_PASSES) in payload["salvage_error"]
    # Same reason on every pass -> exactly one deferral drift event.
    assert len(_drift(paths.state_file, number)) == 1


# ---------------------------------------------------------------------------
# #654 backstop lane: the cap, and one drift event per distinct reason.
# ---------------------------------------------------------------------------


def test_backstop_deferral_is_bounded_and_escalation_carries_probe_error(
    tmp_path: Path, shallow_wts: Path
) -> None:
    labels_cfg = LabelConfig()
    number = 1983
    _, config, gh, paths, branch = _worktree_gone_setup(tmp_path, shallow_wts, number, ())
    _seed_dead_dispatched(paths.state_file, number, branch)
    sessions_dir = _sessions_dir(tmp_path)

    for expected_count in range(1, LOCAL_PARK_DEFER_MAX_PASSES):
        with _probe_error():
            _run_sweep(
                sessions_dir,
                paths.state_file,
                config,
                gh,
                _wg(paths.state_file),
                tmp_path / "fleet",
            )
        entry = load_state(paths.state_file)["issues"][str(number)]
        assert entry["status"] == "dispatched"
        assert entry["local_park_defer_count"] == expected_count
        assert _events(paths.state_file, "dead_dispatched_worker_reaped", number) == []
        # Two consecutive deferred passes with the SAME reason -> ONE event.
        assert len(_drift(paths.state_file, number)) == 1

    with _probe_error():
        _run_sweep(
            sessions_dir, paths.state_file, config, gh, _wg(paths.state_file), tmp_path / "fleet"
        )

    entry = load_state(paths.state_file)["issues"][str(number)]
    assert entry["status"] == "escalated"
    assert "bad revision" in entry["local_park_defer_reason"]
    reaped = _events(paths.state_file, "dead_dispatched_worker_reaped", number)
    assert len(reaped) == 1
    assert "bad revision" in reaped[0]["payload"]["local_park_defer_error"]
    assert labels_cfg.review_ready not in _label_names(gh, number)


def test_defer_emits_one_drift_event_per_distinct_reason() -> None:
    entry: dict[str, Any] = {"status": "dispatched"}
    events: list[tuple[str, dict[str, Any]]] = []
    kwargs = {"issue_number": 7, "orphan_drift_at": None, "sweep_events": events}

    def defer(reason: str, minutes: int) -> bool:
        return defer_or_expire_local_park(
            entry, reason=reason, now=_NOW + timedelta(minutes=minutes), **kwargs
        )

    assert defer("reason-a", 0) and defer("reason-a", 1)
    assert [e[1]["detail"] for e in events] == ["reason-a"]  # same reason: one event
    assert entry["local_park_defer_count"] == 2
    entry["local_park_defer_count"] = 0  # fresh budget, new reason
    entry["local_park_defer_pass_key"] = None
    assert defer("reason-b", 2)
    assert [e[1]["detail"] for e in events] == ["reason-a", "reason-b"]
    assert all(e[0] == "orphaned_worker_drift" for e in events)


# ---------------------------------------------------------------------------
# park_backstop_due_local_orphans skip guards: each fails if removed.
# ---------------------------------------------------------------------------


def _drain_setup(tmp_path: Path, shallow_wts: Path, number: int, labels: tuple[str, ...]):
    repo_root, config, gh, paths, branch = _worktree_gone_setup(
        tmp_path, shallow_wts, number, labels
    )
    _seed_dead_dispatched(paths.state_file, number, branch)
    state = load_state(paths.state_file)
    return repo_root, config, gh, paths, state


def _drain(
    number: int,
    tmp_path: Path,
    shallow_wts: Path,
    *,
    labels: tuple[str, ...] = (),
    escalations: tuple[frozenset[int], ...] = (),
    reclaim_results: dict[int, dict[str, Any]] | None = None,
    park_verdicts: dict[int, LocalParkResult] | None = None,
) -> tuple[dict[int, str], MagicMock]:
    repo_root, config, gh, paths, state = _drain_setup(tmp_path, shallow_wts, number, labels)
    with patch(
        "charlie_work.local_work_park.park_salvageable_local_orphan",
        return_value=LocalParkResult("parked"),
    ) as park:
        deferred = park_backstop_due_local_orphans(
            gh=gh,
            config=config,
            repo_root=repo_root,
            worktrees_dir=None,
            state=state,
            state_file=paths.state_file,
            no_pr_orphans=[number],
            issues_by_number={number: gh.issue_view(number)},
            worker_outcomes={},
            reclaim_results=reclaim_results or {},
            escalations=escalations,
            park_verdicts=park_verdicts or {},
            dead_dispatched_reap_minutes=config.watchdog.dead_dispatched_reap_minutes,
            now=datetime.now(UTC),
            write_gate=_wg(paths.state_file),
        )
    return deferred, park


def test_drain_control_parks_an_unguarded_due_orphan(tmp_path: Path, shallow_wts: Path) -> None:
    deferred, park = _drain(1990, tmp_path, shallow_wts)
    assert park.call_count == 1  # the positive control: the mock path is live
    assert deferred == {}


@pytest.mark.parametrize("terminal", ["human_needed", "done"])
def test_drain_skips_terminal_labeled_issue(
    tmp_path: Path, shallow_wts: Path, terminal: str
) -> None:
    label = getattr(LabelConfig(), terminal)
    _, park = _drain(1991, tmp_path, shallow_wts, labels=(label,))
    park.assert_not_called()


@pytest.mark.parametrize("slot", [0, 1, 2])
def test_drain_skips_same_pass_escalations(tmp_path: Path, shallow_wts: Path, slot: int) -> None:
    groups = [frozenset(), frozenset(), frozenset()]
    groups[slot] = frozenset({1992})
    _, park = _drain(1992, tmp_path, shallow_wts, escalations=tuple(groups))
    park.assert_not_called()


def test_drain_skips_reclaimed_issue(tmp_path: Path, shallow_wts: Path) -> None:
    _, park = _drain(1993, tmp_path, shallow_wts, reclaim_results={1993: {"label_write_ok": True}})
    park.assert_not_called()


def test_drain_skips_prior_verdict_and_forwards_failed_ones(
    tmp_path: Path, shallow_wts: Path
) -> None:
    deferred, park = _drain(
        1994, tmp_path, shallow_wts, park_verdicts={1994: LocalParkResult("parked")}
    )
    park.assert_not_called()
    assert deferred == {}
    deferred, park = _drain(
        1995,
        tmp_path,
        shallow_wts,
        park_verdicts={1995: LocalParkResult("probe_failed", "boom")},
    )
    park.assert_not_called()  # never re-probed in the same pass...
    assert deferred == {1995: "boom"}  # ...but the failure still defers the backstop


# ---------------------------------------------------------------------------
# park_labelless_dead_local_session guards.
# ---------------------------------------------------------------------------


def _labelless_setup(tmp_path: Path, shallow_wts: Path, number: int, entry: dict[str, Any]):
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _local_config(repo_root, shallow_wts)
    issues_dir = repo_root / config.local_issues.issues_dir
    _write_issue(issues_dir, number, labels=())
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)
    state_file = tmp_path / "state.json"
    state = {"version": 1, "issues": {str(number): entry}, "prs": {}, "events": []}
    save_state(state_file, state)
    return repo_root, config, gh, state_file


def _call_labelless(number: int, tmp_path: Path, shallow_wts: Path, entry, worker_kwargs):
    repo_root, config, gh, state_file = _labelless_setup(tmp_path, shallow_wts, number, entry)
    worker = SimpleNamespace(
        issue_number=number, branch=worker_kwargs.get("branch", ""), worktree_path=""
    )
    worker.worktree_path = worker_kwargs.get("worktree_path", "")
    with (
        patch(
            "charlie_work.local_work_park.park_salvageable_local_orphan",
            return_value=LocalParkResult("parked"),
        ) as park,
        patch("charlie_work.local_work_park.read_worker_outcome") as read_outcome,
    ):
        result = park_labelless_dead_local_session(
            gh=gh,
            config=config,
            repo_root=repo_root,
            issue_labels=set(),
            worker=worker,
            write_gate=_wg(state_file),
        )
    return result, park, read_outcome


def test_labelless_session_park_control(tmp_path: Path, shallow_wts: Path) -> None:
    result, park, _ = _call_labelless(1996, tmp_path, shallow_wts, {"status": "dispatched"}, {})
    assert result is True
    assert park.call_count == 1


def test_labelless_session_park_skips_non_dispatched_entry(
    tmp_path: Path, shallow_wts: Path
) -> None:
    result, park, _ = _call_labelless(1997, tmp_path, shallow_wts, {"status": "open_passive"}, {})
    assert result is False
    park.assert_not_called()


def test_labelless_session_empty_worktree_path_never_reads_outcome(
    tmp_path: Path, shallow_wts: Path
) -> None:
    result, park, read_outcome = _call_labelless(
        1998, tmp_path, shallow_wts, {"status": "dispatched"}, {"worktree_path": ""}
    )
    assert result is True
    read_outcome.assert_not_called()  # "" must not resolve to cwd
    assert park.call_args.kwargs["worker_outcome"] is None


def test_labelless_session_backfills_branch_from_sidecar(
    tmp_path: Path, shallow_wts: Path
) -> None:
    _, park, _ = _call_labelless(
        1999, tmp_path, shallow_wts, {"status": "dispatched"}, {"branch": "agent/issue-1999-old"}
    )
    passed_entry = park.call_args.kwargs["state"]["issues"]["1999"]
    assert passed_entry["branch_name"] == "agent/issue-1999-old"


def test_labelless_session_keeps_recorded_branch_over_sidecar(
    tmp_path: Path, shallow_wts: Path
) -> None:
    _, park, _ = _call_labelless(
        2000,
        tmp_path,
        shallow_wts,
        {"status": "dispatched", "branch_name": "agent/recorded"},
        {"branch": "agent/sidecar"},
    )
    assert park.call_args.kwargs["state"]["issues"]["2000"]["branch_name"] == "agent/recorded"


# ---------------------------------------------------------------------------
# dead_dispatched_reap_due parity with the in-lock timer.
# ---------------------------------------------------------------------------


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


# (failure kind, drift armed N minutes ago, throttled_until offset in minutes, due)
# Expected values are derived by hand from the documented semantics (60-minute
# grace; a throttle death is exempt while the window is open and otherwise
# anchors the grace at max(drift, window end)) -- NOT by calling the timer,
# which delegates to the predicate under test and would make this circular.
_REAP_DUE_CASES = [
    (None, 10, None, False),
    (None, 120, None, True),
    (None, 120, 30, True),  # non-throttle kind: throttled_until is irrelevant
    (None, 120, -4, True),
    ("rate_limited", 10, None, False),
    ("rate_limited", 120, None, True),  # no stamped window -> plain drift grace
    ("rate_limited", 120, 30, False),  # window still open -> exempt
    ("rate_limited", 500, 30, False),
    ("quota_exhausted", 120, -4, False),  # window closed 4m ago -> anchor there
    ("quota_exhausted", 10, -90, False),  # drift newer than window end
    ("quota_exhausted", 120, -90, True),  # grace elapsed since window end
]


@pytest.mark.parametrize(("kind", "drift_ago", "until_offset", "expected"), _REAP_DUE_CASES)
def test_reap_due_matches_documented_semantics(
    tmp_path: Path, kind: str | None, drift_ago: int, until_offset: int | None, expected: bool
) -> None:
    entry: dict[str, Any] = {
        "status": "dispatched",
        "orphan_drift_at": _iso(_NOW - timedelta(minutes=drift_ago)),
    }
    if kind is not None:
        entry["dead_worker_failure_kind"] = kind
    state: dict[str, Any] = {"issues": {"42": dict(entry)}, "prs": {}}
    if until_offset is not None:
        state["throttled_until"] = _iso(_NOW + timedelta(minutes=until_offset))
    assert (
        dead_dispatched_reap_due(
            state=state, entry=entry, dead_dispatched_reap_minutes=60, now=_NOW
        )
        is expected
    )
    # The sweep's decision must agree (``max_rearms=0`` disables the bounded
    # re-arm, which mutates the entry and is not part of the predicate).
    run = run_reap(
        state["issues"]["42"],
        issue=42,
        reap_minutes=60,
        max_rearms=0,
        throttled_until=state.get("throttled_until"),
        now=_NOW,
    )
    assert run.reaped is expected


# ---------------------------------------------------------------------------
# local_lane.probe_branch_ref -- the tri-state ref probe the park lane uses.
# ---------------------------------------------------------------------------


def test_probe_branch_ref_true_false_and_none(tmp_path: Path) -> None:
    """rc 0 -> (True, None), rc 1 -> (False, None), rc >= 2 / spawn error -> (None, detail)."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    _git(repo, "branch", "agent/issue-7-x")
    assert probe_branch_ref(repo, "agent/issue-7-x") == (True, None)  # rc 0
    assert probe_branch_ref(repo, "ghost-branch") == (False, None)  # rc 1

    for result in (
        RunResult(returncode=128, stdout="", stderr="fatal: not a git repository"),
        RunResult(returncode=None, stdout="", stderr="", error="spawn failed"),
        RunResult(returncode=None, stdout="", stderr="", timed_out=True, error="timeout"),
    ):
        with patch("charlie_work.local_lane.run_captured", return_value=result):
            exists, detail = probe_branch_ref(repo, "agent/issue-7-x")
            assert exists is None and detail  # the third state carries a reason


# ---------------------------------------------------------------------------
# park_backstop_due_local_orphans -- the stale-snapshot due-check edge.
# ---------------------------------------------------------------------------


def test_drain_rechecks_due_against_fresh_state(tmp_path: Path, shallow_wts: Path) -> None:
    """Drift cleared between the sweep's pre-lock snapshot and the drain
    must not be probed: the in-lock backstop re-reads the entry before
    escalating, so the drain re-verifies due-ness on a fresh locked read --
    the snapshot-only check would park an already-resolved entry."""
    number = 2004
    repo_root, config, gh, paths, state = _drain_setup(tmp_path, shallow_wts, number, ())
    # An earlier pre-lock lane cleared the drift under its own state_lock:
    # the snapshot still shows it armed (due), disk does not.
    fresh = load_state(paths.state_file)
    fresh["issues"][str(number)].pop("orphan_drift_at", None)
    save_state(paths.state_file, fresh)

    with patch(
        "charlie_work.local_work_park.park_salvageable_local_orphan",
        return_value=LocalParkResult("parked"),
    ) as park:
        deferred = park_backstop_due_local_orphans(
            gh=gh,
            config=config,
            repo_root=repo_root,
            worktrees_dir=None,
            state=state,
            state_file=paths.state_file,
            no_pr_orphans=[number],
            issues_by_number={number: gh.issue_view(number)},
            worker_outcomes={},
            reclaim_results={},
            escalations=(),
            park_verdicts={},
            dead_dispatched_reap_minutes=config.watchdog.dead_dispatched_reap_minutes,
            now=datetime.now(UTC),
            write_gate=_wg(paths.state_file),
        )

    park.assert_not_called()
    assert deferred == {}


# ---------------------------------------------------------------------------
# _reset_probe_deferral -- the probe_failed -> conclusive transition.
# ---------------------------------------------------------------------------


def test_probe_failed_then_conclusive_verdict_clears_deferral(
    tmp_path: Path, shallow_wts: Path
) -> None:
    """A conclusive verdict after a deferred probe clears the stamped budget.

    The recovery test covers probe_failed -> ``parked``, which clears the
    counters via the park's own status flip. A conclusive verdict that does
    NOT park -- a provably absent branch ref -> ``no_commits`` -- reaches
    ``_reset_probe_deferral`` in ``park_or_reclaim_local_orphan``'s else
    branch; this pins that path.
    """
    labels_cfg = LabelConfig()
    number = 2005
    _, config, gh, paths, branch = _worktree_gone_setup(
        tmp_path, shallow_wts, number, (labels_cfg.in_progress,)
    )
    _seed_dead_dispatched(paths.state_file, number, branch, armed_drift_minutes_ago=5)
    sessions_dir = _sessions_dir(tmp_path)

    # Pass 1: inconclusive probe -> bounded deferral stamped on the entry.
    with _probe_error():
        _run_sweep(
            sessions_dir,
            paths.state_file,
            config,
            gh,
            _wg(paths.state_file),
            tmp_path / "fleet",
        )
    entry = load_state(paths.state_file)["issues"][str(number)]
    assert entry["local_park_defer_count"] == 1

    # Pass 2: provably empty ref -> ``no_commits`` -> the reclaim proceeds
    # and the deferral bookkeeping is cleared.
    with (
        _probe_error(),
        patch("charlie_work.local_work_park.probe_branch_ref", return_value=(False, None)),
    ):
        _run_sweep(
            sessions_dir,
            paths.state_file,
            config,
            gh,
            _wg(paths.state_file),
            tmp_path / "fleet",
        )

    entry = load_state(paths.state_file)["issues"][str(number)]
    for field_name in LOCAL_PARK_DEFER_FIELDS:
        assert field_name not in entry
    assert labels_cfg.ready in _label_names(gh, number)


def test_reap_due_throttle_exemption_is_not_vacuous() -> None:
    """Control: the grid above contains both a True and a False outcome for a
    throttle death, so a predicate that ignored the exemption would diverge."""
    entry = {
        "orphan_drift_at": _iso(_NOW - timedelta(minutes=500)),
        "dead_worker_failure_kind": "rate_limited",
    }
    open_window = {"throttled_until": _iso(_NOW + timedelta(minutes=30))}
    assert not dead_dispatched_reap_due(
        state=open_window, entry=entry, dead_dispatched_reap_minutes=60, now=_NOW
    )
    assert dead_dispatched_reap_due(
        state={}, entry=entry, dead_dispatched_reap_minutes=60, now=_NOW
    )
