"""Merge evidence, escalation categories, lifecycle names and global-kind history (rollup review)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from _dashboard_metrics_fixtures import day, dispatch
from _dashboard_rollup_fixtures import Fleet

from charlie_work.dashboard import metrics_flow as flow
from charlie_work.dashboard import rollup
from charlie_work.dashboard.metrics_base import MetricQuery, open_dashboard_ro
from charlie_work.dashboard.rollup_derive import derive_event
from charlie_work.dashboard.rollup_flow_handlers import lifecycle_state, reason_category


NOW = datetime(2026, 10, 8, tzinfo=UTC)


def _ev(kind: str, payload: dict, issue=None, pr=None) -> dict:
    return {"id": 1, "ts": "2026-10-01T00:00:00Z", "kind": kind, "payload": payload,
            "issue_number": issue, "pr_number": pr}  # fmt: skip


def _rows(table: str, kind: str, payload: dict) -> list[dict]:
    return [c for t, c in derive_event("repo", _ev(kind, payload)) if t == table]


def _merged(kind: str, payload: dict) -> list[tuple]:
    return [(c["issue"], c["pr"]) for c in _rows("issue_milestones", kind, payload)
            if c["milestone"] == "merged"]  # fmt: skip


def test_every_merge_kind_emits_a_merged_milestone() -> None:
    assert _merged("merge_succeeded", {"issue_number": 5, "pr_number": 50}) == [(5, 50)]
    assert _merged("dispatch_merged_pr_references_closed", {"issue_numbers": [6, 7]}) == [
        (6, None),
        (7, None),
    ]
    assert _merged("reconcile", {"kind": "merged_outside_orchestrator", "issue_number": 8}) == [
        (8, None)
    ]
    # reaped names a PR only and cannot be joined to issue-only evidence: not a merge signal
    assert (
        _merged("review_dispatch_lifecycle_reaped", {"pr_number": 9, "github_state": "merged"})
        == []
    )


def test_finalize_uses_every_ref() -> None:
    p = {"issue_numbers": [1, 2], "pr_numbers": [11, 12]}
    assert _merged("finalize_externally_merged", p) == [(1, 11), (2, 12)]
    assert _merged(
        "finalize_externally_merged", {"issue_numbers": [1, 2], "pr_numbers": [11]}
    ) == [
        (1, None),
        (2, None),
    ]


def test_escalation_reason_falls_back_to_failure_kind_and_is_bounded() -> None:
    (row,) = _rows("escalations", "session_failed_escalated", {"failure_kind": "worktree_unsafe"})
    assert (row["reason"], row["detail"]) == ("worktree_unsafe", None)
    raw = "cross_repo_target: all 3 referenced file path(s) are absent (C:\\Users\\x\\repos\\y)"
    (row,) = _rows("escalations", "dispatch_cross_repo_escalated", {"reason": raw})
    assert (row["reason"], row["detail"]) == ("cross_repo_target", raw)
    assert reason_category("Something odd 12 happened here") == "something_odd_n"
    assert reason_category("  ") is None


def test_lifecycle_accepts_label_cache_and_human_names() -> None:
    assert lifecycle_state("agent:in-progress") == "in_progress"
    assert lifecycle_state("agent:done") == "done"
    assert lifecycle_state("automated-ready") == "ready"
    assert lifecycle_state("merged") == "done"
    assert lifecycle_state("PR open") == "pr_open"
    (m,) = _rows(
        "issue_milestones",
        "lifecycle_transition",
        {"issue_number": 3, "to_state": "agent:pr-open"},
    )
    assert m["milestone"] == "pr_open"


def test_starved_event_is_attributed_to_the_named_repo() -> None:
    named = _rows("capped_demand", "runner_capacity_starved", {"demand": 3, "repo": "o/starved"})
    assert named[0]["repo"] == "o/starved"
    assert "repo" in derive_event("fleet", _ev("runner_capacity_starved", {}))[0][1]  # == source


def _roll(f: Fleet):
    for state in (f.alpha, f.beta, f.fleet_state):  # every source DB must exist
        f.emit(state, day(0), "supervisor_started", {})
    result = rollup.run_rollup(f.sources(), NOW)
    assert result.errors == ()
    db, err = open_dashboard_ro(f.sources().db_path)
    assert db is not None, err
    return db


def test_union_of_merge_kinds_counts_once_and_feeds_lead_time(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path / "fleet"))
    f = Fleet(tmp_path, monkeypatch)
    a, b = f.alpha, f.beta
    dispatch(f, a, day(2, 0), [10])
    # one merge, three signals: the orchestrator's own, the dispatch-side close, finalize
    f.emit(a, day(2, 3), "merge_succeeded", {"issue_number": 10, "pr_number": 110})
    f.emit(a, day(2, 4), "dispatch_merged_pr_references_closed", {"issue_numbers": [10]})
    f.emit(
        a, day(2, 5), "finalize_externally_merged", {"issue_numbers": [10], "pr_numbers": [110]}
    )
    # issue-only close, no other evidence
    dispatch(f, a, day(2, 1), [11])
    f.emit(a, day(2, 7), "dispatch_merged_pr_references_closed", {"issue_numbers": [11]})
    # local lane (no remote): only merge_succeeded
    dispatch(f, b, day(3, 0), [20])
    f.emit(b, day(3, 2), "merge_succeeded", {"issue_number": 20, "pr_number": 120, "local": True})
    db = _roll(f)
    q = MetricQuery(datetime(2026, 10, 1, tzinfo=UTC), NOW, timedelta(days=1))
    series = flow.merges_per_day(db, q)
    assert sum(v for _, v in series.points) == 3.0
    assert {s: sum(v for _, v in pts) for s, pts in series.per_repo.items()} == {
        "owner/alpha": 2.0,
        "local/beta": 1.0,
    }
    lead = flow.lead_time(db, q)
    # medians per day: day 2 holds issue 10 (3h) and 11 (6h); day 3 holds issue 20 (2h)
    assert sorted(v for _, v in lead.points) == [2.0, 4.5]
    db.close()
    f.close()


def test_global_kind_copies_are_taken_only_before_the_global_first_row(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path / "fleet"))
    f = Fleet(tmp_path, monkeypatch)
    target = {"budget": 4, "targets": [{"repo": "o/x", "capacity": 2, "demand": 1, "running": 1}]}
    for ts in (day(1, 0), day(2, 0)):  # before the global DB started writing: history
        f.emit(f.alpha, ts, "runner_allocation", target)
    f.emit(f.fleet_state, day(3, 0), "runner_allocation", target)  # global first row
    f.emit(f.alpha, day(3, 0), "runner_allocation", target)  # same moment: not before, skipped
    f.emit(f.alpha, day(4, 0), "runner_allocation", target)  # overlap period: skipped
    f.emit(f.alpha, day(1, 1), "fleet_job_observations", {"jobs": [_job_done("old")]})
    f.emit(f.fleet_state, day(3, 1), "fleet_job_observations", {"jobs": [_job_done("new")]})
    db = _roll(f)
    got = db.execute("SELECT source, ts FROM runner_samples ORDER BY ts").fetchall()
    assert got == [("owner/alpha", day(1, 0)), ("owner/alpha", day(2, 0)), ("fleet", day(3, 0))]
    assert db.execute("SELECT job_id FROM job_observations ORDER BY job_id").fetchall() == [
        ("new",),
        ("old",),
    ]
    db.close()
    f.close()


def _job_done(job_id: str) -> dict:
    m = {"kind": "measured", "seconds": 1.0}
    d = {"queue_wait": m, "execution": m, "wall": m}
    return {"job_id": job_id, "name": "T", "status": "completed", "durations": d}
