# ruff: noqa: F811  (the imported ``fleet`` fixture is re-bound as a test parameter)
"""Drill-down read models (``dashboard/drill``) over DBs built by the real writers."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta, timezone

import pytest
from _dashboard_rollup_fixtures import ALPHA, BETA, NOW, fleet  # noqa: F401

from charlie_work import instrumentation
from charlie_work.config import LabelConfig
from charlie_work.dashboard import drill, rollup, sources
from charlie_work.dashboard.drill.loop_pass import MAX_PASS_EVENTS, payload_preview
from charlie_work.dashboard.drill.repo import merges_for
from charlie_work.dashboard.metrics_flow import BACKFILL_BURST
from charlie_work.dashboard.now_model import build_now_model
from charlie_work.dashboard.now_types import RepoRead, SourcesRead
from charlie_work.dashboard.sources import SnapshotRead

CID = "ed62ee91e2e5"
L = LabelConfig()
TZ = timezone(timedelta(hours=-7))


@pytest.fixture
def built(fleet):
    """Rolled-up dashboard.db plus alpha events carrying the loop pass's correlation id."""
    for i, (kind, payload) in enumerate(
        [
            ("loop_started", {"pass": 1}),
            ("dispatch_deferred", {"reason": "cap", "nested": {"a": 1}, "items": [1, 2, 3]}),
            ("merge_failed", {"error": "boom\x1b[31m\nline2" + "x" * 300}),
        ]
    ):
        fleet.monkeypatch.setattr(
            instrumentation, "_now_iso", lambda i=i: f"2026-10-01T08:00:0{i}Z"
        )
        instrumentation.log_event(fleet.alpha / "state.json", kind, payload, correlation_id=CID)
    result = rollup.run_rollup(fleet.sources(), NOW)
    assert result.errors == ()
    return fleet


def _snapshot(data: dict | None) -> SnapshotRead:
    return SnapshotRead(NOW, 5.0, data, None)


def test_issue_timeline_stage_times_and_approx(built) -> None:
    got = drill.issue_drill(
        ALPHA, 2199, built.sources().db_path, None, NOW + timedelta(hours=1), tz=TZ
    )
    assert isinstance(got, drill.IssueDrill)
    assert got.kind == "issue" and got.prs == (2208,) and got.approx is True
    labels = [(e.label, e.event_kind) for e in got.timeline]
    assert labels == [
        ("Review verdict: approved", "record_review"),
        ("Merged", "reconcile"),
    ]
    assert got.timeline[0].ts == "2026-10-01T08:06:00Z"
    assert got.timeline[0].ts_local == "2026-10-01T01:06:00-07:00"  # local, not UTC
    assert got.timeline[1].approx is True  # reconcile-detected merge
    assert got.known and got.history_from is not None


def test_issue_with_dispatch_open_stage_and_worker_exit(built) -> None:
    now = NOW + timedelta(hours=1)  # 2h after the 08:00 dispatch... 09:xx later
    got = drill.issue_drill(ALPHA, 2195, built.sources().db_path, None, now)
    assert isinstance(got, drill.IssueDrill)
    kinds = [e.event_kind for e in got.timeline]
    assert kinds == ["dispatch_rework", "session_exited"]
    assert got.timeline[1].detail == "stalled, DEAD"
    # no PR-opened/dispatched pair for 2195: nothing to time, and it is flagged approx
    assert got.stage_times == () and got.lead_seconds is None and got.approx


def test_stage_time_accumulates_across_revisits_and_counts_open_visit(built) -> None:
    for ts, kind, p in [
        ("2026-10-01T08:00:00Z", "dispatch", {"issue_numbers": [900]}),
        (
            "2026-10-01T08:10:00Z",
            "worker_handoff_pr_opened",
            {"issue_number": 900, "pr_number": 901},
        ),
        (
            "2026-10-01T08:20:00Z",
            "record_review",
            {"decision": "request_changes", "issue_number": 900, "pr_number": 901},
        ),
        ("2026-10-01T08:30:00Z", "dispatch_rework", {"issue_numbers": [900], "pr_number": 901}),
        (
            "2026-10-01T08:40:00Z",
            "worker_handoff_pr_opened",
            {"issue_number": 900, "pr_number": 901},
        ),
        ("2026-10-01T08:50:00Z", "review_dispatch_claim", {"pr_numbers": [901]}),
    ]:
        built.emit(built.alpha, ts, kind, p)
    assert rollup.run_rollup(built.sources(), NOW).errors == ()
    now = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
    got = drill.issue_drill(ALPHA, 900, built.sources().db_path, None, now)
    assert isinstance(got, drill.IssueDrill)
    by = {s.stage: s for s in got.stage_times}
    # the in_progress span closes at the first PR-opened signal; the later pr_opened row is a
    # revisit only for pr_open (08:10 -> claim 08:50, then 08:40 start has no prior close).
    assert by["in_progress"].seconds == 600 and by["in_progress"].visits == 1
    assert by["reviewing"].open_now and by["reviewing"].seconds == 600  # claimed 08:50, now 09:00
    assert by["needs_rework"].visits == 1 and by["needs_rework"].seconds == 600
    assert got.approx is True
    # PR view of the same item maps to the issue and sees its PR-less dispatch row
    pr = drill.pr_drill(ALPHA, 901, built.sources().db_path, None, now)
    assert isinstance(pr, drill.IssueDrill)
    assert pr.kind == "pr" and pr.issue == 900
    assert pr.timeline[0].label == "Dispatched to a worker"


def test_exact_path_once_lifecycle_transitions_exist(built) -> None:
    for ts, to in [
        ("2026-10-01T07:00:00Z", "agent:in-progress"),
        ("2026-10-01T07:30:00Z", "agent:pr-open"),
        ("2026-10-01T08:00:00Z", "agent:in-progress"),
        ("2026-10-01T08:10:00Z", "agent:pr-open"),
    ]:
        built.emit(built.alpha, ts, "lifecycle_transition", {"issue_number": 950, "to_state": to})
    assert rollup.run_rollup(built.sources(), NOW).errors == ()
    got = drill.issue_drill(ALPHA, 950, built.sources().db_path, None, NOW)
    assert isinstance(got, drill.IssueDrill)
    assert got.approx is False
    ip = next(s for s in got.stage_times if s.stage == "in_progress")
    assert ip.visits == 2 and ip.seconds == 1800 + 600  # accumulated across revisits


def test_escalations_show_reason_and_current_state_from_snapshot(built) -> None:
    snap = _snapshot(
        {
            "issues": [{"number": 2060, "title": "No-op cap", "labels": [L.human_needed]}],
            "prs": [{"number": 70, "issue_number": 2060, "is_draft": True, "reviewDecision": ""}],
        }
    )
    got = drill.issue_drill(ALPHA, 2060, built.sources().db_path, snap, NOW, labels=L)
    assert isinstance(got, drill.IssueDrill)
    esc = [e for e in got.timeline if e.category == "escalation"]
    assert [(e.label, e.detail) for e in esc] == [("Escalated", "no_op_rework_cap_exceeded")]
    assert got.current is not None
    assert (got.current.stage, got.current.title, got.current.pr, got.current.is_draft) == (
        "Human needed",
        "No-op cap",
        70,
        True,
    )
    # snapshot-only item: unknown to history, still known via the snapshot
    only = drill.pr_drill(ALPHA, 70, built.sources().db_path, snap, NOW)
    assert isinstance(only, drill.IssueDrill) and only.issue == 2060 and only.known


def test_unknown_item_is_a_value_not_an_error(built) -> None:
    got = drill.issue_drill(ALPHA, 424242, built.sources().db_path, None, NOW)
    assert isinstance(got, drill.IssueDrill) and not got.known and got.timeline == ()


@pytest.mark.parametrize(
    ("repo", "number"),
    [("../etc", 1), ("owner/alpha/x", 1), ("owner/..", 1), ("owner/alpha", 0),
     ("owner/alpha", -3), ("owner/alpha", True), ("owner/alpha", "7"), ("owner/alpha", 10**12)],
)  # fmt: skip
def test_invalid_input_returns_error_before_touching_db(tmp_path, repo, number) -> None:
    missing = tmp_path / "nope.db"
    for fn in (drill.issue_drill, drill.pr_drill):
        got = fn(repo, number, missing, None, NOW)
        assert got == drill.DrillError("invalid", got.message)  # type: ignore[union-attr]
    assert not missing.exists()  # a read never creates the database


def test_missing_dashboard_db_is_unavailable(tmp_path) -> None:
    got = drill.issue_drill(ALPHA, 1, tmp_path / "nope.db", None, NOW)
    assert isinstance(got, drill.DrillError) and got.code == "unavailable"


def test_naive_now_rejected(built) -> None:
    got = drill.issue_drill(ALPHA, 1, built.sources().db_path, None, datetime(2026, 10, 1))
    assert isinstance(got, drill.DrillError) and got.code == "invalid"


# --- repo -----------------------------------------------------------------------------------


def _model(*keys: str):
    repos = tuple(
        RepoRead(
            key=k,
            repo_root=f"C:/{k}",
            snapshot=SnapshotRead(
                NOW - timedelta(seconds=30),
                30.0,
                {"issues": [{"number": 1, "labels": [L.queued, L.ready]}], "workers": [{}]},
                None,
            ),
            reviewers_live=1,
            worker_cap=2,
            review_cap=3,
        )
        for k in keys
    )
    return build_now_model(SourcesRead(repos=repos, global_worker_cap=4), NOW)


def test_repo_drill_live_and_history(built) -> None:
    model = _model(ALPHA, BETA)
    got = drill.repo_drill(ALPHA, model, built.sources().db_path, tz=TZ)
    assert isinstance(got, drill.RepoDrill)
    assert got.freshness is not None and got.freshness.repo == ALPHA
    assert next(s for s in got.stages if s.name == "Queued").count == 1
    assert got.workers is not None and (got.workers.live, got.workers.cap) == (1, 2)
    assert got.reviewers is not None and got.reviewers.live == 1
    assert all(i.repo == ALPHA for i in got.needs_me)
    p = got.passes[0]
    assert (p.correlation_id, p.ok, p.error_count, p.merge_count, p.review_count) == (
        CID, True, 0, 2, 3
    )  # fmt: skip
    assert p.started_local == "2026-10-01T01:00:00-07:00"
    assert [m.issue for m in got.merges] == [2199]  # reconcile-detected merge, flagged
    assert got.merges[0].approx is True
    assert {e.issue for e in got.escalations} == {2060, 2200}
    assert got.history_error is None and got.history_from is not None
    # per-repo scoping: beta's escalations/merges never leak into alpha's page
    beta = drill.repo_drill(BETA, model, built.sources().db_path)
    assert isinstance(beta, drill.RepoDrill) and beta.passes == () and beta.merges == ()


def test_repo_drill_degrades_without_history(tmp_path) -> None:
    got = drill.repo_drill(ALPHA, _model(ALPHA), tmp_path / "nope.db")
    assert isinstance(got, drill.RepoDrill)
    assert got.history_error is not None and got.passes == () and got.stages


def test_repo_drill_errors(built) -> None:
    db = built.sources().db_path
    assert drill.repo_drill("bad slug", _model(ALPHA), db) == drill.DrillError(
        "invalid", "not a repo slug (expected owner/name)"
    )
    unknown = drill.repo_drill("owner/other", _model(ALPHA), db)
    assert isinstance(unknown, drill.DrillError) and unknown.code == "not_found"
    early = drill.repo_drill(ALPHA, None, db)
    assert isinstance(early, drill.DrillError) and early.code == "unavailable"
    lim = drill.repo_drill(ALPHA, _model(ALPHA), db, limit=0)
    assert isinstance(lim, drill.DrillError) and lim.code == "invalid"


def test_a_fresh_dashboard_db_carries_the_merge_scan_index(fleet) -> None:
    """merges_for's firsts scan is backed by issue_milestones (source, milestone)."""
    assert rollup.run_rollup(fleet.sources(), NOW).errors == ()
    conn = fleet.db()
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
    assert "issue_milestones_merge" in names


def test_merges_for_dedups_a_pr_only_row_onto_its_linked_issue(fleet) -> None:
    """A PR-only milestone shares its linked issue's key: two rows, one merge, first wins."""
    fleet.emit(
        fleet.beta, "2026-10-01T09:10:00Z", "finalize_externally_merged", {"pr_numbers": [555]}
    )
    fleet.emit(
        fleet.beta,
        "2026-10-01T09:20:00Z",
        "reconcile",
        {"kind": "merged_outside_orchestrator", "issue_number": 42, "pr_number": 555},
    )
    assert rollup.run_rollup(fleet.sources(), NOW).errors == ()
    got = merges_for(fleet.db(), BETA, 20, TZ)
    # without the linked-issue key both rows would survive; instead the earlier
    # PR-only row is the item's first signal
    assert [(m.ts, m.issue, m.pr) for m in got] == [("2026-10-01T09:10:00Z", None, 555)]


def test_merges_for_keeps_a_pr_with_no_linked_issue(fleet) -> None:
    fleet.emit(
        fleet.beta, "2026-10-01T09:10:00Z", "finalize_externally_merged", {"pr_numbers": [700]}
    )
    assert rollup.run_rollup(fleet.sources(), NOW).errors == ()
    got = merges_for(fleet.db(), BETA, 20, TZ)
    assert [(m.issue, m.pr, m.evidence) for m in got] == [
        (None, 700, "finalize_externally_merged")
    ]


def test_merges_for_drops_a_reconcile_backfill_burst(fleet) -> None:
    """BACKFILL_BURST reconcile-detected merges in one repo-hour are a catch-up pass."""
    for i in range(BACKFILL_BURST):
        fleet.emit(
            fleet.beta,
            f"2026-10-01T10:{i:02d}:00Z",
            "reconcile",
            {
                "kind": "merged_outside_orchestrator",
                "issue_number": 900 + i,
                "pr_number": 1900 + i,
            },
        )
    assert rollup.run_rollup(fleet.sources(), NOW).errors == ()
    assert merges_for(fleet.db(), BETA, 20, TZ) == ()


def test_merges_for_applies_limit_after_the_backfill_drop(fleet) -> None:
    """limit cuts what survives the drop — a burst newer than a real merge cannot starve it."""
    fleet.emit(
        fleet.beta,
        "2026-10-01T08:30:00Z",
        "merge_succeeded",
        {"issue_number": 7, "pr_number": 8},
    )
    for i in range(BACKFILL_BURST):
        fleet.emit(
            fleet.beta,
            f"2026-10-01T09:{i:02d}:00Z",
            "reconcile",
            {
                "kind": "merged_outside_orchestrator",
                "issue_number": 900 + i,
                "pr_number": 1900 + i,
            },
        )
    assert rollup.run_rollup(fleet.sources(), NOW).errors == ()
    got = merges_for(fleet.db(), BETA, 1, TZ)
    # newest-first with limit=1: if the drop ran after the limit the only returned row
    # would be a burst reconcile, leaving nothing
    assert [(m.issue, m.pr, m.evidence) for m in got] == [(7, 8, "merge_succeeded")]


# --- loop pass ------------------------------------------------------------------------------


def test_pass_drill_full_ordered_sequence_read_only(built) -> None:
    repos = sources.enumerate_repos(str(built.dir))
    got = drill.pass_drill(ALPHA, CID, repos, tz=TZ)
    assert isinstance(got, drill.PassDrill)
    assert [e.kind for e in got.events] == ["loop_started", "dispatch_deferred", "merge_failed"]
    assert [e.id for e in got.events] == sorted(e.id for e in got.events)
    assert got.events[0].ts_local == "2026-10-01T01:00:00-07:00"
    assert got.level_counts == (("error", 1), ("info", 1), ("warning", 1))
    assert [e.level for e in got.events] == ["info", "warning", "error"]
    assert (got.ok, got.elapsed_seconds, got.completed_at) == (
        True,
        102.17,
        "2026-10-01T08:01:42Z",
    )
    deferred = got.events[1].preview
    assert (
        "reason=cap" in deferred
        and "nested={1 keys}" in deferred
        and "items=[3 items]" in deferred
    )
    failed = got.events[2].preview
    assert len(failed) <= 200 and "\x1b" not in failed and "\n" not in failed
    assert "x" * 100 not in failed  # a long value is cut, never the whole blob


def test_pass_drill_does_not_write(built) -> None:
    repo = next(r for r in sources.enumerate_repos(str(built.dir)) if r.key == ALPHA)
    before = repo.events_db.stat().st_mtime_ns
    assert isinstance(drill.pass_drill(ALPHA, CID, [repo]), drill.PassDrill)
    assert repo.events_db.stat().st_mtime_ns == before


def test_pass_drill_errors(built, tmp_path) -> None:
    repos = sources.enumerate_repos(str(built.dir))
    for bad in ("", "a b", "x;DROP", "../x", "a" * 65, "id'--"):
        got = drill.pass_drill(ALPHA, bad, repos)
        assert isinstance(got, drill.DrillError) and got.code == "invalid"
    bad_slug = drill.pass_drill("../x/y", CID, repos)
    assert isinstance(bad_slug, drill.DrillError) and bad_slug.code == "invalid"
    unknown_repo = drill.pass_drill("owner/other", CID, repos)
    assert isinstance(unknown_repo, drill.DrillError) and unknown_repo.code == "not_found"
    none = drill.pass_drill(ALPHA, "deadbeef0000", repos)
    assert isinstance(none, drill.DrillError) and none.code == "not_found"
    ghost = sources.RepoSource(
        ALPHA, tmp_path, tmp_path, tmp_path / "s.json", tmp_path / "gone" / "events.db"
    )
    gone = drill.pass_drill(ALPHA, CID, [ghost])
    assert isinstance(gone, drill.DrillError) and gone.code == "unavailable"
    assert not (tmp_path / "gone").exists()


def test_pass_drill_truncates_at_cap(built) -> None:
    conn = sqlite3.connect(str(built.alpha / "events.db"))
    conn.executemany(
        "INSERT INTO events (ts, kind, payload, correlation_id, level)"
        " VALUES (?, 'bulk', '{}', 'bulk0000', 'info')",
        [("2026-10-01T08:00:00Z",)] * (MAX_PASS_EVENTS + 5),
    )
    conn.commit()
    conn.close()
    got = drill.pass_drill(ALPHA, "bulk0000", sources.enumerate_repos(str(built.dir)))
    assert isinstance(got, drill.PassDrill)
    assert got.truncated and len(got.events) == MAX_PASS_EVENTS


def test_payload_preview_shapes() -> None:
    assert payload_preview("{}") == ""
    assert payload_preview("not json") == "(unreadable payload)"
    assert payload_preview('["a"]') == "[1 items]"
    many = payload_preview("{" + ",".join(f'"k{i}": "{"v" * 30}"' for i in range(30)) + "}")
    assert len(many) <= 200 and many.endswith("\u2026")


def test_table_headers_right_align_over_numeric_columns() -> None:
    from charlie_work.dashboard.pages.drill_shell import table

    html = table("t", ("Pass", "Took", "Errors"), ['<tr><td>a</td><td class="num">1s</td>'
                 '<td class="num">0</td></tr>'], "")  # fmt: skip
    assert '<th scope="col">Pass</th>' in html
    assert '<th scope="col" class="num">Took</th>' in html
    assert '<th scope="col" class="num">Errors</th>' in html
