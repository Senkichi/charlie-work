# ruff: noqa: F811  (the imported ``fleet`` fixture is re-bound as a test parameter)
"""Drill pages over corrupt or hostile stored rows: typed at the seam, escaped at the sink."""

from __future__ import annotations

import sqlite3
from datetime import timedelta

from _dashboard_rollup_fixtures import ALPHA, NOW, fleet  # noqa: F401

from charlie_work import instrumentation
from charlie_work.dashboard import drill, rollup
from charlie_work.dashboard.now_model import build_now_model
from charlie_work.dashboard.now_types import RepoRead, SourcesRead
from charlie_work.dashboard.pages.drill_repo import render_repo
from charlie_work.dashboard.read_model import ModelState
from charlie_work.dashboard.server_http import CSP
from charlie_work.dashboard.sources import SnapshotRead

HOSTILE = '<zzx onmouseover=alert(1)><meta http-equiv="refresh" content="0;url=x">'


def _model():
    snap = SnapshotRead(NOW - timedelta(seconds=30), 30.0, {"issues": [], "workers": []}, None)
    repo = RepoRead(key=ALPHA, repo_root="C:/a", snapshot=snap, reviewers_live=0,
                    worker_cap=1, review_cap=1)  # fmt: skip
    return build_now_model(SourcesRead(repos=(repo,), global_worker_cap=1), NOW)


def _plant(path, sql: str) -> None:
    conn = sqlite3.connect(path)
    try:
        with conn:
            conn.execute(sql, (HOSTILE, HOSTILE, HOSTILE))
    finally:
        conn.close()


def test_rollup_never_copies_a_text_count_into_dashboard_db(fleet) -> None:
    instrumentation.close_db(fleet.alpha / "state.json")
    _plant(
        fleet.alpha / "events.db",
        "UPDATE loop_passes SET error_count = ?, merge_count = ?, review_count = ?",
    )
    assert rollup.run_rollup(fleet.sources(), NOW).errors == ()
    rows = (
        fleet.db()
        .execute(
            "SELECT typeof(error_count), typeof(merge_count), typeof(review_count) FROM loop_passes"
        )
        .fetchall()
    )
    assert rows and all(t != "text" for row in rows for t in row)


def test_repo_page_escapes_and_types_a_text_count_already_in_dashboard_db(fleet) -> None:
    assert rollup.run_rollup(fleet.sources(), NOW).errors == ()
    db_path = fleet.sources().db_path
    _plant(db_path, "UPDATE loop_passes SET error_count = ?, merge_count = ?, review_count = ?")
    model = _model()
    got = drill.repo_drill(ALPHA, model, db_path)
    assert isinstance(got, drill.RepoDrill) and got.passes
    assert {(p.error_count, p.merge_count, p.review_count) for p in got.passes} == {(0, 0, 0)}
    html = render_repo(ModelState(model=model), got)
    assert "<zzx" not in html and "http-equiv" not in html


def test_csp_closes_the_directives_that_do_not_fall_back_to_default_src() -> None:
    directives = {d.strip().split()[0]: d.strip() for d in CSP.split(";")}
    assert directives["base-uri"] == "base-uri 'none'"
    assert directives["form-action"] == "form-action 'none'"


def test_batch_rework_pr_never_joins_another_issues_timeline(fleet) -> None:
    e = fleet.emit
    e(fleet.alpha, "2026-10-01T07:00:00Z", "worker_handoff_pr_opened",
      {"issue_number": 3010, "pr_number": 3050})  # fmt: skip
    e(fleet.alpha, "2026-10-01T07:01:00Z", "worker_handoff_pr_opened",
      {"issue_number": 3011, "pr_number": 3051})  # fmt: skip
    # one batch event, one representative pr_number (3010's), three issues
    e(fleet.alpha, "2026-10-01T07:30:00Z", "dispatch_rework",
      {"issue_numbers": [3010, 3011, 3012], "pr_number": 3050})  # fmt: skip
    assert rollup.run_rollup(fleet.sources(), NOW).errors == ()
    rows = (
        fleet.db()
        .execute(
            "SELECT issue, pr FROM issue_milestones WHERE event_kind = 'dispatch_rework'"
            " AND issue IN (3010, 3011, 3012) ORDER BY issue"
        )
        .fetchall()
    )
    assert rows == [(3010, None), (3011, None), (3012, None)]
    got = drill.issue_drill(ALPHA, 3011, fleet.sources().db_path, None, NOW)
    assert isinstance(got, drill.IssueDrill) and got.prs == (3051,)
    assert any(t.event_kind == "dispatch_rework" for t in got.timeline)  # still on 3011
