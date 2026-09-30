"""Issue #1681: a required check that is terminally absent because every
workflow run for the head completed with no jobs (GitHub rejected the
workflow file) must route to rework, not park in ``janitor_blocked`` forever.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from charlie_work.ci_absence import runs_terminally_without_jobs
from charlie_work.state import load_state

from test_stale_checks_retrigger import _app_with_conflict_and_missing_checks, _events

_REJECTED_RUN: dict[str, Any] = {"status": "completed", "conclusion": "failure", "jobs": []}


def test_completed_run_with_no_jobs_routes_to_rework(tmp_path: Path) -> None:
    app = _app_with_conflict_and_missing_checks(tmp_path, runs=[_REJECTED_RUN])

    result = app.review(456)

    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["status"] != "janitor_blocked"
    assert state["issues"]["123"]["status"] == "rework_requested"
    events = _events(state, "workflow_no_jobs")
    assert len(events) == 1
    assert events[0]["payload"]["head_sha"] == app.gh.prs[0]["headRefOid"]
    assert events[0]["payload"]["missing_checks"]
    assert result.data["decision"] == "request_changes"
    written = "".join(
        f.read_text(encoding="utf-8", errors="ignore") for f in tmp_path.rglob("*") if f.is_file()
    )
    assert "workflow file invalid: run completed with no jobs" in written
    # Retrigger cannot fix a rejected workflow file.
    assert app.gh.pr_close_calls == []


def test_in_progress_run_without_jobs_stays_pending(tmp_path: Path) -> None:
    """Mirror-image control: not yet completed => still pending, no terminal route."""
    app = _app_with_conflict_and_missing_checks(
        tmp_path, runs=[{"status": "in_progress", "conclusion": None, "jobs": []}]
    )

    result = app.review(456)

    state = load_state(app.paths.state_file)
    assert result.ok is False
    assert state["prs"]["456"]["status"] == "janitor_blocked"
    assert state["issues"]["123"]["status"] == "dispatched"
    assert _events(state, "workflow_no_jobs") == []


def test_no_jobs_event_deduped_per_head_but_route_repeats(tmp_path: Path) -> None:
    app = _app_with_conflict_and_missing_checks(tmp_path, runs=[_REJECTED_RUN])

    app.review(456)
    app.review(456)

    state = load_state(app.paths.state_file)
    assert len(_events(state, "workflow_no_jobs")) == 1
    assert state["prs"]["456"]["workflow_no_jobs_head"] == app.gh.prs[0]["headRefOid"]


@pytest.mark.parametrize(
    ("runs", "expected"),
    [
        ([], False),
        ([_REJECTED_RUN], True),
        ([{"status": "completed", "conclusion": "failure"}], True),
        ([{"status": "completed", "conclusion": "startup_failure"}], True),
        ([{"status": "completed", "conclusion": "success"}], False),
        ([{"status": "completed", "conclusion": "failure", "jobs": [{"id": 1}]}], False),
        ([_REJECTED_RUN, {"status": "queued", "conclusion": None}], False),
    ],
)
def test_runs_terminally_without_jobs(runs: list[dict[str, Any]], expected: bool) -> None:
    assert runs_terminally_without_jobs(runs) is expected
