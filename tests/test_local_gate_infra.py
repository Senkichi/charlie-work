"""Issue #2127: the local merge gate separates infra outcomes from code defects.

A timeout or a summary-less suite death (externally killed runner) is host
weather, not something a worker can fix: it must relaunch (bounded) and never
touch the merge-rework counters. A failure with a pytest terminal summary is a
real code failure and still routes to rework unchanged.
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from charlie_work.local_gate_infra import (
    SuiteOutcome,
    classify_suite_outcome,
    derived_suite_timeout,
    effective_suite_timeout,
)
from charlie_work.state import load_state, load_state_locked, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

# The gate delegate may only be imported after ``charlie_work.workflow`` (#1798).
from charlie_work.orchestration.local_merge_gate import (  # noqa: E402
    LOCAL_SUITE_GATE_MAX_INFRA_RELAUNCHES,
)
from _local_gate_async_fixtures import (  # noqa: E402
    FAIL_SUITE,
    KILLED_SUITE,
    SLEEP_SUITE,
    _adopt_and_approve,
    _event_kinds,
    _events_of_kind,
    _gate_paths,
    _init_repo,
    _kill_claimed_gate,
    _lane_app,
    _lane_config,
    _make_branch,
    _wait_for_result,
)

# ---------------------------------------------------------------------------
# classify_suite_outcome
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tail",
    [
        "F.\n=== 2 failed, 10 passed in 3.2s ===\n",
        "...F [100%]\r\n3 failed, 2100 passed, 3 skipped in 2728.09s (0:45:28)\n",
        "3 failed, 2100 passed, 3 skipped, 3 deselected, 1321 warnings in 2728.09s (0:45:28)",
        "=== no tests ran in 0.01s ===",
        "===== short test summary info =====\nFAILED tests/test_x.py::test_y\n",
        "INTERNALERROR> Traceback (most recent call last):",
        "!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!",
        "!!!!!!!!!!!!!!!!!!! KeyboardInterrupt !!!!!!!!!!!!!!!!!!!",
        "ERROR collecting tests/test_x.py\n",
    ],
)
def test_summary_markers_classify_as_code_failure(tail: str) -> None:
    assert classify_suite_outcome(timed_out=False, tail=tail) is SuiteOutcome.CODE_FAILURE


@pytest.mark.parametrize(
    "tail",
    [
        "....... [ 12%]\r\n.........",
        "",
        "....... [ 12%]\r.......... [ 13%]\r",
    ],
)
def test_summary_less_tail_classifies_as_suite_killed(tail: str) -> None:
    assert classify_suite_outcome(timed_out=False, tail=tail) is SuiteOutcome.SUITE_KILLED


def test_timed_out_wins_even_with_a_summary_looking_tail() -> None:
    assert (
        classify_suite_outcome(timed_out=True, tail="=== 1 failed in 1.0s ===")
        is SuiteOutcome.SUITE_TIMED_OUT
    )


# ---------------------------------------------------------------------------
# derived_suite_timeout
# ---------------------------------------------------------------------------


def _ok(duration: float, ended: str) -> dict:
    return {"ok": True, "duration_seconds": duration, "ended_at": ended}


def test_fewer_than_three_samples_falls_back_to_default() -> None:
    assert derived_suite_timeout([]) == 3600
    assert derived_suite_timeout([_ok(9000, "2026-01-01T00:00:01Z")] * 2) == 3600


def test_median_times_four_is_used() -> None:
    results = [_ok(d, f"2026-01-01T00:00:0{i}Z") for i, d in enumerate((1500, 1650, 1800))]
    assert derived_suite_timeout(results) == 6600


def test_floor_and_ceiling_clamp() -> None:
    small = [_ok(10, f"2026-01-01T00:00:0{i}Z") for i in range(3)]
    huge = [_ok(50000, f"2026-01-01T00:00:0{i}Z") for i in range(3)]
    assert derived_suite_timeout(small) == 3600
    assert derived_suite_timeout(huge) == 10800


def test_only_the_ten_most_recent_ok_results_count() -> None:
    old = [_ok(50000, f"2025-01-01T00:00:{i:02d}Z") for i in range(10)]
    recent = [_ok(1500, f"2026-01-01T00:00:{i:02d}Z") for i in range(10)]
    assert derived_suite_timeout(old + recent) == 6000


def test_malformed_and_not_ok_results_are_ignored() -> None:
    junk = [
        {"ok": False, "duration_seconds": 50000, "ended_at": "2026-02-01T00:00:00Z"},
        {"ok": True, "ended_at": "2026-02-01T00:00:01Z"},
        {"ok": True, "duration_seconds": "9000", "ended_at": "2026-02-01T00:00:02Z"},
        {"ok": True, "duration_seconds": True, "ended_at": "2026-02-01T00:00:03Z"},
        {"ok": True, "duration_seconds": -5, "ended_at": "2026-02-01T00:00:04Z"},
    ]
    good = [_ok(1500, f"2026-01-01T00:00:0{i}Z") for i in range(3)]
    assert derived_suite_timeout(junk + good) == 6000
    assert derived_suite_timeout(junk) == 3600


def test_reader_skips_truncated_files_and_reads_real_results(tmp_path: Path) -> None:
    gate_root = tmp_path / "local-merge-gate"
    for pr, body in (
        (1, json.dumps(_ok(1500, "2026-01-01T00:00:01Z"))),
        (2, json.dumps(_ok(1500, "2026-01-01T00:00:02Z"))),
        (3, json.dumps(_ok(1500, "2026-01-01T00:00:03Z"))),
        (4, '{"ok": true, "duration_sec'),
    ):
        (gate_root / f"pr-{pr}").mkdir(parents=True)
        (gate_root / f"pr-{pr}" / "suite-result.json").write_text(body, encoding="utf-8")
    (gate_root / "pr-5").mkdir()  # no result file at all
    assert effective_suite_timeout(tmp_path) == 6000
    assert effective_suite_timeout(tmp_path / "missing") == 3600


# ---------------------------------------------------------------------------
# Gate integration (real git repo + real runner processes)
# ---------------------------------------------------------------------------


@pytest.fixture
def lane_repo() -> Path:
    return Path(tempfile.mkdtemp(prefix="cw-gate-infra-"))


def _setup(lane_repo: Path, suite: str) -> OrchestratorApp:
    _init_repo(lane_repo)
    issues_dir = lane_repo / "docs" / "issues"
    head = _make_branch(lane_repo, "agent/issue-7-x", "a.py", "a = 1\n")
    config = _lane_config(lane_repo, issues_dir, dispatch={"test_command": suite})
    app = _lane_app(lane_repo, issues_dir, config=config)
    _adopt_and_approve(app, issues_dir, 7, "agent/issue-7-x", head)
    return app


def _pr(app: OrchestratorApp) -> dict:
    return load_state_locked(app.paths.state_file)["prs"]["7"]


def _age_claim(app: OrchestratorApp, seconds: int) -> None:
    stale = (datetime.now(UTC) - timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["7"]["local_suite_started_at"] = stale
        save_state(app.paths.state_file, state)


def test_summary_less_death_relaunches_without_touching_rework(lane_repo: Path) -> None:
    app = _setup(lane_repo, KILLED_SUITE)
    assert app._local_merge_approved()[0]["outcome"] == "suite_launched"
    result = _wait_for_result(app, 7)
    assert result["ok"] is False

    try:
        results = app._local_merge_approved()

        assert results[0]["outcome"] == "suite_launched"
        pr = _pr(app)
        assert pr["status"] == "approved"
        assert pr["local_suite_infra_relaunch_count"] == 1
        assert not pr.get("local_suite_failed_rework_attempts")
        assert "routed_to" not in results[0]
        events = _events_of_kind(app, "local_suite_infra_relaunched")
        assert len(events) == 1
        payload = events[0]["payload"]
        assert payload["outcome"] == "suite_killed"
        assert payload["returncode"] == 1
        assert payload["relaunch_count"] == 1
        assert payload["pr_number"] == 7 and payload["issue_number"] == 7
        assert "local_suite_failed" not in _event_kinds(app)
    finally:
        _kill_claimed_gate(app, 7)


def test_summary_ful_failure_still_routes_to_rework(lane_repo: Path) -> None:
    app = _setup(lane_repo, FAIL_SUITE)
    app._local_merge_approved()
    _wait_for_result(app, 7)

    results = app._local_merge_approved()

    assert results[0]["outcome"] == "suite_failed"
    assert results[0]["routed_to"] == "rework"
    pr = _pr(app)
    assert pr["status"] == "rework_requested"
    assert pr["local_suite_failed_rework_attempts"] == 1
    assert not pr.get("local_suite_infra_relaunch_count")
    assert "local_suite_infra_relaunched" not in _event_kinds(app)


def test_in_flight_timeout_relaunches_and_records_effective_timeout(lane_repo: Path) -> None:
    app = _setup(lane_repo, SLEEP_SUITE)
    app._local_merge_approved()
    _age_claim(app, 3700)

    try:
        results = app._local_merge_approved()

        assert results[0]["outcome"] == "suite_launched"
        assert _pr(app)["status"] == "approved"
        assert not _pr(app).get("local_suite_failed_rework_attempts")
        timed_out = [
            e for e in _events_of_kind(app, "local_suite_result") if e["payload"].get("timed_out")
        ]
        assert timed_out[0]["payload"]["timeout_seconds"] == 3600
        relaunch = _events_of_kind(app, "local_suite_infra_relaunched")[0]["payload"]
        assert relaunch["outcome"] == "suite_timed_out"
    finally:
        _kill_claimed_gate(app, 7)


def test_derived_timeout_spares_a_run_within_the_longer_limit(lane_repo: Path) -> None:
    """Three 1650s ok results derive a 6600s limit: a 3700s-old suite is NOT killed."""
    app = _setup(lane_repo, SLEEP_SUITE)
    for pr in (101, 102, 103):
        paths = _gate_paths(app, pr)
        paths.gate_dir.mkdir(parents=True)
        paths.result.write_text(json.dumps(_ok(1650, f"2026-01-01T00:00:0{pr - 100}Z")))
    app._local_merge_approved()
    _age_claim(app, 3700)

    try:
        results = app._local_merge_approved()

        assert results[0]["outcome"] == "suite_running"
        assert not _events_of_kind(app, "local_suite_infra_relaunched")
    finally:
        _kill_claimed_gate(app, 7)


def test_infra_relaunch_bound_exhausted_escalates_with_infra_reason(lane_repo: Path) -> None:
    app = _setup(lane_repo, KILLED_SUITE)
    app._local_merge_approved()
    for relaunch in range(1, LOCAL_SUITE_GATE_MAX_INFRA_RELAUNCHES + 1):
        _wait_for_result(app, 7)
        assert app._local_merge_approved()[0]["outcome"] == "suite_launched"
        assert _pr(app)["local_suite_infra_relaunch_count"] == relaunch
        # Drop the previous result so the next wait observes the NEW run's.
        _gate_paths(app, 7).result.unlink(missing_ok=True)

    _wait_for_result(app, 7)
    results = app._local_merge_approved()

    assert results[0]["outcome"] == "error"
    assert "suite_killed" in results[0]["detail"]
    pr = _pr(app)
    assert pr["status"] == "escalated"
    assert pr["escalation_reason"] == "local_merge_gate_infra_exhausted"
    assert not pr.get("local_suite_failed_rework_attempts")
    failed = _events_of_kind(app, "local_merge_failed")
    assert failed[-1]["payload"]["reason"] == "local_merge_gate_infra_exhausted"
    assert len(_events_of_kind(app, "local_suite_infra_relaunched")) == (
        LOCAL_SUITE_GATE_MAX_INFRA_RELAUNCHES
    )


def test_unescalate_reset_map_clears_the_infra_counter() -> None:
    from charlie_work.unescalate_reset_fields import (
        REWORK_BUDGET_RESET_BY_ESCALATION_REASON,
    )

    counters, _ = REWORK_BUDGET_RESET_BY_ESCALATION_REASON["local_merge_gate_infra_exhausted"]
    assert "local_suite_infra_relaunch_count" in counters


# ---------------------------------------------------------------------------
# Durable-sample timeout (events.db, not pruned per-PR result dirs)
# ---------------------------------------------------------------------------


def test_timeout_derives_from_durable_events_without_result_files(tmp_path: Path) -> None:
    from charlie_work.instrumentation import log_event

    state_path = tmp_path / "state.json"
    for pr in (1, 2, 3):
        log_event(
            state_path,
            "local_suite_ok",
            {"pr_number": pr, "duration_seconds": 1500},
        )
    dispatches = tmp_path / "dispatches"  # no pr-* result dirs at all
    dispatches.mkdir()
    assert effective_suite_timeout(dispatches, state_path) == 6000


def test_timeout_degrades_to_default_on_missing_or_corrupt_db(tmp_path: Path) -> None:
    dispatches = tmp_path / "dispatches"
    dispatches.mkdir()
    assert effective_suite_timeout(dispatches, tmp_path / "nodir" / "state.json") == 3600
    corrupt = tmp_path / "bad"
    corrupt.mkdir()
    (corrupt / "events.db").write_bytes(b"not a sqlite database" * 50)
    assert effective_suite_timeout(dispatches, corrupt / "state.json") == 3600
