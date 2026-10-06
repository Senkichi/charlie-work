# ruff: noqa: F811  (the imported ``fleet`` fixture is re-bound as a test parameter)
"""Tests for the dashboard rollup (``dashboard/rollup.py``); fixtures in ``_dashboard_rollup_fixtures``."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import timedelta

import pytest
from _dashboard_rollup_fixtures import (  # noqa: F401  (fleet is a pytest fixture)
    ALPHA,
    BETA,
    NOW,
    _all,
    _facts,
    fleet,
)

from charlie_work.dashboard import history_data, rollup
from charlie_work.dashboard.rollup_derive import (
    HANDLERS,
    KNOWN_IGNORED,
    is_classified,
    reason_group,
)
from charlie_work.dashboard.rollup_schema import SCHEMA_VERSION


def test_facts_exact_values(fleet) -> None:
    result = rollup.run_rollup(fleet.sources(), NOW)
    assert result.errors == ()
    by = {s.source: s for s in result.sources}
    # handled-kind events only: unhandled supervisor_started and the 2 noise rows never count
    assert (by[ALPHA].ingested, by[BETA].ingested, by["fleet"].ingested) == (18, 2, 4)
    db = fleet.db()

    assert _all(
        db,
        "SELECT source, ts, repo, live_sessions, fleet_live_sessions, concurrency_limit, fleet_concurrency_limit, available_slots, dispatch_limit, clamped, deferred_by_concurrency, launched, open_total, dispatchable, active_label, missing_ready, parked_unready, terminal_label, blocked_by_open_dependency, operator_claimed FROM pass_samples ORDER BY ts",
    ) == [
        (
            ALPHA,
            "2026-10-01T08:00:00Z",
            ALPHA,
            1,
            3,
            5,
            4,
            4,
            1,
            0,
            4,
            2,
            47,
            5,
            3,
            18,
            7,
            15,
            0,
            0,
        ),
        (
            BETA,
            "2026-10-01T09:00:00Z",
            BETA,
            1,
            3,
            5,
            4,
            4,
            1,
            0,
            None,
            0,
            47,
            5,
            3,
            18,
            7,
            15,
            0,
            0,
        ),
    ]
    assert _all(
        db,
        "SELECT source, available_slots, live_reviews, review_limit, launched, failed, quota_hit FROM review_samples",
    ) == [(ALPHA, 6, 0, 6, 2, 1, 0)]
    assert _all(
        db,
        "SELECT source, ts, issue, pr, milestone, event_kind, approx FROM issue_milestones WHERE source = ? ORDER BY ts, seq",
        ALPHA,
    ) == [
        (ALPHA, "2026-10-01T08:00:00Z", 2226, None, "dispatched", "dispatch", 0),
        (ALPHA, "2026-10-01T08:00:00Z", 2227, None, "dispatched", "dispatch", 0),
        (ALPHA, "2026-10-01T08:01:00Z", 2195, 2197, "rework_dispatched", "dispatch_rework", 0),
        (
            ALPHA,
            "2026-10-01T08:02:00Z",
            2185,
            2188,
            "pr_opened_by_worker",
            "worker_handoff_pr_opened",
            0,
        ),
        (
            ALPHA,
            "2026-10-01T08:03:00Z",
            1939,
            1940,
            "pr_opened_by_salvage",
            "orphaned_worker_opened_pr",
            1,
        ),
        (ALPHA, "2026-10-01T08:04:00Z", None, 2214, "review_claimed", "review_dispatch_claim", 0),
        (ALPHA, "2026-10-01T08:04:00Z", None, 2215, "review_claimed", "review_dispatch_claim", 0),
        (ALPHA, "2026-10-01T08:06:00Z", 2199, 2208, "verdict_approved", "record_review", 0),
        (ALPHA, "2026-10-01T08:07:00Z", 2200, 2209, "verdict_request_changes", "record_review", 0),
        (ALPHA, "2026-10-01T08:07:00Z", 2200, 2209, "escalated", "record_review", 0),
        (ALPHA, "2026-10-01T08:08:00Z", 2199, 2208, "merged", "reconcile", 1),
        (ALPHA, "2026-10-01T08:11:00Z", 2060, None, "escalated", "session_failed_escalated", 0),
        (ALPHA, "2026-10-01T08:12:00Z", 1808, None, "unescalated", "unescalate", 0),
    ]
    assert _all(
        db, "SELECT source, issue, failure_kind, worker_health FROM worker_exits ORDER BY ts"
    ) == [
        (ALPHA, 2195, "stalled", "DEAD"),
        (BETA, 77, None, "DEAD"),
    ]
    assert _all(db, "SELECT issue, pr, event_kind, reason FROM escalations ORDER BY ts") == [
        (2200, 2209, "record_review", "review_verdict_escalated"),
        (2060, None, "session_failed_escalated", "no_op_rework_cap_exceeded"),
        (1808, None, "unescalate", "dead_dispatched_worker_reap"),
    ]
    assert _all(
        db,
        "SELECT pr, reason, reason_group, exit_code, turn_count, tool_call_count FROM verdict_missed ORDER BY ts",
    ) == [
        (2087, "launch_failed", "launch_failed", 0, 0, 0),
        (2088, "PR #2088 is MERGED on GitHub", "pr #", None, None, None),
        (2089, "PR #2089 is MERGED on GitHub", "pr #", None, None, None),
    ]
    assert _all(
        db,
        "SELECT source, ts, target_repo, capacity, demand, running, target, budget, oldest_queued_seconds FROM runner_samples ORDER BY target_repo",
    ) == [
        ("fleet", "2026-10-01T10:00:00Z", "Senkichi/fresh-eyes", 1, 0, 1, 1, 8, 0),
        ("fleet", "2026-10-01T10:00:00Z", "Senkichi/swole", 5, 2, 2, 3, 8, 30),
    ]
    assert _all(
        db,
        "SELECT job_id, source, src_id, queue_wait_seconds, execution_seconds, wall_seconds FROM job_observations ORDER BY job_id",
    ) == [
        ("j1", "fleet", 3, 2.0, 2.0, 40.0),
        ("j2", "fleet", 2, 2.0, 2.0, 40.0),
    ]
    assert _all(db, "SELECT source, ok, changed, from_sha, to_sha, error FROM deploys") == [
        (ALPHA, 1, 1, "32e8", "d6c3", None)
    ]
    assert _all(db, "SELECT event_kind, until, detail FROM throttles") == [
        ("review_quota_exhausted", "2026-10-01T13:00:00Z", "stalled_review_sweep")
    ]
    assert _all(
        db, "SELECT source, event_kind, requested, granted, reason FROM capped_demand ORDER BY ts"
    ) == [
        (ALPHA, "dispatch_backpressure", 3, 0, "host_load"),
        ("fleet", "runner_capacity_starved", 7, 5, None),
    ]
    assert _all(
        db,
        "SELECT source, correlation_id, started_at, completed_at, ok, elapsed_seconds, merge_count, review_count, sink_population FROM loop_passes",
    ) == [
        (ALPHA, "ed62ee91e2e5", "2026-10-01T08:00:00Z", "2026-10-01T08:01:42Z", 1, 102.17, 2, 3, 9)
    ]
    # coverage is honest about everything in the source, including noise and unhandled kinds
    assert _all(
        db,
        "SELECT kind, first_ts, last_ts, n FROM coverage WHERE source = ? AND kind IN ('*', 'unauthorized_merge_queue_sync_covered', 'dispatch')",
        ALPHA,
    ) == [
        ("*", "2026-10-01T08:00:00Z", "2026-10-01T10:06:00Z", 23),
        ("dispatch", "2026-10-01T08:00:00Z", "2026-10-01T08:00:00Z", 1),
        (
            "unauthorized_merge_queue_sync_covered",
            "2026-10-01T08:09:01Z",
            "2026-10-01T08:09:02Z",
            2,
        ),
    ]
    assert _all(
        db, "SELECT kind, n FROM coverage WHERE source='fleet' AND kind = 'supervisor_started'"
    ) == [("supervisor_started", 1)]


def test_noise_and_per_repo_global_copies_excluded(fleet) -> None:
    rollup.run_rollup(fleet.sources(), NOW)
    db = fleet.db()
    assert _all(db, "SELECT COUNT(*) FROM runner_samples WHERE source != 'fleet'") == [(0,)]
    assert _all(db, "SELECT COUNT(*) FROM runner_samples") == [(2,)]  # global only, not 4
    assert _all(db, "SELECT COUNT(*) FROM job_observations") == [(2,)]  # j1 seen 3x, counted once
    for table in ("issue_milestones", "escalations"):
        assert _all(
            db,
            f"SELECT COUNT(*) FROM {table} WHERE event_kind IN ('unauthorized_merge_queue_sync_covered') OR ts = '2026-10-01T08:09:00Z'",
        ) == [(0,)]
    # source attribution ignores the (wrong) repo column logged with every event
    assert {r[0] for r in _all(db, "SELECT DISTINCT repo FROM pass_samples")} == {ALPHA, BETA}


def test_second_run_is_idempotent(fleet) -> None:
    rollup.run_rollup(fleet.sources(), NOW)
    before = _facts(fleet.db())
    result = rollup.run_rollup(fleet.sources(), NOW)
    assert result.errors == ()
    assert result.ingested == 0
    assert {s.source: s.rederived for s in result.sources} == {ALPHA: 18, BETA: 2, "fleet": 4}
    assert _facts(fleet.db()) == before  # re-derived window leaves no duplicates


def test_new_events_ingest_incrementally(fleet) -> None:
    rollup.run_rollup(fleet.sources(), NOW)
    fleet.emit(
        fleet.beta,
        "2026-10-01T11:00:00Z",
        "session_exited",
        {"failure_kind": "rate_limited", "issue_number": 78, "worker_health": "DEAD"},
    )
    result = rollup.run_rollup(fleet.sources(), NOW + timedelta(minutes=5))
    assert {s.source: s.ingested for s in result.sources} == {ALPHA: 0, BETA: 1, "fleet": 0}
    assert _all(
        fleet.db(),
        "SELECT issue, failure_kind FROM worker_exits WHERE source = ? ORDER BY ts",
        BETA,
    ) == [(77, None), (78, "rate_limited")]


def test_rebuild_from_scratch_reproduces_identical_facts(fleet) -> None:
    rollup.run_rollup(fleet.sources(), NOW)
    before = _facts(fleet.db())
    fleet.release()
    fleet.sources().db_path.unlink()
    result = rollup.run_rollup(fleet.sources(), NOW)
    assert result.ingested == 24
    assert _facts(fleet.db()) == before


def test_rederive_window_drops_deduped_rows_but_keeps_old_ones(fleet) -> None:
    rollup.run_rollup(fleet.sources(), NOW)
    fleet.close()
    conn = sqlite3.connect(fleet.alpha / "events.db")
    # an old (outside the 6h window) and a recent (inside) event vanish from the source
    conn.execute("DELETE FROM events WHERE kind = 'dispatch'")  # 08:00, 4h before NOW: inside
    conn.execute("DELETE FROM events WHERE kind = 'unescalate'")
    conn.commit()
    conn.close()
    later = NOW + timedelta(hours=3)  # window now starts 09:00Z; both deleted rows are older
    rollup.run_rollup(fleet.sources(), later)
    assert _all(fleet.db(), "SELECT COUNT(*) FROM pass_samples WHERE source = ?", ALPHA) == [
        (1,)
    ]  # stale, kept
    rollup.run_rollup(fleet.sources(), NOW)  # window from 06:00Z reaches 08:00Z rows
    db = fleet.db()
    assert _all(db, "SELECT COUNT(*) FROM pass_samples WHERE source = ?", ALPHA) == [(0,)]
    assert _all(db, "SELECT COUNT(*) FROM escalations WHERE event_kind = 'unescalate'") == [(0,)]


def test_schema_version_mismatch_drops_and_rebuilds(fleet) -> None:
    rollup.run_rollup(fleet.sources(), NOW)
    before = _facts(fleet.db())
    fleet.release()
    db = fleet.db()
    db.execute(
        "UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(SCHEMA_VERSION + 1),)
    )
    db.execute(
        "INSERT INTO worker_exits (source, src_id, seq, ts, repo) VALUES ('ghost', 1, 0, 't', 'ghost')"
    )
    db.commit()
    db.close()
    result = rollup.run_rollup(fleet.sources(), NOW)
    assert result.db_rebuilt is True
    assert result.ingested == 24
    assert _facts(fleet.db()) == before  # ghost row gone, facts reproduced


def test_missing_source_is_an_error_value_not_a_failure(fleet) -> None:
    fleet.close()
    (fleet.beta / "events.db").unlink(missing_ok=True)
    for suffix in ("-wal", "-shm"):
        (fleet.beta / f"events.db{suffix}").unlink(missing_ok=True)
    result = rollup.run_rollup(fleet.sources(), NOW)
    by = {s.source: s for s in result.sources}
    assert by[BETA].error is not None and by[BETA].error.startswith("missing:")
    assert (by[ALPHA].ingested, by["fleet"].ingested) == (18, 4)
    assert len(result.errors) == 1


def test_replaced_source_db_is_rebuilt(fleet) -> None:
    rollup.run_rollup(fleet.sources(), NOW)
    db = fleet.db()
    db.execute("UPDATE watermarks SET max_id = 9999 WHERE source = ?", (BETA,))
    db.commit()
    db.close()
    result = rollup.run_rollup(fleet.sources(), NOW)
    by = {s.source: s for s in result.sources}
    assert by[BETA].rebuilt is True and by[BETA].ingested == 2
    assert _all(fleet.db(), "SELECT COUNT(*) FROM worker_exits WHERE source = ?", BETA) == [(1,)]


@pytest.mark.parametrize(
    ("reason", "group"),
    [
        ("died_mid_session", "died_mid_session"),
        ("launch_failed", "launch_failed"),
        ("PR #2087 is MERGED on GitHub", "pr #"),
        ("escalated: max attempts 3", "escalated"),
        ("", None),
        (None, None),
    ],
)
def test_reason_group(reason, group) -> None:
    assert reason_group(reason) == group


def test_reaper_sweep_summaries_expand_into_per_issue_milestones(fleet, monkeypatch) -> None:
    """The real reaper writer folds same-kind events into ``<kind>_sweep`` (numbers only);
    the rollup must still give every issue in the batch its PR-open milestone."""
    from charlie_work import instrumentation
    from charlie_work.stalled_review_reap import _append_sweep_events
    from charlie_work.write_gate import WriteGate

    gate = WriteGate(dry_run=False, state_path=fleet.alpha / "state.json", repo=ALPHA)
    kind = "orphaned_worker_advanced_to_pr_open"
    batches = [
        (
            "2026-10-01T10:00:00Z",
            [{"issue_number": 11, "pr_number": 21}, {"issue_number": 12, "pr_number": 22}],
        ),
        ("2026-10-01T10:05:00Z", [{"issue_number": 13, "pr_number": 23}]),
        ("2026-10-01T10:10:00Z", [{"issue_number": 14}, {"issue_number": 15}]),
    ]
    for ts, payloads in batches:
        monkeypatch.setattr(instrumentation, "_now_iso", lambda ts=ts: ts)
        _append_sweep_events({}, [(kind, p) for p in payloads], write_gate=gate)
    batches_alt = [{"issue_number": 16}, {"issue_number": 17}]
    monkeypatch.setattr(instrumentation, "_now_iso", lambda: "2026-10-01T10:15:00Z")
    _append_sweep_events(
        {}, [("worker_handoff_pr_opened", p) for p in batches_alt], write_gate=gate
    )

    result = rollup.run_rollup(fleet.sources(), NOW)
    assert result.errors == ()
    rows = _all(
        fleet.db(),
        "SELECT issue, pr, milestone, event_kind, approx FROM issue_milestones "
        "WHERE issue BETWEEN 11 AND 17 ORDER BY issue",
    )
    assert rows == [
        (11, None, "pr_open_after_dead_worker", f"{kind}_sweep", 1),
        (12, None, "pr_open_after_dead_worker", f"{kind}_sweep", 1),
        (13, 23, "pr_open_after_dead_worker", kind, 1),
        (14, None, "pr_open_after_dead_worker", f"{kind}_sweep", 1),
        (15, None, "pr_open_after_dead_worker", f"{kind}_sweep", 1),
        (16, None, "pr_opened_by_worker", "worker_handoff_pr_opened_sweep", 0),
        (17, None, "pr_opened_by_worker", "worker_handoff_pr_opened_sweep", 0),
    ]


def test_sweep_expansion_only_wraps_ref_only_handlers() -> None:
    from charlie_work.dashboard.rollup_derive import HANDLERS, REF_ONLY, SWEEP_SUFFIX, derive_event

    sweeps = {k for k in HANDLERS if k.endswith(SWEEP_SUFFIX)}
    assert sweeps == {k + SWEEP_SUFFIX for k in REF_ONLY}
    ev = {
        "id": 1,
        "ts": "2026-10-01T10:00:00Z",
        "kind": "orphaned_worker_opened_pr_sweep",
        "issue_number": None,
        "pr_number": None,
        "payload": {"count": 2, "pr_numbers": [40, 41]},
    }
    rows = derive_event(ALPHA, ev)
    assert [(c["issue"], c["pr"], c["seq"]) for _t, c in rows] == [(None, 40, 0), (None, 41, 1)]


def test_unhandled_kind_is_counted_unclassified_per_source(fleet) -> None:
    """#2269: a kind with neither a handler nor a KNOWN_IGNORED entry must surface.

    Emitted through the real ``log_event`` (the fixture's writer); before this
    change it produced no fact rows and no signal. Now the rollup counts it per
    source and persists the set on ``meta`` for the History page.
    """
    fleet.emit(fleet.alpha, "2026-10-01T10:30:00Z", "brand_new_unclassified_kind", {"x": 1})
    fleet.emit(fleet.alpha, "2026-10-01T10:31:00Z", "brand_new_unclassified_kind", {"x": 2})
    result = rollup.run_rollup(fleet.sources(), NOW)
    assert result.errors == ()
    by = {s.source: s for s in result.sources}
    assert by[ALPHA].unclassified == {"brand_new_unclassified_kind": 2}
    # known-ignored (supervisor_started, unauthorized_merge_queue_sync_covered) and
    # handled kinds never land in the count
    assert by[BETA].unclassified == {}
    assert by["fleet"].unclassified == {}
    stored = _all(fleet.db(), "SELECT value FROM meta WHERE key = 'unclassified_kinds'")
    assert json.loads(stored[0][0]) == {ALPHA: {"brand_new_unclassified_kind": 2}}


def test_unclassified_warning_fires_only_when_the_set_changes(fleet, caplog) -> None:
    """#2269: each rollup pass warns when the unclassified-kind set changes -- a new
    kind appearing, one disappearing -- and stays quiet while it is stable."""
    with caplog.at_level(logging.WARNING, "charlie_work.dashboard"):
        rollup.run_rollup(fleet.sources(), NOW)  # nothing unclassified yet
        fleet.emit(fleet.alpha, "2026-10-01T10:30:00Z", "brand_new_unclassified_kind", {})
        rollup.run_rollup(fleet.sources(), NOW)
    assert "1 event kind(s) not interpreted: brand_new_unclassified_kind" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, "charlie_work.dashboard"):
        rollup.run_rollup(fleet.sources(), NOW)  # same set: quiet
    assert "not interpreted" not in caplog.text
    # a previously unclassified kind disappearing is a set change too: warn once
    fleet.close()
    conn = sqlite3.connect(fleet.alpha / "events.db")
    conn.execute("DELETE FROM events WHERE kind = 'brand_new_unclassified_kind'")
    conn.commit()
    conn.close()
    caplog.clear()
    with caplog.at_level(logging.WARNING, "charlie_work.dashboard"):
        rollup.run_rollup(fleet.sources(), NOW)
    assert "0 event kind(s) not interpreted: " in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, "charlie_work.dashboard"):
        rollup.run_rollup(fleet.sources(), NOW)  # still empty: quiet again
    assert "not interpreted" not in caplog.text


def test_unclassified_kinds_survive_a_transient_source_error(fleet, caplog) -> None:
    """#2269: an errored source keeps its last recorded unclassified kinds.

    A pass that cannot open a source's events.db must not read as "that source
    has no unclassified kinds": the stored entry carries forward, so neither the
    error pass nor the recovery pass fires the set-changed warning, and the
    History page note keeps naming the kind throughout.
    """
    fleet.emit(fleet.alpha, "2026-10-01T10:30:00Z", "brand_new_unclassified_kind", {"x": 1})
    rollup.run_rollup(fleet.sources(), NOW)  # pass 1: records the kind
    stored = _all(fleet.db(), "SELECT value FROM meta WHERE key = 'unclassified_kinds'")
    assert json.loads(stored[0][0]) == {ALPHA: {"brand_new_unclassified_kind": 1}}

    fleet.close()
    backups: dict[str, bytes] = {}
    for suffix in ("", "-wal", "-shm"):
        path = fleet.alpha / f"events.db{suffix}"
        if path.is_file():
            backups[suffix] = path.read_bytes()
            path.unlink()
    caplog.clear()
    with caplog.at_level(logging.WARNING, "charlie_work.dashboard"):
        errored = rollup.run_rollup(fleet.sources(), NOW)  # pass 2: source unopenable
    alpha_error = {s.source: s for s in errored.sources}[ALPHA].error
    assert alpha_error is not None and alpha_error.startswith("missing:")
    assert "not interpreted" not in caplog.text  # a transient error is not a set change
    stored = _all(fleet.db(), "SELECT value FROM meta WHERE key = 'unclassified_kinds'")
    assert json.loads(stored[0][0]) == {ALPHA: {"brand_new_unclassified_kind": 1}}
    view = history_data.load_tab(fleet.sources().db_path, "flow", "7d", NOW)
    assert isinstance(view, history_data.HistoryView)
    assert view.unclassified_kinds == ("brand_new_unclassified_kind",)

    for suffix, blob in backups.items():
        (fleet.alpha / f"events.db{suffix}").write_bytes(blob)
    caplog.clear()
    with caplog.at_level(logging.WARNING, "charlie_work.dashboard"):
        recovered = rollup.run_rollup(fleet.sources(), NOW)  # pass 3: source back
    assert {s.source: s for s in recovered.sources}[ALPHA].error is None
    assert "not interpreted" not in caplog.text  # recovery is not a set change either
    stored = _all(fleet.db(), "SELECT value FROM meta WHERE key = 'unclassified_kinds'")
    assert json.loads(stored[0][0]) == {ALPHA: {"brand_new_unclassified_kind": 1}}


# Issue #2459: the 25 kinds the live "not interpreted" warning listed on 2026-10-06.
_ISSUE_2459_KINDS = (
    "cross_family_regen_not_reached",
    "cross_family_report_regen_exhausted",
    "cross_family_report_regen_forced",
    "cross_family_verdict_abandoned",
    "cross_family_verdict_head_indeterminate",
    "cross_family_verdict_unparseable",
    "dead_dispatched_throttle_rearmed_sweep",
    "dead_dispatched_worker_reaped_sweep",
    "label_ensure_incomplete",
    "label_ensure_ok",
    "operator_local_park",
    "operator_orphan_push_completed",
    "operator_orphan_requeue",
    "operator_probe_advanced",
    "operator_queue_depth",
    "operator_reviewer_quota_cleared",
    "operator_rework_rearmed",
    "operator_state_correction",
    "orphaned_worker_recovered_sweep",
    "review_packet_discarded_head_moved",
    "rework_label_skipped_issue_closed",
    "salvage_push_failed_sweep",
    "session_failed_relabeled_sweep",
    "unauthorized_merge_ack_revoked",
    "worker_token_missing",
)


@pytest.mark.parametrize("kind", _ISSUE_2459_KINDS)
def test_issue_2459_kind_is_classified(kind: str) -> None:
    assert is_classified(kind)


def test_ignored_sweep_variant_inherits_its_sibling_classification() -> None:
    """A ``<kind>_sweep`` of an ignored kind is classified without its own entry."""
    for base in (
        "dead_dispatched_worker_reaped",
        "salvage_push_failed",
        "orphaned_worker_recovered",
    ):
        assert base in KNOWN_IGNORED
        assert base + "_sweep" not in KNOWN_IGNORED
        assert is_classified(base + "_sweep")
        assert base + "_sweep" not in HANDLERS  # ignored, not silently handled


def test_sweep_of_unknown_kind_stays_unclassified() -> None:
    assert not is_classified("brand_new_unclassified_kind_sweep")
    assert not is_classified("brand_new_unclassified_kind")


def test_issue_2459_kinds_are_not_counted_unclassified_by_the_rollup(fleet) -> None:
    for n, kind in enumerate(_ISSUE_2459_KINDS):
        fleet.emit(fleet.alpha, f"2026-10-01T10:{n:02d}:00Z", kind, {"issue_numbers": [1]})
    result = rollup.run_rollup(fleet.sources(), NOW)
    assert result.errors == ()
    assert {s.source: s for s in result.sources}[ALPHA].unclassified == {}
