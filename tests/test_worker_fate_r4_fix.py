"""wf-r4 fixes: one #2010 permission-denial gate for every blocked escalation,
and rule 1's stale-evidence event for a dropped stale terminal record."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _orphan_sweep_fixtures import _dead_worker_rework_bed, _run_orphan_sweep, _write_outcome

from charlie_work import worker_fate
from charlie_work.instrumentation import query_events
from charlie_work.orphaned_worker_sweep import maybe_reap_dead_dispatched_worker
from charlie_work.process_utils import write_worker_terminal_status
from charlie_work.state import load_state, parse_iso_timestamp

DENIAL = "Bash was denied. If you approve command execution, I can finish these steps."


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _blocked_payload(detail: str) -> dict[str, Any]:
    return {
        "outcome": "blocked",
        "reason_kind": "other",
        "detail": detail,
        "push_succeeded": True,
        "pr_created": False,
        "head_sha": "abc123",
    }


def _assert_escalation(
    tmp_path: Path, bed: tuple[Any, ...], *, detail: str, escalated: bool
) -> None:
    config, paths, fake_gh, _dispatched_at = bed
    _write_outcome(paths, tmp_path, _blocked_payload(detail), mtime=datetime.now(UTC))

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    declared = [e for e in state.get("events", []) if e.get("kind") == "worker_declared_blocked"]
    if escalated:
        assert entry["status"] == "escalated"
        assert entry["escalation_reason"] == "worker_declared_blocked"
        assert declared
    else:
        assert entry.get("escalation_reason") != "worker_declared_blocked"
        assert entry["status"] != "escalated"
        assert not declared
        assert (207, config.labels.operator_queue) not in fake_gh.labels_added


@pytest.mark.parametrize(
    ("decision", "pr_state_status"),
    [("request_changes", None), ("approved", "rework_requested")],
)
def test_with_pr_permission_denial_blocked_outcome_is_not_operator_escalated(
    tmp_path: Path, decision: str, pr_state_status: str | None
) -> None:
    """Issue #2010, with-PR lane (both sweep sites): the headless
    permission-denial signature is a worker-config defect, not a blocked task,
    so it takes the ordinary reset path exactly as on origin/main."""
    bed = _dead_worker_rework_bed(tmp_path, decision=decision, pr_state_status=pr_state_status)
    _assert_escalation(tmp_path, bed, detail=DENIAL, escalated=False)


@pytest.mark.parametrize(
    ("decision", "pr_state_status"),
    [("request_changes", None), ("approved", "rework_requested")],
)
def test_with_pr_genuine_blocked_outcome_still_escalates(
    tmp_path: Path, decision: str, pr_state_status: str | None
) -> None:
    """Positive control for the test above: same bed, non-denial detail."""
    bed = _dead_worker_rework_bed(tmp_path, decision=decision, pr_state_status=pr_state_status)
    _assert_escalation(tmp_path, bed, detail="needs a human to disambiguate", escalated=True)


def _write_stale_terminal(tmp_path: Path) -> Path:
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    write_worker_terminal_status(
        sessions_dir / "issue-207.claude-code.terminal.json",
        pid=4242,
        exit_code=0,
        started_at=_iso(now - timedelta(hours=3, minutes=5)),
        ended_at=_iso(now - timedelta(hours=3)),
        duration_seconds=300.0,
    )
    return sessions_dir


def test_sweep_emits_stale_evidence_event_for_dropped_stale_terminal_record(
    tmp_path: Path,
) -> None:
    config, paths, fake_gh, _dispatched_at = _dead_worker_rework_bed(
        tmp_path, decision="request_changes"
    )
    _write_stale_terminal(tmp_path)

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    events = query_events(paths.state_file, kind="worker_evidence_stale")
    stale = [e for e in events if e["payload"]["source"] == "terminal"]
    assert len(stale) == 1
    assert stale[0]["payload"]["reason"] == "older_than_dispatch"
    assert load_state(paths.state_file)["issues"]["207"]["stale_evidence_reported"]


def test_dead_dispatched_reap_reports_dropped_stale_terminal_record(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    sessions_dir = _write_stale_terminal(tmp_path)
    entry = {
        "status": "dispatched",
        "worker_pid": 99999,
        "dispatched_at": _iso(now - timedelta(hours=1)),
        "orphan_drift_at": _iso(now - timedelta(hours=2)),
    }
    fates: list[worker_fate.WorkerFate] = []

    _state, reaped = maybe_reap_dead_dispatched_worker(
        state={"issues": {"207": dict(entry)}, "prs": {}, "events": []},
        entry=entry,
        issue_number=207,
        sessions_dir=sessions_dir,
        pr_data=None,
        dead_dispatched_reap_minutes=60,
        now=now,
        sweep_events=[],
        max_throttle_rearms=0,
        on_fate=fates.append,
    )

    assert reaped is True
    assert [s.source for f in fates for s in f.basis.stale] == [
        worker_fate.EvidenceSource.TERMINAL
    ]


def test_fresh_terminal_record_reports_only_what_it_drops() -> None:
    dispatched = parse_iso_timestamp("2026-09-29T10:00:00Z")
    stale = {"ended_at": "2026-09-29T09:00:00Z", "exit_code": 0}
    fresh = {"ended_at": "2026-09-29T10:05:00Z", "exit_code": 0}
    fates: list[worker_fate.WorkerFate] = []

    assert (
        worker_fate.fresh_terminal_record(fresh, dispatched, issue_number=7, on_fate=fates.append)
        is fresh
    )
    assert fates == []
    assert (
        worker_fate.fresh_terminal_record(stale, dispatched, issue_number=7, on_fate=fates.append)
        is None
    )
    (fate,) = fates
    assert fate.basis.issue_number == 7
    (evidence,) = fate.basis.stale
    assert evidence.source is worker_fate.EvidenceSource.TERMINAL
    assert evidence.written_at == parse_iso_timestamp("2026-09-29T09:00:00Z")
