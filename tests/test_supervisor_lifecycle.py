"""Tests for the fleet supervisor lifecycle instrumentation (issue #627).

Covers the heartbeat sidecar I/O, the ``supervisor_started`` /
``supervisor_exited`` event recording, the retroactive prior-abnormal-exit
detection, and the ``is_exit_alertable`` policy.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from charlie_work.instrumentation import close_db, query_events
from charlie_work.supervisor_lifecycle import (
    HEARTBEAT_FILENAME,
    SUPERVISOR_EXITED,
    SUPERVISOR_STARTED,
    detect_prior_abnormal_exit,
    is_exit_alertable,
    record_prior_abnormal_exit,
    record_supervisor_exit,
    record_supervisor_started,
    supervisor_heartbeat_path,
    update_supervisor_heartbeat,
)


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


# All test timestamps are anchored to "now" so date-window-style assertions
# cannot rot as the calendar advances (issue #627 tests).
_started = datetime.now(UTC).replace(microsecond=0)
STARTED_AT = _iso(_started)
BEAT_AT = _iso(_started + timedelta(seconds=2609))


@pytest.fixture(autouse=True)
def _close_db_after_test(tmp_path: Path) -> None:
    yield
    close_db(supervisor_heartbeat_path(str(tmp_path / "fleet")))


def _heartbeat_file(fleet_dir: Path) -> Path:
    return fleet_dir / HEARTBEAT_FILENAME


def test_supervisor_heartbeat_path_respects_override(tmp_path: Path) -> None:
    assert (
        supervisor_heartbeat_path(str(tmp_path / "fleet"))
        == tmp_path / "fleet" / HEARTBEAT_FILENAME
    )


def test_record_supervisor_started_writes_heartbeat_and_event(tmp_path: Path) -> None:
    fleet_dir = str(tmp_path / "fleet")
    record_supervisor_started(
        fleet_dir,
        pid=12345,
        started_at=STARTED_AT,
        full_pass_interval_seconds=300,
        max_pass_runtime_seconds=300,
    )

    hb = json.loads(_heartbeat_file(tmp_path / "fleet").read_text(encoding="utf-8"))
    assert hb["pid"] == 12345
    assert hb["started_at"] == STARTED_AT
    assert hb["last_beat_at"] == STARTED_AT
    assert hb["pass_number"] == 0
    assert hb["full_pass_interval_seconds"] == 300
    assert hb["max_pass_runtime_seconds"] == 300
    assert hb["exited_at"] is None
    assert hb["exit_code"] is None

    events = query_events(supervisor_heartbeat_path(fleet_dir), kind=SUPERVISOR_STARTED)
    assert len(events) == 1
    assert events[0]["payload"]["pid"] == 12345
    assert events[0]["payload"]["full_pass_interval_seconds"] == 300
    assert events[0]["payload"]["max_pass_runtime_seconds"] == 300
    assert events[0]["repo"] == "fleet"


def test_update_supervisor_heartbeat_preserves_started_and_updates_beat(
    tmp_path: Path,
) -> None:
    fleet_dir = str(tmp_path / "fleet")
    record_supervisor_started(
        fleet_dir, pid=99, started_at=STARTED_AT, full_pass_interval_seconds=300
    )
    update_supervisor_heartbeat(fleet_dir, pass_number=4, last_beat_at=BEAT_AT)

    hb = json.loads(_heartbeat_file(tmp_path / "fleet").read_text(encoding="utf-8"))
    assert hb["started_at"] == STARTED_AT
    assert hb["pid"] == 99
    assert hb["last_beat_at"] == BEAT_AT
    assert hb["pass_number"] == 4
    assert hb["exited_at"] is None


def test_detect_prior_abnormal_exit_none_when_no_heartbeat(tmp_path: Path) -> None:
    assert detect_prior_abnormal_exit(str(tmp_path / "fleet")) is None


def test_detect_prior_abnormal_exit_none_when_clean_exit_recorded(
    tmp_path: Path,
) -> None:
    fleet_dir = str(tmp_path / "fleet")
    record_supervisor_started(
        fleet_dir, pid=1, started_at=STARTED_AT, full_pass_interval_seconds=300
    )
    record_supervisor_exit(
        fleet_dir, exit_code=0, passes=3, started_at=STARTED_AT, reason="completed"
    )
    assert detect_prior_abnormal_exit(fleet_dir) is None


def test_detect_prior_abnormal_exit_returns_payload_when_killed(tmp_path: Path) -> None:
    """A heartbeat with no exited_at means the prior supervisor was killed."""
    fleet_dir = str(tmp_path / "fleet")
    record_supervisor_started(
        fleet_dir, pid=4242, started_at=STARTED_AT, full_pass_interval_seconds=300
    )
    update_supervisor_heartbeat(fleet_dir, pass_number=9, last_beat_at=BEAT_AT)
    # No record_supervisor_exit — simulates a TerminateProcess kill.

    prior = detect_prior_abnormal_exit(fleet_dir)
    assert prior is not None
    assert prior["prior_pid"] == 4242
    assert prior["prior_started_at"] == STARTED_AT
    assert prior["prior_last_beat_at"] == BEAT_AT
    assert prior["prior_pass_number"] == 9
    assert prior["uptime_seconds"] is not None
    assert prior["uptime_seconds"] == pytest.approx(2609.0, abs=1.0)


def test_detect_prior_abnormal_exit_none_when_heartbeat_corrupt(
    tmp_path: Path,
) -> None:
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir(parents=True)
    _heartbeat_file(fleet_dir).write_text("{not json", encoding="utf-8")
    assert detect_prior_abnormal_exit(str(fleet_dir)) is None


def test_record_prior_abnormal_exit_emits_alertable_event(tmp_path: Path) -> None:
    fleet_dir = str(tmp_path / "fleet")
    prior = {
        "prior_pid": 4242,
        "prior_started_at": STARTED_AT,
        "prior_last_beat_at": BEAT_AT,
        "prior_pass_number": 9,
        "uptime_seconds": 2609.0,
    }
    payload = record_prior_abnormal_exit(fleet_dir, prior)

    assert payload["exit_code"] is None
    assert payload["reason"] == "prior_supervisor_terminated_without_exit_event"
    assert payload["passes"] == 9
    assert payload["prior_pid"] == 4242

    events = query_events(supervisor_heartbeat_path(fleet_dir), kind=SUPERVISOR_EXITED)
    assert len(events) == 1
    assert events[0]["payload"]["exit_code"] is None
    assert events[0]["repo"] == "fleet"


def test_record_supervisor_exit_stamps_heartbeat_and_emits_event(
    tmp_path: Path,
) -> None:
    fleet_dir = str(tmp_path / "fleet")
    record_supervisor_started(
        fleet_dir, pid=7, started_at=STARTED_AT, full_pass_interval_seconds=300
    )
    payload = record_supervisor_exit(
        fleet_dir, exit_code=0, passes=5, started_at=STARTED_AT, reason="completed"
    )

    assert payload["exit_code"] == 0
    assert payload["passes"] == 5
    assert payload["reason"] == "completed"
    assert payload["uptime_seconds"] is not None

    hb = json.loads(_heartbeat_file(tmp_path / "fleet").read_text(encoding="utf-8"))
    assert hb["exited_at"] is not None
    assert hb["exit_code"] == 0

    events = query_events(supervisor_heartbeat_path(fleet_dir), kind=SUPERVISOR_EXITED)
    assert len(events) == 1
    assert events[0]["payload"]["exit_code"] == 0


def test_record_supervisor_exit_nonzero_exit_code(tmp_path: Path) -> None:
    fleet_dir = str(tmp_path / "fleet")
    record_supervisor_started(
        fleet_dir, pid=7, started_at=STARTED_AT, full_pass_interval_seconds=300
    )
    record_supervisor_exit(
        fleet_dir, exit_code=1, passes=2, started_at=STARTED_AT, reason="exception"
    )
    hb = json.loads(_heartbeat_file(tmp_path / "fleet").read_text(encoding="utf-8"))
    assert hb["exit_code"] == 1
    assert hb["exited_at"] is not None


def test_is_exit_alertable_policy() -> None:
    assert is_exit_alertable(0) is False
    assert is_exit_alertable(1) is True
    assert is_exit_alertable(-1) is True
    assert is_exit_alertable(None) is True


def test_full_lifecycle_started_then_clean_exit_then_new_start_detects_no_gap(
    tmp_path: Path,
) -> None:
    """A clean exit stamps exited_at, so the next start detects no prior abnormal exit."""
    fleet_dir = str(tmp_path / "fleet")
    record_supervisor_started(
        fleet_dir, pid=1, started_at=STARTED_AT, full_pass_interval_seconds=300
    )
    update_supervisor_heartbeat(fleet_dir, pass_number=3, last_beat_at=BEAT_AT)
    record_supervisor_exit(
        fleet_dir, exit_code=0, passes=3, started_at=STARTED_AT, reason="completed"
    )

    # New supervisor starts: no prior abnormal exit because exited_at is set.
    assert detect_prior_abnormal_exit(fleet_dir) is None


def test_full_lifecycle_killed_then_new_start_detects_gap(tmp_path: Path) -> None:
    """A killed supervisor leaves no exited_at; the next start detects the gap."""
    fleet_dir = str(tmp_path / "fleet")
    record_supervisor_started(
        fleet_dir, pid=1, started_at=STARTED_AT, full_pass_interval_seconds=300
    )
    update_supervisor_heartbeat(fleet_dir, pass_number=9, last_beat_at=BEAT_AT)
    # Killed — no record_supervisor_exit.

    prior = detect_prior_abnormal_exit(fleet_dir)
    assert prior is not None
    record_prior_abnormal_exit(fleet_dir, prior)

    exited = query_events(supervisor_heartbeat_path(fleet_dir), kind=SUPERVISOR_EXITED)
    assert len(exited) == 1
    assert exited[0]["payload"]["exit_code"] is None
    assert exited[0]["payload"]["reason"] == "prior_supervisor_terminated_without_exit_event"


def test_record_supervisor_started_swallows_heartbeat_write_failure(
    tmp_path: Path,
    monkeypatch: Any,
    caplog: Any,
) -> None:
    """A disk error writing the heartbeat must not crash the supervisor start path."""
    fleet_dir = str(tmp_path / "fleet")

    def failing_write(_path: Path, _payload: dict[str, Any]) -> None:
        raise OSError("heartbeat disk full")

    monkeypatch.setattr(
        "charlie_work.supervisor_lifecycle._write_heartbeat",
        failing_write,
    )

    with caplog.at_level("WARNING", logger="charlie_work.supervisor_lifecycle"):
        record_supervisor_started(
            fleet_dir,
            pid=12345,
            started_at=STARTED_AT,
            full_pass_interval_seconds=300,
        )

    assert "heartbeat disk full" in caplog.text
    events = query_events(supervisor_heartbeat_path(fleet_dir), kind=SUPERVISOR_STARTED)
    assert len(events) == 1
    assert events[0]["payload"]["pid"] == 12345


def test_record_supervisor_started_swallows_event_log_failure(
    tmp_path: Path,
    monkeypatch: Any,
    caplog: Any,
) -> None:
    """A disk error logging the started event must not crash the supervisor start path."""
    fleet_dir = str(tmp_path / "fleet")

    def failing_log(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("event db full")

    monkeypatch.setattr(
        "charlie_work.supervisor_lifecycle.log_event",
        failing_log,
    )

    with caplog.at_level("WARNING", logger="charlie_work.supervisor_lifecycle"):
        record_supervisor_started(
            fleet_dir,
            pid=12345,
            started_at=STARTED_AT,
            full_pass_interval_seconds=300,
        )

    assert "event db full" in caplog.text
    # The heartbeat sidecar is still written even though the event log failed.
    hb = json.loads(_heartbeat_file(tmp_path / "fleet").read_text(encoding="utf-8"))
    assert hb["pid"] == 12345
