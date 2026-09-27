"""Doctor ``dependency sync starvation`` check (issue #1855).

Exercises ``doctor_sync_starvation._check_sync_starvation`` directly with a
collector ``add`` -- the same shape ``run_doctor`` passes -- against a real
events.db under ``tmp_path``. Direct-call keeps the test hermetic: in
production the check reads the *orchestrator* checkout's own self-deploy
state file (``supervise._self_deploy_state_path(orchestrator_root())``), not
the repo under examination, so a ``run_doctor``-level test would log
synthetic events into the real checkout's events.db.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from charlie_work.doctor import DoctorCheck
from charlie_work.doctor_sync_starvation import _check_sync_starvation
from charlie_work.instrumentation import log_event


def _collect() -> tuple[list[DoctorCheck], Any]:
    checks: list[DoctorCheck] = []

    def add(name: str, ok: bool, detail: str, *, severity: str = "error") -> None:
        checks.append(DoctorCheck(name=name, ok=ok, detail=detail, severity=severity))

    return checks, add


def test_check_sync_starvation_surfaces_recent_event(tmp_path: Path) -> None:
    """A recent ``self_deploy_sync_starved`` event surfaces as a warning --
    not an error: starvation means the drain posture already engaged, so the
    finding must not by itself flip a doctor run red."""
    state_path = tmp_path / "self-deploy-state.json"
    log_event(
        state_path,
        "self_deploy_sync_starved",
        {
            "pending_seconds": 15000,
            "starvation_seconds": 14400,
            "live_count": 3,
            "from_sha": "abc123",
            "to_sha": "def456",
        },
    )

    checks, add = _collect()
    _check_sync_starvation(add, state_path)

    assert len(checks) == 1
    check = checks[0]
    assert check.name == "dependency sync starvation"
    assert check.ok is False
    assert check.severity == "warning"
    assert "1 self_deploy_sync_starved event(s)" in check.detail
    assert "15000s" in check.detail
    assert "14400s" in check.detail


def test_check_sync_starvation_counts_multiple_events(tmp_path: Path) -> None:
    """Two distinct episodes (marker cleared + re-deferred between them)
    report a count of two -- the event fires once per episode, so the count
    is an episode count."""
    state_path = tmp_path / "self-deploy-state.json"
    for pending in (14500, 16000):
        log_event(
            state_path,
            "self_deploy_sync_starved",
            {
                "pending_seconds": pending,
                "starvation_seconds": 14400,
                "live_count": 1,
            },
        )

    checks, add = _collect()
    _check_sync_starvation(add, state_path)

    assert len(checks) == 1
    assert "2 self_deploy_sync_starved event(s)" in checks[0].detail


def test_check_sync_starvation_silent_when_no_events(tmp_path: Path) -> None:
    """No starvation events -> no finding at all; a healthy fleet's doctor
    output gains no new noise (mirrors the other events.db lookback checks)."""
    checks, add = _collect()
    _check_sync_starvation(add, tmp_path / "self-deploy-state.json")

    assert checks == []
