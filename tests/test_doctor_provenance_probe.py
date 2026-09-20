"""ci_fleet provenance-refusal streak probe for ``run_doctor`` (issue #1753).

``provenance-refusals.json`` lives in the host-wide fleet dir and is written
by ``ci_fleet.runner_allocation_pass`` inside the installed ``ci_fleet``
package — structurally outside ``tests/test_event_kind_consumers.py``'s
``src/charlie_work`` scan (issue #1364). Until this check landed, the
escalated streak had no consumer at all: a 22-day ``no_anchor`` escalation
was invisible to ``charlie doctor``. These tests pin the consumer, matching
the seam-named sibling convention of ``test_doctor_allocation_probe.py``.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from ci_fleet.provenance import (
    PROVENANCE_ESCALATION_THRESHOLD,
    REFUSAL_STATE_FILENAME,
    UNVERIFIED_ESCALATION_SECONDS,
)

from charlie_work.doctor import run_doctor
from charlie_work.paths import runtime_paths
from _doctor_fixtures import FakeDoctorGitHub, _config


def _collect_provenance_checks(fleet_dir: Path) -> list[tuple[str, bool, str, str]]:
    from charlie_work.doctor import _check_provenance_refusal_streak

    collected: list[tuple[str, bool, str, str]] = []

    def add(name: str, ok: bool, detail: str, *, severity: str = "error") -> None:
        collected.append((name, ok, detail, severity))

    _check_provenance_refusal_streak(add, fleet_dir_override=str(fleet_dir))
    return collected


def _write_refusal_streak(
    fleet_dir: Path,
    *,
    status: str,
    consecutive: int,
    span_seconds: float,
    detail: str = "test refusal detail",
) -> None:
    """Write a streak file whose ``first_seen``→``last_seen`` span is ``span_seconds``.

    Timestamps derive from the real clock: ``RefusalStreak.escalated`` compares
    the recorded span against a wall-clock threshold, so a frozen literal date
    would silently rot the fixture's relationship to that threshold.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "consecutive": consecutive,
        "first_seen": (now - datetime.timedelta(seconds=span_seconds)).isoformat(),
        "last_seen": now.isoformat(),
        "detail": detail,
        "status": status,
    }
    (fleet_dir / REFUSAL_STATE_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


def test_provenance_probe_reports_ok_when_no_streak_file_exists(tmp_path: Path) -> None:
    """A never-run / freshly-installed fleet is not a finding."""
    checks = _collect_provenance_checks(tmp_path)
    assert len(checks) == 1
    name, ok, detail, _severity = checks[0]
    assert name == "ci_fleet provenance"
    assert ok is True
    assert "absent" in detail


def test_provenance_probe_flags_an_escalated_no_anchor_streak(tmp_path: Path) -> None:
    """The exact shape of the live 22-day streak this issue was filed for."""
    _write_refusal_streak(
        tmp_path,
        status="no_anchor",
        consecutive=4405,
        span_seconds=UNVERIFIED_ESCALATION_SECONDS + 3600,
        detail="provenance anchor is installed but could not determine an expected root",
    )
    checks = _collect_provenance_checks(tmp_path)
    assert len(checks) == 1
    name, ok, detail, severity = checks[0]
    assert name == "ci_fleet provenance"
    assert ok is False
    assert severity == "warning"
    assert "no_anchor" in detail
    assert "4405" in detail
    assert "could not determine an expected root" in detail


def test_provenance_probe_passes_a_sub_threshold_no_anchor_streak(tmp_path: Path) -> None:
    """A routine abstention inside the deploy window must not page anyone.

    ``no_anchor`` escalates on wall-clock span, not pass count — a high
    ``consecutive`` with a short span is still routine.
    """
    _write_refusal_streak(
        tmp_path,
        status="no_anchor",
        consecutive=5000,
        span_seconds=UNVERIFIED_ESCALATION_SECONDS - 3600,
    )
    checks = _collect_provenance_checks(tmp_path)
    assert len(checks) == 1
    _, ok, detail, _severity = checks[0]
    assert ok is True
    assert "below the escalation threshold" in detail


def test_provenance_probe_flags_an_escalated_mismatch_streak(tmp_path: Path) -> None:
    """``mismatch`` blocks actuation — it must surface, not only ``no_anchor``."""
    _write_refusal_streak(
        tmp_path,
        status="mismatch",
        consecutive=PROVENANCE_ESCALATION_THRESHOLD,
        span_seconds=60,
        detail="ci_fleet is imported from X but the provider declares Y",
    )
    checks = _collect_provenance_checks(tmp_path)
    assert len(checks) == 1
    _, ok, detail, severity = checks[0]
    assert ok is False
    assert severity == "warning"
    assert "mismatch" in detail


def test_provenance_probe_passes_a_sub_threshold_mismatch_streak(tmp_path: Path) -> None:
    """``mismatch`` escalates on consecutive count, below the threshold is routine."""
    _write_refusal_streak(
        tmp_path,
        status="mismatch",
        consecutive=PROVENANCE_ESCALATION_THRESHOLD - 1,
        span_seconds=UNVERIFIED_ESCALATION_SECONDS + 3600,
    )
    checks = _collect_provenance_checks(tmp_path)
    assert len(checks) == 1
    _, ok, detail, _severity = checks[0]
    assert ok is True
    assert "below the escalation threshold" in detail


def test_provenance_probe_notes_a_present_but_unparseable_file(tmp_path: Path) -> None:
    """``load_refusal_streak`` reads corruption as no-streak; the detail must
    say the file was there rather than reporting a clean absence."""
    (tmp_path / REFUSAL_STATE_FILENAME).write_text("{not json", encoding="utf-8")
    checks = _collect_provenance_checks(tmp_path)
    assert len(checks) == 1
    _, ok, detail, _severity = checks[0]
    assert ok is True
    assert "did not parse" in detail


def test_run_doctor_wires_the_provenance_probe(tmp_path: Path) -> None:
    """Pin the wiring, not just the probe body.

    Every other test here calls ``_check_provenance_refusal_streak`` directly,
    so deleting its call in ``run_doctor`` would leave them all green while the
    probe silently stopped running for operators.
    """
    config = _config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)
    _write_refusal_streak(
        tmp_path,
        status="no_anchor",
        consecutive=10,
        span_seconds=UNVERIFIED_ESCALATION_SECONDS + 60,
    )

    _, checks = run_doctor(
        tmp_path, paths, config, tmp_path / "c.yaml", gh, fleet_dir_override=str(tmp_path)
    )

    provenance_checks = [c for c in checks if c.name == "ci_fleet provenance"]
    assert len(provenance_checks) == 1
    assert provenance_checks[0].ok is False
    assert "escalated" in provenance_checks[0].detail
    # Warning-only: the probe must never change doctor's exit code.
    assert provenance_checks[0].severity == "warning"
