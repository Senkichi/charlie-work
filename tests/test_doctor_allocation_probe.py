"""Runner-allocation pass-evidence probe for ``run_doctor``.

Split out of ``tests/test_doctor.py`` (issue #1563, Track 1 shoulder) --
bodies are verbatim relocations; shared helpers live in
``tests/_doctor_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from charlie_work.doctor import run_doctor
from charlie_work.paths import runtime_paths
from _doctor_fixtures import (
    FakeDoctorGitHub,
    _collect_allocation_checks,
    _doctor_allocation_config,
    _write_allocation_stamp,
    _write_supervisor_heartbeat,
)


def test_allocation_probe_is_silent_when_the_feature_is_disabled(tmp_path: Path) -> None:
    checks = _collect_allocation_checks(_doctor_allocation_config(enabled=False), tmp_path)
    assert checks == []


def test_allocation_probe_reports_enabled_but_never_run(tmp_path: Path) -> None:
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    assert len(checks) == 1
    name, ok, detail = checks[0]
    assert name == "runner allocation"
    assert ok is False
    assert "never run" in detail


def test_allocation_probe_passes_on_a_fresh_pass(tmp_path: Path) -> None:
    import datetime

    from ci_fleet.charlie_work_adapter import ALLOCATION_STATE_FILENAME

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (tmp_path / ALLOCATION_STATE_FILENAME).write_text(
        json.dumps({"version": 1, "updated_at": now, "source": "prologue", "repos": {}}),
        encoding="utf-8",
    )
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    assert len(checks) == 1
    _, ok, detail = checks[0]
    assert ok is True
    assert "budget 8" in detail
    # Only an unattended write is evidence, so the ok line has to say which it saw.
    assert "unattended" in detail


def test_allocation_probe_flags_a_stale_pass(tmp_path: Path) -> None:
    """A configured-but-inert allocator is the exact shape of issue #590."""
    import datetime

    from ci_fleet.charlie_work_adapter import ALLOCATION_STATE_FILENAME

    config = _doctor_allocation_config()
    # The staleness bound is the pass runtime cap plus three intervals
    # (issue #1852): 1800 + 300*3 = 2700 s under the default config.
    stale_by = (
        config.supervisor.max_pass_runtime_seconds
        + config.supervisor.full_pass_interval_seconds * 3
        + 60
    )
    old = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=stale_by)
    (tmp_path / ALLOCATION_STATE_FILENAME).write_text(
        json.dumps(
            {"version": 1, "updated_at": old.isoformat(), "source": "prologue", "repos": {}}
        ),
        encoding="utf-8",
    )
    checks = _collect_allocation_checks(config, tmp_path)
    assert len(checks) == 1
    _, ok, detail = checks[0]
    assert ok is False
    assert "not running unattended" in detail


def test_allocation_probe_survives_a_corrupt_state_file(tmp_path: Path) -> None:
    from ci_fleet.charlie_work_adapter import ALLOCATION_STATE_FILENAME

    (tmp_path / ALLOCATION_STATE_FILENAME).write_text("{not json", encoding="utf-8")
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    assert len(checks) == 1
    _, ok, detail = checks[0]
    assert ok is False
    assert "updated_at" in detail


def test_allocation_probe_does_not_accept_a_manual_pass_as_evidence(tmp_path: Path) -> None:
    """A fresh manual allocate must not make the probe read healthy.

    CLAUDE.md requires post-reboot procedures to delegate to `charlie runners
    allocate`, so this is the routine case -- and it writes the same host-wide file
    the unattended pass does. Treating its timestamp as proof would blind the probe
    for three intervals during exactly the window an operator is diagnosing #590.
    """
    _write_allocation_stamp(tmp_path, age_seconds=5, source="cli")
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    assert len(checks) == 1
    _, ok, detail = checks[0]
    assert ok is False
    assert "manual" in detail
    assert "cannot confirm" in detail


def test_allocation_probe_reports_unrecorded_provenance_rather_than_assuming(
    tmp_path: Path,
) -> None:
    """A file written before provenance tracking is unknown, not unattended."""
    _write_allocation_stamp(tmp_path, age_seconds=5, source=None)
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    assert len(checks) == 1
    _, ok, detail = checks[0]
    assert ok is False
    assert "unrecorded" in detail


def test_allocation_probe_names_an_unrecognised_writer(tmp_path: Path) -> None:
    """A future writer that forgets to extend AllocationSource is named, not hidden."""
    _write_allocation_stamp(tmp_path, age_seconds=5, source="some-new-path")
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    _, ok, detail = checks[0]
    assert ok is False
    assert "some-new-path" in detail


def test_allocation_probe_clamps_a_future_dated_stamp(tmp_path: Path) -> None:
    """Clock skew must not print a negative age, and must not read as stale.

    Regression for issue #828 (originally #822's class): the stamp is
    future-dated by 600s, and the clamp (doctor.py's non-negative `max(0, ...)`
    -- preserved, not removed) only reads as exactly "0s ago" while the
    probe's own `now` sample has not yet caught up to the future-dated
    `updated_at`. Two independently-sampled `now()`s would flip this the
    moment a stall pushes the probe's read past 600s after the fixture write
    -- the same order of magnitude as an observed CI stall. `now` is frozen
    and passed to both the fixture write and the probe so the comparison is
    exact regardless of any stall in between.
    """
    import datetime

    frozen_now = datetime.datetime(2026, 7, 29, 12, 0, 0, tzinfo=datetime.timezone.utc)
    _write_allocation_stamp(tmp_path, age_seconds=-600, source="prologue", now=frozen_now)
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path, now=frozen_now)
    _, ok, detail = checks[0]
    assert ok is True
    assert "-" not in detail
    assert "0s ago" in detail


def test_allocation_probe_measures_staleness_against_the_recorded_interval(
    tmp_path: Path,
) -> None:
    """The bound comes from the interval the pass was driven at, not re-resolved.

    A per-repo layer setting a different interval would otherwise make the probe
    measure against a cadence the daemon is not running at. Here the recorded
    interval is far shorter than the config default, so a stamp that is fresh
    under the config bound is stale under the recorded one.
    """
    # Recorded interval 10s + heartbeat cap 30s -> bound 60s. Config defaults
    # are interval 300s / cap 1800s -> bound 2700s. Age 90s is stale under the
    # recorded bound but fresh under the config-resolved one.
    _write_allocation_stamp(
        tmp_path,
        age_seconds=90,
        source="prologue",
        full_pass_interval_seconds=10,
    )
    _write_supervisor_heartbeat(tmp_path, {"max_pass_runtime_seconds": 30})
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    _, ok, detail = checks[0]
    assert ok is False
    assert "60s staleness bound" in detail
    assert "not running unattended" in detail


def test_allocation_probe_falls_back_to_config_interval_when_none_recorded(
    tmp_path: Path,
) -> None:
    """A file written before interval recording uses the config bound."""
    config = _doctor_allocation_config()
    # Age just past the config-resolved bound — cap 1800 + interval 300*3
    # = 2700 (issue #1852) — with no recorded interval. The 2700s bound in
    # the detail pins that the interval fell back to config's 300s.
    _write_allocation_stamp(
        tmp_path,
        age_seconds=(
            config.supervisor.max_pass_runtime_seconds
            + config.supervisor.full_pass_interval_seconds * 3
            + 60
        ),
        source="prologue",
    )
    checks = _collect_allocation_checks(config, tmp_path)
    _, ok, detail = checks[0]
    assert ok is False
    assert "2700s staleness bound" in detail
    assert "not running unattended" in detail


def test_allocation_probe_tolerates_a_stamp_inside_the_pass_runtime_cap(
    tmp_path: Path,
) -> None:
    """Issue #1852: a mid-pass stamp is not stale while the pass is in its cap.

    The stamp is rewritten only in the pass prologue, so a healthy pass holds
    it unrewritten for up to ``max_pass_runtime_seconds``. This is the
    observed false positive: on 2026-09-23 doctor flagged a stamp 2,132 s old
    as stale at the bare 900 s bound while the supervisor was mid-pass. With
    the heartbeat's cap the bound is 1800 + 3 x 300 = 2700 s, so 2,132 s is
    fresh.
    """
    _write_allocation_stamp(
        tmp_path,
        age_seconds=2132,
        source="prologue",
        full_pass_interval_seconds=300,
    )
    _write_supervisor_heartbeat(tmp_path, {"max_pass_runtime_seconds": 1800})
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    _, ok, detail = checks[0]
    assert ok is True
    assert "unattended" in detail


def test_allocation_probe_flags_a_stamp_beyond_cap_plus_intervals(
    tmp_path: Path,
) -> None:
    """Past cap + 3 x interval the stamp is genuinely stale -- and the warning
    names the bound it used and where the cap came from."""
    _write_allocation_stamp(
        tmp_path,
        age_seconds=3000,
        source="prologue",
        full_pass_interval_seconds=300,
    )
    _write_supervisor_heartbeat(tmp_path, {"max_pass_runtime_seconds": 1800})
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    _, ok, detail = checks[0]
    assert ok is False
    assert "2700s staleness bound" in detail
    assert "supervisor-heartbeat.json" in detail
    assert "not running unattended" in detail


def test_allocation_probe_uses_the_heartbeat_cap_over_config(tmp_path: Path) -> None:
    """The heartbeat's recorded cap wins over the probe's own config load.

    Heartbeat cap 3600 + interval 300 x 3 = 4500 s bound; age 3000 s is fresh
    under the heartbeat cap but stale under config's (1800 + 900 = 2700 s), so
    an ok here can only mean the heartbeat supplied the cap.
    """
    _write_allocation_stamp(
        tmp_path,
        age_seconds=3000,
        source="prologue",
        full_pass_interval_seconds=300,
    )
    _write_supervisor_heartbeat(tmp_path, {"max_pass_runtime_seconds": 3600})
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    _, ok, _detail = checks[0]
    assert ok is True


def test_allocation_probe_falls_back_to_config_cap_when_heartbeat_absent(
    tmp_path: Path,
) -> None:
    """No heartbeat file -> the cap falls back to config.supervisor's knob.

    Age 960 s is over the bare 3 x 300 s bound but inside 1800 + 900 = 2700 s,
    so an ok here proves the config cap was applied. The bound named in a
    stale verdict must say the cap came from config (checked below at 3000 s).
    """
    _write_allocation_stamp(
        tmp_path,
        age_seconds=960,
        source="prologue",
        full_pass_interval_seconds=300,
    )
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    _, ok, _detail = checks[0]
    assert ok is True

    _write_allocation_stamp(
        tmp_path,
        age_seconds=3000,
        source="prologue",
        full_pass_interval_seconds=300,
    )
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    _, ok, detail = checks[0]
    assert ok is False
    assert "2700s staleness bound" in detail
    assert "config supervisor.max_pass_runtime_seconds" in detail


def test_allocation_probe_falls_back_to_config_cap_when_heartbeat_lacks_field(
    tmp_path: Path,
) -> None:
    """A heartbeat that predates the field -> config cap, same as absent."""
    _write_allocation_stamp(
        tmp_path,
        age_seconds=960,
        source="prologue",
        full_pass_interval_seconds=300,
    )
    _write_supervisor_heartbeat(tmp_path, {"pid": 1, "pass_number": 4})
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    _, ok, _detail = checks[0]
    assert ok is True


def test_allocation_probe_falls_back_to_bare_interval_bound_when_no_cap(
    tmp_path: Path,
) -> None:
    """Neither heartbeat nor config records a cap -> the bound is 3 x interval.

    ``config.supervisor`` is replaced with a namespace that predates the
    ``max_pass_runtime_seconds`` knob (the same "config object built by code
    that predates the section" shape the probe already tolerates for the
    runner_allocation section itself), and the heartbeat lacks the field, so
    the bound collapses to today's 3 x 300 = 900 s.
    """
    import dataclasses
    from types import SimpleNamespace

    config = dataclasses.replace(
        _doctor_allocation_config(),
        supervisor=SimpleNamespace(full_pass_interval_seconds=300),
    )
    _write_allocation_stamp(
        tmp_path,
        age_seconds=960,
        source="prologue",
        full_pass_interval_seconds=300,
    )
    _write_supervisor_heartbeat(tmp_path, {"pid": 1, "pass_number": 4})
    checks = _collect_allocation_checks(config, tmp_path)
    _, ok, detail = checks[0]
    assert ok is False
    assert "900s staleness bound" in detail
    assert "not running unattended" in detail


def test_allocation_probe_reports_a_recorded_skip_reason(tmp_path: Path) -> None:
    """A fresh unattended skip names the cause instead of asserting #590."""
    _write_allocation_stamp(
        tmp_path,
        age_seconds=5,
        source="prologue",
        full_pass_interval_seconds=300,
        skip_reason="no configured runners found under /actions-runners",
    )
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    _, ok, detail = checks[0]
    assert ok is False
    assert "declined to act" in detail
    assert "no configured runners found under /actions-runners" in detail
    # A fresh unattended skip is the daemon *reaching* allocation and declining —
    # not the "never reached it" shape #590 describes — so #590 must not appear.
    assert "#590" not in detail


def test_allocation_probe_joins_skip_reason_and_staleness_when_stale(
    tmp_path: Path,
) -> None:
    """A stale skip reports both the recorded reason and the #590 reading."""
    _write_allocation_stamp(
        tmp_path,
        age_seconds=3000,
        source="prologue",
        full_pass_interval_seconds=300,
        skip_reason="no configured runners found under /actions-runners",
    )
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    _, ok, detail = checks[0]
    assert ok is False
    assert "declined to act" in detail
    assert "no configured runners found under /actions-runners" in detail
    # Stale: the daemon last found this and has not been back, so #590 joins.
    assert "not running unattended" in detail
    assert "#590" in detail


def test_allocation_probe_does_not_flag_a_mid_pass_skip_as_stale(
    tmp_path: Path,
) -> None:
    """Issue #1852: a mid-pass *skip* stamp is fresh inside cap + 3 x interval.

    The skip_reason branch shares the widened bound (cap + 3 x interval =
    2700 s under the default config). Age 2,132 s sits between the pre-#1852
    bound (3 x 300 = 900 s) and the new one: a mutant that reverts only this
    branch to ``interval * 3`` would append the staleness clause and fail
    both negative assertions below.
    """
    _write_allocation_stamp(
        tmp_path,
        age_seconds=2132,
        source="prologue",
        full_pass_interval_seconds=300,
        skip_reason="no configured runners found under /actions-runners",
    )
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    _, ok, detail = checks[0]
    assert ok is False
    assert "declined to act" in detail
    assert "no configured runners found under /actions-runners" in detail
    # Fresh unattended skip: neither staleness nor source-mismatch clause
    # applies, so the #590 framing must not appear at all.
    assert "not running unattended" not in detail
    assert "#590" not in detail


def test_allocation_probe_skip_branch_names_the_cap_aware_staleness_bound(
    tmp_path: Path,
) -> None:
    """A stale skip's joined clause states the bound used and the cap source.

    Heartbeat cap 1,800 + interval 300 x 3 = 2,700 s bound; age 3,000 s is
    stale, and the detail must render the same ``bound_detail`` string the
    plain stale branch produces — including where the cap was read from.
    """
    _write_allocation_stamp(
        tmp_path,
        age_seconds=3000,
        source="prologue",
        full_pass_interval_seconds=300,
        skip_reason="no configured runners found under /actions-runners",
    )
    _write_supervisor_heartbeat(tmp_path, {"max_pass_runtime_seconds": 1800})
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    _, ok, detail = checks[0]
    assert ok is False
    assert "declined to act" in detail
    assert "no configured runners found under /actions-runners" in detail
    assert "2700s staleness bound" in detail
    assert "supervisor-heartbeat.json" in detail
    assert "not running unattended" in detail


def test_allocation_probe_reports_a_manual_skip_with_the_recorded_reason(
    tmp_path: Path,
) -> None:
    """A CLI skip names the reason; the writer is still flagged as non-daemon.

    A fresh manual skip records *why* it declined (issue #606) but, like a
    fresh manual non-skip, cannot confirm the daemon is rebalancing — the
    writer overwrites the same host-wide file, so its skip reason is not
    evidence the unattended pass reached allocation (issue #590). Both
    signals must appear together.
    """
    _write_allocation_stamp(
        tmp_path,
        age_seconds=5,
        source="cli",
        full_pass_interval_seconds=300,
        skip_reason="no configured runners found under /actions-runners",
    )
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    _, ok, detail = checks[0]
    assert ok is False
    assert "declined to act" in detail
    assert "no configured runners found under /actions-runners" in detail
    assert "manual" in detail
    # The writer is non-unattended, so the #590 "cannot confirm" framing must
    # join the recorded reason — exactly the distinction this probe preserves.
    assert "cannot confirm" in detail
    assert "#590" in detail


def test_allocation_probe_cannot_confirm_clause_for_a_fresh_manual_skip(
    tmp_path: Path,
) -> None:
    """Regression: a fresh manual (CLI) skip surfaces the #590 clause.

    The skip_reason branch returns before the source-mismatch check, so the
    'cannot confirm the daemon is rebalancing (issue #590)' framing must be
    appended within that branch when the writer is not unattended — otherwise
    an operator's manual run reads as a named daemon skip and blinds the probe
    for three intervals during the very window #590 is being diagnosed.
    """
    _write_allocation_stamp(
        tmp_path,
        age_seconds=5,
        source="cli",
        full_pass_interval_seconds=300,
        skip_reason="no configured runners found under /actions-runners",
    )
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    _, ok, detail = checks[0]
    assert ok is False
    # The #590 clause is the authoritative signal that a manual write cannot
    # stand in for daemon health.
    assert "cannot confirm the daemon is rebalancing" in detail
    assert "issue #590" in detail
    # Fresh (5s < 900s bound): staleness must NOT also be cited — only the
    # source-mismatch clause applies, so "not running unattended" is absent.
    assert "not running unattended" not in detail


def test_allocation_probe_joins_source_and_staleness_for_a_stale_manual_skip(
    tmp_path: Path,
) -> None:
    """A stale manual skip cites both clauses, with #590 named once.

    Both the source mismatch (manual write cannot confirm the daemon) and
    staleness (daemon has not been back) apply; the clauses join so #590 is
    cited once rather than duplicated.
    """
    _write_allocation_stamp(
        tmp_path,
        age_seconds=3000,
        source="cli",
        full_pass_interval_seconds=300,
        skip_reason="no configured runners found under /actions-runners",
    )
    checks = _collect_allocation_checks(_doctor_allocation_config(), tmp_path)
    _, ok, detail = checks[0]
    assert ok is False
    assert "declined to act" in detail
    assert "cannot confirm the daemon is rebalancing" in detail
    assert "not running unattended" in detail
    # #590 cited exactly once, not duplicated.
    assert detail.count("#590") == 1


def test_run_doctor_wires_the_allocation_probe(tmp_path: Path) -> None:
    """Pin the wiring, not just the probe body.

    Every other allocation test calls ``_check_runner_allocation`` directly, so
    deleting its call in ``run_doctor`` would leave them all green while the probe
    silently stopped running for operators.
    """
    config = _doctor_allocation_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    _, checks = run_doctor(
        tmp_path, paths, config, tmp_path / "c.yaml", gh, fleet_dir_override=str(tmp_path)
    )

    allocation_checks = [c for c in checks if c.name == "runner allocation"]
    assert len(allocation_checks) == 1
    # No state file was written, so the probe should report the never-run case.
    assert allocation_checks[0].ok is False
    assert "never run" in allocation_checks[0].detail
    # Warning-only: the probe must never change doctor's exit code.
    assert allocation_checks[0].severity == "warning"
