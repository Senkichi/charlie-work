"""Tests for the preflight gate (issue #1363, PART 1).

Covers acceptance criteria 1 (module shape + config-driven thresholds), 2
(the disk-floor refusal path, at the unit level -- including the
event-write-failure -> stderr fallback), 4 (venv_identity), and 5
(config_freshness fires exactly once per change).
"""

from __future__ import annotations

import io
from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from charlie_work.config import PreflightConfig, build_config_from_data
from charlie_work.instrumentation import _LEVEL_BY_KIND, close_db, query_events
from charlie_work.preflight import (
    PreflightCheck,
    PreflightPaths,
    PreflightResult,
    emit_preflight_refusal,
    run_preflight,
)

UTC = timezone.utc


@pytest.fixture(autouse=True)
def _close_db_after_test(tmp_path: Path) -> None:
    yield
    close_db(tmp_path / "state.json")


def _paths(tmp_path: Path) -> PreflightPaths:
    repo_root = tmp_path / "repo"
    state_dir = repo_root / ".var" / "charlie-work"
    state_dir.mkdir(parents=True)
    (repo_root / ".venv" / "Scripts").mkdir(parents=True)
    return PreflightPaths(repo_root=repo_root, state_dir=state_dir)


def _disk_usage_free(free_bytes: int):
    def _fn(anchor: str) -> SimpleNamespace:
        return SimpleNamespace(total=1_000_000_000_000, used=0, free=free_bytes)

    return _fn


# ---------------------------------------------------------------------------
# AC1: module shape, frozen dataclasses, config-driven thresholds
# ---------------------------------------------------------------------------


def test_preflight_check_is_frozen() -> None:
    check = PreflightCheck(name="disk_floor", ok=True, detail="fine", fatal=True)
    with pytest.raises(FrozenInstanceError):
        check.ok = False  # type: ignore[misc]


def test_preflight_result_is_frozen() -> None:
    result = PreflightResult(checks=())
    with pytest.raises(FrozenInstanceError):
        result.checks = ()  # type: ignore[misc]


def test_preflight_config_defaults_match_issue_spec() -> None:
    cfg = PreflightConfig()
    assert cfg.disk_floor_gb == 10
    assert cfg.disk_floor_fatal is True
    assert cfg.clock_max_skew_hours == 48.0
    assert cfg.clock_sanity_fatal is False
    assert cfg.venv_identity_fatal is True
    assert cfg.config_freshness_fatal is False


def test_preflight_config_is_frozen() -> None:
    cfg = PreflightConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.disk_floor_gb = 1  # type: ignore[misc]


def test_runtime_preflight_section_parses_from_yaml_data() -> None:
    config = build_config_from_data(
        {
            "runtime": {
                "preflight": {
                    "disk_floor_gb": 25,
                    "disk_floor_fatal": False,
                    "clock_max_skew_hours": 12.5,
                }
            }
        }
    )
    assert config.runtime.preflight.disk_floor_gb == 25
    assert config.runtime.preflight.disk_floor_fatal is False
    assert config.runtime.preflight.clock_max_skew_hours == 12.5
    # Untouched fields keep their defaults.
    assert config.runtime.preflight.venv_identity_fatal is True


def test_runtime_preflight_section_rejects_unknown_key() -> None:
    from charlie_work.config import ConfigError

    with pytest.raises(ConfigError, match="unknown key"):
        build_config_from_data({"runtime": {"preflight": {"bogus_key": 1}}})


def test_runtime_preflight_section_rejects_bad_type() -> None:
    from charlie_work.config import ConfigError

    with pytest.raises(ConfigError, match="disk_floor_gb"):
        build_config_from_data({"runtime": {"preflight": {"disk_floor_gb": "ten"}}})


def test_run_preflight_runs_all_four_checks_in_order(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    result = run_preflight(
        paths,
        PreflightConfig(),
        now=datetime.now(UTC),
        disk_usage=_disk_usage_free(100 * 1024**3),
        stat_fn=lambda p: SimpleNamespace(st_mtime=datetime.now(UTC).timestamp()),
        sys_executable=str(paths.venv_dir / "Scripts" / "python.exe"),
        package_file=str(paths.repo_root / "src" / "charlie_work" / "preflight.py"),
    )
    assert [c.name for c in result.checks] == [
        "disk_floor",
        "clock_sanity",
        "venv_identity",
        "config_freshness",
    ]
    assert result.ok is True


# ---------------------------------------------------------------------------
# AC2: disk_floor refusal path, unit-level
# ---------------------------------------------------------------------------


def test_disk_floor_below_threshold_is_fatal_and_names_measured_value(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    cfg = PreflightConfig(disk_floor_gb=10)
    result = run_preflight(
        paths,
        cfg,
        disk_usage=_disk_usage_free(1 * 1024**3),  # 1 GB free, floor is 10
        sys_executable=str(paths.venv_dir / "Scripts" / "python.exe"),
        package_file=str(paths.repo_root / "src" / "charlie_work" / "preflight.py"),
    )
    disk_check = next(c for c in result.checks if c.name == "disk_floor")
    assert disk_check.ok is False
    assert disk_check.fatal is True
    assert result.ok is False
    assert result.fatal_failures == (disk_check,)
    # Detail must name both the measured value and the threshold.
    assert "1.00 GB" in disk_check.detail
    assert "10 GB" in disk_check.detail


def test_disk_floor_above_threshold_is_ok(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    result = run_preflight(
        paths,
        PreflightConfig(disk_floor_gb=10),
        disk_usage=_disk_usage_free(50 * 1024**3),
        sys_executable=str(paths.venv_dir / "Scripts" / "python.exe"),
        package_file=str(paths.repo_root / "src" / "charlie_work" / "preflight.py"),
    )
    disk_check = next(c for c in result.checks if c.name == "disk_floor")
    assert disk_check.ok is True
    assert disk_check.fatal is True


def test_disk_floor_respects_config_fatal_override(tmp_path: Path) -> None:
    """AC1: fatal/non-fatal classification comes from config, not a hardcoded rule."""
    paths = _paths(tmp_path)
    result = run_preflight(
        paths,
        PreflightConfig(disk_floor_gb=10, disk_floor_fatal=False),
        disk_usage=_disk_usage_free(1 * 1024**3),
        sys_executable=str(paths.venv_dir / "Scripts" / "python.exe"),
        package_file=str(paths.repo_root / "src" / "charlie_work" / "preflight.py"),
    )
    disk_check = next(c for c in result.checks if c.name == "disk_floor")
    assert disk_check.ok is False
    assert disk_check.fatal is False
    # A non-fatal failure does not flip overall result.ok.
    assert result.ok is True
    assert result.non_fatal_failures == (disk_check,)


def test_emit_preflight_refusal_writes_event_on_success(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text("{}", encoding="utf-8")
    check = PreflightCheck(name="disk_floor", ok=False, detail="1.00 GB free", fatal=True)

    emit_preflight_refusal(state_path, check)

    events = query_events(state_path, kind="loop_refused_preflight")
    assert len(events) == 1
    assert events[0]["payload"]["check"] == "disk_floor"
    assert events[0]["payload"]["detail"] == "1.00 GB free"
    assert events[0]["level"] == "error"
    assert "loop_refused_preflight" in _LEVEL_BY_KIND
    assert _LEVEL_BY_KIND["loop_refused_preflight"] == "error"


def test_emit_preflight_refusal_falls_back_to_stderr_when_db_unavailable(tmp_path: Path) -> None:
    """AC2: 'the refusal reaches stderr when [the event write] does not [succeed]'.

    Simulated here via an injected ``db_available_fn`` returning False --
    representing the disk-full case where events.db cannot even be opened.
    """
    state_path = tmp_path / "state.json"
    check = PreflightCheck(name="disk_floor", ok=False, detail="0.00 GB free", fatal=True)
    fake_stderr = io.StringIO()
    log_event_calls: list[tuple] = []

    def _fake_log_event(*args, **kwargs) -> None:
        log_event_calls.append((args, kwargs))

    emit_preflight_refusal(
        state_path,
        check,
        log_event_fn=_fake_log_event,
        db_available_fn=lambda _p: False,
        stderr=fake_stderr,
    )

    assert not log_event_calls, "must not attempt the write when the db is unavailable"
    output = fake_stderr.getvalue()
    assert "disk_floor" in output
    assert "0.00 GB free" in output
    assert "unavailable" in output.lower()


def test_emit_preflight_refusal_falls_back_to_stderr_when_log_event_raises(tmp_path: Path) -> None:
    """AC2: simulate the event-write failure itself raising, not just a closed db."""
    state_path = tmp_path / "state.json"
    check = PreflightCheck(name="disk_floor", ok=False, detail="0.00 GB free", fatal=True)
    fake_stderr = io.StringIO()

    def _raising_log_event(*args, **kwargs) -> None:
        raise OSError("No space left on device")

    emit_preflight_refusal(
        state_path,
        check,
        log_event_fn=_raising_log_event,
        db_available_fn=lambda _p: True,
        stderr=fake_stderr,
    )

    output = fake_stderr.getvalue()
    assert "disk_floor" in output
    assert "No space left on device" in output


def test_run_preflight_refusal_means_loop_body_not_invoked(tmp_path: Path) -> None:
    """Unit-level stand-in for the (not-yet-wired) workflow.py call site: a
    fatal check failing must be something a caller can branch on to skip
    ``_loop_body`` entirely, without needing workflow.py in this test."""
    paths = _paths(tmp_path)
    result = run_preflight(
        paths,
        PreflightConfig(disk_floor_gb=10),
        disk_usage=_disk_usage_free(0),
        sys_executable=str(paths.venv_dir / "Scripts" / "python.exe"),
        package_file=str(paths.repo_root / "src" / "charlie_work" / "preflight.py"),
    )

    loop_body_invoked = False
    if result.ok:
        loop_body_invoked = True  # pragma: no cover -- not exercised on this path

    assert result.ok is False
    assert loop_body_invoked is False


# ---------------------------------------------------------------------------
# AC4: venv_identity
# ---------------------------------------------------------------------------


def test_venv_identity_ok_when_executable_and_package_match(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    result = run_preflight(
        paths,
        PreflightConfig(),
        disk_usage=_disk_usage_free(50 * 1024**3),
        sys_executable=str(paths.venv_dir / "Scripts" / "python.exe"),
        package_file=str(paths.repo_root / "src" / "charlie_work" / "preflight.py"),
    )
    venv_check = next(c for c in result.checks if c.name == "venv_identity")
    assert venv_check.ok is True
    assert venv_check.fatal is True


def test_venv_identity_fails_on_wrong_executable(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    other_venv_python = tmp_path / "other-repo" / ".venv" / "Scripts" / "python.exe"
    result = run_preflight(
        paths,
        PreflightConfig(),
        disk_usage=_disk_usage_free(50 * 1024**3),
        sys_executable=str(other_venv_python),
        package_file=str(paths.repo_root / "src" / "charlie_work" / "preflight.py"),
    )
    venv_check = next(c for c in result.checks if c.name == "venv_identity")
    assert venv_check.ok is False
    assert venv_check.fatal is True
    # Detail must name both the observed and expected paths (AC4).
    assert str(other_venv_python.resolve()) in venv_check.detail
    assert str(paths.venv_dir.resolve()) in venv_check.detail


def test_venv_identity_fails_on_wrong_checkout(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    wrong_repo_package = tmp_path / "wrong-checkout" / "src" / "charlie_work" / "preflight.py"
    result = run_preflight(
        paths,
        PreflightConfig(),
        disk_usage=_disk_usage_free(50 * 1024**3),
        sys_executable=str(paths.venv_dir / "Scripts" / "python.exe"),
        package_file=str(wrong_repo_package),
    )
    venv_check = next(c for c in result.checks if c.name == "venv_identity")
    assert venv_check.ok is False
    assert venv_check.fatal is True
    assert str(wrong_repo_package.resolve()) in venv_check.detail
    assert str(paths.repo_root.resolve()) in venv_check.detail


def test_venv_identity_respects_config_fatal_override(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    result = run_preflight(
        paths,
        PreflightConfig(venv_identity_fatal=False),
        disk_usage=_disk_usage_free(50 * 1024**3),
        sys_executable=str(tmp_path / "wrong" / "python.exe"),
        package_file=str(paths.repo_root / "src" / "charlie_work" / "preflight.py"),
    )
    venv_check = next(c for c in result.checks if c.name == "venv_identity")
    assert venv_check.ok is False
    assert venv_check.fatal is False
    assert result.ok is True


# ---------------------------------------------------------------------------
# AC5: config_freshness fires exactly once per change
# ---------------------------------------------------------------------------


def test_config_freshness_first_observation_is_baseline_not_a_change(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    config_file = tmp_path / "charlie.yaml"
    config_file.write_text("a: 1", encoding="utf-8")
    known_mtimes: dict[str, float] = {}

    result = run_preflight(
        paths,
        PreflightConfig(),
        disk_usage=_disk_usage_free(50 * 1024**3),
        sys_executable=str(paths.venv_dir / "Scripts" / "python.exe"),
        package_file=str(paths.repo_root / "src" / "charlie_work" / "preflight.py"),
        config_sources=(str(config_file),),
        known_config_mtimes=known_mtimes,
    )
    freshness_check = next(c for c in result.checks if c.name == "config_freshness")
    assert freshness_check.ok is True
    assert str(config_file) in known_mtimes


def test_config_freshness_flags_exactly_once_per_change(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    config_file = tmp_path / "charlie.yaml"
    config_file.write_text("a: 1", encoding="utf-8")
    known_mtimes: dict[str, float] = {}

    def _run(mtime: float) -> PreflightCheck:
        result = run_preflight(
            paths,
            PreflightConfig(),
            disk_usage=_disk_usage_free(50 * 1024**3),
            stat_fn=lambda p: (
                SimpleNamespace(st_mtime=mtime)
                if Path(p) == config_file
                else SimpleNamespace(st_mtime=datetime.now(UTC).timestamp())
            ),
            sys_executable=str(paths.venv_dir / "Scripts" / "python.exe"),
            package_file=str(paths.repo_root / "src" / "charlie_work" / "preflight.py"),
            config_sources=(str(config_file),),
            known_config_mtimes=known_mtimes,
        )
        return next(c for c in result.checks if c.name == "config_freshness")

    # Pass 1: baseline, mtime=100. Not a change.
    check1 = _run(100.0)
    assert check1.ok is True

    # Pass 2: same mtime. Still ok -- no change since baseline.
    check2 = _run(100.0)
    assert check2.ok is True

    # Pass 3: operator edits the file, mtime changes to 200. Flags once.
    check3 = _run(200.0)
    assert check3.ok is False
    assert check3.fatal is False  # non-fatal per issue spec
    assert "100.0" in check3.detail
    assert "200.0" in check3.detail

    # Pass 4: same new mtime (200). Back to ok -- the change was already
    # reported once; it must not repeat every subsequent pass.
    check4 = _run(200.0)
    assert check4.ok is True

    # Pass 5: edited again, to 300. Flags again -- a NEW change.
    check5 = _run(300.0)
    assert check5.ok is False
    assert "200.0" in check5.detail
    assert "300.0" in check5.detail


def test_config_freshness_no_sources_is_ok(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    result = run_preflight(
        paths,
        PreflightConfig(),
        disk_usage=_disk_usage_free(50 * 1024**3),
        sys_executable=str(paths.venv_dir / "Scripts" / "python.exe"),
        package_file=str(paths.repo_root / "src" / "charlie_work" / "preflight.py"),
        config_sources=(),
    )
    freshness_check = next(c for c in result.checks if c.name == "config_freshness")
    assert freshness_check.ok is True


def test_config_freshness_respects_config_fatal_override(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    config_file = tmp_path / "charlie.yaml"
    config_file.write_text("a: 1", encoding="utf-8")
    known_mtimes = {str(config_file): 100.0}

    result = run_preflight(
        paths,
        PreflightConfig(config_freshness_fatal=True),
        disk_usage=_disk_usage_free(50 * 1024**3),
        stat_fn=lambda p: SimpleNamespace(st_mtime=200.0),
        sys_executable=str(paths.venv_dir / "Scripts" / "python.exe"),
        package_file=str(paths.repo_root / "src" / "charlie_work" / "preflight.py"),
        config_sources=(str(config_file),),
        known_config_mtimes=known_mtimes,
    )
    freshness_check = next(c for c in result.checks if c.name == "config_freshness")
    assert freshness_check.ok is False
    assert freshness_check.fatal is True
    assert result.ok is False


# ---------------------------------------------------------------------------
# clock_sanity (supporting coverage; not one of the four named ACs but part
# of the module's exact-per-issue shape from AC1)
# ---------------------------------------------------------------------------


def test_clock_sanity_ok_for_recent_state_file(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    now = datetime.now(UTC)
    result = run_preflight(
        paths,
        PreflightConfig(),
        now=now,
        disk_usage=_disk_usage_free(50 * 1024**3),
        stat_fn=lambda p: SimpleNamespace(st_mtime=(now - timedelta(hours=1)).timestamp()),
        sys_executable=str(paths.venv_dir / "Scripts" / "python.exe"),
        package_file=str(paths.repo_root / "src" / "charlie_work" / "preflight.py"),
    )
    clock_check = next(c for c in result.checks if c.name == "clock_sanity")
    assert clock_check.ok is True
    assert clock_check.fatal is False


def test_clock_sanity_flags_state_file_from_the_future(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    now = datetime.now(UTC)
    result = run_preflight(
        paths,
        PreflightConfig(),
        now=now,
        disk_usage=_disk_usage_free(50 * 1024**3),
        stat_fn=lambda p: SimpleNamespace(st_mtime=(now + timedelta(hours=1)).timestamp()),
        sys_executable=str(paths.venv_dir / "Scripts" / "python.exe"),
        package_file=str(paths.repo_root / "src" / "charlie_work" / "preflight.py"),
    )
    clock_check = next(c for c in result.checks if c.name == "clock_sanity")
    assert clock_check.ok is False
    # Non-fatal by default per issue spec -- overall result stays ok.
    assert clock_check.fatal is False


def test_clock_sanity_flags_age_beyond_bound(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    now = datetime.now(UTC)
    result = run_preflight(
        paths,
        PreflightConfig(clock_max_skew_hours=1.0),
        now=now,
        disk_usage=_disk_usage_free(50 * 1024**3),
        stat_fn=lambda p: SimpleNamespace(st_mtime=(now - timedelta(hours=2)).timestamp()),
        sys_executable=str(paths.venv_dir / "Scripts" / "python.exe"),
        package_file=str(paths.repo_root / "src" / "charlie_work" / "preflight.py"),
    )
    clock_check = next(c for c in result.checks if c.name == "clock_sanity")
    assert clock_check.ok is False


def test_clock_sanity_missing_state_file_is_ok_not_a_failure(tmp_path: Path) -> None:
    """A fresh checkout with no state.json yet has nothing to measure the
    clock against -- must not spuriously trip the tripwire."""
    paths = _paths(tmp_path)
    result = run_preflight(
        paths,
        PreflightConfig(),
        disk_usage=_disk_usage_free(50 * 1024**3),
        stat_fn=lambda p: (_ for _ in ()).throw(FileNotFoundError(str(p))),
        sys_executable=str(paths.venv_dir / "Scripts" / "python.exe"),
        package_file=str(paths.repo_root / "src" / "charlie_work" / "preflight.py"),
    )
    clock_check = next(c for c in result.checks if c.name == "clock_sanity")
    assert clock_check.ok is True


def test_preflight_config_field_names_match_defaults_in_issue() -> None:
    """Locks the exact field set so a future rename doesn't silently drop a
    knob the issue's config section promises."""
    names = {f.name for f in fields(PreflightConfig)}
    assert names == {
        "disk_floor_gb",
        "disk_floor_fatal",
        "clock_max_skew_hours",
        "clock_sanity_fatal",
        "venv_identity_fatal",
        "config_freshness_fatal",
    }
