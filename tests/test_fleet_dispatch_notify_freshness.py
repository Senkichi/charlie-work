"""Notify-digest staleness dead-man's-switch tests (issue #1859).

Covers ``check_notify_digest_freshness`` (once-per-fleet-pass probe that
emits at most one ``notify_digest_stale`` warning event per staleness-bound
interval while stale), ``report_notify_resolution`` (the once-per-startup
``notify_resolution`` event the heartbeat consumes), their supervisor
wiring, and the emit_digest isolation added to
``_patch_self_deploy_for_fleet_tests`` so suite runs can no longer append
synthetic entries to a checkout's real digest.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from _fleet_dispatch_fixtures import (
    ISOLATED_NOTIFY_DIGEST_REL,
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
    _drained_fleet_result,
)
from charlie_work import layout, notify_freshness
from charlie_work.config import NotifyConfig, OrchestratorConfig
from charlie_work.fleet_dispatch import fleet_loop, run_fleet_supervise
from charlie_work.fleet_paths import fleet_dir
from charlie_work.instrumentation import query_events
from charlie_work.notify_freshness import (
    check_notify_digest_freshness,
    report_notify_resolution,
)
from charlie_work.notify import AttentionDigest, AttentionEntry


@pytest.fixture(autouse=True)
def _clear_stale_latch() -> Any:
    """The episode latch is module state; keep tests independent of order."""
    notify_freshness._stale_episodes.clear()
    yield
    notify_freshness._stale_episodes.clear()


def _config(tmp_path: Path, **kwargs: Any) -> NotifyConfig:
    defaults: dict[str, Any] = dict(
        enabled=True, sink="file", file_path=str(tmp_path / "digest.jsonl")
    )
    defaults.update(kwargs)
    return NotifyConfig(**defaults)


def _patch_log_event(monkeypatch: Any) -> MagicMock:
    log_event = MagicMock(name="log_event")
    monkeypatch.setattr("charlie_work.notify_freshness.log_event", log_event)
    return log_event


def test_stale_check_emits_event_when_digest_missing(tmp_path: Path, monkeypatch: Any) -> None:
    log_event = _patch_log_event(monkeypatch)
    check_notify_digest_freshness(_config(tmp_path), tmp_path / "state.json", now=1000.0)
    log_event.assert_called_once()
    args = log_event.call_args[0]
    assert args[0] == tmp_path / "state.json"
    assert args[1] == "notify_digest_stale"
    assert args[2]["exists"] is False


def test_stale_check_emits_event_when_digest_older_than_bound(
    tmp_path: Path, monkeypatch: Any
) -> None:
    digest = tmp_path / "digest.jsonl"
    digest.write_text("{}\n", encoding="utf-8")
    stale_instant = digest.stat().st_mtime + notify_freshness.NOTIFY_DIGEST_STALE_SECONDS + 1
    log_event = _patch_log_event(monkeypatch)
    check_notify_digest_freshness(_config(tmp_path), tmp_path / "state.json", now=stale_instant)
    log_event.assert_called_once()
    assert log_event.call_args[0][1] == "notify_digest_stale"


def test_stale_check_silent_when_digest_fresh(tmp_path: Path, monkeypatch: Any) -> None:
    digest = tmp_path / "digest.jsonl"
    digest.write_text("{}\n", encoding="utf-8")
    log_event = _patch_log_event(monkeypatch)
    check_notify_digest_freshness(
        _config(tmp_path), tmp_path / "state.json", now=digest.stat().st_mtime + 60
    )
    log_event.assert_not_called()


def test_stale_check_emits_once_per_episode(tmp_path: Path, monkeypatch: Any) -> None:
    """A persistently-stale digest fires at most one event per staleness-bound
    interval, not one per pass."""
    log_event = _patch_log_event(monkeypatch)
    cfg = _config(tmp_path)
    check_notify_digest_freshness(cfg, tmp_path / "state.json", now=1000.0)
    check_notify_digest_freshness(cfg, tmp_path / "state.json", now=2000.0)
    log_event.assert_called_once()


def test_stale_check_refires_after_bound(tmp_path: Path, monkeypatch: Any) -> None:
    """The latch is per-bound-interval, not once-ever: still stale after
    NOTIFY_DIGEST_STALE_SECONDS -> a second event, so a long-lived dead
    writer keeps a fresh event inside the heartbeat's lookback window."""
    log_event = _patch_log_event(monkeypatch)
    cfg = _config(tmp_path)
    check_notify_digest_freshness(cfg, tmp_path / "state.json", now=1000.0)
    check_notify_digest_freshness(
        cfg,
        tmp_path / "state.json",
        now=1000.0 + notify_freshness.NOTIFY_DIGEST_STALE_SECONDS + 1,
    )
    assert log_event.call_count == 2


def test_stale_check_event_carries_resolved_path(tmp_path: Path, monkeypatch: Any) -> None:
    """The event payload publishes the absolute path the daemon resolves,
    so the heartbeat never re-derives the location from its own checkout."""
    log_event = _patch_log_event(monkeypatch)
    check_notify_digest_freshness(_config(tmp_path), tmp_path / "state.json", now=1000.0)
    payload = log_event.call_args[0][2]
    assert payload["resolved_file_path"] == str((tmp_path / "digest.jsonl").resolve())


def test_stale_check_refires_after_recovery(tmp_path: Path, monkeypatch: Any) -> None:
    """A fresh digest clears the episode so a *new* outage re-fires."""
    digest = tmp_path / "digest.jsonl"
    log_event = _patch_log_event(monkeypatch)
    cfg = _config(tmp_path)
    check_notify_digest_freshness(cfg, tmp_path / "state.json", now=1000.0)
    digest.write_text("{}\n", encoding="utf-8")
    fresh_instant = digest.stat().st_mtime + 60
    check_notify_digest_freshness(cfg, tmp_path / "state.json", now=fresh_instant)
    assert log_event.call_count == 1
    digest.unlink()
    check_notify_digest_freshness(cfg, tmp_path / "state.json", now=fresh_instant + 60)
    assert log_event.call_count == 2


def test_stale_check_noop_for_non_file_sink(tmp_path: Path, monkeypatch: Any) -> None:
    log_event = _patch_log_event(monkeypatch)
    check_notify_digest_freshness(
        _config(tmp_path, sink="webhook", webhook_url="https://example.test/hook"),
        tmp_path / "state.json",
        now=1000.0,
    )
    log_event.assert_not_called()


def test_stale_check_noop_when_file_path_unset(tmp_path: Path, monkeypatch: Any) -> None:
    """enabled/file/empty-path is reported at supervisor startup instead."""
    log_event = _patch_log_event(monkeypatch)
    check_notify_digest_freshness(
        _config(tmp_path, file_path=""), tmp_path / "state.json", now=1000.0
    )
    log_event.assert_not_called()


def test_stale_check_swallows_probe_errors(tmp_path: Path, monkeypatch: Any) -> None:
    """The check itself must never take down the pass it rides on."""
    monkeypatch.setattr(
        "charlie_work.notify_freshness.digest_file_status",
        MagicMock(side_effect=RuntimeError("boom")),
    )
    check_notify_digest_freshness(_config(tmp_path), tmp_path / "state.json", now=1000.0)


def test_stale_check_swallows_log_event_errors(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "charlie_work.notify_freshness.log_event",
        MagicMock(side_effect=RuntimeError("db gone")),
    )
    check_notify_digest_freshness(_config(tmp_path), tmp_path / "state.json", now=1000.0)


# ---------------------------------------------------------------------------
# fleet_loop wiring gate
# ---------------------------------------------------------------------------


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_runs_stale_check_when_notify_enabled(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    mock_load_registry.return_value = {"repos": {}}
    check = MagicMock(name="check_notify_digest_freshness")
    monkeypatch.setattr("charlie_work.fleet_dispatch.check_notify_digest_freshness", check)
    notify = NotifyConfig(enabled=True, sink="file", file_path="digest.jsonl")
    fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=SimpleNamespace(notify=notify),
        repos=None,
        limit=1,
        merge=None,
        dry_run=True,
        work_only=True,
    )
    check.assert_called_once()
    assert check.call_args[0][0] is notify


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_skips_stale_check_when_notify_disabled(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    mock_load_registry.return_value = {"repos": {}}
    check = MagicMock(name="check_notify_digest_freshness")
    monkeypatch.setattr("charlie_work.fleet_dispatch.check_notify_digest_freshness", check)
    fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=SimpleNamespace(notify=NotifyConfig()),
        repos=None,
        limit=1,
        merge=None,
        dry_run=True,
        work_only=True,
    )
    check.assert_not_called()


# ---------------------------------------------------------------------------
# Fixture isolation: the "test no-op" leak into the real digest (issue #1859)
# ---------------------------------------------------------------------------


def test_autouse_fixture_redirects_file_sink_to_tmp_path(tmp_path: Path) -> None:
    """Regression: fleet emits under the suite land in the test's tmp dir, never
    at the config's file_path -- the leak that wrote ``"test no-op"`` entries
    into the development checkout's real digest for a month."""
    # Import inside the test body so the name binds the autouse-patched
    # wrapper, not the real emit_digest captured at collection time.
    from charlie_work.fleet_dispatch import emit_digest

    real_path = tmp_path / "real-checkout" / "digest.jsonl"
    config = NotifyConfig(enabled=True, sink="file", file_path=str(real_path))
    digest = AttentionDigest(
        generated_at="2026-09-18T00:00:00Z",
        repo="fleet",
        transitions=(
            AttentionEntry(
                issue_number=1,
                adapter_kind="owner/repo",
                health="STALLED",
                previous_health="RUNNING",
                last_log_line="test no-op",
                pid=None,
            ),
        ),
    )
    result = emit_digest(config, digest)
    assert result.ok is True
    assert not real_path.exists()
    isolated = tmp_path / ISOLATED_NOTIFY_DIGEST_REL
    assert isolated.exists()
    assert "test no-op" in isolated.read_text(encoding="utf-8")


def test_autouse_fixture_noops_non_file_sinks(tmp_path: Path) -> None:
    """Shell/desktop/webhook sinks are real side effects too; the fixture
    suppresses them rather than redirecting."""
    from charlie_work.fleet_dispatch import emit_digest

    digest = AttentionDigest(
        generated_at="2026-09-18T00:00:00Z",
        repo="fleet",
        transitions=(),
    )
    config = NotifyConfig(enabled=True, sink="shell", shell_command=("false",))
    result = emit_digest(config, digest)
    assert result.ok is True


def test_autouse_fixture_preserves_empty_file_path_error(tmp_path: Path) -> None:
    """enabled/file/empty-path still returns the real 'file_path is empty'
    error -- the redirect must not mask the misconfiguration."""
    from charlie_work.fleet_dispatch import emit_digest

    digest = AttentionDigest(
        generated_at="2026-09-18T00:00:00Z",
        repo="fleet",
        transitions=(),
    )
    config = NotifyConfig(enabled=True, sink="file")
    result = emit_digest(config, digest)
    assert result.ok is False
    assert "file_path is empty" in (result.error or "")


# ---------------------------------------------------------------------------
# report_notify_resolution: the notify_resolution event + startup wiring
# ---------------------------------------------------------------------------


def _resolution_events(state_path: Path) -> list[dict[str, Any]]:
    return query_events(state_path, kind="notify_resolution")


def test_report_notify_resolution_none_config_logs_info_and_emits(
    tmp_path: Path, caplog: Any
) -> None:
    """No notify section -> INFO readout, and the event still publishes
    enabled=False so the heartbeat learns the daemon resolved 'off'
    instead of learning nothing."""
    state_path = tmp_path / "state.json"
    with caplog.at_level(logging.INFO, logger="charlie_work.notify_freshness"):
        report_notify_resolution(None, tmp_path / "config.yaml", state_path)
    assert any("no notify section" in r.getMessage() for r in caplog.records)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    events = _resolution_events(state_path)
    assert len(events) == 1
    assert events[0]["payload"]["enabled"] is False


def test_report_notify_resolution_disabled_logs_info(tmp_path: Path, caplog: Any) -> None:
    """enabled=False is a legitimate opt-out: INFO, never WARNING."""
    state_path = tmp_path / "state.json"
    with caplog.at_level(logging.INFO, logger="charlie_work.notify_freshness"):
        report_notify_resolution(NotifyConfig(enabled=False), tmp_path / "config.yaml", state_path)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    events = _resolution_events(state_path)
    assert len(events) == 1
    assert events[0]["payload"]["enabled"] is False
    assert events[0]["level"] == "info"


def test_report_notify_resolution_empty_file_path_warns_and_flags(
    tmp_path: Path, caplog: Any
) -> None:
    """enabled + sink=file + empty file_path -> WARNING log AND a
    warning-level event with file_path_empty=True -- the consumed signal
    the heartbeat anomalies on (the host's daemon resolves exactly this)."""
    state_path = tmp_path / "state.json"
    with caplog.at_level(logging.INFO, logger="charlie_work.notify_freshness"):
        report_notify_resolution(
            NotifyConfig(enabled=True, sink="file"),
            tmp_path / "config.yaml",
            state_path,
        )
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("file_path is unset" in r.getMessage() for r in warnings)
    events = _resolution_events(state_path)
    assert len(events) == 1
    assert events[0]["payload"]["file_path_empty"] is True
    assert events[0]["payload"]["resolved_file_path"] is None
    assert events[0]["level"] == "warning"


def test_report_notify_resolution_enabled_with_path_does_not_warn(
    tmp_path: Path, caplog: Any
) -> None:
    """The coherent enabled+file+path resolution logs INFO only, and the
    event publishes the cwd-anchored absolute path the heartbeat probes."""
    state_path = tmp_path / "state.json"
    digest = tmp_path / "notify" / "digest.jsonl"
    with caplog.at_level(logging.INFO, logger="charlie_work.notify_freshness"):
        report_notify_resolution(
            NotifyConfig(enabled=True, sink="file", file_path=str(digest)),
            tmp_path / "config.yaml",
            state_path,
        )
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    events = _resolution_events(state_path)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["file_path_empty"] is False
    assert payload["resolved_file_path"] == str(digest.resolve())
    assert events[0]["level"] == "info"


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_reports_notify_resolution_at_startup(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """The startup report must be invoked once per supervisor start with the
    resolved notify config and the fleet state path its events.db lives
    beside -- removing the wiring fails this test."""
    cfg = OrchestratorConfig(
        notify=NotifyConfig(enabled=True, sink="file", file_path="digest.jsonl")
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()
    reporter = MagicMock(name="report_notify_resolution")
    monkeypatch.setattr("charlie_work.fleet_dispatch.report_notify_resolution", reporter)

    run_fleet_supervise(
        max_passes=1,
        fleet_dir_override=str(tmp_path / "fleet"),
        clock=lambda: 0.0,
        sleep=lambda _s: None,
    )

    reporter.assert_called_once()
    args = reporter.call_args[0]
    assert args[0] is cfg.notify
    assert args[2] == layout.state_file_path(fleet_dir(override=str(tmp_path / "fleet")))
