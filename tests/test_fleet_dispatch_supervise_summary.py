"""Pass-summary and final-summary rendering for ``run_fleet_supervise``.

Split out of ``tests/test_fleet_dispatch.py`` (issue #1557, Track 1) --
bodies are verbatim relocations; shared helpers and the autouse hermeticity
fixtures live in ``tests/_fleet_dispatch_fixtures.py``.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch
import pytest
from _fleet_dispatch_fixtures import (
    _FakeClock,
    _drained_fleet_result,
    _failed_fleet_result,
    _mixed_fleet_result,
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
)
from charlie_work.config import (
    OrchestratorConfig,
    SupervisorConfig,
)
from charlie_work.fleet_dispatch import run_fleet_supervise


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_final_summary_splits_errored_from_conditions(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
) -> None:
    """Issue #738: the final supervisor summary line also splits the counts.

    The aggregate ``fleet supervisor complete`` line used the same
    ``total_failed_repos`` counter as the per-pass headline and had the same
    defect. It must now report ``N errored, N with conditions`` instead of
    ``N failed repo(s)``.
    """
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_fleet_loop.return_value = _mixed_fleet_result(
        conditions={"owner/repo1": "loop completed with 1 PR error(s)"},
        errored={"owner/repo2": "fleet pass error: RuntimeError: boom"},
    )

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=1, clock=fc.now, sleep=fc.sleep)

    assert "1 errored" in result.message
    assert "1 with conditions" in result.message
    assert "failed repo(s)" not in result.message
    # The return data carries the split counters alongside the legacy sum.
    assert result.data["total_errored_repos"] == 1
    assert result.data["total_conditions_repos"] == 1
    assert result.data["total_failed_repos"] == 2


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_pass_summary_all_errored(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Issue #738: a pass where every repo crashed reports N errored, 0 conditions.

    This is the genuine-outage case the old gauge could not distinguish from
    a routine pass -- two repos both crashing is now unambiguously ``2 errored``.
    """
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_fleet_loop.return_value = _mixed_fleet_result(
        errored={
            "owner/repo1": "fleet pass error: RuntimeError: boom",
            "owner/repo2": "fleet pass error: ConfigError: bad",
        }
    )

    fc = _FakeClock(auto_advance=1.0)
    run_fleet_supervise(max_passes=1, clock=fc.now, sleep=fc.sleep)

    captured = capsys.readouterr()
    summary_lines = [line for line in captured.out.splitlines() if "fleet pass 1:" in line]
    assert summary_lines
    line = summary_lines[0]
    assert "2 errored" in line
    assert "0 with conditions" in line


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_pass_summary_bounds_long_reason(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A single pathological message must not let the log line grow unbounded."""
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    huge_reason = "x" * 5000
    mock_fleet_loop.return_value = _failed_fleet_result({"owner/repo1": huge_reason})

    fc = _FakeClock(auto_advance=1.0)
    run_fleet_supervise(max_passes=1, clock=fc.now, sleep=fc.sleep)

    captured = capsys.readouterr()
    summary_lines = [line for line in captured.out.splitlines() if "fleet pass 1:" in line]
    assert summary_lines
    assert len(summary_lines[0]) < 2000, (
        f"an unbounded per-repo message must not dominate the summary line "
        f"(got {len(summary_lines[0])} chars)"
    )


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_pass_summary_dedupes_repeated_reasons(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """N repos failing for one shared cause must not repeat the string N times."""
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    same_reason = "unauthorized-merge tripwire: unacked finding #502"
    mock_fleet_loop.return_value = _failed_fleet_result(
        {"owner/repo1": same_reason, "owner/repo2": same_reason}
    )

    fc = _FakeClock(auto_advance=1.0)
    run_fleet_supervise(max_passes=1, clock=fc.now, sleep=fc.sleep)

    captured = capsys.readouterr()
    summary_lines = [line for line in captured.out.splitlines() if "fleet pass 1:" in line]
    assert summary_lines
    assert summary_lines[0].count(same_reason) == 1, (
        f"a shared failure reason must be deduped, not repeated per repo: {summary_lines[0]!r}"
    )


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_pass_summary_guards_missing_message(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A failed repo with no/empty/whitespace-only message must not crash or append junk.

    Whitespace-only is the case that a naive ``if r.get("message")`` guard
    (truthy on unstripped text) lets through as a dangling ``" []"`` -- the
    filter must run *after* stripping, not before.
    """
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_fleet_loop.return_value = _failed_fleet_result(
        {"owner/repo1": None, "owner/repo2": "", "owner/repo3": "   "}
    )

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=1, clock=fc.now, sleep=fc.sleep)

    assert result.ok is True  # the supervisor loop itself must not crash
    captured = capsys.readouterr()
    summary_lines = [line for line in captured.out.splitlines() if "fleet pass 1:" in line]
    assert summary_lines
    # Issue #738: non-fatal ok=False conditions are "with conditions", not
    # "failed" -- the three repos here have no ``errored`` flag, so they land
    # in the conditions bucket and the errored count stays at zero.
    assert "3 with conditions" in summary_lines[0]
    assert "0 errored" in summary_lines[0]
    # No dangling empty reason marker when every message is absent -- check
    # only the text after "fleet pass N:" so the leading "[HH:MM:SS]"
    # timestamp bracket (unrelated to the reason suffix) is not confused for it.
    after_prefix = summary_lines[0].split("fleet pass 1:", 1)[1]
    assert "[]" not in after_prefix
    assert not after_prefix.rstrip().endswith("[")


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_pass_summary_includes_failure_reason(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Issue #893: the per-pass summary line must surface *why* repos failed.

    ``repos_data[key]["message"]`` (fleet_dispatch.py:1564) already carries the
    per-repo failure reason -- e.g. "loop completed with N PR error(s)" -- all
    the way to the print site, but the summary line only ever counted ``ok``
    and dropped the message. A repeating "0 ok, N failed" with no reason reads
    identically to a real outage as it does to a known, acked-releasable
    control (e.g. the unauthorized-merge tripwire) firing every pass.
    """
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_fleet_loop.return_value = _failed_fleet_result(
        {"owner/repo1": "loop completed with 2 PR error(s)"}
    )

    fc = _FakeClock(auto_advance=1.0)
    run_fleet_supervise(max_passes=1, clock=fc.now, sleep=fc.sleep)

    captured = capsys.readouterr()
    summary_lines = [line for line in captured.out.splitlines() if "fleet pass 1:" in line]
    assert summary_lines, f"no per-pass summary line found in output: {captured.out!r}"
    assert "loop completed with 2 PR error(s)" in summary_lines[0], (
        f"the failure reason must be on the summary line, not just counted: {summary_lines[0]!r}"
    )


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_pass_summary_logs_non_ok_reason_at_warning(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Issue #738: every non-ok repo's reason must be emitted at WARNING.

    The exception path already logs via ``logger.exception`` inside
    ``fleet_loop``; this covers the non-fatal ``ok=False`` conditions from
    ``app.loop()`` that were previously silent in the supervisor log. The
    reason must be recoverable from the log after the fact, not just from the
    deduped/truncated summary suffix.
    """
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_fleet_loop.return_value = _mixed_fleet_result(
        conditions={"owner/repo1": "loop completed with 2 PR error(s)"},
    )

    fc = _FakeClock(auto_advance=1.0)
    with caplog.at_level(logging.WARNING, logger="charlie_work.fleet_dispatch"):
        run_fleet_supervise(max_passes=1, clock=fc.now, sleep=fc.sleep)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "owner/repo1" in r.getMessage() and "loop completed with 2 PR error(s)" in r.getMessage()
        for r in warnings
    ), f"non-ok repo reason must be logged at WARNING, got: {[r.getMessage() for r in warnings]!r}"


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_pass_summary_silent_when_all_ok(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A fully healthy pass must not grow a reason suffix at all."""
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_fleet_loop.return_value = _drained_fleet_result()

    fc = _FakeClock(auto_advance=1.0)
    run_fleet_supervise(max_passes=1, clock=fc.now, sleep=fc.sleep)

    captured = capsys.readouterr()
    summary_lines = [line for line in captured.out.splitlines() if "fleet pass 1:" in line]
    assert summary_lines
    # Issue #738: a fully healthy pass reports zero errored and zero
    # conditions, not a single "0 failed" blob.
    assert "0 errored" in summary_lines[0]
    assert "0 with conditions" in summary_lines[0]
    # No reason suffix at all when nothing failed -- check only the text after
    # "fleet pass N:" so the leading "[HH:MM:SS]" timestamp bracket is not
    # confused for a (nonexistent) reason marker.
    after_prefix = summary_lines[0].split("fleet pass 1:", 1)[1]
    assert "[" not in after_prefix


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_pass_summary_splits_errored_from_conditions(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Issue #738: the headline must separate lane crashes from non-fatal conditions.

    A pass with one crashed repo (``errored: True``) and one repo that
    completed with a non-fatal condition (``ok=False``, no ``errored``) must
    produce ``1 errored, 1 with conditions`` -- not the old ``2 failed`` that
    painted both red and gated nothing.
    """
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_fleet_loop.return_value = _mixed_fleet_result(
        conditions={"owner/repo1": "loop completed with 2 PR error(s)"},
        errored={"owner/repo2": "fleet pass error: RuntimeError: boom"},
    )

    fc = _FakeClock(auto_advance=1.0)
    run_fleet_supervise(max_passes=1, clock=fc.now, sleep=fc.sleep)

    captured = capsys.readouterr()
    summary_lines = [line for line in captured.out.splitlines() if "fleet pass 1:" in line]
    assert summary_lines
    line = summary_lines[0]
    assert "1 errored" in line, f"crashed repo must count as errored: {line!r}"
    assert "1 with conditions" in line, f"non-fatal repo must count as conditions: {line!r}"
    # The old undifferentiated "failed" count must not appear.
    assert "2 failed" not in line, f"old red-everywhere count must be gone: {line!r}"
