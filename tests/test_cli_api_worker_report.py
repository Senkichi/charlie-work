"""api-worker fleet report line rendering (``_render_api_worker_report``) and its threading through ``run_fleet_status`` / ``fleet status`` / ``fleet work`` output.

Split out of ``tests/test_cli.py`` (issue #1561, Track 1) -- bodies are
verbatim relocations; shared fakes live in ``tests/_cli_fixtures.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from charlie_work import cli
from charlie_work.fleet_dispatch import ApiWorkerFleetReport
from charlie_work.workflow import CommandResult


# ---------------------------------------------------------------------------
# api-worker fleet report CLI wiring (issue #483)
#
# The standalone compute_api_worker_fleet_report() is tested in
# test_fleet_dispatch.py. These tests cover the CLI-visible wiring the issue
# actually asks for: _render_api_worker_report() and the api_worker_report key
# threaded through run_fleet_status / fleet_loop's CommandResult.data. A silent
# breakage in the rendering/key-lookup path would otherwise ship undetected.
# ---------------------------------------------------------------------------


def _sample_api_worker_report() -> ApiWorkerFleetReport:
    return ApiWorkerFleetReport(
        provider="kimi-k3",
        today_usd=1.50,
        lifetime_usd=7.25,
        cap_usd=15.00,
        live=2,
        enabled_k=1,
        enabled_m=4,
    )


def test_render_api_worker_report_prints_line_when_present(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """_render_api_worker_report prints the line (indented) when a dict with a line is present."""
    report = _sample_api_worker_report().to_dict()
    cli._render_api_worker_report({"api_worker_report": report})
    out = capsys.readouterr().out
    assert "api-worker: kimi-k3" in out
    assert "enabled 1/4 repos" in out
    # Rendered with the two-space indent used in fleet status output.
    assert out.startswith("  ")


def test_render_api_worker_report_silent_when_none(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When api_worker_report is None (no repo configures the section), nothing is printed."""
    cli._render_api_worker_report({"api_worker_report": None})
    assert capsys.readouterr().out == ""


def test_render_api_worker_report_silent_when_key_missing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When the api_worker_report key is absent entirely, nothing is printed."""
    cli._render_api_worker_report({"repos": {}, "errors": []})
    assert capsys.readouterr().out == ""


def test_render_api_worker_report_silent_when_empty_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A dict with a falsy 'line' (e.g. empty string) is omitted, not a blank print."""
    cli._render_api_worker_report({"api_worker_report": {"line": "", "provider": "x"}})
    assert capsys.readouterr().out == ""


def test_render_api_worker_report_silent_when_not_dict(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A non-dict api_worker_report value (defensive) is omitted, not raised."""
    cli._render_api_worker_report({"api_worker_report": "unexpected string"})
    assert capsys.readouterr().out == ""


def test_run_fleet_status_threads_api_worker_report_into_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_fleet_status places compute_api_worker_fleet_report's dict into CommandResult.data.

    The autouse conftest fixture points CHARLIE_WORK_FLEET_DIR at an empty
    per-test directory, so the per-repo status loop iterates an empty registry
    and only the api_worker_report wiring is exercised. Mocking the compute
    function isolates the threading from the standalone-function coverage.
    """
    report = _sample_api_worker_report()
    monkeypatch.setattr(cli, "compute_api_worker_fleet_report", lambda **_k: report)

    args = cli.build_parser().parse_args(["fleet", "status"])
    result = cli.run_fleet_status(args)

    assert result.ok is True
    assert result.data["api_worker_report"] == report.to_dict()
    assert "line" in result.data["api_worker_report"]
    assert result.data["api_worker_report"]["provider"] == "kimi-k3"


def test_run_fleet_status_api_worker_report_none_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no repo configures the section, api_worker_report is None (line omitted)."""
    monkeypatch.setattr(cli, "compute_api_worker_fleet_report", lambda **_k: None)

    args = cli.build_parser().parse_args(["fleet", "status"])
    result = cli.run_fleet_status(args)

    assert result.ok is True
    assert result.data["api_worker_report"] is None


def test_cli_fleet_status_main_renders_api_worker_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end: `charlie fleet status` prints the api-worker line when configured.

    Mocks compute_api_worker_fleet_report so the rendering path is exercised
    without a full fleet setup; the empty registry (autouse fixture) means no
    per-repo status lines are printed, so the api-worker line is isolated.
    """
    report = _sample_api_worker_report()
    monkeypatch.setattr(cli, "compute_api_worker_fleet_report", lambda **_k: report)

    rc = cli.main(["fleet", "status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "api-worker: kimi-k3" in out
    assert "enabled 1/4 repos" in out
    assert "$1.50 today" in out


def test_cli_fleet_status_main_omits_line_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end: `charlie fleet status` prints no api-worker line when unconfigured."""
    monkeypatch.setattr(cli, "compute_api_worker_fleet_report", lambda **_k: None)

    rc = cli.main(["fleet", "status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "api-worker:" not in out


def test_cli_fleet_work_main_renders_api_worker_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end: `charlie fleet work` prints the api-worker line from fleet_loop's data.

    fleet_loop is mocked to return a CommandResult carrying the api_worker_report
    dict, so the work/bash-rats rendering branch in main() is exercised directly.
    """
    report_dict = _sample_api_worker_report().to_dict()
    fleet_loop_mock = MagicMock(
        return_value=CommandResult(
            True,
            "fleet pass complete: 0 repo(s) processed",
            {
                "repos": {},
                "digest": {"count": 0, "orphan_sweep_calls": 0},
                "api_worker_report": report_dict,
            },
        )
    )
    monkeypatch.setattr(cli, "fleet_loop", fleet_loop_mock)
    monkeypatch.setattr(cli, "load_layered_config", lambda *a, **k: None)

    rc = cli.main(["fleet", "work", "--limit", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "api-worker: kimi-k3" in out
    assert "enabled 1/4 repos" in out


def test_cli_fleet_work_main_omits_line_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end: `charlie fleet work` prints no api-worker line when report is None."""
    fleet_loop_mock = MagicMock(
        return_value=CommandResult(
            True,
            "fleet pass complete: 0 repo(s) processed",
            {
                "repos": {},
                "digest": {"count": 0, "orphan_sweep_calls": 0},
                "api_worker_report": None,
            },
        )
    )
    monkeypatch.setattr(cli, "fleet_loop", fleet_loop_mock)
    monkeypatch.setattr(cli, "load_layered_config", lambda *a, **k: None)

    rc = cli.main(["fleet", "work", "--limit", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "api-worker:" not in out
