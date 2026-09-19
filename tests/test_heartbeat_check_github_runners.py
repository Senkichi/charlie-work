"""GitHub-rate and runner-check tests for ``scripts/heartbeat_check.py``.

Split out of ``tests/test_heartbeat_check.py`` (issue #1556, Track-1):
``check_github_rate`` ok/low/failure paths and ``check_runners``
Windows-gated ok/anomaly paths.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from _heartbeat_check_fixtures import (
    _gh_dispatch,
    _load_heartbeat_check,
)


@pytest.fixture(scope="module")
def hb() -> ModuleType:
    return _load_heartbeat_check()


def test_check_github_rate_ok(hb: ModuleType, monkeypatch: Any, tmp_path: Path) -> None:
    _gh_dispatch(
        monkeypatch,
        hb,
        lambda args, cwd: (True, {"resources": {"graphql": {"remaining": 5000}}}, ""),
    )
    report = hb.Report()
    hb.check_github_rate(report, tmp_path)
    assert not report.anomaly
    assert "graphql_remaining=5000" in report.lines[0]


def test_check_github_rate_anomaly_when_low(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    _gh_dispatch(
        monkeypatch,
        hb,
        lambda args, cwd: (True, {"resources": {"graphql": {"remaining": 100}}}, ""),
    )
    report = hb.Report()
    hb.check_github_rate(report, tmp_path)
    assert report.anomaly
    assert "below threshold" in report.lines[0]


def test_check_github_rate_anomaly_on_gh_failure(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    _gh_dispatch(
        monkeypatch,
        hb,
        lambda args, cwd: (False, None, "gh exploded"),
    )
    report = hb.Report()
    hb.check_github_rate(report, tmp_path)
    assert report.anomaly


def test_check_runners_skipped_off_windows(hb: ModuleType, monkeypatch: Any) -> None:
    monkeypatch.setattr(hb.sys, "platform", "linux")
    report = hb.Report()
    hb.check_runners(report)
    assert not report.anomaly
    assert "skipped on linux" in report.lines[0]


def test_check_runners_ok_on_windows_with_good_result(hb: ModuleType, monkeypatch: Any) -> None:
    monkeypatch.setattr(hb.sys, "platform", "win32")

    class FakeProc:
        returncode = 0
        stdout = "Last Result: 0\n"
        stderr = ""

    monkeypatch.setattr(hb.subprocess, "run", lambda *a, **k: FakeProc())
    report = hb.Report()
    hb.check_runners(report)
    assert not report.anomaly
    assert "last_result=0" in report.lines[0]


def test_check_runners_anomaly_on_windows_with_bad_result(
    hb: ModuleType, monkeypatch: Any
) -> None:
    monkeypatch.setattr(hb.sys, "platform", "win32")

    class FakeProc:
        returncode = 0
        stdout = "Last Result: 1\n"
        stderr = ""

    monkeypatch.setattr(hb.subprocess, "run", lambda *a, **k: FakeProc())
    report = hb.Report()
    hb.check_runners(report)
    assert report.anomaly
    assert "last run result 1" in report.lines[0]
