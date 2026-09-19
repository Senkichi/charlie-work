"""``check_armable_backlog`` tests for ``scripts/heartbeat_check.py``.

Split out of ``tests/test_heartbeat_check.py`` (issue #1556, Track-1):
runway/armable counting, label-gating (``blocked``/``needs-design``/
``human-action``/``question``/``wontfix``/``duplicate``/``invalid``),
gh-failure anomaly, preview truncation, blocked-err caveat, and the
mutation-arming regression.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from _heartbeat_check_fixtures import (
    _gh_dispatch,
    _load_heartbeat_check,
    _make_repo,
)


@pytest.fixture(scope="module")
def hb() -> ModuleType:
    return _load_heartbeat_check()


# ---------------------------------------------------------------------------
# check_armable_backlog (issue #armable-backlog, added 2026-08-23)
# ---------------------------------------------------------------------------


def _issue(number: int, labels: tuple[str, ...] = ()) -> dict[str, Any]:
    return {"number": number, "labels": [{"name": name} for name in labels]}


def test_check_armable_backlog_ok_when_runway_plenty(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """runway >= floor -> OK, regardless of any armable backlog.

    Uses an explicit dispatch.max_concurrent_sessions cap (floor=3) rather
    than the default, so get_dispatch_cap's config-reading path is exercised
    too, not just the ARMABLE_RUNWAY_FLOOR_DEFAULT fallback.
    """
    repo = _make_repo(hb, tmp_path)
    repo.config_path.write_text("dispatch:\n  max_concurrent_sessions: 3\n", encoding="utf-8")
    issues = [
        _issue(1, ("automated-ready",)),
        _issue(2, ("automated-ready",)),
        _issue(3, ("automated-ready",)),
    ]
    _gh_dispatch(monkeypatch, hb, lambda args, cwd: (True, issues, ""))
    report = hb.Report()
    hb.check_armable_backlog(report, repo, blocked_numbers=None, blocked_err="")
    assert not report.anomaly
    assert "plenty armed" in report.lines[0]
    assert "runway=3 floor=3" in report.lines[0]


def test_check_armable_backlog_ok_when_genuinely_empty(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """runway < floor but no armable issues -> OK, backlog genuinely empty."""
    repo = _make_repo(hb, tmp_path)
    issues = [_issue(1, ("automated-ready",)), _issue(2, ("blocked",))]
    _gh_dispatch(monkeypatch, hb, lambda args, cwd: (True, issues, ""))
    report = hb.Report()
    hb.check_armable_backlog(report, repo, blocked_numbers=None, blocked_err="")
    assert not report.anomaly
    assert "genuinely empty" in report.lines[0]
    assert "runway=1 floor=3 armable=0" in report.lines[0]
    assert "gated=1" in report.lines[0]


def test_check_armable_backlog_anomaly_when_armable_nonempty(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """runway < floor and armable non-empty -> ANOMALY listing numbers + gating labels."""
    repo = _make_repo(hb, tmp_path)
    issues = [_issue(10), _issue(5)]
    _gh_dispatch(monkeypatch, hb, lambda args, cwd: (True, issues, ""))
    report = hb.Report()
    hb.check_armable_backlog(report, repo, blocked_numbers=None, blocked_err="")
    assert report.anomaly
    line = report.lines[0]
    assert "runway thin (0 < floor 3)" in line
    assert "2 un-triaged armable issue(s)" in line
    assert "[5, 10]" in line
    for gating_label in sorted(hb.ARMABLE_GATING_LABELS):
        assert gating_label in line
    assert hb.ARMED_LABEL in line


def test_check_armable_backlog_agent_label_counts_active_never_armable(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """An agent:* label buckets an issue as active, so it can never inflate armable."""
    repo = _make_repo(hb, tmp_path)
    issues = [_issue(7, ("agent:in-progress",))]
    _gh_dispatch(monkeypatch, hb, lambda args, cwd: (True, issues, ""))
    report = hb.Report()
    hb.check_armable_backlog(report, repo, blocked_numbers=None, blocked_err="")
    assert not report.anomaly
    line = report.lines[0]
    assert "genuinely empty" in line
    assert "armable=0" in line
    assert "active=1" in line


@pytest.mark.parametrize(
    "gating_label",
    [
        "blocked",
        "needs-design",
        "human-action",
        "question",
        "wontfix",
        "duplicate",
        "invalid",
    ],
)
def test_check_armable_backlog_gating_label_counts_gated_never_armable(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path, gating_label: str
) -> None:
    """Each gating label buckets an issue as gated, so it can never inflate armable.

    Parametrized over the literal label strings rather than iterating
    hb.ARMABLE_GATING_LABELS directly, so a future accidental narrowing of
    that set (e.g. dropping "wontfix") would fail this test loudly instead
    of silently shrinking the parametrization along with the constant.
    """
    assert gating_label in hb.ARMABLE_GATING_LABELS  # keeps the literal list honest
    repo = _make_repo(hb, tmp_path)
    issues = [_issue(7, (gating_label,))]
    _gh_dispatch(monkeypatch, hb, lambda args, cwd: (True, issues, ""))
    report = hb.Report()
    hb.check_armable_backlog(report, repo, blocked_numbers=None, blocked_err="")
    assert not report.anomaly
    line = report.lines[0]
    assert "genuinely empty" in line
    assert "armable=0" in line
    assert "gated=1" in line


def test_check_armable_backlog_blocked_number_counts_gated_never_armable(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """An issue in blocked_numbers buckets as gated even with no labels at all."""
    repo = _make_repo(hb, tmp_path)
    issues = [_issue(7)]
    _gh_dispatch(monkeypatch, hb, lambda args, cwd: (True, issues, ""))
    report = hb.Report()
    hb.check_armable_backlog(report, repo, blocked_numbers={7}, blocked_err="")
    assert not report.anomaly
    line = report.lines[0]
    assert "genuinely empty" in line
    assert "armable=0" in line
    assert "gated=1" in line


def test_check_armable_backlog_anomaly_when_gh_fails(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    _gh_dispatch(monkeypatch, hb, lambda args, cwd: (False, None, "gh: rate limited"))
    report = hb.Report()
    hb.check_armable_backlog(report, repo, blocked_numbers=None, blocked_err="")
    assert report.anomaly
    assert "gh: rate limited" in report.lines[0]


def test_check_armable_backlog_preview_truncation(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """More than ARMABLE_PREVIEW_LIMIT armable issues -> preview + "(+N more)" suffix."""
    repo = _make_repo(hb, tmp_path)
    total_armable = hb.ARMABLE_PREVIEW_LIMIT + 2
    issues = [_issue(n) for n in range(1, total_armable + 1)]
    _gh_dispatch(monkeypatch, hb, lambda args, cwd: (True, issues, ""))
    report = hb.Report()
    hb.check_armable_backlog(report, repo, blocked_numbers=None, blocked_err="")
    assert report.anomaly
    line = report.lines[0]
    assert f"{total_armable} un-triaged armable issue(s)" in line
    expected_preview = list(range(1, hb.ARMABLE_PREVIEW_LIMIT + 1))
    assert str(expected_preview) in line
    assert "(+2 more)" in line


def test_check_armable_backlog_blocked_err_caveat_on_ok_line(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    issues = [
        _issue(1, ("automated-ready",)),
        _issue(2, ("automated-ready",)),
        _issue(3, ("automated-ready",)),
    ]
    _gh_dispatch(monkeypatch, hb, lambda args, cwd: (True, issues, ""))
    report = hb.Report()
    hb.check_armable_backlog(
        report,
        repo,
        blocked_numbers=None,
        blocked_err="charlie fleet status --json timed out",
    )
    assert not report.anomaly
    assert "plenty armed" in report.lines[0]
    assert (
        "(blocked-issue lookup degraded: charlie fleet status --json timed out)" in report.lines[0]
    )


def test_check_armable_backlog_blocked_err_caveat_on_anomaly_line(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    issues = [_issue(10), _issue(5)]
    _gh_dispatch(monkeypatch, hb, lambda args, cwd: (True, issues, ""))
    report = hb.Report()
    hb.check_armable_backlog(
        report,
        repo,
        blocked_numbers=None,
        blocked_err="blocked-issue lookup failed",
    )
    assert report.anomaly
    assert "(blocked-issue lookup degraded: blocked-issue lookup failed)" in report.lines[0]


def test_check_armable_backlog_mutation_arming_issue_clears_anomaly(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Sanity check on test 3's anomaly: arming the sole armable issue with
    ``automated-ready`` (rather than leaving it un-triaged) moves it out of the
    armable bucket entirely, which must clear the anomaly -- confirming the
    anomaly is actually driven by the un-triaged/armable state and not some
    other coincidental property of this issue.
    """
    repo = _make_repo(hb, tmp_path)
    issues = [_issue(10)]
    _gh_dispatch(monkeypatch, hb, lambda args, cwd: (True, issues, ""))
    report_before = hb.Report()
    hb.check_armable_backlog(report_before, repo, blocked_numbers=None, blocked_err="")
    assert report_before.anomaly

    armed_issues = [_issue(10, ("automated-ready",))]
    _gh_dispatch(monkeypatch, hb, lambda args, cwd: (True, armed_issues, ""))
    report_after = hb.Report()
    hb.check_armable_backlog(report_after, repo, blocked_numbers=None, blocked_err="")
    assert not report_after.anomaly
    assert "genuinely empty" in report_after.lines[0]
    assert "armable=0" in report_after.lines[0]
