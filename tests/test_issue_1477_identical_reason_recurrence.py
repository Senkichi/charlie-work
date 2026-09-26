"""Issue #1477 (Option A): identical-reason recurrence after a manual
``charlie unescalate`` is promoted to ``reason_class="judgment"``.

The observed loop (issue #1306 / PR #1409): a ``no_op_rework`` escalation
fires, an operator runs ``charlie unescalate`` (which resets
``auto_deescalation_count``), the identical failure re-escalates, the sweep
auto-clears it again (fresh budget), the failure recurs, re-escalates, the
operator clears it again -- an unbounded escalate -> clear -> re-escalate
cycle that burns paid worker sessions forever because the reset cannot tell
"a human fixed the underlying cause" apart from "a human just cleared the
label".

The fix is bookkeeping, in two halves:

- ``OrchestratorApp.unescalate`` stamps ``unescalate_cleared_reason`` /
  ``unescalate_cleared_at`` on the issue entry, recording WHICH escalation
  reason the re-arm cleared and when.
- ``OrchestratorApp._deescalate_mechanical_issue`` checks that marker
  before any clear accounting: if the issue's ``escalation_reason``
  identically matches the cleared reason and the clear happened within
  ``config.deescalation.identical_reason_recurrence_window_minutes``,
  the recurrence is reclassified ``reason_class="judgment"`` (never
  auto-cleared) and moved to ``agent:human-needed`` via the ``escalated``
  label edge -- instead of burning a fresh auto-clear slot re-entering a
  failure the human did not actually fix.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from _fakes_github import FakeGitHub
from _unescalate_fixtures import _app, _dry_run_app, _events
from charlie_work.config import (
    ConfigError,
    DeescalationConfig,
    OrchestratorConfig,
    PostMortemConfig,
    build_config_from_data,
)
from charlie_work.escalation import _escalate_issue
from charlie_work.paths import runtime_paths
from charlie_work.state import PASSIVE_OPEN_STATUS, load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

_REASON = "no_op_rework_attempts_cap_exceeded"


def _seed_escalated_issue(
    app: OrchestratorApp,
    *,
    reason: str = _REASON,
    reason_class: str = "mechanical",
    issue_extra: dict | None = None,
) -> None:
    """Plant issue 123 / PR 456 in the escalated state the fake repo ships."""
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "escalated",
            "escalation_reason": reason,
        }
        issue_entry: dict = {
            "number": 123,
            "status": "escalated",
            "escalation_reason": reason,
            "reason_class": reason_class,
        }
        if issue_extra:
            issue_entry.update(issue_extra)
        state["issues"]["123"] = issue_entry
        save_state(app.paths.state_file, state)


def _reescalate(app: OrchestratorApp, reason: str) -> None:
    """Re-escalate issue 123 the same way a cap/routing call site does."""
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state = _escalate_issue(
            state,
            123,
            reason=reason,
            reason_class="mechanical",
            pr_number=456,
        )
        save_state(app.paths.state_file, state)


def _windowed_app(tmp_path: Path, window_minutes: int) -> OrchestratorApp:
    config = OrchestratorConfig(
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db")),
        deescalation=DeescalationConfig(identical_reason_recurrence_window_minutes=window_minutes),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    return OrchestratorApp(tmp_path, paths, config, FakeGitHub())


# ---------------------------------------------------------------------------
# unescalate() marker stamping
# ---------------------------------------------------------------------------


def test_unescalate_stamps_cleared_reason_markers(tmp_path: Path) -> None:
    """The re-arm must record WHICH escalation reason it cleared and when --
    the de-escalation sweep's recurrence check reads exactly this pair."""
    app = _app(tmp_path)
    _seed_escalated_issue(app)

    result = app.unescalate(pr_number=456)

    assert result.ok is True and result.data["changed"] is True
    issue = load_state(app.paths.state_file)["issues"]["123"]
    assert issue["unescalate_cleared_reason"] == _REASON
    # Stamped at re-arm time; must parse and be fresh (inside the window).
    cleared_at = datetime.fromisoformat(issue["unescalate_cleared_at"].replace("Z", "+00:00"))
    assert datetime.now(UTC) - cleared_at <= timedelta(minutes=1)
    # The reset still ran in full.
    assert issue["status"] == PASSIVE_OPEN_STATUS
    assert "escalation_reason" not in issue
    assert "reason_class" not in issue

    assert result.data["cleared_escalation_reason"] == _REASON
    events = _events(load_state(app.paths.state_file), "unescalate")
    assert len(events) == 1
    assert events[0]["payload"]["cleared_escalation_reason"] == _REASON


def test_unescalate_pops_stale_markers_when_no_reason_cleared(tmp_path: Path) -> None:
    """A re-arm that clears no escalation_reason must drop a stale marker
    pair left by an earlier unescalate, or it could mis-fire on an unrelated
    later episode."""
    app = _app(tmp_path)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "status": "escalated",
            # No escalation_reason -- a bare stuck record.
            "unescalate_cleared_reason": _REASON,
            "unescalate_cleared_at": "2000-01-01T00:00:00Z",
        }
        save_state(app.paths.state_file, state)

    result = app.unescalate(issue_number=123)

    assert result.ok is True and result.data["changed"] is True
    issue = load_state(app.paths.state_file)["issues"]["123"]
    assert "unescalate_cleared_reason" not in issue
    assert "unescalate_cleared_at" not in issue


# ---------------------------------------------------------------------------
# Sweep promotion (the sticky judgment half)
# ---------------------------------------------------------------------------


def test_identical_reason_recurrence_promoted_to_judgment(tmp_path: Path) -> None:
    """The full #1306 shape, end to end: escalate mechanical -> manual
    unescalate -> identical reason re-escalates -> the next sweep pass must
    promote it to ``judgment`` (never auto-cleared) and move it to
    ``agent:human-needed`` rather than clearing it again."""
    app = _app(tmp_path)
    _seed_escalated_issue(app)
    assert app.unescalate(pr_number=456).ok is True
    _reescalate(app, _REASON)

    app._maybe_deescalate_mechanical()

    state = load_state(app.paths.state_file)
    issue = state["issues"]["123"]
    # Promoted: judgment class, still parked in the sink.
    assert issue["status"] == "escalated"
    assert issue["reason_class"] == "judgment"
    assert issue["escalation_reason"] == _REASON
    # Never auto-cleared: no sweep bookkeeping was consumed.
    assert "auto_deescalation_count" not in issue
    assert _events(state, "deescalation_cleared") == []

    # Label edge: operator_queue (applied when the recurrence escalated as
    # mechanical) replaced by human_needed.
    assert (123, app.config.labels.human_needed) in app.gh.labels_added
    assert (123, app.config.labels.operator_queue) in app.gh.labels_removed

    promoted = _events(state, "deescalation_recurrence_promoted")
    assert len(promoted) == 1
    assert promoted[0]["payload"]["issue_number"] == 123
    assert promoted[0]["payload"]["escalation_reason"] == _REASON
    assert promoted[0]["payload"]["unescalate_cleared_at"]
    assert promoted[0]["payload"]["window_minutes"] == 1440

    passes = _events(state, "deescalation_pass_completed")
    assert len(passes) == 1
    assert passes[0]["payload"]["candidates"] == 1
    assert len(passes[0]["payload"]["promoted_to_judgment"]) == 1
    assert passes[0]["payload"]["cleared"] == []


def test_promoted_issue_is_terminal_on_later_passes(tmp_path: Path) -> None:
    """After promotion the issue is no longer a sweep candidate at all --
    no re-promotion, no clear, no cap bookkeeping, forever."""
    app = _app(tmp_path)
    _seed_escalated_issue(
        app,
        issue_extra={
            "unescalate_cleared_reason": _REASON,
            "unescalate_cleared_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        },
    )

    app._maybe_deescalate_mechanical()
    assert load_state(app.paths.state_file)["issues"]["123"]["reason_class"] == "judgment"

    # Force a second sweep pass.
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["deescalation_pass"] = {"next_deescalation_at": "2000-01-01T00:00:00Z"}
        save_state(app.paths.state_file, state)
    app._maybe_deescalate_mechanical()

    state = load_state(app.paths.state_file)
    assert len(_events(state, "deescalation_recurrence_promoted")) == 1
    second_pass = _events(state, "deescalation_pass_completed")[-1]
    assert second_pass["payload"]["candidates"] == 0


def test_identical_reason_promoted_without_open_pr(tmp_path: Path) -> None:
    """Promotion must not depend on PR health: it decides WHAT the
    escalation is, not whether it is safe to clear, so an issue with no
    tracked open PR promotes all the same."""
    app = _app(tmp_path)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "status": "escalated",
            "escalation_reason": "dispatch_failed_cap_exceeded",
            "reason_class": "mechanical",
            "unescalate_cleared_reason": "dispatch_failed_cap_exceeded",
            "unescalate_cleared_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        save_state(app.paths.state_file, state)

    outcome = app._deescalate_mechanical_issue(123)

    assert outcome == {
        "promoted_to_judgment": True,
        "issue_number": 123,
        "escalation_reason": "dispatch_failed_cap_exceeded",
    }
    issue = load_state(app.paths.state_file)["issues"]["123"]
    assert issue["reason_class"] == "judgment"
    assert (123, app.config.labels.human_needed) in app.gh.labels_added


# ---------------------------------------------------------------------------
# Negative / boundary cases -- the marker must not over-promote
# ---------------------------------------------------------------------------


def test_different_reason_recurrence_clears_normally(tmp_path: Path) -> None:
    """A DIFFERENT mechanical reason after a manual unescalate is a new
    failure: the reset budget legitimately applies and the normal
    mergeable+janitor_ok clear proceeds."""
    app = _app(tmp_path)
    _seed_escalated_issue(
        app,
        reason="session_failed_escalated",
        issue_extra={
            "unescalate_cleared_reason": _REASON,
            "unescalate_cleared_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        },
    )

    app._maybe_deescalate_mechanical()

    state = load_state(app.paths.state_file)
    issue = state["issues"]["123"]
    assert issue["status"] == PASSIVE_OPEN_STATUS
    assert issue["auto_deescalation_count"] == 1
    assert _events(state, "deescalation_recurrence_promoted") == []
    assert len(_events(state, "deescalation_cleared")) == 1


def test_stale_marker_outside_window_clears_normally(tmp_path: Path) -> None:
    """A marker older than the recurrence window is not 'shortly after' --
    the issue gets the benefit of the doubt and normal sweep treatment."""
    app = _app(tmp_path)
    stale_at = (datetime.now(UTC) - timedelta(hours=48)).isoformat().replace("+00:00", "Z")
    _seed_escalated_issue(
        app,
        issue_extra={
            "unescalate_cleared_reason": _REASON,
            "unescalate_cleared_at": stale_at,
        },
    )

    app._maybe_deescalate_mechanical()

    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == PASSIVE_OPEN_STATUS
    assert _events(state, "deescalation_recurrence_promoted") == []
    assert len(_events(state, "deescalation_cleared")) == 1


def test_missing_cleared_at_marker_does_not_promote(tmp_path: Path) -> None:
    """A reason marker without its timestamp cannot establish 'shortly
    after' -- fail toward normal mechanical handling, not a guess."""
    app = _app(tmp_path)
    _seed_escalated_issue(
        app,
        issue_extra={"unescalate_cleared_reason": _REASON},
    )

    app._maybe_deescalate_mechanical()

    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == PASSIVE_OPEN_STATUS
    assert _events(state, "deescalation_recurrence_promoted") == []


def test_zero_window_disables_promotion(tmp_path: Path) -> None:
    """``identical_reason_recurrence_window_minutes: 0`` is the kill
    switch: the recurrence is cleared as ordinary mechanical."""
    app = _windowed_app(tmp_path, 0)
    _seed_escalated_issue(
        app,
        issue_extra={
            "unescalate_cleared_reason": _REASON,
            "unescalate_cleared_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        },
    )

    app._maybe_deescalate_mechanical()

    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == PASSIVE_OPEN_STATUS
    assert _events(state, "deescalation_recurrence_promoted") == []


def test_dry_run_does_not_promote(tmp_path: Path) -> None:
    """Under --dry-run the sweep writes nothing, promotion included."""
    app = _dry_run_app(tmp_path)
    _seed_escalated_issue(
        app,
        issue_extra={
            "unescalate_cleared_reason": _REASON,
            "unescalate_cleared_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        },
    )

    outcome = app._deescalate_mechanical_issue(123)

    assert outcome == {"skipped": "dry_run", "issue_number": 123}
    issue = load_state(app.paths.state_file)["issues"]["123"]
    assert issue["reason_class"] == "mechanical"
    assert app.gh.labels_added == []


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def test_recurrence_window_config_default_and_parse() -> None:
    assert DeescalationConfig().identical_reason_recurrence_window_minutes == 1440
    config = build_config_from_data(
        {"deescalation": {"identical_reason_recurrence_window_minutes": 120}}
    )
    assert config.deescalation.identical_reason_recurrence_window_minutes == 120


@pytest.mark.parametrize("bad", [-1, "sixty", True, 1.5])
def test_recurrence_window_config_rejects_bad_values(bad) -> None:
    with pytest.raises(ConfigError, match="identical_reason_recurrence_window_minutes"):
        build_config_from_data(
            {"deescalation": {"identical_reason_recurrence_window_minutes": bad}}
        )
