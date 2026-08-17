"""Regression tests for issue #612.

A reviewer session that dies on Claude Code's account-level session-limit
notice exits with ``result.subtype == "success"`` and produces no verdict.
The notice names a specific reset clock time in an IANA zone, e.g.
"resets 1:20am (America/Los_Angeles)". Before this fix that named reset
time was thrown away — ``match_throttle_tail`` only parses the
"resets in N minutes" form — and the reviewer-quota backoff fell back to
a fixed ``quota_reset_hours`` window, backing off 5h whether the limit
reset in 30 min or 8h.

These tests cover the two remaining gaps from #612 (points 1 and 3 were
already addressed by #652's launch-failure reclassification):

* ``parse_reset_clock_time`` resolves the clock-time form to the next UTC
  occurrence (point 4: back off until the named reset time).
* ``_set_reviewer_quota_exhausted_with_backoff`` uses the parsed reset as
  ``throttled_until`` instead of ``now + quota_reset_hours``.
* Both detection paths (the stalled-review sweep and the launch-time
  quota hit) emit a distinct, queryable ``review_quota_exhausted`` event
  carrying the parsed reset time (point 2).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.config import OrchestratorConfig, ReviewDispatchConfig
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.throttle_signatures import parse_reset_clock_time
from charlie_work.workflow import (
    _detect_and_handle_stalled_reviews,
    _set_reviewer_quota_exhausted_with_backoff,
)

from _helpers import _init_git_repo
from _review_fixtures import _dispatch_reviews_app, _write_review_packet


# Claude Code's session-limit notice, verbatim as observed 2026-07-21.
_SESSION_LIMIT_NOTICE = "You've hit your session limit \u00b7 resets 4:40pm (America/Los_Angeles)"


# ---------------------------------------------------------------------------
# parse_reset_clock_time — pure unit tests
# ---------------------------------------------------------------------------


def test_parse_reset_clock_time_future_today() -> None:
    """A reset clock time later today resolves to today's occurrence in UTC."""
    # 2026-07-28 22:40 UTC == 3:40pm America/Los_Angeles (PDT, UTC-7).
    now = datetime(2026, 7, 28, 22, 40, 0, tzinfo=UTC)
    # 4:40pm PDT is after 3:40pm PDT, so today: 16:40 + 7h = 23:40 UTC.
    reset = parse_reset_clock_time("resets 4:40pm (America/Los_Angeles)", now)
    assert reset == datetime(2026, 7, 28, 23, 40, 0, tzinfo=UTC)


def test_parse_reset_clock_time_already_passed_wraps_to_tomorrow() -> None:
    """A reset clock time already passed today wraps to tomorrow."""
    now = datetime(2026, 7, 28, 22, 40, 0, tzinfo=UTC)  # 3:40pm PDT
    # 1:20am PDT already passed today, so tomorrow: 01:20 + 7h = 08:20 UTC.
    reset = parse_reset_clock_time("resets 1:20am (America/Los_Angeles)", now)
    assert reset == datetime(2026, 7, 29, 8, 20, 0, tzinfo=UTC)


def test_parse_reset_clock_time_12am_midnight() -> None:
    """12am maps to midnight (hour 0), not noon."""
    now = datetime(2026, 7, 28, 22, 40, 0, tzinfo=UTC)  # 3:40pm PDT
    reset = parse_reset_clock_time("resets 12:00am (America/Los_Angeles)", now)
    # midnight PDT already passed today -> tomorrow 00:00 + 7h = 07:00 UTC.
    assert reset == datetime(2026, 7, 29, 7, 0, 0, tzinfo=UTC)


def test_parse_reset_clock_time_12pm_noon() -> None:
    """12pm maps to noon (hour 12), not midnight."""
    now = datetime(2026, 7, 28, 22, 40, 0, tzinfo=UTC)  # 3:40pm PDT
    reset = parse_reset_clock_time("resets 12:00pm (America/Los_Angeles)", now)
    # noon PDT already passed today (it's 3:40pm) -> tomorrow 12:00 + 7h = 19:00 UTC.
    assert reset == datetime(2026, 7, 29, 19, 0, 0, tzinfo=UTC)


def test_parse_reset_clock_time_handles_full_notice() -> None:
    """The parser finds the clock-time form inside the full notice text."""
    now = datetime(2026, 7, 28, 22, 40, 0, tzinfo=UTC)
    reset = parse_reset_clock_time(_SESSION_LIMIT_NOTICE, now)
    assert reset == datetime(2026, 7, 28, 23, 40, 0, tzinfo=UTC)


def test_parse_reset_clock_time_no_clock_form_returns_none() -> None:
    """A tail without the clock-time form returns None (caller falls back)."""
    now = datetime(2026, 7, 28, 22, 40, 0, tzinfo=UTC)
    assert parse_reset_clock_time("resets in 30 minutes", now) is None
    assert parse_reset_clock_time("usage limit exceeded", now) is None
    assert parse_reset_clock_time("", now) is None


def test_parse_reset_clock_time_unknown_zone_returns_none() -> None:
    """An unknown/unavailable IANA zone returns None — never guess the offset."""
    now = datetime(2026, 7, 28, 22, 40, 0, tzinfo=UTC)
    assert parse_reset_clock_time("resets 1:20am (Mars/Olympus)", now) is None


def test_parse_reset_clock_time_resolves_in_named_zone() -> None:
    """The reset wall-clock in the named zone is the stated H:MMam/pm."""
    now = datetime(2026, 7, 28, 22, 40, 0, tzinfo=UTC)
    reset = parse_reset_clock_time("resets 11:00pm (America/Los_Angeles)", now)
    assert reset is not None
    assert reset.astimezone(ZoneInfo("America/Los_Angeles")).strftime("%I:%M%p") == "11:00PM"


# ---------------------------------------------------------------------------
# _set_reviewer_quota_exhausted_with_backoff — reset_at wiring
# ---------------------------------------------------------------------------


def _iso_to_dt(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def test_backoff_uses_parsed_reset_as_throttled_until() -> None:
    """When reset_at is provided, throttled_until is reset_at + resume margin,
    not now + quota_reset_hours (issue #612 point 4)."""
    config = OrchestratorConfig()
    now = datetime(2026, 7, 28, 22, 40, 0, tzinfo=UTC)
    reset_at = datetime(2026, 7, 29, 8, 20, 0, tzinfo=UTC)  # 1:20am PDT tomorrow

    state, record = _set_reviewer_quota_exhausted_with_backoff({}, config, now, reset_at=reset_at)

    margin = timedelta(seconds=config.runtime.throttle_resume_margin_s)
    assert _iso_to_dt(record["throttled_until"]) == reset_at + margin
    assert record["reset_at"] == reset_at.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    # The fixed quota_reset_hours window would have been now + 5h = 03:40 UTC;
    # the parsed reset (08:20 + margin) is later and is what we actually use.
    assert _iso_to_dt(record["throttled_until"]) != now + timedelta(
        hours=config.review_dispatch.quota_reset_hours
    )
    # probe_after still follows the exponential interval, independent of reset_at.
    assert _iso_to_dt(record["probe_after"]) == now + timedelta(minutes=15)


def test_backoff_falls_back_to_fixed_window_without_reset_at() -> None:
    """Without a parsed reset, the fixed quota_reset_hours window is used."""
    config = OrchestratorConfig()
    now = datetime(2026, 7, 28, 22, 40, 0, tzinfo=UTC)

    state, record = _set_reviewer_quota_exhausted_with_backoff({}, config, now)

    assert _iso_to_dt(record["throttled_until"]) == now + timedelta(
        hours=config.review_dispatch.quota_reset_hours
    )
    assert record["reset_at"] is None


def test_backoff_reset_at_overrides_shorter_fixed_window() -> None:
    """A reset sooner than the fixed window is honored (no re-spend into a
    still-closed window AND no over-long stall when the real reset is near)."""
    config = OrchestratorConfig()
    now = datetime(2026, 7, 28, 22, 40, 0, tzinfo=UTC)
    # Reset in 30 minutes — much sooner than the 5h fixed window.
    reset_at = now + timedelta(minutes=30)

    _state, record = _set_reviewer_quota_exhausted_with_backoff({}, config, now, reset_at=reset_at)

    margin = timedelta(seconds=config.runtime.throttle_resume_margin_s)
    assert _iso_to_dt(record["throttled_until"]) == reset_at + margin
    # The fixed window would stall 5h; we back off only ~32 min instead.
    assert _iso_to_dt(record["throttled_until"]) < now + timedelta(
        hours=config.review_dispatch.quota_reset_hours
    )


# ---------------------------------------------------------------------------
# _detect_and_handle_stalled_reviews — event + reset-time backoff
# ---------------------------------------------------------------------------


def _seed_stalled(tmp_path: Path, pr_numbers: list[int]):
    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    config = OrchestratorConfig(review_dispatch=ReviewDispatchConfig(enabled=True))
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    started = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    with state_lock(state_file):
        state = load_state(state_file)
        for pr in pr_numbers:
            state["prs"][str(pr)] = {
                "number": pr,
                "review_dispatch_status": "review_dispatch_dispatched",
                "review_dispatched_at": started,
                "reviewer_pid": 999999999,
                "reviewer_process_start_time": 1.0,
            }
        save_state(state_file, state)
    return repo_root, reviews_dir, config, state_file


def _write_session_limit_reviewer(reviews_dir: Path, pr_number: int, tmp_path: Path) -> Path:
    """A dead reviewer whose log shows the session-limit notice with a clock-time reset."""
    log_path = reviews_dir / f"issue-{pr_number}-review.claude.log"
    log_path.write_text(_SESSION_LIMIT_NOTICE + "\n", encoding="utf-8")
    started = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    sidecar = {
        "issue_number": pr_number,
        "branch": f"agent/issue-{pr_number}-fix",
        "worktree_path": str(tmp_path / "worktrees" / f"issue-{pr_number}"),
        "prompt_path": str(tmp_path / "prompt.md"),
        "command": ["claude", "-p"],
        "pid": 999999999,
        "started_at": started,
        "log_path": str(log_path),
        "error": None,
        "process_start_time": 1.0,
    }
    sidecar_path = reviews_dir / f"issue-{pr_number}.claude.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    return sidecar_path


def test_stalled_sweep_emits_review_quota_exhausted_with_parsed_reset(
    tmp_path: Path,
) -> None:
    """A dead reviewer whose log shows the session-limit notice emits a distinct
    ``review_quota_exhausted`` event carrying the parsed reset time, and backs off
    until that reset (plus the resume margin) instead of a fixed window (issue #612)."""
    repo_root, reviews_dir, config, state_file = _seed_stalled(tmp_path, [100])
    _write_session_limit_reviewer(reviews_dir, 100, tmp_path)

    _detect_and_handle_stalled_reviews(reviews_dir, state_file, config, repo_root)

    state = load_state(state_file)
    quota = state.get("reviewer_quota", {})
    assert quota.get("consecutive_probe_failures") == 1

    qe_events = [e for e in state.get("events", []) if e.get("kind") == "review_quota_exhausted"]
    assert len(qe_events) == 1
    payload = qe_events[0]["payload"]
    assert payload["source"] == "stalled_review_sweep"
    assert payload["consecutive_probe_failures"] == 1
    # reset_at is parsed from the notice and recorded.
    assert payload["reset_at"] is not None
    reset_at = _iso_to_dt(payload["reset_at"])
    # The reset wall-clock in America/Los_Angeles is 4:40pm.
    assert reset_at.astimezone(ZoneInfo("America/Los_Angeles")).strftime("%I:%M%p") == "04:40PM"
    # throttled_until == reset_at + resume margin (the provider's own reset, not a fixed guess).
    margin = timedelta(seconds=config.runtime.throttle_resume_margin_s)
    assert _iso_to_dt(payload["throttled_until"]) == reset_at + margin
    # And the state's throttled_until matches the event.
    assert quota.get("throttled_until") == payload["throttled_until"]


def test_stalled_sweep_falls_back_when_no_clock_reset(tmp_path: Path) -> None:
    """A throttle marker without the clock-time form still emits the event with
    reset_at=None and falls back to the fixed quota_reset_hours window."""
    repo_root, reviews_dir, config, state_file = _seed_stalled(tmp_path, [100])
    # "usage limit" is a throttle_error_marker but carries no clock-time reset.
    log_path = reviews_dir / "issue-100-review.claude.log"
    log_path.write_text("usage limit exceeded\n", encoding="utf-8")
    started = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    sidecar_path = reviews_dir / "issue-100.claude.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": 100,
                "branch": "agent/issue-100-fix",
                "worktree_path": str(tmp_path / "wt"),
                "prompt_path": str(tmp_path / "prompt.md"),
                "command": ["claude", "-p"],
                "pid": 999999999,
                "started_at": started,
                "log_path": str(log_path),
                "error": None,
                "process_start_time": 1.0,
            }
        ),
        encoding="utf-8",
    )

    # frozen_now (issue #828) is injected so the fallback-window assertion
    # below is exact instead of racing wall-clock time under CI runner
    # contention -- no downstream real-clock-dependent step follows in this
    # test, so no future offset is needed (contrast
    # test_loop_classifies_dead_sessions_and_sets_throttle_state in
    # test_charlie_work.py, which offsets +1h because a later dispatch()
    # call reads real wall clock).
    frozen_now = datetime.now(UTC)
    _detect_and_handle_stalled_reviews(reviews_dir, state_file, config, repo_root, now=frozen_now)

    state = load_state(state_file)
    qe_events = [e for e in state.get("events", []) if e.get("kind") == "review_quota_exhausted"]
    assert len(qe_events) == 1
    payload = qe_events[0]["payload"]
    assert payload["reset_at"] is None
    # Fixed window fallback: no clock-time reset was parsed, so
    # _set_reviewer_quota_exhausted_with_backoff falls back to
    # frozen_now + quota_reset_hours exactly (see workflow.py) -- exact
    # equality, no wall-clock tolerance window.
    throttled = _iso_to_dt(payload["throttled_until"])
    expected = (frozen_now + timedelta(hours=config.review_dispatch.quota_reset_hours)).replace(
        microsecond=0
    )
    assert throttled == expected


# ---------------------------------------------------------------------------
# dispatch_reviews launch-time quota hit — event + reset-time backoff
# ---------------------------------------------------------------------------


_PR = {
    "number": 100,
    "title": "Fix #10",
    "url": "https://example.test/pull/100",
    "headRefName": "agent/issue-10-fix",
    "baseRefName": "main",
    "headRefOid": "sha-100",
    "mergeStateStatus": "CLEAN",
    "body": "Closes #10",
    "labels": [],
    "isCrossRepository": False,
    "state": "OPEN",
}


def test_launch_quota_hit_emits_review_quota_exhausted_with_parsed_reset(
    monkeypatch, tmp_path: Path
) -> None:
    """A launch-time quota hit carrying the session-limit notice emits the
    ``review_quota_exhausted`` event with the parsed reset and backs off until
    the named reset time (issue #612)."""
    app = _dispatch_reviews_app(tmp_path, prs=[_PR])
    _write_review_packet(tmp_path, 100, "sha-100")

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        return ClaudeWorkerRecord(
            issue_number=kwargs.get("issue_number") or args[0],
            branch=kwargs.get("branch") or args[1],
            worktree_path="/fake/worktree",
            prompt_path="/fake/prompt.md",
            command=("claude", "-p", "--permission-mode", "plan"),
            pid=None,
            started_at="2026-07-28T22:00:00Z",
            log_path="/fake/log.log",
            error=_SESSION_LIMIT_NOTICE,
            process_start_time=1.0,
        )

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    result = app.dispatch_reviews()
    state = load_state(app.paths.state_file)

    assert result.data.get("quota_hit") is True
    qe_events = [e for e in state.get("events", []) if e.get("kind") == "review_quota_exhausted"]
    assert len(qe_events) == 1
    payload = qe_events[0]["payload"]
    assert payload["source"] == "launch_quota_hit"
    assert payload["reset_at"] is not None
    reset_at = _iso_to_dt(payload["reset_at"])
    assert reset_at.astimezone(ZoneInfo("America/Los_Angeles")).strftime("%I:%M%p") == "04:40PM"
    margin = timedelta(seconds=app.config.runtime.throttle_resume_margin_s)
    assert _iso_to_dt(payload["throttled_until"]) == reset_at + margin
    assert state["reviewer_quota"]["throttled_until"] == payload["throttled_until"]
    assert state["reviewer_quota"]["reset_at"] == payload["reset_at"]


def test_clear_reviewer_quota_clears_reset_at(tmp_path: Path) -> None:
    """A successful verdict reap clears reset_at so a stale value does not
    linger after the quota window is proven open (issue #612)."""
    from charlie_work.state import clear_reviewer_quota

    state = {
        "reviewer_quota": {
            "throttled_until": "2026-07-29T08:21:30Z",
            "probe_after": "2026-07-28T22:55:00Z",
            "alerted_at": "2026-07-28T22:40:00Z",
            "consecutive_probe_failures": 2,
            "reset_at": "2026-07-29T08:20:00Z",
        }
    }
    cleared = clear_reviewer_quota(state)
    quota = cleared["reviewer_quota"]
    assert "throttled_until" not in quota
    assert "probe_after" not in quota
    assert "alerted_at" not in quota
    assert "reset_at" not in quota
    # consecutive_probe_failures is reset separately by dispatch_reviews, not here.
    assert quota.get("consecutive_probe_failures") == 2
