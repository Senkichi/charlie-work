"""Tests for issue #970: the rework-stall escalation must consume
``rework_issue_fetch_skipped``.

``rework_issue_fetch_skipped`` (added by #968, closing #939) recorded *why* an
issue was never dispatched for rework, but had no reader: the one place an
operator looks for that answer -- the stall escalation raised by
``_check_janitor_rework_stall`` (``workflow.py``) -- did not correlate with it.
This file pins the consumer: the escalation's ``janitor_rework_stalled`` event
payload, its ``CommandResult.data``, and its ``CommandResult.message`` now
surface the ``rework_issue_fetch_skipped`` events from the stall window, the
way #940 wired ``unauthorized_merge_check_skipped`` into ``tripwire_status`` as
``last_skipped_reason``.

It also pins the secondary change from #970: ``_build_rework_issue_fetch_skip_payload``
keeps up to 5 *distinct* ``(reason, error_type)`` pairs (modelled on
``summarize_loop_errors``' ``error_details``), not just the first exception as a
single representative -- so a pass that mixes a deleted-issue 404 (stop
retrying) and a transient timeout (retry) does not collapse the two into one.

Skip events carry a real-wall-clock ``ts`` with no way to backdate it through
the public API (``instrumentation.log_event`` always stamps ``_now_iso()``), so
testing the ``stall_since`` window bound requires writing rows into
``events.db`` directly. ``_write_skip_event`` below follows the
``test_tripwire_status_skip_consumer.py`` precedent, mirroring
``instrumentation._SCHEMA_SQL`` exactly so the app's later
``CREATE TABLE IF NOT EXISTS`` is a no-op against the same file.

The stall-escalation setup reuses the ``_conflicting_app`` /
``_set_decision`` helpers from ``test_fix_janitor_routing.py`` (the default
``FakeGitHub`` fixture wires PR #456 <-> issue #123 with a CONFLICTING merge
state), and the same past-timestamp-injection pattern
``test_janitor_conflict_stalled_rework_requested_escalates`` uses to drive the
stall clock past threshold without sleeping.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from charlie_work.config import ReviewConfig
from charlie_work.github import GitHubError
from charlie_work.state import load_state, save_state
from charlie_work.workflow import _build_rework_issue_fetch_skip_payload

from test_fix_janitor_routing import _conflicting_app, _set_decision


def _write_skip_event(
    paths,
    ts: str,
    *,
    issue_numbers: list[int],
    reason: str,
    error_type: str = "GitHubError",
    reasons: list[dict] | None = None,
) -> None:
    """Insert one ``rework_issue_fetch_skipped`` row with a caller-chosen ``ts``.

    Writes directly to ``events.db`` next to ``state.json`` -- see module
    docstring for why the public ``log_event`` API cannot be used here.
    """
    db_path = paths.state_file.parent / "events.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts              TEXT    NOT NULL,
                kind            TEXT    NOT NULL,
                payload         TEXT    NOT NULL,
                repo            TEXT,
                correlation_id  TEXT,
                pr_number       INTEGER,
                issue_number    INTEGER,
                level           TEXT DEFAULT 'info'
            );
            """
        )
        payload_dict: dict = {
            "issue_numbers": issue_numbers,
            "issue_numbers_truncated": 0,
            "reason": reason,
            "error_type": error_type,
        }
        if reasons is not None:
            payload_dict["reasons"] = reasons
            payload_dict["reasons_truncated"] = 0
        payload = json.dumps(payload_dict, sort_keys=True)
        conn.execute(
            """INSERT INTO events
               (ts, kind, payload, repo, correlation_id, pr_number, issue_number, level)
               VALUES (?, 'rework_issue_fetch_skipped', ?, NULL, NULL, NULL, NULL, 'warning')""",
            (ts, payload),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Primary: the stall escalation consumes rework_issue_fetch_skipped.
# ---------------------------------------------------------------------------


def test_stall_escalation_surfaces_fetch_skips_in_payload_and_message(
    tmp_path: Path,
) -> None:
    """The stall escalation must correlate with ``rework_issue_fetch_skipped``.

    An operator reading "rework has not progressed for N cycles" must see
    "...and the last N passes could not fetch issues #A, #B (reason: ...)" in
    the same object -- in the ``janitor_rework_stalled`` event payload, in
    ``CommandResult.data``, and appended to ``CommandResult.message`` -- not
    have to know the event kind exists and query events.db for it.
    """
    app = _conflicting_app(
        tmp_path,
        review=ReviewConfig(max_conflict_rework_attempts=2, rework_stall_minutes=60),
    )
    _set_decision(app, 456, "request_changes")
    result1 = app.review(456)
    assert result1.data["routed_to_rework"] is True

    # Anchor the stall clock 61 minutes ago (past the 60-minute threshold),
    # matching the real first-passive-wait write shape.
    stall_since = (datetime.now(UTC) - timedelta(minutes=61)).isoformat()
    state = load_state(app.paths.state_file)
    state["prs"]["456"]["conflict_rework_attempts_stall_since"] = stall_since
    state["prs"]["456"]["conflict_rework_attempts_stall_head"] = app.gh.prs[0].get("headRefOid")
    save_state(app.paths.state_file, state)

    # Two fetch-skip passes during the stall window -- the second is the most
    # recent and must be the one surfaced (not the first).
    since_dt = datetime.fromisoformat(stall_since)
    _write_skip_event(
        app.paths,
        (since_dt + timedelta(minutes=10)).isoformat(),
        issue_numbers=[123],
        reason="gh: command not found",
    )
    _write_skip_event(
        app.paths,
        (since_dt + timedelta(minutes=30)).isoformat(),
        issue_numbers=[123, 456],
        reason="gh: timeout",
    )

    result2 = app.review(456)

    assert result2.ok is False
    assert result2.data["escalated"] is True

    # CommandResult.data carries the skip summary.
    assert result2.data["rework_fetch_skips"] == 2
    assert result2.data["last_rework_fetch_skip_reason"] == "gh: timeout"
    assert result2.data["last_rework_fetch_skip_issue_numbers"] == [123, 456]
    assert (
        result2.data["last_rework_fetch_skip_at"] == (since_dt + timedelta(minutes=30)).isoformat()
    )

    # The message appends (not substitutes) the warning clause and names the
    # most recent skip, not the first.
    assert "rework stalled" in result2.message
    assert "could not fetch" in result2.message
    assert "2 rework pass(es)" in result2.message
    assert "gh: timeout" in result2.message
    assert "[123, 456]" in result2.message

    # The janitor_rework_stalled event payload carries the same summary.
    state = load_state(app.paths.state_file)
    stalled = [e for e in state["events"] if e["kind"] == "janitor_rework_stalled"]
    assert len(stalled) == 1
    payload = stalled[0]["payload"]
    assert payload["rework_fetch_skips"] == 2
    assert payload["last_rework_fetch_skip_reason"] == "gh: timeout"
    assert payload["last_rework_fetch_skip_issue_numbers"] == [123, 456]


def test_stall_escalation_window_excludes_skips_before_stall_since(
    tmp_path: Path,
) -> None:
    """The semantic core of #970: only fetch skips at/after ``stall_since`` count.

    A skip recorded before the stall clock began says nothing about why *this*
    stall never progressed -- it predates there being a stall to explain.
    Mirrors #940's ``armed_at`` window bound.
    """
    app = _conflicting_app(
        tmp_path,
        review=ReviewConfig(max_conflict_rework_attempts=2, rework_stall_minutes=60),
    )
    _set_decision(app, 456, "request_changes")
    app.review(456)

    stall_since = (datetime.now(UTC) - timedelta(minutes=61)).isoformat()
    state = load_state(app.paths.state_file)
    state["prs"]["456"]["conflict_rework_attempts_stall_since"] = stall_since
    state["prs"]["456"]["conflict_rework_attempts_stall_head"] = app.gh.prs[0].get("headRefOid")
    save_state(app.paths.state_file, state)

    since_dt = datetime.fromisoformat(stall_since)
    # Before the stall began -- must be excluded.
    _write_skip_event(
        app.paths,
        (since_dt - timedelta(hours=2)).isoformat(),
        issue_numbers=[999],
        reason="pre-stall outage",
    )
    # During the stall window -- must be included.
    _write_skip_event(
        app.paths,
        (since_dt + timedelta(minutes=5)).isoformat(),
        issue_numbers=[123],
        reason="in-window outage",
    )

    result = app.review(456)

    assert result.data["escalated"] is True
    assert result.data["rework_fetch_skips"] == 1, (
        "only the in-window skip should count; the pre-stall skip must be excluded"
    )
    assert result.data["last_rework_fetch_skip_reason"] == "in-window outage"


def test_stall_escalation_no_skips_reports_zero_and_omits_warning_clause(
    tmp_path: Path,
) -> None:
    """A stall with no fetch skips must not be decorated with a vacuous clause.

    The base escalation text stays true and self-contained; the zero/None
    fields are present in ``data`` so a consumer never has to guess whether
    a missing key means "not computed" or "zero".
    """
    app = _conflicting_app(
        tmp_path,
        review=ReviewConfig(max_conflict_rework_attempts=2, rework_stall_minutes=60),
    )
    _set_decision(app, 456, "request_changes")
    app.review(456)

    state = load_state(app.paths.state_file)
    state["prs"]["456"]["conflict_rework_attempts_stall_since"] = (
        datetime.now(UTC) - timedelta(minutes=61)
    ).isoformat()
    state["prs"]["456"]["conflict_rework_attempts_stall_head"] = app.gh.prs[0].get("headRefOid")
    save_state(app.paths.state_file, state)

    result = app.review(456)

    assert result.data["escalated"] is True
    assert result.data["rework_fetch_skips"] == 0
    assert result.data["last_rework_fetch_skip_at"] is None
    assert result.data["last_rework_fetch_skip_reason"] is None
    assert result.data["last_rework_fetch_skip_issue_numbers"] is None
    assert "could not fetch" not in result.message
    assert "rework stalled" in result.message


# ---------------------------------------------------------------------------
# Secondary: _build_rework_issue_fetch_skip_payload keeps distinct reasons.
# ---------------------------------------------------------------------------


def test_skip_payload_keeps_distinct_reasons_not_single_representative() -> None:
    """A pass mixing a deleted-issue 404 and a transient timeout must keep both.

    Those two want opposite operator responses (stop retrying vs. retry), so
    collapsing to ``failures[0][1]`` as the single ``reason`` loses the
    distinction #970 asks the payload to preserve. ``reasons`` keeps up to
    ``max_reasons`` distinct ``(reason, error_type)`` pairs, deduped so a
    repeated identical outage does not crowd out a rarer distinct one.
    """
    failures = [
        (101, GitHubError("HTTP 404: Not Found")),
        (102, GitHubError("gh: timeout")),
        (103, GitHubError("HTTP 404: Not Found")),  # dup of 101's root cause
    ]
    payload = _build_rework_issue_fetch_skip_payload(failures)

    # The single-representative fields are retained for backward compat.
    assert payload["reason"] == "HTTP 404: Not Found"
    assert payload["error_type"] == "GitHubError"

    reasons = payload["reasons"]
    assert len(reasons) == 2, "the duplicated 404 must collapse, leaving 2 distinct"
    assert reasons[0] == {"reason": "HTTP 404: Not Found", "error_type": "GitHubError"}
    assert reasons[1] == {"reason": "gh: timeout", "error_type": "GitHubError"}
    assert payload["reasons_truncated"] == 0


def test_skip_payload_reasons_capped_with_truncated_count() -> None:
    """Distinct reasons are capped at ``max_reasons`` with an explicit count."""
    failures = [
        (n, GitHubError(f"err mode {n}"))
        for n in range(1, 9)  # 8 distinct
    ]
    payload = _build_rework_issue_fetch_skip_payload(failures, max_reasons=5)

    assert len(payload["reasons"]) == 5
    assert payload["reasons_truncated"] == 3
    # First-seen order preserved; reasons[0] is the representative.
    assert payload["reasons"][0]["reason"] == "err mode 1"
    assert payload["reason"] == "err mode 1"


def test_skip_payload_empty_failures_has_empty_reasons() -> None:
    """The no-failures branch must produce the same shape with empty lists."""
    payload = _build_rework_issue_fetch_skip_payload([])
    assert payload["reasons"] == []
    assert payload["reasons_truncated"] == 0
    assert payload["reason"] == ""
    assert payload["error_type"] == ""


def test_skip_payload_distinct_reasons_preserve_first_as_representative() -> None:
    """``reasons[0]`` must always equal the ``reason``/``error_type`` representative.

    The representative is ``failures[0]``; the distinct set preserves
    first-seen order, so the two cannot disagree. A reader correlating the
    single ``reason`` with ``reasons`` relies on this.
    """
    failures = [
        (1, GitHubError("first failure")),
        (2, GitHubError("second failure")),
    ]
    payload = _build_rework_issue_fetch_skip_payload(failures)
    assert payload["reason"] == payload["reasons"][0]["reason"]
    assert payload["error_type"] == payload["reasons"][0]["error_type"]
