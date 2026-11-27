"""A verdict-voiding unescalate must stale a same-head packet's PR body (#3182).

After ``charlie unescalate`` voids a still-valid verdict, the next review of a
PR whose head did not move must be rendered from the LIVE PR body: an operator
decision written into the body -- the resolution the voided verdict was waiting
on -- is exactly what the fresh review exists to read. The reap-loop same-head
packet skip used to keep serving the pre-unescalation packet, so the re-review
repeated the verdict it was meant to resolve (PR #3143).

The trigger is the recorded unescalation -- ``unescalate_verdict_voided_at`` on
the PR record compared against the packet's own ``packet_built_at`` -- never
"the body changed": a passive body edit on a PR nobody unescalated still
rebuilds nothing (#2826's rule).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from _review_fixtures import _make_loop_app

from charlie_work.instrumentation import query_events
from charlie_work.packet_currency import packet_body_current_for_skip
from charlie_work.review_decision import ReviewDecision
from charlie_work.state import load_state
from charlie_work.unescalate_reset_fields import UNESCALATE_VERDICT_VOIDED_AT_KEY

_HEAD = "sha-same"
_OLD_BODY = "Closes #123\n\nTests: stale description."
_NEW_BODY = "Closes #123\n\nTests: corrected description.\n\n## Operator decision\n\nMerge as-is."
_PR_NUMBER = 456
_ISSUE_NUMBER = 123


def _ts(**delta_kwargs: float) -> str:
    """A ``format_utc``-shaped stamp relative to now -- never a literal date."""
    return (
        (datetime.now(UTC) + timedelta(**delta_kwargs))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _pr(body: str) -> dict[str, Any]:
    return {
        "number": _PR_NUMBER,
        "title": "fix: search (#123)",
        "url": "https://example.test/pull/456",
        "headRefName": "agent/issue-123-fix-search",
        "headRefOid": _HEAD,
        "body": body,
        "labels": [],
        "isCrossRepository": False,
    }


def _write_packet(app: Any, *, body: str, built_at: str | None) -> Path:
    """Write the review packet's ``pr.json`` the way ``review()`` lays it down."""
    pr_dir = app.paths.prs / f"pr-{_PR_NUMBER}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"number": _PR_NUMBER, "headRefOid": _HEAD, "body": body}
    if built_at is not None:
        payload["packet_built_at"] = built_at
    (pr_dir / "pr.json").write_text(json.dumps(payload), encoding="utf-8")
    return pr_dir


def _state_with_void(voided_at: str | None) -> dict[str, Any]:
    pr_record: dict[str, Any] = {"number": _PR_NUMBER, "issue_number": _ISSUE_NUMBER}
    if voided_at is not None:
        pr_record[UNESCALATE_VERDICT_VOIDED_AT_KEY] = voided_at
    return {"prs": {str(_PR_NUMBER): pr_record}}


# The decision placeholder unescalate writes over the voided verdict.
_PENDING = ReviewDecision(
    decision="pending",
    reviewed_head_sha=_HEAD,
    recorded_at=None,
    source_round=None,
    stale=False,
    missing=False,
)


def _packet_body(app: Any) -> str:
    packet = json.loads(
        (app.paths.prs / f"pr-{_PR_NUMBER}" / "pr.json").read_text(encoding="utf-8")
    )
    return packet["body"]


# ---------------------------------------------------------------------------
# packet_body_current_for_skip: the #3182 arm
# ---------------------------------------------------------------------------


def test_packet_built_before_void_with_drifted_body_is_not_current(tmp_path: Path) -> None:
    """AC: unescalation recorded later than the packet was built + live body
    differs from the packet body -> not current, and the event fires."""
    app, _gh = _make_loop_app(tmp_path, prs=[_pr(_NEW_BODY)])
    _write_packet(app, body=_OLD_BODY, built_at=_ts(minutes=-10))
    state = _state_with_void(_ts(minutes=-5))

    assert (
        packet_body_current_for_skip(app, state, _PR_NUMBER, {"body": _NEW_BODY}, _PENDING)
        is False
    )
    events = query_events(app.paths.state_file, kind="unescalate_packet_body_stale")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["pr_number"] == _PR_NUMBER
    assert payload["issue_number"] == _ISSUE_NUMBER
    assert payload[UNESCALATE_VERDICT_VOIDED_AT_KEY]


def test_packet_built_before_void_with_identical_body_stays_current(tmp_path: Path) -> None:
    """AC: unescalation recorded later than the packet was built + live body
    identical to the packet body -> current (no pointless rebuild)."""
    app, _gh = _make_loop_app(tmp_path, prs=[_pr(_OLD_BODY)])
    _write_packet(app, body=_OLD_BODY, built_at=_ts(minutes=-10))
    state = _state_with_void(_ts(minutes=-5))

    assert (
        packet_body_current_for_skip(app, state, _PR_NUMBER, {"body": _OLD_BODY}, _PENDING) is True
    )
    assert query_events(app.paths.state_file, kind="unescalate_packet_body_stale") == []


def test_packet_built_after_void_keeps_passive_edit_passive(tmp_path: Path) -> None:
    """A packet rebuilt after the void is exempt: drift against it is an
    unrelated body edit, and #2826's rule rebuilds nothing."""
    app, _gh = _make_loop_app(tmp_path, prs=[_pr(_NEW_BODY)])
    _write_packet(app, body=_OLD_BODY, built_at=_ts(minutes=-1))
    state = _state_with_void(_ts(minutes=-5))

    assert (
        packet_body_current_for_skip(app, state, _PR_NUMBER, {"body": _NEW_BODY}, _PENDING) is True
    )
    assert query_events(app.paths.state_file, kind="unescalate_packet_body_stale") == []


def test_no_void_marker_body_drift_stays_current(tmp_path: Path) -> None:
    """Without an unescalation the #2826 default holds: a passive body edit
    on a verdictless PR rebuilds nothing."""
    app, _gh = _make_loop_app(tmp_path, prs=[_pr(_NEW_BODY)])
    _write_packet(app, body=_OLD_BODY, built_at=_ts(minutes=-10))
    state = _state_with_void(None)

    assert (
        packet_body_current_for_skip(app, state, _PR_NUMBER, {"body": _NEW_BODY}, _PENDING) is True
    )


def test_same_second_void_stales_a_drifted_packet(tmp_path: Path) -> None:
    """``format_utc`` is second-resolution, so a same-second (build, void)
    pair cannot prove the packet postdates the void -- fail safe toward one
    rebuild rather than possibly serving the pre-decision body."""
    stamp = _ts()
    app, _gh = _make_loop_app(tmp_path, prs=[_pr(_NEW_BODY)])
    _write_packet(app, body=_OLD_BODY, built_at=stamp)
    state = _state_with_void(stamp)

    assert (
        packet_body_current_for_skip(app, state, _PR_NUMBER, {"body": _NEW_BODY}, _PENDING)
        is False
    )


def test_legacy_packet_without_built_at_and_drift_is_not_current(tmp_path: Path) -> None:
    """A packet predating the ``packet_built_at`` field cannot prove it
    postdates the void: missing stamp + drift -> not current."""
    app, _gh = _make_loop_app(tmp_path, prs=[_pr(_NEW_BODY)])
    _write_packet(app, body=_OLD_BODY, built_at=None)
    state = _state_with_void(_ts(minutes=-5))

    assert (
        packet_body_current_for_skip(app, state, _PR_NUMBER, {"body": _NEW_BODY}, _PENDING)
        is False
    )


# ---------------------------------------------------------------------------
# End to end: unescalate a blocked PR -> one pass -> packet carries the live body
# ---------------------------------------------------------------------------


def test_unescalate_blocked_pr_rebuilds_packet_with_live_body(tmp_path: Path) -> None:
    """The PR #3143 shape: same head, ``blocked`` verdict voided by unescalate,
    operator decision appended to the live body. One pass must rebuild the
    packet from the live body and emit a ``review_packet`` event for it."""
    app, gh = _make_loop_app(tmp_path, prs=[_pr(_OLD_BODY)])

    assert app.review(_PR_NUMBER).ok is True
    assert _packet_body(app) == _OLD_BODY
    recorded = app.record_review(
        _PR_NUMBER,
        "blocked",
        summary="HUMAN DECISION: the store APIs have no production caller",
        verdict_provenance="fresh_llm_review",
    )
    assert recorded.ok is True, recorded.message
    assert load_state(app.paths.state_file)["prs"][str(_PR_NUMBER)]["status"] == "blocked"

    # The operator answers the verdict where the reviewer reads it: the PR body.
    gh.prs[0]["body"] = _NEW_BODY
    result = app.unescalate(_PR_NUMBER)
    assert result.ok is True
    assert result.data["verdict_voided"] is True
    pr_record = load_state(app.paths.state_file)["prs"][str(_PR_NUMBER)]
    assert pr_record[UNESCALATE_VERDICT_VOIDED_AT_KEY].endswith("Z")

    review_calls: list[int] = []
    original_review = app.review

    def tracking_review(pr_number: int) -> object:
        review_calls.append(pr_number)
        return original_review(pr_number)

    app.review = tracking_review  # type: ignore[method-assign]
    loop_result = app.loop(limit=0)

    assert review_calls == [_PR_NUMBER]
    assert _packet_body(app) == _NEW_BODY
    events = query_events(app.paths.state_file, kind="review_packet")
    assert len(events) == 2  # the initial packet + the forced rebuild
    stale_events = query_events(app.paths.state_file, kind="unescalate_packet_body_stale")
    assert len(stale_events) == 1
    assert stale_events[0]["payload"]["pr_number"] == _PR_NUMBER
    assert loop_result.data["skipped_reviews"] == 0


def test_unescalate_blocked_pr_identical_body_still_skips(tmp_path: Path) -> None:
    """Counterpart: the same unescalate with an UNCHANGED body keeps the
    same-head skip -- the void marker alone never forces a rebuild."""
    app, gh = _make_loop_app(tmp_path, prs=[_pr(_OLD_BODY)])

    assert app.review(_PR_NUMBER).ok is True
    recorded = app.record_review(
        _PR_NUMBER,
        "blocked",
        summary="HUMAN DECISION",
        verdict_provenance="fresh_llm_review",
    )
    assert recorded.ok is True, recorded.message
    result = app.unescalate(_PR_NUMBER)
    assert result.ok is True and result.data["verdict_voided"] is True

    review_calls: list[int] = []
    app.review = lambda n: review_calls.append(n)  # type: ignore[method-assign]
    loop_result = app.loop(limit=0)

    assert review_calls == []
    assert loop_result.data["skipped_reviews"] == 1
    assert query_events(app.paths.state_file, kind="unescalate_packet_body_stale") == []
