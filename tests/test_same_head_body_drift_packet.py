"""Same-head body drift must refresh the review packet (issue #1983).

A body-only rework leaves the head unchanged, so ``loop()``'s same-head
packet skip used to keep the packet rendered from the OLD body: the fresh
review read stale text and ``record_review`` stamped ``reviewed_body_sha256``
from that packet, so the drift (and the requeue) repeated forever.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from _review_fixtures import _make_loop_app, _review_queue_app, _write_review_packet
from charlie_work.instrumentation import query_events
from charlie_work.no_op_rework_body import _request_changes_body_drifted

_HEAD = "sha-same"
_OLD_BODY = "Closes #123\n\nTests: stale description."
_NEW_BODY = "Closes #123\n\nTests: corrected description."


def _sha(body: str | None) -> str:
    text = (body or "").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pr(body: str) -> dict:
    return {
        "number": 456,
        "title": "fix: search (#123)",
        "url": "https://example.test/pull/456",
        "headRefName": "agent/issue-123-fix-search",
        "headRefOid": _HEAD,
        "body": body,
        "labels": [],
        "isCrossRepository": False,
    }


def _decision(app) -> dict:
    return json.loads(
        (app.paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )


def test_same_head_body_drift_regenerates_packet_and_converges(tmp_path: Path) -> None:
    """End to end: verdict at OLD body, body-only edit, loop() must regenerate
    the packet, the new verdict is baselined to the LIVE body, and the next
    review_queue() pass queues nothing and emits no further requeue event."""
    app, fake_gh = _make_loop_app(tmp_path, prs=[_pr(_OLD_BODY)])

    assert app.review(456).ok is True
    assert app.record_review(
        456, "request_changes", summary="fix body", verdict_provenance="fresh_llm_review"
    ).ok
    assert _decision(app)["reviewed_body_sha256"] == _sha(_OLD_BODY)

    fake_gh.prs[0]["body"] = _NEW_BODY  # body-only rework: no push

    review_calls: list[int] = []
    original_review = app.review

    def tracking_review(pr_number: int) -> object:
        review_calls.append(pr_number)
        return original_review(pr_number)

    app.review = tracking_review  # type: ignore[method-assign]
    result = app.loop(limit=0)

    assert 456 in review_calls
    assert result.data["skipped_reviews"] == 0
    packet = json.loads((app.paths.prs / "pr-456" / "pr.json").read_text(encoding="utf-8"))
    assert packet["body"] == _NEW_BODY

    assert app.record_review(
        456, "request_changes", summary="still needs work", verdict_provenance="fresh_llm_review"
    ).ok
    assert _decision(app)["reviewed_body_sha256"] == _sha(_NEW_BODY)

    assert app.review_queue().data["queue"] == []
    assert query_events(app.paths.state_file, kind="request_changes_body_changed_requeued") == []


def test_same_head_unchanged_body_still_skips_packet_regen(tmp_path: Path) -> None:
    """No drift -> the same-head packet skip is unchanged."""
    app, _ = _make_loop_app(tmp_path, prs=[_pr(_OLD_BODY)])
    assert app.review(456).ok is True
    assert app.record_review(
        456, "request_changes", summary="fix body", verdict_provenance="fresh_llm_review"
    ).ok

    review_calls: list[int] = []
    app.review = lambda n: review_calls.append(n)  # type: ignore[method-assign]
    result = app.loop(limit=0)

    assert review_calls == []
    assert result.data["skipped_reviews"] == 1


def test_packet_body_current(tmp_path: Path) -> None:
    app = _review_queue_app(tmp_path, prs=[_pr(_NEW_BODY)])
    pr_dir = _write_review_packet(tmp_path, 456, _HEAD)
    # Packet without a body cannot prove currency.
    assert app._packet_body_current(456, {"body": _NEW_BODY}) is False
    # Live payload without a body key: nothing to compare.
    assert app._packet_body_current(456, {}) is True
    (pr_dir / "pr.json").write_text(
        json.dumps({"number": 456, "headRefOid": _HEAD, "body": _OLD_BODY}), encoding="utf-8"
    )
    assert app._packet_body_current(456, {"body": _NEW_BODY}) is False
    assert app._packet_body_current(456, {"body": _OLD_BODY}) is True
    # CRLF normalization matches the wire contract.
    assert app._packet_body_current(456, {"body": _OLD_BODY.replace("\n", "\r\n")}) is True


def test_same_head_drift_with_stale_packet_head_or_template_does_not_queue(
    tmp_path: Path,
) -> None:
    """Drift only queues against a packet at the live head with a current
    template; otherwise the review_queue leaves it for regeneration."""
    decision = {
        "decision": "request_changes",
        "escalated": False,
        "reviewed_head_sha": _HEAD,
        "reviewed_body_sha256": _sha(_OLD_BODY),
        "required_changes": ["fix body"],
        "summary": "fix body",
    }
    # Stale packet head.
    app = _review_queue_app(tmp_path / "a", prs=[_pr(_NEW_BODY)], dry_run=False)
    _write_review_packet(tmp_path / "a", 456, "sha-older", decision)
    assert app.review_queue().data["queue"] == []
    assert query_events(app.paths.state_file, kind="request_changes_body_changed_requeued") == []

    # Stale template digest at the live head.
    app = _review_queue_app(tmp_path / "b", prs=[_pr(_NEW_BODY)], dry_run=False)
    pr_dir = _write_review_packet(tmp_path / "b", 456, _HEAD, decision)
    (pr_dir / "pr.json").write_text(
        json.dumps({"number": 456, "headRefOid": _HEAD, "prompt_template_sha": "stale"}),
        encoding="utf-8",
    )
    assert app.review_queue().data["queue"] == []
    assert query_events(app.paths.state_file, kind="request_changes_body_changed_requeued") == []


@pytest.mark.parametrize(
    ("decision", "pr", "expected"),
    [
        (
            {"decision": "blocked", "reviewed_body_sha256": _sha(_OLD_BODY)},
            {"body": _NEW_BODY},
            False,
        ),
        (
            {"decision": "approved", "reviewed_body_sha256": _sha(_OLD_BODY)},
            {"body": _NEW_BODY},
            False,
        ),
        ({"decision": "request_changes", "reviewed_body_sha256": _sha(_OLD_BODY)}, {}, False),
        ({"decision": "request_changes", "reviewed_body_sha256": ""}, {"body": _NEW_BODY}, False),
        ({"decision": "request_changes"}, {"body": _NEW_BODY}, False),
        (
            {"decision": "request_changes", "reviewed_body_sha256": _sha(_OLD_BODY)},
            {"body": _OLD_BODY},
            False,
        ),
        (
            {"decision": "request_changes", "reviewed_body_sha256": _sha(_OLD_BODY)},
            {"body": _NEW_BODY},
            True,
        ),
        (None, {"body": _NEW_BODY}, False),
    ],
)
def test_request_changes_body_drifted_unit(decision, pr, expected) -> None:
    assert _request_changes_body_drifted(decision, pr) is expected
