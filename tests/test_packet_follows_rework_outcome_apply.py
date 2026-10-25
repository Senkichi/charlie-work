"""A verdictless review packet is rebuilt after a rework outcome apply rewrote the body (#2826).

The reap-loop same-head packet skip keys currency on the head, the template digest and --
for ``request_changes`` verdicts only (#1983) -- the body. With the apply now landing on
every dead-worker route, a PR with no verdict at its head can have its body rewritten
AFTER the packet was built; without a follow-through the reviewer is shown the OLD body
and asks for a fix no worker can make. The trigger is the apply's own record
(``rework_outcome_applied_bodies``), not "the body changed": a body edit nobody applied
through the orchestrator never rebuilds a packet.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from _fakes_github import FakeGitHub
from _review_fixtures import _make_loop_app
from _rework_dispatch_fixtures import _wg

from charlie_work import rework_outcome
from charlie_work.config import WORKER_OUTCOME_FILENAME
from charlie_work.no_op_rework_body import _body_content_sha256
from charlie_work.rework_outcome import APPLIED_BODIES_KEY, applied_body_outdates_packet
from charlie_work.state import load_state, save_state
from charlie_work.worktree import worktree_path_for_branch

_HEAD = "sha-same"
_BRANCH = "agent/issue-123-fix-search"
_OLD_BODY = "Closes #123\n\nTests: stale description."
_NEW_BODY = "Closes #123\n\nTests: corrected description."


def _pr(body: str) -> dict[str, Any]:
    return {
        "number": 456,
        "title": "fix: search (#123)",
        "url": "https://example.test/pull/456",
        "headRefName": _BRANCH,
        "headRefOid": _HEAD,
        "body": body,
        "labels": [],
        "isCrossRepository": False,
    }


def _packet_body(app: Any) -> str:
    packet = json.loads((app.paths.prs / "pr-456" / "pr.json").read_text(encoding="utf-8"))
    return packet["body"]


def _verdictless_packet_app(tmp_path: Path, *, every_route: bool = True) -> tuple[Any, FakeGitHub]:
    """An app whose PR 456 has a packet built from ``_OLD_BODY`` and no verdict."""
    app, gh = _make_loop_app(tmp_path, prs=[_pr(_OLD_BODY)])
    app.config = dataclasses.replace(
        app.config,
        watchdog=dataclasses.replace(
            app.config.watchdog, apply_rework_outcome_on_every_route=every_route
        ),
    )
    assert app.review(456).ok is True
    decision = json.loads(
        (app.paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["decision"] == "pending"  # a packet awaiting its first verdict
    assert _packet_body(app) == _OLD_BODY
    return app, gh


def _apply_outcome(app: Any, gh: FakeGitHub, tmp_path: Path, body: str) -> None:
    """Run the real outcome apply: edits the PR body and records the applied body."""
    state = load_state(app.paths.state_file)
    state["issues"]["123"] = {"status": "pr_open", "branch_name": _BRANCH}
    save_state(app.paths.state_file, state)
    worktree = worktree_path_for_branch(tmp_path, _BRANCH, app.paths.worktrees)
    worktree.mkdir(parents=True, exist_ok=True)
    (worktree / WORKER_OUTCOME_FILENAME).write_text(
        json.dumps(
            {"push_succeeded": True, "pr_created": False, "head_sha": _HEAD, "pr_body": body}
        ),
        encoding="utf-8",
    )
    with patch.object(rework_outcome, "remote_branch_head_sha", lambda *_a: _HEAD):
        rework_outcome.apply_rework_worker_outcome(
            gh,
            repo_root=tmp_path,
            worktrees_dir=app.paths.worktrees,
            sessions_dir=app.paths.root / "sessions",
            state_file=app.paths.state_file,
            write_gate=_wg(app.paths.state_file),
            issue_number=123,
            pr_number=456,
        )
    assert gh.prs[0]["body"] == body


def _loop_reviews(app: Any) -> tuple[list[int], Any]:
    calls: list[int] = []
    original = app.review

    def tracking(pr_number: int) -> object:
        calls.append(pr_number)
        return original(pr_number)

    app.review = tracking
    return calls, app.loop(limit=0)


# ---------------------------------------------------------------------------
# Criterion 7: packet follow-through
# ---------------------------------------------------------------------------


def test_applied_body_rebuilds_a_verdictless_packet_before_reviewer_dispatch(
    tmp_path: Path,
) -> None:
    app, gh = _verdictless_packet_app(tmp_path)
    _apply_outcome(app, gh, tmp_path, _NEW_BODY)
    assert load_state(app.paths.state_file)[APPLIED_BODIES_KEY]["456"] == _body_content_sha256(
        _NEW_BODY
    )

    calls, result = _loop_reviews(app)

    assert calls == [456]
    assert result.data["skipped_reviews"] == 0
    assert _packet_body(app) == _NEW_BODY


def test_rebuilt_packet_is_current_so_the_next_pass_skips_it(tmp_path: Path) -> None:
    app, gh = _verdictless_packet_app(tmp_path)
    _apply_outcome(app, gh, tmp_path, _NEW_BODY)
    first_calls, _ = _loop_reviews(app)
    assert first_calls == [456]  # the rebuild this test converges from

    calls, result = _loop_reviews(app)

    assert calls == []
    assert result.data["skipped_reviews"] == 1


def test_body_edit_with_no_recorded_apply_does_not_rebuild(tmp_path: Path) -> None:
    app, gh = _verdictless_packet_app(tmp_path)
    gh.prs[0]["body"] = _NEW_BODY  # edited, but not through the outcome apply

    calls, result = _loop_reviews(app)

    assert calls == []
    assert result.data["skipped_reviews"] == 1
    assert _packet_body(app) == _OLD_BODY


def test_third_party_edit_after_the_apply_does_not_rebuild(tmp_path: Path) -> None:
    app, gh = _verdictless_packet_app(tmp_path)
    _apply_outcome(app, gh, tmp_path, _NEW_BODY)
    gh.prs[0]["body"] = _NEW_BODY + "\n\nEdited by a human afterwards."

    calls, result = _loop_reviews(app)

    assert calls == []
    assert result.data["skipped_reviews"] == 1


def test_kill_switch_off_leaves_the_packet_alone(tmp_path: Path) -> None:
    app, gh = _verdictless_packet_app(tmp_path, every_route=False)
    _apply_outcome(app, gh, tmp_path, _NEW_BODY)

    calls, result = _loop_reviews(app)

    assert calls == []
    assert result.data["skipped_reviews"] == 1
    assert _packet_body(app) == _OLD_BODY


# ---------------------------------------------------------------------------
# applied_body_outdates_packet: every return path
# ---------------------------------------------------------------------------


def _state_with_applied(pr_number: int, body: str) -> dict[str, Any]:
    return {APPLIED_BODIES_KEY: {str(pr_number): _body_content_sha256(body)}}


@pytest.mark.parametrize(
    ("state", "live_body", "packet_body", "expected"),
    [
        pytest.param(_state_with_applied(456, _NEW_BODY), _NEW_BODY, _OLD_BODY, True, id="stale"),
        pytest.param(_state_with_applied(456, _NEW_BODY), _NEW_BODY, None, True, id="no-body"),
        pytest.param(
            _state_with_applied(456, _NEW_BODY), _NEW_BODY, _NEW_BODY, False, id="packet-current"
        ),
        pytest.param(
            _state_with_applied(456, _NEW_BODY),
            _NEW_BODY.replace("\n", "\r\n"),
            _NEW_BODY,
            False,
            id="crlf-normalized",
        ),
        pytest.param({}, _NEW_BODY, _OLD_BODY, False, id="never-applied"),
        pytest.param(
            _state_with_applied(456, _NEW_BODY), _OLD_BODY, _OLD_BODY, False, id="live-differs"
        ),
        pytest.param(
            _state_with_applied(999, _NEW_BODY), _NEW_BODY, _OLD_BODY, False, id="other-pr"
        ),
        pytest.param({APPLIED_BODIES_KEY: "garbage"}, _NEW_BODY, _OLD_BODY, False, id="bad-map"),
        pytest.param({APPLIED_BODIES_KEY: {"456": ""}}, "", _OLD_BODY, False, id="empty-record"),
        pytest.param(
            {APPLIED_BODIES_KEY: {"456": 7}}, _NEW_BODY, _OLD_BODY, False, id="bad-record"
        ),
    ],
)
def test_applied_body_outdates_packet(
    state: dict[str, Any], live_body: object, packet_body: str | None, expected: bool
) -> None:
    assert applied_body_outdates_packet(state, 456, live_body, packet_body) is expected
