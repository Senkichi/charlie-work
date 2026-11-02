"""``record_local_review`` refusal branches carry stable ``reason`` tokens (issue #2476).

The local-lane mirror of ``test_charlie_work_record_review.py``'s parametrized
guard: ``_reap_review_verdicts`` emits ``result.data["reason"]`` as the
``review_verdict_missed`` event's reason (the free-text message moves to
``detail``), so every ``CommandResult(False, ...)`` branch must set it -- a
branch that forgets it degrades to the opaque ``record_review_refused``
fallback.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _local_lane_fixtures import (
    _app,
    _commit_file,
    _git,
    _init_repo,
    _make_branch,
    _parked_issue,
    new_repo_root,
)
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp


@pytest.fixture
def repo() -> Path:
    return new_repo_root()


def _adopted(repo: Path) -> tuple[OrchestratorApp, str]:
    """Park issue 7's branch and build its review packet.

    Same setup as ``test_local_lane.TestRecordLocalReview._adopted`` -- after
    this, ``prs["7"]`` is a ``local`` lane record in ``reviewing`` status with
    a packet pinned to the branch head.
    """
    _init_repo(repo)
    issues_dir = repo / "docs" / "issues"
    head = _make_branch(repo, "agent/issue-7-x", "a.py", "a = 1\n")
    app = _app(repo, issues_dir)
    _parked_issue(app, issues_dir, 7, "agent/issue-7-x")
    app._local_review_packets()
    return app, head


def _seed_pr_record(app: OrchestratorApp, fields: dict) -> None:
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"][str(fields["number"])] = fields
        save_state(app.paths.state_file, state)


def _set_pr_status(app: OrchestratorApp, pr_number: int, status: str) -> None:
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"][str(pr_number)]["status"] = status
        save_state(app.paths.state_file, state)


@pytest.mark.parametrize(
    ("expected_reason", "call", "prepare"),
    [
        pytest.param("invalid_decision", {"decision": "lgtm"}, None, id="invalid_decision"),
        pytest.param(
            "invalid_verdict_provenance",
            {"verdict_provenance": "made_up"},
            None,
            id="invalid_verdict_provenance",
        ),
        pytest.param(
            "missing_summary",
            {"decision": "request_changes", "summary": ""},
            None,
            id="missing_summary",
        ),
        pytest.param(
            "not_local_record", {"pr_number": 42}, "non_local_record", id="not_local_record"
        ),
        pytest.param("pr_terminal_state", {}, "record_merged", id="pr_terminal_state"),
        pytest.param("pr_escalated", {}, "record_escalated", id="pr_escalated"),
        # The other head_moved_during_build site (``reviewed_head`` equal to
        # the packet head while the live branch moved -- the CAS-guard
        # variant) was tokenized before #2476; this pins the
        # no-``reviewed_head`` divergence site.
        pytest.param(
            "head_moved_during_build", {}, "advance_branch", id="head_moved_during_build"
        ),
        pytest.param(
            "reviewed_head_mismatch",
            {"reviewed_head": "0" * 40},
            None,
            id="reviewed_head_mismatch",
        ),
        pytest.param(
            "no_head_available", {"pr_number": 9}, "branchless_record", id="no_head_available"
        ),
    ],
)
def test_refusals_carry_reason_token(
    repo: Path, expected_reason: str, call: dict, prepare: str | None
) -> None:
    app, head = _adopted(repo)
    if prepare == "non_local_record":
        # prs[42] exists but carries no local marker.
        _seed_pr_record(app, {"number": 42, "status": "reviewing"})
    elif prepare == "branchless_record":
        # A lane record with no branch and no packet: nothing to pin to.
        _seed_pr_record(
            app, {"number": 9, "issue_number": 9, "local": True, "status": "reviewing"}
        )
    elif prepare in ("record_merged", "record_escalated"):
        _set_pr_status(
            app, 7, {"record_merged": "merged", "record_escalated": "escalated"}[prepare]
        )
    elif prepare == "advance_branch":
        # The packet head was the branch tip at packet time; a commit then
        # lands on the branch before the verdict is recorded.
        _git(repo, "checkout", "agent/issue-7-x")
        _commit_file(repo, "b.py", "b = 1\n", "feat: more work")
        _git(repo, "checkout", "main")

    kwargs = {
        "pr_number": 7,
        "decision": "approved",
        "verdict_provenance": "fresh_llm_review",
        **call,
    }
    result = app.record_local_review(**kwargs)

    assert not result.ok
    assert result.data["reason"] == expected_reason
