"""Operator-pinned local verdicts carry forward across the gate's sync-merge (issue #2151).

``record_local_review`` used to take its patch-id from the packet's
``diff.patch`` even when the verdict was pinned to a different (live) head, so
the gate's own base-sync merge failed the patch-id carry-forward and the review
lane voided the approval mid-gate.
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
from charlie_work.janitor import _calculate_patch_id
from charlie_work.local_approval_carry import approval_survives_head_move
from charlie_work.local_lane import branch_diff
from charlie_work.state import load_state_locked

BRANCH = "agent/issue-7-x"


@pytest.fixture
def repo() -> Path:
    return new_repo_root()


def _patch_id(repo: Path, head: str) -> str:
    diff = branch_diff(repo, "main", head)
    assert diff
    return _calculate_patch_id(diff)


def _advance_branch(repo: Path, relpath: str, content: str) -> str:
    _git(repo, "checkout", BRANCH)
    head = _commit_file(repo, relpath, content, f"feat: {relpath}")
    _git(repo, "checkout", "main")
    return head


def _gate_sync_merge(repo: Path) -> str:
    """What the gate's base-sync does: master moves, then is merged into the branch."""
    _commit_file(repo, "unrelated.py", "u = 1\n", "base moved")
    _git(repo, "checkout", BRANCH)
    _git(repo, "merge", "--no-ff", "main", "-m", "merge: sync main")
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "checkout", "main")
    return head


def _zero_diff_branch(repo: Path) -> str:
    """Branch whose three-dot diff against main is empty.

    The content lands on ``main`` by another path, then a sync merge
    advances the merge base past it -- the shape a branch reaches once
    everything it carried is already on the base (issue #2739).
    """
    _make_branch(repo, BRANCH, "a.py", "a = 1\n")
    _commit_file(repo, "a.py", "a = 1\n", "feat: same content landed directly")
    _git(repo, "checkout", BRANCH)
    _git(repo, "merge", "--no-ff", "main", "-m", "merge: sync main")
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "checkout", "main")
    assert branch_diff(repo, "main", BRANCH) == ""
    return head


def _stale_packet_then_operator_approval(repo: Path):
    """Packet built at head A; branch advanced to B; operator approves B."""
    _init_repo(repo)
    issues_dir = repo / "docs" / "issues"
    packet_head = _make_branch(repo, BRANCH, "a.py", "a = 1\n")
    app = _app(repo, issues_dir)
    _parked_issue(app, issues_dir, 7, BRANCH)
    app._local_review_packets()
    approved_head = _advance_branch(repo, "b.py", "b = 1\n")
    result = app.record_local_review(
        7, "approved", reviewed_head=approved_head, verdict_provenance="operator_manual"
    )
    assert result.ok, result.message
    return app, packet_head, approved_head


class TestOperatorVerdictPatchId:
    def test_patch_id_describes_the_pinned_head_not_the_packet(self, repo: Path) -> None:
        app, packet_head, approved_head = _stale_packet_then_operator_approval(repo)

        decision = app._review_decision(7)

        assert decision["reviewed_head_sha"] == approved_head
        assert decision["reviewed_patch_id"] == _patch_id(repo, approved_head)
        assert decision["reviewed_patch_id"] != _patch_id(repo, packet_head)

    def test_sync_merge_after_operator_approval_is_not_re_reviewed(self, repo: Path) -> None:
        app, packet_head, _ = _stale_packet_then_operator_approval(repo)
        _gate_sync_merge(repo)

        outcome = app._local_review_packets()

        assert outcome["packets"] == []
        state = load_state_locked(app.paths.state_file)
        assert state["prs"]["7"]["status"] == "approved"
        assert app._read_packet_head_oid(7) == packet_head

    def test_real_content_change_after_approval_still_invalidates(self, repo: Path) -> None:
        app, _, _ = _stale_packet_then_operator_approval(repo)
        _gate_sync_merge(repo)
        _advance_branch(repo, "c.py", "c = 1\n")

        outcome = app._local_review_packets()

        assert outcome["packets"], "new content must be re-reviewed"
        state = load_state_locked(app.paths.state_file)
        assert state["prs"]["7"]["status"] != "approved"


class TestApprovalSurvivesHeadMove:
    def test_self_heals_a_verdict_recorded_with_the_wrong_patch_id(self, repo: Path) -> None:
        """Verdicts recorded before the fix hold the packet head's patch-id."""
        app, packet_head, approved_head = _stale_packet_then_operator_approval(repo)
        sync_head = _gate_sync_merge(repo)
        decision = {
            **app._review_decision(7),
            "reviewed_patch_id": _patch_id(repo, packet_head),
        }

        assert sync_head != approved_head
        assert approval_survives_head_move(repo, "main", BRANCH, decision)

    def test_rejects_changed_content(self, repo: Path) -> None:
        app, _, _ = _stale_packet_then_operator_approval(repo)
        _advance_branch(repo, "c.py", "c = 1\n")

        assert not approval_survives_head_move(repo, "main", BRANCH, app._review_decision(7))

    def test_empty_live_diff_survives_when_verdict_recorded_empty_patch_id(
        self, repo: Path
    ) -> None:
        """A verdict pinned while the diff was already empty (recorded
        patch-id ``""``) must survive a later sync-merge head move: an empty
        three-dot diff is the "already landed" shape, not a failed diff."""
        _init_repo(repo)
        head = _zero_diff_branch(repo)
        decision = {"reviewed_patch_id": "", "reviewed_head_sha": head}

        _commit_file(repo, "other.py", "o = 1\n", "base moved")
        _git(repo, "checkout", BRANCH)
        _git(repo, "merge", "--no-ff", "main", "-m", "merge: sync main again")
        _git(repo, "checkout", "main")

        assert branch_diff(repo, "main", BRANCH) == ""
        assert approval_survives_head_move(repo, "main", BRANCH, decision)

    def test_empty_live_diff_does_not_match_a_real_patch_id(self, repo: Path) -> None:
        """An emptied diff does not carry a verdict recorded over real
        content -- the landed path (not approval carry-forward) owns that
        shape."""
        _init_repo(repo)
        head = _zero_diff_branch(repo)
        decision = {"reviewed_patch_id": "a-real-patch-id", "reviewed_head_sha": head}

        assert not approval_survives_head_move(repo, "main", BRANCH, decision)

    def test_diff_failure_never_counts_as_empty(self, repo: Path) -> None:
        """``branch_diff`` returning None (unresolvable ref) is not the
        empty string: a failed diff carries no approval, even onto an
        empty-patch-id verdict."""
        _init_repo(repo)
        head = _zero_diff_branch(repo)
        decision = {"reviewed_patch_id": "", "reviewed_head_sha": head}

        assert not approval_survives_head_move(repo, "no-such-base", BRANCH, decision)
