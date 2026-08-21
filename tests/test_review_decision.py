"""Tests for the single review-decision reader (issue #1362 Stage 1).

Covers the module's own read semantics in isolation -- flat-file precedence,
rounds-directory fallback, staleness, missing/corrupt handling, and the
fail-safe guarantee that a head mismatch or unparseable content is never
reported as "approved". Wiring these results into workflow.py's 24 call
sites is a later stage and is not exercised here.
"""

from __future__ import annotations

import json
from pathlib import Path

from charlie_work.review_decision import ReviewDecision, review_decision


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_flat_file_read_matching_head_is_not_stale(tmp_path: Path) -> None:
    pr_dir = tmp_path / "pr-1"
    _write_json(
        pr_dir / "review-decision.json",
        {"decision": "approved", "reviewed_head_sha": "sha-1"},
    )

    result = review_decision(pr_dir, pr_state={}, current_head_sha="sha-1")

    assert result == ReviewDecision(
        decision="approved",
        reviewed_head_sha="sha-1",
        recorded_at=None,
        source_round=None,
        stale=False,
        missing=False,
    )


def test_flat_file_wins_over_rounds_when_both_present(tmp_path: Path) -> None:
    pr_dir = tmp_path / "pr-1"
    _write_json(
        pr_dir / "review-decision.json",
        {"decision": "pending", "reviewed_head_sha": "sha-1"},
    )
    _write_json(
        pr_dir / "rounds" / "round-1" / "review-decision.json",
        {"decision": "approved", "reviewed_head_sha": "sha-1"},
    )

    result = review_decision(pr_dir, pr_state={}, current_head_sha="sha-1")

    assert result.decision == "pending"
    assert result.source_round is None


def test_rounds_fallback_picks_the_highest_round_when_flat_file_missing(
    tmp_path: Path,
) -> None:
    pr_dir = tmp_path / "pr-1"
    assert not (pr_dir / "review-decision.json").exists()
    _write_json(
        pr_dir / "rounds" / "round-1" / "review-decision.json",
        {"decision": "request_changes", "reviewed_head_sha": "sha-1"},
    )
    _write_json(
        pr_dir / "rounds" / "round-2" / "review-decision.json",
        {"decision": "approved", "reviewed_head_sha": "sha-2"},
    )

    result = review_decision(pr_dir, pr_state={}, current_head_sha="sha-2")

    assert result.decision == "approved"
    assert result.source_round == 2
    assert result.stale is False
    assert result.missing is False


def test_rounds_fallback_used_when_flat_file_is_torn(tmp_path: Path) -> None:
    """AC4 read side: flat file present but truncated/unparseable JSON must
    fall through to the round archive exactly like a missing flat file --
    never raise, never silently report the torn payload."""
    pr_dir = tmp_path / "pr-1"
    flat_path = pr_dir / "review-decision.json"
    flat_path.parent.mkdir(parents=True, exist_ok=True)
    flat_path.write_text('{"decision": "approved", "reviewed_head', encoding="utf-8")
    _write_json(
        pr_dir / "rounds" / "round-1" / "review-decision.json",
        {"decision": "request_changes", "reviewed_head_sha": "sha-1"},
    )

    result = review_decision(pr_dir, pr_state={}, current_head_sha="sha-1")

    assert result.decision == "request_changes"
    assert result.source_round == 1
    assert result.missing is False


def test_stale_on_head_mismatch_never_reports_approved_as_fresh(tmp_path: Path) -> None:
    pr_dir = tmp_path / "pr-1"
    _write_json(
        pr_dir / "review-decision.json",
        {"decision": "approved", "reviewed_head_sha": "sha-old"},
    )

    result = review_decision(pr_dir, pr_state={}, current_head_sha="sha-new")

    assert result.decision == "approved"
    assert result.stale is True
    # The fail-safe contract this reader exists to enforce: a caller must
    # check `not stale` alongside `decision == "approved"`, so express the
    # already_approved-shaped check here directly.
    already_approved = result.decision == "approved" and not result.stale
    assert already_approved is False


def test_missing_when_neither_flat_nor_round_files_exist(tmp_path: Path) -> None:
    pr_dir = tmp_path / "pr-1"

    result = review_decision(pr_dir, pr_state={}, current_head_sha="sha-1")

    assert result == ReviewDecision(
        decision=None,
        reviewed_head_sha=None,
        recorded_at=None,
        source_round=None,
        stale=False,
        missing=True,
    )
    assert result.decision != "approved"


def test_missing_when_rounds_dir_exists_but_every_round_file_is_torn(
    tmp_path: Path,
) -> None:
    """Mirrors the crash-window case ``_round_history_entries`` documents:
    a round directory exists but its decision file never got written (or is
    corrupt) -- must degrade to missing, not raise, and never fabricate a
    decision."""
    pr_dir = tmp_path / "pr-1"
    round_dir = pr_dir / "rounds" / "round-1"
    round_dir.mkdir(parents=True)
    (round_dir / "review-decision.json").write_text("not json{{{", encoding="utf-8")

    result = review_decision(pr_dir, pr_state={}, current_head_sha="sha-1")

    assert result.missing is True
    assert result.decision is None


def test_corrupt_flat_file_never_reports_approved_even_with_no_rounds_fallback(
    tmp_path: Path,
) -> None:
    """Fail-safe contract, isolated: a torn flat file with nothing to fall
    back to must resolve to missing/None, never to a guessed verdict."""
    pr_dir = tmp_path / "pr-1"
    flat_path = pr_dir / "review-decision.json"
    flat_path.parent.mkdir(parents=True, exist_ok=True)
    flat_path.write_text("{", encoding="utf-8")

    result = review_decision(pr_dir, pr_state={}, current_head_sha="sha-1")

    assert result.missing is True
    assert result.decision is None
    assert result.decision != "approved"


def test_pending_decision_is_reported_as_pending_never_terminal(tmp_path: Path) -> None:
    """Regression shape for #1357: an escalated PR's pending placeholder
    must read back as "pending", never coerced into a terminal decision by
    this reader."""
    pr_dir = tmp_path / "pr-1"
    _write_json(pr_dir / "review-decision.json", {"decision": "pending"})

    result = review_decision(pr_dir, pr_state={}, current_head_sha="sha-1")

    assert result.decision == "pending"
    assert result.decision not in ("approved", "request_changes", "blocked")
    assert result.missing is False


def test_state_approved_old_head_but_file_pending_is_not_already_approved(
    tmp_path: Path,
) -> None:
    """Regression shape for #1340: state.json may still carry a stale
    "approved" while the file has already been voided to "pending" after a
    head advance. The file is authoritative for control flow in Stage 1, so
    the reader must report "pending", and `pr_state` must not override it."""
    pr_dir = tmp_path / "pr-1"
    _write_json(
        pr_dir / "review-decision.json",
        {"decision": "pending"},
    )
    stale_state = {"decision": "approved", "reviewed_head_sha": "sha-old"}

    result = review_decision(pr_dir, pr_state=stale_state, current_head_sha="sha-new")

    assert result.decision == "pending"
    already_approved = result.decision == "approved" and not result.stale
    assert already_approved is False
