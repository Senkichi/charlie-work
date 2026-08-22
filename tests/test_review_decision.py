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

import pytest

from charlie_work import review_decision as review_decision_module
from charlie_work.review_decision import ReviewDecision, record_decision, review_decision


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


# --- record_decision (issue #1362 Stage 2: single writer) ------------------


def test_record_decision_writes_round_file_and_flat_file(tmp_path: Path) -> None:
    pr_dir = tmp_path / "pr-1"
    payload = {
        "decision": "request_changes",
        "summary": "fix the thing",
        "required_changes": ["do X"],
        "verdict_provenance": "fresh_llm_review",
    }

    result = record_decision(pr_dir, payload, "sha-1")

    round_path = pr_dir / "rounds" / "round-1" / "review-decision.json"
    flat_path = pr_dir / "review-decision.json"
    assert round_path.exists()
    assert flat_path.exists()
    round_payload = json.loads(round_path.read_text(encoding="utf-8"))
    flat_payload = json.loads(flat_path.read_text(encoding="utf-8"))
    assert round_payload == flat_payload
    assert flat_payload["reviewed_head_sha"] == "sha-1"
    assert flat_payload["verdict_provenance"] == "fresh_llm_review"
    assert result.decision == "request_changes"
    assert result.reviewed_head_sha == "sha-1"
    assert result.stale is False
    assert result.missing is False


def test_record_decision_stamps_head_sha_overwriting_payload_value(tmp_path: Path) -> None:
    pr_dir = tmp_path / "pr-1"
    payload = {"decision": "approved", "reviewed_head_sha": "stale-sha"}

    record_decision(pr_dir, payload, "fresh-sha")

    flat_payload = json.loads((pr_dir / "review-decision.json").read_text(encoding="utf-8"))
    assert flat_payload["reviewed_head_sha"] == "fresh-sha"


def test_record_decision_leaves_reviewed_head_sha_untouched_when_head_sha_is_none(
    tmp_path: Path,
) -> None:
    pr_dir = tmp_path / "pr-1"
    payload = {"decision": "approved", "reviewed_head_sha": "already-resolved"}

    record_decision(pr_dir, payload, None)

    flat_payload = json.loads((pr_dir / "review-decision.json").read_text(encoding="utf-8"))
    assert flat_payload["reviewed_head_sha"] == "already-resolved"


def test_record_decision_pending_placeholder_is_head_stamped(tmp_path: Path) -> None:
    """Spec shape: even a bare {"decision": "pending"} placeholder becomes
    head-stamped through record_decision, making a pending-for-a-dead-head
    verdict detectable downstream."""
    pr_dir = tmp_path / "pr-1"

    result = record_decision(pr_dir, {"decision": "pending"}, "sha-live")

    flat_payload = json.loads((pr_dir / "review-decision.json").read_text(encoding="utf-8"))
    assert flat_payload["decision"] == "pending"
    assert flat_payload["reviewed_head_sha"] == "sha-live"
    assert result.decision == "pending"
    assert result.reviewed_head_sha == "sha-live"
    assert result.stale is False


def test_record_decision_second_call_same_verdict_same_head_reuses_round(
    tmp_path: Path,
) -> None:
    """A byte-identical retry (same decision/summary/required_changes/head)
    must land back in round-1, never mint round-2 -- mirrors
    ``_next_round_number``'s retry requirement."""
    pr_dir = tmp_path / "pr-1"
    payload = {
        "decision": "request_changes",
        "summary": "fix it",
        "required_changes": ["do X"],
        "verdict_provenance": "fresh_llm_review",
    }

    record_decision(pr_dir, payload, "sha-1")
    record_decision(pr_dir, dict(payload), "sha-1")

    rounds_dir = pr_dir / "rounds"
    assert sorted(p.name for p in rounds_dir.iterdir()) == ["round-1"]


def test_record_decision_second_call_distinct_verdict_mints_new_round(tmp_path: Path) -> None:
    pr_dir = tmp_path / "pr-1"
    record_decision(
        pr_dir,
        {
            "decision": "request_changes",
            "summary": "first pass",
            "required_changes": ["do X"],
            "verdict_provenance": "fresh_llm_review",
        },
        "sha-1",
    )
    record_decision(
        pr_dir,
        {
            "decision": "approved",
            "summary": "looks good now",
            "required_changes": [],
            "verdict_provenance": "fresh_llm_review",
        },
        "sha-2",
    )

    rounds_dir = pr_dir / "rounds"
    assert sorted(p.name for p in rounds_dir.iterdir()) == ["round-1", "round-2"]


def test_pending_placeholder_with_archive_round_false_does_not_shift_real_verdict_round(
    tmp_path: Path,
) -> None:
    """Regression for the F1 review finding: workflow.py's packet-build
    placeholder must pass ``archive_round=False`` so a content-free
    "pending" head-stamp never occupies round-1 ahead of the first real
    reviewer verdict. This mirrors the actual call sequence -- placeholder
    write, then record_review's live verdict write -- at the record_decision
    level, without needing the full workflow.py app fixture."""
    pr_dir = tmp_path / "pr-1"

    # The packet-build placeholder (workflow.py ~:9903), correctly not
    # archiving a round.
    record_decision(pr_dir, {"decision": "pending"}, "sha-1", archive_round=False)

    # The first real reviewer verdict (record_review's write).
    record_decision(
        pr_dir,
        {
            "decision": "request_changes",
            "summary": "first pass",
            "required_changes": ["do X"],
            "verdict_provenance": "fresh_llm_review",
        },
        "sha-1",
    )

    rounds_dir = pr_dir / "rounds"
    # The real verdict must land in round-1, not round-2 -- no phantom
    # "pending" round was minted by the placeholder.
    assert sorted(p.name for p in rounds_dir.iterdir()) == ["round-1"]
    round_payload = json.loads((rounds_dir / "round-1" / "review-decision.json").read_text())
    assert round_payload["decision"] == "request_changes"


def test_merge_authorize_override_before_any_verdict_does_not_mint_a_round(
    tmp_path: Path,
) -> None:
    """Regression for the F2 review finding: merge_authorize's override
    patch (workflow.py ~:13955) must also pass ``archive_round=False``.
    Once the placeholder no longer archives (F1's fix), a PR can reach
    merge_authorize with zero archived rounds; without this flag the
    override patch would become the round-1 minter -- a phantom "reviewer
    round" that contains only an operator override, never a verdict."""
    pr_dir = tmp_path / "pr-1"

    # No prior reviewer verdict and no placeholder round -- decision_path
    # does not even exist yet, matching the F1-fixed placeholder's
    # behavior (or a PR that skipped review entirely).
    updated = {"authorized_override": {"by": "senkichi", "reason": "CI green"}}
    record_decision(pr_dir, updated, None, archive_round=False)

    rounds_dir = pr_dir / "rounds"
    assert not rounds_dir.exists() or list(rounds_dir.iterdir()) == []
    flat_payload = json.loads((pr_dir / "review-decision.json").read_text())
    assert flat_payload["authorized_override"]["by"] == "senkichi"


def test_record_decision_crash_between_round_and_flat_write_recovers_from_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC4: simulate a crash between the round-file write and the flat-file
    write by monkeypatching the flat-file step to raise. The exception must
    propagate (never be swallowed into a silently-missing verdict), and a
    fresh review_decision() read afterward must recover the round verdict --
    proving round-first ordering, not the return value of the failed call."""
    pr_dir = tmp_path / "pr-1"
    flat_path = pr_dir / "review-decision.json"
    original_write = review_decision_module._write_json_atomic

    def flaky_write(path: Path, value: object) -> None:
        if path == flat_path:
            raise OSError("simulated crash before the flat-file write completes")
        original_write(path, value)

    monkeypatch.setattr(review_decision_module, "_write_json_atomic", flaky_write)

    payload = {
        "decision": "request_changes",
        "summary": "fix it",
        "required_changes": ["do X"],
        "verdict_provenance": "fresh_llm_review",
    }
    with pytest.raises(OSError):
        record_decision(pr_dir, payload, "sha-1")

    # The round file landed; the flat file never did.
    round_path = pr_dir / "rounds" / "round-1" / "review-decision.json"
    assert round_path.exists()
    assert not flat_path.exists()

    # A fresh, independent read recovers the round verdict via fallback.
    recovered = review_decision(pr_dir, pr_state=None, current_head_sha="sha-1")
    assert recovered.missing is False
    assert recovered.decision == "request_changes"
    assert recovered.source_round == 1
    assert recovered.stale is False


def test_write_json_atomic_never_leaves_a_partial_real_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Atomicity of the shared primitive itself: a failure while serializing
    must never produce a partially-written file at the real path -- only
    (at most) a ``.tmp`` sibling, which the next write overwrites and no
    reader ever looks at directly."""
    path = tmp_path / "pr-1" / "review-decision.json"

    def boom(*_args: object, **_kwargs: object) -> None:
        raise ValueError("boom mid-serialize")

    monkeypatch.setattr(review_decision_module.json, "dump", boom)

    with pytest.raises(ValueError):
        review_decision_module._write_json_atomic(path, {"decision": "approved"})

    assert not path.exists()

    # Recovery: a subsequent successful write must still work (no leftover
    # .tmp file wedges future writes) and the real path never shows torn
    # content in between.
    monkeypatch.undo()
    review_decision_module._write_json_atomic(path, {"decision": "approved"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"decision": "approved"}
