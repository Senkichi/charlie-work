"""Turn output schema and action parsing (issue #3119).

The schema fails the *whole* Turn output as one error value — and the
forbidden effects (other label edits, verdicts, merging, unescalating,
escape hatches, config/charter edits) must be unexpressible by any
schema-valid action.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from charlie_work.foreman.actions import (
    ACTION_TYPES,
    AmendIssueAction,
    CommentIssueAction,
    DeleteMemoryAction,
    FileIssueAction,
    ProposeDoneAction,
    RaiseItemAction,
    TURN_OUTPUT_SCHEMA,
    WithdrawIssueAction,
    WriteMemoryAction,
    load_turn_output,
    parse_turn_output,
    validate_turn_output,
)


def _doc(*actions: dict) -> dict:
    return {"turn_summary": "a turn", "actions": list(actions)}


def _file_issue(**overrides) -> dict:
    base = {
        "type": "file_issue",
        "key": "a",
        "repo": "acme/widgets",
        "title": "Title",
        "body": "Body.",
    }
    base.update(overrides)
    return base


def test_every_action_type_round_trips() -> None:
    doc = _doc(
        _file_issue(blocked_by=["key:x", "#12"]),
        {"type": "amend_issue", "repo": "acme/widgets", "number": 3, "body": "new"},
        {"type": "withdraw_issue", "repo": "acme/widgets", "number": 4, "reason": "done"},
        {"type": "comment_issue", "repo": "acme/widgets", "number": 5, "body": "note"},
        {
            "type": "raise_item",
            "key": "q1",
            "kind": "decision",
            "question": "Pick",
            "options": ["a", "b"],
            "recommended": 1,
            "command": "gh issue edit 1 --add-label x",
        },
        {"type": "write_memory", "path": "notes.md", "content": "remember"},
        {"type": "delete_memory", "path": "old"},
        {"type": "propose_done", "evidence": "all landed"},
    )
    result = parse_turn_output(doc)
    assert result.ok, result.error
    output = result.value
    assert output.turn_summary == "a turn"
    assert isinstance(output.actions[0], FileIssueAction)
    assert output.actions[0].blocked_by == ("key:x", "#12")
    assert isinstance(output.actions[1], AmendIssueAction)
    assert isinstance(output.actions[2], WithdrawIssueAction)
    assert isinstance(output.actions[3], CommentIssueAction)
    assert isinstance(output.actions[4], RaiseItemAction)
    assert output.actions[4].options == ("a", "b")
    assert output.actions[4].recommended == 1
    assert isinstance(output.actions[5], WriteMemoryAction)
    assert isinstance(output.actions[6], DeleteMemoryAction)
    assert isinstance(output.actions[7], ProposeDoneAction)


def test_empty_actions_list_is_valid() -> None:
    result = parse_turn_output(_doc())
    assert result.ok
    assert result.value.actions == ()


def test_schema_document_is_self_consistent() -> None:
    """Every $defs action branch validates a doc carrying that action type."""
    for action_type in sorted(ACTION_TYPES):
        assert any(
            branch.get("$ref") == f"#/$defs/{action_type}"
            for branch in TURN_OUTPUT_SCHEMA["$defs"]["action"]["oneOf"]
        )


@pytest.mark.parametrize(
    "document,fragment",
    [
        ({"actions": []}, "turn_summary"),
        ({"turn_summary": "s"}, "actions"),
        ({"turn_summary": "s", "actions": [], "extra": 1}, "additional property"),
        (_doc({"type": "teleport_issue"}), "action shape"),
        (_doc({"type": "file_issue"}), "required property missing"),
        (_doc(_file_issue(key="bad key!")), "key"),
        (_doc(_file_issue(blocked_by=["a"])), "blocked_by"),
        (_doc(_file_issue(number=3)), "additional property"),
        (
            _doc({"type": "amend_issue", "repo": "r", "number": 0, "body": "b"}),
            "number",
        ),
        (_doc({"type": "raise_item", "key": "k", "kind": "wish", "question": "q"}), "kind"),
        (
            _doc(
                {
                    "type": "raise_item",
                    "key": "k",
                    "kind": "decision",
                    "question": "q",
                    "recommended": "first",
                }
            ),
            "recommended",
        ),
        (_doc({"type": "delete_memory", "path": 4}), "path"),
        (_doc({"type": "propose_done"}), "required property missing"),
    ],
)
def test_schema_failures_are_one_error_value(document, fragment: str) -> None:
    result = parse_turn_output(document)
    assert not result.ok
    assert "turn output failed schema" in result.error
    if fragment:
        assert fragment in result.error


def test_load_turn_output_reads_actions_json(tmp_path: Path) -> None:
    path = tmp_path / "actions.json"
    path.write_text(json.dumps(_doc(_file_issue())), encoding="utf-8")
    assert load_turn_output(path).ok


def test_load_turn_output_errors_on_missing_and_bad_json(tmp_path: Path) -> None:
    missing = load_turn_output(tmp_path / "nope.json")
    assert not missing.ok and "not found" in missing.error
    bad = tmp_path / "actions.json"
    bad.write_text("{not json", encoding="utf-8")
    result = load_turn_output(bad)
    assert not result.ok and "invalid JSON" in result.error


# ---------------------------------------------------------------------------
# Forbidden effects (issue #3119 step 3): no schema-valid action may express
# other label edits, verdicts, merging, unescalating, escape hatches, or
# config/charter edits — one test per effect.
# ---------------------------------------------------------------------------


def test_no_schema_valid_action_edits_other_labels() -> None:
    """Only Ready may ever be added (by Apply's arming step, never a field)."""
    # A dedicated label action type does not exist.
    for action_type in ("set_labels", "add_label", "remove_label", "label_issue"):
        assert not parse_turn_output(_doc({"type": action_type, "number": 1, "labels": ["x"]})).ok
    # And no existing action carries a labels field.
    assert not parse_turn_output(_doc(_file_issue(labels=["agent:done"]))).ok
    assert not parse_turn_output(
        _doc(
            {
                "type": "amend_issue",
                "repo": "acme/widgets",
                "number": 1,
                "body": "b",
                "labels": ["agent:done"],
            }
        )
    ).ok
    assert not parse_turn_output(
        _doc(
            {
                "type": "withdraw_issue",
                "repo": "acme/widgets",
                "number": 1,
                "reason": "r",
                "remove_labels": ["agent:human-needed"],
            }
        )
    ).ok


def test_no_schema_valid_action_records_a_verdict() -> None:
    for action_type in ("record_verdict", "verdict", "approve", "request_changes"):
        assert not parse_turn_output(
            _doc({"type": action_type, "number": 1, "verdict": "approved"})
        ).ok
    # Nor can a verdict be smuggled onto an existing action.
    assert not parse_turn_output(
        _doc(
            {
                "type": "comment_issue",
                "repo": "acme/widgets",
                "number": 1,
                "body": "b",
                "verdict": "approved",
            }
        )
    ).ok


def test_no_schema_valid_action_merges() -> None:
    for action_type in ("merge_pr", "merge", "squash_merge", "complete_pr"):
        assert not parse_turn_output(_doc({"type": action_type, "number": 1})).ok


def test_no_schema_valid_action_unescalates() -> None:
    """Removing human-needed/operator-queue is a label edit with no action."""
    for action_type in ("unescalate", "deescalate", "resolve_escalation"):
        assert not parse_turn_output(_doc({"type": action_type, "number": 1})).ok


def test_no_schema_valid_action_applies_an_escape_hatch() -> None:
    for action_type in ("escape_hatch", "reset_issue", "force_done"):
        assert not parse_turn_output(_doc({"type": action_type, "number": 1})).ok


def test_no_schema_valid_action_edits_config_or_charter() -> None:
    for action_type in (
        "edit_charter",
        "write_config",
        "update_charter",
        "edit_config",
        "set_bounds",
    ):
        assert not parse_turn_output(_doc({"type": action_type})).ok
    # A charter path cannot be reached through write_memory either: the
    # schema keeps ``path`` shape-only (a bare stem), and the escape refusal
    # lives in Apply's ``memory/`` containment — covered by the apply tests.
    assert not validate_turn_output(
        _doc({"type": "write_memory", "path": "../charter.yaml", "content": "x"})
    )
