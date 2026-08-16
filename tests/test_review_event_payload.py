"""Tests for issue #1268 (W11), events.db payload + atomic live writes.

Second of three W11 sub-items -- see ``tests/test_review_round_archive.py``
(the first sub-item, the round-numbered archive) for the overall issue
context and its own docstring. This file covers:

  AC3 -- the ``record_review`` events.db payload carries non-null
         ``summary`` and ``required_changes`` keys (previously never
         assigned at all: confirmed live, 0 non-null across every
         pre-W11 row -- the "never produced" bug variant, not "produced
         then filtered"). A summary/required_changes string longer than
         16KB is truncated to <=16KB with a literal ``"...truncated"``
         marker in the STORED events.db payload; the archived round-K
         file copy (written by the first W11 sub-item) keeps the full,
         untruncated text -- only the events.db copy is size-guarded.
  AC6 (atomicity half only -- round-archive-write atomicity is exercised
       by test_review_round_archive.py's own fixtures, not re-tested
       here) -- the two live writes inside the module-level
       ``_write_rework_prompt`` (``rework-prompt.md`` and its
       ``rework-dispatch-note.txt`` sidecar) now route through
       ``_write_text_atomic`` (tmp+replace) instead of a bare
       ``write_text``, and a write interrupted before the atomic rename
       never leaves a torn file at the live path.

The GitHub PR comment gate (the third W11 sub-item) is out of scope here;
AC4/AC5 are not exercised by this file and are expected to still be red at
this checkpoint.
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
from pathlib import Path

import pytest

import charlie_work.workflow as workflow
from charlie_work.instrumentation import query_events
from charlie_work.workflow import (
    _EVENT_TEXT_MAX_BYTES,
    _EVENT_TRUNCATE_MARKER,
    _truncate_for_event,
    _write_text_atomic,
)

from test_review_round_archive import _PR_NUMBER, _pr_dir, _record, _round_archive_app, _round_dir

# ---------------------------------------------------------------------------
# AC3 -- events.db payload carries summary/required_changes, size-bounded
# ---------------------------------------------------------------------------


def test_ac3_two_rounds_each_produce_a_record_review_row_with_non_null_verdict_text(
    tmp_path: Path,
) -> None:
    app, paths = _round_archive_app(tmp_path)

    _record(app, head="sha-r1", summary="round one summary", required_changes=["round one change"])
    _record(app, head="sha-r2", summary="round two summary", required_changes=["round two change"])

    rows = query_events(paths.state_file, kind="record_review")
    assert len(rows) == 2, "expected exactly one record_review row per round"

    round1_payload = rows[0]["payload"]
    round2_payload = rows[1]["payload"]

    assert round1_payload["summary"] == "round one summary"
    assert round1_payload["required_changes"] == "round one change"
    assert round2_payload["summary"] == "round two summary"
    assert round2_payload["required_changes"] == "round two change"

    # Additive-only regression guard: the pre-existing keys are still
    # present and hold their pre-W11 values -- this change must extend the
    # payload dict, never replace or rename anything in it.
    for payload in (round1_payload, round2_payload):
        assert payload["pr_number"] == _PR_NUMBER
        assert payload["decision"] == "request_changes"
        assert payload["escalated"] is False
        assert payload["verdict_provenance"] == "fresh_llm_review"


def test_ac3_oversized_verdict_text_truncated_in_events_db_but_archive_stays_full(
    tmp_path: Path,
) -> None:
    app, paths = _round_archive_app(tmp_path)

    long_summary = "S" * 20_000  # > 16KB
    long_change = "C" * 20_000  # > 16KB, becomes the sole required_changes entry
    _record(app, head="sha-big", summary=long_summary, required_changes=[long_change])

    rows = query_events(paths.state_file, kind="record_review")
    assert len(rows) == 1
    payload = rows[0]["payload"]

    stored_summary = payload["summary"]
    stored_required_changes = payload["required_changes"]

    # Byte-budget, not character-count: assert <=, never ==, per the
    # helper's own contract (errors="ignore" can drop a partial multi-byte
    # char at the cut boundary, so the encoded length need not land exactly
    # on the budget).
    assert len(stored_summary.encode("utf-8")) <= _EVENT_TEXT_MAX_BYTES
    assert len(stored_required_changes.encode("utf-8")) <= _EVENT_TEXT_MAX_BYTES
    assert stored_summary.endswith(_EVENT_TRUNCATE_MARKER)
    assert stored_required_changes.endswith(_EVENT_TRUNCATE_MARKER)
    # And it really did cut something -- not just append the marker to the
    # untouched original.
    assert len(stored_summary) < len(long_summary)
    assert len(stored_required_changes) < len(long_change)

    # The archived round-1 copy (first W11 sub-item) must be untruncated:
    # events.db is observability-only, the archive is the reconstruction
    # source of truth W13 reads from.
    archived_decision = json.loads(
        (_round_dir(paths, 1) / "review-decision.json").read_text(encoding="utf-8")
    )
    assert archived_decision["summary"] == long_summary
    assert archived_decision["required_changes"] == [long_change]

    # Same for the live (not-yet-superseded) file.
    live_decision = json.loads(
        (_pr_dir(paths) / "review-decision.json").read_text(encoding="utf-8")
    )
    assert live_decision["summary"] == long_summary
    assert live_decision["required_changes"] == [long_change]


def test_truncate_for_event_short_text_passes_through_unchanged() -> None:
    text = "well under budget"
    assert _truncate_for_event(text) == text


def test_truncate_for_event_multibyte_cut_boundary_never_raises_and_stays_in_budget() -> None:
    # Every character is a 3-byte UTF-8 codepoint, so a budget not evenly
    # divisible by 3 forces the cut to land mid-character somewhere in the
    # run -- exactly the case errors="ignore" exists to survive without
    # raising. max_bytes is kept comfortably larger than the marker's own
    # byte length (12) so this exercises the ordinary case, not the
    # marker-larger-than-budget corner the helper doesn't promise to bound.
    text = "☃" * 50  # SNOWMAN, 3 bytes each in UTF-8
    result = _truncate_for_event(text, max_bytes=20)
    assert len(result.encode("utf-8")) <= 20
    assert result.endswith(_EVENT_TRUNCATE_MARKER)


# ---------------------------------------------------------------------------
# AC6 (atomicity half) -- the two live writes inside the module-level
# _write_rework_prompt now route through _write_text_atomic.
# ---------------------------------------------------------------------------


def test_write_text_atomic_happy_path_leaves_no_tmp_sibling(tmp_path: Path) -> None:
    target = tmp_path / "artifact.txt"
    _write_text_atomic(target, "hello world")

    assert target.read_text(encoding="utf-8") == "hello world"
    assert not target.with_suffix(target.suffix + ".tmp").exists()


def test_write_text_atomic_crash_before_rename_leaves_final_path_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core atomicity guarantee: if the process dies after the tmp file
    is written but before the rename, the live path must never observe a
    torn/partial write -- it must show either the old content (this test)
    or nothing at all, never a half-written new value."""
    target = tmp_path / "artifact.txt"
    target.write_text("original content", encoding="utf-8")

    def _boom(self: Path, other: object) -> None:
        raise OSError("simulated crash before rename")

    monkeypatch.setattr(Path, "replace", _boom)

    with pytest.raises(OSError, match="simulated crash"):
        _write_text_atomic(target, "new content that must never land")

    assert target.read_text(encoding="utf-8") == "original content"


def test_write_rework_prompt_live_writes_route_through_atomic_helper() -> None:
    """AST-derived, not string/regex matching: enumerate what the function's
    two live-write call sites actually call, rather than grepping for a
    forbidden substring that a comment or docstring could accidentally
    trip (or a rephrasing could silently evade)."""
    source = textwrap.dedent(inspect.getsource(workflow._write_rework_prompt))
    tree = ast.parse(source)
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)

    write_text_calls = 0
    atomic_calls = 0
    # Call-count alone would still pass if a future edit pointed both calls
    # at the same filename (or the wrong one) -- collecting the string
    # literals that actually appear among the atomic calls' own arguments
    # ties the assertion to *which* artifact, not just how many calls exist.
    atomic_call_literals: set[str] = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "write_text":
            write_text_calls += 1
        elif isinstance(node.func, ast.Name) and node.func.id == "_write_text_atomic":
            atomic_calls += 1
            for arg_node in ast.walk(node):
                if isinstance(arg_node, ast.Constant) and isinstance(arg_node.value, str):
                    atomic_call_literals.add(arg_node.value)

    assert write_text_calls == 0, (
        "_write_rework_prompt must not call .write_text(...) directly for "
        "either of its two live artifacts -- both must route through "
        "_write_text_atomic"
    )
    assert atomic_calls == 2, (
        "expected exactly two _write_text_atomic calls: rework-prompt.md "
        "and the rework-dispatch-note.txt sidecar"
    )
    assert "rework-dispatch-note.txt" in atomic_call_literals, (
        "the sidecar's atomic call must still name rework-dispatch-note.txt "
        "-- a future edit that pointed both _write_text_atomic calls at the "
        "same path would otherwise still pass the count-only assertion above"
    )


def test_record_review_leaves_no_tmp_files_after_live_writes(tmp_path: Path) -> None:
    """Hygiene check, not the regression guard: a plain ``write_text`` would
    also leave zero ``.tmp`` siblings (it never creates one), so this test
    alone would still pass on a revert of the atomic-write fix. The real
    guard against that regression is
    ``test_write_rework_prompt_live_writes_route_through_atomic_helper``
    above, which asserts the call sites themselves via AST rather than an
    on-disk side effect that both the fixed and buggy code share."""
    app, paths = _round_archive_app(tmp_path)
    _record(app, head="sha-1", summary="s", required_changes=["c"])

    pr_dir = _pr_dir(paths)
    tmp_files = list(pr_dir.rglob("*.tmp"))
    assert tmp_files == [], f"stray .tmp files left behind: {tmp_files}"
    assert (pr_dir / "rework-prompt.md").is_file()
    assert (pr_dir / "rework-dispatch-note.txt").is_file()
