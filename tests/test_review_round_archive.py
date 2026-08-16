"""Tests for issue #1268 (W11), round-numbered archive only.

This file covers the first of three W11 sub-items: before `record_review`
overwrites `review-decision.json` / `rework-prompt.md` /
`rework-dispatch-note.txt` in place, the new content is additionally copied
to `prs/pr-N/rounds/round-K/<same-filename>`, atomically. The events.db
payload and the GitHub PR comment gate (the other two W11 sub-items) are
deliberately out of scope here -- see w11-impl-notes.md / issue #1268's
binding comment.

  AC1 -- two simulated rounds (different head SHAs) leave both
         `rounds/round-1/` and `rounds/round-2/` on disk, each with all
         three files, and round-1's archived content differs from round-2's
         and from the final live content -- proving round-1 actually
         survived round-2's overwrite of the live files.
  AC2 -- three cases, all in this module:
         (a) RETRY -- identical decision/summary/required_changes on the
             SAME head reuses round-K (simulates crash-then-rerun).
         (b) DISTINCT-SAME-HEAD -- different content on the SAME head mints
             round-(K+1) and leaves round-K's bytes on disk untouched. This
             is the case a naive head-advancement-only discriminator gets
             wrong.
         (c) ADVANCED-HEAD -- new content on an advanced head mints
             round-(K+1).
  AC8 -- two mutations, each shown to break exactly one AC2 case:
         (i) always minting `highest + 1` (drops the retry short-circuit)
             breaks AC2(a) -- a byte-identical same-head retry wrongly
             mints a new round.
         (ii) narrowing `_ROUND_COMPARE_KEYS` to only `reviewed_head_sha`
              (reintroducing the rejected head-advanced-only rule) breaks
              AC2(b) -- a distinct verdict on an unchanged head wrongly
              overwrites round-K's archived content. This is the mutation
              that reproduces the actual data-loss bug W11 exists to fix,
              so it is the more important of the two.

Round-directory naming: unpadded (`round-1`, `round-2`, ..., `round-10`).
Any reader -- including W13/#1270, which will extend
`_build_prior_review_section` to read this archive -- must parse the
trailing digits as `int` before ordering; a lexicographic string sort over
the directory names is wrong (`"round-10"` < `"round-2"` as strings). The
tests below assert via `_existing_round_numbers`, the same int-parsing
helper the writer uses, rather than re-deriving their own sort.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

import charlie_work.workflow as workflow
from charlie_work.workflow import OrchestratorApp, _existing_round_numbers

from _review_fixtures import (  # noqa: F401  (_PR_NUMBER: deliberate re-export)
    _PR_NUMBER,
    _pr_dir,
    _record,
    _round_archive_app,
    _round_dir,
)


_ARCHIVE_FILES = ("review-decision.json", "rework-prompt.md", "rework-dispatch-note.txt")


# ---------------------------------------------------------------------------
# AC1
# ---------------------------------------------------------------------------


def test_ac1_two_rounds_leave_distinct_archived_content_for_each_round(tmp_path: Path) -> None:
    app, paths = _round_archive_app(tmp_path)

    _record(app, head="sha-r1", summary="round one summary", required_changes=["round one change"])
    _record(app, head="sha-r2", summary="round two summary", required_changes=["round two change"])

    round1_dir = _round_dir(paths, 1)
    round2_dir = _round_dir(paths, 2)
    pr_dir = _pr_dir(paths)

    for name in _ARCHIVE_FILES:
        assert (round1_dir / name).is_file(), f"round-1/{name} missing"
        assert (round2_dir / name).is_file(), f"round-2/{name} missing"

    round1_decision = json.loads((round1_dir / "review-decision.json").read_text(encoding="utf-8"))
    round2_decision = json.loads((round2_dir / "review-decision.json").read_text(encoding="utf-8"))
    live_decision = json.loads((pr_dir / "review-decision.json").read_text(encoding="utf-8"))

    assert round1_decision["summary"] == "round one summary"
    assert round1_decision["required_changes"] == ["round one change"]
    assert round2_decision["summary"] == "round two summary"
    assert round2_decision["required_changes"] == ["round two change"]
    assert live_decision["summary"] == "round two summary"

    # The load-bearing assertion: round-1's archive differs from round-2's
    # AND from the final live content -- proving round-1's text actually
    # survived round-2's in-place overwrite of review-decision.json, not
    # just that two directories happen to exist.
    assert round1_decision != round2_decision
    assert round1_decision != live_decision
    # round-2 is the round that ended up live, so its archive matches.
    assert round2_decision == live_decision

    round1_prompt = (round1_dir / "rework-prompt.md").read_text(encoding="utf-8")
    round2_prompt = (round2_dir / "rework-prompt.md").read_text(encoding="utf-8")
    live_prompt = (pr_dir / "rework-prompt.md").read_text(encoding="utf-8")
    assert round1_prompt != round2_prompt
    assert round1_prompt != live_prompt
    assert round2_prompt == live_prompt

    round1_note = (round1_dir / "rework-dispatch-note.txt").read_text(encoding="utf-8")
    round2_note = (round2_dir / "rework-dispatch-note.txt").read_text(encoding="utf-8")
    live_note = (pr_dir / "rework-dispatch-note.txt").read_text(encoding="utf-8")
    assert round1_note != round2_note
    assert round1_note != live_note
    assert round2_note == live_note


# ---------------------------------------------------------------------------
# AC2 -- three cases, reusable as assertion helpers so AC8's mutation tests
# can prove each one specifically fails under the mutation it targets,
# rather than merely asserting "some test goes red".
# ---------------------------------------------------------------------------


def _assert_retry_reuses_round(app: OrchestratorApp, paths: Any) -> None:
    """AC2(a): a byte-identical same-head retry (simulated crash-then-rerun)
    overwrites the SAME round-K, never mints round-(K+1)."""
    rounds_dir = _pr_dir(paths) / "rounds"
    _record(app, head="sha-retry", summary="retry summary", required_changes=["retry change"])
    assert _existing_round_numbers(rounds_dir) == [1]

    # Identical decision/summary/required_changes, identical head.
    _record(app, head="sha-retry", summary="retry summary", required_changes=["retry change"])
    assert _existing_round_numbers(rounds_dir) == [1], (
        "a byte-identical same-head retry must reuse round-1, not mint round-2"
    )


def _assert_distinct_same_head_mints_round_and_preserves_prior(
    app: OrchestratorApp, paths: Any
) -> None:
    """AC2(b): a distinct verdict on an unchanged head mints round-(K+1) and
    leaves round-K's prior content on disk unchanged -- the case a naive
    head-advancement-only discriminator gets wrong."""
    rounds_dir = _pr_dir(paths) / "rounds"
    _record(app, head="sha-distinct", summary="first summary", required_changes=["first change"])
    round1_dir = _round_dir(paths, 1)
    before = {name: (round1_dir / name).read_bytes() for name in _ARCHIVE_FILES}

    # Same head, genuinely different decision content.
    _record(
        app,
        head="sha-distinct",
        summary="second summary -- different from the first",
        required_changes=["second change -- different from the first"],
    )
    # Checked first, and separately from the round-count assertion below: a
    # discriminator that misreads this call as a retry (mutation (ii))
    # reuses round-1 and overwrites it in place -- the direct symptom of the
    # data-loss bug this archive exists to fix. Asserting byte-preservation
    # before the mint-count means that failure surfaces as "content was
    # overwritten", not merely "round-2 is missing".
    for name in _ARCHIVE_FILES:
        assert (round1_dir / name).read_bytes() == before[name], (
            f"round-1's archived {name} must survive a distinct same-head verdict untouched"
        )
    assert _existing_round_numbers(rounds_dir) == [1, 2], (
        "a distinct verdict on an unchanged head must mint round-2"
    )


def _assert_advanced_head_mints_round(app: OrchestratorApp, paths: Any) -> None:
    """AC2(c): new content on an advanced head mints round-(K+1)."""
    rounds_dir = _pr_dir(paths) / "rounds"
    _record(app, head="sha-adv-1", summary="first summary", required_changes=["first change"])
    _record(app, head="sha-adv-2", summary="second summary", required_changes=["second change"])
    assert _existing_round_numbers(rounds_dir) == [1, 2]


def test_ac2_case_a_retry_reuses_same_round(tmp_path: Path) -> None:
    app, paths = _round_archive_app(tmp_path)
    _assert_retry_reuses_round(app, paths)


def test_ac2_case_b_distinct_same_head_mints_new_round_and_preserves_prior(
    tmp_path: Path,
) -> None:
    app, paths = _round_archive_app(tmp_path)
    _assert_distinct_same_head_mints_round_and_preserves_prior(app, paths)


def test_ac2_case_c_advanced_head_mints_new_round(tmp_path: Path) -> None:
    app, paths = _round_archive_app(tmp_path)
    _assert_advanced_head_mints_round(app, paths)


# ---------------------------------------------------------------------------
# AC8 -- two mutations, mutated toward the forbidden state (per the task's
# own double-write/crash framing), each shown to break the ONE AC2 case it
# targets by reusing that case's assertion helper under the mutation.
# ---------------------------------------------------------------------------


def test_ac8_mutation_i_always_next_breaks_retry_idempotency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation (i): drop the retry short-circuit -- always mint
    ``highest + 1`` regardless of content. AC2(a) must fail specifically
    because a spurious round-2 was created for a byte-identical same-head
    retry."""

    def _always_mint_next(rounds_dir: Path, decision_payload: Any) -> int:
        highest = max(_existing_round_numbers(rounds_dir), default=0)
        return highest + 1

    monkeypatch.setattr(workflow, "_next_round_number", _always_mint_next)

    app, paths = _round_archive_app(tmp_path)
    with pytest.raises(AssertionError, match="must reuse round-1"):
        _assert_retry_reuses_round(app, paths)


def test_ac8_mutation_ii_head_only_compare_breaks_distinct_verdict_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation (ii): narrow the compare keys to `reviewed_head_sha` only --
    reintroducing the rejected head-advanced-only rule. AC2(b) must fail
    specifically because round-1's archived content was overwritten/lost
    when a distinct verdict arrived on an unchanged head -- this is the
    mutation that reproduces the actual data-loss bug W11 exists to fix."""
    monkeypatch.setattr(workflow, "_ROUND_COMPARE_KEYS", ("reviewed_head_sha",))

    app, paths = _round_archive_app(tmp_path)
    with pytest.raises(AssertionError, match="must survive a distinct"):
        _assert_distinct_same_head_mints_round_and_preserves_prior(app, paths)


# ---------------------------------------------------------------------------
# AC6 -- module-wide static coverage. test_review_event_payload.py's
# ``test_write_rework_prompt_live_writes_route_through_atomic_helper`` only
# inspects ``_write_rework_prompt``'s own body, so it cannot see the two
# inline archive-copy writes added directly inside ``record_review`` (the
# round_dir / "rework-prompt.md" and its sidecar, plus the
# round_dir / "review-decision.json" copy). This check derives coverage of
# all three archived filenames across the WHOLE module, per
# w11-impl-notes.md item 5: enumerate the target filename literals and
# assert every write reaching a path built from one of them routes through
# _write_json / _write_text_atomic -- never grep for a forbidden substring
# (that either false-positives on unrelated write_text calls, e.g. the
# worker-prompt.md / review-comment.md / diff.patch / interdiff.patch /
# citation-drift-comment.md sites elsewhere in this module, or gets narrowed
# until it fails open -- the same trap as the
# ast-guard-fails-open-on-value-form memory).
# ---------------------------------------------------------------------------

_AC6_TARGET_FILENAMES = ("review-decision.json", "rework-prompt.md", "rework-dispatch-note.txt")
_AC6_APPROVED_WRITERS = {"_write_json", "_write_text_atomic"}
_AC6_RAW_WRITE_ATTRS = {"write_text", "write_bytes"}


def _ac6_call_func_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def test_ac6_module_wide_write_sites_for_archived_filenames_are_atomic() -> None:
    source = inspect.getsource(workflow)
    tree = ast.parse(source)

    function_nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    violations: list[str] = []
    approved_targets: set[str] = set()

    for func in function_nodes:
        if func.name in _AC6_APPROVED_WRITERS:
            # The two approved helpers' own bodies legitimately contain the
            # raw write_text/open/write primitives everything else must not
            # use directly -- that is what makes them the approved choke
            # point, not a violation of it.
            continue

        # Local, function-scoped resolution only: name -> every RHS text
        # ever assigned to it within this function. Deliberately not a
        # module-wide dataflow pass -- cheap, and the three filenames are
        # distinctive literals unlikely to collide across unrelated locals
        # sharing a generic name (e.g. a `prompt_path` in one function
        # resolving against an assignment made in a different function).
        assigned_rhs: dict[str, list[str]] = {}
        for node in ast.walk(func):
            if isinstance(node, ast.Assign):
                rhs_text = ast.unparse(node.value)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigned_rhs.setdefault(target.id, []).append(rhs_text)

        def _resolve(expr: ast.expr, _assigned=assigned_rhs) -> list[str]:
            if isinstance(expr, ast.Name):
                return _assigned.get(expr.id, [ast.unparse(expr)])
            return [ast.unparse(expr)]

        for node in ast.walk(func):
            if not isinstance(node, ast.Call):
                continue
            called_name = _ac6_call_func_name(node)

            if called_name in _AC6_APPROVED_WRITERS and node.args:
                approved_targets.update(_resolve(node.args[0]))
                continue

            if not (
                isinstance(node.func, ast.Attribute) and node.func.attr in _AC6_RAW_WRITE_ATTRS
            ):
                continue

            candidate_texts = _resolve(node.func.value)
            for filename in _AC6_TARGET_FILENAMES:
                if any(filename in text for text in candidate_texts):
                    violations.append(
                        f"{func.name}:{node.lineno} raw .{node.func.attr}(...) targets "
                        f"{filename!r} outside _write_json/_write_text_atomic"
                    )

    assert violations == [], violations

    # The check above proves "no raw write reaches these filenames"; this
    # proves it is not vacuous -- each of the three filenames really is
    # still written through an approved helper somewhere in the module. A
    # future rename/refactor that silently dropped one of these three
    # archive/live writes entirely would otherwise make the assertion above
    # trivially pass with zero violations for the wrong reason.
    for filename in _AC6_TARGET_FILENAMES:
        assert any(filename in text for text in approved_targets), (
            f"expected at least one _write_json/_write_text_atomic call "
            f"targeting {filename!r}; none found -- the violations check "
            f"above may be vacuously passing"
        )
