"""Tests for _is_carry_forward_eligible, carved out of test_charlie_work.py (#1284).

Also covers _summary_is_vacuous as the inner dependency: workflow.py's
_is_carry_forward_eligible calls _summary_is_vacuous directly, and both
clusters' banners cross-reference each other by issue number.
"""

from __future__ import annotations

from charlie_work.cross_family import LEGACY_VACUOUS_SUMMARY
from charlie_work.workflow import _is_carry_forward_eligible, _summary_is_vacuous


# --------------------------------------------------------------------------
# Issue #784 AC-8: _is_carry_forward_eligible -- the single-point-of-
# enforcement predicate that rejects carry-forward for a content-free
# recorded verdict. Operates on plain dicts (never reconstructs a
# CrossFamilyVerdict), which is the load-bearing fact resolving the
# deserialization hazard: it must never raise on any of the 8 pre-#784
# on-disk broken records, however malformed their shape.
# --------------------------------------------------------------------------


def test_is_carry_forward_eligible_true_for_approved() -> None:
    assert _is_carry_forward_eligible({"decision": "approved", "summary": ""}) is True


def test_is_carry_forward_eligible_true_for_blocked() -> None:
    assert _is_carry_forward_eligible({"decision": "blocked", "summary": "why"}) is True


def test_is_carry_forward_eligible_true_for_request_changes_with_required_changes() -> None:
    assert (
        _is_carry_forward_eligible(
            {"decision": "request_changes", "required_changes": ["fix x"], "summary": ""}
        )
        is True
    )


def test_is_carry_forward_eligible_true_for_request_changes_with_real_summary() -> None:
    assert (
        _is_carry_forward_eligible(
            {"decision": "request_changes", "required_changes": [], "summary": "Real prose."}
        )
        is True
    )


def test_is_carry_forward_eligible_false_for_empty_summary_and_required_changes() -> None:
    assert (
        _is_carry_forward_eligible(
            {"decision": "request_changes", "required_changes": [], "summary": ""}
        )
        is False
    )


def test_is_carry_forward_eligible_false_for_missing_keys() -> None:
    """A pre-#784 broken on-disk record may not even carry a
    ``required_changes`` or ``summary`` key at all -- must classify as
    ineligible via ``.get()`` defaults, never raise (KeyError or otherwise),
    on a plain dict this predicate was built specifically to tolerate."""
    assert _is_carry_forward_eligible({"decision": "request_changes"}) is False


def test_is_carry_forward_eligible_false_for_legacy_placeholder_summary() -> None:
    """The historical hardcoded placeholder is content-free by definition
    even though ``summary`` is technically non-empty -- one of the 8
    pre-#784 on-disk records has exactly this shape."""
    assert (
        _is_carry_forward_eligible(
            {
                "decision": "request_changes",
                "required_changes": [],
                "summary": LEGACY_VACUOUS_SUMMARY,
            }
        )
        is False
    )


def test_is_carry_forward_eligible_false_for_whitespace_padded_placeholder() -> None:
    """``summary`` is ``.strip()``-normalized before the comparison, so
    whitespace padding around the exact placeholder cannot smuggle it past
    the guard as if it were a distinct, real summary."""
    assert (
        _is_carry_forward_eligible(
            {
                "decision": "request_changes",
                "required_changes": [],
                "summary": f"  {LEGACY_VACUOUS_SUMMARY}  ",
            }
        )
        is False
    )


# --------------------------------------------------------------------------
# Issue #792: _summary_is_vacuous is the single shared discriminator between
# "nothing to derive" (record_review's write-time marker) and "not eligible
# for carry-forward" (_is_carry_forward_eligible's read-time check above).
# It must classify real, specific reviewer prose as non-vacuous even when
# that prose is terse or lacks a file/line reference -- only a blank string
# or the one known historical placeholder is vacuous.
# --------------------------------------------------------------------------


def test_summary_is_vacuous_true_for_blank_string() -> None:
    assert _summary_is_vacuous("") is True


def test_summary_is_vacuous_true_for_whitespace_only() -> None:
    assert _summary_is_vacuous("   \n\t  ") is True


def test_summary_is_vacuous_true_for_legacy_placeholder() -> None:
    assert _summary_is_vacuous(LEGACY_VACUOUS_SUMMARY) is True


def test_summary_is_vacuous_true_for_whitespace_padded_placeholder() -> None:
    assert _summary_is_vacuous(f"  {LEGACY_VACUOUS_SUMMARY}  ") is True


def test_summary_is_vacuous_false_for_substantive_architectural_prose() -> None:
    """A real, specific finding with no file/line reference (pr-774's shape)
    must not be misclassified as vacuous just because it lacks structure."""
    prose = (
        "The retry wrapper swallows the underlying exception type, so a "
        "caller cannot distinguish a transient network failure from a "
        "permanent 4xx and will retry requests that can never succeed."
    )
    assert _summary_is_vacuous(prose) is False


def test_summary_is_vacuous_false_for_terse_but_real_ci_summary() -> None:
    """The CI-failure producer's own summary shape (pr-529/683's pattern)
    is short but names a specific, real cause -- not content-free."""
    assert _summary_is_vacuous("CI failed on Lint; push a fix") is False


def test_summary_is_vacuous_false_for_prefix_or_suffix_of_placeholder() -> None:
    """Only an exact match on the known placeholder is vacuous -- a summary
    that merely contains it as a substring (e.g. a reviewer quoting the old
    bug) is real, distinguishing text and must not be swept in by a loose
    substring check."""
    assert _summary_is_vacuous(f"{LEGACY_VACUOUS_SUMMARY} but I also checked X") is False
