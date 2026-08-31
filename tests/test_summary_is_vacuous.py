"""Tests for _summary_is_vacuous, carved out of test_charlie_work.py (#1284).

This file used to also cover _is_carry_forward_eligible (issue #784 AC-8),
_summary_is_vacuous's own caller, since workflow.py's now-deleted
_is_carry_forward_eligible called _summary_is_vacuous directly and both
clusters' banners cross-referenced each other by issue number. The
auto-gate-only carry-forward eligibility check (and its sole caller,
review_queue()'s vacuous-decision branch) was deleted in the role-config
Phase 2 cleanup -- _summary_is_vacuous itself survives as record_review's
write-time vacuous-marker check, so only its own tests remain here.
"""

from __future__ import annotations

from charlie_work.rescue_review import LEGACY_VACUOUS_SUMMARY
from charlie_work.workflow import _summary_is_vacuous


# --------------------------------------------------------------------------
# Issue #792: _summary_is_vacuous is the single shared discriminator between
# "nothing to derive" (record_review's write-time marker) and "not eligible
# for carry-forward" (the now-deleted _is_carry_forward_eligible's read-time
# check). It must classify real, specific reviewer prose as non-vacuous even
# when that prose is terse or lacks a file/line reference -- only a blank
# string or the one known historical placeholder is vacuous.
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
