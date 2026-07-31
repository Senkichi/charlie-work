"""Tests for scripts/ac1b_findings_actionability.py.

Loads the script as a module without adding scripts/ to sys.path, mirroring
tests/test_backfill_stale_rework_briefs.py's pattern for the other
standalone script.

Covers two real bugs found running the harness against the live corpus
(docs/plans/rework-findings-channel.md section 8):

1. ``derive_cross_family_collapse_sentinel`` assumed
   ``cross_family.parse_cross_family_verdict`` returns a ``(decision,
   summary)`` tuple. It actually returns a ``CrossFamilyVerdict`` dataclass
   (or ``None``) -- see cross_family.py:396-474 and
   test_charlie_work.py's ``test_parse_cross_family_verdict_*`` tests, which
   already assert attribute access (``result.decision`` / ``result.summary``).
   The tuple-shaped `isinstance` check always failed, raising RuntimeError
   and collapsing every verdict to ``UNKNOWN_provenance_unavailable``.
2. ``find_concrete_referents`` treated any bare ``identifier(`` token as a
   ``code_symbol`` referent, so ordinary prose/verification-command text
   (e.g. `python -c "import charlie_work; print(charlie_work.__file__)"`)
   scored as "actionable" purely because it called a Python builtin.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from charlie_work import cross_family


def _load_ac1b_script() -> ModuleType:
    path = Path(__file__).parent.parent / "scripts" / "ac1b_findings_actionability.py"
    spec = importlib.util.spec_from_file_location("ac1b_findings_actionability", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["ac1b_findings_actionability"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ac1b() -> ModuleType:
    return _load_ac1b_script()


# --------------------------------------------------------------------------
# Bug 1: sentinel derivation must handle the REAL CrossFamilyVerdict return
# shape, not a (decision, summary) tuple.
# --------------------------------------------------------------------------


def test_derive_cross_family_collapse_sentinel_matches_real_parser(ac1b: ModuleType) -> None:
    """The derived sentinel must be the exact legacy-fallback summary the
    REAL parser produces for a BLOCKER-only report with no Verdict: marker
    -- proving derivation actually exercises cross_family's live contract
    rather than raising and falling back to UNKNOWN.
    """
    sentinel = ac1b.derive_cross_family_collapse_sentinel()
    assert sentinel == "Cross-family review found BLOCKER/MAJOR findings"

    # Cross-check directly against the parser this is meant to track, so a
    # future change to the legacy fallback string is caught by BOTH tests
    # moving together, not just trusted to match by construction.
    probe = "## Report\n\n**BLOCKER** unparseable body with no Verdict: marker\n"
    parsed = cross_family.parse_cross_family_verdict(probe)
    assert isinstance(parsed, cross_family.CrossFamilyVerdict)
    assert parsed.summary == sentinel


def test_classify_verdict_uses_derived_sentinel_for_cross_family_collapse(
    ac1b: ModuleType,
) -> None:
    """A verdict whose summary is exactly the derived sentinel classifies as
    cross_family_generic_collapse -- the category this whole script exists
    to isolate (docs/plans/rework-findings-channel.md section 8).
    """
    sentinel = ac1b.derive_cross_family_collapse_sentinel()
    assert ac1b.classify_verdict(sentinel, sentinel) == ac1b.CROSS_FAMILY_COLLAPSE
    assert ac1b.classify_verdict("CI failed on Lint; push a fix", sentinel) == (
        ac1b.SYNTHETIC_CI_FAILURE
    )
    assert ac1b.classify_verdict("Some real reviewer prose.", sentinel) == (
        ac1b.REAL_REVIEWER_PROSE
    )


# --------------------------------------------------------------------------
# Bug 2: bare builtin calls (print(, int(, str(, ...) in prose must not
# count as concrete code_symbol referents.
# --------------------------------------------------------------------------


def test_verification_command_prose_is_not_actionable(ac1b: ModuleType) -> None:
    """The exact false-positive shape found in the live corpus (pr-182,
    pr-187, pr-188, pr-190): a verification-command snippet whose only
    identifier-shaped token is a bare `print(` call. This must not count
    as a reviewer naming a symbol to change.
    """
    text = 'python -c "import charlie_work; print(charlie_work.__file__)"'
    referents = ac1b.find_concrete_referents(text)
    assert referents == []
    assert ac1b.is_actionable(text) is False


def test_bare_int_call_is_not_actionable(ac1b: ModuleType) -> None:
    """The pr-500 false-positive shape: an error-message snippet whose only
    identifier-shaped token is a bare `int(` call.
    """
    text = "ValueError invalid literal for int() with base 10: 'two'"
    referents = ac1b.find_concrete_referents(text)
    assert referents == []
    assert ac1b.is_actionable(text) is False


def test_bare_non_builtin_call_still_counts(ac1b: ModuleType) -> None:
    """A bare call to a real project symbol (not a Python builtin) must
    still be flagged -- the fix narrows to builtins specifically, it does
    not disable the bare-call referent shape altogether.
    """
    text = "the fix should call _is_review_dispatchable(pr) before dispatching"
    referents = ac1b.find_concrete_referents(text)
    assert ("code_symbol", "_is_review_dispatchable(") in referents
    assert ac1b.is_actionable(text) is True


def test_backtick_quoted_symbols_unaffected_by_builtin_filter(ac1b: ModuleType) -> None:
    """Backtick-quoted symbols (the legitimate signal the task calls out to
    keep) are a separate regex alternative with no `(` in its character
    class, so they are untouched by the builtin-call filter either way.
    """
    text = "See `_diff_content_signature` at src/charlie_work/workflow.py:3700"
    referents = ac1b.find_concrete_referents(text)
    assert ("code_symbol", "`_diff_content_signature`") in referents
    assert ("file_path", "src/charlie_work/workflow.py") in referents
    assert ("line_number", ":3700") in referents
    assert ac1b.is_actionable(text) is True


def test_mixed_builtin_and_real_referent_keeps_only_the_real_one(ac1b: ModuleType) -> None:
    """A verdict mentioning both a bare builtin call and a real file path
    (the pr-182/pr-188 shape) stays actionable, but the builtin call is
    dropped from the referent list -- the fix removes noise without
    flipping an already-correct classification.
    """
    text = 'python -c "import charlie_work; print(charlie_work.__file__)"\ntests/test_worker.py'
    referents = ac1b.find_concrete_referents(text)
    assert ("code_symbol", "print(") not in referents
    assert ("file_path", "tests/test_worker.py") in referents
    assert ac1b.is_actionable(text) is True
