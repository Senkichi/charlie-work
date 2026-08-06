"""Tests for ``charlie_work.cross_family.report_is_reusable`` -- the single
predicate shared by both cross-family reuse callers (issue #1081):
``workflow._cross_family_for_pr`` (reuse-vs-rerun) and the same-head packet
skip in ``OrchestratorApp.loop`` (stale-vs-fresh).

Its body is four sequential guards:

    if not text.strip(): return False
    if "(UNAVAILABLE)" in text.splitlines()[0]: return False
    if not report_body_is_valid(extract_report_body(text)): return False
    return extract_head_ref_oid(text) == current_head_sha

Each test below is written to pin exactly one of these clauses.
"""

from __future__ import annotations

from pathlib import Path

from charlie_work.cross_family import _fail, _report, report_is_reusable

# A body with a real strict severity marker and a Verdict line -- passes
# report_body_is_valid via the _SEVERITY_RE branch.
_REAL_BODY = "**BLOCKER**\nsomething needs fixing\n\nVerdict: request changes"


def test_well_formed_report_reusable_when_head_sha_matches() -> None:
    """Pins clause 4's True branch: valid body + a head-SHA comment equal to
    current_head_sha -> reusable."""
    text = _report("glm-5.2", _REAL_BODY, "sha-abc123")
    assert report_is_reusable(text, "sha-abc123") is True


def test_well_formed_report_not_reusable_when_head_sha_differs() -> None:
    """Pins clause 4's False branch on a genuine mismatch: same report, a
    different current_head_sha."""
    text = _report("glm-5.2", _REAL_BODY, "sha-abc123")
    assert report_is_reusable(text, "sha-different") is False


def test_well_formed_report_not_reusable_when_current_sha_is_none() -> None:
    """Pins clause 4's False branch when the report DOES carry a real head
    SHA but the caller doesn't know the current one: a real sha is never
    equal to None."""
    text = _report("glm-5.2", _REAL_BODY, "sha-abc123")
    assert report_is_reusable(text, None) is False


def test_headerless_report_not_reusable_against_a_concrete_sha() -> None:
    """Pins clause 4's False branch for the documented "no head SHA -> not
    reusable" case: extract_head_ref_oid returns None for a report with no
    ``<!-- PR head SHA: -->`` comment (built via _report(head_ref_oid=None)),
    so None != any concrete sha."""
    text = _report("glm-5.2", _REAL_BODY, None)
    assert report_is_reusable(text, "sha-abc123") is False


def test_headerless_report_is_not_reusable_when_the_current_head_is_also_unknown() -> None:
    """Regression for a fail-open found while writing these tests.

    The head clause was originally a bare equality
    (``extract_head_ref_oid(text) == current_head_sha``). It reads as though it
    already rejects a report with no head SHA, but with BOTH sides ``None`` --
    a headerless report compared against an unknown current head -- it is
    ``None == None``, i.e. True. So the report was judged *reusable* precisely
    when its head was unadjudicable, which is the exact condition issue #1081
    was filed about (``headRefOid`` absent at generation time) and the same
    indeterminate-comparison-collapses-to-permissive shape #1079 closed one
    layer up.

    Both sides must now be known before they are compared, so this fails
    closed. Without the ``is None`` guard this assertion flips to True.
    """
    text = _report("glm-5.2", _REAL_BODY, None)
    assert report_is_reusable(text, None) is False


def test_empty_string_not_reusable() -> None:
    """Pins clause 1 (``if not text.strip()``) and that it guards the later
    ``.splitlines()[0]`` indexing in clause 2 -- an empty string must not
    raise IndexError."""
    assert report_is_reusable("", "sha-abc123") is False


def test_whitespace_only_not_reusable() -> None:
    """Same clause as above (clause 1), whitespace-only variant."""
    assert report_is_reusable("   \n\t  \n", "sha-abc123") is False


def test_unavailable_stub_not_reusable(tmp_path: Path) -> None:
    """Pins clause 2: a real ``_fail``-written stub carries "(UNAVAILABLE)"
    on its first line and is never reusable, even against a sha that would
    otherwise look plausible."""
    report_path = tmp_path / "cross-family-review.md"
    _fail(report_path, "glm-5.2", "cross-family review timed out after 600s")
    text = report_path.read_text(encoding="utf-8")
    assert text == (
        "# Cross-family adversarial review — `glm-5.2` (UNAVAILABLE)\n\n"
        "> cross-family review timed out after 600s\n"
    )
    assert report_is_reusable(text, "sha-abc123") is False


def test_blocked_refusal_body_not_reusable_despite_matching_sha_and_marker() -> None:
    """Pins clause 3: report_body_is_valid's refusal check (_BLOCKED_RE)
    rejects a blocked/refusal message even when it also contains a severity
    marker and even when the head SHA matches exactly -- the refusal check
    runs before the severity-marker check inside report_body_is_valid."""
    blocked_body = (
        "I am blocked from performing the review due to tool restrictions.\n\n"
        "**BLOCKER** unreachable finding"
    )
    text = _report("glm-5.2", blocked_body, "sha-abc123")
    assert report_is_reusable(text, "sha-abc123") is False


def test_unavailable_word_in_body_not_first_line_is_reusable() -> None:
    """Pins that clause 2's stub check is scoped to
    ``text.splitlines()[0]`` only: the literal string "(UNAVAILABLE)"
    appearing later in a genuinely valid body (e.g. quoting a prior failed
    run) must not cause a real review to be rejected."""
    body = (
        "**BLOCKER**\nreal finding here\n\n"
        "Note: the previous automated run was (UNAVAILABLE) due to a timeout.\n\n"
        "Verdict: request changes"
    )
    text = _report("glm-5.2", body, "sha-xyz789")
    assert text.splitlines()[0] == "# Cross-family adversarial review — `glm-5.2`"
    assert report_is_reusable(text, "sha-xyz789") is True
