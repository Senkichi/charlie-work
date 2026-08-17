"""Tests for parse_cross_family_verdict and CrossFamilyVerdict parsing, carved out of test_charlie_work.py (#1284)."""

from __future__ import annotations

import pytest

from charlie_work.cross_family import (
    CrossFamilyVerdict,
    MalformedCrossFamilyVerdict,
    _CAVEAT,
    parse_cross_family_verdict,
    report_body_is_valid,
)


def test_parse_cross_family_verdict_approved_no_blockers() -> None:
    """A report with only MINOR/NIT findings parses to approved."""
    body = "**MINOR**\nsmall issue\n\nVerdict: No BLOCKERs or MAJORs — fix is correct"
    wrapped = f"# Cross-family adversarial review — `glm-5.2`\n\n{_CAVEAT}\n\n---\n\n{body}\n"
    result = parse_cross_family_verdict(wrapped)
    assert result is not None
    assert result.decision == "approved"
    assert "No BLOCKERs" in result.summary
    assert result.required_changes == ()


def test_parse_cross_family_verdict_request_changes_with_blocker() -> None:
    """A report with a BLOCKER finding parses to request_changes."""
    body = "**BLOCKER**\ncritical bug\n\nVerdict: BLOCKER — does not fix the issue"
    wrapped = f"# Cross-family adversarial review — `glm-5.2`\n\n{_CAVEAT}\n\n---\n\n{body}\n"
    result = parse_cross_family_verdict(wrapped)
    assert result is not None
    assert result.decision == "request_changes"
    assert "BLOCKER" in result.summary
    assert result.required_changes == ()


def test_parse_cross_family_verdict_request_changes_with_major() -> None:
    """A report with a MAJOR finding parses to request_changes."""
    body = "**MAJOR**\nreal bug\n\nVerdict: MAJOR issues block merge"
    wrapped = f"# Cross-family adversarial review — `glm-5.2`\n\n{_CAVEAT}\n\n---\n\n{body}\n"
    result = parse_cross_family_verdict(wrapped)
    assert result is not None
    assert result.decision == "request_changes"


def test_parse_cross_family_verdict_heading_style_major() -> None:
    """Heading-style ``### MAJOR`` markers are detected as request_changes."""
    body = "### MAJOR — bug.py:10: off-by-one\n\nfix it\n\n## Verdict\n\nMAJOR should be fixed"
    wrapped = f"# Cross-family adversarial review — `glm-5.2`\n\n{_CAVEAT}\n\n---\n\n{body}\n"
    result = parse_cross_family_verdict(wrapped)
    assert result is not None
    assert result.decision == "request_changes"


def test_parse_cross_family_verdict_unavailable_returns_none() -> None:
    """An UNAVAILABLE stub report returns None (skip, don't record a wrong verdict)."""
    stub = "# Cross-family adversarial review — `glm-5.2` (UNAVAILABLE)\n\n> timed out\n"
    assert parse_cross_family_verdict(stub) is None


def test_parse_cross_family_verdict_empty_returns_none() -> None:
    """Empty/blank report text returns None."""
    assert parse_cross_family_verdict("") is None
    assert parse_cross_family_verdict("   ") is None


def test_parse_cross_family_verdict_invalid_body_returns_none() -> None:
    """A report body that fails report_body_is_valid returns None."""
    body = "some random text with no severity or verdict"
    wrapped = f"# Cross-family adversarial review — `glm-5.2`\n\n{_CAVEAT}\n\n---\n\n{body}\n"
    assert parse_cross_family_verdict(wrapped) is None


def test_parse_cross_family_verdict_json_block_populates_required_changes() -> None:
    """The defect this covers: a new-format report's JSON verdict block carries
    itemized findings into ``required_changes`` instead of leaving it empty."""
    body = (
        "**MAJOR**\nfile.py:10 does the wrong thing\n\n"
        "Verdict: MAJOR issue blocks merge\n\n"
        "```json\n"
        '{"decision": "request_changes", '
        '"summary": "file.py:10 has a real bug that breaks X", '
        '"required_changes": ["Fix the off-by-one in file.py:10", '
        '"Add a regression test for the empty-list case"]}\n'
        "```\n"
    )
    wrapped = f"# Cross-family adversarial review — `glm-5.2`\n\n{_CAVEAT}\n\n---\n\n{body}\n"
    result = parse_cross_family_verdict(wrapped)
    assert result is not None
    assert result == CrossFamilyVerdict(
        decision="request_changes",
        summary="file.py:10 has a real bug that breaks X",
        required_changes=(
            "Fix the off-by-one in file.py:10",
            "Add a regression test for the empty-list case",
        ),
    )


def test_parse_cross_family_verdict_json_approved_no_markdown_severity() -> None:
    """A JSON-only body (no ``**SEVERITY**``/``Verdict:`` markers) still parses --
    exercises report_body_is_valid's JSON-block fallback."""
    body = (
        '```json\n{"decision": "approved", "summary": "clean PR, only a style nit", '
        '"required_changes": []}\n```\n'
    )
    wrapped = f"# Cross-family adversarial review — `glm-5.2`\n\n{_CAVEAT}\n\n---\n\n{body}\n"
    assert report_body_is_valid(body) is True
    result = parse_cross_family_verdict(wrapped)
    assert result == CrossFamilyVerdict(decision="approved", summary="clean PR, only a style nit")


def test_parse_cross_family_verdict_json_approved_overridden_by_body_severity() -> None:
    """Fail-safe: a **MAJOR** marker in the Markdown findings always overrides a
    JSON block that claims "approved" -- this verdict auto-records and can
    unblock the merge lane, so a self-contradicting downgrade is never trusted."""
    body = (
        "**MAJOR**\nfile.py:20 real bug\n\nVerdict: MAJOR should block\n\n"
        '```json\n{"decision": "approved", "summary": "looks fine overall", '
        '"required_changes": []}\n```\n'
    )
    wrapped = f"# Cross-family adversarial review — `glm-5.2`\n\n{_CAVEAT}\n\n---\n\n{body}\n"
    result = parse_cross_family_verdict(wrapped)
    assert result is not None
    assert result.decision == "request_changes"


def test_parse_cross_family_verdict_json_request_changes_empty_list_is_malformed() -> None:
    """Issue #784 (AC-2): a JSON verdict block that declares request_changes
    with an empty/missing required_changes list is a contract violation and
    must NOT silently fall back to the legacy Markdown parse -- even though
    the legacy parse would find a perfectly usable ``Verdict:`` line here.
    Pre-#784, this test asserted exactly that fall-through as correct
    behavior; that assertion encoded the defect itself (issue #784's root
    cause: "a JSON verdict saying request_changes with empty
    required_changes falls through to the weaker legacy path"), so it is
    replaced rather than preserved. The contract is: once a reviewer elects
    the structured JSON format, empty required_changes is untrusted on its
    own, full stop -- regardless of what the Markdown body also says."""
    body = (
        "**MAJOR**\nfile.py:30 bug\n\nVerdict: MAJOR should block\n\n"
        '```json\n{"decision": "request_changes", "summary": "bad but no list", '
        '"required_changes": []}\n```\n'
    )
    wrapped = f"# Cross-family adversarial review — `glm-5.2`\n\n{_CAVEAT}\n\n---\n\n{body}\n"
    result = parse_cross_family_verdict(wrapped)
    assert isinstance(result, MalformedCrossFamilyVerdict)
    assert result.reason == "json_verdict_request_changes_missing_required_changes"


def test_parse_cross_family_verdict_json_block_drives_decision_legacy_would_not_reach() -> None:
    """The JSON block is the decision authority, not a passenger on a legacy
    verdict that happens to agree: a body with only a **MINOR** marker and no
    ``Verdict:`` line -- which the legacy parser alone would call "approved"
    (no BLOCKER/MAJOR marker present) -- must come back as request_changes
    when the JSON block says so with a populated required_changes list. This
    would fail if the JSON check were ever reordered below the legacy path."""
    body = (
        "**MINOR**\nfile.py:5 minor style issue, not a real bug\n\n"
        "```json\n"
        '{"decision": "request_changes", '
        '"summary": "actually needs a behavioral fix despite the minor framing", '
        '"required_changes": ["Handle the None case in file.py:5"]}\n'
        "```\n"
    )
    wrapped = f"# Cross-family adversarial review — `glm-5.2`\n\n{_CAVEAT}\n\n---\n\n{body}\n"
    result = parse_cross_family_verdict(wrapped)
    assert result == CrossFamilyVerdict(
        decision="request_changes",
        summary="actually needs a behavioral fix despite the minor framing",
        required_changes=("Handle the None case in file.py:5",),
    )


def test_parse_cross_family_verdict_legacy_report_unchanged_by_json_support() -> None:
    """Backward-compatibility regression guard: a historical report with no
    JSON verdict block at all parses to the exact same decision/summary the
    pre-JSON-support parser produced, with an empty required_changes."""
    body = "**BLOCKER**\ncritical bug\n\nVerdict: BLOCKER — does not fix the issue"
    wrapped = f"# Cross-family adversarial review — `glm-5.2`\n\n{_CAVEAT}\n\n---\n\n{body}\n"
    result = parse_cross_family_verdict(wrapped)
    assert result == CrossFamilyVerdict(
        decision="request_changes",
        summary="BLOCKER — does not fix the issue",
        required_changes=(),
    )


def test_parse_cross_family_verdict_legacy_blocker_with_no_summary_is_malformed() -> None:
    """Issue #784 AC-1: a BLOCKER/MAJOR marker with no ``Verdict:`` line to
    extract a summary from is genuinely content-free -- it must return
    MalformedCrossFamilyVerdict, never a request_changes verdict asserting
    blockers exist while naming none. No hardcoded placeholder summary is
    substituted (the pre-#784 behavior this fix replaces)."""
    body = "**BLOCKER**\ncritical bug, but no Verdict: line anywhere in this report"
    wrapped = f"# Cross-family adversarial review — `glm-5.2`\n\n{_CAVEAT}\n\n---\n\n{body}\n"
    result = parse_cross_family_verdict(wrapped)
    assert isinstance(result, MalformedCrossFamilyVerdict)
    assert result.reason == "blocker_or_major_with_no_extractable_summary"
    assert "BLOCKER" in result.raw_body


def test_parse_cross_family_verdict_bold_inline_verdict_marker() -> None:
    """Regression: some cross-family models (e.g. glm-5.2) emit the verdict as
    a bold-inline ``**Verdict:**`` marker within a paragraph, rather than a
    bare ``Verdict:`` line or a ``## Verdict`` heading. The pre-fix
    ``_VERDICT_RE`` matched neither the bare-colon nor the heading form, so
    every such report (PRs #680, #690, #692, #699, #700 in production, all
    with real BLOCKER/MAJOR findings and a readable verdict) fell through to
    the "no extractable summary" branch and was misclassified as
    ``MalformedCrossFamilyVerdict`` despite the verdict being right there."""
    body = (
        "**MAJOR**\nfile.py:10 real bug\n\n"
        "**Verdict:** Approve with a required follow-up — MAJOR 1 is a real "
        "correctness bug that must be fixed before this claim can be trusted."
    )
    wrapped = f"# Cross-family adversarial review — `glm-5.2`\n\n{_CAVEAT}\n\n---\n\n{body}\n"
    result = parse_cross_family_verdict(wrapped)
    assert result == CrossFamilyVerdict(
        decision="request_changes",
        summary=(
            "Approve with a required follow-up — MAJOR 1 is a real "
            "correctness bug that must be fixed before this claim can be trusted."
        ),
        required_changes=(),
    )


def test_parse_cross_family_verdict_json_block_after_language_tagged_code_fences() -> None:
    """Regression for PR #802's real failure shape: a report that cites code
    in ```python fences before its final ```json verdict fence. The pre-fix
    ``_VERDICT_FENCE_RE`` (``` ```(?:json)?\\s*\\n `` ``) only recognized an
    opening fence tagged bare or ``json`` -- a ```python fence's own opening
    backtick never matched, so ``finditer`` instead paired that block's
    *closing* bare ``` with the *next* fence's opening as a bogus "match",
    permanently desynchronizing every fence pair after it and hiding the
    genuinely well-formed trailing JSON verdict entirely (confirmed
    byte-for-byte against PR #802's on-disk report)."""
    body = (
        "**MAJOR**\nfile.py:10 real bug\n\n"
        "```python\n"
        "total_running = sum(t.running for t in plan.targets)\n"
        "```\n\n"
        "some prose explaining the first citation\n\n"
        "```python\n"
        "planned_running = sum(t.target for t in plan.targets)\n"
        "```\n\n"
        "some prose explaining the second citation\n\n"
        "```json\n"
        '{"decision": "request_changes", "summary": "real bug in the spare-budget gate", '
        '"required_changes": ["Fix the gate to use planned running, not actual running"]}\n'
        "```\n"
    )
    wrapped = f"# Cross-family adversarial review — `glm-5.2`\n\n{_CAVEAT}\n\n---\n\n{body}\n"
    result = parse_cross_family_verdict(wrapped)
    assert result == CrossFamilyVerdict(
        decision="request_changes",
        summary="real bug in the spare-budget gate",
        required_changes=("Fix the gate to use planned running, not actual running",),
    )


def test_cross_family_verdict_post_init_rejects_content_free_request_changes() -> None:
    """Issue #784 AC-6: the invalid state -- request_changes with neither
    itemized required_changes nor a real summary -- must be unrepresentable
    at construction, not just avoided by callers that remember to check."""
    with pytest.raises(ValueError, match="content-free"):
        CrossFamilyVerdict(decision="request_changes", summary="", required_changes=())


def test_cross_family_verdict_post_init_rejects_whitespace_only_summary() -> None:
    """Whitespace-only is not a real summary either -- ``.strip()`` is
    applied before the emptiness check, so padding cannot smuggle a
    content-free verdict past the guard."""
    with pytest.raises(ValueError, match="content-free"):
        CrossFamilyVerdict(decision="request_changes", summary="   \n  ", required_changes=())


def test_cross_family_verdict_post_init_allows_request_changes_with_only_summary() -> None:
    """Narrower than "always require required_changes": the legacy Markdown
    parse path never itemizes findings, so a request_changes verdict with a
    real extracted summary and empty required_changes remains legitimate
    and constructible -- this is exactly what the legacy-path tests above
    rely on."""
    verdict = CrossFamilyVerdict(
        decision="request_changes", summary="a real extracted summary", required_changes=()
    )
    assert verdict.summary == "a real extracted summary"


def test_cross_family_verdict_post_init_allows_request_changes_with_only_required_changes() -> (
    None
):
    """A JSON-block verdict with itemized required_changes but an empty
    summary is also legitimate -- required_changes alone is something a
    rework brief can act on."""
    verdict = CrossFamilyVerdict(
        decision="request_changes", summary="", required_changes=("fix the null check",)
    )
    assert verdict.required_changes == ("fix the null check",)


def test_cross_family_verdict_post_init_allows_approved_with_empty_summary() -> None:
    """The guard is scoped to ``request_changes`` only -- an approved
    verdict never needs anything for a rework brief to act on, so an empty
    summary there is unaffected."""
    verdict = CrossFamilyVerdict(decision="approved", summary="")
    assert verdict.decision == "approved"
