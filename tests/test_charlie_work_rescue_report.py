"""Rescue-review report validation: report_body_is_valid markers and extract_report_body unwrapping.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

from charlie_work.rescue_review import (
    _CAVEAT,
    extract_report_body,
    report_body_is_valid,
)


def test_report_body_is_valid_detects_real_review_vs_blocked() -> None:
    assert report_body_is_valid("**MAJOR**\nissue\n\nVerdict: safe") is True
    assert report_body_is_valid("Verdict: safe") is True
    assert report_body_is_valid("Verdict: no permission issues found") is True
    blocked = (
        "I'm blocked from performing the review. All tool calls are being rejected. Please re-run."
    )
    assert report_body_is_valid(blocked) is False
    assert report_body_is_valid("Verdict: blocked from performing the review") is False
    assert report_body_is_valid("") is False


def test_report_body_is_valid_rejects_blocked_output_with_bold_markers() -> None:
    """Regression for issue #38: bold markdown in a blocked refusal must not
    short-circuit validation and allow the blocked output to be cached.
    """
    blocked_with_bold = "**Unable to review** — all tool calls are being rejected. Please re-run."
    assert report_body_is_valid(blocked_with_bold) is False


def test_report_body_is_valid_accepts_heading_style_markers() -> None:
    """Cross-family models (e.g. kimi-k3) emit findings as ``### NIT —`` headings
    and a ``## Verdict`` heading instead of ``**NIT**`` bold / ``Verdict:`` line.
    These are real reviews and must not be falsely rejected as UNAVAILABLE.
    """
    heading_severity = (
        "### NIT — worktree.py:2977 (new): dead weight\n\n"
        "## Verdict\n\n**Approve** — claims verified."
    )
    assert report_body_is_valid(heading_severity) is True
    # Heading verdict alone (no severity heading) is also valid.
    assert report_body_is_valid("## Verdict\n\nApprove") is True
    # Heading severity alone (no verdict heading) is also valid.
    assert report_body_is_valid("### MAJOR — bug.py:10: off-by-one\n\nfix it") is True


def test_extract_report_body_strips_wrapper_but_preserves_model_output() -> None:
    body = "**MAJOR**\nissue\n\nVerdict: safe"
    wrapped = f"# Cross-family adversarial review — `codex`\n\n{_CAVEAT}\n\n---\n\n{body}\n"
    assert extract_report_body(wrapped) == body
    assert extract_report_body(body) == body
