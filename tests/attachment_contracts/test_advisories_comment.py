"""Tests for issue #1466: worker-published advisories PR-comment channel.

Covers ``parse_advisories_comment`` (the pure parser) and
``ADVISORY_COMMENT_MARKER``. The I/O path that scans PR comments and calls
the parser (``OrchestratorApp._read_advisories_from_pr_comment``) is
exercised end-to-end in ``tests/test_attachment_budget_packet.py`` against
``FakeGitHub``.
"""

from __future__ import annotations

import json

from charlie_work.attachment_contracts.hook_entry import (
    ADVISORY_COMMENT_MARKER,
    parse_advisories_comment,
)
from charlie_work.attachment_contracts.model import AdvisoryRecord


def _comment(records: list[dict]) -> str:
    body = ADVISORY_COMMENT_MARKER + "\n```json\n" + json.dumps(records) + "\n```\n"
    return body


def test_non_marker_body_returns_none() -> None:
    """A body that does not start with the marker is not an advisories
    comment -- ``None`` signals "no channel present" to the caller."""
    assert parse_advisories_comment("just a regular review comment") is None
    assert parse_advisories_comment("") is None
    # Marker as a substring but not a prefix -- must not match (prefix test,
    # matching ORCHESTRATOR_COMMENT_MARKER's discipline).
    assert parse_advisories_comment("see " + ADVISORY_COMMENT_MARKER + " below") is None


def test_marker_present_empty_array_returns_empty_tuple() -> None:
    """A present marker with an empty JSON array is a present channel with
    zero records -- ``()`` (not ``None``) so the caller does NOT fall back
    to the local log."""
    result = parse_advisories_comment(_comment([]))
    assert result is not None
    assert result == ()


def test_marker_present_parses_records() -> None:
    records = [
        {
            "severity": "block",
            "file": "src/foo.py",
            "identity": "Foo",
            "message": "Foo is saturated",
            "redirect": "src/foo_extra.py",
            "timestamp": "2026-08-25T00:00:00+00:00",
        },
        {
            "severity": "error",
            "file": "src/bar.py",
            "identity": "Bar",
            "message": "Bar tamper",
            "redirect": None,
            "timestamp": None,
        },
    ]
    result = parse_advisories_comment(_comment(records))
    assert result is not None
    assert result == (
        AdvisoryRecord(
            severity="block",
            file="src/foo.py",
            identity="Foo",
            message="Foo is saturated",
            redirect="src/foo_extra.py",
            timestamp="2026-08-25T00:00:00+00:00",
        ),
        AdvisoryRecord(
            severity="error",
            file="src/bar.py",
            identity="Bar",
            message="Bar tamper",
            redirect=None,
            timestamp=None,
        ),
    )


def test_old_shape_record_without_redirect_or_timestamp_parses() -> None:
    """Same tolerance as ``read_advisories``: records missing the optional
    ``redirect``/``timestamp`` fields still parse (both default to None)."""
    result = parse_advisories_comment(
        _comment(
            [
                {
                    "severity": "block",
                    "file": "src/old.py",
                    "identity": "Old",
                    "message": "old shape",
                }
            ]
        )
    )
    assert result is not None
    assert len(result) == 1
    rec = result[0]
    assert rec.redirect is None
    assert rec.timestamp is None


def test_malformed_elements_are_skipped_best_effort() -> None:
    """Non-object / missing-field elements are skipped, never raise --
    mirrors ``read_advisories``'s best-effort contract."""
    result = parse_advisories_comment(
        _comment(
            [
                "not an object",
                {"severity": "block"},  # missing required fields
                {
                    "severity": "block",
                    "file": "src/ok.py",
                    "identity": "Ok",
                    "message": "fine",
                },
            ]
        )
    )
    assert result is not None
    assert len(result) == 1
    assert result[0].identity == "Ok"


def test_marker_present_no_fence_returns_empty_tuple() -> None:
    """Marker present but no fenced block -> present channel, zero records."""
    result = parse_advisories_comment(ADVISORY_COMMENT_MARKER + "\nno fence here\n")
    assert result is not None
    assert result == ()


def test_marker_present_malformed_json_returns_empty_tuple() -> None:
    """Marker present, fence present, but body is not valid JSON -> present
    channel, zero records (best-effort, never raise)."""
    body = ADVISORY_COMMENT_MARKER + "\n```json\nnot json {{{\n```\n"
    result = parse_advisories_comment(body)
    assert result is not None
    assert result == ()


def test_marker_present_json_object_not_array_returns_empty_tuple() -> None:
    """The fence must contain a JSON ARRAY. A bare object is the wrong shape
    -> present channel, zero records."""
    body = (
        ADVISORY_COMMENT_MARKER
        + "\n```json\n"
        + json.dumps({"severity": "block", "file": "x", "identity": "y", "message": "z"})
        + "\n```\n"
    )
    result = parse_advisories_comment(body)
    assert result is not None
    assert result == ()


def test_leading_whitespace_before_marker_still_matches() -> None:
    """GitHub may preserve leading whitespace; the marker check is
    ``lstrip().startswith``."""
    body = "  \n" + _comment([{"severity": "block", "file": "x", "identity": "y", "message": "z"}])
    result = parse_advisories_comment(body)
    assert result is not None
    assert len(result) == 1


def test_untagged_fence_still_parsed() -> None:
    """A fence opening with bare backticks (no ``json`` tag) is still a
    fence -- the parser does not require the tag."""
    body = (
        ADVISORY_COMMENT_MARKER
        + "\n```\n"
        + json.dumps([{"severity": "block", "file": "x", "identity": "y", "message": "z"}])
        + "\n```\n"
    )
    result = parse_advisories_comment(body)
    assert result is not None
    assert len(result) == 1
