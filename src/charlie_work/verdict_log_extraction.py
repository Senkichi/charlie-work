"""Plaintext reviewer-log verdict extraction (issue #2143).

Split out of ``verdict_parsing.py`` to keep that module under its file-size
ratchet mark. ``verdict_parsing._parse_review_verdict_from_log`` calls
``extract_verdict_from_log_text`` with a function-level import: this module
needs ``_validate_review_verdict`` and ``_extract_verdict_from_text`` from it,
so a top-level import would be a cycle.
"""

from __future__ import annotations

import json
import re
from typing import Any

from . import markdown_fence, markdown_guard
from .verdict_parsing import (
    _extract_verdict_from_stream_json,
    _extract_verdict_from_text,
    _validate_review_verdict,
)

_LINE_START_JSON_OPENER_RE = re.compile(r"^```json[ \t]*$")


def extract_trailing_fenced_verdict(text: str) -> dict[str, Any] | None:
    """Verdict from the well-formed ```json block that ENDS the text, else ``None``.

    Scans backwards from the end: the last non-blank line must be a bare
    closing fence, and the nearest earlier fence-looking line must be a
    line-start ```json opener. Nothing before that opener is consulted, so a
    verdict-shaped object inside an earlier illustrative example (with or
    without balanced fences) can never be selected. Anything else -- no
    trailing fence, a glued or differently tagged opener, an invalid verdict
    body -- returns ``None`` and the caller falls back to the whole-text scan.
    """
    lines = text.rstrip().splitlines()
    if not lines or lines[-1].rstrip() != "```":
        return None
    for index in range(len(lines) - 2, -1, -1):
        line = lines[index]
        if not line.startswith("```"):
            continue
        if not _LINE_START_JSON_OPENER_RE.match(line):
            return None
        try:
            data = json.loads("\n".join(lines[index + 1 : -1]).strip())
        except json.JSONDecodeError:
            return None
        return _validate_review_verdict(data)
    return None


def extract_verdict_from_plaintext_log(log_text: str) -> dict[str, Any] | None:
    """Whole-text verdict extraction for a plaintext log.

    The raw text is consulted first, so boundary normalization can never
    destroy a verdict the unmodified text yields. ``restore_message_boundaries``
    is only a fallback when the raw pass finds nothing, or a way to settle a
    raw-pass legacy/scan disagreement when the normalized text confirms the
    raw pass's own decision; it never changes the decision the raw pass chose.
    """
    raw_events: list[markdown_guard.Disagreement] = []
    verdict = _extract_verdict_from_text(log_text, on_disagreement=raw_events.append)
    normalized = markdown_fence.restore_message_boundaries(log_text)
    if verdict is None:
        return _extract_verdict_from_text(normalized)
    if raw_events and normalized != log_text:
        normalized_events: list[markdown_guard.Disagreement] = []
        confirmed = _extract_verdict_from_text(
            normalized, on_disagreement=normalized_events.append
        )
        if (
            confirmed is not None
            and confirmed["decision"] == verdict["decision"]
            and not normalized_events
        ):
            return verdict
    for event in raw_events:
        markdown_guard.emit_disagreement(event)
    return verdict


def extract_verdict_from_log_text(log_text: str) -> dict[str, Any] | None:
    """Verdict from a reviewer sidecar log (plaintext or stream-json), else ``None``.

    Order: the trailing well-formed block, the whole-text plaintext scan, then
    stream-json event decoding.
    """
    verdict = extract_trailing_fenced_verdict(log_text)
    if verdict is not None:
        return verdict
    verdict = extract_verdict_from_plaintext_log(log_text)
    if verdict is not None:
        return verdict
    return _extract_verdict_from_stream_json(log_text)
