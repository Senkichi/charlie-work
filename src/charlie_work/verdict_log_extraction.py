"""Plaintext reviewer-log verdict extraction (issue #2143).

Split out of ``verdict_parsing.py`` to keep that module under its file-size
ratchet mark. The two modules are mutually dependent --
``verdict_parsing._parse_review_verdict_from_log`` calls
``extract_verdict_from_log_text`` here, and this module calls
``_validate_review_verdict``/``_extract_verdict_from_text`` back -- so each
binds the other's MODULE (``from . import verdict_parsing`` /
``from . import verdict_log_extraction``) and resolves names through
attribute access at call time. That keeps either import order safe (a
partially-initialized module binds fine; its attributes are only needed
after both modules finish loading) and keeps the monkeypatch seam honest:
patching ``verdict_parsing._extract_verdict_from_text`` propagates here,
and patching this module's own names propagates back.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .opencode_log import OpencodeProgress

from . import markdown_fence, markdown_guard, verdict_parsing
from .claude_code import extract_event_text, iter_stream_json_events
from .config import OrchestratorConfig
from .throttle_signatures import match_quota_tail, match_throttle_tail

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
        return verdict_parsing._validate_review_verdict(data)
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
    verdict = verdict_parsing._extract_verdict_from_text(
        log_text, on_disagreement=raw_events.append
    )
    normalized = markdown_fence.restore_message_boundaries(log_text)
    if verdict is None:
        return verdict_parsing._extract_verdict_from_text(normalized)
    if raw_events and normalized != log_text:
        normalized_events: list[markdown_guard.Disagreement] = []
        confirmed = verdict_parsing._extract_verdict_from_text(
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

    Order: the trailing well-formed block, the whole-text plaintext scan,
    stream-json event decoding, then opencode JSONL decoding (``--format
    json`` emits the same JSON-escaped-in-a-string shape as stream-json, but
    with ``{"type":"text","part":{"text":...}}`` events instead of claude's
    assistant/result shapes).
    """
    verdict = extract_trailing_fenced_verdict(log_text)
    if verdict is not None:
        return verdict
    verdict = extract_verdict_from_plaintext_log(log_text)
    if verdict is not None:
        return verdict
    verdict = verdict_parsing._extract_verdict_from_stream_json(log_text)
    if verdict is not None:
        return verdict
    from .opencode_log import last_text_event

    final_text = last_text_event(log_text)
    return verdict_parsing._extract_verdict_from_text(final_text) if final_text else None


def recover_review_log_texts(log_text: str) -> tuple[list[str], "OpencodeProgress | None"]:
    """Assistant texts (and progress) recovered from a reviewer session log.

    Tries the claude stream-json decode first. An opencode ``--format json``
    log is JSONL too, but its ``text``/``step_start``/``tool_use``/``error``
    shapes are invisible to the claude decoder -- without this leg the
    caller's raw-prose fallback dumps raw JSONL into the PR comment and a
    worked session misclassifies as launch_failed on 0/0 counts (the
    ``_extract_review_session_summary`` consumer). Returns the decoded texts
    plus ``OpencodeProgress`` for opencode logs, ``(texts, None)`` otherwise.
    """
    texts: list[str] = []
    for event in iter_stream_json_events(log_text):
        text = extract_event_text(event)
        if text.strip():
            texts.append(text.strip())
    if texts:
        return texts, None
    from .opencode_log import session_progress

    oc = session_progress(log_text)
    if oc is not None:
        texts.extend(oc.texts)
    return texts, oc


# --- reviewer-log throttle tails (moved out of verdict_parsing.py) -----------
#
# ``_extract_review_session_summary`` consults these to distinguish an
# environmental provider death (launch_failed) from a PR-level one
# (died_mid_session). They live here -- next to the log-decoding helpers they
# share a boundary with -- to keep verdict_parsing.py under its file-size
# ratchet mark.

# Tail length for the raw-log session-limit match. Mirrors the 2048-char tail
# used by the stalled-session sweep (the ``log_text[-2048:]`` slice at the
# ``Path(w.log_path).read_text(...)`` call in
# _handle_stalled_review_sessions): the CLI prints its session-limit notice at
# the very end of the log, so the tail isolates the death message from the
# multi-turn analysis prose earlier in the log.
_REVIEW_THROTTLE_TAIL_CHARS = 2048


# Broad throttle/quota marker sets, only ever applied to provider_error_digest
# output -- opencode's OWN error records -- never to raw log tails, where the
# same generic phrasing ("rate limit", "usage limit") legitimately appears in
# review commentary quoting the source under review (the #652 boundary).
#
# Resolved at CALL time, not import time: this module is on verdict_parsing's
# import path, and instantiating OrchestratorConfig() at module scope would
# make every importer pay config construction (and freeze the markers before a
# caller that rebinds the runtime config could take effect). The one
# production call site passes its own config's markers anyway; these are the
# fallback for direct callers.
def _default_review_throttle_error_markers() -> tuple[str, ...]:
    """The ``RuntimeConfig.throttle_error_markers`` defaults, resolved now."""
    return tuple(OrchestratorConfig().runtime.throttle_error_markers)


def _default_review_quota_error_markers() -> tuple[str, ...]:
    """The ``RuntimeConfig.quota_error_markers`` defaults, resolved now."""
    return tuple(OrchestratorConfig().runtime.quota_error_markers)


def _log_died_on_provider_error(
    log_path: Path,
    throttle_markers: Sequence[str],
    quota_markers: Sequence[str],
) -> bool:
    """True when an opencode reviewer's own error records carry a throttle/quota signature.

    Unlike ``_log_tail_throttled`` this is safe to run against the broad
    ``throttle_error_markers``/``quota_error_markers`` sets:
    ``provider_error_digest`` strips everything that is not an opencode-owned
    error record -- assistant text, tool output, print-log noise -- so review
    commentary quoting the PR's source cannot trigger it. Only the digest's
    tail is matched (same 2048-char isolation as the raw-tail sweep).
    """
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    if not log_text:
        return False
    from .opencode_log import provider_error_digest

    digest = provider_error_digest(log_text)
    if not digest:
        return False
    tail = (
        digest[-_REVIEW_THROTTLE_TAIL_CHARS:]
        if len(digest) > _REVIEW_THROTTLE_TAIL_CHARS
        else digest
    )
    return match_throttle_tail(tail, throttle_markers)[0] or match_quota_tail(tail, quota_markers)


def _log_tail_throttled(log_path: Path, markers: Sequence[str]) -> bool:
    """Return True when the raw process log's tail contains a session-limit marker.

    Reads the last ``_REVIEW_THROTTLE_TAIL_CHARS`` chars of ``log_path`` and
    matches against ``markers`` via ``match_throttle_tail``. This is the same
    raw-log-tail boundary the stalled-session sweep uses (the
    ``log_text[-2048:]`` slice at the ``Path(w.log_path).read_text(...)`` call
    in ``_handle_stalled_review_sessions``). Missing or unreadable logs do not
    match.

    Note (issue #652 review): a tee'd claude-code reviewer launch writes the
    same stream to ``log_path`` and ``events_path``, so the raw log tail IS
    the events content and matching against it does NOT avoid the reviewer's
    own parsed assistant text. (That is per-harness: the opencode launch path
    never tees -- ``cli_pins`` skips the stream-json tee -- and its digest
    leg, :func:`_log_died_on_provider_error`, is the one that can safely use
    the broad marker sets.) The false-positive protection HERE comes from the
    NARROW ``session_limit_markers`` list (specific CLI death-message
    phrasing, not generic domain terms), not from the raw-vs-parsed
    distinction. The ``tool_call_count == 0`` guard in the caller is
    defense-in-depth.
    """
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    if not log_text:
        return False
    if len(log_text) > _REVIEW_THROTTLE_TAIL_CHARS:
        tail = log_text[-_REVIEW_THROTTLE_TAIL_CHARS:]
    else:
        tail = log_text
    return match_throttle_tail(tail, markers)[0]
