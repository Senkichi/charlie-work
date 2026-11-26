"""Provider-error digest of an opencode worker log (``AdapterFateProfile.log_digest``).

An opencode worker's log (``opencode run --format json --print-logs``, stdout and
stderr merged) carries tool OUTPUT -- file contents, grep hits, test logs. In this
repo those routinely contain the very strings the quota/throttle classifiers
match ("rate limit", "usage limit"), so a bare substring match over the raw tail
would restrict the opencode fallback fleet-wide on a worker that merely read
throttle code. The digest keeps only opencode's OWN provider-error records:

- a ``--print-logs`` line ``timestamp=... level=ERROR ... error.error="<msg>"``
  (one per failed attempt, written before opencode sleeps out retry-after), and
- the terminal stdout event ``{"type":"error", ...}`` (name, status, message and
  the response body that names ``GoUsageLimitError``).

Both surfaces ship TWO wire shapes, one per opencode major version, and every
reader below normalizes BOTH (runtime-verified against the installed 2.3.0
binary, 2026-11-18; the 1.x shapes were verified 2026-10-06 against 1.18.35):

- 1.x: the event's error payload nests the fields under ``error.data``
  (``{name, data: {statusCode, isRetryable, message, responseBody}}``) and the
  print-log line carries ``error.error="..."``.
- 2.x: the structured error is flat
  (``{type, message, status?, response?: {body}, retryAfterMs?}`` -- the
  bundle's ``Session.StructuredError`` schema, embedded in the binary), the
  print-log line carries ``cause="AI.Error: <msg>"`` instead of
  ``error.error=``, and a failed attempt additionally emits a
  ``{"type":"retry", ..., "error": {...}}`` event before the retry sleep
  (``session.retry.scheduled`` -> JSON ``retry``, with ``attempt``,
  ``nextRetryAt`` and the SAME structured error). A 2.x session killed inside
  the retry sleep may therefore never emit the terminal ``error`` event -- the
  ``retry`` events are the only provider-error records, so the digest collects
  them (since the last progress event, like every other error record).

Only errors since the last sign of progress count: a stdout event carrying
work product (``text`` with non-blank content, ``tool_use``) resets the
digest, so a 429 that opencode retried through successfully never classifies
a later, unrelated death. A bare ``step_start`` is NOT progress -- a retry
attempt emits it before its HTTP response arrives, so resetting on it would
discard the stream errors that explain why the session died silent inside
the retry. ``step_finish`` is not progress either: it terminates the step
whose output (text/tool_use) already reset the digest, and after an
``error``/``retry``/print-log record it is the last event before the session
dies -- treating it as progress would erase the only provider-error evidence
from both this digest and the ``session_progress`` text fallback. Unknown event
kinds are likewise ignored, not treated as progress: a future opencode event
kind must not silently erase error evidence. An empty
``part.text`` is not progress either: ``SessionProcessor.cleanup`` flushes
``{"type":"text","part":{"text":""}}`` after an aborted stream.
Tool output is always JSON-escaped inside a stdout event line, so a quoted
``level=ERROR`` in tool output cannot forge a print-logs record.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_EVENT_TYPE = re.compile(r'^\{"type":"([A-Za-z_]+)"')
# --print-logs stream-error lines, both major-version shapes: 1.x carries
# ``error.error="<msg>"``; 2.x carries ``cause="AI.Error: <msg>"`` (the
# "Failed to drain Session" line). Both are written once per failed provider
# attempt, before opencode sleeps out retry-after.
_PRINT_LOG_ERROR = re.compile(
    r'^timestamp=(\S+) level=ERROR .*?(?:error\.error|cause)="((?:[^"\\]|\\.)*)"'
)
# Issue #2767: OpenCode Zen's free tier is providerID=opencode with a *-free
# modelID (Go is providerID=opencode-go and never serves *-free models).
_ZEN_FREE_MODEL = re.compile(
    r"\bproviderID=opencode\b.*\bmodelID=\S*-free\b|\bmodelID=\S*-free\b.*\bproviderID=opencode\b"
)
_ZEN_RATE_LIMIT = re.compile(r"AI_(?:APICall|Retry)Error:.*Rate limit exceeded", re.IGNORECASE)
_FIELD_CAP = 400

# Stamped on a Zen free-tier stream-error digest line (inside the
# "FreeUsageLimitError ..." tag) so ``go_quota_reset`` can tell that death
# apart from a plain FreeUsageLimitError body and skip the Go probe.
ZEN_FREE_LIMIT_MARKER = "ZenFreeUsageLimitError"


def opencode_data_dir(sessions_dir: Path, issue_number: int) -> Path:
    """The per-worker opencode ``XDG_DATA_HOME`` (db, logs, session transcripts).

    Single source for the launcher (which wipes it before a launch) and
    ``WorkerView.reap_sidecar`` (which removes it with the sidecar), so the
    directory never outlives the worker that owns it.
    """
    return sessions_dir / "opencode-data" / f"issue-{issue_number}"


def _error_summary_from_event(event: dict) -> str:
    """Normalized provider-error summary of one opencode error/retry event.

    Both major-version wire shapes are read (see the module docstring):

    - 1.x nests the fields under ``error.data`` (``statusCode``,
      ``isRetryable``, ``message``, ``responseBody``) with the error class in
      ``error.name``;
    - 2.x is flat (``type``, ``message``, ``status``, ``response.body``,
      ``retryAfterMs``) and ``retry`` events carry the SAME structured error
      at the event's top level instead of under ``part``.
    """
    error = event.get("error") if isinstance(event, dict) else None
    if not isinstance(error, dict):
        return ""
    raw_data = error.get("data")
    data: dict = raw_data if isinstance(raw_data, dict) else {}
    # 1.x name=APIError / 2.x type=provider.quota -- same slot.
    name = error.get("name") or error.get("type") or ""
    # 2.x emits the retry-after the binary itself honours (the structured
    # error's retryAfterMs; the captured 2.3.0 event carries it at the EVENT
    # top level) so ``go_quota_reset`` can cool down the exact window without
    # probing.
    retry_after_ms = event.get(
        "retryAfterMs", error.get("retryAfterMs", data.get("retryAfterMs", ""))
    )
    response = error.get("response") if isinstance(error.get("response"), dict) else {}
    body = data.get("responseBody", response.get("body", ""))
    parts = (
        f"name={name}",
        f"status={data.get('statusCode', error.get('status', ''))}",
        f"retryable={data.get('isRetryable', error.get('retryable', ''))}",
        f"retry-after-ms={retry_after_ms}",
        f"message={str(data.get('message', error.get('message', '')))[:_FIELD_CAP]}",
        f"body={str(body)[:_FIELD_CAP]}",
    )
    return "opencode error: " + " ".join(parts)


def _error_event_summary(line: str) -> str:
    try:
        event = json.loads(line)
    except ValueError:
        return line[:_FIELD_CAP]
    if isinstance(event, dict) and event.get("type") == "retry":
        # A 2.x retry event: the error lives at the event's top level (the
        # terminal {"type":"error"} event wraps it under "error" too, so the
        # same extraction covers both).
        return _error_summary_from_event(event)
    error = event.get("error") if isinstance(event, dict) else None
    if not isinstance(error, dict):
        return line[:_FIELD_CAP]
    summary = _error_summary_from_event(event)
    return summary or line[:_FIELD_CAP]


def last_text_event(log_text: str) -> str | None:
    """The final assistant ``text`` part of an opencode JSONL stream, else None.

    ``opencode run --format json`` emits ``{"type":"text","part":{"text":...}}``
    events for assistant output; a reviewer's verdict fence lives inside the
    last one's ``part.text``, JSON-escaped, so the plaintext fence scan can
    never see it. Only the LAST text event is returned -- a mid-session draft
    or an echo of the prompt's own example verdict must not count (the same
    rule ``_extract_verdict_from_stream_json`` applies to claude events).

    A ``step_start``, ``tool_use`` or ``error`` event after a text resets the
    candidate: that text was mid-session output, not the final answer, so a
    draft verdict it carried must not be reaped when the session dies before
    producing its real one. A 2.x ``retry`` event resets too -- it carries the
    failed provider call that the text preceded (the 2.x counterpart of the
    print-log ERROR line: opencode emits it BEFORE sleeping out retry-after,
    and the process may be killed inside that sleep). A ``--print-logs``
    ``level=ERROR`` line resets as well:
    it is written when the provider call fails inside ``h.stream`` before the
    next ``step_start``/``error`` event exists, so a text before it is likewise
    mid-session output from an earlier turn. ``step_finish`` does not reset
    -- text followed by step_finish is a completed turn, exactly the shape a
    real verdict ends on -- UNLESS the finish reason says the turn continued
    (``"tool-calls"``, ``"length"``, ...): that text preceded a tool call the
    process died inside, so it was a draft, not the answer.
    """
    last: str | None = None
    for line in log_text.splitlines():
        event = _EVENT_TYPE.match(line)
        if event is None:
            if _PRINT_LOG_ERROR.match(line) is not None:
                last = None
            continue
        kind = event.group(1)
        if kind == "text":
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            part = payload.get("part") if isinstance(payload, dict) else None
            if (
                isinstance(part, dict)
                and isinstance(part.get("text"), str)
                and part["text"].strip()
            ):
                # Empty/whitespace parts are SessionProcessor.cleanup
                # artifacts (text-start opened before the stream aborted),
                # not content -- they neither qualify nor displace `last`.
                last = part["text"]
        elif kind in ("step_start", "tool_use", "error", "retry"):
            last = None
        elif kind == "step_finish":
            try:
                payload = json.loads(line)
            except ValueError:
                # A truncated step_finish cannot certify the text before it
                # as a completed turn -- its reason might say the turn
                # continued (a draft, per above).
                last = None
                continue
            part = payload.get("part") if isinstance(payload, dict) else None
            reason = part.get("reason") if isinstance(part, dict) else None
            if reason not in (None, "", "stop", "end-turn"):
                last = None
    return last


def event_text(event: dict[str, Any]) -> str:
    """Assistant text of one parsed opencode event (``""`` when none)."""
    if event.get("type") != "text":
        return ""
    part = event.get("part")
    if isinstance(part, dict) and isinstance(part.get("text"), str):
        return part["text"]
    return ""


@dataclass(frozen=True)
class OpencodeProgress:
    """Per-session metrics recovered from an opencode ``--format json`` log.

    The opencode counterpart of ``claude_code.ClaudeProgress``: ``step_start``
    events open attempts, ``tool_use`` events count tool calls, ``text``
    events carry assistant analysis and ``error`` events carry the session's
    own failure output (summarized via ``_error_event_summary``). A 2.x
    ``retry`` event marks a failed provider attempt (the 2.x counterpart of a
    ``--print-logs`` ERROR line): it is not progress, discounts a still-open
    output-less step, and its structured error lands in ``texts`` like an
    error event's does. A
    ``step_start`` whose step produced no NON-BLANK ``text`` and no
    ``tool_use`` before the next ``step_start``/``error``/``retry``/print-log
    ``level=ERROR``/EOF does NOT count as a turn: the attempt died inside
    its provider call (a quota wall at launch, an internal retry, or a
    watchdog kill), so counting it would fake completed review work that
    never happened. An empty or whitespace-only ``part.text`` -- the
    ``SessionProcessor.cleanup`` flush emitted after a stream aborts -- is
    not output either.
    """

    turn_count: int
    tool_call_count: int
    texts: tuple[str, ...]


def session_progress(log_text: str) -> OpencodeProgress | None:
    """Turn/tool-call counts and assistant texts of an opencode JSONL log.

    Returns None when no opencode event is recognized -- a claude stream-json
    or plaintext log -- so callers keep their existing fallback path rather
    than misattributing the format. ``_extract_review_session_summary`` uses
    this so a worked opencode reviewer is classified on real turn/tool
    counts (not launch_failed) and its decoded text lands in the PR comment
    instead of raw JSONL.
    """
    texts: list[str] = []
    turns = tools = 0
    seen = False
    step_has_output = False
    for line in log_text.splitlines():
        event = _EVENT_TYPE.match(line)
        if event is None:
            if _PRINT_LOG_ERROR.match(line) is not None:
                seen = True
                if turns and not step_has_output:
                    turns -= 1
                step_has_output = True  # the failure was already discounted
                # The stream error lands in the summary like an error
                # event's does: a death after assistant text must still
                # surface the real provider message in the PR comment.
                texts.append(provider_error_digest(line))
            continue
        kind = event.group(1)
        if kind == "retry":
            # 2.x failed provider attempt: same treatment as the 1.x
            # print-log ERROR lane for turn discounting, and the same
            # treatment as the 1.x error event for the text record -- the
            # structured error lands in the summary so the PR comment
            # carries the real provider message, not just the draft text
            # before it.
            seen = True
            if turns and not step_has_output:
                turns -= 1
            step_has_output = True
            texts.append(_error_event_summary(line))
        elif kind == "step_start":
            seen = True
            if turns and not step_has_output:
                turns -= 1  # previous attempt produced nothing
            turns += 1
            step_has_output = False
        elif kind == "tool_use":
            seen = True
            tools += 1
            step_has_output = True
        elif kind == "text":
            seen = True
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            part = payload.get("part") if isinstance(payload, dict) else None
            if (
                isinstance(part, dict)
                and isinstance(part.get("text"), str)
                and part["text"].strip()
            ):
                # Only a non-blank text part is step output: an empty
                # part.text (cleanup flush after an abort) or a truncated
                # line would otherwise fake a completed turn and hide the
                # digest fallback below.
                step_has_output = True
                texts.append(part["text"])
        elif kind == "error":
            seen = True
            # A step that emitted no assistant text and no tool call before
            # dying on a provider error never completed a review turn: the
            # canonical shape is ``step_start`` + ``error`` for a quota wall
            # at turn 1. Discounting it keeps the 0/0 launch_failed path in
            # ``_extract_review_session_summary`` reachable, so the death is
            # classified environmental rather than died_mid_session (which
            # would defeat the #583 throttle-rollback guard).
            if turns and not step_has_output:
                turns -= 1
            step_has_output = True  # later errors must not decrement again
            texts.append(_error_event_summary(line))
    if turns and not step_has_output:
        turns -= 1  # trailing bare step_start: killed inside the provider call
    if seen and not texts:
        # --print-logs stream errors never emit {"type":"error"}; surface the
        # real provider error (the digest already keeps only errors since the
        # last progress event) rather than letting the caller substitute its
        # "produced no assistant text" placeholder over the actual cause.
        digest = provider_error_digest(log_text)
        if digest:
            texts.append(digest)
    return OpencodeProgress(turns, tools, tuple(texts)) if seen else None


def provider_error_digest(log_text: str) -> str:
    """opencode's own provider-error records since the last progress event."""
    collected: list[str] = []
    for line in log_text.splitlines():
        event = _EVENT_TYPE.match(line)
        if event is not None:
            kind = event.group(1)
            if kind in ("error", "retry"):
                # A 2.x retry event carries the SAME structured provider
                # error as the terminal error event (session.retry.scheduled
                # -> JSON "retry") and may be the ONLY record when the
                # process is killed inside the retry sleep -- ignoring it
                # left 2.x quota/throttle deaths unclassifiable.
                collected.append(_error_event_summary(line))
            elif kind == "step_start":
                pass  # a retry begins; earlier failures still count if it dies silent
            elif kind == "text":
                try:
                    payload = json.loads(line)
                except ValueError:
                    continue
                part = payload.get("part") if isinstance(payload, dict) else None
                text = part.get("text") if isinstance(part, dict) else None
                if isinstance(text, str) and text.strip():
                    collected.clear()
            elif kind == "tool_use":
                collected.clear()
            # step_finish and any unrecognized future kind are ignored, not
            # treated as progress: step_finish terminates the step whose
            # output already reset the digest, and after an error it is the
            # last event before the session dies -- clearing there would
            # erase the only provider-error evidence (and empty the
            # session_progress text fallback with it). A future opencode
            # event kind must not silently discard error records either;
            # keeping them can only over-report stale errors, which the
            # callers' turn/tool-count guards absorb, while clearing them
            # loses the classification signal outright.
            continue
        logged = _PRINT_LOG_ERROR.match(line)
        if logged is not None:
            ts = logged.group(1)
            header = line[: logged.start(2)]
            err = logged.group(2)[:_FIELD_CAP]
            if _ZEN_FREE_MODEL.search(header) and _ZEN_RATE_LIMIT.search(err):
                # Zen free-tier rate limit, not a Go usage limit: tag it with
                # FreeUsageLimitError (already a quota_error_markers member, so
                # it classifies quota_exhausted) plus ZenFreeUsageLimitError so
                # go_quota_reset picks the 1h cooldown over the Go probe.
                collected.append(
                    f"{ts} opencode stream error "
                    f"(FreeUsageLimitError {ZEN_FREE_LIMIT_MARKER}): {err}"
                )
            else:
                collected.append(f"{ts} opencode stream error: {err}")
    return "\n".join(collected)


def carries_work_output(log_text: str) -> bool:
    """True when the log holds assistant text or tool output.

    This is the precondition for matching throttle/quota markers against the
    RAW log tail: a reviewed file quoting "rate limit" in the last 2KB can
    only reach the tail through assistant text or tool output, so when
    neither exists the raw tail is safe to classify (and is the only signal
    left when a provider error arrives in a shape the digest does not
    recognize). A blank ``part.text`` is the ``SessionProcessor.cleanup``
    flush, not output.
    """
    for line in log_text.splitlines():
        event = _EVENT_TYPE.match(line)
        if event is None:
            continue
        kind = event.group(1)
        if kind == "tool_use":
            return True
        if kind != "text":
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        part = payload.get("part") if isinstance(payload, dict) else None
        text = part.get("text") if isinstance(part, dict) else None
        if isinstance(text, str) and text.strip():
            return True
    return False
