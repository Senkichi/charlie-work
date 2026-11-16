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

Only errors since the last sign of progress count: any other stdout event
(``step_start``, ``text``, ``tool_use`` ...) resets the digest, so a 429 that
opencode retried through successfully never classifies a later, unrelated death.
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
_PRINT_LOG_ERROR = re.compile(r'^timestamp=(\S+) level=ERROR .*?error\.error="((?:[^"\\]|\\.)*)"')
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


def _error_event_summary(line: str) -> str:
    try:
        event = json.loads(line)
    except ValueError:
        return line[:_FIELD_CAP]
    error = event.get("error") if isinstance(event, dict) else None
    if not isinstance(error, dict):
        return line[:_FIELD_CAP]
    raw_data = error.get("data")
    data: dict = raw_data if isinstance(raw_data, dict) else {}
    parts = (
        f"name={error.get('name', '')}",
        f"status={data.get('statusCode', '')}",
        f"retryable={data.get('isRetryable', '')}",
        f"message={str(data.get('message', ''))[:_FIELD_CAP]}",
        f"body={str(data.get('responseBody', ''))[:_FIELD_CAP]}",
    )
    return "opencode error: " + " ".join(parts)


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
    producing its real one. ``step_finish`` does not reset -- text followed by
    step_finish is a completed turn, exactly the shape a real verdict ends on.
    """
    last: str | None = None
    for line in log_text.splitlines():
        event = _EVENT_TYPE.match(line)
        if event is None:
            continue
        kind = event.group(1)
        if kind == "text":
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            part = payload.get("part") if isinstance(payload, dict) else None
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                last = part["text"]
        elif kind in ("step_start", "tool_use", "error"):
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
    events count turns, ``tool_use`` events count tool calls, ``text`` events
    carry assistant analysis and ``error`` events carry the session's own
    failure output (summarized via ``_error_event_summary``).
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
    for line in log_text.splitlines():
        event = _EVENT_TYPE.match(line)
        if event is None:
            continue
        kind = event.group(1)
        if kind == "step_start":
            seen = True
            turns += 1
        elif kind == "tool_use":
            seen = True
            tools += 1
        elif kind == "text":
            seen = True
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            part = payload.get("part") if isinstance(payload, dict) else None
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
        elif kind == "error":
            seen = True
            texts.append(_error_event_summary(line))
    return OpencodeProgress(turns, tools, tuple(texts)) if seen else None


def provider_error_digest(log_text: str) -> str:
    """opencode's own provider-error records since the last progress event."""
    collected: list[str] = []
    for line in log_text.splitlines():
        event = _EVENT_TYPE.match(line)
        if event is not None:
            if event.group(1) == "error":
                collected.append(_error_event_summary(line))
            else:
                collected.clear()
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
