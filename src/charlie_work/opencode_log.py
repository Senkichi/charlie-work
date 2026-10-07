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
from pathlib import Path

_EVENT_TYPE = re.compile(r'^\{"type":"([A-Za-z_]+)"')
_PRINT_LOG_ERROR = re.compile(r'^timestamp=\S+ level=ERROR .*?error\.error="((?:[^"\\]|\\.)*)"')
_FIELD_CAP = 400


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
            collected.append(f"opencode stream error: {logged.group(1)[:_FIELD_CAP]}")
    return "\n".join(collected)
