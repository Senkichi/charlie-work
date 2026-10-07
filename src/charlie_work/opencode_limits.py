"""Reset time of an OpenCode Go usage limit (``AdapterFateProfile.quota_reset``).

OpenCode Go meters three windows -- a rolling 5-hour, a weekly and a monthly
budget -- and a ``GoUsageLimitError`` 429 says which one tripped
(``metadata.limitName``) and when it resets (the ``retry-after`` header, in
seconds). opencode itself never writes either to a worker's log: it sleeps out
``retry-after`` in memory and ``--print-logs`` shows only "Usage limit reached".
So a quota death would otherwise get the classifier's fixed 24h cooldown --
too long for the 5-hour window, too short for the weekly or monthly one.

``go_quota_reset`` recovers the real reset with one probe of the Go API, made
only when a death has already classified ``quota_exhausted``:

- a ``GoUsageLimitError`` 429 -> ``retry-after`` (else the ``limitName``
  window length); the request is refused, so it costs nothing;
- a 200 -> the limit has already reset; a short cooldown (the request is
  capped at one output token);
- anything else (no credential, network error, a different error) -> None, and
  the caller keeps its 24h default.

The credential is read from the operator's opencode ``auth.json`` and is only
ever sent as the request's bearer token: it never reaches a log, an event or
an exception message.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

from .opencode_worker import _host_auth_path

log = logging.getLogger(__name__)

GO_PROVIDER = "opencode-go"
GO_API_BASE = "https://opencode.ai/zen/go/v1"
GO_LIMIT_ERROR = "GoUsageLimitError"

# A limit that has already reset when probed: re-dispatch soon, but not into
# the same second (the death may have raced the reset).
CLEARED_COOLDOWN = timedelta(minutes=15)
# Repeated classifications of one death (dead reap, sweep, reconcile) within a
# pass reuse the first probe instead of re-asking the API.
_CACHE_SECONDS = 120.0
_TIMEOUT_SECONDS = 10.0

# opencode's humanized message ("It will reset in 2 hours 5 minutes"), should a
# future opencode release print it to the log.
_RESET_IN = re.compile(
    r"reset in ((?:\d+\s*(?:days?|hours?|minutes?|seconds?)[\s,]*(?:and\s+)?)+)"
)
_RESET_PART = re.compile(r"(\d+)\s*(day|hour|minute|second)")
_LIMIT_HOURS = re.compile(r"(\d+)[- ]?hour", re.IGNORECASE)
_LIMIT_WINDOWS = {"day": timedelta(days=1), "week": timedelta(days=7), "month": timedelta(days=30)}

# The one network seam: tests replace it (conftest refuses real connections).
_open = urllib.request.urlopen

_cache_lock = threading.Lock()
_cache: tuple[float, timedelta | None] | None = None


def reset_from_text(text: str) -> timedelta | None:
    """The last "reset in <n days/hours/minutes>" duration in ``text``, or None."""
    matches = list(_RESET_IN.finditer(text))
    if not matches:
        return None
    total = timedelta()
    for amount, unit in _RESET_PART.findall(matches[-1].group(1)):
        total += timedelta(**{f"{unit}s": int(amount)})
    return total or None


def window_for_limit_name(limit_name: str) -> timedelta | None:
    """Window length for a ``limitName`` ("5-hour", "weekly", "monthly"), or None."""
    hours = _LIMIT_HOURS.search(limit_name)
    if hours:
        return timedelta(hours=int(hours.group(1)))
    lowered = limit_name.lower()
    for word, window in _LIMIT_WINDOWS.items():
        if word in lowered:
            return window
    return None


def _retry_after(headers: Any) -> timedelta | None:
    """``retry-after-ms`` / ``retry-after`` (seconds or HTTP-date) as a duration."""
    raw_ms = headers.get("retry-after-ms")
    if raw_ms:
        try:
            return timedelta(milliseconds=max(0.0, float(raw_ms)))
        except ValueError:
            pass
    raw = headers.get("retry-after")
    if not raw:
        return None
    try:
        return timedelta(seconds=max(0.0, float(raw)))
    except ValueError:
        pass
    try:
        moment = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return max(timedelta(), moment - datetime.now(UTC))


def reset_from_429(headers: Any, body: str) -> timedelta | None:
    """Reset duration of a ``GoUsageLimitError`` 429, or None for any other 429."""
    if GO_LIMIT_ERROR not in body:
        return None
    retry = _retry_after(headers)
    if retry is not None:
        return retry
    try:
        payload = json.loads(body)
    except ValueError:
        return None
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    limit_name = metadata.get("limitName") if isinstance(metadata, dict) else None
    return window_for_limit_name(limit_name) if isinstance(limit_name, str) else None


def _go_api_key() -> str | None:
    try:
        entries = json.loads(_host_auth_path(os.environ).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    entry = entries.get(GO_PROVIDER) if isinstance(entries, dict) else None
    key = entry.get("key") if isinstance(entry, dict) else None
    return key if isinstance(key, str) and key else None


def _request(
    key: str, method: str, path: str, body: dict[str, Any] | None
) -> tuple[int, Any, str]:
    headers = {
        "Authorization": f"Bearer {key}",
        "content-type": "application/json",
        "user-agent": "charlie-work-quota-probe",
        # The Go API refuses a request without a session id (400 MissingSessionID).
        "x-opencode-session": f"ses_cw_probe_{uuid.uuid4().hex[:16]}",
    }
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        f"{GO_API_BASE}{path}", data=data, headers=headers, method=method
    )
    try:
        with _open(request, timeout=_TIMEOUT_SECONDS) as response:
            return response.status, response.headers, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read().decode("utf-8", "replace")


def _probe_model(key: str) -> str | None:
    """Any model the subscription serves: the limits are per workspace, not per model."""
    status, _, body = _request(key, "GET", "/models", None)
    if status != 200:
        return None
    payload = json.loads(body)
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                return entry["id"]
    return None


def probe_go_reset() -> timedelta | None:
    """Ask the Go API how long until the tripped usage limit resets. Never raises."""
    key = _go_api_key()
    if key is None:
        return None
    try:
        model = _probe_model(key)
        if model is None:
            return None
        status, headers, body = _request(
            key,
            "POST",
            "/chat/completions",
            {"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": "."}]},
        )
    except (OSError, ValueError) as exc:
        # The exception names the URL at most, never the request headers.
        log.warning("opencode Go quota probe failed: %s", type(exc).__name__)
        return None
    if status == 200:
        return CLEARED_COOLDOWN
    if status == 429:
        return reset_from_429(headers, body)
    log.warning("opencode Go quota probe: unexpected HTTP %s", status)
    return None


def go_quota_reset(tail: str) -> timedelta | None:
    """``AdapterFateProfile.quota_reset`` for opencode: the reset the log states, else a probe."""
    stated = reset_from_text(tail)
    if stated is not None:
        return stated
    if "FreeUsageLimitError" in tail:
        # opencode Zen's free tier, not the Go subscription: a Go probe would
        # answer for the wrong account.
        return None
    global _cache
    with _cache_lock:
        now = time.monotonic()
        if _cache is not None and now - _cache[0] < _CACHE_SECONDS:
            return _cache[1]
        reset = probe_go_reset()
        _cache = (now, reset)
        return reset
