"""Post-exit failure classification: match a dead worker's log tail against
provider throttle/quota/auth/suspension signatures (``classify_failure``).

Split out of ``worker_fate.py`` (architecture-deepening candidate 1; file-size
ratchet). Pure over its inputs -- a log path, marker lists and an injectable
clock -- and independent of the fate resolver's evidence types, so it has no
import of ``worker_fate``. ``worker_fate`` re-exports the public names; the
per-harness flag lookup lives in ``adapter_fate_profile.classify_for``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .config import OrchestratorConfig
from .throttle_signatures import (
    PERMISSION_DENIED_FAILURE_KIND,
    is_headless_permission_denial,
    is_provider_auth_failure,
    match_quota_tail,
    match_throttle_tail,
)

# --------------------------------------------------------------------------
# Failure classification (§6) and the internal Adapter seam (§7).
#
# ``classify_failure`` merged the two per-adapter ``_classify_session_failure``
# copies (both now deleted; ``classify_for`` is the entry point). The two were
# identical except for two things, both now data instead of two copies of
# the function:
#   - account-error detection (``provider_suspended``/``provider_auth``,
#     api only) was ``adapter_kind == "api"``; now ``account_error_detection``.
#   - the ``rate_limited`` cooldown anchor: devin already anchored at the
#     message's *emission* time (issue #1997 -- classification can run tens
#     of minutes after the log line was written, so anchoring "now" plus a
#     15-minute cooldown overshoots the real provider reset by that much);
#     claude-code/api used to anchor at *classification* time instead.
#     Rule 6 (wf-design.md §9) ports #1997 to claude-code/api fleet-wide:
#     every harness now anchors at emission time, unconditionally.
# --------------------------------------------------------------------------

_CLASSIFY_DEFAULT_THROTTLE_ERROR_MARKERS = OrchestratorConfig().runtime.throttle_error_markers
_CLASSIFY_DEFAULT_QUOTA_ERROR_MARKERS = OrchestratorConfig().runtime.quota_error_markers
_DEFAULT_RATE_LIMIT_COOLDOWN_MINUTES = 15
_DEFAULT_QUOTA_COOLDOWN_HOURS = 24

# Provider authentication failures (issue #484). Matched against the log tail
# of account-error-detecting (api) sessions only.
#
# N3 (wf-review-opus.md, wf-8-review-fixes): this used to be a byte-duplicate
# of claude_code.py's own compiled copy ("moved verbatim from claude_code.py"
# -- the two could drift silently). Both now call
# ``throttle_signatures.is_provider_auth_failure`` (single point of
# enforcement) -- see its docstring for the word-boundary 401/403
# false-positive rationale.
#
# Provider account suspension / insufficient-balance responses (issue #1342).
# Moved verbatim from claude_code.py. The billing phrase alone is not
# enough -- ``_provider_suspension_in_tail`` requires it to co-occur on the
# same log line as a structural API-error signal (HTTP 402 or a CLI
# ``Error:``/``API Error:`` prefix), or a worker merely quoting/reviewing the
# trigger phrase would misclassify.
_PROVIDER_SUSPENDED_PHRASE = re.compile(
    r"insufficient\s+(?:balance|funds|credit)"
    r"|account\s+(?:is\s+)?suspended"
    r"|suspended\s+due\s+to\s+(?:insufficient\s+balance|billing|payment|unpaid)"
    r"|recharge\s+your\s+account"
    r"|please\s+recharge",
    re.IGNORECASE,
)
_PROVIDER_SUSPENDED_ANCHOR = re.compile(
    r"^\s*(?:api\s+)?error\s*:|\b402\b",
    re.IGNORECASE,
)


def _provider_suspension_in_tail(tail: str) -> bool:
    """True if ``tail`` has a structurally-anchored account-suspension
    signature: the billing phrase and the API-error anchor on the SAME line.
    """
    for line in tail.splitlines():
        if _PROVIDER_SUSPENDED_PHRASE.search(line) and _PROVIDER_SUSPENDED_ANCHOR.search(line):
            return True
    return False


# Issue #1997: a tz-aware ISO-8601 timestamp on a tail line marks when the
# provider emitted the throttle message -- a better anchor than the
# classification-time clock. Moved verbatim from devin_failure_classification.py.
_TAIL_LINE_TS_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})"
)


def _tail_emission_timestamp(tail: str) -> datetime | None:
    """The tz-aware timestamp on the tail's last timestamped PLAIN-TEXT line,
    scanning backwards so the most recent one wins. None when no line has one
    -- naive (offset-less) timestamps are skipped since their zone is unknown.

    JSON lines (whole objects, or a ``{...`` fragment cut by the tail's byte
    slice) are skipped: a timestamp embedded in an event or quoted inside a
    ``tool_result`` is not when the provider emitted anything.
    """
    for line in reversed(tail.splitlines()):
        if line.lstrip().startswith("{"):
            continue
        match = _TAIL_LINE_TS_PATTERN.search(line)
        if match is None:
            continue
        try:
            return datetime.fromisoformat(match.group(0))
        except ValueError:
            continue
    return None


def _parse_event(line: str) -> dict | None:
    """The stream-json event on ``line``: a JSON object with a string ``"type"``.
    None for a partial line, prose, stderr, or any other JSON."""
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        event = json.loads(line)
    except ValueError:
        return None
    if isinstance(event, dict) and isinstance(event.get("type"), str):
        return event
    return None


def _is_stream_json(tail: str) -> bool:
    """True when the log's shape is stream-json: ANY whole tail line parses as
    a stream-json event. The shape is read off the log itself, not the
    harness: ``ClaudeCodeConfig.tee_stream_json`` defaults to False, so a
    claude-code ``.log`` is plain CLI prose unless the tee is on, while api
    and the reviewers always write stream-json. The LAST line must not decide:
    ``claude_code`` merges stderr into the same ``.log`` (``stderr=STDOUT``),
    so a stream-json log routinely ends on a non-JSON line (a SessionEnd hook
    failure). Only the tail's first line can be a partial event.
    """
    return any(_parse_event(line) is not None for line in tail.splitlines())


def _event_timestamp(event: dict) -> datetime | None:
    """The event's own tz-aware top-level ``"timestamp"``, else None. Naive
    (offset-less) and non-string values are skipped."""
    raw = event.get("timestamp")
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _stream_json_emission_anchor(
    log_path: Path, tail: str, throttle_markers: Sequence[str]
) -> datetime | None:
    """When the provider emitted the throttle, read off a stream-json ``tail``.

    Real CLI output carries a top-level ``"timestamp"`` only on ``assistant``
    and ``user`` events; the terminal ``result`` event (and ``system`` /
    ``rate_limit_event``) has none. Non-JSON lines (merged stderr) are skipped
    throughout. So:

    1. Find the newest event whose own text carries a throttle marker -- the
       event that IS the emission. Its top-level ``"timestamp"`` is trusted
       when present. An earlier assistant/user turn is never used: the CLI
       retries a 429 with backoff for minutes before dying, so that turn
       predates the emission by an unbounded gap.
    2. Without a timestamp: if that event is the terminal ``result`` and the
       last JSON event of the tail, the CLI wrote it last, so the log's mtime
       is the emission time (stderr lines after it are seconds of hook noise).
    3. Otherwise None (the caller anchors at ``now``). In particular a
       timestamp quoted inside a later ``tool_result`` is never taken.
    """
    events = [
        (line, event) for line in tail.splitlines() if (event := _parse_event(line)) is not None
    ]
    for index in range(len(events) - 1, -1, -1):
        line, event = events[index]
        matched, _ = match_throttle_tail(line, throttle_markers)
        if not matched:
            continue
        stamped = _event_timestamp(event)
        if stamped is not None:
            return stamped
        if event.get("type") == "result" and index == len(events) - 1:
            try:
                return datetime.fromtimestamp(log_path.stat().st_mtime, tz=UTC)
            except OSError:
                return None
        return None
    return None


def _throttle_emission_anchor(
    log_path: Path,
    tail: str,
    *,
    now: datetime,
    throttle_markers: Sequence[str],
) -> datetime:
    """Anchor for a provider-throttle window: when the message was emitted,
    not when it was classified. Clamped to ``now``: a future mtime/tail
    timestamp is clock/mtime skew, not evidence the reset also moved.

    The log's shape picks the reader (see ``_is_stream_json``):

    - plain text (devin; claude-code without the stream-json tee): the tail's
      last timestamped line, else the log's mtime (the last write to a dead
      worker's log is the death message), else ``now``.
    - stream-json (api, reviewers, claude-code with the tee): see
      ``_stream_json_emission_anchor``, else ``now``. Never a bare mtime --
      for a stream-json log it says nothing about which event was the
      throttle and would re-open the embedded-timestamp hole.
    """
    if _is_stream_json(tail):
        anchor = _stream_json_emission_anchor(log_path, tail, throttle_markers)
        return min(anchor if anchor is not None else now, now)
    anchor = _tail_emission_timestamp(tail)
    if anchor is None:
        try:
            anchor = datetime.fromtimestamp(log_path.stat().st_mtime, tz=UTC)
        except OSError:
            anchor = now
    return min(anchor, now)


def classify_failure(
    log_path: Path,
    throttle_error_markers: Sequence[str] | None = None,
    *,
    quota_error_markers: Sequence[str] | None = None,
    resume_margin_seconds: int = 0,
    account_error_detection: bool = False,
    headless_permission_detection: bool = False,
    now: datetime | None = None,
) -> tuple[str | None, str | None]:
    """Classify a session failure by matching the log tail against provider
    throttle/auth/suspension signatures. Called after a session exits.

    Returns (failure_kind, throttled_until_iso):
    - failure_kind: "provider_suspended" | "provider_auth" | "rate_limited" |
      "quota_exhausted" | "permission_denied" | None
    - throttled_until_iso: ISO timestamp the cooldown ends, or None (always
      None for "provider_suspended" -- terminal, no cooldown)

    ``account_error_detection`` (api only) enables the ``provider_suspended``
    (#1342) and ``provider_auth`` (#484) checks, both checked before
    quota/throttle so neither masquerades as a transient issue.

    ``headless_permission_detection`` (claude-code, issue #2010) enables the
    ``permission_denied`` check. It is checked LAST so throttle/auth
    signatures still win.

    The emission time is read out of the log according to the log's own
    shape (N9 -- see ``_throttle_emission_anchor``), not the harness.
    Callers normally reach this through ``classify_for``, which reads the two
    detection flags off the adapter profile.

    The ``rate_limited`` anchor is always the message's emission time
    (issue #1997 -- see ``_throttle_emission_anchor``), for every harness.
    Rule 6 (design doc, FLIP 6): claude-code/api used to anchor at
    classification time instead; devin already used the emission anchor, so
    this is the one fleet-wide throttle-timing change in the nine-rule
    resolution. ``quota_exhausted`` always anchors at classification time --
    its fixed 24h cooldown has no provider-stated reset to anchor against,
    so rule 6 does not affect it.

    ``now`` is the injectable clock: defaults to ``datetime.now(UTC)`` when
    not supplied, so production behaviour is byte-identical (issue #822).
    """
    if not log_path.exists():
        return None, None

    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None

    resolved_now = now if now is not None else datetime.now(UTC)
    # Check the last 2KB of the log (where error messages appear).
    tail = log_text[-2048:] if len(log_text) > 2048 else log_text

    if account_error_detection and _provider_suspension_in_tail(tail):
        return "provider_suspended", None

    if account_error_detection and is_provider_auth_failure(tail):
        cooldown = timedelta(hours=_DEFAULT_QUOTA_COOLDOWN_HOURS, seconds=resume_margin_seconds)
        throttled_until = resolved_now + cooldown
        return "provider_auth", throttled_until.replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        )

    quota_markers = (
        quota_error_markers
        if quota_error_markers is not None
        else _CLASSIFY_DEFAULT_QUOTA_ERROR_MARKERS
    )
    if match_quota_tail(tail, quota_markers):
        cooldown = timedelta(hours=_DEFAULT_QUOTA_COOLDOWN_HOURS, seconds=resume_margin_seconds)
        throttled_until = resolved_now + cooldown
        return "quota_exhausted", throttled_until.replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        )

    markers = (
        throttle_error_markers
        if throttle_error_markers is not None
        else _CLASSIFY_DEFAULT_THROTTLE_ERROR_MARKERS
    )
    matched, reset_minutes = match_throttle_tail(tail, markers)
    if matched:
        cooldown = timedelta(
            minutes=reset_minutes
            if reset_minutes is not None
            else _DEFAULT_RATE_LIMIT_COOLDOWN_MINUTES,
            seconds=resume_margin_seconds,
        )
        # The byte slice can cut the first line mid-event: its embedded
        # timestamps are not emission times, so the anchor never reads it.
        whole_lines = tail.partition("\n")[2] if len(log_text) > len(tail) else tail
        emitted_at = _throttle_emission_anchor(
            log_path, whole_lines, now=resolved_now, throttle_markers=markers
        )
        throttled_until = max(resolved_now, emitted_at + cooldown)
        return "rate_limited", throttled_until.replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        )

    # Issue #2010: last, so throttle/auth signatures still win.
    if headless_permission_detection and is_headless_permission_denial(tail):
        return PERMISSION_DENIED_FAILURE_KIND, None

    return None, None
