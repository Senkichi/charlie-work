"""Devin-shell log-tail failure classification and throttle-window derivation.

Extracted from ``devin_shell.py`` for the issue #1997 emission-time anchoring
work: ``devin_shell`` sits at its file-size ratchet mark, and this block —
``_classify_session_failure``, ``get_rate_limit_defer_until``, and the
emission-anchor helpers they share — is the cohesive "what does the session
log tail say about how this worker died, and when does its provider cooldown
end" surface. ``devin_shell`` re-exports the two public names so existing
callers and test imports keep resolving through ``charlie_work.devin_shell``.

Signature matching itself is NOT here — the substring/"resets in N minutes"
extraction lives in ``throttle_signatures`` (the single point of enforcement
shared with the claude-code sibling adapter); this module consumes it and
owns the devin-adapter-specific cooldown policy around it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .config import OrchestratorConfig
from .throttle_signatures import match_quota_tail, match_throttle_tail

# Provider throttle signatures — matched against session log tails to classify
# failure kinds. The defaults are sourced from RuntimeConfig so there is a single
# default list; callers can override via config for new provider phrasings.
# Matching itself (substring + "resets in N minutes" extraction) is unified in
# throttle_signatures.match_throttle_tail — used here and by
# get_rate_limit_defer_until below (PR #262 review findings F1/F5).
_DEFAULT_THROTTLE_ERROR_MARKERS = OrchestratorConfig().runtime.throttle_error_markers
# Quota-exhaustion prose fallback markers — defaults sourced from
# RuntimeConfig so there is a single default list; the structured
# "cognition.ai/errorKind": "resource_exhausted" trailer and the
# period-agnostic prose match live in throttle_signatures.match_quota_tail
# (issue #1684), shared with the claude_code sibling adapter.
_DEFAULT_QUOTA_ERROR_MARKERS = OrchestratorConfig().runtime.quota_error_markers

# Default cooldown durations when we can't parse a specific reset time
_DEFAULT_RATE_LIMIT_COOLDOWN_MINUTES = 15
_DEFAULT_QUOTA_COOLDOWN_HOURS = 24

# Issue #1997: a tz-aware ISO-8601 timestamp on a tail line marks when the
# provider emitted the throttle message — a better anchor than the
# classification-time clock. The offset (``Z`` or ``±HH:MM``/``±HHMM``) is
# deliberately required: a naive timestamp's zone is ambiguous, and the file
# mtime is strictly better evidence than a guessed zone.
_TAIL_LINE_TS_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})"
)


def _tail_emission_timestamp(tail: str) -> datetime | None:
    """Return the tz-aware ISO-8601 timestamp on the tail's last timestamped line.

    Scans backwards so the most recent timestamped line wins. Returns None
    when no line carries a tz-aware timestamp — naive (offset-less)
    timestamps are skipped because their zone is unknowable (issue #1997).
    """
    for line in reversed(tail.splitlines()):
        match = _TAIL_LINE_TS_PATTERN.search(line)
        if match is None:
            continue
        try:
            return datetime.fromisoformat(match.group(0))
        except ValueError:
            continue
    return None


def _throttle_emission_anchor(log_path: Path, tail: str, *, now: datetime) -> datetime:
    """Anchor for a provider-throttle window: when the message was emitted.

    The provider's "Your limit will reset in N minutes" counts from emission,
    but classification can run tens of minutes later (the activity-staleness
    threshold), so anchoring at classification time overshoots the real reset
    by that latency (issue #1997). Anchors at the tail's timestamp when a
    line carries one, else the log's mtime — the last write to a dead
    worker's log is the death message — else ``now`` when neither is usable.

    The result is clamped to ``now``: a future mtime or tail timestamp is
    clock/mtime skew, not evidence the provider's reset also moved forward.
    """
    anchor = _tail_emission_timestamp(tail)
    if anchor is None:
        try:
            anchor = datetime.fromtimestamp(log_path.stat().st_mtime, tz=UTC)
        except OSError:
            anchor = now
    return min(anchor, now)


def _classify_session_failure(
    log_path: Path,
    throttle_error_markers: Sequence[str] | None = None,
    *,
    quota_error_markers: Sequence[str] | None = None,
    resume_margin_seconds: int = 0,
    now: datetime | None = None,
) -> tuple[str | None, str | None]:
    """Classify a session failure by matching the log tail against provider throttle signatures.

    Returns a tuple of (failure_kind, throttled_until_iso):
    - failure_kind: "rate_limited" | "quota_exhausted" | None
    - throttled_until_iso: ISO timestamp when the cooldown ends, or None if not applicable

    This is called after a session exits to detect provider throttling and set a cool-down window.

    ``quota_error_markers`` is the prose-fallback list for quota exhaustion
    (``RuntimeConfig.quota_error_markers``); the structured
    ``cognition.ai/errorKind`` trailer is always checked first, regardless
    of the marker list (issue #1684). Defaults to the config module's
    default list when not provided.

    ``resume_margin_seconds`` is an extra safety margin past the provider's
    reported reset (or fixed quota cooldown) time. Provider reset estimates are
    floors, not guarantees, and dispatching at T+0 races the actual reset
    (issue #499).

    The ``rate_limited`` cooldown window is anchored at the message's
    emission time (issue #1997): the tail's timestamp when a line carries
    one, else the log's mtime, else ``now`` — not at classification time,
    which trails emission by the activity-staleness threshold and would
    overshoot the real reset by that much. ``quota_exhausted`` keeps a
    classification-time anchor because its fixed 24h cooldown has no
    provider-stated reset to anchor against.

    ``now`` is the injectable clock (mirrors ``get_rate_limit_defer_until``
    below and ``post_mortem.classify_and_record``): defaults to
    ``datetime.now(UTC)`` when not supplied, so production behavior is
    byte-identical. Tests that need an exact (not wall-clock-tolerance)
    assertion on the returned ``throttled_until_iso`` should pass a frozen
    value instead of racing real time (issue #822).
    """
    if not log_path.exists():
        return None, None

    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None

    resolved_now = now if now is not None else datetime.now(UTC)

    # Check the last 2KB of the log (where error messages appear)
    tail = log_text[-2048:] if len(log_text) > 2048 else log_text

    # Check for quota exhaustion first (more severe). Single point of
    # enforcement (throttle_signatures.match_quota_tail) shared with the
    # claude_code sibling adapter — the structured "cognition.ai/errorKind":
    # "resource_exhausted" trailer is matched before the config-driven prose
    # markers so provider wording drift ("daily" -> "weekly", issue #1684)
    # cannot defeat the classification.
    quota_markers = (
        quota_error_markers if quota_error_markers is not None else _DEFAULT_QUOTA_ERROR_MARKERS
    )
    if match_quota_tail(tail, quota_markers):
        # Quota exhaustion uses a fixed 24-hour cooldown regardless of reset time
        cooldown = timedelta(hours=_DEFAULT_QUOTA_COOLDOWN_HOURS, seconds=resume_margin_seconds)
        throttled_until = resolved_now + cooldown
        return "quota_exhausted", throttled_until.replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        )

    # Check for rate limiting / provider throttling using configurable substrings.
    # Single point of enforcement (throttle_signatures.match_throttle_tail) shared
    # with get_rate_limit_defer_until below — see that function's docstring.
    markers = (
        throttle_error_markers
        if throttle_error_markers is not None
        else _DEFAULT_THROTTLE_ERROR_MARKERS
    )
    matched, reset_minutes = match_throttle_tail(tail, markers)
    if matched:
        cooldown = timedelta(
            minutes=reset_minutes
            if reset_minutes is not None
            else _DEFAULT_RATE_LIMIT_COOLDOWN_MINUTES,
            seconds=resume_margin_seconds,
        )
        # Issue #1997: the provider's reset countdown starts when the message
        # was emitted, not when it is classified — anchor at the log's
        # emission time so detection latency does not overshoot the real
        # reset. A window already in the past clamps to ``resolved_now``: the
        # stored deadline is then a zero-length window ending "now", which
        # ``is_throttled``/``_is_deferred`` already treat as expired — so the
        # fleet resumes immediately instead of idling past the real reset.
        emitted_at = _throttle_emission_anchor(log_path, tail, now=resolved_now)
        throttled_until = max(resolved_now, emitted_at + cooldown)
        return "rate_limited", throttled_until.replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        )

    return None, None


def get_rate_limit_defer_until(
    log_path: Path,
    slack_minutes: int,
    now: datetime | None = None,
    throttle_error_markers: Sequence[str] | None = None,
    resume_margin_seconds: int = 0,
) -> str | None:
    """Return a defer-until ISO timestamp for a log tail containing a rate-limit signature.

    Reads the same 2KB tail as ``_classify_session_failure`` and matches
    against the same config-driven markers via ``throttle_signatures.
    match_throttle_tail`` (issue #247; unified with ``_classify_session_
    failure`` per PR #262 review findings F1/F5 — previously each function
    carried its own copy of this matching logic). If the tail matches and a
    ``"resets in N minutes"`` value is found, the defer deadline is anchored
    at the message's emission time (issue #1997, shared with
    ``_classify_session_failure`` via ``_throttle_emission_anchor``):
    ``emitted_at + N minutes + slack + resume_margin_seconds``. Otherwise the
    fallback ``_DEFAULT_RATE_LIMIT_COOLDOWN_MINUTES`` is used. A computed
    deadline already in the past is clamped to ``now`` — a zero-length
    window that ``_is_deferred`` treats as already expired.

    ``throttle_error_markers`` defaults to ``RuntimeConfig``'s default list
    when not provided (backward compatible with pre-#260 callers).

    ``resume_margin_seconds`` is an extra safety margin past the provider's
    reported reset time. Provider reset estimates are floors, not guarantees,
    and dispatching at T+0 races the actual reset (issue #499).

    Returns None when the log is missing, unreadable, or does not contain a
    rate-limit signature. Quota exhaustion is intentionally not deferred here.
    """
    if now is None:
        now = datetime.now(UTC)

    if not log_path.exists():
        return None

    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    tail = log_text[-2048:] if len(log_text) > 2048 else log_text

    markers = (
        throttle_error_markers
        if throttle_error_markers is not None
        else _DEFAULT_THROTTLE_ERROR_MARKERS
    )
    matched, reset_minutes = match_throttle_tail(tail, markers)
    if not matched:
        return None

    minutes = reset_minutes if reset_minutes is not None else _DEFAULT_RATE_LIMIT_COOLDOWN_MINUTES

    # Issue #1997: same emission-time anchor as _classify_session_failure —
    # the provider's countdown started when the log line was written, not
    # now. max(now, ...) keeps a born-expired window a zero-length deferral
    # rather than a negative one.
    emitted_at = _throttle_emission_anchor(log_path, tail, now=now)
    defer_until = max(
        now,
        emitted_at + timedelta(minutes=minutes + slack_minutes, seconds=resume_margin_seconds),
    )
    return defer_until.replace(microsecond=0).isoformat().replace("+00:00", "Z")
