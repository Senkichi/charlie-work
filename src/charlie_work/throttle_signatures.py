"""Shared provider-throttle tail-matching helper.

``_classify_session_failure`` (``devin_shell.py`` + ``claude_code.py``) and
``get_rate_limit_defer_until`` (``devin_shell.py``, issue #247) each answer
the same question — "does this log tail contain a provider-throttle
signature, and if so, when does the cooldown end?" — against the same
config-driven marker list (``RuntimeConfig.throttle_error_markers``).

Before this module existed, each of the three call sites carried its own
copy of the substring-match-plus-"resets in N minutes" logic (PR #262
review findings F1/F5), and the copies drifted: one still referenced a
module-level ``_RATE_LIMIT_PATTERN`` regex that a sibling change had
deleted, a latent ``NameError``. This module is the single point of
enforcement — a new throttle signature only needs a config change
(``RuntimeConfig.throttle_error_markers``), never a source edit in three
places.

Issue #260 (corrected premise): "A tool was rejected by the user" is the
Devin CLI's own surfacing of a PreToolUse hook block, not a provider
throttle condition — it must never reach this matcher via the default
marker list. See ``post_mortem.py`` for the ``worker_blocked`` classifier
that owns that signature instead.

Issue #612: Claude Code's account-level session-limit notice names a
specific clock-time reset in an IANA zone, e.g. "resets 1:20am
(America/Los_Angeles)". ``match_throttle_tail`` only extracts the
"resets in N minutes" form, so this clock-time form was thrown away and
the reviewer-quota backoff fell back to a fixed ``quota_reset_hours``
window — backing off 5h when the limit resets in 30 min, or re-spending
into a still-closed window when it resets in 8h. ``parse_reset_clock_time``
is the single parser for the clock-time form, returning the next UTC
occurrence of the named reset time so callers can back off until the
provider's own stated reset instead of a fixed guess.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# "resets in N minutes" / "reset in N minutes" — extracts a specific
# provider-reported cooldown duration when present in the log tail.
_RESETS_IN_PATTERN = re.compile(r"resets? in (\d+) minutes?", re.IGNORECASE)

# "resets H:MMam/pm (IANA zone)" — Claude Code's session-limit notice names
# a specific clock time and the IANA timezone it resets in (observed
# verbatim 2026-07-21: "resets 4:40pm (America/Los_Angeles)"). Captures the
# hour, minute, am/pm, and zone name so ``parse_reset_clock_time`` can
# resolve the next occurrence of that clock time in that zone and convert to
# UTC. ``\s*`` tolerates the "4:40pm" and "4:40 pm" forms.
_RESETS_CLOCK_PATTERN = re.compile(
    r"resets?\s+(\d{1,2}):(\d{2})\s*(am|pm)\s*\(([^)]+)\)",
    re.IGNORECASE,
)


def match_throttle_tail(tail: str, markers: Sequence[str]) -> tuple[bool, int | None]:
    """Match ``tail`` against ``markers`` (case-insensitive substrings).

    Returns ``(matched, reset_minutes)``:
    - ``matched`` is True when any marker is found in ``tail``.
    - ``reset_minutes`` is the parsed "resets in N minutes" value when the
      tail matched and included one, else None (callers apply their own
      default cooldown in that case). Always None when ``matched`` is False.
    """
    tail_lower = tail.lower()
    matched = any(marker.lower() in tail_lower for marker in markers)
    if not matched:
        return False, None
    reset_match = _RESETS_IN_PATTERN.search(tail)
    reset_minutes = int(reset_match.group(1)) if reset_match else None
    return True, reset_minutes


def parse_reset_clock_time(tail: str, now: datetime) -> datetime | None:
    """Parse a "resets H:MMam/pm (IANA zone)" notice into the next UTC datetime.

    Claude Code's session-limit death message names the reset as a clock
    time in an IANA timezone (e.g. "resets 1:20am (America/Los_Angeles)"),
    not as "resets in N minutes". This parser resolves the next occurrence
    of that clock time in the named zone relative to ``now``: if today's
    occurrence has already passed (in that zone), the reset is tomorrow at
    that clock time. The result is a timezone-aware UTC datetime.

    Returns ``None`` when the tail carries no clock-time reset notice, or
    when the named zone cannot be loaded (e.g. ``tzdata`` not installed on
    a host without system tz data). Callers fall back to a fixed cooldown
    in that case rather than guessing the offset — a wrong offset would
    either re-spend into a still-closed window or stall far longer than the
    provider intends (issue #612).

    ``now`` must be timezone-aware (UTC). It is the reference for the
    "today vs tomorrow" decision; passing a fixed ``now`` in tests makes
    the result deterministic regardless of when the test runs.
    """
    match = _RESETS_CLOCK_PATTERN.search(tail)
    if match is None:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    meridiem = match.group(3).lower()
    zone_name = match.group(4).strip()

    # 12-hour → 24-hour: 12am → 0, 12pm → 12, else add 12 for pm.
    if meridiem == "am":
        if hour == 12:
            hour = 0
    else:  # pm
        if hour != 12:
            hour += 12

    try:
        zone = ZoneInfo(zone_name)
    except (ZoneInfoNotFoundError, KeyError, ValueError):
        # Unknown/unavailable zone — do not guess the offset (issue #612).
        return None

    now_in_zone = now.astimezone(zone)
    candidate = now_in_zone.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now_in_zone:
        candidate += timedelta(days=1)
    return candidate.astimezone(UTC)


__all__ = ["match_throttle_tail", "parse_reset_clock_time"]
