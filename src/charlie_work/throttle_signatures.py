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
"""

from __future__ import annotations

import re
from collections.abc import Sequence

# "resets in N minutes" / "reset in N minutes" — extracts a specific
# provider-reported cooldown duration when present in the log tail.
_RESETS_IN_PATTERN = re.compile(r"resets? in (\d+) minutes?", re.IGNORECASE)


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


__all__ = ["match_throttle_tail"]
