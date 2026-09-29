"""Devin-shell log-tail failure classification and throttle-window derivation.

``_classify_session_failure`` is now a thin wrapper around
``worker_fate.classify_failure`` (design doc §7, issue #1997 unification with
the claude_code sibling adapter's copy) — devin has no account-error
detection and already uses the emission-time throttle anchor (the anchor
itself, ``_throttle_emission_anchor``, now lives in ``worker_fate`` too).
``devin_shell`` re-exports the public names so existing callers and test
imports keep resolving through ``charlie_work.devin_shell``.

``get_rate_limit_defer_until`` stays here: it is not part of the
``classify_failure`` merge (claude_code has no equivalent function to unify
it with), but shares the emission-anchor helper, imported lazily from
``worker_fate`` to avoid a cycle.

Signature matching itself is NOT here — the substring/"resets in N minutes"
extraction lives in ``throttle_signatures`` (the single point of enforcement
shared with the claude-code sibling adapter); this module consumes it and
owns the devin-adapter-specific cooldown policy around it.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .config import OrchestratorConfig
from .throttle_signatures import match_throttle_tail

# Provider throttle signatures — matched against session log tails to classify
# failure kinds. The defaults are sourced from RuntimeConfig so there is a single
# default list; callers can override via config for new provider phrasings.
# Matching itself (substring + "resets in N minutes" extraction) is unified in
# throttle_signatures.match_throttle_tail — used by get_rate_limit_defer_until
# below (PR #262 review findings F1/F5).
_DEFAULT_THROTTLE_ERROR_MARKERS = OrchestratorConfig().runtime.throttle_error_markers

# Default cooldown duration when we can't parse a specific reset time
_DEFAULT_RATE_LIMIT_COOLDOWN_MINUTES = 15


def _classify_session_failure(
    log_path: Path,
    throttle_error_markers: Sequence[str] | None = None,
    *,
    quota_error_markers: Sequence[str] | None = None,
    resume_margin_seconds: int = 0,
    now: datetime | None = None,
) -> tuple[str | None, str | None]:
    """Classify a session failure from its log tail. Thin wrapper around
    ``worker_fate.classify_failure`` (design doc §7) — devin has no
    account-error detection (``account_error_detection=False``) and already
    uses the emission-time throttle anchor (``legacy_classification_
    anchor=False``, issue #1997), so no FLIP marker applies here: this
    module's observable behaviour is unchanged by the merge.

    Returns (failure_kind, throttled_until_iso):
    - failure_kind: "rate_limited" | "quota_exhausted" | None
    - throttled_until_iso: ISO timestamp when the cooldown ends, or None
    """
    from .worker_fate import classify_failure

    return classify_failure(
        log_path,
        throttle_error_markers,
        quota_error_markers=quota_error_markers,
        resume_margin_seconds=resume_margin_seconds,
        account_error_detection=False,
        legacy_classification_anchor=False,
        now=now,
    )


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
    # rather than a negative one. Lazy import: the anchor helper now lives
    # in worker_fate (design doc §7), and a top-level import here would
    # cycle (worker_fate -> worker -> claude_code/devin_shell already
    # imports this module's sibling functions).
    from .worker_fate import _throttle_emission_anchor

    emitted_at = _throttle_emission_anchor(log_path, tail, now=now)
    defer_until = max(
        now,
        emitted_at + timedelta(minutes=minutes + slack_minutes, seconds=resume_margin_seconds),
    )
    return defer_until.replace(microsecond=0).isoformat().replace("+00:00", "Z")
