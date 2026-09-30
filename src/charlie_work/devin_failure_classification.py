"""Devin-shell failure-classification policy.

The failure classifier itself is ``worker_fate.classify_for`` (design doc §7;
the per-adapter ``_classify_session_failure`` copies are deleted). What lives
here is the devin-adapter-specific policy around it:

* ``get_rate_limit_defer_until`` — the rate-limit deferral window derivation.
  It is not part of the ``classify_failure`` merge, but shares its
  emission-anchor helper, imported lazily from ``worker_fate`` to avoid a
  cycle.
* ``update_session_record_with_failure_classification`` — the devin sidecar
  classification writer (moved here from ``devin_shell`` in the PR #2069
  file-size-ratchet rework). It resolves ``devin_shell._sidecar_path`` /
  ``devin_shell._write_json`` and ``worker_fate`` at call time: top-level
  imports of either would cycle (``devin_shell`` imports this module for its
  re-exports; ``worker_fate``'s profile table reaches back into
  ``devin_shell``), and attribute access keeps ``devin_shell.*`` monkeypatch
  points reaching this code.

``devin_shell`` re-exports both so existing callers keep resolving through
``charlie_work.devin_shell``.

Signature matching itself is NOT here — the substring/"resets in N minutes"
extraction lives in ``throttle_signatures`` (the single point of enforcement
shared with the claude-code sibling adapter); this module consumes it and
owns the devin-adapter-specific cooldown policy around it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .config import OrchestratorConfig
from .process_utils import terminal_record_proves_completion
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


def get_rate_limit_defer_until(
    log_path: Path,
    slack_minutes: int,
    now: datetime | None = None,
    throttle_error_markers: Sequence[str] | None = None,
    resume_margin_seconds: int = 0,
) -> str | None:
    """Return a defer-until ISO timestamp for a log tail containing a rate-limit signature.

    Reads the same 2KB tail as ``worker_fate.classify_for`` and matches
    against the same config-driven markers via ``throttle_signatures.
    match_throttle_tail`` (issue #247; unified with the classifier per PR
    #262 review findings F1/F5 — previously each function
    carried its own copy of this matching logic). If the tail matches and a
    ``"resets in N minutes"`` value is found, the defer deadline is anchored
    at the message's emission time (issue #1997, shared with
    ``worker_fate.classify_failure`` via ``_throttle_emission_anchor``):
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

    # Issue #1997: same emission-time anchor as worker_fate.classify_for —
    # the provider's countdown started when the log line was written, not
    # now. max(now, ...) keeps a born-expired window a zero-length deferral
    # rather than a negative one. Lazy import: the anchor helper now lives
    # in worker_fate (design doc §7), and a top-level import here would
    # cycle (worker_fate -> worker -> claude_code/devin_shell already
    # imports this module's sibling functions).
    from .failure_classifier import _throttle_emission_anchor

    emitted_at = _throttle_emission_anchor(log_path, tail, now=now, throttle_markers=markers)
    defer_until = max(
        now,
        emitted_at + timedelta(minutes=minutes + slack_minutes, seconds=resume_margin_seconds),
    )
    return defer_until.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def update_session_record_with_failure_classification(
    sessions_dir: Path,
    issue_number: int,
    *,
    fallback_kind: str | None = None,
    config: OrchestratorConfig | None = None,
    session_completed: bool = False,
    now: datetime | None = None,
) -> tuple[str | None, str | None]:
    """Update a session record with failure classification after the session exits.

    This reads the existing sidecar, classifies the failure from the log tail,
    and writes back an updated record with failure_kind set.

    Log-tail classification (``worker_fate.classify_for``) always runs first.
    If it detects a provider throttle signature (``rate_limited`` /
    ``quota_exhausted``), that classification wins — including its computed
    ``throttled_until`` cooldown — regardless of ``fallback_kind``. Only when
    the log shows no throttle signature does ``fallback_kind`` apply (e.g. the
    stall watchdog's "stalled" default, or the launch-stall watchdog's
    "launch_stalled" default). This ordering matters: a worker that dies
    because it hit a provider rate limit must be classified as such even when
    the caller only knows "this looked stalled" — otherwise ``throttled_until``
    never gets set and dispatch keeps relaunching workers into the same limit.

    ``session_completed`` (issue #656): when the caller has already confirmed
    via worktree inspection that this session produced complete, committable
    work, log-tail classification is skipped entirely and ``fallback_kind`` is
    used directly. A session that finished real work cannot also have been
    killed by a provider rate-limit failure — that's ground truth, not a
    heuristic. See ``claude_code.update_worker_record_with_failure_
    classification`` for the sibling fix and the live false-positive this
    protects against (a worker's own completion-summary prose quoting
    throttle-marker text). Since #2052 the launcher runs the terminal-status
    watcher, so completion is derived here too via
    ``terminal_record_proves_completion`` -- a record proving this pid exited
    0 with a worker outcome makes the log tail completion prose.

    ``config`` is optional for backward compatibility; when provided, its
    ``runtime.throttle_error_markers``, ``runtime.quota_error_markers``, and
    ``runtime.throttle_resume_margin_s`` are used instead of the defaults.

    ``now`` is forwarded to ``worker_fate.classify_for`` (issue #822's
    injectable clock); defaults to ``datetime.now(UTC)`` there when omitted.

    Returns a tuple of (failure_kind, throttled_until_iso) for the caller to
    update runtime state if needed. ``throttled_until_iso`` is only non-None
    when log-tail classification actually matched a throttle signature.
    """
    # Lazy: this module is devin_shell's own extraction -- ``devin_shell``
    # imports it at module top for the re-export, so importing back at module
    # level would cycle. Attribute access at call time also keeps
    # ``devin_shell._sidecar_path``/``_write_json`` monkeypatch points live.
    from . import devin_shell

    sidecar_path = devin_shell._sidecar_path(sessions_dir, issue_number)
    if not sidecar_path.exists():
        return None, None

    try:
        with sidecar_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None, None

    if not isinstance(payload, dict):
        return None, None

    # Skip if already classified
    if payload.get("failure_kind") is not None:
        return payload.get("failure_kind"), None

    # Same derivation as claude_code's sibling writer (#656/#2022): the
    # terminal record the #2052 watcher leaves behind is ground truth that
    # this exact pid exited 0 with a worker outcome -- its log tail is the
    # model's own completion prose, not a provider error.
    completed = session_completed or terminal_record_proves_completion(
        sessions_dir, issue_number, "devin", payload.get("pid")
    )
    classified_kind: str | None = None
    throttled_until: str | None = None
    log_path_str = payload.get("log_path") if not completed else None
    if log_path_str:
        if config is not None:
            throttle_markers = config.runtime.throttle_error_markers
            quota_markers = config.runtime.quota_error_markers
            resume_margin_seconds = config.runtime.throttle_resume_margin_s
        else:
            throttle_markers = None
            quota_markers = None
            resume_margin_seconds = 0
        # Lazy: worker_fate reaches back into devin_shell (profile table).
        from . import worker_fate

        classified_kind, throttled_until = worker_fate.classify_for(
            "devin",
            Path(log_path_str),
            throttle_markers,
            quota_error_markers=quota_markers,
            resume_margin_seconds=resume_margin_seconds,
            now=now,
        )

    resolved_kind = classified_kind or fallback_kind
    if resolved_kind is None:
        return None, None

    payload["failure_kind"] = resolved_kind
    devin_shell._write_json(sidecar_path, payload)
    return resolved_kind, throttled_until
