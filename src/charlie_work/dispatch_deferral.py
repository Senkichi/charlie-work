"""Visibility for silent dispatch deferrals (issue #1986).

``OrchestratorApp.dispatch`` / ``dispatch_rework`` defer -- return
``ok=True`` with ``data["deferred_reason"]`` and launch nothing -- when the
fleet lock is held, the state lock is busy, the GraphQL budget is low, or
GitHub errors. None of those wrote an event or a log line, so a repo that
deferred every pass looked exactly like a repo with nothing to do (swole,
2026-09-29: 2h+ without a dispatch, ``loop_completed ok: true`` throughout).

This module is the single enforcement point: :func:`records_deferral`
decorates the two entry points, so every ``return`` path -- present and
future -- is covered without per-site calls. It emits a ``dispatch_deferred``
event + log line per deferral, and keeps a per-lane consecutive-deferral
streak in a sidecar (not ``state.json``: one reason *is* the state lock
being busy). At :data:`DEFAULT_STARVATION_THRESHOLD` consecutive deferrals
it emits one error-level ``dispatch_starved`` event.

Only ``ok=True`` deferrals are recorded: an ``ok=False`` deferral (provider
throttle) already surfaces as a ``not ok:`` WARNING in ``fleet_dispatch``,
and is neither a reset nor a streak increment.
"""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from charlie_work import layout
from charlie_work.instrumentation import log_event
from charlie_work.state import StateLockBusy, state_lock

logger = logging.getLogger(__name__)

DISPATCH_DEFERRED_KIND = "dispatch_deferred"
DISPATCH_STARVED_KIND = "dispatch_starved"
DEFAULT_STARVATION_THRESHOLD = 3


def deferral_reason(result: Any) -> str | None:
    """Return the silent-deferral reason carried by ``result``, else ``None``."""
    if not getattr(result, "ok", False):
        return None
    reason = (getattr(result, "data", None) or {}).get("deferred_reason")
    return reason if isinstance(reason, str) and reason else None


def _event_payload(lane: str, reason: str, result: Any, streak: int) -> dict[str, Any]:
    # merged_pr* fields are bulky finalize bookkeeping, not deferral context.
    detail = {k: v for k, v in result.data.items() if not k.startswith("merged_pr")}
    return {
        **detail,
        "lane": lane,
        "deferred_reason": reason,
        "message": result.message,
        "consecutive_deferrals": streak,
    }


def _read_streaks(path: Path) -> dict[str, int]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, int)}


def _write_streaks(path: Path, streaks: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(streaks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _advance_streak(state_file: Path, lane: str, deferred: bool) -> int:
    """Increment (deferred) or clear (not) ``lane``'s streak; return the new count."""
    path = layout.dispatch_deferral_streak_path(state_file.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    with state_lock(path):
        streaks = _read_streaks(path)
        count = streaks.get(lane, 0) + 1 if deferred else 0
        if count != streaks.get(lane, 0):
            _write_streaks(
                path,
                {**streaks, lane: count}
                if count
                else {k: v for k, v in streaks.items() if k != lane},
            )
        return count


def record_lane_result(app: Any, lane: str, result: Any) -> None:
    """Emit/track the deferral outcome of one ``dispatch``-family call.

    Never raises: this is diagnostics and must not be why a pass fails.
    """
    if getattr(app, "dry_run", False):
        return
    try:
        reason = deferral_reason(result)
        if reason is None and not result.ok:
            return  # ok=False deferral (throttle): already logged, streak untouched
        state_file = app.paths.state_file
        try:
            streak = _advance_streak(state_file, lane, reason is not None)
        except StateLockBusy:
            streak = -1  # counter lock contended this instant; the event still fires
        if reason is None:
            return
        repo = app.repo_root.name
        logger.warning(
            "%s deferred for %s: %s (%s; consecutive=%s)",
            lane,
            repo,
            reason,
            result.message,
            streak if streak > 0 else "?",
        )
        log_event(
            state_file,
            DISPATCH_DEFERRED_KIND,
            _event_payload(lane, reason, result, streak),
            repo=repo,
        )
        if streak == DEFAULT_STARVATION_THRESHOLD:
            log_event(
                state_file,
                DISPATCH_STARVED_KIND,
                {
                    "lane": lane,
                    "deferred_reason": reason,
                    "consecutive_deferrals": streak,
                    "threshold": DEFAULT_STARVATION_THRESHOLD,
                },
                repo=repo,
            )
    except Exception:  # noqa: BLE001 -- diagnostics never break dispatch
        logger.warning("dispatch deferral recording failed", exc_info=True)


def records_deferral(lane: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate an ``OrchestratorApp`` dispatch entry point (``self`` first)."""

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = fn(self, *args, **kwargs)
            record_lane_result(self, lane, result)
            return result

        return wrapper

    return decorate
