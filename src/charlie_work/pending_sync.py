"""Deferred ``uv sync`` marker and its starvation bound (issue #1855).

Extracted from ``supervise.py`` so new code does not land in an over-cap
monolith (file-size ratchet, issue #1442). This module owns the
pending-sync marker schema (read/write/clear under
``layout.pending_sync_path``), the episode ``written_at`` timestamp the
marker carries, the age computation the starvation bound measures, the
default bound, and the once-per-episode ``self_deploy_sync_starved``
event emission. ``supervise.py`` re-exports this surface so existing
imports and tests keep working.

Self-deploy defers ``uv sync`` while fleet workers are live. Under
continuous load that deferral was unbounded (issue #1855). The marker's
``written_at`` records the FIRST deferral of an episode and is carried
forward verbatim on every rewrite, so its age bounds the whole episode --
not the latest pass. Once the age crosses ``starvation_seconds`` the
deferred result reports ``starved=True`` and the caller drains new
dispatches so the live-worker count reaches zero and the sync can land;
nothing is killed. A single ``self_deploy_sync_starved`` event fires per
episode, deduplicated by the ``starved_notified`` latch in the marker
itself so the signal survives the head-moved supervisor restarts that
happen mid-episode.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import layout
from .instrumentation import log_event
from .state import utc_now

#: Event kind for the deferred-sync starvation signal (issue #1855). Read
#: here (not re-declared at call sites) for the same reason label strings
#: are read from ``LabelConfig``: ``doctor_sync_starvation`` queries this
#: kind back out of events.db, and a literal scattered across emit/query
#: sites drifts silently.
SELF_DEPLOY_SYNC_STARVED_KIND = "self_deploy_sync_starved"

#: Default starvation bound (seconds) for a deferred ``uv sync`` that stays
#: pending under continuous live fleet workers before the supervisor stops
#: admitting new dispatches (issue #1855). Mirrors
#: ``config.SupervisorConfig.dependency_sync_starvation_seconds``'s default;
#: <= 0 disables the bound.
DEFAULT_SYNC_STARVATION_SECONDS = 14400


@dataclass(frozen=True)
class SyncDeferral:
    """Outcome of recording one deferred pass of a pending-sync episode.

    ``starved`` is True when the episode's age -- measured from the
    marker's ``written_at`` -- has crossed the configured bound this pass,
    including passes where the ``starved_notified`` latch was already set.
    ``marker_age_seconds`` is ``None`` when the marker has no usable
    ``written_at`` (a pre-#1855 marker or a hand-edit).
    """

    starved: bool
    marker_age_seconds: float | None


def _pending_sync_marker_path(state_root: Path) -> Path:
    """Return the path to the deferred-``uv sync`` marker under ``state_root``."""
    return layout.pending_sync_path(state_root)


def _write_marker(
    path: Path, from_sha: str, to_sha: str, *, starved_notified: bool = False
) -> None:
    """Persist the pending-sync marker atomically (temp-file + replace).

    ``written_at`` records the FIRST deferral of this episode and is carried
    forward verbatim on every later rewrite (each deferred pass refreshes
    ``to_sha`` as new pulls land) -- it is what the starvation bound
    (issue #1855) measures, and it must survive the head-moved supervisor
    restarts that happen mid-episode. A missing or unparseable prior
    ``written_at`` re-arms to now (a pre-#1855 marker, or a hand-edit, gets
    one fresh window rather than an unverifiable age). Same shape for
    ``starved_notified``: latched true once the ``self_deploy_sync_starved``
    event fires, preserved across rewrites, so the event fires once per
    episode rather than once per deferred pass.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    prior = _read_marker(path)
    prior_written = prior.get("written_at")
    payload: dict[str, Any] = {
        "from_sha": from_sha,
        "to_sha": to_sha,
        "written_at": (
            prior_written if _parse_marker_timestamp(prior_written) is not None else utc_now()
        ),
    }
    if starved_notified or prior.get("starved_notified"):
        payload["starved_notified"] = True
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_marker(path: Path) -> dict[str, Any]:
    """Read the marker, returning an empty dict on any read/parse error."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    # A valid-JSON non-mapping marker (hand-edit, torn schema change) reads
    # as absent rather than crashing the deferral branch on ``marker.get``.
    return data if isinstance(data, dict) else {}


def _parse_marker_timestamp(value: Any) -> datetime.datetime | None:
    """Parse a marker ``written_at`` ISO-8601 string, or ``None`` on any miss.

    Naive timestamps are read as UTC -- the writer (``utc_now``) always emits
    ``Z``, so a naive value can only come from a hand-edited marker, and
    assuming UTC errs toward a *larger* measured age (the starvation-safe
    direction) when the local convention was UTC anyway.
    """
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.UTC)
    return parsed


def _pending_sync_age_seconds(
    marker: dict[str, Any] | None, *, now: datetime.datetime
) -> float | None:
    """Age of the pending-sync episode in seconds, or ``None`` when unknown.

    ``None`` covers an absent marker and a marker with no usable
    ``written_at`` -- the safe direction is to keep deferring on an
    unverifiable age rather than drain-dispatch on one.
    """
    if not marker:
        return None
    start = _parse_marker_timestamp(marker.get("written_at"))
    if start is None:
        return None
    return (now - start).total_seconds()


def _clear_marker(path: Path) -> None:
    """Remove the pending-sync marker, if it exists."""
    path.unlink(missing_ok=True)


def record_sync_deferral(
    marker_path: Path,
    prior_marker: dict[str, Any] | None,
    *,
    from_sha: str,
    to_sha: str,
    live_count: int,
    starvation_seconds: int,
    state_path: Path,
    now: datetime.datetime | None = None,
) -> SyncDeferral:
    """Persist one deferred pass and emit the once-per-episode starvation event.

    Bounds how long a pending sync may starve under continuous load (issue
    #1855). The marker's ``written_at`` is the FIRST deferral of this
    episode (``_write_marker`` carries it forward), so the bound measures
    the whole episode, not the latest pass. ``starved`` reports the trip to
    the caller -- which stops admitting new dispatches -- rather than
    killing anything live, the same non-destructive posture as the rest of
    the self-deploy path.

    ``prior_marker`` is the marker read before this pass's rewrite (or
    ``None`` when absent); the rewrite happens unconditionally so a fresh
    ``to_sha`` lands even on a starved pass. When the bound has tripped and
    the latch is unset, one ``self_deploy_sync_starved`` event is emitted to
    ``state_path``'s events.db and ``starved_notified`` is latched --
    emission precedes the latch so a crash between the two re-emits on the
    next pass, the safe direction for a once-per-episode signal.
    ``starvation_seconds <= 0`` disables the bound.
    """
    marker_age = _pending_sync_age_seconds(
        prior_marker, now=now or datetime.datetime.now(datetime.UTC)
    )
    starved = (
        starvation_seconds > 0 and marker_age is not None and marker_age >= starvation_seconds
    )
    _write_marker(marker_path, from_sha, to_sha)
    if starved and not (prior_marker or {}).get("starved_notified"):
        # write-gate-exempt(issue=1855): durable events.db signal, outside state lock
        log_event(
            state_path,
            SELF_DEPLOY_SYNC_STARVED_KIND,
            {
                "pending_seconds": int(marker_age or 0),
                "starvation_seconds": starvation_seconds,
                "live_count": live_count,
                "from_sha": from_sha,
                "to_sha": to_sha,
            },
        )
        _write_marker(marker_path, from_sha, to_sha, starved_notified=True)
    return SyncDeferral(starved=starved, marker_age_seconds=marker_age)
