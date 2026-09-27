"""Fleet-supervisor notify-digest freshness supervision (issue #1859).

Every emit path in ``notify.py`` returns errors as values, and the only
record of a failing sink is ``digest["notify_error"]`` on the pass result
-- a writer that stops landing entries (lost config section, deleted
directory, wedged volume) is otherwise invisible indefinitely. The daemon's
real digest writer was dead for three weeks before anyone noticed because
nothing ever looked at the file. This module is the writer-side half of
the fix: a once-per-pass staleness check the fleet supervisor runs, plus
the startup ``notify_resolution`` event that publishes what the daemon
*actually resolved* -- enabled/sink and the absolute digest path after
cwd-anchoring -- so the heartbeat consumer answers questions about the
daemon's world, not its own checkout's. The consumer half (the heartbeat
script's verdict) lives in ``scripts/heartbeat_check.py`` on top of
``notify_digest_check.py``.

Imported by ``fleet_dispatch`` and ``cli`` -- this module pulls in
``instrumentation`` (and transitively ``ci_fleet``), so it is deliberately
NOT a leaf like ``notify_digest_check``.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path
from typing import Any

from . import layout
from .instrumentation import log_event
from .notify import digest_file_status
from .notify_digest_check import NOTIFY_DIGEST_STALE_HOURS

logger = logging.getLogger(__name__)

#: Writer-side staleness bound, derived from the shared consumer constant
#: so the two tripwires can never drift. Sized from the real daemon
#: digest's measured healthy-gap distribution (max 47.1h) -- see the
#: constant's own comment in ``notify_digest_check``.
NOTIFY_DIGEST_STALE_SECONDS = NOTIFY_DIGEST_STALE_HOURS * 60 * 60

#: Module-level latch so a persistently-stale digest emits at most one
#: ``notify_digest_stale`` event per staleness-bound interval per process
#: rather than one per pass (~144/day at the default cadence). Keyed on the
#: configured file path; the key is cleared when the file goes fresh again
#: so a *new* stale episode re-fires, a supervisor restart re-fires once,
#: and a still-stale digest re-fires once per bound -- the re-fire is what
#: keeps a fresh event inside the heartbeat consumer's lookback window for
#: the whole episode. Both edge directions stay visible.
_stale_episodes: dict[str, float] = {}


def _resolve_for_event(file_path: str) -> str | None:
    """Absolute form of a configured ``file_path`` for event payloads.

    ``_file_sink`` interprets a relative path against the supervisor's cwd,
    so the resolved form is the location the daemon is actually writing to
    -- the value the heartbeat consumer needs, since its own checkout can
    differ from the daemon's (round-1 review). ``Path.resolve()`` can raise
    on unresolvable paths; degrade to ``absolute()`` (which itself reads
    cwd and can raise when the cwd is gone), then to the raw string, rather
    than lose the event.
    """
    if not file_path:
        return None
    try:
        return str(Path(file_path).resolve())
    except (OSError, RuntimeError, ValueError):
        try:
            return str(Path(file_path).absolute())
        except (OSError, RuntimeError, ValueError):
            return file_path


def resolve_fleet_notify(notify_config: Any, state_root: Path) -> Any:
    """Resolve the ``file_path=""`` sentinel for a fleet-level notify config.

    ``NotifyConfig.file_path`` documents the empty string as "derive from
    ``runtime.state_dir``" -- a substitution ``paths.resolved_layout``
    performs per repo against that repo's own state root. The fleet
    supervisor never runs that pass (it has no single repo root), so the
    raw sentinel reached ``_file_sink`` and every fleet-level emit failed
    "file_path is empty" while ``report_notify_resolution`` flagged the
    state as misconfigured on every supervisor start (issue #1899). The
    fleet analog of the per-repo anchor is the supervisor's own
    bookkeeping root -- ``supervisor_runtime_paths(runtime.state_dir).root``
    -- which callers compute once and pass in here so the report, the
    staleness probe, and every emit all agree on one path.

    ``None`` stays ``None`` (absent notify section); non-dataclass
    stand-ins (test doubles) pass through untouched because
    ``dataclasses.replace`` cannot rebuild them; an explicit
    ``file_path`` -- relative or absolute -- is the operator's own choice
    and is preserved verbatim. The frozen source config is never mutated.
    """
    if notify_config is None or not dataclasses.is_dataclass(notify_config):
        return notify_config
    if getattr(notify_config, "file_path", ""):
        return notify_config
    return dataclasses.replace(
        notify_config, file_path=str(layout.notify_digest_default(state_root))
    )


def check_notify_digest_freshness(
    notify_config: Any, fleet_state_path: Path, *, now: float | None = None
) -> None:
    """Warn into the fleet events.db when an enabled file sink has gone stale.

    Runs once per fleet pass (``fleet_loop`` calls it only when
    ``notify.enabled`` resolved True). ``digest_file_status`` probes the
    file the same way ``_file_sink`` resolves it; a missing/unreadable file
    and an mtime older than ``NOTIFY_DIGEST_STALE_SECONDS`` are both stale
    -- the former is the "writer never landed a line" shape. Emits at most
    one ``notify_digest_stale`` warning event per staleness-bound interval
    while stale (see ``_stale_episodes``) -- a long-lived dead writer keeps
    a fresh event inside the heartbeat's lookback window instead of firing
    once and going quiet -- and clears the latch on recovery so the *next*
    episode re-fires. Never raises: the check itself failing must not take
    the pass down with the pipeline it watches.
    """
    try:
        status = digest_file_status(notify_config, now=now)
    except Exception:
        return
    if status is None:
        # Not a file sink, or file_path unset (the incoherent enabled/file/
        # empty-path combination is published by report_notify_resolution's
        # notify_resolution event at supervisor startup instead).
        return
    now_ts = time.time() if now is None else now
    file_key = str(getattr(notify_config, "file_path", "") or "")
    stale = not status.exists or (
        status.age_seconds is not None and status.age_seconds > NOTIFY_DIGEST_STALE_SECONDS
    )
    if not stale:
        _stale_episodes.pop(file_key, None)
        return
    last_reported = _stale_episodes.get(file_key)
    if last_reported is not None and (now_ts - last_reported) < NOTIFY_DIGEST_STALE_SECONDS:
        return
    _stale_episodes[file_key] = now_ts
    try:
        # write-gate-exempt(issue=1859): fleet-level supervisor telemetry; no repo WriteGate in scope
        log_event(
            fleet_state_path,
            "notify_digest_stale",
            {
                "file_path": file_key,
                "resolved_file_path": _resolve_for_event(file_key),
                "exists": status.exists,
                "age_seconds": (None if status.age_seconds is None else round(status.age_seconds)),
                "stale_after_seconds": NOTIFY_DIGEST_STALE_SECONDS,
            },
        )
    except Exception:
        logger.debug("Failed to record notify_digest_stale", exc_info=True)


def report_notify_resolution(
    notify_config: Any, global_config_path: Path, fleet_state_path: Path
) -> None:
    """Publish the resolved notify sink once per supervisor start (issue #1859).

    ``NotifyConfig.enabled`` defaults False and an absent ``notify:``
    section is a deliberate no-op, so a config file that is lost or never
    carried forward degrades the entire notification pipeline to silence
    with no error raised anywhere -- the 2026-08-31 outage ran three weeks
    before anyone noticed. An ``enabled=False`` readout is INFO (a fleet
    that never opted in is legitimate); the incoherent
    enabled/file-sink/empty-file_path combination is a WARNING because
    every emit fails "file_path is empty" and nothing else surfaces it.

    The same resolution is written to the fleet ``events.db`` as a
    ``notify_resolution`` event carrying the absolute, cwd-anchored digest
    path -- the heartbeat consumer reads it to learn what the daemon
    actually resolved rather than re-deriving the notify config from the
    script's own checkout (which resolves a *different* tree on this host).
    The event is also what surfaces the enabled/file/empty-path state
    through a consumed signal: the heartbeat anomalies on it, where the
    startup log line alone was write-only.
    """
    if notify_config is None:
        enabled, sink, file_path = False, "", ""
    else:
        enabled = bool(getattr(notify_config, "enabled", False))
        sink = str(getattr(notify_config, "sink", "") or "").lower()
        file_path = str(getattr(notify_config, "file_path", "") or "")
    resolved = _resolve_for_event(file_path) if sink == "file" else None
    misconfigured = enabled and sink == "file" and not file_path

    if notify_config is None:
        logger.info("Fleet supervisor notify config: no notify section (disabled)")
    else:
        logger.info(
            "Fleet supervisor notify config: enabled=%s sink=%s file_path=%s",
            enabled,
            sink or "(unset)",
            file_path or "(unset)",
        )
    if misconfigured:
        logger.warning(
            "Fleet supervisor notify config: enabled with sink=file but "
            "file_path is unset -- every emit_digest call will fail "
            "'file_path is empty'; set notify.file_path in the config "
            "layer that supplies it (see %s or the checkout's "
            "orchestrator.config.yaml)",
            global_config_path,
        )

    try:
        # write-gate-exempt(issue=1859): fleet-level supervisor telemetry; no repo WriteGate in scope
        log_event(
            fleet_state_path,
            "notify_resolution",
            {
                "enabled": enabled,
                "sink": sink,
                "file_path": file_path,
                "resolved_file_path": resolved,
                "file_path_empty": misconfigured,
                "global_config_path": str(global_config_path),
            },
            level="warning" if misconfigured else "info",
        )
    except Exception:
        logger.debug("Failed to record notify_resolution", exc_info=True)
