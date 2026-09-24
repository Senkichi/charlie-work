"""Fleet-supervisor notify-digest freshness supervision (issue #1859).

Every emit path in ``notify.py`` returns errors as values, and the only
record of a failing sink is ``digest["notify_error"]`` on the pass result
-- a writer that stops landing entries (lost config section, deleted
directory, wedged volume) is otherwise invisible indefinitely. The daemon's
real digest writer was dead for three weeks before anyone noticed because
nothing ever looked at the file. This module is the writer-side half of
the fix: a once-per-pass staleness check the fleet supervisor runs, plus
the startup resolution report that makes the sink's configured target
observable in the supervisor log. The consumer half (the heartbeat
script's verdict) lives in ``notify_digest_check.py``.

Imported only by ``fleet_dispatch`` -- this module pulls in
``instrumentation`` (and transitively ``ci_fleet``), so it is deliberately
NOT a leaf like ``notify_digest_check``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .instrumentation import log_event
from .notify import digest_file_status
from .notify_digest_check import NOTIFY_DIGEST_STALE_HOURS

logger = logging.getLogger(__name__)

#: Writer-side staleness bound, derived from the shared consumer constant
#: so the two tripwires can never drift. A digest older than this means the
#: enabled writer has produced nothing for a full day -- itself the anomaly.
NOTIFY_DIGEST_STALE_SECONDS = NOTIFY_DIGEST_STALE_HOURS * 60 * 60

#: Module-level latch so a persistently-stale digest emits one warning event
#: per episode per supervisor process rather than one per pass (~144/day at
#: the default cadence). Keyed on the configured file path; the key is
#: cleared when the file goes fresh again so a *new* stale episode
#: re-fires, and a supervisor restart re-fires once -- both edge directions
#: stay visible.
_stale_episodes: dict[str, float] = {}


def check_notify_digest_freshness(
    notify_config: Any, fleet_state_path: Path, *, now: float | None = None
) -> None:
    """Warn into the fleet events.db when an enabled file sink has gone stale.

    Runs once per fleet pass (``fleet_loop`` calls it only when
    ``notify.enabled`` resolved True). ``digest_file_status`` probes the
    file the same way ``_file_sink`` resolves it; a missing/unreadable file
    and an mtime older than ``NOTIFY_DIGEST_STALE_SECONDS`` are both stale
    -- the former is the "writer never landed a line" shape. Emits at most
    one ``notify_digest_stale`` warning event per stale episode per process
    (see ``_stale_episodes``); a quiet-but-healthy fleet can go a day
    without attention transitions, so this is deliberately a warning event
    for the events.db record, not a pass failure. Never raises: the check
    itself failing must not take the pass down with the pipeline it
    watches.
    """
    try:
        status = digest_file_status(notify_config, now=now)
    except Exception:
        return
    if status is None:
        # Not a file sink, or file_path unset (the incoherent enabled/file/
        # empty-path combination is reported at supervisor startup instead).
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
        log_event(
            fleet_state_path,
            "notify_digest_stale",
            {
                "file_path": file_key,
                "exists": status.exists,
                "age_seconds": (None if status.age_seconds is None else round(status.age_seconds)),
                "stale_after_seconds": NOTIFY_DIGEST_STALE_SECONDS,
            },
        )
    except Exception:
        logger.debug("Failed to record notify_digest_stale", exc_info=True)


def report_notify_resolution(notify_config: Any, global_config_path: Path) -> None:
    """Log the resolved notify sink once per supervisor start (issue #1859).

    ``NotifyConfig.enabled`` defaults False and an absent ``notify:``
    section is a deliberate no-op, so a config file that is lost or never
    carried forward degrades the entire notification pipeline to silence
    with no error raised anywhere -- the 2026-08-31 outage ran three weeks
    before anyone noticed. An ``enabled=False`` readout is INFO (a fleet
    that never opted in is legitimate); the incoherent
    enabled/file-sink/empty-file_path combination is a WARNING because
    every emit fails "file_path is empty" and nothing else surfaces it.
    """
    if notify_config is None:
        logger.info("Fleet supervisor notify config: no notify section (disabled)")
        return
    sink = str(getattr(notify_config, "sink", "") or "").lower()
    file_path = getattr(notify_config, "file_path", "") or ""
    logger.info(
        "Fleet supervisor notify config: enabled=%s sink=%s file_path=%s",
        getattr(notify_config, "enabled", False),
        sink or "(unset)",
        file_path or "(unset)",
    )
    if getattr(notify_config, "enabled", False) and sink == "file" and not file_path:
        logger.warning(
            "Fleet supervisor notify config: enabled with sink=file but "
            "file_path is unset -- every emit_digest call will fail "
            "'file_path is empty'; set notify.file_path in the config "
            "layer that supplies it (see %s or the checkout's "
            "orchestrator.config.yaml)",
            global_config_path,
        )
