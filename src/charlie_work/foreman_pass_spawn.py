"""The supervise loop's periodic ``charlie scheme`` spawn (issue #3122).

Extracted from ``fleet_dispatch`` under the file-size ratchet (#1442): the
loop's only foreman edge — the cadence lookup, the cheap ``foremen/``
presence gate, and the interval-gated spawn — lives here so the monolith
keeps only the per-iteration wiring that cannot leave the loop body. The
names are re-exported through ``fleet_dispatch`` so existing callers and
test patch seams keep resolving on the ``fleet_dispatch.<name>`` facade.

ADR-0010 keeps this module hub-side and import-clean: the pass is spawned
as ``sys.executable -m charlie_work … scheme`` argv through the
``foreman_pass_launch`` host port — never an import of
``charlie_work.foreman`` (``tests/test_foreman_import_direction.py``), and
"a foreman exists" is answered by directory shape alone, not by any of the
package's own readers.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from . import layout
from .fleet_paths import fleet_dir
from .fleet_pause import fleet_pause_pending
from .foreman_config import ForemanConfig
from .host import current as _host_current
from .instrumentation import log_event
from .supervise import orchestrator_root

logger = logging.getLogger(__name__)


def _foreman_pass_cadence(global_config: Any, *, dry_run: bool) -> tuple[bool, float]:
    """``(enabled, interval_seconds)`` for the loop's periodic foreman pass.

    The loop never imports ``charlie_work.foreman`` — ``ForemanConfig`` is
    the host-wide section in ``charlie_work.foreman_config``, outside the
    package, so this read is a config lookup, not a foreman edge. The
    ``getattr`` fallback tolerates config stand-ins predating the section.
    ``--dry-run`` previews the loop itself; a spawned pass is real work, so
    it stays off under dry_run.
    """
    foreman_cfg = getattr(global_config, "foreman", None)
    if foreman_cfg is None:
        foreman_cfg = ForemanConfig()
    enabled = bool(foreman_cfg.enabled) and not dry_run
    return enabled, float(foreman_cfg.pass_interval_minutes) * 60.0


def _foremen_present(fleet_dir_override: str | None) -> bool:
    """Whether ``<fleet_dir>/foremen`` holds at least one foreman directory.

    The supervise loop's cheap gate for spawning ``charlie scheme``: it
    cannot import ``charlie_work.foreman`` (ADR-0010), so "a foreman exists"
    is answered by the directory shape alone — any non-``_``-prefixed dir
    under ``foremen/`` (``_shared`` is not a foreman).
    """
    root = fleet_dir(override=fleet_dir_override) / layout.FOREMEN_DIRNAME
    try:
        return any(entry.is_dir() and not entry.name.startswith("_") for entry in root.iterdir())
    except OSError:
        return False


def _maybe_start_foreman_pass(
    child: Any | None,
    last_started_at: float,
    *,
    now: float,
    interval_seconds: float,
    enabled: bool,
    fleet_dir_override: str | None,
) -> tuple[Any | None, float]:
    """Start the periodic ``charlie scheme`` child when due; never waits (issue #3122).

    Returns ``(live_child, last_started_at)``. The previous child still
    running is left alone — the loop skips the spawn rather than stacking a
    second pass (the pass's own ``foremen/pass.lock`` is the backstop). A
    skipped tick (disabled, paused, no foremen) leaves the anchor so an
    overdue pass starts the moment conditions clear; a failed spawn moves it
    so a wedged spawn retries next interval, not next poll tick.
    """
    if child is not None:
        if child.poll() is None:
            return child, last_started_at
        child = None
    if (now - last_started_at) < interval_seconds:
        return None, last_started_at
    if (
        not enabled
        or not _foremen_present(fleet_dir_override)
        or fleet_pause_pending(fleet_dir_override)
    ):
        return None, last_started_at
    argv = (
        sys.executable,
        "-m",
        "charlie_work",
        "--fleet-dir",
        str(fleet_dir(override=fleet_dir_override)),
        "scheme",
    )
    started = _host_current().foreman_pass_launch.launch(argv, cwd=orchestrator_root())
    if started.error is not None or started.child is None:
        # write-gate-exempt(issue=3122): fleet-level event, no write_gate receiver — mirrors the spawn failure into the fleet events.db.
        log_event(  # event-consumer: audit-only — the retry next interval is the recovery path
            layout.state_file_path(fleet_dir(override=fleet_dir_override)),
            "foreman_pass_start_failed",
            {"argv": list(argv), "error": started.error or "no child returned"},
        )
        return None, now
    logger.info("foreman pass child spawned: %s", " ".join(argv))
    return started.child, now
