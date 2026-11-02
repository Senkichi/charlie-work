"""``charlie dashboard serve`` runtime: bind, serve, and stop on orchestrator HEAD drift.

The dashboard is long-lived, so a pull of the orchestrator checkout would leave it
running stale code. A watcher thread polls the checkout HEAD and, on a change, stops the
server cleanly and reports ``restart_requested`` -- ``cli.main`` turns that into
``supervise_loop.EXIT_RESTART_REQUESTED`` (the wrapper's cross-version wire contract; the
number is never re-declared here). This module never calls ``self_deploy``, never writes
fleet state and never calls GitHub.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from ..command_result import CommandResult
from .config import DashboardConfig
from .server import ServerError, default_sources, make_server, run_server
from .theme import find_swole_root, swole_drift

log = logging.getLogger("charlie_work.dashboard")

DEFAULT_DRIFT_INTERVAL_SECONDS = 10.0
DISABLED_MESSAGE = "dashboard is disabled (dashboard.enabled: false); not serving"

HeadReader = Callable[[], str | None]


def _default_head_reader() -> str | None:
    from ..supervise import orchestrator_root, read_head_sha

    return read_head_sha(orchestrator_root())


def _log_theme_drift(fleet_dir_override: str | None) -> None:
    """Log (only) whether the vendored theme tokens differ from the swole checkout."""
    try:
        from ..fleet_registry import _load_registry
        from ..layout import fleet_registry_path

        root: Path | None = find_swole_root(
            _load_registry(fleet_registry_path(fleet_dir_override))
        )
        result = swole_drift(root)
    except Exception as exc:  # noqa: BLE001 - startup diagnostics must never block serving
        log.warning("dashboard theme drift check failed: %s", exc)
        return
    if result.status == "drifted":
        log.warning("dashboard theme drifted from swole: %s", ", ".join(result.differing))
    else:
        log.info("dashboard theme drift check: %s %s", result.status, result.detail)


def watch_head_drift(
    read_head: HeadReader,
    stop: threading.Event,
    drifted: threading.Event,
    interval_seconds: float,
    baseline: str | None = None,
) -> None:
    """Set ``drifted`` and ``stop`` once ``read_head`` reports a SHA unlike the baseline.

    ``baseline`` is the HEAD read when the process started, before any lazy import, so a
    pull landing during startup is still a drift. ``None`` reads (git hiccup) carry no
    information and are skipped; with no ``baseline`` the first non-``None`` read becomes
    it, so a transient startup failure cannot fake a drift.
    """
    while not stop.is_set():
        sha = read_head()
        if sha is not None:
            if baseline is None:
                baseline = sha
            elif sha != baseline:
                log.warning("orchestrator HEAD moved %s -> %s; restarting", baseline, sha)
                drifted.set()
                stop.set()
                return
        stop.wait(interval_seconds)


def serve_dashboard(
    config: DashboardConfig,
    fleet_dir_override: str | None,
    *,
    read_head: HeadReader | None = None,
    drift_interval_seconds: float | None = None,
) -> CommandResult:
    """Serve until interrupted or HEAD drift; bind/config problems return as values."""
    if not config.enabled:
        return CommandResult(True, DISABLED_MESSAGE, {})
    reader = read_head or _default_head_reader
    # Earliest reliable point: before default_sources lazily imports the collector modules
    # and before binding, i.e. as close to "the HEAD this code was loaded at" as we can get.
    baseline = reader()
    server = make_server(
        config, default_sources(fleet_dir_override, config.collector_interval_seconds)
    )
    if isinstance(server, ServerError):
        return CommandResult(False, server.message, {})
    log.info("dashboard serving on http://%s:%d/", config.host, server.port)
    print(f"dashboard serving on http://{config.host}:{server.port}/ (Ctrl-C to stop)")
    _log_theme_drift(fleet_dir_override)
    stop, drifted = threading.Event(), threading.Event()
    threading.Thread(
        target=watch_head_drift,
        args=(
            reader,
            stop,
            drifted,
            DEFAULT_DRIFT_INTERVAL_SECONDS
            if drift_interval_seconds is None
            else drift_interval_seconds,
            baseline,
        ),
        name="dashboard-head-watch",
        daemon=True,
    ).start()
    try:
        run_server(server, stop)
    except KeyboardInterrupt:
        stop.set()
    if drifted.is_set():
        return CommandResult(
            True, "dashboard stopped: orchestrator HEAD changed", {"restart_requested": True}
        )
    return CommandResult(True, "dashboard stopped", {})
