from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from .fleet_paths import fleet_dir

LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"

# How long a day-stamped log file is kept before _prune_old_logs deletes it.
# See _prune_old_logs' docstring for the worst-case on-disk footprint this
# bounds.
LOG_RETENTION_DAYS = 14

_LOG_FILENAME_GLOB = "charlie-work-*.log"


def _log_file_path(now: datetime | None = None) -> Path:
    """Return today's UTC-dated orchestrator log file path.

    ``fleet_dir()`` (fleet_paths.py) is the existing host-wide, repo-agnostic
    state directory (``%LOCALAPPDATA%\\charlie-work`` on Windows,
    ``${XDG_STATE_HOME:-~/.local/state}/charlie-work`` on POSIX, already used
    for ``fleet.json``/``fleet-supervisor.lock``/``config.yaml``). It is
    deliberately NOT derived from CWD: the same 5-minute Task Scheduler job
    (``examples/schedule/charlie-fleet-task.xml``) can run from whatever
    directory Task Scheduler happens to set as its working directory, and
    ``fleet`` commands dispatch into N different repos from one process, so
    there is no single "repo root" to anchor a fleet-wide log on anyway.

    The filename embeds the UTC date rather than using one fixed name rotated
    by ``logging.handlers.RotatingFileHandler``/``TimedRotatingFileHandler``.
    charlie-work routinely runs as several *concurrent OS processes* sharing
    this file -- an operator's manual ``charlie work`` overlapping a
    Task-Scheduler-fired ``charlie fleet bash-rats``, or two fleet passes
    five minutes apart that both outlive that window. Both rotating handlers
    roll over via ``os.rename()`` on the currently-open file; that is safe
    for a single writer, but two processes racing a rollover on Windows can
    hit ``PermissionError: [WinError 32] The process cannot access the file
    because it is being used by another process`` -- Windows refuses to
    rename a file another process still has open, unlike POSIX where
    rename-over-an-open-fd is unconditionally allowed. A day-stamped filename
    sidesteps the race entirely: "rotation" is just every process
    independently computing the same deterministic name for today and
    opening it in append mode -- no renaming, no cross-process coordination,
    no multi-writer hazard.
    """
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d")
    return fleet_dir() / "logs" / f"charlie-work-{stamp}.log"


def _prune_old_logs(logs_dir: Path, *, retention_days: int = LOG_RETENTION_DAYS) -> None:
    """Delete day-stamped log files older than *retention_days*.

    Worst-case on-disk footprint with this cap, generously overestimated:
    the busiest expected cadence is the shipped 5-minute Task Scheduler
    trigger (``examples/schedule/charlie-fleet-task.xml``); at ~20KB of INFO
    lines per pass (comfortably above one worker-census line plus the
    handful of other lines a pass emits) and a worst-case back-to-back
    5-minute cadence (288 passes/day) that is ~5.6MB/day. At
    ``LOG_RETENTION_DAYS=14`` that is a ~80MB steady-state ceiling
    regardless of actual volume -- bounded, and self-pruning, on a box where
    58.5GB of leaked temp files was already the incident that prompted this
    audit (see MEMORY.md ops_runner_topology_repo_scoped).

    Never raises: a prune failure (permissions, a file deleted concurrently
    by another process, a read-only filesystem) must not block logging
    setup -- same principle as the rest of this module.
    """
    try:
        if not logs_dir.is_dir():
            return
        cutoff = datetime.now(UTC).timestamp() - retention_days * 86400
        for path in logs_dir.glob(_LOG_FILENAME_GLOB):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                continue
    except OSError:
        return


def configure_logging(*, verbose: bool = False) -> None:
    """Configure root logging: stderr always, a dated on-disk file when possible.

    Issue #646 follow-up: the worker census (and everything else logged at
    this level) previously only reached ``sys.stderr``
    (``logging.basicConfig(stream=sys.stderr)``), with no ``FileHandler``
    anywhere in the codebase. stderr is destroyed the instant a Task
    Scheduler-fired invocation exits, and the shipped scheduler template
    (``examples/schedule/charlie-fleet-task.xml``) ships its
    ``>> charlie-work.log 2>&1`` redirection commented out by default -- so a
    default install persisted nothing, and the census answered "how many
    suites were running at 11:33" only for an operator who had already
    edited the XML template. This function is the single point of
    enforcement for both streams: every entry point that reaches this
    (currently just ``cli.py::main``) gets a file on disk independent of how
    the process was launched or whether the launcher redirected its stream.

    Idempotent the same way ``logging.basicConfig`` is: if the root logger
    already has handlers (e.g. a test process that calls ``main()``
    repeatedly in one interpreter), this returns immediately rather than
    opening -- and abandoning -- another file handle each call.

    Never raises. If the fleet dir can't be created or the log file can't be
    opened (permissions, a read-only filesystem, disk full), this warns on
    stderr and continues with the stream handler alone -- the same principle
    applied to ``_log_worker_census`` itself (workflow.py): a diagnostic that
    can crash the fleet dispatcher is worse than a diagnostic that is
    sometimes incomplete.
    """
    if logging.root.handlers:
        return

    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(LOG_FORMAT)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)

    try:
        logs_dir = fleet_dir() / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        _prune_old_logs(logs_dir)
        file_handler: logging.Handler | None = logging.FileHandler(
            _log_file_path(), mode="a", encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
    except OSError as exc:
        file_handler = None
        setup_error = exc
    else:
        setup_error = None

    handlers: list[logging.Handler] = [stream_handler]
    if file_handler is not None:
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers)

    if setup_error is not None:
        logging.getLogger(__name__).warning(
            "could not open charlie-work log file under %s -- continuing with stderr only: %s",
            fleet_dir() / "logs",
            setup_error,
        )
