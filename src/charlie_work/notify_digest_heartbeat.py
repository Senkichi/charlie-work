"""Heartbeat-side consumer of the fleet supervisor's notify-digest events
(issue #1859).

The notify digest shipped as a signal without a consumer: the live daemon's
file-sink writer was dead for three weeks (last real entry 2026-08-31) while
nothing read the file at all. ``scripts/heartbeat_check.py`` fixed that by
reading two kinds the fleet supervisor's own ``events.db`` carries
(``check_wedge_kill_loop`` precedent):

* ``notify_resolution`` -- once per supervisor start, publishing what the
  daemon *actually resolved*: enabled/sink and the absolute, cwd-anchored
  digest path. Consumed instead of re-deriving the ``notify:`` section from
  the script's own checkout -- on the host that motivated this, the script's
  checkout and the daemon's config root are different trees and resolved
  different answers (round-1 review).
* ``notify_digest_stale`` -- the writer-side per-pass tripwire, surfaced in
  the facts line so the heartbeat's verdict and the supervisor's own
  detector visibly agree or disagree.

This module holds the actual events.db-read-plus-verdict logic. It was
extracted out of ``scripts/heartbeat_check.py`` (round-2 review, file-size
ratchet) to keep that file at or under its high-water mark; the script keeps
only the guarded import plus a thin wrapper (see its
``check_notify_digest_freshness``) that degrades to WARN, never raising,
when this module or the ``notify_digest_check`` leaf it delegates the file
probe to is not importable at all -- the script's own contract
(``scripts/README.md``: "a broken package install can never break the check
that would detect it") is unaffected by this move, since the guarded
try/except in the script now covers this module the same way it already
covered ``notify_digest_check``.

Deliberately takes the report sink, the resolved fleet dir, the script's own
``parse_iso`` and the already-guarded ``notify_digest_check`` module object
as parameters rather than importing or redefining any of them: the caller's
``_ndc`` is never re-imported here, so a test that monkeypatches the
caller's ``_ndc`` attribute (including replacing it with ``None``) or
patches ``probe_digest_file`` on it observes the identical effect no matter
which module ends up calling it -- module objects are process-wide
singletons, and the caller's guard still runs before this module is ever
reached.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol

from charlie_work.heartbeat_alarms import emit
from charlie_work.heartbeat_alarms_fleet import eval_notify_digest

NOTIFY_RESOLUTION_EVENT_KIND = "notify_resolution"
NOTIFY_DIGEST_STALE_EVENT_KIND = "notify_digest_stale"


class SupportsReport(Protocol):
    """Structural stand-in for ``scripts.heartbeat_check.Report``.

    Kept as a Protocol (stdlib, no import of the script or the package's own
    types) rather than importing ``Report`` itself -- the script is loaded
    ad hoc by tests via ``importlib`` and this module must not need to
    import it back.
    """

    def ok(self, check: str, facts: str) -> None: ...

    def warn(self, check: str, detail: str) -> None: ...

    def anom(self, check: str, detail: str) -> None: ...


def check_notify_digest_freshness(
    report: SupportsReport,
    *,
    now: datetime,
    ndc: Any,
    fleet_dir: Path,
    parse_iso: Callable[[str | None], datetime | None],
    stale_hours: int,
) -> None:
    """Flag an enabled file sink whose digest is dead.

    The verdict is ``heartbeat_alarms_fleet.eval_notify_digest``; this reads the
    rows and injects the file probe.

    Reads the FLEET-level ``events.db`` (the sibling of
    ``supervisor-heartbeat.json`` directly under ``fleet_dir`` that
    ``check_wedge_kill_loop`` already reads) for the two kinds described in
    this module's docstring, then probes the supervisor-published path
    read-only (``ndc.probe_digest_file``). Every cannot-tell outcome --
    missing/unreadable events.db, no resolution row, unparseable payload --
    degrades to WARN rather than flipping the exit code on a fleet that may
    simply never have opted in. Callers wrap this in their own try/except so
    an unexpected exception here also degrades to WARN instead of crashing
    the beat before ``save_state`` and report output.
    """
    check = "notify-digest"
    db_path = fleet_dir / "events.db"
    if not db_path.exists():
        report.warn(
            check,
            "no fleet events.db -- the supervisor has never reported a "
            "notify resolution on this host",
        )
        return

    try:
        conn = sqlite3.connect(str(db_path))
    except sqlite3.Error as exc:
        report.warn(check, f"cannot check notify digest: fleet events.db unreadable: {exc}")
        return

    try:
        try:
            table_row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'"
            ).fetchone()
            if table_row is None:
                report.warn(check, "fleet events.db has no events table yet")
                return
            res_row = conn.execute(
                "SELECT ts, payload FROM events WHERE kind = ? ORDER BY id DESC LIMIT 1",
                (NOTIFY_RESOLUTION_EVENT_KIND,),
            ).fetchone()
            stale_rows = conn.execute(
                "SELECT ts FROM events WHERE kind = ?",
                (NOTIFY_DIGEST_STALE_EVENT_KIND,),
            ).fetchall()
        except sqlite3.Error as exc:
            report.warn(check, f"cannot check notify digest: fleet events.db unreadable: {exc}")
            return
    finally:
        conn.close()

    emit(
        report,
        eval_notify_digest(
            res_row,
            [ts for (ts,) in stale_rows],
            now,
            stale_hours,
            lambda path: ndc.probe_digest_file(path, now=now, parse_iso=parse_iso),
        ),
    )
