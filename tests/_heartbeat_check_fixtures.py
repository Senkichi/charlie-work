"""Shared fakes/helpers for the heartbeat-check test modules.

Hoisted verbatim out of ``tests/test_heartbeat_check.py`` (issue #1556,
Track-1 seam split) when that module was split into seam-named siblings
-- the ``tests/_*.py`` hoisted-fixture convention is the sanctioned
import target for shared test helpers (see
``tests/test_zero_cross_test_import_guard.py``).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

from _script_loader import load_script_module


def _load_heartbeat_check() -> ModuleType:
    """Load scripts/heartbeat_check.py as a module without adding scripts to sys.path."""
    path = Path(__file__).parent.parent / "scripts" / "heartbeat_check.py"
    return load_script_module(path, "heartbeat_check")


def _iso(minutes_ago: float = 0.0, *, base: datetime | None = None) -> str:
    """Return an ISO-8601 UTC timestamp `minutes_ago` before `base`.

    `base` defaults to the real wall clock, sampled here, for the many
    wide-margin callers below. Tests with a tight margin against a rounded
    or exact-value assertion (issue #828) should pass a frozen `base` so the
    fixture and the production `now` it is compared against derive from the
    same instant instead of racing an unbounded CI stall.
    """
    reference = base if base is not None else datetime.now(timezone.utc)
    ts = reference - timedelta(minutes=minutes_ago)
    return ts.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _make_repo(hb: ModuleType, tmp_path: Path) -> Any:
    return hb.RepoInfo(
        slug="owner/repo",
        repo_root=tmp_path,
        state_dir=tmp_path / "state",
        config_path=tmp_path / "orchestrator.config.yaml",
    )


def _write_events_db(
    state_dir: Path,
    rows: list[tuple[str, str] | tuple[str, str, str]] | None = None,
) -> Path:
    """Create an events.db next to state.json with the production `events` schema.

    `rows` is a list of either (ts, kind) pairs (defaulting `level` to
    `'info'`, matching the production schema's default) or (ts, kind, level)
    triples for tests that need to seed error/warning-level rows. Mirrors
    `charlie_work.instrumentation`'s `events` table by hand rather than
    importing the package, since heartbeat_check.py deliberately avoids that
    import (see `fleet_dir`'s docstring) and this check must be tested the
    same way.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    db_path = state_dir / "events.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts              TEXT    NOT NULL,
                kind            TEXT    NOT NULL,
                payload         TEXT    NOT NULL,
                repo            TEXT,
                correlation_id  TEXT,
                pr_number       INTEGER,
                issue_number    INTEGER,
                level           TEXT DEFAULT 'info'
            )
            """
        )
        for row in rows or []:
            ts, kind = row[0], row[1]
            level = row[2] if len(row) > 2 else "info"
            conn.execute(
                "INSERT INTO events (ts, kind, payload, level) VALUES (?, ?, '{}', ?)",
                (ts, kind, level),
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _gh_dispatch(monkeypatch: Any, hb: ModuleType, handler: Any) -> None:
    """Install a fake run_gh_json that dispatches to ``handler(args, cwd)``."""

    def fake_run_gh_json(args: list[str], cwd: Path) -> tuple[bool, Any, str]:
        return handler(args, cwd)

    monkeypatch.setattr(hb, "run_gh_json", fake_run_gh_json)


_REAL_PR824_BODY_EXCERPT = (
    "## Summary\n\nFor issue #817: `_filter_fleet_health_transitions` "
    "(`src/charlie_work/fleet_dispatch.py`) is a correct edge-detector for "
    "the fleet health digest's dedup baseline, but its producers only ever "
    "constructed `AttentionEntry` objects for *unhealthy* observations."
)


def _stale_mention_gh_dispatch(
    monkeypatch: Any,
    hb: ModuleType,
    *,
    open_numbers: list[int],
    merged_prs: list[dict[str, Any]],
    captured: list[list[str]] | None = None,
) -> None:
    def handler(args: list[str], cwd: Path) -> tuple[bool, Any, str]:
        if captured is not None:
            captured.append(list(args))
        if args[:2] == ["issue", "list"]:
            return True, [{"number": n} for n in open_numbers], ""
        if args[:2] == ["pr", "list"]:
            return True, merged_prs, ""
        raise AssertionError(f"unexpected gh call in check_stale_open_issue_mentions: {args}")

    _gh_dispatch(monkeypatch, hb, handler)
