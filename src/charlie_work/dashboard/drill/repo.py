"""Repo drill-down: live state from the Now model plus recent history from ``dashboard.db``."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import tzinfo
from pathlib import Path

from ..metrics_flow import drop_reconcile_backfill
from ..now_types import NowModel
from ..rollup_common import _int
from .types import (
    DrillError,
    EscalationRow,
    LoopPassRow,
    MergeRow,
    RepoDrill,
    check_slug,
    history_start,
    local_text,
    open_history,
)

DEFAULT_LIMIT = 20
MAX_LIMIT = 200


def _passes(db: sqlite3.Connection, slug: str, limit: int, tz: tzinfo | None):
    rows = db.execute(
        "SELECT correlation_id, started_at, completed_at, ok, elapsed_seconds, error_count,"
        " merge_count, review_count FROM loop_passes WHERE source = ?"
        " ORDER BY started_at DESC, correlation_id DESC LIMIT ?",
        (slug, limit),
    ).fetchall()
    return tuple(
        LoopPassRow(
            cid,
            started,
            local_text(started, tz),
            done,
            local_text(done, tz),
            None if ok is None else bool(ok),
            elapsed if isinstance(elapsed, (int, float)) else None,
            _int(errors) or 0,  # a TEXT count in a corrupt db is 0, never markup
            _int(merges) or 0,
            _int(reviews) or 0,
        )  # fmt: skip
        for cid, started, done, ok, elapsed, errors, merges, reviews in rows
    )


def merges_for(db: sqlite3.Connection, slug: str, limit: int, tz: tzinfo | None):
    """Newest merges, each issue (or PR) once at its first signal, as History's Merges counts.

    The first-signal dedup runs in SQL: a PR-only row shares its linked issue's key
    (``'i' || issue``) and a PR with no linked issue keeps its own (``'p' || pr``), so
    the fetch returns one row per merged item — not every milestone row the repo has.
    The reconcile-backfill drop still sees all firsts (its burst count is per repo-hour),
    and only after it does ``limit`` apply.
    """
    rows = db.execute(
        "WITH linked AS ("
        " SELECT pr, MAX(issue) AS issue FROM issue_milestones"
        " WHERE source = ? AND issue IS NOT NULL AND pr IS NOT NULL GROUP BY pr"
        "), firsts AS ("
        " SELECT m.ts, m.issue, m.pr, m.event_kind, m.src_id, m.seq,"
        " ROW_NUMBER() OVER ("
        "  PARTITION BY COALESCE('i' || m.issue, 'i' || linked.issue, 'p' || m.pr)"
        "  ORDER BY m.ts, m.src_id, m.seq) AS rn"
        " FROM issue_milestones m LEFT JOIN linked ON linked.pr = m.pr"
        " WHERE m.source = ? AND m.milestone IN ('merged', 'done'))"
        " SELECT ts, issue, pr, event_kind, src_id, seq FROM firsts"
        " WHERE rn = 1 ORDER BY ts, src_id, seq",
        (slug, slug),
    ).fetchall()
    kept = set(drop_reconcile_backfill([(ts, slug, kind) for ts, _, _, kind, _, _ in rows]))
    out = [
        MergeRow(ts, local_text(ts, tz), issue, pr, kind, kind == "reconcile")
        for ts, issue, pr, kind, _sid, _seq in rows
        if (ts, slug, kind) in kept
    ]
    return tuple(reversed(out))[:limit]


def _escalations(db: sqlite3.Connection, slug: str, limit: int, tz: tzinfo | None):
    rows = db.execute(
        "SELECT ts, issue, pr, event_kind, reason, detail FROM escalations"
        " WHERE source = ? AND event_kind != 'unescalate' ORDER BY ts DESC, src_id DESC LIMIT ?",
        (slug, limit),
    ).fetchall()
    return tuple(EscalationRow(ts, local_text(ts, tz), *rest) for ts, *rest in rows)


def repo_drill(
    slug: str,
    model: NowModel | None,
    db_path: Path,
    *,
    tz: tzinfo | None = None,
    limit: int = DEFAULT_LIMIT,
) -> RepoDrill | DrillError:
    """One repo: freshness, stage counts, its Needs-me rows, capacity, last passes, merges.

    ``model`` is the collector's current Now model (None before its first tick). The repo
    must be one the model knows (registry-only); an unreadable ``dashboard.db`` leaves the
    history fields empty and sets ``history_error`` rather than failing the live part.
    """
    if (bad := check_slug(slug)) is not None:
        return bad
    if isinstance(limit, bool) or not isinstance(limit, int) or not 0 < limit <= MAX_LIMIT:
        return DrillError("invalid", f"limit must be 1..{MAX_LIMIT}")
    if model is None:
        return DrillError("unavailable", "the Now model has not been collected yet")
    freshness = next((f for f in model.freshness if f.repo == slug), None)
    if freshness is None:
        return DrillError("not_found", "no such repo in the fleet registry")
    flow = next((r for r in model.repos if r.repo == slug), None)
    cap = model.capacity
    live = dict(
        stages=flow.stages if flow else (),
        needs_me=tuple(i for i in model.needs_me if i.repo == slug),
        workers=next((w for w in cap.workers_by_repo if w.repo == slug), None),
        reviewers=next((w for w in cap.reviewers_by_repo if w.repo == slug), None),
        runners=next((r for r in cap.runners if r.repo == slug), None),
    )
    empty = RepoDrill(slug, freshness, **live, history_error=None, history_from=None,
                      passes=(), merges=(), escalations=())  # fmt: skip
    db, err = open_history(db_path)
    if db is None:
        return replace(empty, history_error=err.message if err else None)
    try:
        return replace(
            empty,
            history_from=history_start(db, slug),
            passes=_passes(db, slug, limit, tz),
            merges=merges_for(db, slug, limit, tz),
            escalations=_escalations(db, slug, limit, tz),
        )
    except sqlite3.Error as exc:
        return replace(empty, history_error=f"history unreadable: {exc}")
    finally:
        db.close()
