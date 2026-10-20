"""Issue and PR drill-downs: one lifecycle timeline from ``dashboard.db`` plus current state.

The two views share one reader. ``issue_drill`` follows an issue and every PR the history
links to it; ``pr_drill`` follows a PR and the issue the history (or, failing that, the
snapshot) maps it to, so a PR page also shows the dispatch that preceded the PR.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, tzinfo
from pathlib import Path

from ...config import LabelConfig
from ..metrics_flow import EXACT_KINDS
from ..now_access import dict_list, label_set, pos_int
from ..sources import SnapshotRead
from .stages import stage_spans, stage_times
from .types import (
    CurrentState,
    DrillError,
    IssueDrill,
    TimelineEntry,
    check_number,
    check_slug,
    history_start,
    local_text,
    open_history,
)

_ROW_LIMIT = 1000
_LABELS = {
    "dispatched": "Dispatched to a worker",
    "rework_dispatched": "Rework dispatched",
    "pr_opened_by_worker": "PR opened by the worker",
    "pr_opened_by_salvage": "PR opened by salvage",
    "pr_open_after_dead_worker": "Worker died; its PR moved to PR open",
    "review_claimed": "Review claimed",
    "verdict_approved": "Review verdict: approved",
    "verdict_request_changes": "Review verdict: changes requested",
    "verdict_blocked": "Review verdict: blocked",
    "merged": "Merged",
    "ready_observed": "Observed ready",
}
_REVIEW = frozenset(
    {"review_claimed", "verdict_approved", "verdict_request_changes", "verdict_blocked"}
)
# Shown from the escalations table (it carries the reason); the milestone only feeds stages.
_ESCALATION_MILESTONES = frozenset({"escalated", "unescalated"})


def _label(milestone: str) -> str:
    return _LABELS.get(milestone) or f"Entered {milestone.replace('_', ' ')}"


def _current(
    snapshot: SnapshotRead | None, labels: LabelConfig, issue: int | None, pr: int | None
) -> CurrentState | None:
    data = snapshot.data if snapshot is not None else None
    if data is None:
        return None
    stages = (
        (labels.queued, "Queued"),
        (labels.in_progress, "In progress"),
        (labels.pr_open, "PR open"),
        (labels.reviewing, "Reviewing"),
        (labels.needs_rework, "Needs rework"),
        (labels.operator_queue, "Operator queue"),
        (labels.human_needed, "Human needed"),
    )
    found = next((i for i in dict_list(data, "issues") if pos_int(i.get("number")) == issue), None)
    pr_row = next(
        (
            p
            for p in dict_list(data, "prs")
            if (pr is not None and pos_int(p.get("number")) == pr)
            or (pr is None and issue is not None and pos_int(p.get("issue_number")) == issue)
        ),
        None,
    )
    if found is None and pr_row is None:
        return None
    have = label_set(found) if found else set()
    title = found.get("title") if found else None
    decision = pr_row.get("reviewDecision") if pr_row else None
    draft = pr_row.get("is_draft") if pr_row else None
    return CurrentState(
        stage=next((name for label, name in stages if label in have), None),
        title=title if isinstance(title, str) else None,
        labels=tuple(sorted(have)),
        pr=pos_int(pr_row.get("number")) if pr_row else None,
        review_decision=decision if isinstance(decision, str) and decision else None,
        is_draft=draft if isinstance(draft, bool) else None,
        as_of=snapshot.written_at if snapshot is not None else None,
    )


def _issue_for_pr(
    db: sqlite3.Connection, repo: str, pr: int, snapshot: SnapshotRead | None
) -> int | None:
    row = db.execute(
        "SELECT issue FROM issue_milestones WHERE source = ? AND pr = ? AND issue IS NOT NULL"
        " ORDER BY ts, src_id, seq LIMIT 1",
        (repo, pr),
    ).fetchone()
    if row:
        return row[0]
    data = snapshot.data if snapshot is not None else None
    for p in dict_list(data or {}, "prs"):
        if pos_int(p.get("number")) == pr:
            return pos_int(p.get("issue_number"))
    return None


def _prs_for_issue(db: sqlite3.Connection, repo: str, issue: int) -> tuple[int, ...]:
    rows = db.execute(
        "SELECT DISTINCT pr FROM issue_milestones WHERE source = ? AND issue = ?"
        " AND pr IS NOT NULL ORDER BY pr",
        (repo, issue),
    ).fetchall()
    return tuple(r[0] for r in rows)


def _scope(issue: int | None, prs: tuple[int, ...], strict_pr: bool) -> tuple[str, list]:
    """WHERE fragment (after ``source = ?``) selecting an item's rows in PR-carrying tables.

    An issue view takes every row naming the issue or one of its PRs. A PR view takes rows
    naming the PR plus the issue's PR-less rows (its dispatch), never a sibling PR's.
    """
    clauses: list[str] = []
    args: list = []
    if prs:
        clauses.append(f"pr IN ({','.join('?' * len(prs))})")
        args += prs
    if issue is not None:
        clauses.append("(issue = ? AND pr IS NULL)" if strict_pr else "issue = ?")
        args.append(issue)
    if not clauses:
        return "0", []
    return "(" + " OR ".join(clauses) + ")", args


def _entry(ts, tz, category, label, detail, kind, approx, issue, pr) -> TimelineEntry:
    return TimelineEntry(ts, local_text(ts, tz), category, label, detail, kind, approx, issue, pr)


def _entries(db: sqlite3.Connection, repo: str, scope: str, args: list, issue, tz):
    """``(timeline entries, milestone rows)``; the rows also feed the stage computation."""
    tail = f"ORDER BY ts, src_id, seq LIMIT {_ROW_LIMIT}"
    ms = db.execute(
        "SELECT ts, issue, pr, milestone, event_kind, approx FROM issue_milestones"
        f" WHERE source = ? AND {scope} {tail}",
        [repo, *args],
    ).fetchall()
    out = [
        _entry(ts, tz, "review" if name in _REVIEW else "lifecycle", _label(name), None,
               kind, bool(approx), i, pr)
        for ts, i, pr, name, kind, approx in ms
        if name not in _ESCALATION_MILESTONES
    ]  # fmt: skip
    for ts, i, pr, kind, reason, detail in db.execute(
        f"SELECT ts, issue, pr, event_kind, reason, detail FROM escalations"
        f" WHERE source = ? AND {scope} {tail}",
        [repo, *args],
    ):
        label = "Escalation cleared" if kind == "unescalate" else "Escalated"
        text = " - ".join(x for x in (reason, detail) if x) or None
        out.append(_entry(ts, tz, "escalation", label, text, kind, False, i, pr))
    for ts, i, pr, reason, detail in db.execute(
        f"SELECT ts, issue, pr, reason, detail FROM verdict_missed"
        f" WHERE source = ? AND {scope} {tail}",
        [repo, *args],
    ):
        # `reason` is the stable token; `detail` is the human-readable message (issue #2476).
        out.append(
            _entry(ts, tz, "review", "Review verdict missed", detail or reason,
                   "review_verdict_missed", False, i, pr)
        )  # fmt: skip
    if issue is not None:  # worker exits carry an issue number only (no PR column)
        for ts, kind, health in db.execute(
            "SELECT ts, failure_kind, worker_health FROM worker_exits"
            f" WHERE source = ? AND issue = ? {tail}",
            (repo, issue),
        ):
            text = ", ".join(x for x in (kind, health) if x) or None
            out.append(
                _entry(ts, tz, "worker", "Worker exited", text, "session_exited",
                       False, issue, None)
            )  # fmt: skip
    out.sort(key=lambda e: e.ts)  # stable: same-second rows keep their derivation order
    return out, ms


def _drill(
    kind: str,
    repo: str,
    number: int,
    db_path: Path,
    snapshot: SnapshotRead | None,
    now: datetime,
    tz: tzinfo | None,
    labels: LabelConfig | None,
) -> IssueDrill | DrillError:
    if (bad := check_slug(repo) or check_number(number)) is not None:
        return bad
    if now.tzinfo is None:
        return DrillError("invalid", "now must be timezone-aware")
    db, err = open_history(db_path)
    if db is None:
        return err or DrillError("unavailable", "history unavailable")
    try:
        if kind == "issue":
            issue, prs = number, _prs_for_issue(db, repo, number)
        else:
            issue, prs = _issue_for_pr(db, repo, number, snapshot), (number,)
        scope, args = _scope(issue, prs, strict_pr=kind == "pr")
        entries, ms = _entries(db, repo, scope, args, issue, tz)
        milestones = [(ts, name, k in EXACT_KINDS) for ts, _i, _p, name, k, _a in ms]
        stages, lead, approx = stage_times(milestones, now)
        current = _current(
            snapshot, labels or LabelConfig(), issue, number if kind == "pr" else None
        )
        return IssueDrill(
            kind=kind,
            repo=repo,
            number=number,
            issue=issue,
            prs=prs,
            current=current,
            timeline=tuple(entries),
            stage_times=stages,
            lead_seconds=lead,
            approx=approx,
            history_from=history_start(db, repo),
            known=bool(entries) or current is not None,
            spans=stage_spans(milestones, now),
        )
    except sqlite3.Error as exc:
        return DrillError("unavailable", f"history unreadable: {exc}")
    finally:
        db.close()


def issue_drill(
    repo: str,
    number: int,
    db_path: Path,
    snapshot: SnapshotRead | None,
    now: datetime,
    *,
    tz: tzinfo | None = None,
    labels: LabelConfig | None = None,
) -> IssueDrill | DrillError:
    """Lifecycle timeline, stage times and current state for ``repo`` issue ``number``."""
    return _drill("issue", repo, number, db_path, snapshot, now, tz, labels)


def pr_drill(
    repo: str,
    number: int,
    db_path: Path,
    snapshot: SnapshotRead | None,
    now: datetime,
    *,
    tz: tzinfo | None = None,
    labels: LabelConfig | None = None,
) -> IssueDrill | DrillError:
    """Same, keyed by PR ``number``; ``issue`` on the result is the issue it maps to."""
    return _drill("pr", repo, number, db_path, snapshot, now, tz, labels)
