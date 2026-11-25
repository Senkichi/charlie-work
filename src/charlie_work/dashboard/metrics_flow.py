"""Flow metrics: merges/day, lead time, stage time, work in progress, queue depth.

Lead time and stage time have two paths (spec "Not yet recorded"):

* approx: reconstructed from ``issue_milestones`` rows the rollup derives from ``dispatch``,
  PR-opened, ``review_dispatch_claim``, ``record_review`` and every merge-evidence kind;
* exact: from ``lifecycle_transition`` / ``ready_observed`` rows (issue #2226).

As soon as the rollup has seen a ``lifecycle_transition`` the exact path takes over from
that first event (``exact_from``); earlier results stay approx. The two never mix inside
one measurement: an interval closing at or after the cutover is exact-only.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .metrics_base import MetricQuery, Sample, Series, SeriesSpec, make_series, parse_ts
from .metrics_coverage import _NON_REPO_SOURCES

EXACT_KINDS = ("lifecycle_transition", "ready_observed")
_VERDICTS = ("verdict_approved", "verdict_request_changes", "verdict_blocked")
_PR_OPENED = ("pr_opened_by_worker", "pr_opened_by_salvage", "pr_open_after_dead_worker")
# stage -> the milestones that enter it, approx path only. A visit ends on ANY
# milestone that is not a start (issue #2473) -- the same ``names - {stage}``
# rule the exact path applies to its own rows: a parked escalation
# (``escalated``, or ``unescalated`` when the escalated row is missing), a dead
# worker's relabel (``session_failed_relabeled``), a re-dispatch, or the merge
# itself all mean the issue left the stage it was in. Naming only the
# happy-path exit let a parked issue's open interval stretch to a far-later
# close (the 1534h phantom samples of #2473).
APPROX_STAGES: dict[str, tuple[str, ...]] = {
    "in_progress": ("dispatched",),
    "pr_open": _PR_OPENED,
    "reviewing": ("review_claimed",),
    "needs_rework": ("verdict_request_changes",),
}
STAGES = tuple(APPROX_STAGES)
# Display names, shared by History titles and the issue drill-down.
STAGE_NAMES = {
    "in_progress": "In progress",
    "pr_open": "PR open",
    "reviewing": "Reviewing",
    "needs_rework": "Needs rework",
}
_LIFECYCLE_KINDS = ("lifecycle_transition", "ready_observed", "reconcile", "dispatch")
_HOURS = 3600.0


@dataclass(frozen=True)
class _Ms:
    ts: str
    source: str
    key: tuple
    name: str
    exact: bool


def exact_cutover(db: sqlite3.Connection) -> str | None:
    """Earliest ``lifecycle_transition`` the rollup has seen (None: not instrumented yet)."""
    row = db.execute(
        "SELECT MIN(first_ts) FROM coverage WHERE kind = 'lifecycle_transition'"
    ).fetchone()
    return row[0] if row and row[0] else None


def _load(db: sqlite3.Connection, q: MetricQuery) -> list[_Ms]:
    """Milestones up to the window end (a lead time starts before its window), keyed per issue.

    A PR-only row (review claim, verdict) is attached to its issue through any row that
    names both; otherwise it keeps a ``("pr", n)`` key of its own.
    """
    rows = db.execute(
        "SELECT ts, source, issue, pr, milestone, event_kind FROM issue_milestones"
        " WHERE ts < ? ORDER BY ts, src_id, seq",
        (q.end_iso,),
    ).fetchall()
    pr_issue = {(s, pr): i for _, s, i, pr, _, _ in rows if i is not None and pr is not None}
    out = []
    for ts, source, issue, pr, name, kind in rows:
        if issue is None and pr is not None:
            issue = pr_issue.get((source, pr))
        key = ("issue", issue) if issue is not None else ("pr", pr)
        out.append(_Ms(ts, source, key, name, kind in EXACT_KINDS))
    return out


def scan_intervals(
    events: Sequence[tuple[str, str]],
    starts: Sequence[str],
    ends: Sequence[str],
    *,
    restart: bool = True,
) -> tuple[list[tuple[str, str]], str | None]:
    """Closed ``(open_ts, close_ts)`` intervals plus the still-open tail's start, if any.

    ``restart`` (the default) makes a start arriving while a visit is open move the
    open to it: the LAST start before a close wins, so a re-dispatch after a parked
    or dead first attempt does not inherit that dead time (issue #2473). Lead time
    passes ``restart=False`` -- its span is first-signal -> done, and parked time
    is genuinely part of it.
    """
    out, opened = [], None
    for ts, name in events:
        if name in starts and (opened is None or restart):
            opened = ts
        elif opened is not None and name in ends:
            out.append((opened, ts))
            opened = None
    return out, opened


# Lead time span: (start milestones, end milestones) per path. The one place the names live.
_LEAD_APPROX = (("dispatched",), ("merged",))
_LEAD_EXACT = (("ready", "ready_observed"), ("done",))


def lead_span(events: Sequence[tuple[str, str]], *, exact: bool) -> tuple[str, str] | None:
    """The first closed first-signal -> done ``(start_ts, end_ts)``, or None (parked time counts)."""
    starts, ends = _LEAD_EXACT if exact else _LEAD_APPROX
    return next(iter(scan_intervals(events, starts, ends, restart=False)[0]), None)


def _hours(a: str, b: str) -> float:
    return (parse_ts(b) - parse_ts(a)).total_seconds() / _HOURS


def _grouped(ms: Sequence[_Ms], exact: bool) -> dict[tuple, list[tuple[str, str]]]:
    out: dict[tuple, list[tuple[str, str]]] = defaultdict(list)
    for m in ms:
        if m.exact == exact:
            out[(m.source, m.key)].append((m.ts, m.name))
    return out


def _timing_series(
    db: sqlite3.Connection,
    q: MetricQuery,
    spec: SeriesSpec,
    samples: list[Sample],
    cut: str | None,
) -> Series:
    approx = cut is None or parse_ts(cut) > q.start
    return make_series(db, q, spec, samples, samples, how="median", approx=approx, exact_from=cut)


def lead_time(db: sqlite3.Connection, q: MetricQuery) -> Series:
    """Ready -> Done in hours (median per bucket, attributed to the Done time)."""
    cut = exact_cutover(db)
    ms = _load(db, q)
    samples: list[Sample] = []
    for (source, _), evs in _grouped(ms, False).items():
        if span := lead_span(evs, exact=False):
            start, end = span
            if cut is None or end < cut:
                samples.append((end, source, _hours(start, end)))
    for (source, _), evs in _grouped(ms, True).items():
        if span := lead_span(evs, exact=True):
            samples.append((span[1], source, _hours(*span)))
    spec = SeriesSpec(
        "lead_time", "Lead time", "hours", "duration", _LIFECYCLE_KINDS, any_kind=True
    )
    return _timing_series(db, q, spec, samples, cut)


def stage_time(db: sqlite3.Connection, q: MetricQuery, stage: str) -> Series:
    """Hours an issue spent in ``stage``, summed over its visits, at its last departure."""
    if stage not in APPROX_STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    cut = exact_cutover(db)
    ms = _load(db, q)
    samples: list[Sample] = []
    starts = APPROX_STAGES[stage]
    # Ends are every approx milestone that is not a start of this stage: the
    # issue has left the stage whenever it records any other milestone.
    ends = tuple({m.name for m in ms if not m.exact} - set(starts) - {"ready_observed"})
    for (source, _), evs in _grouped(ms, False).items():
        spans, _ = scan_intervals(evs, starts, ends)
        if spans and (cut is None or spans[-1][1] < cut):
            samples.append((spans[-1][1], source, sum(_hours(a, b) for a, b in spans)))
    exact_names = {m.name for m in ms if m.exact} - {stage, "ready_observed"}
    for (source, _), evs in _grouped(ms, True).items():
        # lifecycle_transition rows name the stage entered, so a repeated start is a
        # data anomaly rather than a re-dispatch: keep the pre-#2473 first-start rule.
        spans, _ = scan_intervals(evs, (stage,), tuple(exact_names), restart=False)
        if spans:
            samples.append((spans[-1][1], source, sum(_hours(a, b) for a, b in spans)))
    spec = SeriesSpec(
        f"stage_time.{stage}", f"Stage time: {STAGE_NAMES.get(stage, stage)}", "hours", "duration", _LIFECYCLE_KINDS,
        any_kind=True,
    )  # fmt: skip
    return _timing_series(db, q, spec, samples, cut)


MERGE_KINDS = (
    "merge_succeeded",
    "dispatch_merged_pr_references_closed",
    "finalize_externally_merged",
    "reconcile",
    "lifecycle_transition",
)


# The milestones a merge reaches: the Lead time end of each path (approx, exact).
_MERGE_MILESTONES = (_LEAD_APPROX[1][0], _LEAD_EXACT[1][0])


def pr_links_last_row(db: sqlite3.Connection, end_iso: str) -> dict[tuple[str, int], int]:
    """``(source, pr) -> issue`` for History: the last row of an unordered SELECT wins."""
    return {
        (s, pr): i
        for s, i, pr in db.execute(
            "SELECT source, issue, pr FROM issue_milestones"
            " WHERE issue IS NOT NULL AND pr IS NOT NULL AND ts < ?",
            (end_iso,),
        )
    }


def pr_links_max_issue(db: sqlite3.Connection, source: str) -> dict[tuple[str, int], int]:
    """``(source, pr) -> issue`` for the repo drill-down: the highest issue naming the PR."""
    return {
        (s, pr): i
        for s, pr, i in db.execute(
            "SELECT source, pr, MAX(issue) FROM issue_milestones"
            " WHERE source = ? AND issue IS NOT NULL AND pr IS NOT NULL GROUP BY source, pr",
            (source,),
        )
    }


def first_merges(
    db: sqlite3.Connection,
    link: Mapping[tuple[str, int], int],
    *,
    end_iso: str | None = None,
    source: str | None = None,
) -> list[tuple[str, str, int | None, int | None, str]]:
    """Each merged issue (or PR with no linked issue) once, at its first merged/done signal.

    Rows are ``(ts, source, issue, pr, evidence kind)`` in signal order, ``issue`` being
    the row's own (None for a PR-only row). A PR-only row is attached to its issue through
    ``link`` (``(source, pr) -> issue``), an explicit input: History and the drill-down
    build it by different rules that disagree on a PR linked to several issues, on purpose,
    pending an operator decision. ``end_iso`` bounds the signals (exclusive) and ``source``
    restricts to one repo. This is the one first-merge dedup behind History, the repo
    drill-down and Now.
    """
    where, args = "", list(_MERGE_MILESTONES)
    if end_iso is not None:
        where += " AND ts < ?"
        args.append(end_iso)
    if source is not None:
        where += " AND source = ?"
        args.append(source)
    rows = db.execute(
        "SELECT ts, source, issue, pr, event_kind FROM issue_milestones"
        f" WHERE milestone IN (?, ?){where} ORDER BY ts, src_id, seq",
        args,
    ).fetchall()
    seen: set = set()
    firsts = []
    for ts, src, issue, pr, kind in rows:
        linked = issue if issue is not None else link.get((src, pr))
        key = (src, linked) if linked is not None else (src, "pr", pr)
        if key not in seen:
            seen.add(key)
            firsts.append((ts, src, issue, pr, kind))
    return firsts


def merges_per_day(db: sqlite3.Connection, q: MetricQuery) -> Series:
    """Issues reaching Done per bucket, each counted once at its first Done signal.

    Every merge-evidence kind feeds the same ``merged``/``done`` milestones (see
    ``rollup_flow_handlers``), the same set lead time reads. A PR-only row is attached to
    its issue through any row naming both (as ``_load`` does), so one merge reported by
    several kinds counts once per ``(repo, issue or PR)``.
    """
    firsts = [
        (ts, source, kind)
        for ts, source, _, _, kind in first_merges(
            db, pr_links_last_row(db, q.end_iso), end_iso=q.end_iso
        )
    ]
    firsts = drop_reconcile_backfill(firsts)
    samples: list[Sample] = [(ts, source, 1.0) for ts, source, _ in firsts]
    # A reconcile-detected merge is stamped when the reconciler noticed it, not when it happened.
    approx = any(kind == "reconcile" for _, _, kind in firsts)
    spec = SeriesSpec("merges_per_day", "Merges", "merges", "count", MERGE_KINDS, any_kind=True)
    return make_series(db, q, spec, samples, samples, how="sum", approx=approx)


BACKFILL_BURST = 20  # reconcile-detected merges in one repo-hour: a catch-up pass, not flow


def drop_reconcile_backfill(
    firsts: Sequence[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """Drop merges first seen by ``reconcile`` in a repo-hour holding ``BACKFILL_BURST`` or more.

    Onboarding a repo (or the reconciler's first pass) reports every historical merged PR at
    once, stamped with the detection time (observed: 308, 158 and 72 in a single hour against
    5-50 a day normally). Those are state bookkeeping, not merges that happened that hour.
    """
    burst: dict[tuple[str, str], int] = defaultdict(int)
    for ts, source, kind in firsts:
        if kind == "reconcile":
            burst[(source, ts[:13])] += 1
    return [
        f
        for f in firsts
        if not (f[2] == "reconcile" and burst[(f[1], f[0][:13])] >= BACKFILL_BURST)
    ]


def pass_gauge(
    db: sqlite3.Connection, q: MetricQuery, spec: SeriesSpec, col: str, fleet_col: str | None
) -> Series:
    """Mean of a ``pass_samples`` column per bucket; ``fleet_col`` feeds the combined line."""
    cols = f"ts, source, {col}" + (f", {fleet_col}" if fleet_col else "")
    # Non-repo sources are excluded so a fleet/global or supervisor-checkout row
    # can never surface as a phantom repo line; for the orchestrator source this
    # is a second fence -- ingest already restricts it to deploys (issue #2475).
    rows = db.execute(
        f"SELECT {cols} FROM pass_samples WHERE ts >= ? AND ts < ? AND source NOT IN (?, ?)",
        (q.start_iso, q.end_iso, *_NON_REPO_SOURCES),
    ).fetchall()
    per_repo = [(r[0], r[1], float(r[2])) for r in rows if r[2] is not None]
    combined = [(r[0], r[1], float(r[3])) for r in rows if fleet_col and r[3] is not None]
    return make_series(db, q, spec, combined or per_repo, per_repo, how="mean")


def work_in_progress(db: sqlite3.Connection, q: MetricQuery) -> Series:
    """Live workers: per repo its own count, all-repos the fleet-wide count."""
    spec = SeriesSpec(
        "wip", "Work in progress", "workers", "gauge", ("dispatch",), check_instrumented=True
    )
    return pass_gauge(db, q, spec, "live_sessions", "fleet_live_sessions")


def queue_depth(db: sqlite3.Connection, q: MetricQuery) -> Series:
    """Dispatchable issues waiting; all-repos is the sum of each repo's bucket mean."""
    spec = SeriesSpec(
        "queue_depth", "Queue depth", "issues", "gauge", ("dispatch",),
        check_instrumented=True, combine="repo_sum",
    )  # fmt: skip
    return pass_gauge(db, q, spec, "dispatchable", None)
