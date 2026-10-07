"""Pure heartbeat alarm evaluators -- the single source for alarm verdicts.

This module: the ``Finding`` value, ``emit`` adapter, timestamp helpers and the
per-repo events.db evaluators. Host/fleet-level evaluators (loop/log freshness,
supervisor heartbeat, wedge loop, notify digest) are in
``charlie_work.heartbeat_alarms_fleet``.

``scripts/heartbeat_check.py`` (the scheduled beat) and the fleet dashboard
both need the same verdicts from the same already-loaded data. This leaf holds
the decision logic only: every ``eval_*`` takes rows/values the caller already
read, with ``now`` / ``baseline`` INJECTED, and returns ``Finding`` values.
No I/O, no clock reads, no ``Report``. Readers (sqlite, mtime stat, JSON files)
stay in their callers; the heartbeat adapts a ``Finding`` back onto its
``Report`` with ``emit`` so the printed lines (and suppression matching on the
check name) are byte-identical to before the extraction.

Stdlib only -- no ``charlie_work`` / ``ci_fleet`` imports -- so the heartbeat
can load it behind its guarded import and stay usable when the rest of the
package is broken (``scripts/README.md``).

Timestamp convention: ``ts`` strings are ISO ``T``/``Z`` text and are compared
here in Python on parsed datetimes, never in SQL (SQLite's space-separated
``datetime()`` strings sort differently and mis-compare silently). An
unparseable ``ts`` fails toward visibility (reported), not silence.

``parse_iso`` is the single copy for every verdict and the heartbeat scripts.
Behavior change vs the pre-extraction script copy: it also swallows
``AttributeError`` (a non-string ``ts``, e.g. an int from a malformed row,
returns ``None`` instead of raising).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Protocol

Severity = Literal["ok", "warn", "anomaly"]


@dataclass(frozen=True)
class Finding:
    """One alarm verdict. ``check`` is the full name (``"<base> <repo-slug>"``
    for per-repo checks) so suppression matching is unchanged; ``facts`` is
    the OK-line payload / trailing ``(facts)`` of a non-ok detail."""

    check: str
    repo: str | None
    severity: Severity
    detail: str
    facts: str = ""


class ReportSink(Protocol):
    def ok(self, check: str, facts: str) -> None: ...
    def warn(self, check: str, detail: str) -> None: ...
    def anom(self, check: str, detail: str) -> None: ...


def emit(report: ReportSink, finding: Finding) -> None:
    """Adapt a ``Finding`` onto the heartbeat ``Report`` (identical line text)."""
    if finding.severity == "ok":
        report.ok(finding.check, finding.facts or finding.detail)
    elif finding.severity == "warn":
        report.warn(finding.check, finding.detail)
    else:
        report.anom(finding.check, finding.detail)


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def new_event_rows(rows: Iterable[Sequence[Any]], baseline: datetime) -> list[Sequence[Any]]:
    """Rows (``ts`` first) strictly after ``baseline``; unparseable ``ts`` is kept."""
    out: list[Sequence[Any]] = []
    for row in rows:
        ts_dt = parse_iso(row[0])
        if ts_dt is None or ts_dt > baseline:
            out.append(row)
    return out


def ok_finding(check: str, repo: str | None, facts: str) -> Finding:
    return Finding(check, repo, "ok", "", facts)


def anomaly_finding(check: str, repo: str | None, detail: str) -> Finding:
    return Finding(check, repo, "anomaly", detail)


def eval_error_events(slug: str, rows: Sequence[tuple[str, str]], baseline: datetime) -> Finding:
    """``rows`` = every ``(ts, kind)`` with ``level='error'`` (persisted level)."""
    check = f"error-events {slug}"
    new = [f"{kind}@{ts}" for ts, kind in new_event_rows(rows, baseline)]
    facts = f"error_rows={len(rows)} new_since_last_beat={len(new)}"
    if new:
        return Finding(
            check,
            slug,
            "anomaly",
            f"new error-level event(s) since last beat: {new} ({facts})",
            facts,
        )
    return ok_finding(check, slug, facts)


def eval_warning_events(
    slug: str,
    rows: Sequence[tuple[str, str]],
    baseline: datetime,
    expected_ops: frozenset[str] | set[str],
) -> list[Finding]:
    """Warnings never escalate to anomaly; ``expected_ops`` kinds are summarized."""
    check = f"warning-events {slug}"
    detail_rows: list[str] = []
    counts: dict[str, int] = {}
    for ts, kind in new_event_rows(rows, baseline):
        if kind in expected_ops:
            counts[kind] = counts.get(kind, 0) + 1
        else:
            detail_rows.append(f"{kind}@{ts}")
    total_new = len(detail_rows) + sum(counts.values())
    facts = f"warning_rows={len(rows)} new_since_last_beat={total_new}"
    findings: list[Finding] = []
    if detail_rows:
        findings.append(
            Finding(
                check,
                slug,
                "warn",
                f"new warning-level event(s) since last beat: {detail_rows} ({facts})",
                facts,
            )
        )
    if counts:
        # Sorted by kind name, never insertion order: byte-identical reruns.
        counts_str = ", ".join(f"{kind}={counts[kind]}" for kind in sorted(counts))
        findings.append(
            Finding(
                check,
                slug,
                "warn",
                f"{sum(counts.values())} routine operational warnings ({counts_str}) ({facts})",
                facts,
            )
        )
    return findings or [ok_finding(check, slug, facts)]


def eval_kind_events(
    check: str,
    slug: str | None,
    rows: Sequence[tuple[str, ...]],
    baseline: datetime,
    severity: Literal["warn", "anomaly"],
    *,
    kind: str,
    rows_label: str,
) -> Finding:
    """Generic single-kind "new since last beat" check (``draft_pr_blocked``)."""
    new = new_event_rows(rows, baseline)
    facts = f"{rows_label}={len(rows)} new_since_last_beat={len(new)}"
    if new:
        return Finding(
            check, slug, severity, f"{kind} since last beat: {len(new)} event(s) ({facts})", facts
        )
    return ok_finding(check, slug, facts)


def eval_draft_pr_blocked(slug: str, rows: Sequence[tuple[str]], baseline: datetime) -> Finding:
    return eval_kind_events(
        f"draft-pr-blocked-events {slug}",
        slug,
        rows,
        baseline,
        "warn",
        kind="draft_pr_blocked",
        rows_label="blocked_rows",
    )


def eval_infra_blocked(
    slug: str,
    blocked_rows: Sequence[tuple[str, ...]],
    escalated_rows: Sequence[tuple[str, ...]],
    baseline: datetime,
) -> Finding:
    """``escalated`` (``infra_blocked_escalated``) is an anomaly; bare blocks warn."""
    check = f"infra-blocked-events {slug}"
    new_blocked = new_event_rows(blocked_rows, baseline)
    new_escalated = [row[0] for row in new_event_rows(escalated_rows, baseline)]
    facts = f"blocked_rows={len(blocked_rows)} escalated_rows={len(escalated_rows)}"
    if new_escalated:
        return Finding(
            check,
            slug,
            "anomaly",
            f"infra_blocked_escalated since last beat: {new_escalated} ({facts})",
            facts,
        )
    if new_blocked:
        return Finding(
            check,
            slug,
            "warn",
            f"check_infra_blocked since last beat: {len(new_blocked)} event(s) ({facts})",
            facts,
        )
    return ok_finding(check, slug, facts)


def _payload_dict(payload_json: Any) -> dict[str, Any] | None:
    try:
        payload = json.loads(payload_json)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def eval_ci_headroom_unavailable(
    slug: str, rows: Sequence[tuple[str, str]], baseline: datetime
) -> Finding:
    """``rows`` = ``(ts, payload_json)`` of ``ci_headroom_unavailable``."""
    check = f"ci-headroom-unavailable-events {slug}"
    new = new_event_rows(rows, baseline)
    reasons: dict[str, int] = {}
    for _ts, payload_json in new:
        payload = _payload_dict(payload_json)
        reason = str(payload.get("reason", "unknown")) if payload is not None else "unknown"
        reasons[reason] = reasons.get(reason, 0) + 1
    facts = f"unavailable_rows={len(rows)} new_since_last_beat={len(new)}"
    if new:
        return Finding(
            check,
            slug,
            "warn",
            f"ci_headroom_unavailable since last beat: {len(new)} event(s), "
            f"reasons={reasons} ({facts})",
            facts,
        )
    return ok_finding(check, slug, facts)


def eval_mergequeue_stalled(
    slug: str, rows: Sequence[tuple[str, str]], baseline: datetime
) -> Finding:
    """``rows`` = ``(ts, payload_json)`` of ``mergequeue_stalled`` (one per queue episode)."""
    check = f"mergequeue_stalled {slug}"
    new = new_event_rows(rows, baseline)
    prs: set[int] = set()
    for _ts, payload_json in new:
        payload = _payload_dict(payload_json)
        number = payload.get("pr_number") if payload is not None else None
        if isinstance(number, int) and not isinstance(number, bool):
            prs.add(number)
    facts = f"stalled_rows={len(rows)} new_since_last_beat={len(new)}"
    if not new:
        return ok_finding(check, slug, facts)
    detail = f"{len(new)} PR(s) queued past the stall threshold since last beat"
    if prs:
        detail += f": PR(s) {sorted(prs)}"
    return Finding(check, slug, "warn", f"{detail} ({facts})", facts)


def eval_local_lane_stalled(
    slug: str, rows: Sequence[tuple[str, str]], baseline: datetime
) -> Finding:
    """``rows`` = ``(ts, payload_json)`` of ``local_lane_kill_switch_stalled``.

    Forwards the event detail only; it does not re-evaluate config.
    """
    check = f"local_lane_kill_switch_stalled {slug}"
    new = new_event_rows(rows, baseline)
    switches: set[str] = set()
    stranded: set[int] = set()
    for _ts, payload_json in new:
        payload = _payload_dict(payload_json)
        if payload is None:
            continue
        switch = payload.get("switch")
        if switch:
            switches.add(str(switch))
        for n in payload.get("issue_numbers") or ():
            if isinstance(n, int) and not isinstance(n, bool):
                stranded.add(n)
    facts = f"stalled_rows={len(rows)} new_since_last_beat={len(new)}"
    if not new:
        return ok_finding(check, slug, facts)
    detail = f"{len(new)} event(s) since last beat"
    if switches:
        detail += f"; disabled switch(es): {sorted(switches)}"
    if stranded:
        detail += f"; stranded issue(s): {sorted(stranded)}"
    return Finding(check, slug, "anomaly", f"{detail} ({facts})", facts)
