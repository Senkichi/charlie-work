"""Issue and PR drill-down page: current state, stage lanes, lead time, full timeline.

The lane chart and the stage-time column beside it are drawn from the same visits
(``IssueDrill.spans``), and the takeaway is computed from those bands. Reconstructed (approx.)
times are drawn hollow and dashed and say so in words; the exact path replaces them once
``lifecycle_transition`` events exist (issue #2226).
"""

from __future__ import annotations

from datetime import datetime, tzinfo

from ..charts.lanes import Band, Lane, lane_chart
from ..drill import IssueDrill, TimelineEntry
from ..metrics_flow import STAGE_NAMES, STAGES
from ..read_model import ModelState
from ..timeutil import parse_ts
from .drill_shell import page, section, table, ts_tag
from .now_fmt import age, esc, issue_url, link, local_time, pr_url, repo_url, short_repo

_CATEGORY = {"lifecycle": "Lifecycle", "review": "Review", "escalation": "Escalation",
             "worker": "Worker"}  # fmt: skip


def _lanes(d: IssueDrill) -> tuple[Lane, ...]:
    return tuple(
        Lane(
            STAGE_NAMES.get(stage, stage.replace("_", " ")),
            tuple(
                Band(parse_ts(s.start), parse_ts(s.end), s.open_now)
                for s in d.spans
                if s.stage == stage
            ),
        )
        for stage in STAGES
    )


def _window(d: IssueDrill, now: datetime) -> tuple[datetime, datetime] | None:
    moments = [parse_ts(s.start) for s in d.spans] + [parse_ts(s.end) for s in d.spans]
    if not moments:
        return None
    end = max(moments)
    if any(s.open_now for s in d.spans):
        end = max(end, now)
    return min(moments), end


def _current(d: IssueDrill) -> str:
    c = d.current
    if c is None:
        return '<p class="calm-sm">Not in the current snapshot (closed, or not ready).</p>'
    rows = [
        ("Stage", esc(c.stage or "no workflow label")),
        ("Title", esc(c.title or "—")),
        ("Labels", esc(", ".join(c.labels)) or "—"),
    ]
    if c.pr is not None:
        draft = " (draft)" if c.is_draft else ""
        rows.append(("PR", link(pr_url(d.repo, c.pr), f"PR #{c.pr}") + esc(draft)))
    if c.review_decision:
        rows.append(("Review decision", esc(c.review_decision)))
    if c.as_of is not None:
        rows.append(("Snapshot", f"{local_time(c.as_of)} (local)"))
    cells = "".join(f'<tr><th scope="row">{k}</th><td>{v}</td></tr>' for k, v in rows)
    return f'<table class="dkv"><caption class="sr">Current state</caption>{cells}</table>'


def _links(d: IssueDrill) -> str:
    parts = []
    if d.issue is not None and (d.kind == "pr" or d.number != d.issue):
        parts.append("issue " + link(issue_url(d.repo, d.issue), f"#{d.issue}"))
    prs = [p for p in d.prs if not (d.kind == "pr" and p == d.number)]
    if prs:
        parts.append("PRs " + ", ".join(link(pr_url(d.repo, p), f"#{p}") for p in prs))
    return " · ".join(parts)


def _stage_rows(d: IssueDrill) -> list[str]:
    rows = []
    for s in sorted(d.stage_times, key=lambda s: (-s.seconds, s.stage)):
        flag = " · open" if s.open_now else ""
        rows.append(
            f'<tr><th scope="row">{esc(STAGE_NAMES.get(s.stage, s.stage))}</th>'
            f'<td class="num">{esc(age(s.seconds))}</td><td class="num">{s.visits}</td>'
            f"<td>{'approx.' if d.approx else 'exact'}{flag}</td></tr>"
        )
    return rows


def _event_row(d: IssueDrill, e: TimelineEntry, tz: tzinfo | None) -> str:
    item = " ".join(
        x
        for x in (
            link(issue_url(d.repo, e.issue), f"#{e.issue}") if e.issue else "",
            link(pr_url(d.repo, e.pr), f"PR #{e.pr}") if e.pr else "",
        )
        if x
    )
    approx = ' <span class="aprx">approx.</span>' if e.approx else ""
    tone = (
        ' class="tone-warn"' if e.category == "escalation" and e.event_kind != "unescalate" else ""
    )
    return (
        f"<tr{tone}><td>{ts_tag(e.ts, tz)}</td><td>{esc(_CATEGORY.get(e.category, e.category))}</td>"
        f"<td>{esc(e.label)}{approx}</td><td>{esc(e.detail or '')}</td><td>{item}</td>"
        f"<td><code>{esc(e.event_kind)}</code></td></tr>"
    )


def render_issue(state: ModelState, d: IssueDrill, now: datetime, tz: tzinfo | None = None) -> str:
    noun = "Issue" if d.kind == "issue" else "PR"
    title = f"{noun} #{d.number}"
    window = _window(d, now)
    chart = (
        lane_chart(_lanes(d), window, "Time in each stage", approx=d.approx, now=now, tz=tz)
        if window is not None
        else '<p class="calm-sm">No stage visits recorded yet.</p>'
    )
    basis = "dispatch to merge, approx." if d.approx else "ready to done"
    lead = (
        f"Lead time ({basis}): <b>{esc(age(d.lead_seconds))}</b>"
        if d.lead_seconds is not None
        else "Lead time: not complete yet"
    )
    hist = (
        f"history from {ts_tag(d.history_from, tz, '%Y-%m-%d')}"
        if d.history_from
        else "no history ingested yet"
    )
    approx_note = (
        '<p class="dnote aprx">Stage times are reconstructed from dispatch, PR, review and '
        "merge events (approx.); exact transitions start with issue #2226.</p>"
        if d.approx and d.spans
        else ""
    )
    links = _links(d)
    body = (
        f'<p class="dlead"><b>{esc(title)}</b> in {link(repo_url(d.repo), d.repo, "repo")}'
        f"{' · ' + links if links else ''} · {lead}</p>"
        + section("current", "Now", _current(d), note="as of snapshot")
        + section(
            "stages",
            "Stages",
            approx_note
            + chart
            + table("Stage times", ("Stage", "Total", "Visits", "Basis"), _stage_rows(d), ""),
        )
        + section(
            "timeline",
            "Timeline",
            table(
                "Lifecycle timeline",
                ("When (local)", "Kind", "Event", "Detail", "Item", "Source"),
                [_event_row(d, e, tz) for e in d.timeline],
                "No events recorded for this item.",
            ),
            note=hist,
        )
    )
    name = short_repo(d.repo)
    return page(state, noun, f"{title} · {name}", ((name, repo_url(d.repo)), (title, None)), body)
