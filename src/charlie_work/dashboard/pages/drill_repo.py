"""Repo drill-down page (``/repo/<owner>/<name>``): live state, then recent history.

The live half (stages, Needs-me rows, capacity) is the same Now model tick the header
reports; the history half (loop passes, merges, escalations) is ``dashboard.db``. Now links
here with ``?stage=<stage>`` or ``?view=needs|workers|runners``: that section is marked as
the focus (and the stage row as current), so the number that was clicked is findable.
"""

from __future__ import annotations

from datetime import tzinfo
from urllib.parse import urlencode

from ..drill import EscalationRow, LoopPassRow, MergeRow, RepoDrill
from ..now_types import NeedsMeItem
from ..read_model import ModelState
from .drill_shell import page, section, table, ts_tag
from .now_fmt import (
    age,
    cap_text,
    esc,
    flow_url,
    issue_url,
    link,
    pass_url,
    pr_url,
    short_repo,
    slug,
)
from .now_needs import TONE

_VIEWS = ("needs", "workers", "runners")


def _num(repo: str, issue: int | None, pr: int | None) -> str:
    parts = []
    if issue is not None:
        parts.append(link(issue_url(repo, issue), f"#{issue}"))
    if pr is not None:
        parts.append(link(pr_url(repo, pr), f"PR #{pr}"))
    return " · ".join(parts) or '<span class="unk">—</span>'


def _stages(d: RepoDrill, focus: str | None) -> str:
    if not d.stages:
        return '<p class="calm-sm">No snapshot stage counts for this repo.</p>'
    rows = []
    for s in d.stages:
        cur = ' aria-current="true"' if focus == slug(s.name) else ""
        href = f"{flow_url(s.name)}?{urlencode({'repo': d.slug})}"
        rows.append(
            f'<tr{cur}><th scope="row">{esc(s.name)}</th>'
            f'<td class="num">{link(href, s.count)}</td></tr>'
        )
    return table("Issues per stage", ("Stage", "Issues"), rows, "")


def _needs(repo: str, items: tuple[NeedsMeItem, ...]) -> str:
    rows = []
    for i in items:
        tone, glyph, word = TONE.get(i.severity, TONE["warn"])
        target = issue_url(repo, i.number) if i.number is not None else None
        cmd = f"<code>{esc(i.command)}</code>" if i.command else '<span class="unk">—</span>'
        rows.append(
            f'<tr class="tone-{tone}"><td class="lvl"><span aria-hidden="true">{glyph}</span> '
            f'{esc(word)}</td><td class="num">{esc(age(i.age_seconds))}</td>'
            f"<td>{link(target, i.reason, 'why')}</td><td>{cmd}</td></tr>"
        )
    return table(
        "Needs me", ("State", "Age", "Reason", "Command"), rows, "Nothing here needs you."
    )


def _capacity(d: RepoDrill) -> str:
    def pair(live: int | None, cap: int | None) -> str:
        shown = "?" if live is None else str(live)
        return f"{esc(shown)} running / {cap_text(cap)}"

    w, r, ru = d.workers, d.reviewers, d.runners
    rows = [
        f'<tr id="workers"><th scope="row">Workers</th><td>{pair(w.live, w.cap) if w else "—"}</td></tr>',
        f'<tr><th scope="row">Reviewers</th><td>{pair(r.live, r.cap) if r else "—"}</td></tr>',
        '<tr id="runners"><th scope="row">CI runners</th><td>'
        + (
            f"{ru.online} online / {ru.capacity} slots · busy "
            f"{'?' if ru.busy is None else ru.busy} · parked {ru.parked} · demand {ru.demand}"
            if ru
            else "not allocated to this repo"
        )
        + "</td></tr>",
    ]
    return table("Capacity", ("Kind", "Now"), rows, "")


def _ok(p: LoopPassRow) -> str:
    if p.ok is None:
        return '<span class="unk">running or unknown</span>'
    return "ok" if p.ok else '<span class="text-danger">failed</span>'


def _passes(repo: str, rows: tuple[LoopPassRow, ...], tz: tzinfo | None) -> str:
    out = [
        f"<tr><td>{link(pass_url(repo, p.correlation_id), p.correlation_id, 'cid')}</td>"
        f"<td>{ts_tag(p.started_at, tz)}</td>"
        f'<td class="num">{esc(age(p.elapsed_seconds))}</td><td>{_ok(p)}</td>'
        f'<td class="num">{esc(p.error_count)}</td><td class="num">{esc(p.merge_count)}</td>'
        f'<td class="num">{esc(p.review_count)}</td></tr>'
        for p in rows
    ]
    return table(
        "Last loop passes",
        ("Pass", "Started (local)", "Took", "Result", "Errors", "Merges", "Reviews"),
        out,
        "No loop passes recorded yet.",
    )


def _merges(repo: str, rows: tuple[MergeRow, ...], tz: tzinfo | None) -> str:
    out = [
        f"<tr><td>{ts_tag(m.ts, tz)}</td><td>{_num(repo, m.issue, m.pr)}</td>"
        f"<td><code>{esc(m.evidence)}</code>"
        + (
            ' <span class="aprx">approx. (noticed, not merged, at this time)</span>'
            if m.approx
            else ""
        )
        + "</td></tr>"
        for m in rows
    ]
    return table(
        "Recent merges", ("Merged (local)", "Item", "Evidence"), out, "No merges recorded."
    )


def _escalations(repo: str, rows: tuple[EscalationRow, ...], tz: tzinfo | None) -> str:
    out = [
        f"<tr><td>{ts_tag(e.ts, tz)}</td><td>{_num(repo, e.issue, e.pr)}</td>"
        f"<td>{esc(e.reason or '—')}</td><td>{esc(e.detail or '')}</td></tr>"
        for e in rows
    ]
    return table(
        "Recent escalations", ("When (local)", "Item", "Reason", "Detail"), out, "No escalations."
    )


def render_repo(
    state: ModelState,
    d: RepoDrill,
    *,
    stage: str | None = None,
    view: str | None = None,
    tz: tzinfo | None = None,
) -> str:
    snap = "as of snapshot"
    hist_note = (
        esc(d.history_error)
        if d.history_error
        else ("history from " + ts_tag(d.history_from, tz, "%Y-%m-%d") if d.history_from else "")
    )
    fresh = d.freshness
    pass_age = (
        f"last pass {esc(age(fresh.age_seconds))} ago" + (" · stale" if fresh.stale else "")
        if fresh
        else ""
    )
    view = view if view in _VIEWS else None
    body = (
        f'<p class="dlead"><b>{esc(d.slug)}</b> · {pass_age}</p>'
        + '<div class="dgrid">'
        + section("stages", "Stages", _stages(d, stage), note=snap, focus=stage is not None)
        + section("needs", "Needs me", _needs(d.slug, d.needs_me), focus=view == "needs")
        + section("cap", "Capacity", _capacity(d), focus=view in ("workers", "runners"))
        + "</div>"
        + section("passes", "Loop passes", _passes(d.slug, d.passes, tz), note=hist_note)
        + section("merges", "Merges", _merges(d.slug, d.merges, tz))
        + section("escalations", "Escalations", _escalations(d.slug, d.escalations, tz))
    )
    name = short_repo(d.slug)
    return page(state, "Repo", name, ((name, None),), body)
