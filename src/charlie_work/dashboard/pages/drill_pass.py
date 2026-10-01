"""Loop-pass drill-down page (``/pass/<repo>/<correlation id>``): one dense event table.

Rows are in insertion order. Level is a glyph plus the word (never colour alone); the payload
is the model's compact, control-stripped preview, escaped here like every other value.
"""

from __future__ import annotations

from datetime import tzinfo

from ..drill import PassDrill, PassEvent
from ..drill.loop_pass import MAX_PASS_EVENTS
from ..read_model import ModelState
from .drill_shell import page, section, table, ts_tag
from .now_fmt import age, esc, issue_url, link, pr_url, repo_url, short_repo

# level -> (row tone, glyph). The word is printed beside the glyph.
_LEVEL = {"error": ("danger", "✕"), "warning": ("warn", "▲"), "info": ("plain", "·")}


def _items(repo: str, e: PassEvent) -> str:
    parts = []
    if e.issue is not None:
        parts.append(link(issue_url(repo, e.issue), f"#{e.issue}"))
    if e.pr is not None:
        parts.append(link(pr_url(repo, e.pr), f"PR #{e.pr}"))
    return " ".join(parts)


def _row(repo: str, e: PassEvent, tz: tzinfo | None) -> str:
    tone, glyph = _LEVEL.get(e.level, ("plain", "?"))
    return (
        f'<tr class="tone-{tone}"><td class="num">{e.id}</td>'
        f"<td>{ts_tag(e.ts, tz, '%H:%M:%S')}</td>"
        f'<td class="lvl"><span aria-hidden="true">{glyph}</span> {esc(e.level)}</td>'
        f"<td><code>{esc(e.kind)}</code></td><td>{_items(repo, e)}</td>"
        f'<td class="preview"><code>{esc(e.preview)}</code></td></tr>'
    )


def _summary(d: PassDrill, tz: tzinfo | None) -> str:
    result = (
        "result unknown (no loop_passes row)"
        if d.ok is None
        else ("ok" if d.ok else '<span class="text-danger">failed</span>')
    )
    levels = " · ".join(f"{n} {esc(level)}" for level, n in d.level_counts) or "no events"
    return (
        f'<p class="dlead">Pass <code>{esc(d.correlation_id)}</code> of <b>{esc(d.repo)}</b>: '
        f"started {ts_tag(d.started_at, tz)}, completed {ts_tag(d.completed_at, tz)} (local), "
        f"took {esc(age(d.elapsed_seconds))} · {result}</p>"
        f'<p class="dnote">{len(d.events)} events: {levels}</p>'
    )


def render_pass(state: ModelState, d: PassDrill, tz: tzinfo | None = None) -> str:
    rows = [_row(d.repo, e, tz) for e in d.events]
    more = (
        f'<p class="dnote text-warn">Showing the first {MAX_PASS_EVENTS} events; '
        "this pass has more.</p>"
        if d.truncated
        else ""
    )
    inner = more + table(
        "Events in this loop pass",
        ("#", "Time (local)", "Level", "Kind", "Item", "Payload"),
        rows,
        "No events carry this correlation id.",
    )
    name = short_repo(d.repo)
    trail = ((name, repo_url(d.repo)), (f"Pass {d.correlation_id}", None))
    return page(
        state,
        "Loop pass",
        f"Pass {d.correlation_id}",
        trail,
        _summary(d, tz) + section("events", "Events", inner),
    )
