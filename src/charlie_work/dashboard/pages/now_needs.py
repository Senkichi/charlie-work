"""The Needs-you panel: the count, three group tiles, and one list at a time.

Rows arrive grouped and sorted by ``now_model`` (the single point of grouping); this module
only draws the decisions (``NEEDS_ME_GROUPS`` minus Exceptions, which are the header health
pill). Every group's rows are on the page; ``now.js`` picks the active group (tile click,
kept in the URL hash across refreshes) and expands rows. A collapsed row is two lines; its
commands and copy buttons exist only inside the expanded detail.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..now_types import NeedsMeItem, NowModel
from .now_fmt import age, age_compact, esc, issue_url, repo_url, short_repo, stable_id
from .now_health import needs_rows
from .routes import routed

# severity -> (tone class, glyph, word). Kept for the drill-down pages that share it.
TONE = {
    "anomaly": ("danger", "✕", "failing"),
    "warn": ("warn", "▲", "at risk"),
    "action": ("plain", "○", "to decide"),
}


@dataclass(frozen=True)
class Group:
    name: str  # NEEDS_ME_GROUPS value
    slug: str  # data-g / URL hash value
    tile: str
    hint: str


GROUPS = (
    Group("Awaiting your verdict", "verdicts", "Verdicts", "a PR is parked until you decide"),
    Group("Human needed", "decisions", "Decisions", "an issue needs a human call"),
    Group("Operator queue", "requeue", "Requeue", "escalated, waiting to be requeued"),
)
ACTIVE_ROWS = 6  # rows the active group shows before "N more"
THEN_ROWS = 4  # rows a background group shows (CSS hides the rest; clicking activates it)

_REPO_PREFIX = re.compile(r"^charlie --repo (?:'[^']*'|\S+) ")
_CHOICES = re.compile(r"<[^<>|]+(?:\|[^<>|]+)+>")
_GROUP_PREFIX = re.compile(
    r"^(?:human needed|operator queue|awaiting (?:your|an operator) verdict):\s*", re.I
)
_VERDICT_PHRASE = re.compile(r"^PR #\d+ \(issue #\d+\) awaits an operator verdict")
_LEAD_REF = re.compile(r"^#\d+\s*")
_PR_REF = re.compile(r"PR #(\d+)")


def _short_command(command: str) -> str:
    """What the row shows; the copy button always carries the full command."""
    return _CHOICES.sub("<…>", _REPO_PREFIX.sub("", command, count=1))


def row_title(item: NeedsMeItem) -> str:
    where = short_repo(item.repo)
    pr = _PR_REF.search(item.reason)
    if pr:
        return f"{where} · PR #{pr.group(1)}"
    return f"{where} · #{item.number}" if item.number is not None else where


def row_why(item: NeedsMeItem) -> str:
    text = _GROUP_PREFIX.sub("", item.reason, count=1) or item.reason
    text = _VERDICT_PHRASE.sub("awaits your verdict", text, count=1)
    return _LEAD_REF.sub("", text, count=1) or text


def _target(item: NeedsMeItem) -> str | None:
    """The row's drill-down (issue, else repo), or ``None`` while unrouted."""
    if item.number is not None and item.repo != "fleet":
        return routed(issue_url(item.repo, item.number))
    return routed(repo_url(item.repo))


def _cmd(command: str, row_id: str, which: str, label: str) -> str:
    short = _short_command(command)
    return (
        f'<div class="cmd"><code title="{esc(command)}">{esc(short)}</code>'
        f'<button type="button" class="copy" id="{esc(row_id)}-{which}" '
        f'data-copy="{esc(command)}" aria-label="{esc(label)}: {esc(short)}">'
        f"{'copy' if which == 'c1' else 'or this'}</button></div>"
    )


def _detail(item: NeedsMeItem, row_id: str) -> str:
    parts = []
    if item.command:
        parts.append(_cmd(item.command, row_id, "c1", "copy"))
        if item.secondary_command:
            parts.append(_cmd(item.secondary_command, row_id, "c2", "copy alternative command"))
    else:
        parts.append('<p class="dim small">No single remedy; open the drill-down.</p>')
    target = _target(item)
    if target:
        parts.append(
            f'<a class="drill small dim" id="{esc(row_id)}-go" href="{esc(target)}">'
            "open drill-down →</a>"
        )
    return "".join(parts)


def _row(item: NeedsMeItem, k: int) -> str:
    row_id = stable_id("nm", item.group, item.kind, item.repo, item.number, item.reason)
    cls = "row" + (" gt4" if k >= THEN_ROWS else "") + (" gt6" if k >= ACTIVE_ROWS else "")
    snap = " · as of snapshot" if item.as_of_snapshot else ""
    return (
        f'<li class="{cls}" id="{esc(row_id)}">'
        f'<button type="button" class="rowbtn" id="{esc(row_id)}-b" aria-expanded="false" '
        f'aria-controls="{esc(row_id)}-d" title="{esc(item.reason + snap)}">'
        f'<span class="what">{esc(row_title(item))}</span>'
        f'<span class="age">{esc(age_compact(item.age_seconds))}</span>'
        f'<span class="why">{esc(row_why(item))}</span></button>'
        f'<div class="detail" id="{esc(row_id)}-d" hidden>{_detail(item, row_id)}</div></li>'
    )


def _section(group: Group, rows: tuple[NeedsMeItem, ...], active: str) -> str:
    more = len(rows) - ACTIVE_ROWS
    more_btn = (
        f'<button type="button" class="more" id="more-{group.slug}" data-more="{group.slug}" '
        f'data-n="{more}" aria-expanded="false">{more} more</button>'
        if more > 0
        else ""
    )
    on = ' data-on="1"' if group.slug == active else ""
    every = ' data-all="0"'
    return (
        f'<section class="grp" data-g="{group.slug}" aria-label="{esc(group.tile)}"{on}{every}>'
        f'<button type="button" class="ghead" id="gh-{group.slug}" data-g="{group.slug}">'
        f"Then: {esc(group.tile)} ({len(rows)})</button>"
        f'<p class="hint dim small">{esc(group.hint)}. Oldest first; click a row for the '
        "command.</p>"
        f'<ul class="rows">{"".join(_row(i, k) for k, i in enumerate(rows))}</ul>{more_btn}'
        "</section>"
    )


def _oldest(rows: tuple[NeedsMeItem, ...]) -> float | None:
    ages = [i.age_seconds for i in rows if i.age_seconds is not None]
    return max(ages) if ages else None


def _tile(group: Group, rows: tuple[NeedsMeItem, ...], active: str) -> str:
    oldest = f"oldest {age_compact(_oldest(rows))}" if rows else "none waiting"
    return (
        f'<button type="button" class="tile" id="tile-{group.slug}" data-g="{group.slug}" '
        f'aria-pressed="{"true" if group.slug == active else "false"}">'
        f'<span class="num">{len(rows)}</span><span class="lbl">{esc(group.tile)}</span>'
        f'<span class="dim small">{esc(oldest)}</span></button>'
    )


def render_needs(model: NowModel) -> str:
    """The left panel: count, tiles and the grouped list, or the calm line when empty."""
    rows = needs_rows(model)
    by_group = {g: tuple(i for i in rows if i.group == g.name) for g in GROUPS}
    active = next((g.slug for g in GROUPS if by_group[g]), GROUPS[0].slug)
    oldest = _oldest(rows)
    sub = f"oldest waiting {age_compact(oldest)}" if oldest is not None else ""
    head = (
        f'<div><div class="hero"><span class="num" id="needs-n">{len(rows)}</span>'
        f'<h2 id="needs-h">need you</h2></div><p class="sub">{esc(sub)}</p></div>'
    )
    if not rows:
        ages = [f.age_seconds for f in model.freshness if f.age_seconds is not None]
        last = f" Last pass {esc(age(min(ages)))} ago." if ages else ""
        return (
            '<section class="panel n-needs" id="needs" aria-labelledby="needs-h">'
            f'{head}<p class="calm">Nothing needs you.{last}</p></section>'
        )
    tiles = "".join(_tile(g, by_group[g], active) for g in GROUPS)
    sections = "".join(_section(g, by_group[g], active) for g in GROUPS if by_group[g])
    return (
        '<section class="panel n-needs" id="needs" aria-labelledby="needs-h">'
        f'{head}<div class="tiles" role="group" aria-label="Needs groups">{tiles}</div>'
        f'<div class="list" id="needs-list" data-active="{active}" data-default="{active}">'
        f"{sections}</div></section>"
    )
