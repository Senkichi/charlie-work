"""The Needs-me zone: headline, group counts and the grouped exception ledger.

Rows arrive already grouped and sorted by ``now_needs_me`` (the single point of grouping);
this module only draws them, in ``NEEDS_ME_GROUPS`` order. Zero rows renders one calm
journal line, never an empty table.
"""

from __future__ import annotations

import re

from ..now_types import NEEDS_ME_GROUPS, NeedsMeItem, NowModel
from .now_fmt import age, esc, issue_url, link, repo_url, slug, stable_id

# severity -> (tone class, glyph, word). Colour is never the only signal: glyph + word too.
_TONE = {
    "anomaly": ("danger", "✕", "failing"),
    "warn": ("warn", "▲", "at risk"),
    "action": ("plain", "○", "to decide"),
}
_KIND = {
    "operator_queue": "Queue",
    "human_needed": "Human",
    "alarm": "Alarm",
    "stale_source": "Stale",
    "paused": "Paused",
    "supervisor": "Supervisor",
}
_GROUP_NOTE = {
    "Exceptions": "alarms and stale or unreadable sources",
    "Awaiting your verdict": "a PR is parked until you record a verdict",
    "Human needed": "an issue needs a human decision",
    "Operator queue": "escalated issues waiting to be requeued",
}
_SEP = " · "
_REPO_PREFIX = re.compile(r"^charlie --repo (?:'[^']*'|\S+) ")


def _short_command(command: str) -> str:
    """What the row shows; the copy button always carries the full command."""
    return _REPO_PREFIX.sub("charlie … ", command, count=1)


def _target(item: NeedsMeItem) -> str:
    if item.number is not None and item.repo != "fleet":
        return issue_url(item.repo, item.number)
    return repo_url(item.repo)


def _copy(command: str, row_id: str, which: str) -> str:
    return (
        f'<code title="{esc(command)}">{esc(_short_command(command))}</code>'
        f'<button type="button" class="copy" id="{esc(row_id)}-{which}" '
        f'data-copy="{esc(command)}" aria-label="copy command">copy</button>'
    )


def _commands(item: NeedsMeItem, row_id: str) -> str:
    if not item.command:
        return '<span class="cmd none" title="no single remedy; open the drill-down">—</span>'
    out = f'<span class="cmd">{_copy(item.command, row_id, "c1")}</span>'
    if item.secondary_command:
        out += (
            f'<span class="cmd cmd2"><span class="or">or</span>'
            f"{_copy(item.secondary_command, row_id, 'c2')}</span>"
        )
    return out


def _row(item: NeedsMeItem) -> str:
    tone, glyph, word = _TONE.get(item.severity, _TONE["warn"])
    kind = _KIND.get(item.kind, item.kind)
    if item.kind == "human_needed" and item.group == "Awaiting your verdict":
        kind = "Verdict"
    row_id = stable_id("nm", item.group, item.kind, item.repo, item.number, item.reason)
    target = _target(item)
    age_title = "age not recorded" if item.age_seconds is None else f"{word}, open drill-down"
    age_cls = "age n unk" if item.age_seconds is None else "age n"
    short = item.repo.rsplit("/", 1)[-1]
    snap = " · as of snapshot" if item.as_of_snapshot else ""
    return (
        f'<li class="row tone-{tone}" id="{esc(row_id)}">'
        f'<span class="g" aria-hidden="true">{glyph}</span>'
        f"{link(target, age(item.age_seconds), age_cls, age_title)}"
        f'<span class="repo" title="{esc(item.repo)}">{esc(short)}</span>'
        f'<span class="kind">{esc(kind)}<span class="sr"> ({esc(word)})</span></span>'
        f'<span class="why" title="{esc(item.reason + snap)}">'
        f'<a id="{esc(row_id)}-go" href="{esc(target)}">{esc(item.reason)}</a></span>'
        f'<span class="cmds">{_commands(item, row_id)}</span></li>'
    )


def _calm(model: NowModel) -> str:
    ages = [f.age_seconds for f in model.freshness if f.age_seconds is not None]
    last = f" Last pass {esc(age(min(ages)))} ago." if ages else ""
    return f'<p class="calm">Nothing needs you.{last}</p>'


def _headline(model: NowModel) -> str:
    total = len(model.needs_me)
    counts = {g.group: g.count for g in model.needs_me_groups}
    parts = []
    for name in NEEDS_ME_GROUPS:
        n = counts.get(name, sum(1 for i in model.needs_me if i.group == name))
        cls = "grp-count" + (" zero" if n == 0 else "")
        parts.append(
            f'<span class="{cls}">{link("#grp-" + slug(name), n)} {esc(name.lower())}</span>'
        )
    return (
        '<div class="needs-head">'
        f'{link("#needs-list", total, "big n")}<h2 id="needs-h">need you</h2>'
        f'<span class="split">{_SEP.join(parts)}</span></div>'
    )


def render_needs(model: NowModel) -> str:
    """The left column: headline + grouped ledger, or the calm line when empty."""
    if not model.needs_me:
        return f'<section class="needs" aria-label="Needs me">{_calm(model)}</section>'
    body: list[str] = []
    for name in NEEDS_ME_GROUPS:
        rows = [i for i in model.needs_me if i.group == name]
        if not rows:
            continue
        body.append(
            f'<li class="grp" id="grp-{slug(name)}"><span class="label">{esc(name)}</span>'
            f'<span class="meta">{len(rows)} · {esc(_GROUP_NOTE.get(name, ""))}</span></li>'
        )
        body.extend(_row(i) for i in rows)
    return (
        '<section class="needs" aria-labelledby="needs-h">'
        f"{_headline(model)}"
        '<div class="cols" aria-hidden="true"><span></span><span class="label">Age</span>'
        '<span class="label">Repo</span><span class="label">Kind</span>'
        '<span class="label">Reason</span><span class="label r">Command</span></div>'
        f'<ol class="ledger-now" id="needs-list">{"".join(body)}</ol></section>'
    )
