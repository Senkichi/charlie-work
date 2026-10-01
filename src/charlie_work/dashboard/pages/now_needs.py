"""The Needs-me zone: headline, group counts and the grouped exception ledger.

Rows arrive already grouped and sorted by ``now_needs_me`` (the single point of grouping);
this module only draws them, in ``NEEDS_ME_GROUPS`` order. Zero rows renders one calm
journal line, never an empty table.
"""

from __future__ import annotations

import re

from ..now_types import NEEDS_ME_GROUPS, NeedsMeItem, NowModel
from .now_fmt import age, esc, issue_url, link, repo_url, slug, stable_id
from .routes import routed

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
_FILTER = (
    '<div class="needs-filter"><label class="sr" for="needs-filter">Filter by repo or issue</label>'
    '<input type="search" id="needs-filter" placeholder="filter repo / issue   ( / )" '
    'autocomplete="off" spellcheck="false"></div>'
)
_REPO_PREFIX = re.compile(r"^charlie --repo (?:'[^']*'|\S+) ")
_CHOICES = re.compile(r"<[^<>|]+(?:\|[^<>|]+)+>")
# The group header already names the decision, so a reason that repeats it is trimmed for
# display (the full reason stays in the row's title and in /api/now.json).
_GROUP_PREFIX = re.compile(
    r"^(?:human needed|operator queue|awaiting (?:your|an operator) verdict):\s*", re.I
)
_LEAD_REF = re.compile(r"^((?:PR )?#\d+)(\s)")
# Visible-row budget for the whole ledger, so the page stays one screen at 1440x900 with
# real row counts; rows past a group's share hide behind a client-side "+N more" toggle
# (a toggle, not a link: the rows are already on the page, and the filter searches them
# all). Exceptions are never capped -- an alarm must not hide behind a click -- and they
# spend the budget first; every other group keeps at least ROW_FLOOR rows visible.
ROW_BUDGET = 16
ROW_FLOOR = 3
_UNCAPPED = frozenset({"Exceptions"})


def visible_counts(sizes: dict[str, int]) -> dict[str, int]:
    """Rows shown per group (``NEEDS_ME_GROUPS`` order) under ``ROW_BUDGET``."""
    shown = {g: (n if g in _UNCAPPED else min(n, ROW_FLOOR)) for g, n in sizes.items()}
    left = ROW_BUDGET - sum(shown.values())
    for g in NEEDS_ME_GROUPS:
        if left <= 0:
            break
        if g in sizes and g not in _UNCAPPED:
            extra = min(left, sizes[g] - shown[g])
            shown[g] += extra
            left -= extra
    return shown


def _short_command(command: str) -> str:
    """What the row shows; the copy button always carries the full command."""
    return _CHOICES.sub("<…>", _REPO_PREFIX.sub("", command, count=1))


def _reason_html(reason: str) -> str:
    text = _GROUP_PREFIX.sub("", reason, count=1) or reason
    m = _LEAD_REF.match(text)
    if m is None:
        return esc(text)
    return f'<b class="ref">{esc(m.group(1))}</b>{esc(text[m.end(1) :])}'


def _target(item: NeedsMeItem) -> str | None:
    """The row's one drill-down link (on the reason), or ``None`` while unrouted."""
    if item.number is not None and item.repo != "fleet":
        return routed(issue_url(item.repo, item.number))
    return routed(repo_url(item.repo))


def _copy(command: str, row_id: str, which: str) -> str:
    return (
        f'<code title="{esc(command)}">{esc(_short_command(command))}</code>'
        f'<button type="button" class="copy" id="{esc(row_id)}-{which}" '
        f'data-copy="{esc(command)}" aria-label="copy: {esc(_short_command(command))}">'
        "copy</button>"
    )


def _commands(item: NeedsMeItem, row_id: str) -> str:
    if not item.command:
        return '<span class="cmd none" title="no single remedy; open the drill-down">—</span>'
    out = f'<span class="cmd">{_copy(item.command, row_id, "c1")}'
    if item.secondary_command:
        # Same line, button only: keeps every row one line high so the ledger fits.
        alt = _short_command(item.secondary_command)
        verb = alt.split(" ", 1)[0] or "alt"
        out += (
            f'<span class="cmd cmd2"><span class="or">or</span>'
            f'<button type="button" class="copy alt" id="{esc(row_id)}-c2" '
            f'data-copy="{esc(item.secondary_command)}" title="{esc(item.secondary_command)}" '
            f'aria-label="copy alternative command: {esc(alt)}">{esc(verb)}</button></span>'
        )
    return out + "</span>"


def _go(row_id: str, target: str | None, reason: str) -> str:
    if target is None:
        return f'<span id="{esc(row_id)}-go">{_reason_html(reason)}</span>'
    return f'<a id="{esc(row_id)}-go" href="{esc(target)}">{_reason_html(reason)}</a>'


def _row(item: NeedsMeItem, over: bool = False) -> str:
    tone, glyph, word = _TONE.get(item.severity, _TONE["warn"])
    kind = _KIND.get(item.kind, item.kind)
    if item.kind == "human_needed" and item.group == "Awaiting your verdict":
        kind = "Verdict"
    row_id = stable_id("nm", item.group, item.kind, item.repo, item.number, item.reason)
    target = _target(item)
    # The age is text, not a second link to the reason's URL (one link per row).
    age_title = "age not recorded" if item.age_seconds is None else f"{word}, waiting this long"
    age_cls = "age n unk" if item.age_seconds is None else "age n"
    short = item.repo.rsplit("/", 1)[-1]
    snap = " · as of snapshot" if item.as_of_snapshot else ""
    return (
        f'<li class="row tone-{tone}{" over" if over else ""}" id="{esc(row_id)}" '
        f'data-grp="{slug(item.group)}">'
        f'<span class="g" aria-hidden="true">{glyph}</span>'
        f"{link(None, age(item.age_seconds), age_cls, age_title)}"
        f'<span class="repo" title="{esc(item.repo)}">{esc(short)}</span>'
        f'<span class="kind">{esc(kind)}<span class="sr"> ({esc(word)})</span></span>'
        f'<span class="why" title="{esc(item.reason + snap)}">'
        f"{_go(row_id, target, item.reason)}</span>"
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
        return f'<section class="needs" id="needs" aria-label="Needs me">{_calm(model)}</section>'
    body: list[str] = []
    grouped = {g: [i for i in model.needs_me if i.group == g] for g in NEEDS_ME_GROUPS}
    shown = visible_counts({g: len(r) for g, r in grouped.items() if r})
    for name in NEEDS_ME_GROUPS:
        rows = grouped[name]
        if not rows:
            continue
        body.append(
            f'<li class="grp" id="grp-{slug(name)}"><span class="label">{esc(name)}</span>'
            f'<span class="meta">{len(rows)} · {esc(_GROUP_NOTE.get(name, ""))}</span></li>'
        )
        cap = shown[name]
        body.extend(_row(i, over=n >= cap) for n, i in enumerate(rows))
        if len(rows) > cap:
            g = slug(name)
            body.append(
                f'<li class="more" data-grp="{g}"><button type="button" class="more-btn" '
                f'id="more-{g}" data-grp="{g}" data-more="{len(rows) - cap}" '
                f'aria-expanded="false">+{len(rows) - cap} more</button></li>'
            )
    return (
        '<section class="needs" id="needs" aria-labelledby="needs-h">'
        f"{_headline(model)}{_FILTER}"
        '<div class="cols" aria-hidden="true"><span></span><span class="label">Age</span>'
        '<span class="label">Repo</span><span class="label">Kind</span>'
        '<span class="label">Reason</span><span class="label r">Command</span></div>'
        f'<ol class="ledger-now" id="needs-list">{"".join(body)}</ol></section>'
    )
