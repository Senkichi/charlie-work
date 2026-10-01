"""Right-rail Flow pipeline and Ready-but-not-dispatchable bars.

Text is HTML (sized in CSS px, so it never shrinks with the rail the way viewBox-scaled
SVG text did) and every number goes through ``link`` (an anchor only once its drill-down
route is registered). Only the bars are SVG: decorative, ``aria-hidden``, never links,
with their geometry in SVG attributes (x/y/width/height) because the CSP forbids inline
styles. Every colour comes from a CSS class bound to an ``--lj-*`` token.
"""

from __future__ import annotations

from ..now_types import FlowModel, NowModel
from .now_fmt import esc, flow_url, fmt_float, link

_PIPE = ("Dispatchable", "Queued", "In progress", "PR open", "Reviewing")
_MAXH, _BAR = 75.0, 48.0
_DONE_H = 6.0
_ND_W = 280.0
_REASON_LABEL = {
    "terminal_label": "Terminal label",
    "active_label": "Active label",
    "operator_claimed": "Operator claimed",
    "mention_covered_awaiting_operator": "Mention awaiting operator",
    "blocked_by_open_dependency": "Blocked by open dependency",
    "unidentified": "Unidentified",
}


def _count(flow: FlowModel, name: str) -> int:
    return next((s.count for s in flow.stages if s.name == name), 0)


def _bar_svg(cls: str, h: float) -> str:
    """A bar exactly ``h`` CSS px tall (viewBox height == svg height, no scaling)."""
    hh = fmt_float(h)
    return (
        f'<svg class="pbar" viewBox="0 0 {fmt_float(_BAR)} {hh}" height="{hh}" '
        f'preserveAspectRatio="none" aria-hidden="true"><rect class="{cls}" x="0" y="0" '
        f'width="{fmt_float(_BAR)}" height="{hh}"/></svg>'
    )


def _stage(name: str, count: int | None, scale: float, done: bool) -> str:
    label = "Done 24h" if done else name
    href = flow_url("done-24h" if done else name)
    cls = "stage done unit-sep" if done else "stage"
    if count is None:
        mark = (
            '<span class="stage-n unk">—</span>'
            '<span class="bar unk"><span class="stage-s">not yet recorded</span></span>'
        )
    elif count == 0:
        mark = f'<span class="stage-n">{link(href, 0)}</span>{_bar_svg("bar zero", 2.0)}'
    else:
        # Done is a 24h throughput, not a queue depth: it never shares the stock scale
        # (77 merged would flatten every stage), so it is a fixed green token bar.
        h = _DONE_H if done else _MAXH * count / scale
        num = "stage-n done" if done else "stage-n"
        mark = (
            f'<span class="{num}">{link(href, int(count))}</span>'
            f"{_bar_svg('bar done' if done else 'bar bar', h)}"
        )
    return f'<li class="{cls}">{mark}<span class="stage-l">{esc(label)}</span></li>'


def _pipeline(flow: FlowModel) -> str:
    counts = [_count(flow, n) for n in _PIPE]
    scale = float(max(counts) if max(counts) > 0 else 0)
    stages = [_stage(n, c, scale, done=False) for n, c in zip(_PIPE, counts, strict=True)]
    stages.append(_stage("Done 24h", flow.done_24h, scale, done=True))
    rework = _count(flow, "Needs rework")
    loop = (
        '<p class="rework"><span aria-hidden="true">↺ </span>'
        f"{link(flow_url('Needs rework'), rework, 'n hn')} needs rework, back to In progress</p>"
    )
    return (
        '<ol class="pipe" aria-label="Pipeline: issues per stage, left to right">'
        f"{''.join(stages)}</ol>{loop}"
    )


def render_flow(model: NowModel) -> str:
    t = model.totals
    sub = (
        f'<p class="sub">In flight {link(flow_url("active"), t.active_issues)} active issues '
        f"· {link('/prs?linked=1', t.open_linked_prs)} linked PRs open "
        f"· {link('/prs?linked=0', t.unlinked_prs)} unlinked PRs "
        '<span class="as-of">as of snapshot</span></p>'
    )
    return (
        '<section class="flow-now" aria-labelledby="flow-h"><h2 id="flow-h">Flow</h2>'
        f"{_pipeline(model.flow)}{sub}</section>"
    )


def _nd_row(reason: str, count: int, top: int) -> str:
    w = _ND_W * count / top
    label = _REASON_LABEL.get(reason, reason.replace("_", " ").capitalize())
    return (
        f'<li class="nd-row"><span class="hl">{esc(label)}</span>'
        f'<svg class="hbar-svg" viewBox="0 0 {fmt_float(_ND_W)} 12" preserveAspectRatio="none" '
        f'aria-hidden="true"><rect class="hbar-track" x="0" y="0" width="{fmt_float(_ND_W)}" '
        f'height="12"/><rect class="hbar" x="0" y="0" width="{fmt_float(w)}" height="12"/>'
        f"</svg>{link(f'/backlog?reason={reason}', int(count), 'n hn')}</li>"
    )


def render_not_dispatchable(model: NowModel) -> str:
    reasons = sorted(model.flow.not_dispatchable, key=lambda r: (-r.count, r.reason))
    held = sum(r.count for r in reasons)
    ready = model.totals.ready_issues
    head = (
        '<h2 id="nd-h">Ready, not dispatchable <span class="meta">'
        f"{link('/backlog?ready=0', held)} of {link('/backlog?ready=1', ready)} ready</span></h2>"
    )
    if not reasons:
        line = "Every ready issue is dispatchable." if ready else "No ready issues."
        return f'<section class="nd-now" aria-labelledby="nd-h">{head}<p class="calm-sm">{line}</p></section>'
    top = max(r.count for r in reasons) or 1
    rows = "".join(_nd_row(r.reason, r.count, top) for r in reasons)
    lead = reasons[0]
    eg = ", ".join(lead.examples[:3])
    note = (
        f'<p class="sub">most held: {esc(_REASON_LABEL.get(lead.reason, lead.reason).lower())}'
        f"{' (e.g. ' + esc(eg) + ')' if eg else ''}</p>"
    )
    return (
        f'<section class="nd-now" aria-labelledby="nd-h">{head}'
        f'<ul class="nd-list" aria-label="Held issues by reason">{rows}</ul>{note}</section>'
    )
