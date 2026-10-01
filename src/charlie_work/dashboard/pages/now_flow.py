"""Right-rail Flow pipeline and Ready-but-not-dispatchable bars (server-rendered SVG).

Geometry goes in SVG attributes (x/y/width/height), never in ``style``: the CSP forbids
inline styles, and every colour comes from a CSS class bound to an ``--lj-*`` token.
"""

from __future__ import annotations

from ..now_types import FlowModel, NowModel
from .now_fmt import esc, flow_url, fmt_float, link

_PIPE = ("Dispatchable", "Queued", "In progress", "PR open", "Reviewing")
_W, _BASE, _MAXH, _BAR = 520.0, 118.0, 75.0, 48.0
_DONE_H = 6.0
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


def _svg_link(href: str, inner: str) -> str:
    return f'<a href="{esc(href)}">{inner}</a>'


def _column(x: float, name: str, count: int | None, scale: float, done: bool) -> str:
    cx = x + _BAR / 2
    label = "Done 24h" if done else name
    href = flow_url("done-24h" if done else name)
    if count is None:
        mark = (
            f'<rect class="bar unk" x="{fmt_float(x)}" y="{fmt_float(_BASE - _MAXH)}" '
            f'width="{fmt_float(_BAR)}" height="{fmt_float(_MAXH)}"/>'
            f'<text class="stage-n unk" x="{fmt_float(cx)}" y="{fmt_float(_BASE - _MAXH - 8)}" '
            'text-anchor="middle">—</text>'
            f'<text class="stage-s" x="{fmt_float(cx)}" y="84" text-anchor="middle">not yet</text>'
            f'<text class="stage-s" x="{fmt_float(cx)}" y="97" '
            'text-anchor="middle">recorded</text>'
        )
    else:
        h = _MAXH * count / scale if scale else 0.0
        kind = "done" if done else "bar"
        if done and count:
            # Done is a 24h throughput, not a queue depth: it never shares the stock scale
            # (77 merged would flatten every stage), so it is a fixed green token bar.
            h = _DONE_H
        if count == 0:
            rect = (
                f'<rect class="bar zero" x="{fmt_float(x)}" y="{fmt_float(_BASE - 2.5)}" '
                f'width="{fmt_float(_BAR)}" height="2"/>'
            )
            h = 2.5
        else:
            rect = (
                f'<rect class="bar {kind}" x="{fmt_float(x)}" y="{fmt_float(_BASE - h)}" '
                f'width="{fmt_float(_BAR)}" height="{fmt_float(h)}"/>'
            )
        num_cls = "stage-n done" if done else "stage-n"
        mark = (
            f'{rect}<text class="{num_cls}" x="{fmt_float(cx)}" y="{fmt_float(_BASE - h - 8)}" '
            f'text-anchor="middle">{int(count)}</text>'
        )
    return (
        _svg_link(href, mark)
        + f'<text class="stage-l" x="{fmt_float(cx)}" y="148" text-anchor="middle">'
        f"{esc(label)}</text>"
    )


def _pipeline_svg(flow: FlowModel) -> str:
    counts = [_count(flow, n) for n in _PIPE]
    scale = float(max(counts) if max(counts) > 0 else 0)
    step = _W / 6
    xs = [i * step + (step - _BAR) / 2 for i in range(6)]
    sep = 5 * step - 1
    parts = [
        f'<line class="base" x1="0" x2="{fmt_float(_W)}" y1="118.5" y2="118.5"/>',
        f'<line class="unit-sep" x1="{fmt_float(sep)}" x2="{fmt_float(sep)}" y1="30" y2="118"/>',
    ]
    for i, name in enumerate(_PIPE):
        parts.append(_column(xs[i], name, counts[i], scale, done=False))
    parts.append(_column(xs[5], "Done 24h", flow.done_24h, scale, done=True))
    for i in range(5):
        a, b = xs[i] + _BAR + 4, xs[i + 1] - 9
        parts.append(
            f'<path class="arrow" d="M{fmt_float(a)} 130 H{fmt_float(b)}"/>'
            f'<path class="arrowhead" d="M{fmt_float(b)} 127 L{fmt_float(b + 5)} 130 '
            f'L{fmt_float(b)} 133 Z"/>'
        )
    rework = _count(flow, "Needs rework")
    src, dst = xs[3] + _BAR / 2, xs[2] + _BAR / 2
    parts.append(
        _svg_link(
            flow_url("Needs rework"),
            f'<path class="arrow rework" d="M{fmt_float(src)} 154 V162 H{fmt_float(dst)} V157"/>'
            f'<path class="arrowhead" d="M{fmt_float(dst - 3)} 158 L{fmt_float(dst)} 153 '
            f'L{fmt_float(dst + 3)} 158 Z"/>'
            f'<text class="stage-s" x="{fmt_float(src + 8)}" y="166"><tspan class="hn">'
            f"{rework}</tspan> needs rework, back to In progress</text>",
        )
    )
    done = "not recorded" if flow.done_24h is None else str(flow.done_24h)
    summary = ", ".join(f"{n} {c}" for n, c in zip(_PIPE, counts, strict=True))
    aria = f"Pipeline: {summary}, Done in 24h {done}; {rework} needs rework"
    return (
        f'<svg class="chart pipe" viewBox="0 0 {fmt_float(_W)} 172" role="img" '
        f'aria-label="{esc(aria)}">{"".join(parts)}</svg>'
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
        f"{_pipeline_svg(model.flow)}{sub}</section>"
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
        return f'<section aria-labelledby="nd-h">{head}<p class="calm-sm">{line}</p></section>'
    top = max(r.count for r in reasons) or 1
    rows = []
    for i, r in enumerate(reasons):
        y = i * 24 + 5
        w = 280.0 * r.count / top
        label = _REASON_LABEL.get(r.reason, r.reason.replace("_", " ").capitalize())
        rows.append(
            _svg_link(
                f"/backlog?reason={r.reason}",
                f'<text class="hl" x="0" y="{y + 10}">{esc(label)}</text>'
                f'<rect class="hbar-track" x="190" y="{y}" width="280" height="12"/>'
                f'<rect class="hbar" x="190" y="{y}" width="{fmt_float(w)}" height="12"/>'
                f'<text class="hn" x="478" y="{y + 11}">{int(r.count)}</text>',
            )
        )
    aria = "Not dispatchable by reason: " + ", ".join(
        f"{_REASON_LABEL.get(r.reason, r.reason)} {r.count}" for r in reasons
    )
    svg = (
        f'<svg class="chart nd" viewBox="0 0 520 {len(reasons) * 24 + 2}" role="img" '
        f'aria-label="{esc(aria)}">{"".join(rows)}</svg>'
    )
    lead = reasons[0]
    eg = ", ".join(lead.examples[:3])
    note = (
        f'<p class="sub">most held: {esc(_REASON_LABEL.get(lead.reason, lead.reason).lower())}'
        f"{' (e.g. ' + esc(eg) + ')' if eg else ''}</p>"
    )
    return f'<section aria-labelledby="nd-h">{head}{svg}{note}</section>'
