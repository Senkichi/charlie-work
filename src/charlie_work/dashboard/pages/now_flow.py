"""The Pipeline panel: one connected track of stage nodes, the longest queue in focus.

The headline is the takeaway ("15 PRs piled up at PR open"). Each node is a button that
opens a per-repo drawer (``now.js`` shows the drawer named by the URL hash, so it survives
refresh); the "why" link opens the ready-but-not-dispatchable breakdown. Node text is HTML;
only the circles are ``aria-hidden`` SVG whose size is an SVG attribute in ``em``.
"""

from __future__ import annotations

import math

from ..now_types import FlowModel, NowModel
from .now_bars import hbars, repo_label
from .now_fmt import esc, slug

_TRACK = ("Dispatchable", "Queued", "In progress", "PR open", "Reviewing")
_PR_STAGES = frozenset({"PR open", "Reviewing"})
DONE = "Merged 24h"
# The rework arc runs Reviewing -> In progress; now-page.css pins it to these grid columns
# (``.arc-curve``), so a reordered ``_TRACK`` must fail loudly (tests assert it).
ARC_COLUMNS = (_TRACK.index("In progress") + 1, _TRACK.index("Reviewing") + 2)
_REASON_LABEL = {
    "terminal_label": "Terminal label",
    "active_label": "Already active",
    "operator_claimed": "Operator claimed",
    "mention_covered_awaiting_operator": "Mention awaiting operator",
    "blocked_by_open_dependency": "Blocked by a dependency",
    "unidentified": "Unidentified",
    "parked_unready": "Parked, not ready",
}


def _count(flow: FlowModel, name: str) -> int:
    return next((s.count for s in flow.stages if s.name == name), 0)


def _focus_stage(flow: FlowModel) -> tuple[str, int]:
    """The queue the operator should look at: the biggest stage past Dispatchable."""
    best = ("", 0)
    for name in _TRACK[1:]:
        if _count(flow, name) > best[1]:
            best = (name, _count(flow, name))
    return best


def headline(flow: FlowModel) -> str:
    name, n = _focus_stage(flow)
    if not n:
        return "Nothing in flight"
    noun = "PRs" if name in _PR_STAGES else "issues"
    return f"{n} {noun} piled up at <b>{esc(name)}</b>"


def _dot(count: int | None) -> str:
    """Circle sized by sqrt(count), in em so it scales with the type; ``None`` is dashed."""
    r = 0.64 if not count else min(2.0, (9 + 3.2 * math.sqrt(count)) / 14)
    d = f"{2 * r:.2f}"
    unknown = ' class="unk"' if count is None else ""
    return (
        f'<svg class="dot" width="{d}em" height="{d}em" viewBox="0 0 100 100" '
        f'aria-hidden="true"><circle{unknown} cx="50" cy="50" r="46"/></svg>'
    )


def _node(name: str, count: int | None, focus: bool, done: bool) -> str:
    key = "merged" if done else slug(name)
    shown = "—" if count is None else str(count)
    cls = "node" + (" focus" if focus else "")
    return (
        f'<li class="step"><button type="button" class="{cls}" id="node-{key}" '
        f'data-pipe="{key}" aria-pressed="false" aria-label="{esc(name)}: {esc(shown)}">'
        f'<span class="cnt num">{esc(shown)}</span>{_dot(count)}'
        f'<span class="lbl">{esc(name)}</span></button></li>'
    )


def _stage_drawer(model: NowModel, name: str) -> str:
    rows = [
        (repo_label(r.repo), float(c), 0.0, str(c))
        for r in model.repos
        if (c := next((s.count for s in r.stages if s.name == name), 0))
    ]
    top = max((v for _, v, _, _ in rows), default=0.0)
    rows = sorted(((lbl, v, top, t) for lbl, v, _, t in rows), key=lambda r: -r[1])
    return hbars(f"{name} by repo", rows)


def _why_drawer(flow: FlowModel) -> str:
    reasons = sorted(flow.not_dispatchable, key=lambda r: (-r.count, r.reason))
    top = max((r.count for r in reasons), default=0)
    rows = [
        (
            esc(_REASON_LABEL.get(r.reason, r.reason.replace("_", " ").capitalize())),
            float(r.count),
            float(top),
            str(r.count),
        )
        for r in reasons
    ]
    return hbars("Why ready issues are held", rows, empty="Every ready issue is dispatchable.")


def _drawer(key: str, inner: str) -> str:
    return f'<div class="drawer" id="pipe-{key}" data-pipe="{key}" hidden>{inner}</div>'


def render_flow(model: NowModel) -> str:
    flow, t = model.flow, model.totals
    focus, _ = _focus_stage(flow)
    nodes = "".join(_node(n, _count(flow, n), n == focus, False) for n in _TRACK)
    nodes += _node(DONE, flow.done_24h, True, True)
    rework = _count(flow, "Needs rework")
    arc = (
        '<div class="arc" aria-hidden="true"><span class="arc-curve"></span>'
        f'<span class="arc-note">{rework} back for rework</span></div>'
        if rework
        else ""
    )
    held = sum(r.count for r in flow.not_dispatchable)
    why = (
        f"{held} of {t.ready_issues} ready issues cannot dispatch "
        '(<button type="button" class="linkbtn" id="why-btn" data-pipe="why" '
        'aria-pressed="false">why</button>).'
        if held
        else "Every ready issue is dispatchable."
        if t.ready_issues
        else "No ready issues."
    )
    done_note = (
        "See the progress chart for merges over time."
        if flow.done_24h is not None
        else "Merged-24h is unknown: the rollup is not current."
    )
    drawers = "".join(_drawer(slug(n), _stage_drawer(model, n)) for n in _TRACK)
    drawers += _drawer("merged", f'<p class="dim small">{done_note}</p>')
    drawers += _drawer("why", _why_drawer(flow))
    return (
        '<section class="panel n-flow" id="flow" aria-labelledby="flow-h">'
        f'<div><h2 id="flow-h">{headline(flow)}</h2>'
        f'<p class="sub">{t.active_issues} issues in flight. {why} '
        '<span class="as-of">as of snapshot</span></p></div>'
        f'<ol class="track" aria-label="Pipeline: issues per stage, left to right">{nodes}</ol>'
        f"{arc}{drawers}</section>"
    )
