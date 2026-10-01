"""MINIMAL placeholder Now page; the real design replaces this module wholesale.

Pure: ``render_now`` / ``render_fragment`` take a ``ModelState`` and return HTML. Every
dynamic value goes through ``html.escape``.
"""

from __future__ import annotations

from datetime import datetime
from html import escape

from ..read_model import ModelState


def _local(moment: datetime | None) -> str:
    return moment.astimezone().strftime("%H:%M:%S") if moment else "n/a"


def _table(head: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    th = "".join(f"<th>{escape(h)}</th>" for h in head)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def render_fragment(state: ModelState, poll_seconds: int) -> str:
    banner = ""
    if state.collector_error:
        banner = (
            f'<p class="banner" role="alert">collector failing since '
            f"{escape(_local(state.collector_failing_since))} (local): "
            f"{escape(state.collector_error)}</p>"
        )
    head = (
        f'<section id="now" hx-get="/now/fragment" '
        f'hx-trigger="every {int(poll_seconds)}s" hx-swap="outerHTML">'
    )
    model = state.model
    if model is None:
        return f"{head}{banner}<p>collecting...</p></section>"
    needs = _table(
        ("severity", "kind", "repo", "reason", "command"),
        [(i.severity, i.kind, i.repo or "", i.reason, i.command or "") for i in model.needs_me],
    )
    flow = _table(("stage", "count"), [(s.name, s.count) for s in model.flow.stages])
    cap = model.capacity
    capacity = _table(
        ("what", "live", "cap"),
        [
            ("workers", cap.workers_live, cap.workers_cap),
            ("reviewers", cap.reviewers_live, cap.reviewers_cap),
        ],
    )
    return (
        f"{head}{banner}<p>as of {escape(_local(model.generated_at))} (local)</p>"
        f"<h2>Needs me</h2>{needs}<h2>Flow</h2>{flow}<h2>Capacity</h2>{capacity}</section>"
    )


def render_now(state: ModelState, theme: str = "auto", *, poll_seconds: int = 20) -> str:
    return (
        f'<!doctype html><html lang="en" data-theme="{escape(theme)}"><head>'
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="htmx-config" content=\'{"includeIndicatorStyles":false}\'>'
        "<title>Fleet dashboard</title>"
        '<link rel="stylesheet" href="/static/dashboard.css">'
        '<script src="/static/htmx.min.js" defer></script></head><body><main>'
        f"<h1>Fleet - Now</h1>{render_fragment(state, poll_seconds)}</main></body></html>"
    )
