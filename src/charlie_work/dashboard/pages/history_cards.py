"""The History metric cards: one button per metric, wrong-way movers first.

A card says the metric's plain name, its window summary (total, median, mean or latest,
from the registry), the change vs the prior window in words, and a sparkline. A card
whose metric moved the wrong way by ``history_model.WRONG_WAY`` or more carries
``data-bad`` (the page's only accent colour) and says so to screen readers too; an
uninstrumented or failed metric shows "—" and says why, never a zero. Every dynamic value
is escaped (``now_fmt.esc``); the sparkline is SVG attributes only (CSP: no inline style).
"""

from __future__ import annotations

from ..history_data import MetricData
from ..history_model import Card, delta_text, fmt_value, on_grid
from .now_fmt import esc, slug


def card_id(metric_id: str) -> str:
    return "card-" + slug(metric_id)


def spark(values: list[float | None]) -> str:
    """A 100x30 sparkline over the bucket grid; gaps where a bucket has no value and a
    dot for an isolated value. Empty when nothing was observed."""
    seen = [v for v in values if v is not None]
    if not seen:
        return ""
    top = max(max(seen), 1e-9)
    n = len(values)
    parts: list[str] = []
    run = 0
    for i, v in enumerate(values):
        if v is None:
            if run == 1:
                parts.append("h0.1")
            run = 0
            continue
        x = 50.0 if n == 1 else i / (n - 1) * 100
        parts.append(f"{'L' if run else 'M'}{x:.1f} {28 - v / top * 26:.1f}")
        run += 1
    if run == 1:
        parts.append("h0.1")
    return (
        '<svg class="spark" viewBox="0 0 100 30" preserveAspectRatio="none" '
        f'aria-hidden="true" focusable="false"><path d="{"".join(parts)}"/></svg>'
    )


def _stat_words(card: Card) -> str:
    words = [card.summary_word]
    if card.approx:
        words.append("approx.")
    if card.partial:
        words.append("partial")
    return " · ".join(words)


def render_card(card: Card, metric: MetricData, grid: tuple[str, ...]) -> str:
    """One card button. ``grid`` is the window's bucket starts (the sparkline's x)."""
    if card.state == "ok":
        value = fmt_value(card.value, card.unit)
        if card.prs is not None:
            # "<prs> PRs (<value> attempts)": the distinct-PR count leads (issue #2476).
            value = f"{fmt_value(card.prs, card.unit)} PRs ({value} attempts)"
        delta = delta_text(card.change)
        stat = _stat_words(card)
        line = spark(on_grid(metric.headline.points, grid))
    else:
        value = "—"
        delta = "not instrumented yet" if card.state == "missing" else "could not be drawn"
        stat = "no data" if card.state == "missing" else "error"
        line = ""
    bad = ' data-bad="1"' if card.bad else ""
    flag = (
        '<span class="sr"> (moved the wrong way: worth your attention)</span>' if card.bad else ""
    )
    return (
        f'<button type="button" class="card" id="{esc(card_id(card.metric_id))}" '
        f'data-k="{esc(card.metric_id)}" data-state="{esc(card.state)}" aria-pressed="false"{bad}>'
        f'<span class="cname">{esc(card.name)}</span>'
        f'<span class="cval num">{esc(value)}</span>'
        f'<span class="cdelta">{esc(delta)}{flag}</span>'
        f'<span class="cstat">{esc(stat)}</span>{line}</button>'
    )
