"""The Capacity panel: three meters (workers, reviewers, CI runners), the full ones in focus.

The headline is the takeaway ("Workers at capacity"). Each meter is a button that opens a
per-repo drawer. A cap that was not reported is an open, dashed track with the words "cap
not reported", never a made-up scale; an over-cap value is stated in words, not colour.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..now_types import CapacityModel, NowModel, RepoWorkers, RunnerRepo
from .now_bars import hbars, repo_label
from .now_fmt import esc, fmt_float


@dataclass(frozen=True)
class Meter:
    key: str
    name: str
    live: int | None  # None: not measured
    cap: int | None  # None / 0: no cap reported

    @property
    def full(self) -> bool:
        return bool(self.cap) and self.live is not None and self.live >= self.cap


def meters(cap: CapacityModel) -> tuple[Meter, ...]:
    online = sum(r.online for r in cap.runners) if cap.runners else None
    slots = sum(r.capacity for r in cap.runners) if cap.runners else None
    return (
        Meter("workers", "Workers", cap.workers_live, cap.workers_cap),
        Meter("reviewers", "Reviewers", cap.reviewers_live, cap.reviewers_cap),
        Meter("runners", "CI runners", online, slots),
    )


def headline(ms: tuple[Meter, ...]) -> str:
    full = [m.name for m in ms if m.full]
    return f"{' and '.join(full)} at capacity" if full else "Capacity has headroom"


def _track(m: Meter) -> str:
    if not m.cap:
        return (
            '<svg class="trk open" viewBox="0 0 100 10" preserveAspectRatio="none" '
            'aria-hidden="true"><line x1="0" x2="100" y1="5" y2="5"/></svg>'
        )
    pct = fmt_float(min(100.0, 100.0 * (m.live or 0) / m.cap))
    return (
        '<svg class="trk" viewBox="0 0 100 10" preserveAspectRatio="none" aria-hidden="true">'
        f'<rect class="bg" x="0" y="0" width="100" height="10"/>'
        f'<rect class="fill" x="0" y="0" width="{pct}" height="10"/></svg>'
    )


def _meter(m: Meter) -> str:
    live = "—" if m.live is None else str(m.live)
    cap = "?" if m.cap is None else "∞" if m.cap == 0 else str(m.cap)
    words = (
        "not measured"
        if m.live is None
        else "cap not reported"
        if m.cap is None
        else "over cap"
        if m.cap and m.live > m.cap
        else ""
    )
    note = f'<span class="note">{words}</span>' if words else ""
    cls = "meter" + (" focus" if m.full else "")
    return (
        f'<button type="button" class="{cls}" id="meter-{m.key}" data-cap="{m.key}" '
        f'aria-pressed="false" aria-label="{esc(m.name)}: {esc(live)} of {esc(cap)}">'
        f'<span class="mname">{esc(m.name)}{note}</span>'
        f'<span class="val"><span class="num">{esc(live)}</span> / {esc(cap)}</span>'
        f"{_track(m)}</button>"
    )


def _workers_rows(rows: tuple[RepoWorkers, ...]) -> list[tuple[str, float, float, str]]:
    shown = [r for r in rows if r.live or r.cap]
    top = float(max((r.live for r in shown), default=0) or 1)
    return [
        (
            repo_label(r.repo),
            float(r.live),
            float(r.cap) if r.cap else top,
            f"{r.live}/{r.cap if r.cap else '∞'}",
        )
        for r in shown
    ]


def _runner_rows(rows: tuple[RunnerRepo, ...]) -> list[tuple[str, float, float, str]]:
    return [
        (
            repo_label(r.repo),
            float(r.online),
            float(r.capacity),
            f"{r.online}/{r.capacity} · {r.parked} parked · demand {r.demand}",
        )
        for r in rows
        if r.online or r.capacity
    ]


def _drawer(m: Meter, cap: CapacityModel) -> str:
    if m.key == "runners":
        inner = hbars("CI runners by repo (online / slots)", _runner_rows(cap.runners))
    else:
        by = cap.workers_by_repo if m.key == "workers" else cap.reviewers_by_repo
        inner = hbars(f"{m.name} by repo (live / cap)", _workers_rows(by))
    return f'<div class="drawer" id="cap-{m.key}" data-cap="{m.key}" hidden>{inner}</div>'


def render_capacity(model: NowModel) -> str:
    cap = model.capacity
    ms = meters(cap)
    waiting = next((s.count for s in model.flow.stages if s.name == "Dispatchable"), 0)
    workers_full = ms[0].full
    sub = (
        f"{waiting} dispatchable issues are waiting on a free worker."
        if workers_full and waiting
        else ""
    )
    return (
        '<section class="panel n-cap" id="capacity" aria-labelledby="cap-h">'
        f'<div><h2 id="cap-h">{esc(headline(ms))}</h2><p class="sub">{esc(sub)}</p></div>'
        f'<div class="meters">{"".join(_meter(m) for m in ms)}</div>'
        f"{''.join(_drawer(m, cap) for m in ms)}</section>"
    )
