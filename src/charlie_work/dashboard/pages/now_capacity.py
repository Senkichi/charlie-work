"""Right-rail Capacity bullet bars and the compact per-repo ledger.

A cap that was not reported is drawn as an open, dashed track with the words
"cap not reported" -- never a made-up scale. The per-repo ledger draws every stage cell on
ONE shared scale and every runner cell on one shared slot scale, so cells compare across
repos.
"""

from __future__ import annotations

from ..now_types import CapacityModel, NowModel, RepoFreshness, RepoWorkers, RunnerRepo
from .now_fmt import age, esc, fmt_float, link, repo_url, short_repo

_TRACK = 240.0
_STAGE_ABBR = (
    ("Dispatchable", "Disp"),
    ("Queued", "Q"),
    ("In progress", "Prog"),
    ("PR open", "PR"),
    ("Reviewing", "Rev"),
    ("Needs rework", "Rwk"),
)


def _bullet(live: int, cap: int | None) -> str:
    if cap:
        scale = float(max(cap, live, 1))
        fill = _TRACK * live / scale
        tick = _TRACK * cap / scale
        hot = " hot" if live > cap else ""
        inner = (
            f'<rect class="trk" x="0" y="5" width="{fmt_float(_TRACK)}" height="6"/>'
            f'<rect class="fill{hot}" x="0" y="3" width="{fmt_float(fill)}" height="10"/>'
            f'<line class="captick" x1="{fmt_float(tick)}" x2="{fmt_float(tick)}" y1="0" y2="16"/>'
        )
    else:
        # Unknown cap: the fill reaches 80% of the track and the rest stays open-ended.
        fill = _TRACK * 0.8 if live else 0.0
        inner = (
            f'<rect class="fill" x="0" y="3" width="{fmt_float(fill)}" height="10"/>'
            f'<line class="trk-open" x1="{fmt_float(fill)}" x2="{fmt_float(_TRACK)}" '
            'y1="8" y2="8"/>'
        )
    return (
        f'<svg class="bullet-svg" viewBox="0 0 {fmt_float(_TRACK)} 16" aria-hidden="true">'
        f"{inner}</svg>"
    )


def _over(live: int, cap: int | None) -> str:
    """Over-cap is stated in words + glyph (text-warn prepends the triangle), not colour only."""
    return '<span class="over text-warn">over cap</span>' if cap and live > cap else ""


def _cap_row(name: str, href: str, live: int | None, cap: int | None, small: str) -> str:
    if live is None:
        bar = '<span class="unknown">not measured</span>'
        val = '<span class="unk">—</span> live'
    else:
        bar = _bullet(live, cap)
        if not cap:
            bar += '<span class="unknown">cap not reported</span>'
        cap_txt = link(href + "#cap", cap) if cap else "cap ?"
        val = f"{link(href, live)} live / {cap_txt}{_over(live, cap)}"
    return (
        f'<div class="cap-row"><span class="who">{esc(name)}<small>{small}</small></span>'
        f'<span class="bar">{bar}</span><span class="val">{val}</span></div>'
    )


def _by_repo(rows: tuple[RepoWorkers, ...]) -> str:
    busy = [r for r in rows if r.live]
    if not busy:
        return "none running"
    return " · ".join(f"{esc(short_repo(r.repo))} {link(repo_url(r.repo), r.live)}" for r in busy)


def _runner_row(cap: CapacityModel) -> str:
    if not cap.runners:
        return (
            '<div class="cap-row"><span class="who">CI runners<small>no allocation event'
            '</small></span><span class="bar"><span class="unknown">not reported</span></span>'
            '<span class="val">—</span></div>'
        )
    online = sum(r.online for r in cap.runners)
    slots = sum(r.capacity for r in cap.runners)
    parked = sum(r.parked for r in cap.runners)
    demand = sum(r.demand for r in cap.runners)
    busy_known = [r.busy for r in cap.runners if r.busy is not None]
    busy = f"busy {sum(busy_known)}" if busy_known else "busy not reported"
    small = f"{link('/capacity/runners#parked', parked)} parked · demand " + link(
        "/capacity/runners#demand", demand
    )
    return (
        f'<div class="cap-row"><span class="who">CI runners<small>{small} · {busy}'
        f'</small></span><span class="bar">{_bullet(online, slots)}</span><span class="val">'
        f"{link('/capacity/runners', online)} online / {link('/capacity/runners#slots', slots)}"
        f"{_over(online, slots)}</span></div>"
    )


def render_capacity(model: NowModel) -> str:
    cap = model.capacity
    if cap.runners_age_seconds is None:
        meta = "runners not reported"
    elif cap.runners_stale:
        old = esc(age(cap.runners_age_seconds))
        meta = f'<span class="text-warn">runners {old} old, stale</span>'
    else:
        meta = f"runners {esc(age(cap.runners_age_seconds))} ago"
    if cap.capped_demand_now:
        who = ", ".join(short_repo(r) for r in cap.capped_repos) or "fleet budget"
        meta += f' · <span class="text-warn">capped demand now: {esc(who)}</span>'
    else:
        meta += " · no capped demand now"
    return (
        '<section class="cap-now" aria-labelledby="cap-h">'
        f'<h2 id="cap-h">Capacity <span class="meta">{meta}</span></h2>'
        + _cap_row(
            "Workers",
            "/capacity/workers",
            cap.workers_live,
            cap.workers_cap,
            _by_repo(cap.workers_by_repo),
        )
        + _cap_row(
            "Reviewers",
            "/capacity/reviewers",
            cap.reviewers_live,
            cap.reviewers_cap,
            _by_repo(cap.reviewers_by_repo),
        )
        + _runner_row(cap)
        + "</section>"
    )


def _match(key: str, rows: tuple) -> object | None:
    """Exact repo-key match, else the same short name (runner targets may omit the owner)."""
    exact = next((r for r in rows if r.repo == key), None)
    if exact is not None:
        return exact
    return next((r for r in rows if short_repo(r.repo) == short_repo(key)), None)


def _mini(count: int, scale: int, cls: str = "mbar") -> str:
    h = 12.0 * count / scale if scale else 0.0
    rect = (
        f'<rect class="{cls}" x="0" y="{fmt_float(12 - h)}" width="4" height="{fmt_float(h)}"/>'
        if count
        else '<rect class="mzero" x="0" y="11" width="4" height="1"/>'
    )
    return f'<svg class="mini" viewBox="0 0 4 12" aria-hidden="true">{rect}</svg>'


def _runner_cell(r: RunnerRepo | None, slot_scale: int, key: str) -> str:
    if r is None:
        return '<span class="muted">—</span>'
    w = 60.0 / slot_scale if slot_scale else 0.0
    parts = [f'<rect class="trk" x="0" y="4" width="{fmt_float(w * r.capacity)}" height="4"/>']
    parts.append(f'<rect class="fill" x="0" y="2" width="{fmt_float(w * r.online)}" height="8"/>')
    if r.parked:
        parts.append(
            f'<rect class="parked" x="{fmt_float(w * r.online)}" y="2" '
            f'width="{fmt_float(w * r.parked)}" height="8"/>'
        )
    if r.demand:
        d = min(w * r.demand, 60.0)
        parts.append(
            f'<line class="demand" x1="{fmt_float(d)}" x2="{fmt_float(d)}" y1="0" y2="12"/>'
        )
    short = r.demand > r.online
    glyph = (
        '<span class="text-warn"><span class="sr">demand above online</span></span>'
        if short
        else ""
    )
    return (
        f'<span class="rnr"><svg class="rsvg" viewBox="0 0 60 12" aria-hidden="true">'
        f"{''.join(parts)}</svg>{glyph}{link(repo_url(key, view='runners'), r.online)}/"
        f"{r.capacity}"
        + (f' <span class="muted">p{r.parked}</span>' if r.parked else "")
        + (f' <span class="muted">d{r.demand}</span>' if r.demand else "")
        + "</span>"
    )


def _pass_cell(f: RepoFreshness | None, key: str) -> str:
    if f is None:
        return '<td class="pass">—</td>'
    if f.error:
        return (
            f'<td class="pass"><span class="text-danger" title="{esc(f.error)}">'
            f'{link(repo_url(key), age(f.age_seconds))}<span class="sr"> read error</span></span></td>'
        )
    if f.stale:
        return (
            f'<td class="pass"><span class="text-warn" title="stale">'
            f'{link(repo_url(key), age(f.age_seconds))}<span class="sr"> stale</span></span></td>'
        )
    return f'<td class="pass">{link(repo_url(key), age(f.age_seconds))}</td>'


def render_repo_ledger(model: NowModel) -> str:
    if not model.repos:
        return ""
    fresh = {f.repo: f for f in model.freshness}
    stage_scale = max((s.count for r in model.repos for s in r.stages), default=0)
    slot_scale = max((r.capacity for r in model.capacity.runners), default=0)
    head = "".join(f'<th class="num" title="{esc(n)}">{esc(a)}</th>' for n, a in _STAGE_ABBR)
    body = []
    for repo in sorted(model.repos, key=lambda r: (-r.need_you, r.repo)):
        f = fresh.get(repo.repo)
        counts = {s.name: s.count for s in repo.stages}
        cells = "".join(
            f'<td class="stg">{_mini(counts.get(n, 0), stage_scale)}'
            f"{link(repo_url(repo.repo, stage=a.lower()), counts.get(n, 0))}</td>"
            for n, a in _STAGE_ABBR
        )
        w = _match(repo.repo, model.capacity.workers_by_repo)
        workers = (
            f"{link(repo_url(repo.repo, view='workers'), w.live)}/{w.cap if w.cap else '?'}"
            if isinstance(w, RepoWorkers)
            else '<span class="muted">—</span>'
        )
        runner = _match(repo.repo, model.capacity.runners)
        rcell = _runner_cell(
            runner if isinstance(runner, RunnerRepo) else None, slot_scale, repo.repo
        )
        stale = " is-stale" if f is not None and f.stale else ""
        need = link(repo_url(repo.repo, view="needs"), repo.need_you)
        body.append(
            f'<tr class="repo-row{stale}"><td class="rname" title="{esc(repo.repo)}">'
            f"{esc(short_repo(repo.repo))}</td>{_pass_cell(f, repo.repo)}{cells}"
            f'<td class="num">{workers}</td><td class="ci">{rcell}</td>'
            f'<td class="num need{" has" if repo.need_you else ""}">{need}</td></tr>'
        )
    return (
        '<section class="repos-now" aria-labelledby="repos-h">'
        '<h2 id="repos-h">By repo <span class="meta">shared scales · sorted by need-you'
        " · CI: p parked, d demand</span></h2>"
        # Its own horizontal scroller: at mid widths the table may be wider than the rail,
        # and the page itself never clips (no overflow-x: hidden on html/body).
        '<div class="table-scroll" role="region" aria-label="By-repo table, scrolls sideways" tabindex="0">'
        '<table class="repos"><thead><tr><th class="l">Repo</th><th>Last pass</th>'
        f"{head}"
        '<th class="num" title="live workers / cap">Work</th>'
        '<th title="CI runners online / slots; p parked, d demand">CI</th>'
        '<th class="num" title="Needs-me rows for this repo">You</th></tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div></section>"
    )
