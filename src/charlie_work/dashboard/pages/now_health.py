"""Source health: the header pill and the panel it opens.

Exceptions (alarms, stale or unreadable sources, a paused fleet, a silent supervisor) are not
decisions the operator makes, so they never count toward "need you". They collapse into one
pill in the header: a red dot and what is wrong, or a grey "N sources fresh". The pill opens
a panel with the reasons and every source's last pass (each linking to its repo drill-down).
"""

from __future__ import annotations

from ..now_types import NeedsMeItem, NowModel, RepoFreshness
from .now_fmt import age, esc, link, repo_url, short_repo

EXCEPTIONS = "Exceptions"
_OTHER_LABEL = {"paused": "fleet paused", "supervisor": "supervisor not beating"}


def needs_rows(model: NowModel) -> tuple[NeedsMeItem, ...]:
    """The rows that are the operator's to decide (everything except Exceptions)."""
    return tuple(i for i in model.needs_me if i.group != EXCEPTIONS)


def exception_rows(model: NowModel) -> tuple[NeedsMeItem, ...]:
    return tuple(i for i in model.needs_me if i.group == EXCEPTIONS)


def _broken(model: NowModel) -> tuple[RepoFreshness, ...]:
    return tuple(f for f in model.freshness if f.stale or f.error)


def _other(model: NowModel) -> tuple[NeedsMeItem, ...]:
    """Exceptions that are not a per-repo source (those come from ``model.freshness``)."""
    return tuple(i for i in exception_rows(model) if i.kind != "stale_source" or i.repo == "fleet")


def pill_text(model: NowModel) -> str:
    """What is wrong, in a few words; ``N sources fresh`` when nothing is."""
    parts: list[str] = []
    unreadable = [short_repo(f.repo) for f in _broken(model) if f.error]
    stale = [short_repo(f.repo) for f in _broken(model) if not f.error]
    if unreadable:
        parts.append(f"{', '.join(unreadable)} unreadable")
    if stale:
        parts.append(f"{', '.join(stale)} stale")
    alarms = sum(1 for i in _other(model) if i.kind == "alarm")
    if alarms:
        parts.append(f"{alarms} alarm{'s' if alarms != 1 else ''}")
    for i in _other(model):
        if i.kind in _OTHER_LABEL:
            if _OTHER_LABEL[i.kind] not in parts:
                parts.append(_OTHER_LABEL[i.kind])
        elif i.kind == "stale_source":
            parts.append("runner allocation stale")
    return " · ".join(parts) or f"{len(model.freshness)} sources fresh"


def _source(f: RepoFreshness) -> str:
    state = " · unreadable" if f.error else " · stale" if f.stale else ""
    name = link(repo_url(f.repo), short_repo(f.repo), "rl", f.repo)
    return (
        f'<li>{name} <span class="dim">last pass {esc(age(f.age_seconds))} ago{state}</span></li>'
    )


def _exception(i: NeedsMeItem) -> str:
    where = short_repo(i.repo) if i.repo != "fleet" else "fleet"
    return f"<li><b>{esc(where)}</b> {esc(i.reason)}</li>"


def render_health(model: NowModel | None) -> str:
    """The pill (a button) and its panel; empty before the first read."""
    if model is None:
        return ""
    bad = bool(_broken(model) or _other(model))
    text = esc(pill_text(model))
    issues = "".join(_exception(i) for i in exception_rows(model))
    problems = f'<ul class="hlist" aria-label="Problems">{issues}</ul>' if issues else ""
    sources = "".join(_source(f) for f in model.freshness)
    return (
        f'<button type="button" id="health-btn" class="health{"" if bad else " ok"}" '
        'aria-expanded="false" aria-controls="health-pop">'
        f'<span class="dot" aria-hidden="true"></span>{text}</button>'
        '<div id="health-pop" class="healthpop" role="group" aria-label="Source health" hidden>'
        f'{problems}<p class="dtitle">Sources · stale after '
        f"{esc(age(model.stale_threshold_seconds))}</p>"
        f'<ul class="hlist">{sources}</ul></div>'
    )
