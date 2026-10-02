"""Stage drill-down (``/flow/<stage>``): the issues behind one Flow number on Now.

Label stages, Dispatchable and Active list issues from the same snapshot read the Now model
was built from (``ModelState.sources``), through the same stage definitions
(``now_model.counted_stages``, ``LabelConfig.active``), so the list and the number come
from one tick. Done (24h) is a throughput, not a snapshot stock: it lists the merges
``dashboard.db`` holds for the last 24h, per registered repo, with the repo drill-down's
first-signal dedup.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo
from pathlib import Path

from ...config import LabelConfig
from ..now_access import dict_list, label_set, pos_int, snapshot_data
from ..now_model import counted_stages
from ..now_types import NowModel, SourcesRead
from ..pages.now_fmt import slug
from ..timeutil import parse_ts
from .repo import MAX_LIMIT, merges_for
from .types import DrillError, MergeRow, check_slug, open_history

DONE = "done-24h"
_DAY = timedelta(hours=24)


@dataclass(frozen=True)
class StageItem:
    repo: str
    number: int
    title: str | None
    labels: tuple[str, ...]


@dataclass(frozen=True)
class StageListing:
    key: str  # the URL slug
    name: str
    count: int | None  # the number Now shows (None: unknown)
    items: tuple[StageItem, ...]
    merges: tuple[tuple[str, MergeRow], ...]  # Done (24h) only: (repo, merge)
    as_of: datetime | None  # the snapshot tick (or the rollup read, for Done)
    unreadable: tuple[str, ...]  # repos whose snapshot could not be read (not counted)
    note: str | None
    repo: str | None = None  # the one repo the list is narrowed to (None: the fleet)


def stage_names(sources: SourcesRead | None) -> dict[str, str]:
    """URL slug -> display name for every stage Now links (derived, never a second list)."""
    names = ["Dispatchable"] + [n for n, _ in counted_stages(_labels(sources))]
    out = {slug(n): n for n in names}
    out["active"] = "Active"
    out[DONE] = "Done (24h)"
    return out


def _labels(sources: SourcesRead | None) -> LabelConfig:
    return sources.labels if sources is not None else LabelConfig()


def _now_count(model: NowModel, name: str, key: str, repo: str | None) -> int | None:
    if repo is not None:  # Now's per-repo ledger counts the label stages and Dispatchable
        flow = next((r for r in model.repos if r.repo == repo), None)
        stages = flow.stages if flow else ()
        return next((s.count for s in stages if s.name == name), None)
    if key == "active":
        return model.totals.active_issues
    if key == DONE:
        return model.flow.done_24h
    return next((s.count for s in model.flow.stages if s.name == name), None)


def _issues(sources: SourcesRead, key: str) -> tuple[StageItem, ...]:
    labels = sources.labels
    wanted = dict((slug(n), lab) for n, lab in counted_stages(labels))
    out: list[StageItem] = []
    for repo in sources.repos:
        for issue in dict_list(snapshot_data(repo), "issues"):
            have = label_set(issue)
            if key == "dispatchable":
                hit = issue.get("dispatchable") is True
            elif key == "active":
                hit = bool(have & labels.active)
            else:
                hit = wanted.get(key) in have
            number = pos_int(issue.get("number"))
            if hit and number is not None:
                title = issue.get("title")
                out.append(
                    StageItem(repo.key, number, title if isinstance(title, str) else None,
                              tuple(sorted(have)))
                )  # fmt: skip
    return tuple(sorted(out, key=lambda i: (i.repo, -i.number)))


def _done(
    model: NowModel, db_path: Path, now: datetime, tz: tzinfo | None
) -> tuple[tuple[tuple[str, MergeRow], ...], str | None]:
    db, err = open_history(db_path)
    if db is None:
        return (), err.message if err else "history unavailable"
    cutoff = now - _DAY
    rows: list[tuple[str, MergeRow]] = []
    try:
        for f in model.freshness:
            for m in merges_for(db, f.repo, MAX_LIMIT, tz):
                if parse_ts(m.ts) >= cutoff:
                    rows.append((f.repo, m))
    except sqlite3.Error as exc:
        return (), f"history unreadable: {exc}"
    finally:
        db.close()
    rows.sort(key=lambda r: r[1].ts, reverse=True)
    return tuple(rows), None


def stage_listing(
    key: str,
    model: NowModel | None,
    sources: SourcesRead | None,
    db_path: Path | None,
    now: datetime,
    *,
    repo: str | None = None,
    tz: tzinfo | None = None,
) -> StageListing | DrillError:
    """The items behind the Now stage ``key`` (a URL slug such as ``in-progress``).

    ``repo`` narrows the list to one registered repo (the repo page links that way).
    """
    names = stage_names(sources)
    if key not in names:
        return DrillError("not_found", "no such Flow stage")
    if repo is not None and (bad := check_slug(repo)) is not None:
        return bad
    if model is None or sources is None:
        return DrillError("unavailable", "the Now model has not been collected yet")
    if repo is not None and all(f.repo != repo for f in model.freshness):
        return DrillError("not_found", "no such repo in the fleet registry")
    name = names[key]
    count = _now_count(model, name, key, repo)
    if key == DONE:
        if db_path is None:
            return DrillError("unavailable", "no dashboard.db configured")
        merges, err = _done(model, db_path, now, tz)
        merges = tuple(m for m in merges if repo is None or m[0] == repo)
        drift = count is not None and count != len({(r, m.issue or m.pr) for r, m in merges})
        note = err or (
            "Now's count was taken at the last collector tick; this list is read now."
            if drift
            else None
        )
        return StageListing(key, name, count, (), merges, now, (), note, repo)
    items = tuple(i for i in _issues(sources, key) if repo is None or i.repo == repo)
    unreadable = tuple(
        r.key for r in sources.repos if r.snapshot.data is None and repo in (None, r.key)
    )
    note = None
    if count is not None and count != len(items):
        note = (
            f"Now counts {count} (from each repo's backlog reachability); the snapshots list "
            f"{len(items)} issue(s) individually."
            if key == "dispatchable"
            else f"Now counts {count}; the snapshots list {len(items)} issue(s)."
        )
    return StageListing(key, name, count, items, (), model.generated_at, unreadable, note, repo)
