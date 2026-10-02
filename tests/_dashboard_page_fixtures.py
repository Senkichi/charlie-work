"""Shared literal Now models and an HTML collector for the Now page render tests."""

from __future__ import annotations

from datetime import UTC, datetime
from html.parser import HTMLParser

from charlie_work.dashboard.now_types import (
    NEEDS_ME_GROUPS,
    CapacityModel,
    FlowModel,
    FlowStage,
    GroupSummary,
    NeedsMeItem,
    NowModel,
    NowTotals,
    RepoFlow,
    RepoFreshness,
    RepoWorkers,
    RunnerRepo,
    UnreachableReason,
)
from charlie_work.dashboard.pages.now import render_now
from charlie_work.dashboard.read_model import ModelState

NOW = datetime(2026, 10, 1, 20, 29, 39, tzinfo=UTC)
CW = "Senkichi/charlie-work"
SW = "Senkichi/swole"
VERDICT_CMD = (
    "charlie --repo 'C:\\r\\cw' verdict --pr 7 --decision <approved|request_changes|blocked>"
)
REQUEUE_CMD = "charlie --repo 'C:\\r\\cw' unescalate --pr 7"
QUEUE_CMD = "charlie --repo 'C:\\r\\sw' unescalate --issue 12"
STAGES = ("Dispatchable", "Queued", "In progress", "PR open", "Reviewing", "Needs rework")


def _stages(*counts: int) -> tuple[FlowStage, ...]:
    return tuple(FlowStage(n, None, c) for n, c in zip(STAGES, counts, strict=True))


def _item(group: str, **kw) -> NeedsMeItem:
    base = dict(
        kind="operator_queue",
        severity="action",
        repo=SW,
        age_seconds=None,
        reason="Operator queue: #12 a title",
        command=QUEUE_CMD,
        as_of_snapshot=True,
        number=12,
        group=group,
    )
    base.update(kw)
    return NeedsMeItem(**base)


ITEMS = (
    _item(
        "Exceptions",
        kind="alarm",
        severity="anomaly",
        repo=CW,
        reason="loop_errors: <script>alert(1)</script> & more",
        command=None,
        number=None,
        as_of_snapshot=False,
    ),
    _item(
        "Exceptions",
        kind="stale_source",
        severity="warn",
        reason="snapshot is 700s old",
        command=None,
        number=None,
        age_seconds=700.0,
    ),
    _item(
        "Awaiting your verdict",
        kind="human_needed",
        repo=CW,
        reason="Human needed: PR #7 (issue #6) awaits an operator verdict",
        command=VERDICT_CMD,
        secondary_command=REQUEUE_CMD,
        number=6,
    ),
    _item("Human needed", kind="human_needed", reason="Human needed: #9 x", number=9),
    _item("Operator queue"),
)


def _model(items: tuple[NeedsMeItem, ...] = ITEMS, **kw) -> NowModel:
    groups = tuple(
        GroupSummary(g, sum(1 for i in items if i.group == g), None) for g in NEEDS_ME_GROUPS
    )
    base = dict(
        generated_at=NOW,
        stale_threshold_seconds=630.0,
        freshness=(
            RepoFreshness(CW, NOW, 24.0, False, None),
            RepoFreshness(SW, NOW, 700.0, True, None),
        ),
        needs_me=items,
        flow=FlowModel(
            _stages(5, 0, 4, 2, 0, 1),
            None,
            (UnreachableReason("blocked_by_open_dependency", 84, (f"{SW}#139",)),),
        ),
        capacity=CapacityModel(
            workers_live=4,
            workers_cap=None,
            workers_by_repo=(RepoWorkers(CW, 3, None), RepoWorkers(SW, 1, 2)),
            reviewers_live=0,
            reviewers_cap=6,
            reviewers_by_repo=(),
            runners=(RunnerRepo(SW, 3, 2, None, 1, 3),),
            runners_age_seconds=72.0,
            runners_stale=False,
            capped_demand_now=False,
            capped_repos=(),
        ),
        totals=NowTotals(119, 7, 11, 15, 4),
        repos=(
            RepoFlow(CW, _stages(2, 0, 3, 0, 0, 0), 3),
            RepoFlow(SW, _stages(1, 0, 0, 1, 0, 0), 2),
        ),
        needs_me_groups=groups,
    )
    base.update(kw)
    return NowModel(**base)


class _Collect(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.text: list[str] = []
        # Bodies of <script> elements as the parser sees them, so an end tag a regex
        # would miss (``</script >``) cannot hide an inline script from the CSP checks.
        self.script_bodies: list[str] = []
        self._in_script = False

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        if tag == "script":
            self._in_script = True
            self.script_bodies.append("")

    def handle_endtag(self, tag):
        if tag == "script":
            self._in_script = False

    def handle_startendtag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_data(self, data):
        if self._in_script:
            self.script_bodies[-1] += data
        self.text.append(data)


def _parse(html_text: str) -> _Collect:
    p = _Collect()
    p.feed(html_text)
    return p


def _page(model: NowModel | None = None, **state) -> str:
    return render_now(ModelState(model=model or _model(), **state), poll_seconds=15)
