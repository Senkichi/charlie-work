"""Pure builder for the dashboard "Now" model (spec section 4, "Now").

``build_now_model`` does no I/O and never calls GitHub: it folds the already-read
sources (``SourcesRead``) and the shared alarm findings into frozen value objects.
Snapshot ``issues`` / ``prs`` / ``backlog_reachability`` were fetched from GitHub when
the loop pass wrote the snapshot, so every field derived from them is marked
``as_of_snapshot`` (snapshot recon section 1).

Commands shown in Needs-me rows use only subcommands that exist in ``cli.py``
(``unescalate``, ``verdict``, ``fleet resume``); ``tests/test_dashboard_now_model.py`` parses
each one through ``cli.build_parser`` so a rename breaks the test, not the page.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from . import now_cadence
from .now_access import as_int, dict_list, label_set, snapshot_data
from .now_needs_me import needs_me_items
from .now_types import (
    CapacityModel,
    FindingLike,
    FlowModel,
    FlowStage,
    NowModel,
    NowTotals,
    RepoFreshness,
    RepoRead,
    RepoWorkers,
    RunnerRepo,
    SourcesRead,
    UnreachableReason,
)
from .sources import _parse_utc

# Reasons a Ready issue is held back, in classifier order (backlog_reachability.py
# :243-299). ``missing_ready`` is deliberately absent: such an issue is not Ready at
# all, so it is not "Ready but not dispatchable". ``dispatchable`` is the other side.
NOT_DISPATCHABLE_REASONS = (
    "terminal_label",
    "active_label",
    "operator_claimed",
    "mention_covered_awaiting_operator",
    "blocked_by_open_dependency",
    "unidentified",
)


def _freshness(
    repos: Sequence[RepoRead], threshold: float
) -> tuple[tuple[RepoFreshness, ...], dict[str, bool]]:
    rows: list[RepoFreshness] = []
    stale: dict[str, bool] = {}
    for repo in repos:
        snap = repo.snapshot
        # An unreadable snapshot is stale by definition: there is nothing to show.
        is_stale = (
            snap.error is not None or snap.age_seconds is None or snap.age_seconds > threshold
        )
        stale[repo.key] = is_stale
        rows.append(
            RepoFreshness(repo.key, snap.written_at, snap.age_seconds, is_stale, snap.error)
        )
    return tuple(rows), stale


def _flow(sources: SourcesRead) -> tuple[FlowModel, dict[str, int]]:
    labels = sources.labels
    counted = (
        ("Queued", labels.queued),
        ("In progress", labels.in_progress),
        ("PR open", labels.pr_open),
        ("Reviewing", labels.reviewing),
        ("Needs rework", labels.needs_rework),
    )
    stage_counts = {name: 0 for name, _ in counted}
    dispatchable_by_repo: dict[str, int] = {}
    reasons = {reason: 0 for reason in NOT_DISPATCHABLE_REASONS}
    examples: dict[str, list[str]] = {reason: [] for reason in NOT_DISPATCHABLE_REASONS}
    for repo in sources.repos:
        data = snapshot_data(repo)
        issues = dict_list(data, "issues")
        for issue in issues:
            have = label_set(issue)
            for name, label in counted:
                if label in have:
                    stage_counts[name] += 1
        reach = data.get("backlog_reachability")
        if isinstance(reach, dict) and reach.get("observed"):
            dispatchable_by_repo[repo.key] = as_int(reach.get("dispatchable"))
            unreachable = reach.get("unreachable_examples")
            for reason in NOT_DISPATCHABLE_REASONS:
                reasons[reason] += as_int(reach.get(reason))
                nums = unreachable.get(reason) if isinstance(unreachable, dict) else None
                if isinstance(nums, list):
                    examples[reason].extend(f"{repo.key}#{n}" for n in nums)
        else:
            # ``observed`` false means unknown, not "none blocked": fall back to the
            # per-issue flag rather than reporting zero.
            dispatchable_by_repo[repo.key] = sum(
                1 for i in issues if i.get("dispatchable") is True
            )
    stages = (FlowStage("Dispatchable", None, sum(dispatchable_by_repo.values())),) + tuple(
        FlowStage(name, label, stage_counts[name]) for name, label in counted
    )
    breakdown = tuple(
        UnreachableReason(reason, reasons[reason], tuple(examples[reason]))
        for reason in NOT_DISPATCHABLE_REASONS
        if reasons[reason] > 0
    )
    return FlowModel(stages, sources.done_24h, breakdown), dispatchable_by_repo


def _capacity(
    sources: SourcesRead,
    dispatchable_by_repo: dict[str, int],
    now: datetime,
    threshold: float,
) -> CapacityModel:
    workers = tuple(
        RepoWorkers(r.key, len(dict_list(snapshot_data(r), "workers")), r.worker_cap or None)
        for r in sources.repos
        if r.snapshot.data is not None
    )
    reviewers = tuple(
        RepoWorkers(r.key, r.reviewers_live, r.review_cap or None)
        for r in sources.repos
        if r.reviewers_live is not None
    )
    live = sum(w.live for w in workers)
    capped = sorted(
        w.repo
        for w in workers
        if w.cap and w.live >= w.cap and dispatchable_by_repo.get(w.repo, 0) > 0
    )
    fleet_cap = sources.global_worker_cap or None
    fleet_full = (
        fleet_cap is not None and live >= fleet_cap and sum(dispatchable_by_repo.values()) > 0
    )
    runners, age = _runners(sources)
    return CapacityModel(
        workers_live=live,
        workers_cap=fleet_cap,
        workers_by_repo=workers,
        # No readable count anywhere is unknown, not an idle fleet.
        reviewers_live=sum(r.live for r in reviewers) if reviewers else None,
        reviewers_cap=sources.global_review_cap or None,
        reviewers_by_repo=reviewers,
        runners=runners,
        runners_age_seconds=age if age is None else max(0.0, (now - age).total_seconds()),
        runners_stale=age is not None and (now - age).total_seconds() > threshold,
        capped_demand_now=fleet_full or bool(capped),
        capped_repos=tuple(capped),
    )


def _runners(sources: SourcesRead) -> tuple[tuple[RunnerRepo, ...], datetime | None]:
    event = sources.runner_allocation
    if not event:
        return (), None
    payload = event.get("payload")
    targets = payload.get("targets") if isinstance(payload, dict) else None
    rows: list[RunnerRepo] = []
    for target in targets if isinstance(targets, list) else []:
        if not isinstance(target, dict) or not isinstance(target.get("repo"), str):
            continue
        capacity, running = as_int(target.get("capacity")), as_int(target.get("running"))
        repo = target["repo"]
        busy = sources.runner_busy.get(repo)
        rows.append(
            RunnerRepo(
                repo,
                capacity,
                running,
                busy,
                max(capacity - running, 0),
                as_int(target.get("demand")),
            )
        )
    return tuple(sorted(rows, key=lambda r: r.repo)), _parse_utc(event.get("ts"))


def _totals(sources: SourcesRead) -> NowTotals:
    ready = active = linked = unlinked = live = 0
    for repo in sources.repos:
        data = snapshot_data(repo)
        ready += as_int(data.get("ready_issue_count"))
        active += as_int(data.get("active_issue_count"))
        linked += as_int(data.get("open_linked_pr_count"))
        unlinked += as_int(data.get("unlinked_pr_count"))
        live += len(dict_list(data, "workers"))
    return NowTotals(ready, active, linked, unlinked, live)


def build_now_model(
    sources_read: SourcesRead,
    now: datetime,
    findings: Sequence[FindingLike] = (),
) -> NowModel:
    """Fold read sources + alarm findings into the Now model (pure; ``now`` injected)."""
    hb = sources_read.supervisor_heartbeat
    threshold = now_cadence.stale_threshold_seconds(
        hb.data if hb is not None else None,
        sources_read.collector_interval_seconds,
        sources_read.snapshot_gap_p90_seconds,
    )
    fresh, _ = _freshness(sources_read.repos, threshold)
    flow, dispatchable = _flow(sources_read)
    capacity = _capacity(sources_read, dispatchable, now, threshold)
    runners_stale_age = capacity.runners_age_seconds if capacity.runners_stale else None
    return NowModel(
        generated_at=now,
        stale_threshold_seconds=threshold,
        freshness=fresh,
        needs_me=needs_me_items(sources_read, fresh, findings, now, threshold, runners_stale_age),
        flow=flow,
        capacity=capacity,
        totals=_totals(sources_read),
    )
