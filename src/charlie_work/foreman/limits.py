"""Structural limits: escalation breaker, unattended limit, verdict (issue #3123).

Design §10/§16. Evaluated at the start of every non-dry-run pass for each
foreman whose state is ``active`` or ``degraded`` — ``pass_run._pass_one``
calls :func:`evaluate_limits` before the wake predicate so a transition
applies to the Turn about to run (a breaker-tripped foreman's new issues
stay unarmed via the ``degraded`` state; a failed verdict's forced
``propose`` is read by ``apply_checks._arm_or_raise``).

The three limits share one bookkeeping shape:

* ``state.degraded_reasons`` records which limits currently hold — the
  ``degraded`` state is the union of independent causes, so clearing one
  (breaker recovered) must not clear the other (a stale decision item).
* One event on each transition, one on each recovery — the audit trail the
  heartbeat alarms and the dashboard read.
* Decision/report items are how each limit reaches the operator; an item
  resolved (answered *or* dismissed — both are operator actions on the
  question) is the recovery path the design calls "the operator's answer".

``state.json`` stays a cache: every field the limits write round-trips
through the tolerant decoder, and a state.json that predates this feature
reads as "no reasons recorded" — the next pass recomputes from scratch.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import layout
from ..api_budget import ledger_path, load_ledger
from ..github import label_names
from ..instrumentation import log_event, query_events
from ..iso_timestamp import parse_iso_timestamp
from . import collect
from . import items as fitems
from . import paths as fpaths
from . import state as fstate
from . import store
from .charter import ARMING_PROPOSE

if TYPE_CHECKING:
    from ..foreman_config import ForemanBreakerConfig
    from .collect import Foreman

#: Degraded causes the limits manage (``state.degraded_reasons`` members).
DEGRADED_BREAKER = "breaker"
DEGRADED_UNATTENDED = "unattended"
DEGRADED_REASONS = frozenset({DEGRADED_BREAKER, DEGRADED_UNATTENDED})

#: Event kinds — each is registered under ``event_levels/`` (the registry
#: test fails the build on an unregistered kind).
EVENT_BREAKER_TRIPPED = "foreman_breaker_tripped"
EVENT_BREAKER_RECOVERED = "foreman_breaker_recovered"
EVENT_UNATTENDED_LIMIT = "foreman_unattended_limit"
EVENT_UNATTENDED_RECOVERED = "foreman_unattended_recovered"
EVENT_VERDICT = "foreman_verdict"

#: Lifecycle ``to_state`` values (``labels._label_state_map`` +
#: ``_EDGE_STATE_OVERRIDES``) that close an issue out. "Finished" for the
#: breaker and the verdict means the issue's *latest* recorded lifecycle
#: state is one of these; the escalation rate is the share that landed in
#: Human needed or Operator queue.
FINISHED_STATES = frozenset({"done", "review_ready", "operator_queue", "human_needed", "closed"})
ESCALATED_STATES = frozenset({"operator_queue", "human_needed"})
DONE_STATE = "done"

#: The verdict's fixed thresholds (design §16 — "Two thresholds, set now").
VERDICT_MIN_ISSUES = 20
VERDICT_MAX_AGE_DAYS = 14
VERDICT_DONE_SHARE_RATIO = 0.8
VERDICT_TOKEN_SHARE_MAX = 0.25

#: Session event-log name fragments for worker/reviewer token accounting —
#: ``issue-<n>.events.jsonl`` (worker), ``-review`` (reviewer), ``-rework``
#: (rework) under the sessions dir (``claude_code._events_path``).
_ISSUE_EVENTS_RE = re.compile(r"issue-(\d+)(|-review|-rework)\.events\.jsonl$")


@dataclass(frozen=True)
class FinishedIssue:
    """One issue whose latest recorded lifecycle state is terminal."""

    number: int
    to_state: str
    event_id: int


def _finished_issues(foreman: Foreman) -> list[FinishedIssue]:
    """Every finished issue in the repo's events.db, newest-finished first.

    An issue "finished" when its *latest* ``lifecycle_transition`` landed on
    a terminal state — a later re-arm (``unescalated_requeued`` → ``ready``)
    takes it back out of the finished set, so only the last transition counts.
    Event id order is the finish order.
    """
    latest: dict[int, dict[str, Any]] = {}
    for event in query_events(foreman.state_file, kind="lifecycle_transition"):
        number = event.get("issue_number")
        if isinstance(number, int):
            latest[number] = event
    finished = [
        FinishedIssue(
            number=number,
            to_state=event["payload"].get("to_state"),
            event_id=int(event.get("id") or 0),
        )
        for number, event in latest.items()
        if isinstance(event.get("payload"), dict)
        and event["payload"].get("to_state") in FINISHED_STATES
    ]
    finished.sort(key=lambda f: f.event_id, reverse=True)
    return finished


def _snapshot_owned(foreman: Foreman) -> set[int]:
    """Open issues in the repo's status snapshot carrying a ``foreman:`` label.

    The same local source Collect reads — an owned issue here needs no
    completed Turn to count, so it closes the register's pickup gap below.
    Any ``foreman:`` name matches (``foreman:other`` included): the baseline
    is "no foreman owns", not "this foreman does not own".
    """
    prefix = foreman.labels.foreman_prefix
    if not prefix:
        return set()
    _written, data = collect._read_snapshot(foreman.state_root)
    if data is None:
        return set()
    issues = data.get("issues")
    if not isinstance(issues, list):
        return set()
    owned: set[int] = set()
    for entry in issues:
        if not isinstance(entry, dict):
            continue
        number = entry.get("number")
        if isinstance(number, int) and any(name.startswith(prefix) for name in label_names(entry)):
            owned.add(number)
    return owned


def _all_foreman_owned(foreman: Foreman, fleet_dir: Path) -> set[int]:
    """Issue numbers the baseline's "no foreman owns" rule excludes (§10/§15).

    Two local sources stand in for the GitHub label read the pass does not
    make:

    * every foreman's ``state.issues`` register — authored *and* tracked
      issues recorded at the last completed Turn, plus entries the register
      keeps after their label left the snapshot (a Turn's register write
      unions prior entries forward, so it only grows);
    * this repo's status snapshot — open issues carrying any ``foreman:``
      label, covering a tracked issue between the operator adding the label
      and its foreman's next completed Turn.

    The approximation errs in *both* directions: a retained or foreign-repo
    register entry can exclude an issue no foreman currently owns, and the
    residual gap — an owned issue that finished and left the snapshot before
    any Turn recorded it — still lands in the baseline. Both residuals age
    out as the ``baseline_sample`` window slides past.
    """
    owned = _snapshot_owned(foreman)
    for _name, root in store.list_foremen(fleet_dir):
        owned.update(fstate.read_state_snapshot(root).issues)
    return owned


def _escalation_rate(window: list[FinishedIssue]) -> float:
    """Share of the window that ended in Human needed or Operator queue."""
    if not window:
        return 0.0
    return sum(1 for f in window if f.to_state in ESCALATED_STATES) / len(window)


def _item_by_id(items: tuple[fitems.Item, ...]) -> dict[str, fitems.Item]:
    return {item.id: item for item in items}


def _emit(foreman: Foreman, kind: str, payload: dict[str, Any]) -> None:
    # write-gate-exempt(issue=3123): no write_gate receiver on the pass — the limit events are the §10 audit trail on the foreman's own events.db.
    log_event(  # event-consumer: audit-only — degraded_reasons in state.json is the record the pass reads
        fpaths.state_path(foreman.root),
        kind,
        {"foreman": foreman.name, **payload},
        repo=foreman.repo_key,
    )


def _breaker_sample_key(sample: list[FinishedIssue]) -> tuple[int, ...]:
    return tuple(sorted(f.number for f in sample))


def _raise_breaker_item(
    foreman: Foreman,
    items: dict[str, fitems.Item],
    prior_item_id: str | None,
    rate: float,
    baseline_rate: float,
) -> str | None:
    """Raise the breaker's course-check decision item; reuse an open one.

    Reuses the recorded open item if one survives (a degraded foreman whose
    operator manually resumed it re-trips without spamming a second copy).
    """
    if prior_item_id is not None:
        prior = items.get(prior_item_id)
        if prior is not None and prior.status in (
            fitems.STATUS_OPEN,
            fitems.STATUS_SNOOZED,
        ):
            return prior_item_id
    created = fitems.create_item(
        foreman.root,
        kind=fitems.ITEM_DECISION,
        question=(
            f"Foreman {foreman.name} escalation breaker tripped: "
            f"{rate:.0%} of its last finished issues ended in Human needed or "
            f"Operator queue (baseline {baseline_rate:.0%}). Confirm or "
            "correct course?"
        ),
        options=(
            "Confirm — resume arming (the foreman is back under review)",
            "Correct — keep it degraded while I retune the charter",
        ),
        recommended=0,
        command=f"charlie gang interventions {foreman.name}",
    )
    if created.ok:
        return created.value.id
    return None


def _stale_decision(
    items: tuple[fitems.Item, ...], *, max_days: int, now: datetime
) -> fitems.Item | None:
    """The oldest open ``decision`` item when it is older than ``max_days``."""
    oldest: fitems.Item | None = None
    oldest_at: datetime | None = None
    for item in fitems.open_items(items):
        if item.kind != fitems.ITEM_DECISION:
            continue
        created = parse_iso_timestamp(item.created_at)
        if created is None:
            continue
        if oldest_at is None or created < oldest_at:
            oldest, oldest_at = item, created
    if oldest is None or oldest_at is None:
        return None
    if (now - oldest_at) <= timedelta(days=max_days):
        return None
    return oldest


def _turn_events(foreman: Foreman) -> list[dict[str, Any]]:
    """Every ``foreman_turn_completed`` event, oldest first."""
    return query_events(fpaths.state_path(foreman.root), kind="foreman_turn_completed")


def _worker_reviewer_tokens(foreman: Foreman, issues: set[int]) -> int | None:
    """Worker+reviewer tokens recorded against ``issues`` — or ``None``.

    Two sources: the api-budget ledger's ``SessionEntry`` rows (api adapter,
    issue #480) and the claude-code session event logs
    ``issue-<n>[-review|-rework].events.jsonl`` under the sessions dir, whose
    last ``tokens`` value is the session's cumulative total
    (``worker.parse_cumulative_usage``). Devin-shell sessions record no
    per-issue token source — ``None`` (no data at all) is reported as
    unknown rather than counted as zero spend, so a fleet without recorded
    usage is not flunked on an absent denominator.
    """
    found = False
    total = 0
    ledger_file = ledger_path(foreman.state_root)
    if ledger_file.is_file():
        for session in load_ledger(ledger_file).sessions:
            if session.issue in issues:
                found = True
                total += session.input_tokens + session.output_tokens
    sessions_dir = layout.resolve_state_child(
        foreman.config.devin.sessions_dir,
        repo_root=foreman.repo_root,
        default=layout.sessions_dir_default(foreman.state_root),
    )
    if sessions_dir.is_dir():
        for entry in sessions_dir.iterdir():
            match = _ISSUE_EVENTS_RE.match(entry.name)
            if match is None or int(match.group(1)) not in issues or not entry.is_file():
                continue
            tokens = _cumulative_tokens(entry)
            if tokens is None:
                continue
            found = True
            total += tokens
    return total if found else None


def _cumulative_tokens(events_path: Path) -> int | None:
    """Last ``tokens`` value in a session events.jsonl (the cumulative total)."""
    try:
        tokens: int | None = None
        with events_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and isinstance(event.get("tokens"), int):
                    tokens = event["tokens"]
        return tokens
    except OSError:
        return None


def _verdict_numbers(
    foreman: Foreman,
    authored_finished: list[FinishedIssue],
    baseline: list[FinishedIssue],
) -> dict[str, Any]:
    """Both §16 numbers: Done share vs baseline, Turn tokens vs worker spend."""
    done_share = (
        sum(1 for f in authored_finished if f.to_state == DONE_STATE) / len(authored_finished)
        if authored_finished
        else 0.0
    )
    baseline_done_share = (
        sum(1 for f in baseline if f.to_state == DONE_STATE) / len(baseline) if baseline else 0.0
    )
    turn_tokens = sum(
        int(event["payload"].get("input_tokens") or 0)
        + int(event["payload"].get("output_tokens") or 0)
        for event in _turn_events(foreman)
        if isinstance(event.get("payload"), dict)
    )
    worker_tokens = _worker_reviewer_tokens(foreman, {f.number for f in authored_finished})
    done_ok = done_share >= VERDICT_DONE_SHARE_RATIO * baseline_done_share
    token_ok = worker_tokens is None or turn_tokens <= VERDICT_TOKEN_SHARE_MAX * worker_tokens
    return {
        "authored_finished": len(authored_finished),
        "done_share": done_share,
        "baseline_done_share": baseline_done_share,
        "baseline_count": len(baseline),
        "done_ok": done_ok,
        "turn_tokens": turn_tokens,
        "worker_reviewer_tokens": worker_tokens,
        "token_ok": token_ok,
        "passed": done_ok and token_ok,
    }


def _verdict_report_text(foreman: Foreman, numbers: dict[str, Any]) -> str:
    worker = numbers["worker_reviewer_tokens"]
    worker_text = str(worker) if worker is not None else "unknown (no recorded sessions)"
    return (
        f"Foreman {foreman.name} verdict ({'passed' if numbers['passed'] else 'FAILED'}): "
        f"{numbers['authored_finished']} authored issues finished; "
        f"Done share {numbers['done_share']:.0%} vs baseline "
        f"{numbers['baseline_done_share']:.0%} over "
        f"{numbers['baseline_count']} non-foreman issues "
        f"(needs >= {VERDICT_DONE_SHARE_RATIO}x); "
        f"Turn tokens {numbers['turn_tokens']} vs worker+reviewer "
        f"{worker_text} (needs <= {VERDICT_TOKEN_SHARE_MAX:.0%})."
    )


def _raise_retune_item(foreman: Foreman) -> str | None:
    """The failed verdict's "retire or retune" decision item; its id or None."""
    created = fitems.create_item(
        foreman.root,
        kind=fitems.ITEM_DECISION,
        question=(
            f"Foreman {foreman.name} missed its verdict (see the report item): "
            "it is now forced to propose arming. Retire it or retune it?"
        ),
        options=(
            "Retune — clear the forced propose and adjust the charter",
            "Retire — keep it forced and wind the foreman down",
        ),
        recommended=0,
        command=f"charlie gang interventions {foreman.name}",
    )
    return created.value.id if created.ok else None


def _resolve_item(items: dict[str, fitems.Item], item_id: str | None) -> bool:
    """Whether the recorded item was resolved by the operator.

    Answered or dismissed both count — each is an operator action on the
    question; only an item still open (or gone, treated as not resolved) holds.
    """
    if item_id is None:
        return False
    item = items.get(item_id)
    return item is not None and item.status in (
        fitems.STATUS_ANSWERED,
        fitems.STATUS_DISMISSED,
    )


def evaluate_limits(
    foreman: Foreman,
    *,
    fleet_dir: Path,
    breaker: ForemanBreakerConfig,
    now: datetime,
) -> fstate.ForemanState:
    """Re-evaluate breaker, unattended limit and verdict for one foreman.

    Called at the start of each pass for ``active``/``degraded`` foremen.
    Returns the persisted state — callers should treat the return value as
    the foreman's current state (it may differ from ``foreman.state`` when a
    transition landed this pass).
    """
    root = foreman.root
    state = fstate.read_state_snapshot(root)
    items = fitems.load_items(root)
    by_id = _item_by_id(items)
    open_status = (fitems.STATUS_OPEN, fitems.STATUS_SNOOZED)
    finished = _finished_issues(foreman)
    owned = _all_foreman_owned(foreman, fleet_dir)

    foreman_finished = [f for f in finished if f.number in state.issues]
    sample = foreman_finished[: breaker.sample]
    baseline = [f for f in finished if f.number not in owned][: breaker.baseline_sample]

    reasons = set(state.degraded_reasons) & DEGRADED_REASONS
    changes: dict[str, Any] = {}

    # --- escalation breaker ------------------------------------------------
    # Only evaluates once the foreman's own sample is full and a non-foreman
    # baseline exists — the design's "no probation" is the open-issue bound,
    # not a grace window here.
    sample_full = len(sample) >= breaker.sample
    over = (
        sample_full
        and bool(baseline)
        and _escalation_rate(sample) > breaker.ratio * _escalation_rate(baseline)
    )
    sample_key = _breaker_sample_key(sample)
    if DEGRADED_BREAKER in reasons:
        resolved = _resolve_item(by_id, state.breaker_item_id)
        if resolved or not over:
            reasons.discard(DEGRADED_BREAKER)
            # An answer while the rate is still over acknowledges *this*
            # sample — the breaker stays quiet until a new finished issue
            # changes it; a rate recovery clears the acknowledgement.
            changes["breaker_ack_sample"] = sample_key if (resolved and over) else ()
            _emit(
                foreman,
                EVENT_BREAKER_RECOVERED,
                {
                    "via": "item_resolved" if resolved else "rate",
                    "rate": round(_escalation_rate(sample), 4),
                    "baseline_rate": round(_escalation_rate(baseline), 4),
                    "sample": len(sample),
                    "baseline": len(baseline),
                },
            )
        else:
            # Still tripped and unanswered: the course-check item is the only
            # operator-facing path back, so a missing one is re-raised — a
            # failed create_item at trip time or a wiped items.json must not
            # leave a degraded foreman with nothing to answer (same shape as
            # the forced-arming re-raise below).
            recorded = by_id.get(state.breaker_item_id) if state.breaker_item_id else None
            if recorded is None or recorded.status not in open_status:
                item_id = _raise_breaker_item(
                    foreman,
                    by_id,
                    state.breaker_item_id,
                    _escalation_rate(sample),
                    _escalation_rate(baseline),
                )
                if item_id is not None:
                    changes["breaker_item_id"] = item_id
    elif over and sample_key != tuple(sorted(state.breaker_ack_sample)):
        reasons.add(DEGRADED_BREAKER)
        item_id = _raise_breaker_item(
            foreman,
            by_id,
            state.breaker_item_id,
            _escalation_rate(sample),
            _escalation_rate(baseline),
        )
        changes["breaker_item_id"] = item_id
        _emit(
            foreman,
            EVENT_BREAKER_TRIPPED,
            {
                "rate": round(_escalation_rate(sample), 4),
                "baseline_rate": round(_escalation_rate(baseline), 4),
                "sample": len(sample),
                "baseline": len(baseline),
                "item_id": item_id,
            },
        )

    # --- unattended limit ---------------------------------------------------
    stale = _stale_decision(items, max_days=foreman.charter.bounds.max_unanswered_days, now=now)
    if DEGRADED_UNATTENDED in reasons:
        if stale is None:
            reasons.discard(DEGRADED_UNATTENDED)
            _emit(foreman, EVENT_UNATTENDED_RECOVERED, {})
    elif stale is not None:
        reasons.add(DEGRADED_UNATTENDED)
        age_days = 0.0
        created = parse_iso_timestamp(stale.created_at)
        if created is not None:
            age_days = (now - created).total_seconds() / 86400.0
        _emit(
            foreman,
            EVENT_UNATTENDED_LIMIT,
            {
                "item_id": stale.id,
                "created_at": stale.created_at,
                "age_days": round(age_days, 2),
                "max_unanswered_days": foreman.charter.bounds.max_unanswered_days,
            },
        )

    # --- verdict ------------------------------------------------------------
    if not state.verdict_raised:
        authored = {n for n, entry in state.issues.items() if entry.body_hash}
        authored_finished = [f for f in foreman_finished if f.number in authored]
        due = len(authored_finished) >= VERDICT_MIN_ISSUES
        if not due:
            turns = _turn_events(foreman)
            first_ts = parse_iso_timestamp(turns[0].get("ts")) if turns else None
            due = first_ts is not None and (now - first_ts) >= timedelta(days=VERDICT_MAX_AGE_DAYS)
        if due:
            numbers = _verdict_numbers(foreman, authored_finished, baseline)
            report = fitems.create_item(
                foreman.root,
                kind=fitems.ITEM_REPORT,
                text=_verdict_report_text(foreman, numbers),
            )
            # A report item that could not be raised leaves the verdict
            # unraised — retrying next pass beats marking a verdict the
            # operator never saw.
            if report.ok:
                changes["verdict_raised"] = True
                if not numbers["passed"]:
                    changes["forced_arming"] = ARMING_PROPOSE
                    changes["verdict_item_id"] = _raise_retune_item(foreman)
                _emit(foreman, EVENT_VERDICT, numbers)

    # A resolved retire/retune item clears the forced arming — the only path
    # back to the charter's own ``arming`` (design §16). While forced, a
    # missing decision item (a failed create, or a wiped items.json) is
    # re-raised so the answer path back never disappears.
    if state.forced_arming is not None:
        if _resolve_item(by_id, state.verdict_item_id):
            changes["forced_arming"] = None
            changes["verdict_item_id"] = None
        else:
            recorded = by_id.get(state.verdict_item_id) if state.verdict_item_id else None
            if recorded is None or recorded.status not in open_status:
                retune = _raise_retune_item(foreman)
                if retune is not None:
                    changes["verdict_item_id"] = retune

    # --- persist -------------------------------------------------------------
    reasons_tuple = tuple(sorted(reasons))
    changes["degraded_reasons"] = reasons_tuple
    if state.state in ("active", "degraded"):
        changes["state"] = "degraded" if reasons_tuple else "active"
    if reasons_tuple != state.degraded_reasons or any(
        changes.get(key) != getattr(state, key) for key in changes if key != "degraded_reasons"
    ):
        # write-gate-exempt(issue=3123): no write_gate receiver on the pass — this save IS the limit's bookkeeping landing.
        saved = fstate.save_state(root, fstate.with_updates(state, **changes))
        if saved.ok:
            state = saved.value
    return state


__all__ = [
    "DEGRADED_BREAKER",
    "DEGRADED_REASONS",
    "DEGRADED_UNATTENDED",
    "ESCALATED_STATES",
    "EVENT_BREAKER_RECOVERED",
    "EVENT_BREAKER_TRIPPED",
    "EVENT_UNATTENDED_LIMIT",
    "EVENT_UNATTENDED_RECOVERED",
    "EVENT_VERDICT",
    "FINISHED_STATES",
    "VERDICT_DONE_SHARE_RATIO",
    "VERDICT_MAX_AGE_DAYS",
    "VERDICT_MIN_ISSUES",
    "VERDICT_TOKEN_SHARE_MAX",
    "evaluate_limits",
]
