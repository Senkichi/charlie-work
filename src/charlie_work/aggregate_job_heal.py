"""Self-heal a required aggregate job that a wedged runner timed out (issue #2666).

On 2026-10-09 two runners each wedged a trivial required aggregate job (``Tests``,
``Gradle Tests``) of a PR sitting in the merge queue. GitHub killed each at its
``timeout-minutes`` and finalized it ``cancelled``. Aviator will not admit a PR whose
required check is cancelled, and nothing in the fleet looks at one: a
``mergequeue``-labelled PR skips ``merge_ready`` for up to
``queue_skip.RECHECK_INTERVAL``, and an ``mq-bot-*`` draft is excluded from every lane
by ``queue_bot.is_queue_bot_pr``. The PR sat until a human ran ``gh run rerun --failed``,
which passed in under a minute.

This lane closes that gap: for every open PR that carries ``auto_merge.mergequeue_label``
or is authored by ``auto_merge.queue_bot_login``, find a required check
(``auto_merge.required_checks``) whose *latest* check run on the PR head concluded
``cancelled`` *because the job timed out*, and re-run that workflow run's failed jobs once.

The timeout signature is the job's own check-run annotation, not the run's conclusion.
The issue proposed "job cancelled while the run is not cancelled", but live payloads show
a timed-out job leaves the **run** ``cancelled`` too, so that rule could never match a
real timeout. What differs between the cases is the annotation GitHub attaches to the job
(captured from the public API 2026-11-02; ``tests/fixtures/gh_check_run_annotations_*``):

* job-level ``timeout-minutes`` kill: ``"The job has exceeded the maximum execution time
  of 15m0s"``;
* cancelled by a newer run's concurrency group: ``"Canceling since a higher priority
  waiting request for CI-refs/pull/2911/merge-ci"`` -- a superseded run, never rerun here.

A run superseded on the *same* head is also excluded structurally: the newer run's check
run has the same name and a higher id, and only the latest per name is judged.

Bounded: one rerun per workflow run id, persisted under ``state["aggregate_job_heal"]``
(``reruns``), so a job that wedges again on the rerun surfaces to a human instead of
looping. A rerun *call* that GitHub refuses is retried on later ticks, at most
``MAX_RERUN_CALLS`` times per run id. Nothing is written to ``state["prs"]``: an
``mq-bot-*`` draft is not fleet-owned and must not gain an entry there.

Cost: REST only (never GraphQL, so it is not a budget-governor lane). One open-PR list per
tick plus one check-run read per candidate, gated by
``auto_merge.aggregate_heal_interval_minutes`` (``0`` is the kill switch); the annotation
read happens only for a required check that is actually cancelled. The lane is dormant
(no API call, no state write) on a repo with no merge queue or no required checks.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from .checks import _run_id_from_link
from .config import OrchestratorConfig
from .github import GitHubError, GitHubRunResult, label_names
from .host import current as _host_current
from .local_work_park import publishes_pull_requests
from .queue_bot import is_queue_bot_pr
from .reconcile import _latest_check_runs_by_name, _normalize_reconcile_pr
from .state import load_state, load_state_locked, state_lock

logger = logging.getLogger(__name__)

STATE_KEY = "aggregate_job_heal"
SOURCE_MERGEQUEUE = "mergequeue"
SOURCE_QUEUE_BOT_DRAFT = "queue_bot_draft"

# Lower-cased substring of the annotation GitHub attaches to a job it killed at
# ``timeout-minutes`` ("The job has exceeded the maximum execution time of 15m0s").
TIMEOUT_MESSAGE = "exceeded the maximum execution time"

# Refused rerun calls (not reruns) tolerated per run id before the lane gives up on it.
MAX_RERUN_CALLS = 3
# Run ids remembered in state; run ids grow monotonically, so the newest are kept.
_KEEP_RUNS = 200
_PAGE_SIZE = 100
_MAX_PAGES = 10


@dataclass(frozen=True)
class QueuePr:
    number: int
    head_sha: str
    source: str  # SOURCE_MERGEQUEUE | SOURCE_QUEUE_BOT_DRAFT


@dataclass(frozen=True)
class CancelledCheck:
    """A required check whose latest run on the head is a completed, cancelled Actions job."""

    check_name: str
    check_run_id: int
    run_id: int  # the containing workflow run, parsed from the job's details URL


# ---------------------------------------------------------------- pure decisions


def queue_prs(prs: Iterable[Mapping[str, Any]], config: OrchestratorConfig) -> tuple[QueuePr, ...]:
    """Open PRs the merge queue owns: queue-bot drafts and ``mergequeue``-labelled PRs.

    ``prs`` carry the normalized ``gh pr list --json`` shape (``_normalize_reconcile_pr``).
    A queue-bot draft is identified by the single bot predicate; it is checked first so a
    bot PR that also carries the label is reported as the draft it is.
    """
    label = config.auto_merge.mergequeue_label
    found: list[QueuePr] = []
    for pr in prs:
        if str(pr.get("state") or "").upper() != "OPEN":
            continue
        number, head_sha = pr.get("number"), pr.get("headRefOid")
        if number is None or not head_sha:
            continue
        if is_queue_bot_pr(dict(pr), config):
            source = SOURCE_QUEUE_BOT_DRAFT
        elif label and label in label_names(dict(pr)):
            source = SOURCE_MERGEQUEUE
        else:
            continue
        found.append(QueuePr(int(number), str(head_sha), source))
    return tuple(found)


def cancelled_required_checks(
    check_runs: Iterable[Mapping[str, Any]], required: Iterable[str]
) -> tuple[CancelledCheck, ...]:
    """Required checks whose latest run is a completed ``cancelled`` Actions job.

    Only the highest-id check run per name is judged (``_latest_check_runs_by_name``,
    the helper every mergequeue detector shares), so a run superseded on the same head by
    a newer run of the same job is never reported. A check whose details URL is not an
    Actions job link has no workflow run to re-run and is skipped.
    """
    latest = _latest_check_runs_by_name([dict(run) for run in check_runs])
    found: list[CancelledCheck] = []
    for name in dict.fromkeys(required):
        run = latest.get(name)
        if run is None or run.get("status") != "completed" or run.get("conclusion") != "cancelled":
            continue
        check_run_id = run.get("id")
        run_id = _run_id_from_link(run.get("details_url"))
        if isinstance(check_run_id, bool) or not isinstance(check_run_id, int) or run_id is None:
            continue
        found.append(CancelledCheck(name, check_run_id, run_id))
    return tuple(found)


def is_timeout_cancel(annotations: Iterable[Any]) -> bool:
    """True when a job's annotations say GitHub killed it at ``timeout-minutes``."""
    return any(
        TIMEOUT_MESSAGE in str(annotation.get("message") or "").lower()
        for annotation in annotations
        if isinstance(annotation, Mapping)
    )


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def rerun_allowed(entry: Any) -> bool:
    """False once a rerun was triggered for the run, or its refused-call budget is spent."""
    if not isinstance(entry, Mapping):
        return True
    if entry.get("triggered"):
        return False
    return _as_int(entry.get("calls")) < MAX_RERUN_CALLS


# ------------------------------------------------------------------ state sections


def _section(state: Mapping[str, Any]) -> dict[str, Any]:
    section = state.get(STATE_KEY)
    return dict(section) if isinstance(section, dict) else {}


def _stamp(moment: datetime) -> str:
    return moment.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def is_due(state: Mapping[str, Any], now: datetime) -> bool:
    """True when the next tick is due. Absent or malformed means due (never wedge off)."""
    raw = _section(state).get("next_due_at")
    if not raw:
        return True
    try:
        due = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return True
    return now >= (due if due.tzinfo else due.replace(tzinfo=UTC))


def armed(state: Mapping[str, Any], next_due_at: datetime) -> dict[str, Any]:
    """New state with the next tick scheduled."""
    return {**state, STATE_KEY: {**_section(state), "next_due_at": _stamp(next_due_at)}}


def with_attempt(
    state: Mapping[str, Any], run_id: int, *, pr_number: int, triggered: bool, at: datetime
) -> dict[str, Any]:
    """New state recording one rerun call for ``run_id`` (pruned to the newest runs)."""
    section = _section(state)
    reruns = dict(section["reruns"]) if isinstance(section.get("reruns"), dict) else {}
    previous = reruns.get(str(run_id))
    previous = previous if isinstance(previous, Mapping) else {}
    reruns[str(run_id)] = {
        "calls": _as_int(previous.get("calls")) + 1,
        "triggered": bool(triggered or previous.get("triggered")),
        "pr_number": pr_number,
        "at": _stamp(at),
    }
    if len(reruns) > _KEEP_RUNS:
        newest = sorted(reruns, key=_as_int)[-_KEEP_RUNS:]
        reruns = {key: reruns[key] for key in newest}
    return {**state, STATE_KEY: {**section, "reruns": reruns}}


# ----------------------------------------------------------------------- I/O shell


def _open_prs(gh: Any) -> list[dict[str, Any]]:
    """Every open PR via the paged REST ``pulls`` endpoint (one call for < 100 open PRs)."""
    prs: list[dict[str, Any]] = []
    for page in range(1, _MAX_PAGES + 1):
        result = gh.run(
            [
                "api",
                f"repos/{{owner}}/{{repo}}/pulls?state=open&per_page={_PAGE_SIZE}&page={page}",
            ],
            json_output=True,
        )
        if not isinstance(result, list):
            raise GitHubError(
                f"aggregate heal: `gh api pulls?state=open` returned {type(result).__name__}, "
                "not a list"
            )
        prs.extend(_normalize_reconcile_pr(pr) for pr in result)
        if len(result) < _PAGE_SIZE:
            break
    return prs


def _rerun_failed_jobs(gh: Any, run_id: int) -> tuple[bool, str | None]:
    """``gh run rerun <id> --failed`` (REST ``rerun-failed-jobs``): (accepted, error)."""
    result = gh.run(["run", "rerun", str(run_id), "--failed"], allow_failure=True)
    if isinstance(result, GitHubRunResult):
        if result.ok:
            return True, None
        return False, result.error or f"gh run rerun {run_id} exited {result.returncode}"
    if isinstance(result, str):  # the dry-run echo
        return True, None
    return False, f"unexpected result from gh run rerun {run_id}: {result!r}"


def _record_attempt(
    app: Any,
    pr: QueuePr,
    run_id: int,
    checks: list[CancelledCheck],
    *,
    error: str | None,
    now: datetime,
) -> None:
    """Persist the attempt and emit its one event, under the state lock."""
    payload: dict[str, Any] = {
        "pr_number": pr.number,
        "head_sha": pr.head_sha,
        "run_id": run_id,
        "source": pr.source,
        "checks": [check.check_name for check in checks],
        "check_run_ids": [check.check_run_id for check in checks],
    }
    state_file = app.paths.state_file
    with state_lock(state_file):
        state = load_state(state_file)
        state = with_attempt(state, run_id, pr_number=pr.number, triggered=error is None, at=now)
        if error is None:
            state = app.write_gate.record_event(
                state,
                "aggregate_job_rerun_triggered",  # event-consumer: audit-only -- the rerun IS the remedy; a wedge that repeats stays visible through mergequeue_stalled
                payload,
            )
        else:
            state = app.write_gate.record_event(
                state,
                "aggregate_job_rerun_failed",  # event-consumer: audit-only -- bounded by MAX_RERUN_CALLS; the stuck PR still surfaces through mergequeue_stalled
                {**payload, "stage": "rerun", "error": error},
            )
        app.write_gate.save_state(state)


def heal_wedged_aggregates(app: Any, *, now: datetime | None = None) -> int:
    """Re-run the failed jobs of each timed-out required check once; return the count.

    Raises ``GitHubError`` on an unreadable open-PR list; ``heal_best_effort`` is the
    containment for the loop.
    """
    cfg = app.config.auto_merge
    required = tuple(cfg.required_checks)
    interval = cfg.aggregate_heal_interval_minutes
    if interval <= 0 or not required or app.dry_run:
        return 0
    if not (cfg.mergequeue_label or cfg.queue_bot_login) or not publishes_pull_requests(app.gh):
        return 0

    moment = now if now is not None else _host_current().clock.now()
    state_file = app.paths.state_file
    with state_lock(state_file):
        state = load_state(state_file)
        if not is_due(state, moment):
            return 0
        # Armed before the work: a failing tick degrades to one error per interval.
        app.write_gate.save_state(armed(state, moment + timedelta(minutes=interval)))

    triggered = 0
    handled_runs: set[int] = set()
    for pr in queue_prs(_open_prs(app.gh), app.config):
        check_runs = app.gh.commit_check_runs(pr.head_sha)
        if check_runs is None:
            continue
        timed_out = [
            check
            for check in cancelled_required_checks(check_runs, required)
            if is_timeout_cancel(app.gh.check_run_annotations(check.check_run_id))
        ]
        for run_id in sorted({check.run_id for check in timed_out} - handled_runs):
            handled_runs.add(run_id)
            entry = (
                load_state_locked(state_file).get(STATE_KEY, {}).get("reruns", {}).get(str(run_id))
            )
            if not rerun_allowed(entry):
                continue
            accepted, error = _rerun_failed_jobs(app.gh, run_id)
            _record_attempt(
                app,
                pr,
                run_id,
                [check for check in timed_out if check.run_id == run_id],
                error=error,
                now=moment,
            )
            triggered += int(accepted)
    return triggered


def heal_best_effort(app: Any, *, now: datetime | None = None) -> int:
    """``heal_wedged_aggregates`` that can never abort the pass.

    The lane is advisory and runs inside ``loop()``, where one uncaught exception ends
    the whole daemon (``supervise.py``'s handler sits outside its ``while True``). A
    failure is logged and recorded as a lock-free event; the cadence was already armed,
    so the next attempt is one interval away.
    """
    try:
        return heal_wedged_aggregates(app, now=now)
    except Exception as exc:  # noqa: BLE001 - advisory path: nothing may escape
        logger.warning("aggregate job heal failed: %s", exc)
        app.write_gate.log_event(
            kind="aggregate_job_rerun_failed",  # event-consumer: audit-only -- pass-level failure; the cadence re-arms and the stuck PR surfaces through mergequeue_stalled
            payload={"stage": "pass", "error": f"{type(exc).__name__}: {exc}"},
        )
        return 0
