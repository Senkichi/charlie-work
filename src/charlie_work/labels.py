"""Single owner of every GitHub-label edge in the issue lifecycle.

Every add/remove pair lives here as a named transition; workflow code names
the event and never touches individual labels. This is the single point of
enforcement for label-state consistency — scattering add/remove calls across
the workflow was how stalled label states happened in production.

Issue #2226: because every issue-label write funnels through
``apply_issue_labels`` (``transition`` is the named-edge front door and the
drift-repair callers pass explicit add/remove sets), this module is also the
single place a ``lifecycle_transition`` event is emitted into events.db.
``from_state`` is the last recorded lifecycle state for the issue — the
event log's own chain — so a re-applied edge against an unchanged state
emits nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from .config import LabelConfig
from .github import GitHubLike
from .instrumentation import log_event, query_events

logger = logging.getLogger(__name__)


class TransitionOutcome(Enum):
    """Result of a label transition operation."""

    APPLIED = "applied"  # All adds and removes succeeded
    PARTIAL_FAILURE = "partial_failure"  # At least one add or remove failed
    NOTHING_CHANGED = "nothing_changed"  # No labels to add or remove


@dataclass(frozen=True)
class TransitionResult:
    """Detailed result of a label transition operation."""

    outcome: TransitionOutcome
    add_failures: list[tuple[int, str]]  # (issue_number, label) pairs that failed to add
    remove_failures: list[tuple[int, str]]  # (issue_number, label) pairs that failed to remove

    @property
    def ok(self) -> bool:
        """True unless at least one individual label write failed."""
        return self.outcome is not TransitionOutcome.PARTIAL_FAILURE


def _edges(labels: LabelConfig) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    # Helper to compute removal set: all workflow labels except the ones being added
    # This ensures label transitions are exclusive (single-state by design)
    def _compute_remove(
        add_labels: tuple[str, ...], extra_remove: tuple[str, ...] = ()
    ) -> tuple[str, ...]:
        if not add_labels:
            return ()
        to_remove = (labels.workflow_labels - set(add_labels)) | set(extra_remove)
        return tuple(sorted(to_remove))

    return {
        # manifest written, worker not yet independently confirmed
        "queued": ((labels.queued,), _compute_remove((labels.queued,))),
        # worker launch confirmed (subprocess ok / independent evidence)
        "dispatched": ((labels.in_progress,), _compute_remove((labels.in_progress,))),
        # Re-review is a fresh cycle: clear needs_rework so repeated loop()
        # passes don't permanently stack reviewing on top of needs_rework.
        "review_started": (
            (labels.pr_open, labels.reviewing),
            _compute_remove((labels.pr_open, labels.reviewing)),
        ),
        "rework_requested": ((labels.needs_rework,), _compute_remove((labels.needs_rework,))),
        # rework worker launched for non-manual adapters
        "rework_dispatched": ((labels.in_progress,), _compute_remove((labels.in_progress,))),
        # reviewer approved; waiting for merge. pr_open (kept) without reviewing
        # is the "approved, not yet merged" state — distinct from under-review.
        "review_approved": ((labels.pr_open,), _compute_remove((labels.pr_open,))),
        # rework cap exhausted or reviewer blocked — a human decision is needed
        "escalated": ((labels.human_needed,), _compute_remove((labels.human_needed,))),
        "blocked": ((labels.human_needed,), _compute_remove((labels.human_needed,))),
        # Issue #1266: mechanical-escalation counterparts of "escalated" /
        # "redispatch_escalated" — same shape, different label, so a
        # mechanical escalation lands here instead of human_needed and
        # reserves that label for judgment calls. escalation.py's
        # _escalation_edge() is the single place that decides which of the
        # two edges a call site should use; nothing else picks between them.
        "operator_queued": (
            (labels.operator_queue,),
            _compute_remove((labels.operator_queue,)),
        ),
        # Issue #427: finalization must also drop the ready marker; a closed
        # issue with a stale automated-ready label pollutes roll-call/metrics.
        # Issue #496: merge-hold is a transient operator signal, not a workflow
        # state, so it is not in ``workflow_labels`` and must be stripped
        # explicitly here. Non-terminal transitions preserve it.
        "merged": (
            (labels.done,),
            _compute_remove((labels.done,), extra_remove=(labels.ready, labels.merge_hold)),
        ),
        # Issue #429: a closed ready issue with no merged PR binding it is stale
        # (e.g. human-closed not-planned/duplicate). Strip the ready marker and
        # any active labels so it drops out of future --state all fetches.
        # Issue #496: the merge-hold label is also transient and must not persist
        # on a closed issue.
        "closed_unmerged": (
            (),
            tuple(sorted(labels.active | {labels.ready, labels.merge_hold})),
        ),
        # redispatch cap exhausted — a human decision is needed
        "redispatch_escalated": ((labels.human_needed,), _compute_remove((labels.human_needed,))),
        # Issue #1266: mechanical counterpart of "redispatch_escalated" — see
        # "operator_queued" above for why this is a distinct named edge
        # rather than reusing that one.
        "redispatch_operator_queued": (
            (labels.operator_queue,),
            _compute_remove((labels.operator_queue,)),
        ),
        # Issue #1598: a bound PR whose issue carries a configured
        # ``dispatch.human_merge_labels`` label is human-merged, never
        # fleet-merged. When the PR reaches merge-ready state (approved,
        # checks green, no conflicts), the fleet transitions the issue to
        # ``agent:operator-queue`` via this edge and escalates with
        # ``reason_class="policy"`` — distinct from ``"mechanical"`` (the
        # de-escalation sweep must NOT auto-clear it) and from
        # ``"judgment"`` (``charlie unescalate`` is not required — the
        # merged-PR reconcile path closes it out once the human merges).
        # Same label as ``operator_queued`` but a distinct named edge so the
        # transition is attributable in events.db.
        "human_merge_required": (
            (labels.operator_queue,),
            _compute_remove((labels.operator_queue,)),
        ),
        # Local-file issue source: the worker committed to its branch and
        # there is no remote to push to, so the branch is the deliverable.
        # A success state, deliberately NOT one of the escalation edges --
        # see ``LabelConfig.review_ready``. ``ready`` is kept (unlike
        # "merged") because the issue is not done until the branch is merged
        # and the issue closed. With the local review/merge lane enabled
        # (the default for local_issues repos, issue #1844) this is only a
        # brief parking spot: the next loop pass adopts the branch into the
        # lane, which runs review + suite + merge automatically. When the
        # lane is disabled, ``review_ready`` being terminal is what holds
        # the issue out of dispatch for a human to merge by hand.
        "local_work_ready": (
            (labels.review_ready,),
            _compute_remove((labels.review_ready,)),
        ),
        # Operator re-arm (`charlie unescalate`) for an issue whose PR is
        # still open: drop human-needed (and any other stale workflow state)
        # and return to the passive pr-open state pending a fresh review.
        "unescalated_pr_open": ((labels.pr_open,), _compute_remove((labels.pr_open,))),
        # Operator re-arm with no live PR: strip every workflow label back to
        # bare automated-ready so dispatch treats the issue as fresh. Never
        # adds queued — queued is an ACTIVE label and would exclude the issue
        # from dispatch, the exact trap this edge exists to avoid.
        "unescalated_requeued": ((), tuple(sorted(labels.workflow_labels))),
        # Issue #1976: the config-retirement sweep marks a deprecated key's
        # removal issue Ready once the key has been absent from every config
        # layer for the quiet window. Removal issues are filed unlabeled, so
        # the add is the whole state change; the remove half clears any stray
        # workflow label so the issue arrives at dispatch as a clean candidate
        # (same shape as "unescalated_requeued" plus the ready marker itself,
        # which is not a workflow_labels member).
        "config_retirement_ready": (
            (labels.ready,),
            _compute_remove((labels.ready,)),
        ),
        # Issue #203: a merged PR only *mentions* the issue in free text, with
        # no hijack-safe branch/closing-keyword binding. That never authorizes
        # a close — flag it for a human decision instead, same label as any
        # other human-needed escalation.
        "merged_pr_mention_flagged": (
            (labels.human_needed,),
            _compute_remove((labels.human_needed,)),
        ),
    }


def _label_state_map(labels: LabelConfig) -> dict[str, str]:
    """Map each label ``LabelConfig`` manages to its lifecycle state name.

    Keys are configured label strings — never literals; values are the
    CONTEXT.md lifecycle terms, which coincide with the ``LabelConfig``
    field names (``ready``/``queued``/``in_progress``/``pr_open``/
    ``reviewing``/``needs_rework``/``done`` plus the terminal escalation and
    local-lane dispositions).
    """
    return {
        labels.ready: "ready",
        labels.queued: "queued",
        labels.in_progress: "in_progress",
        labels.pr_open: "pr_open",
        labels.reviewing: "reviewing",
        labels.needs_rework: "needs_rework",
        labels.done: "done",
        labels.review_ready: "review_ready",
        labels.operator_queue: "operator_queue",
        labels.human_needed: "human_needed",
    }


# When an edge adds more than one lifecycle label at once ("review_started"
# adds pr_open + reviewing) the recorded to_state is the most advanced one.
_STATE_PRECEDENCE = (
    "ready",
    "queued",
    "in_progress",
    "pr_open",
    "reviewing",
    "needs_rework",
    "done",
    "review_ready",
    "operator_queue",
    "human_needed",
)

# Edges with an empty add-set cannot derive their to_state from the labels
# they write; the override names the CONTEXT.md disposition they land on.
_EDGE_STATE_OVERRIDES = {
    "closed_unmerged": "closed",
    "unescalated_requeued": "ready",
}


def _state_for_labels(labels: LabelConfig, add: Iterable[str]) -> str | None:
    """Derive the lifecycle state a label write lands on from its add-set."""
    state_by_label = _label_state_map(labels)
    states = [state_by_label[label] for label in add if label in state_by_label]
    if not states:
        return None
    return max(states, key=_STATE_PRECEDENCE.index)


def _current_lifecycle_state(state_path: Path, issue_number: int) -> str | None:
    """Return the issue's last recorded lifecycle state from events.db.

    The event log is the authoritative record of *observed* transitions:
    a ``ready_observed`` event opens a Ready episode and each
    ``lifecycle_transition`` moves the chain forward. ``None`` means the
    issue has no recorded lifecycle history yet.
    """
    state: str | None = None
    for event in query_events(state_path, issue_number=issue_number):
        if event["kind"] == "lifecycle_transition":
            to_state = event["payload"].get("to_state")
            if isinstance(to_state, str):
                state = to_state
        elif event["kind"] == "ready_observed":
            state = "ready"
    return state


# The states whose label disposition strips ``ready`` (``merged`` removes it
# via extra_remove, ``closed_unmerged`` lists it explicitly). Landing on one
# ends the Ready episode: a later sighting of the label is a new episode.
# Every other recorded state keeps the ready label on the issue, so the
# episode stays open while it persists — which is also why an issue that is
# dispatched, escalated, or under review does not re-emit ``ready_observed``
# on every intake pass even though it still carries ``ready``.
_READY_ABSENT_STATES = frozenset({"done", "closed"})


def ready_episode_open(state_path: Path, issue_number: int) -> bool:
    """True when the issue's recorded lifecycle already has a Ready episode open.

    The event chain is authoritative: ``ready_observed`` opens an episode and
    a ``lifecycle_transition`` to a ``_READY_ABSENT_STATES`` member closes it.
    ``None`` (no recorded history) means the episode is not open — the first
    sighting always emits.
    """
    return _current_lifecycle_state(state_path, issue_number) not in (
        None,
        *_READY_ABSENT_STATES,
    )


def _emit_lifecycle_transition(
    state_path: Path,
    *,
    repo: str | None,
    issue_number: int,
    pr_number: int | None,
    to_state: str,
    cause: str | None,
) -> None:
    """Emit ``lifecycle_transition`` unless it re-observes the current state.

    ``from_state`` is read back from the issue's own event chain, so a pass
    that re-applies an edge whose state is already recorded emits nothing —
    that is the no-duplicate guarantee of issue #2226.
    """
    from_state = _current_lifecycle_state(state_path, issue_number)
    if from_state == to_state:
        return
    log_event(
        state_path,
        "lifecycle_transition",
        {
            "issue_number": issue_number,
            "pr_number": pr_number,
            "from_state": from_state,
            "to_state": to_state,
            "cause": cause,
        },
        repo=repo,
    )


def apply_issue_labels(
    gh: GitHubLike,
    labels: LabelConfig,
    issue_number: int,
    *,
    add: Iterable[str] = (),
    remove: Iterable[str] = (),
    to_state: str | None = None,
    state_path: Path | None,
    repo: str | None = None,
    pr_number: int | None = None,
    cause: str | None = None,
) -> TransitionResult:
    """The single issue-label write seam; emits ``lifecycle_transition``.

    Every write to an issue's labels — a named edge via ``transition`` or an
    explicit add/remove repair set (reconcile's drift fixes) — funnels here.
    After a fully-applied write, when ``to_state`` resolves (explicitly, or
    from the add-set via ``LabelConfig``) and ``state_path`` is supplied, a
    ``lifecycle_transition`` event is recorded unless the issue's recorded
    state already equals it.

    ``state_path`` is a required keyword so a production caller can never
    omit it by accident: pass ``None`` explicitly to opt out of the event
    (e.g. a dry-run salvage probe that writes no PR), or a real path so the
    lifecycle record stays complete. ``WriteGate.apply_issue_labels`` binds
    it automatically.
    """
    add = tuple(add)
    remove = tuple(remove)
    add_failures: list[tuple[int, str]] = []
    remove_failures: list[tuple[int, str]] = []

    for label in add:
        if not gh.add_issue_label(issue_number, label):
            add_failures.append((issue_number, label))

    for label in remove:
        if not gh.remove_issue_label(issue_number, label):
            remove_failures.append((issue_number, label))

    if add_failures or remove_failures:
        logger.warning(
            "label_transition issue=%d cause=%s outcome=partial_failure "
            "add_failures=%s remove_failures=%s",
            issue_number,
            cause,
            add_failures,
            remove_failures,
        )
        return TransitionResult(TransitionOutcome.PARTIAL_FAILURE, add_failures, remove_failures)

    # Emit on both non-failure outcomes: a repair whose computed set is empty
    # still *observes* the state it converged on (e.g. a closed issue whose
    # active labels were already gone), and the event chain is the record of
    # observed state, not of write side-effects. ``gh.dry_run`` suppresses the
    # emit at the seam itself — under dry-run the label writes are transport
    # no-ops, so no transition actually occurred and no event may claim one;
    # this holds even for callers that bypass ``WriteGate``.
    resolved_state = to_state or _state_for_labels(labels, add)
    if resolved_state is not None and state_path is not None and not getattr(gh, "dry_run", False):
        _emit_lifecycle_transition(
            state_path,
            repo=repo,
            issue_number=issue_number,
            pr_number=pr_number,
            to_state=resolved_state,
            cause=cause,
        )
    if not add and not remove:
        logger.info(
            "label_transition issue=%d cause=%s outcome=nothing_changed",
            issue_number,
            cause,
        )
        return TransitionResult(TransitionOutcome.NOTHING_CHANGED, [], [])
    logger.info(
        "label_transition issue=%d cause=%s outcome=applied add=%s remove=%s",
        issue_number,
        cause,
        add,
        remove,
    )
    return TransitionResult(TransitionOutcome.APPLIED, [], [])


def transition(
    gh: GitHubLike,
    labels: LabelConfig,
    issue_number: int,
    event: str,
    *,
    state_path: Path | None,
    repo: str | None = None,
    pr_number: int | None = None,
    cause: str | None = None,
) -> TransitionResult:
    """Apply the named lifecycle edge and record the transition (issue #2226).

    ``state_path`` locates the events.db the ``lifecycle_transition`` event
    is written to and is a required keyword so no caller can omit it by
    accident — pass ``None`` explicitly to opt out of the event while still
    applying the label writes. ``WriteGate.transition`` binds it
    automatically.
    """
    add, remove = _edges(labels)[event]
    return apply_issue_labels(
        gh,
        labels,
        issue_number,
        add=add,
        remove=remove,
        to_state=_EDGE_STATE_OVERRIDES.get(event),
        state_path=state_path,
        repo=repo,
        pr_number=pr_number,
        cause=cause or event,
    )
