"""Worker fate: one module for liveness and post-exit classification.

Architecture-deepening candidate 1 (``docs/superpowers/plans/
2026-09-29-architecture-deepening.md``, design note in the Wave A scratchpad
``wf-design.md``). Today, "what happened to this worker" is decided by 13
scattered sites (``workflow.py``, ``orphaned_worker_sweep.py``,
``live_handoff_finalize.py``, ``rework_outcome.py``, ``dead_worker_reap.py``,
``misc_worker_dispatch.py``) that disagree on nine points. This module is the
single point of enforcement for all nine; ``resolve_fate`` is a pure function
over gathered evidence, so every rule is testable as a decision table with no
fakes, no ``tmp_path`` and no git.

Shape (design doc section 1, "Shape A: a pure resolver over gathered
evidence"):

    evidence = gather_evidence(subject, readers, now=now)   # thin I/O
    fate     = resolve_fate(evidence, now=now)               # pure, 9 rules

``process_utils.is_pid_alive`` stays the liveness primitive, unchanged,
including its asymmetric fail-open/fail-closed behaviour; ``is_alive`` here
only adds the ``pid is None`` case (no process, e.g. a manual-adapter
subject). ``classify_worker_health`` and ``is_worker_confirmed_dead`` stay in
``worker.py`` — they own the inconclusive-probe deferral counter (#755) and
this module only consumes their output through ``FateEvidence.health``.

Deliberately NOT in this commit (see the design doc sections 5-7 and the
plan's Group A table, steps A2-A4): the ``AdapterFateProfile`` registry,
``classify_failure``, ``profile_for``, ``persisted_failure``/``persist_fate``/
``stale_evidence_events``, and ``default_readers`` production wiring. Those
depend on merging two still-drifted adapter classifiers
(``claude_code._classify_session_failure`` vs
``devin_failure_classification._classify_session_failure``, issue #1997) and
on new ``state.py``/``process_utils.py`` fields, and are wiring-adjacent in a
way this slice's "no consumers yet" boundary is meant to keep out. Until
then, ``FateEvidence.failure`` is supplied by the caller (or a test) as
already-classified data — ``resolve_fate`` only ever reads
``FailureEvidence.kind``/``throttled_until``, never produces them.

No consumer in ``src/`` calls into this module yet. Wiring happens in later
commits named after the plan's Group A/B steps.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from .process_utils import is_pid_alive as _process_is_pid_alive
from .throttle_signatures import is_provider_throttle_failure
from .worker import WorkerHealth

_EMPTY_RAW: Mapping[str, Any] = MappingProxyType({})


class EvidenceSource(StrEnum):
    """Where a piece of outcome evidence was read from."""

    TERMINAL = "terminal"
    WORKTREE = "worktree"


class StaleReason(StrEnum):
    """Why a candidate outcome was ignored under rule 1 (freshness)."""

    OLDER_THAN_DISPATCH = "older_than_dispatch"
    HEAD_MISMATCH = "head_mismatch"


# --------------------------------------------------------------------------
# Evidence types (§2). All frozen per the repo's config/value-object
# invariant (CLAUDE.md).
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OutcomeEvidence:
    """One parsed ``.worker-outcome.json`` claim, from one source."""

    source: EvidenceSource
    written_at: datetime | None  # worktree: file mtime; terminal: see rule 1
    outcome: str | None  # "completed" | "blocked" | ... (raw string)
    push_succeeded: bool | None
    pr_created: bool | None  # a claim only; workers have no gh token (#1771)
    head_sha: str | None
    raw: Mapping[str, Any] = field(default_factory=lambda: _EMPTY_RAW)


@dataclass(frozen=True)
class TerminalEvidence:
    """The durable terminal-status record, if the watcher wrote one."""

    ended_at: datetime
    exit_code: int | None
    outcome: OutcomeEvidence | None = None  # embedded copy, source=TERMINAL


@dataclass(frozen=True)
class BranchEvidence:
    """Local/remote branch state. Two differently-named ``ahead`` fields:
    ``remote_ahead`` (pushed commits, rules 3/9) and ``unpushed`` (rules
    3/9's "stranded" count) are never the same field under two names.
    """

    has_remote: bool  # False on no-remote repos (Park lane)
    remote_head_sha: str | None  # live head of origin/<branch>; None = unknown/absent
    remote_ahead: int | None  # remote branch ahead of base (pushed commits)
    unpushed: int | None  # local HEAD ahead of remote (or of base if never pushed)
    open_pr_number: int | None  # from gh pr_list; None = no open PR
    pr_known: bool  # False when the PR lookup was not done or failed


@dataclass(frozen=True)
class FailureEvidence:
    """Adapter classifier output (rule 6). Produced upstream of this module
    until the adapter seam (design doc §7) lands; consumed here as data.
    """

    kind: str | None  # rate_limited|quota_exhausted|provider_suspended|provider_auth|...|None
    throttled_until: datetime | None
    fresh: bool  # True = classified from the log this pass; False = persisted fallback


@dataclass(frozen=True)
class FateEvidence:
    """Everything ``resolve_fate`` needs. Gathered once per pass by
    ``gather_evidence``; consumers that already hold some of it (the
    with-PR lane already has ``live_head_sha``, the no-PR lane already has
    ``pr_by_issue``) may build it directly instead, to avoid a double
    ``ls-remote`` during migration.
    """

    issue_number: int
    adapter: str  # harness name, key into the profile registry
    dispatched_at: datetime | None
    pid_alive: bool
    health: WorkerHealth | None  # only meaningful when pid_alive
    terminal: TerminalEvidence | None
    worktree_outcome: OutcomeEvidence | None
    branch: BranchEvidence
    failure: FailureEvidence | None = None  # None while alive or with no log/persisted kind


@dataclass(frozen=True)
class StaleEvidence:
    """One piece of evidence ignored under rule 1 (freshness)."""

    source: EvidenceSource
    reason: StaleReason
    written_at: datetime | None
    evidence_head: str | None
    live_head: str | None


@dataclass(frozen=True)
class FateBasis:
    """Fields shared by every fate variant, for logging/events."""

    issue_number: int
    pid_alive: bool
    outcome: OutcomeEvidence | None  # the fresh outcome that decided, if any
    exit_code: int | None
    stale: tuple[StaleEvidence, ...]
    rule: str  # "R2", "R4-nodispatch", ...: which rule decided


# --------------------------------------------------------------------------
# Fate union (§2). Invalid combinations cannot be represented: ``Live`` has
# no failure kind, ``Throttled`` always has one, and "completed but no PR"
# is a distinct variant (``PushedWithoutPr``), not an optional field on
# ``Completed``.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Live:
    basis: FateBasis
    health: WorkerHealth | None  # None only when the health reader itself failed


@dataclass(frozen=True)
class Completed:
    basis: FateBasis
    pr_number: int
    head_sha: str | None


@dataclass(frozen=True)
class PushedWithoutPr:
    basis: FateBasis
    head_sha: str | None


@dataclass(frozen=True)
class Stranded:
    basis: FateBasis
    unpushed: int
    park: bool  # not branch.has_remote
    failure: FailureEvidence | None


@dataclass(frozen=True)
class Blocked:
    basis: FateBasis
    reason: str | None


@dataclass(frozen=True)
class Throttled:
    basis: FateBasis
    failure: FailureEvidence


@dataclass(frozen=True)
class Crashed:
    basis: FateBasis
    failure: FailureEvidence | None


WorkerFate = Live | Completed | PushedWithoutPr | Stranded | Blocked | Throttled | Crashed


# --------------------------------------------------------------------------
# Liveness (§2 "is_alive is the one liveness entry point").
# --------------------------------------------------------------------------


def is_alive(pid: int | None, process_start_time: float | None) -> bool:
    """True when ``pid`` names a live process matching ``process_start_time``.

    Delegates to ``process_utils.is_pid_alive`` (the unchanged primitive,
    including its asymmetric fail-open/fail-closed behaviour) for every
    real PID. The only addition is ``pid is None``, which is always dead —
    e.g. a manual-adapter subject that never had a process to begin with.
    """
    if pid is None:
        return False
    return _process_is_pid_alive(pid, process_start_time)


# --------------------------------------------------------------------------
# Freshness (rules 1 and 7): step 0 of resolution.
# --------------------------------------------------------------------------


def _rule_suffix(dispatched_at: datetime | None) -> str:
    return "-nodispatch" if dispatched_at is None else ""


def _dispatch_fresh(written_at: datetime | None, dispatched_at: datetime | None) -> bool:
    """``written_at > dispatched_at``; legacy mode (``dispatched_at is None``)
    accepts unconditionally. An unknown ``written_at`` cannot prove
    freshness, so it fails the check (except in legacy mode).
    """
    if dispatched_at is None:
        return True
    return written_at is not None and written_at > dispatched_at


def _head_fresh(evidence_head: str | None, live_head: str | None) -> bool:
    """``head_sha == remote_head_sha`` only when both are known; an unknown
    head never proves staleness.
    """
    if evidence_head is None or live_head is None:
        return True
    return evidence_head == live_head


def _evaluate_candidate(
    candidate: OutcomeEvidence | None,
    *,
    dispatched_at: datetime | None,
    live_head: str | None,
) -> tuple[bool, StaleEvidence | None]:
    """Return (fresh, stale_entry_or_None) for one outcome candidate.

    A candidate that is absent (``None``, i.e. "no claim") is neither fresh
    nor stale — the caller skips it entirely.
    """
    if candidate is None:
        return False, None
    dispatch_ok = _dispatch_fresh(candidate.written_at, dispatched_at)
    head_ok = _head_fresh(candidate.head_sha, live_head)
    if dispatch_ok and head_ok:
        return True, None
    reason = StaleReason.HEAD_MISMATCH if not head_ok else StaleReason.OLDER_THAN_DISPATCH
    stale = StaleEvidence(
        source=candidate.source,
        reason=reason,
        written_at=candidate.written_at,
        evidence_head=candidate.head_sha,
        live_head=live_head,
    )
    return False, stale


@dataclass(frozen=True)
class _FreshOutcome:
    """The step-0 result: the winning candidate (or None) plus the
    freshness-gated exit code, and every stale entry found along the way.
    """

    outcome: OutcomeEvidence | None
    outcome_suffix: str  # "-nodispatch" iff the winning candidate used legacy mode
    exit_code: int | None
    stale: tuple[StaleEvidence, ...]


def _resolve_freshness(evidence: FateEvidence) -> _FreshOutcome:
    live_head = evidence.branch.remote_head_sha
    stale: list[StaleEvidence] = []

    terminal_outcome = evidence.terminal.outcome if evidence.terminal else None
    terminal_fresh, terminal_stale = _evaluate_candidate(
        terminal_outcome, dispatched_at=evidence.dispatched_at, live_head=live_head
    )
    if terminal_stale is not None:
        stale.append(terminal_stale)

    worktree_fresh, worktree_stale = _evaluate_candidate(
        evidence.worktree_outcome, dispatched_at=evidence.dispatched_at, live_head=live_head
    )
    if worktree_stale is not None:
        stale.append(worktree_stale)

    if terminal_fresh:
        chosen = terminal_outcome
        suffix = _rule_suffix(evidence.dispatched_at)
    elif worktree_fresh:
        chosen = evidence.worktree_outcome
        suffix = _rule_suffix(evidence.dispatched_at)
    else:
        chosen = None
        suffix = ""

    exit_code: int | None = None
    if evidence.terminal is not None:
        if _dispatch_fresh(evidence.terminal.ended_at, evidence.dispatched_at):
            exit_code = evidence.terminal.exit_code
        else:
            stale.append(
                StaleEvidence(
                    source=EvidenceSource.TERMINAL,
                    reason=StaleReason.OLDER_THAN_DISPATCH,
                    written_at=evidence.terminal.ended_at,
                    evidence_head=None,
                    live_head=live_head,
                )
            )

    return _FreshOutcome(
        outcome=chosen, outcome_suffix=suffix, exit_code=exit_code, stale=tuple(stale)
    )


# --------------------------------------------------------------------------
# resolve_fate: the nine rules as one ordered table (§3).
# --------------------------------------------------------------------------


def _is_pushed(outcome: OutcomeEvidence, branch: BranchEvidence) -> bool:
    """``remote_head_sha == outcome.head_sha``, or head unknown and
    ``remote_ahead > 0`` (rows 2-4). An unknown value never proves a push.
    """
    if branch.remote_head_sha is not None:
        return outcome.head_sha is not None and branch.remote_head_sha == outcome.head_sha
    return branch.remote_ahead is not None and branch.remote_ahead > 0


def _completed_or_pushed(
    basis: FateBasis, branch: BranchEvidence, *, head_sha: str | None
) -> Completed | PushedWithoutPr:
    if branch.open_pr_number is not None:
        return Completed(basis=basis, pr_number=branch.open_pr_number, head_sha=head_sha)
    return PushedWithoutPr(basis=basis, head_sha=head_sha)


def resolve_fate(evidence: FateEvidence, *, now: datetime) -> WorkerFate:  # noqa: ARG001
    """Resolve one worker's fate. Pure: no I/O, no clock read (``now`` is
    accepted for interface symmetry with ``gather_evidence``/``worker_fate``
    and future use by a rule that needs it; none of the current nine do).

    First match wins, evaluated only after step 0 (freshness, rules 1/7)
    picks the winning outcome candidate. See the design doc §3 for the
    rule-by-rule rationale, especially the rows-2-4-vs-6 precedence note:
    a fresh declared push beats a later, trailing unpushed commit so a
    salvage push after completion cannot turn a completion into a stray
    stranded classification.

    Corollary (pinned by ``test_worker_fate.py::test_row4_known_head_mismatch_*``):
    row 4 ("declared push, remote doesn't show it") can only ever see an
    outcome whose ``head_sha`` is unknown. A *known* ``head_sha`` that
    mismatches the live remote head is discarded as stale by step 0 before
    row 4 is reached at all -- which is exactly what the precedence note
    above depends on: a salvage push moves the remote head, so a stale
    outcome's head claim stops matching on the very next resolve.
    """
    fresh = _resolve_freshness(evidence)
    outcome = fresh.outcome
    branch = evidence.branch

    def basis(rule: str, *, outcome_used: OutcomeEvidence | None = outcome) -> FateBasis:
        return FateBasis(
            issue_number=evidence.issue_number,
            pid_alive=evidence.pid_alive,
            outcome=outcome_used,
            exit_code=fresh.exit_code,
            stale=fresh.stale,
            rule=rule,
        )

    # Row 1 (R2): fresh outcome "blocked" beats everything, alive or dead.
    if outcome is not None and outcome.outcome == "blocked":
        reason = outcome.raw.get("reason") if isinstance(outcome.raw, Mapping) else None
        return Blocked(basis=basis(f"R2{fresh.outcome_suffix}"), reason=reason)

    if outcome is not None and outcome.push_succeeded is True:
        pushed = _is_pushed(outcome, branch)
        head_sha = outcome.head_sha or branch.remote_head_sha
        # Row 2/3 (R4+R5+R8 / R8): a declared, remote-confirmed push.
        if pushed:
            if branch.open_pr_number is not None:
                return Completed(
                    basis=basis(f"R4+R5+R8{fresh.outcome_suffix}"),
                    pr_number=branch.open_pr_number,
                    head_sha=head_sha,
                )
            if branch.pr_known:
                return PushedWithoutPr(
                    basis=basis(f"R8{fresh.outcome_suffix} (pr_created claim ignored)"),
                    head_sha=head_sha,
                )
            # PR existence unknown: neither Completed nor PushedWithoutPr can
            # be proven. Falls through to the liveness/dead rows below.
        # Row 4 (R3+R9): declared push, remote doesn't show it, local
        # unpushed commits exist -> stranded, not silently dropped.
        elif branch.unpushed is not None and branch.unpushed > 0:
            return Stranded(
                basis=basis(f"R3+R9{fresh.outcome_suffix}"),
                unpushed=branch.unpushed,
                park=not branch.has_remote,
                failure=evidence.failure,
            )

    # Row 5 (R5): still running, and rows 1-4 above didn't already decide it.
    if evidence.pid_alive:
        return Live(basis=basis("R5"), health=evidence.health)

    # Row 6 (R3+R9): dead, local-only commits -> stranded (salvage; park on
    # no-remote repos). Carries the failure kind too, so a throttle rearm
    # can still apply after the commits are salvaged.
    if branch.unpushed is not None and branch.unpushed > 0:
        return Stranded(
            basis=basis("R3+R9"),
            unpushed=branch.unpushed,
            park=not branch.has_remote,
            failure=evidence.failure,
        )

    # Row 7 (R6): dead, no fresh outcome, a provider throttle classified it.
    if (
        outcome is None
        and evidence.failure is not None
        and is_provider_throttle_failure(evidence.failure.kind)
    ):
        return Throttled(basis=basis("R6"), failure=evidence.failure)

    # Row 8 (R4): dead, no fresh outcome, a fresh clean exit with published
    # commits -- the exit code decides only because no outcome file exists.
    if (
        outcome is None
        and fresh.exit_code == 0
        and (
            (branch.remote_ahead is not None and branch.remote_ahead > 0)
            or branch.open_pr_number is not None
        )
    ):
        return _completed_or_pushed(basis("R4"), branch, head_sha=branch.remote_head_sha)

    # Row 9 (R3): dead, no outcome at all, remote shows pushed commits
    # regardless of exit code.
    if outcome is None and branch.remote_ahead is not None and branch.remote_ahead > 0:
        return _completed_or_pushed(basis("R3"), branch, head_sha=branch.remote_head_sha)

    # Row 10: otherwise, a dead worker with nothing to credit or salvage.
    return Crashed(basis=basis("R10"), failure=evidence.failure)


# --------------------------------------------------------------------------
# Evidence gathering (§4). The only laziness: the remote branch read runs
# only when the PID is dead or a candidate outcome claims a push.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkerSubject:
    """What ``gather_evidence`` needs to know about one dispatched worker."""

    issue_number: int
    adapter: str
    pid: int | None
    process_start_time: float | None
    dispatched_at: datetime | None
    worktree_path: str | None
    branch: str | None
    log_path: str | None


@dataclass(frozen=True)
class FateReaders:
    """A bundle of plain callables. Each returns a plain value (or ``None``
    on failure -- readers never raise) and takes no arguments beyond what
    is bound in by the caller (typically via ``functools.partial`` or a
    closure over ``subject``). Kept this shallow on purpose: the decision
    logic lives entirely in ``resolve_fate``, never behind this bundle.
    """

    pid_alive: Callable[[], bool]
    health: Callable[[], WorkerHealth | None]
    terminal_record: Callable[[], TerminalEvidence | None]
    worktree_outcome: Callable[[], OutcomeEvidence | None]
    branch: Callable[[bool], BranchEvidence]  # arg: whether to do the remote read
    log_tail: Callable[[], str | None]
    persisted_failure_kind: Callable[[], FailureEvidence | None]


def _claims_push(outcome: OutcomeEvidence | None) -> bool:
    return outcome is not None and outcome.push_succeeded is True


def gather_evidence(
    subject: WorkerSubject, readers: FateReaders, *, now: datetime
) -> FateEvidence:  # noqa: ARG001
    """Read exactly what ``resolve_fate`` needs, in cost order (§4):

    1. ``pid_alive``
    2. ``health`` (alive only)
    3. ``terminal_record``
    4. ``worktree_outcome``
    5. ``branch``: local ``unpushed`` always; the remote read (``ls-remote``,
       ``remote_ahead``) only when the PID is dead **or** a candidate
       outcome claims ``push_succeeded`` -- the one place this function is
       lazy, pinned by the gather-policy tests below.
    6. ``log_tail`` (dead only) -- kept as raw evidence; classifying it into
       a ``FailureEvidence`` is the adapter seam (design doc §7), not built
       yet, so a dead worker's ``failure`` currently comes only from
       ``persisted_failure_kind`` (step 7).
    7. ``persisted_failure_kind`` (dead only, no fresh classification yet)

    ``now`` is accepted for interface symmetry with ``resolve_fate`` /
    ``worker_fate`` and so a future reader can be handed a frozen clock;
    none of the current readers need it directly (each closes over its own
    clock if it needs one).
    """
    pid_alive = readers.pid_alive()
    health = readers.health() if pid_alive else None
    terminal = readers.terminal_record()
    worktree_outcome = readers.worktree_outcome()

    wants_remote = (
        (not pid_alive)
        or _claims_push(terminal.outcome if terminal else None)
        or _claims_push(worktree_outcome)
    )
    branch = readers.branch(wants_remote)

    failure: FailureEvidence | None = None
    if not pid_alive:
        readers.log_tail()  # read for future classification; not yet interpreted here
        failure = readers.persisted_failure_kind()

    return FateEvidence(
        issue_number=subject.issue_number,
        adapter=subject.adapter,
        dispatched_at=subject.dispatched_at,
        pid_alive=pid_alive,
        health=health,
        terminal=terminal,
        worktree_outcome=worktree_outcome,
        branch=branch,
        failure=failure,
    )


def worker_fate(subject: WorkerSubject, readers: FateReaders, *, now: datetime) -> WorkerFate:
    """Convenience: gather, then resolve."""
    evidence = gather_evidence(subject, readers, now=now)
    return resolve_fate(evidence, now=now)
