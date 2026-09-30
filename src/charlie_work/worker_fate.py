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

``wf-4-wire-a`` adds the internal Adapter seam (design doc §7):
``classify_failure`` merges ``claude_code._classify_session_failure`` and
``devin_failure_classification._classify_session_failure`` (issue #1997 —
the devin-only emission-time throttle anchor, later ported fleet-wide by
rule 6, so it is no longer a per-profile flag); ``AdapterFateProfile``/
``profile_for`` replace the 14
``w.adapter_kind ==`` branches in ``dead_worker_reap.py`` and the two
``_classify_session_failure`` duplicates in ``claude_code.py`` /
``devin_shell.py`` (now thin wrappers around ``classify_failure``).

Still deliberately NOT in this commit (see the design doc §6 and the
plan's Group A/B tables): ``persisted_failure``/``persist_fate``/
``stale_evidence_events`` (they depend on a fate-resolution loop actually
running over ``dead_worker_reap.py``'s workers, which is Group B, not this
wiring step — the 3 direct ``record_dead_worker_failure_kind`` calls stay
in place), and ``default_readers`` production wiring. Until then,
``FateEvidence.failure`` is still supplied by the caller (or a test) as
already-classified data — ``resolve_fate`` only ever reads
``FailureEvidence.kind``/``throttled_until``, never produces them.

``is_worker_alive`` (claude_code.py) / ``is_session_alive`` (devin_shell.py)
are deliberately NOT touched by this commit either, despite being the
other half of design doc §7's "what it deletes" list (A2, not A3/A4).
``tests/test_charlie_work_dispatch_phantom.py`` monkeypatches
``charlie_work.claude_code.is_pid_alive`` directly (with an explicit
comment explaining why: it is the only way to pin PID 4242's liveness
without depending on host process-table state) — a patch that only stays
effective while ``is_worker_alive``'s body keeps calling its own module's
``is_pid_alive`` name literally. Redirecting the body to
``worker_fate.is_alive`` would silently make that patch inert. A2's
26-call-site rewiring (including ``WorkerView.is_alive`` in ``worker.py``,
outside this commit's file list) is left for its own step, where the
monkeypatch can be updated deliberately alongside the body change instead
of as a side effect of this one.

No consumer resolves a full ``WorkerFate`` yet (Group B). Wiring happens
in later commits named after the plan's Group A/B steps.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .config import OrchestratorConfig
from .process_utils import is_pid_alive as _process_is_pid_alive
from .state import parse_iso_timestamp as _state_parse_iso_timestamp
from .throttle_signatures import (
    is_provider_auth_failure,
    is_provider_throttle_failure,
    match_quota_tail,
    match_throttle_tail,
)
from .worker import WorkerHealth, WorkerView

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

    def __post_init__(self) -> None:
        # N7 (wf-review-opus.md): every construction site passes the
        # live, mutable dict it just parsed as `raw`. Freezing the
        # dataclass does not freeze `raw`'s own contents -- a caller
        # mutation (e.g. of `worker_outcome` in a live-handoff candidate
        # dict built alongside this evidence) would otherwise be visible
        # through the "frozen" evidence after the fact. Wrap in
        # `MappingProxyType` so `raw` is genuinely immutable regardless of
        # what the caller does to its own copy afterward.
        if not isinstance(self.raw, MappingProxyType):
            object.__setattr__(self, "raw", MappingProxyType(dict(self.raw)))


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


def _carries_no_claim(candidate: OutcomeEvidence) -> bool:
    """N1 (wf-review-opus.md) / design doc §3 step 0: an empty terminal or
    worktree outcome dict (``{}``) is "no claim: neither stale nor
    decisive" -- distinct from *absent* (``None``), but the SAME as absent
    for freshness-arbitration purposes. ``_no_pr_outcome_evidence`` /
    ``rework_outcome._outcome_evidence`` both still build a real
    ``OutcomeEvidence`` for ``{}`` (all four claim fields ``None``) rather
    than returning ``None`` themselves, so the check belongs here -- the
    single point every consumer's freshness step already funnels through --
    instead of being duplicated at (and possibly missed by) each of their
    construction sites.
    """
    return (
        candidate.outcome is None
        and candidate.push_succeeded is None
        and candidate.pr_created is None
        and candidate.head_sha is None
    )


def _evaluate_candidate(
    candidate: OutcomeEvidence | None,
    *,
    dispatched_at: datetime | None,
    live_head: str | None,
) -> tuple[bool, StaleEvidence | None]:
    """Return (fresh, stale_entry_or_None) for one outcome candidate.

    A candidate that is absent (``None``, i.e. "no claim") is neither fresh
    nor stale — the caller skips it entirely. The same holds for a present
    candidate that carries no actual claim (every field ``None`` --
    ``_carries_no_claim``): it must not out-rank a sibling candidate that
    does carry real content just because it happens to be "fresher" by
    timestamp alone.
    """
    if candidate is None or _carries_no_claim(candidate):
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
    """``remote_head_sha == outcome.head_sha`` when both are known, else fall
    back to ``remote_ahead > 0`` (rows 2-4). An unknown value never proves a
    push -- but "unknown" means either side is missing, not just the branch
    side: an outcome that never reported its own ``head_sha`` (a common,
    legitimate shape -- the worker still confirmed the push) must not lose a
    real, available ``remote_ahead`` signal just because the head-SHA compare
    itself could not run.
    """
    if branch.remote_head_sha is not None and outcome.head_sha is not None:
        return branch.remote_head_sha == outcome.head_sha
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

    # Rows 8/9 read git-confirmed evidence (exit code / remote ahead-count),
    # not the outcome file. They must fire not only when there is no fresh
    # outcome at all, but also when a fresh outcome exists but never
    # confirmed a push (``push_succeeded`` False or missing, e.g. a
    # rework-shaped outcome): by this point rows 1-4 have already returned
    # for every case where the outcome itself decides push/PR status
    # (``blocked``, or a push confirmed at the remote), so a non-confirming
    # outcome has nothing left to say and must not shadow real remote
    # evidence. Gating on bare ``outcome is None`` silently dropped
    # confirmed pushed work whenever any non-push-claiming outcome existed
    # -- including the #1248 salvage-push case, where the push lands on the
    # remote after the worker's own (non-confirming) outcome file was
    # written (B1, wf-review-opus.md). An outcome that *does* claim
    # ``push_succeeded is True`` but couldn't be confirmed here (unknown PR
    # existence, or no branch evidence at all) still defers to the liveness/
    # dead rows below rather than rows 8/9 -- that self-report-with-no-
    # evidence case stays exactly as before (see ``tests/test_issue_1006.py``).
    outcome_has_no_push_opinion = outcome is None or outcome.push_succeeded is not True

    # Row 8 (R4): dead, no outcome deciding push/PR status, a fresh clean
    # exit with published commits -- the exit code decides only because no
    # outcome file settles it.
    if (
        outcome_has_no_push_opinion
        and fresh.exit_code == 0
        and (
            (branch.remote_ahead is not None and branch.remote_ahead > 0)
            or branch.open_pr_number is not None
        )
    ):
        return _completed_or_pushed(basis("R4"), branch, head_sha=branch.remote_head_sha)

    # Row 9 (R3): dead, no outcome deciding push/PR status, remote shows
    # pushed commits regardless of exit code.
    if outcome_has_no_push_opinion and branch.remote_ahead is not None and branch.remote_ahead > 0:
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


# --------------------------------------------------------------------------
# Failure classification (§6) and the internal Adapter seam (§7).
#
# ``classify_failure`` merges ``claude_code._classify_session_failure`` and
# ``devin_failure_classification._classify_session_failure``. The two were
# identical except for two things, both now data instead of two copies of
# the function:
#   - account-error detection (``provider_suspended``/``provider_auth``,
#     api only) was ``adapter_kind == "api"``; now ``account_error_detection``.
#   - the ``rate_limited`` cooldown anchor: devin already anchored at the
#     message's *emission* time (issue #1997 -- classification can run tens
#     of minutes after the log line was written, so anchoring "now" plus a
#     15-minute cooldown overshoots the real provider reset by that much);
#     claude-code/api used to anchor at *classification* time instead.
#     Rule 6 (wf-design.md §9) ports #1997 to claude-code/api fleet-wide:
#     every harness now anchors at emission time, unconditionally.
# --------------------------------------------------------------------------

_CLASSIFY_DEFAULT_THROTTLE_ERROR_MARKERS = OrchestratorConfig().runtime.throttle_error_markers
_CLASSIFY_DEFAULT_QUOTA_ERROR_MARKERS = OrchestratorConfig().runtime.quota_error_markers
_DEFAULT_RATE_LIMIT_COOLDOWN_MINUTES = 15
_DEFAULT_QUOTA_COOLDOWN_HOURS = 24

# Provider authentication failures (issue #484). Matched against the log tail
# of account-error-detecting (api) sessions only.
#
# N3 (wf-review-opus.md, wf-8-review-fixes): this used to be a byte-duplicate
# of claude_code.py's own compiled copy ("moved verbatim from claude_code.py"
# -- the two could drift silently). Both now call
# ``throttle_signatures.is_provider_auth_failure`` (single point of
# enforcement) -- see its docstring for the word-boundary 401/403
# false-positive rationale.
#
# Provider account suspension / insufficient-balance responses (issue #1342).
# Moved verbatim from claude_code.py. The billing phrase alone is not
# enough -- ``_provider_suspension_in_tail`` requires it to co-occur on the
# same log line as a structural API-error signal (HTTP 402 or a CLI
# ``Error:``/``API Error:`` prefix), or a worker merely quoting/reviewing the
# trigger phrase would misclassify.
_PROVIDER_SUSPENDED_PHRASE = re.compile(
    r"insufficient\s+(?:balance|funds|credit)"
    r"|account\s+(?:is\s+)?suspended"
    r"|suspended\s+due\s+to\s+(?:insufficient\s+balance|billing|payment|unpaid)"
    r"|recharge\s+your\s+account"
    r"|please\s+recharge",
    re.IGNORECASE,
)
_PROVIDER_SUSPENDED_ANCHOR = re.compile(
    r"^\s*(?:api\s+)?error\s*:|\b402\b",
    re.IGNORECASE,
)


def _provider_suspension_in_tail(tail: str) -> bool:
    """True if ``tail`` has a structurally-anchored account-suspension
    signature: the billing phrase and the API-error anchor on the SAME line.
    """
    for line in tail.splitlines():
        if _PROVIDER_SUSPENDED_PHRASE.search(line) and _PROVIDER_SUSPENDED_ANCHOR.search(line):
            return True
    return False


# Headless permission-denial signature (issue #2010): a ``claude -p`` session
# that cannot answer a permission prompt ends by asking the operator to
# approve command execution. That is a config defect, not a blocked task, so
# it gets its own failure kind (``permission_denied``) instead of escalating.
# Moved verbatim from claude_code.py (origin/main 7293cb71).
PERMISSION_DENIED_FAILURE_KIND = "permission_denied"
_HEADLESS_PERMISSION_DENIAL_PATTERN = re.compile(
    r"approve\s+(?:the\s+)?(?:command|bash|tool)\s+execution|"
    r"requires?\s+(?:your\s+)?approval|"
    r"(?:command|tool)\s+(?:was|were)\s+(?:denied|not\s+allowed)|"
    r"permission\s+to\s+run\s+(?:this|these|the)\s+(?:command|bash)",
    re.IGNORECASE,
)


def is_headless_permission_denial(text: str) -> bool:
    """True when ``text`` (a log tail or outcome detail) shows the headless
    permission-denial signature."""
    return bool(_HEADLESS_PERMISSION_DENIAL_PATTERN.search(text))


# Issue #1997: a tz-aware ISO-8601 timestamp on a tail line marks when the
# provider emitted the throttle message -- a better anchor than the
# classification-time clock. Moved verbatim from devin_failure_classification.py.
_TAIL_LINE_TS_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})"
)


def _tail_emission_timestamp(tail: str) -> datetime | None:
    """The tz-aware timestamp on the tail's last timestamped line, scanning
    backwards so the most recent one wins. None when no line has one --
    naive (offset-less) timestamps are skipped since their zone is unknown.
    """
    for line in reversed(tail.splitlines()):
        match = _TAIL_LINE_TS_PATTERN.search(line)
        if match is None:
            continue
        try:
            return datetime.fromisoformat(match.group(0))
        except ValueError:
            continue
    return None


def _throttle_emission_anchor(log_path: Path, tail: str, *, now: datetime) -> datetime:
    """Anchor for a provider-throttle window: when the message was emitted,
    not when it was classified. The tail's own timestamp when a line carries
    one, else the log's mtime (the last write to a dead worker's log is the
    death message), else ``now``. Clamped to ``now``: a future mtime/tail
    timestamp is clock/mtime skew, not evidence the reset also moved.
    """
    anchor = _tail_emission_timestamp(tail)
    if anchor is None:
        try:
            anchor = datetime.fromtimestamp(log_path.stat().st_mtime, tz=UTC)
        except OSError:
            anchor = now
    return min(anchor, now)


def classify_failure(
    log_path: Path,
    throttle_error_markers: Sequence[str] | None = None,
    *,
    quota_error_markers: Sequence[str] | None = None,
    resume_margin_seconds: int = 0,
    account_error_detection: bool = False,
    headless_permission_detection: bool = False,
    now: datetime | None = None,
) -> tuple[str | None, str | None]:
    """Classify a session failure by matching the log tail against provider
    throttle/auth/suspension signatures. Called after a session exits.

    Returns (failure_kind, throttled_until_iso):
    - failure_kind: "provider_suspended" | "provider_auth" | "rate_limited" |
      "quota_exhausted" | "permission_denied" | None
    - throttled_until_iso: ISO timestamp the cooldown ends, or None (always
      None for "provider_suspended" -- terminal, no cooldown)

    ``account_error_detection`` (api only) enables the ``provider_suspended``
    (#1342) and ``provider_auth`` (#484) checks, both checked before
    quota/throttle so neither masquerades as a transient issue.

    ``headless_permission_detection`` (claude-code, issue #2010) enables the
    ``permission_denied`` check. It is checked LAST so throttle/auth
    signatures still win.

    The ``rate_limited`` anchor is always the message's emission time
    (issue #1997 -- see ``_throttle_emission_anchor``), for every harness.
    Rule 6 (design doc, FLIP 6): claude-code/api used to anchor at
    classification time instead; devin already used the emission anchor, so
    this is the one fleet-wide throttle-timing change in the nine-rule
    resolution. ``quota_exhausted`` always anchors at classification time --
    its fixed 24h cooldown has no provider-stated reset to anchor against,
    so rule 6 does not affect it.

    ``now`` is the injectable clock: defaults to ``datetime.now(UTC)`` when
    not supplied, so production behaviour is byte-identical (issue #822).
    """
    if not log_path.exists():
        return None, None

    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None

    resolved_now = now if now is not None else datetime.now(UTC)
    # Check the last 2KB of the log (where error messages appear).
    tail = log_text[-2048:] if len(log_text) > 2048 else log_text

    if account_error_detection and _provider_suspension_in_tail(tail):
        return "provider_suspended", None

    if account_error_detection and is_provider_auth_failure(tail):
        cooldown = timedelta(hours=_DEFAULT_QUOTA_COOLDOWN_HOURS, seconds=resume_margin_seconds)
        throttled_until = resolved_now + cooldown
        return "provider_auth", throttled_until.replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        )

    quota_markers = (
        quota_error_markers
        if quota_error_markers is not None
        else _CLASSIFY_DEFAULT_QUOTA_ERROR_MARKERS
    )
    if match_quota_tail(tail, quota_markers):
        cooldown = timedelta(hours=_DEFAULT_QUOTA_COOLDOWN_HOURS, seconds=resume_margin_seconds)
        throttled_until = resolved_now + cooldown
        return "quota_exhausted", throttled_until.replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        )

    markers = (
        throttle_error_markers
        if throttle_error_markers is not None
        else _CLASSIFY_DEFAULT_THROTTLE_ERROR_MARKERS
    )
    matched, reset_minutes = match_throttle_tail(tail, markers)
    if matched:
        cooldown = timedelta(
            minutes=reset_minutes
            if reset_minutes is not None
            else _DEFAULT_RATE_LIMIT_COOLDOWN_MINUTES,
            seconds=resume_margin_seconds,
        )
        emitted_at = _throttle_emission_anchor(log_path, tail, now=resolved_now)
        throttled_until = max(resolved_now, emitted_at + cooldown)
        return "rate_limited", throttled_until.replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        )

    # Issue #2010: last, so throttle/auth signatures still win.
    if headless_permission_detection and is_headless_permission_denial(tail):
        return PERMISSION_DENIED_FAILURE_KIND, None

    return None, None


@dataclass(frozen=True)
class AdapterFateProfile:
    """The internal Adapter seam (design doc §7): one place that knows how
    each harness reports liveness and failure, replacing the 14
    ``w.adapter_kind ==`` branches in ``dead_worker_reap.py``.
    """

    harness: str  # key in harnesses.HARNESS_REGISTRY / WORKER_HARNESSES
    view_kinds: frozenset[str]  # WorkerView.adapter_kind spellings ("devin" for devin-shell)
    writes_terminal_record: bool  # claude-code, api: True; devin-shell: False (follow-up)
    probes_process: bool  # manual: False (no PID); others: True
    account_error_detection: bool  # api only: provider_suspended / provider_auth
    record_failure: Callable[..., tuple[str | None, str | None]] | None
    # (sessions_dir, issue_number, *, fallback_kind, config, now) -> the
    # existing update_worker_record_with_failure_classification /
    # update_session_record_with_failure_classification sidecar writer.
    over_budget: Callable[[WorkerView, OrchestratorConfig], bool] | None  # api only


_PROFILES: dict[str, AdapterFateProfile] | None = None


def _build_profiles() -> dict[str, AdapterFateProfile]:
    """Build the by-harness profile table, then index it by ``view_kinds``.

    Imports the adapter modules lazily (function body, not module top
    level): ``claude_code``/``devin_shell`` reach back into this module's
    ``classify_failure`` from inside their own functions, so a top-level
    import here would cycle. This function only ever runs at call time
    (from ``profile_for``), by which point every module involved has
    already finished loading, so the cycle risk does not apply to a lazy
    import -- only to a module-level one.

    ``update_worker_record_with_failure_classification`` (claude-code, api)
    takes an ``adapter_kind`` kwarg that selects both the sidecar filename
    suffix and account-error detection -- unlike ``update_session_record_
    with_failure_classification`` (devin), which has no such parameter.
    ``functools.partial`` binds each profile's value in at registry-build
    time so every call site can call ``record_failure`` with the exact same
    positional/keyword shape regardless of which adapter it resolved to,
    without changing either function's own signature.
    """
    from functools import partial

    from .claude_code import update_worker_record_with_failure_classification
    from .devin_shell import update_session_record_with_failure_classification
    from .harnesses import WORKER_HARNESSES
    from .worker import _api_session_over_budget

    by_harness = {
        "devin-shell": AdapterFateProfile(
            harness="devin-shell",
            view_kinds=frozenset({"devin"}),
            writes_terminal_record=False,
            probes_process=True,
            account_error_detection=False,
            record_failure=update_session_record_with_failure_classification,
            over_budget=None,
        ),
        "claude-code": AdapterFateProfile(
            harness="claude-code",
            view_kinds=frozenset({"claude-code"}),
            writes_terminal_record=True,
            probes_process=True,
            account_error_detection=False,
            record_failure=partial(
                update_worker_record_with_failure_classification, adapter_kind="claude-code"
            ),
            over_budget=None,
        ),
        "api": AdapterFateProfile(
            harness="api",
            view_kinds=frozenset({"api"}),
            writes_terminal_record=True,
            probes_process=True,
            account_error_detection=True,
            record_failure=partial(
                update_worker_record_with_failure_classification, adapter_kind="api"
            ),
            over_budget=_api_session_over_budget,
        ),
        # "command" and "manual" have no failure-classification or budget
        # consumer today: dead_worker_reap.py's 14 sites never branch on
        # either adapter_kind, and neither harness has an issue driving
        # account-error detection or an over-budget check. Declared here
        # (all capabilities off) purely so the WORKER_HARNESSES completeness
        # assert below covers all 5 harnesses, matching the adapters.py
        # #1513 pattern this seam follows -- not because either capability
        # is known to be correct for them. A future consumer that needs
        # real values for these two should fill them in then, not infer
        # them from this placeholder.
        "command": AdapterFateProfile(
            harness="command",
            view_kinds=frozenset({"command"}),
            writes_terminal_record=False,
            probes_process=True,
            account_error_detection=False,
            record_failure=None,
            over_budget=None,
        ),
        "manual": AdapterFateProfile(
            harness="manual",
            view_kinds=frozenset({"manual"}),
            writes_terminal_record=False,
            probes_process=False,
            account_error_detection=False,
            record_failure=None,
            over_budget=None,
        ),
    }
    # N6 (wf-review-opus.md): an explicit raise, not a bare `assert` --
    # this runs lazily on first `profile_for` call (mid-reap-pass, not at
    # import), and a bare `assert` here is silently stripped under
    # `python -O`, disabling the completeness guard exactly when a
    # harness/profile drift would otherwise be caught.
    if {p.harness for p in by_harness.values()} != WORKER_HARNESSES:
        raise AssertionError(
            "worker_fate._build_profiles must declare exactly the harnesses "
            "harnesses.WORKER_HARNESSES declares valid -- keep both in sync "
            "(adapters.py #1513 completeness-assert pattern)"
        )
    return by_harness


def profile_for(adapter_kind: str) -> AdapterFateProfile | None:
    """Look up the seam profile by ``WorkerView.adapter_kind`` spelling
    (e.g. ``"devin"``, not the harness name ``"devin-shell"``).

    Returns ``None`` for an unrecognized value rather than raising. The
    design doc's §7 says an unknown value "raises KeyError at the seam" --
    deviated from deliberately here: the 4 triplet call sites in
    ``dead_worker_reap.py`` disagree today on what an unrecognized
    ``adapter_kind`` should do (one keeps a pre-set ``fallback_kind``,
    three default to ``(None, None)``), so a raising ``profile_for`` would
    turn each site's existing graceful degradation into a crash. Returning
    ``None`` lets every call site keep its own pre-existing fallback.
    """
    global _PROFILES
    if _PROFILES is None:
        by_harness = _build_profiles()
        _PROFILES = {
            view_kind: profile
            for profile in by_harness.values()
            for view_kind in profile.view_kinds
        }
    return _PROFILES.get(adapter_kind)


# --------------------------------------------------------------------------
# Group B shared helpers: consumers build ``FateEvidence`` from data they
# already hold (design doc §8) rather than going through ``gather_evidence``,
# so no double ``ls-remote`` runs during migration. These two are the pieces
# every consumer site needs and none of them should reimplement.
# --------------------------------------------------------------------------


# N8 (wf-review-opus.md, wf-8-review-fixes): ``parse_iso_timestamp`` used to
# be a byte-for-byte duplicate of ``workflow._parse_iso_timestamp``, kept
# separate only to dodge an import cycle (``workflow.py`` imports this
# module). ``state.parse_iso_timestamp`` is the shared leaf-module copy both
# now use; re-exported under this name so every existing
# ``worker_fate.parse_iso_timestamp(...)`` call site keeps working unchanged.
parse_iso_timestamp = _state_parse_iso_timestamp


# --------------------------------------------------------------------------
# B6 (wf-review-opus.md) / design doc §5: rule 1's stale-evidence event.
# ``FateBasis.stale`` (rule 1/7's freshness step) was populated from the
# start, but nothing built the event the plan promised ("Stale evidence is
# ignored **and emits a stale-evidence event**") or read ``basis.stale`` at
# all -- a prior-dispatch outcome (including a real pushed branch) could be
# silently dropped with zero signal in events.db. These two functions are
# the module's own answer (design doc §5); callers wire them in per site,
# same as every other Group B consumer.
# --------------------------------------------------------------------------


def stale_evidence_key(stale: StaleEvidence) -> str:
    """Dedup key for one ``StaleEvidence``: once per ``(source, written_at)``.

    Design doc §5's dedup key, keyed by ``entry["stale_evidence_reported"]``
    at the caller -- a dead worker re-swept every pass must not re-emit the
    same stale candidate and flood the 2000-entry events ring.
    """
    written = stale.written_at.isoformat() if stale.written_at is not None else "unknown"
    return f"{stale.source}:{written}"


def stale_evidence_events(
    entry: Mapping[str, Any], fate: WorkerFate
) -> list[tuple[str, dict[str, Any]]]:
    """Build ``(kind, payload)`` pairs for not-yet-reported rule-1 stale evidence.

    Design doc §5: kind ``worker_evidence_stale`` (level warning), one event
    per ``StaleEvidence`` in ``fate.basis.stale`` whose :func:`stale_evidence_key`
    is not already in ``entry.get("stale_evidence_reported")``. Payload:
    ``issue_number, source, reason, written_at, dispatched_at, evidence_head,
    live_head, adapter`` -- ``dispatched_at`` and ``adapter`` come from
    ``entry`` itself (``FateBasis`` does not carry either; every fate-computing
    call site already stamps ``entry["dispatched_at"]``/``entry["adapter"]``
    from the same evidence used to resolve the fate).

    Pure: does not mutate ``entry``. The caller emits the returned events
    through ``_record_event``/``append_event``/``log_event`` (ADR-0005; see
    CLAUDE.md's instrumentation invariant for which one applies where) and
    merges :func:`stale_evidence_key` for each of ``fate.basis.stale`` into
    ``entry["stale_evidence_reported"]`` itself, in the same state-lock
    section, so the dedup marker and the emitted events never drift apart.
    """
    basis = fate.basis
    already_reported = set(entry.get("stale_evidence_reported") or ())
    dispatched_at = entry.get("dispatched_at")
    adapter = entry.get("adapter")
    events: list[tuple[str, dict[str, Any]]] = []
    for stale in basis.stale:
        if stale_evidence_key(stale) in already_reported:
            continue
        events.append(
            (
                "worker_evidence_stale",
                {
                    "issue_number": basis.issue_number,
                    "source": str(stale.source),
                    "reason": str(stale.reason),
                    "written_at": stale.written_at.isoformat() if stale.written_at else None,
                    "dispatched_at": dispatched_at,
                    "evidence_head": stale.evidence_head,
                    "live_head": stale.live_head,
                    "adapter": adapter,
                },
            )
        )
    return events


@dataclass(frozen=True)
class PersistedFailure:
    """Read-side view of the persisted ``dead_worker_failure_kind`` (§6).

    The single accessor consumer sites use instead of reading
    ``entry.get("dead_worker_failure_kind")`` directly -- a guard test
    (``tests/test_worker_fate_seam.py``) AST-walks ``src/`` and fails if a
    ``.get("dead_worker_failure_kind")`` call or a
    ``[...]["dead_worker_failure_kind"]`` read appears outside ``state.py``
    (the field's owner) and this module.
    """

    kind: str | None
    is_throttle: bool


def persisted_failure(entry: Mapping[str, Any]) -> PersistedFailure:
    """Read ``dead_worker_failure_kind`` off a state entry (rule 6, read side).

    Never raises: a missing or malformed entry yields ``PersistedFailure(None,
    False)``, the same as no persisted classification existing at all.
    """
    kind = entry.get("dead_worker_failure_kind") if isinstance(entry, Mapping) else None
    return PersistedFailure(kind=kind, is_throttle=is_provider_throttle_failure(kind))
