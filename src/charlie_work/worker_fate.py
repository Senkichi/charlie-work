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
evidence"): every consumer already does its own reads (some under a lock)
and builds a ``FateEvidence`` by hand; ``resolve_fate`` is the pure, 9-rule
decision over it. The originally planned evidence-gathering reader bundle was
deleted (wf-r2-s1): wiring it in would only add a second ``ls-remote``/outcome
read per candidate.

``process_utils.is_pid_alive`` stays the liveness primitive, unchanged,
including its asymmetric fail-open/fail-closed behaviour; ``is_alive`` here
only adds the ``pid is None`` case (no process, e.g. a manual-adapter
subject). ``classify_worker_health`` and ``is_worker_confirmed_dead`` stay in
``worker.py`` — they own the inconclusive-probe deferral counter (#755) and
this module only consumes their output through ``FateEvidence.health``.

``wf-4-wire-a`` adds the internal Adapter seam (design doc §7):
``classify_failure`` merged the two ``_classify_session_failure`` copies
that lived in ``claude_code.py`` and ``devin_failure_classification.py``
(issue #1997 — the devin-only emission-time throttle anchor, later ported
fleet-wide by rule 6, so it is no longer a per-profile flag);
``AdapterFateProfile``/``profile_for`` replace the 14
``w.adapter_kind ==`` branches in ``dead_worker_reap.py``. Both copies are
now deleted: ``classify_for`` looks the profile up and passes its two
flags (``account_error_detection``, ``headless_permission_detection``)
to ``classify_failure``, and the adapters' sidecar writers
call ``classify_for``.

``FateEvidence.failure`` is supplied by the caller as already-classified
data — ``resolve_fate`` only ever reads ``FailureEvidence.kind``/
``throttled_until``, never produces them. :func:`report_stale_evidence` is the
one path every ``resolve_fate`` consumer uses to report rule 1's ignored
evidence (built from ``stale_evidence_events``). Rule 6's write side is :func:`persist_failure`, the
single primitive every ``dead_worker_failure_kind`` writer goes through
(``dead_worker_reap``, ``dead_worker_classification``, ``reconcile``); its
read side feeds the persisted kind back in as ``FateEvidence.failure``
(``PersistedFailure.as_evidence``) so ``Throttled`` is reachable, and
:func:`throttle_failure` reads the throttle decision off the fate.

``is_alive`` is the single liveness seam (design doc A2): ``WorkerView.is_alive``,
``dead_worker_reap`` and ``doctor`` call it through the module attribute, so
tests patch ``charlie_work.worker_fate.is_alive`` and nothing else.

Known exception to "this module owns post-exit fate": three consumers
(``live_handoff_finalize``, ``misc_worker_dispatch``, ``rework_outcome``) use
``resolve_fate`` only as the rule-1/7 freshness filter and then route on the
surviving ``basis.outcome``'s self-reported ``push_succeeded``/``head_sha``
themselves. Their evidence shapes never resolve to ``PushedWithoutPr`` or
``Completed`` (no PR knowledge), so the fate variant carries no decision for
them; those reads are legacy-compatible by design (wf-review-opus B9).

The ``entry["stale_evidence_reported"]`` dedup marker is deliberately keyed
``(source, written_at)`` with no dispatch epoch and is never cleared: a
leftover file is one piece of evidence however many dispatches it outlives, so
it is reported once per issue entry (design doc §5).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .write_gate import WriteGate

# Deliberate re-exports (file-size ratchet split): the public surface stays
# ``worker_fate.<name>`` for every consumer and test.
from .adapter_fate_profile import AdapterFateProfile, classify_for, profile_for  # noqa: F401
from .failure_classifier import classify_failure  # noqa: F401
from . import process_utils as _process_utils
from .state import parse_iso_timestamp as _state_parse_iso_timestamp
from .state import (
    record_dead_worker_failure_kind,
    set_throttled_until,
)
from .throttle_signatures import (
    is_provider_throttle_failure,
)
from .worker import WorkerHealth
from .worker_fate_stale import (  # noqa: F401
    collect_fate,
    report_stale_evidence,
    stale_evidence_events,
    stale_evidence_key,
    stale_terminal_fate,
)

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
    """Local/remote branch state. Three differently-named ``ahead`` fields:
    ``remote_ahead`` (pushed commits, rules 3/9), ``unpushed`` (rules 3/9's
    proven-"stranded" count) and ``local_ahead`` (local vs base only, N5) are
    never the same field under two names.
    """

    remote_head_sha: str | None  # live head of origin/<branch>; None = unknown/absent
    remote_ahead: int | None  # remote branch ahead of base (pushed commits)
    unpushed: int | None  # local HEAD ahead of remote (or of base if never pushed)
    open_pr_number: int | None  # from gh pr_list; None = no open PR
    pr_known: bool  # False when the PR lookup was not done or failed
    local_ahead: int | None = None
    # Local HEAD ahead of base; says nothing about the remote (N5). Used only
    # by row 6 when the remote is unknown, so a consumer that does no remote
    # read (dispatch-time routing) can still surface stranded-or-unknown work
    # without pretending the count is proven-unpushed.


@dataclass(frozen=True)
class FailureEvidence:
    """Adapter classifier output (rule 6). Produced upstream of this module
    until the adapter seam (design doc §7) lands; consumed here as data.
    """

    kind: str | None  # rate_limited|quota_exhausted|provider_suspended|provider_auth|...|None
    throttled_until: datetime | None
    fresh: bool  # True = classified from the log this pass; False = persisted fallback

    @classmethod
    def from_classification(
        cls, kind: str | None, throttled_until_iso: str | None, *, fresh: bool
    ) -> FailureEvidence:
        """Build from an adapter classifier's ``(failure_kind, throttled_until_iso)``.

        The classifier reports the cooldown end as an ISO string; parsing it
        here keeps ``throttled_until`` a real ``datetime`` for consumers,
        while :func:`persist_failure` formats it back in the classifier's own
        canonical form so the persisted value round-trips unchanged.
        """
        return cls(
            kind=kind,
            throttled_until=_state_parse_iso_timestamp(throttled_until_iso),
            fresh=fresh,
        )


@dataclass(frozen=True)
class FateEvidence:
    """Everything ``resolve_fate`` needs. Each consumer builds it from the
    reads it already does (the with-PR lane already has ``live_head_sha``,
    the no-PR lane already has ``pr_by_issue``), so no second ``ls-remote``
    runs on behalf of the resolver.
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
    # The dispatch epoch the freshness gate ran against; lets a stale-evidence
    # event carry it for a consumer whose state entry does not (wf-r2-s6).
    dispatched_at: datetime | None = None


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
    unpushed: int  # commits not proven on the remote
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
    real PID. The only additions are ``pid is None`` and ``pid <= 0``, which
    are always dead -- e.g. a manual-adapter subject that never had a process
    to begin with, or a sentinel pid.

    The single liveness seam for worker-fate resolution (see module docstring).
    The primitive is looked up at call time, so a patch of
    ``process_utils.is_pid_alive`` reaches it too. Reviewer, merge-gate and
    worktree-marker liveness checks (and ``worker.py``'s own probe, which this
    module imports) deliberately keep calling the primitive directly.
    """
    if pid is None or pid <= 0:
        return False
    return _process_utils.is_pid_alive(pid, process_start_time)


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
    accepted for interface symmetry and future use by a rule that needs it;
    none of the current nine do).

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
            dispatched_at=evidence.dispatched_at,
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
                failure=evidence.failure,
            )

    # Row 5 (R5): still running, and rows 1-4 above didn't already decide it.
    if evidence.pid_alive:
        return Live(basis=basis("R5"), health=evidence.health)

    # Row 6 (R3+R9): dead, local-only commits -> stranded (salvage; the no-origin
    # archive park is the salvage path's own call, not the fate's). Carries the failure kind too, so a throttle rearm
    # can still apply after the commits are salvaged.
    if branch.unpushed is not None and branch.unpushed > 0:
        return Stranded(
            basis=basis("R3+R9"),
            unpushed=branch.unpushed,
            failure=evidence.failure,
        )

    # Row 6, remote unknown (N5): the consumer did no remote read (or it
    # failed), so local commits cannot be split into pushed vs unpushed. They
    # are still commits not proven on the remote, so they are stranded --
    # under a distinct rule so an event reader can tell the count is
    # unverified. A known remote (any of remote_ahead/remote_head_sha) makes
    # ``local_ahead`` irrelevant: rows 8/9 read the remote evidence instead.
    if (
        branch.unpushed is None
        and branch.remote_ahead is None
        and branch.remote_head_sha is None
        and branch.local_ahead is not None
        and branch.local_ahead > 0
    ):
        return Stranded(
            basis=basis("R3+R9-remote-unknown"),
            unpushed=branch.local_ahead,
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
# Group B shared helpers: consumers build ``FateEvidence`` from data they
# already hold (design doc §8), so no double ``ls-remote`` runs. These are the
# pieces every consumer site needs and none of them should reimplement.
# --------------------------------------------------------------------------


# N8 (wf-review-opus.md, wf-8-review-fixes): ``parse_iso_timestamp`` used to
# be a byte-for-byte duplicate of ``workflow._parse_iso_timestamp``, kept
# separate only to dodge an import cycle (``workflow.py`` imports this
# module). ``state.parse_iso_timestamp`` is the shared leaf-module copy both
# now use; re-exported under this name so every existing
# ``worker_fate.parse_iso_timestamp(...)`` call site keeps working unchanged.
parse_iso_timestamp = _state_parse_iso_timestamp


def fresh_terminal_record(
    record: Mapping[str, Any] | None,
    dispatched_at: datetime | None,
    *,
    issue_number: int | None = None,
    on_fate: Callable[[WorkerFate], None] | None = None,
) -> Mapping[str, Any] | None:
    """The terminal record only when it belongs to the current dispatch.

    ``find_worker_terminal_status`` returns the newest ``issue-<n>.*.terminal.json``
    and records are never deleted, so an earlier dispatch's record is still
    returned after a redispatch whose own watcher never ran. Rule 1: the
    record's ``ended_at`` must be ``> dispatched_at`` before its ``exit_code``,
    ``pid`` or ``duration_seconds`` count; legacy entries with no
    ``dispatched_at`` accept unconditionally (``_dispatch_fresh``).

    A dropped record is reported, not silent: with ``issue_number`` and
    ``on_fate`` given, an evidence-only fate carrying the ``StaleEvidence`` goes
    to ``on_fate`` (see ``stale_terminal_fate``).
    """
    if not isinstance(record, Mapping):
        return None
    ended_at = parse_iso_timestamp(record.get("ended_at"))
    if _dispatch_fresh(ended_at, dispatched_at):
        return record
    if on_fate is not None and issue_number is not None:
        on_fate(stale_terminal_fate(issue_number, ended_at, dispatched_at))
    return None


def _format_utc_z(moment: datetime) -> str:
    """The canonical persisted timestamp form (matches ``state.utc_now`` and
    the classifier's ``throttled_until``): UTC, whole seconds, ``Z`` suffix."""
    aware = moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)
    return aware.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def persist_failure(
    state: dict[str, Any],
    issue_number: int,
    failure: FailureEvidence,
    *,
    adapter_kind: str | None,
    now: datetime,
    source: str,
    write_gate: WriteGate | None = None,
) -> dict[str, Any]:
    """Rule 6, write side: the single primitive that persists a classified failure.

    Every writer of ``dead_worker_failure_kind`` goes through this function
    (pinned by ``tests/test_worker_fate_seam.py``), so the cooldown and the
    epoch-scoped kind stamp can never drift apart.

    Pure: returns a new state. ``failure.throttled_until`` is written verbatim
    (never recomputed -- #1993/#1997); ``None`` leaves ``throttled_until`` alone,
    for callers that must not touch the fleet-wide cooldown. The kind stamp is a
    no-op when the issue has no entry or ``failure.kind`` is ``None``.
    ``source`` (issue #2006) names the caller (a literal, pinned by
    ``tests/test_throttle_window_set.py``); ``write_gate`` routes the event.
    """
    new_state = state
    if failure.throttled_until is not None:
        new_state = set_throttled_until(
            new_state,
            _format_utc_z(failure.throttled_until),
            source=source,
            reason=failure.kind,
            adapter_kind=adapter_kind,
            write_gate=write_gate,
        )
    if failure.kind is not None:
        new_state = record_dead_worker_failure_kind(
            new_state,
            issue_number,
            failure.kind,
            classified_at=_format_utc_z(now),
        )
    return new_state


@dataclass(frozen=True)
class PersistedFailure:
    """Read-side view of the persisted ``dead_worker_failure_kind`` (§6).

    The single accessor consumer sites use instead of reading
    ``entry.get("dead_worker_failure_kind")`` directly -- a guard test
    (``tests/test_worker_fate_seam.py``) AST-walks ``src/`` and fails if a
    ``.get("dead_worker_failure_kind")`` call or a
    ``[...]["dead_worker_failure_kind"]`` read appears outside ``state.py``
    (the field's owner) and this module. Writes are confined the same way
    (:func:`persist_failure`).
    """

    kind: str | None
    is_throttle: bool
    classified_at: datetime | None = None

    def as_evidence(self) -> FailureEvidence | None:
        """The persisted classification as ``FateEvidence.failure`` (``fresh=False``).

        ``throttled_until`` is deliberately ``None``: the cooldown window lives
        in ``state["throttled_until"]`` and is never re-derived from the stamp.
        """
        if self.kind is None:
            return None
        return FailureEvidence(kind=self.kind, throttled_until=None, fresh=False)


def persisted_failure(entry: Mapping[str, Any]) -> PersistedFailure:
    """Read ``dead_worker_failure_kind`` off a state entry (rule 6, read side).

    Never raises: a missing or malformed entry yields ``PersistedFailure(None,
    False)``, the same as no persisted classification existing at all.
    """
    if not isinstance(entry, Mapping):
        return PersistedFailure(kind=None, is_throttle=False)
    kind = entry.get("dead_worker_failure_kind")
    return PersistedFailure(
        kind=kind,
        is_throttle=is_provider_throttle_failure(kind),
        classified_at=parse_iso_timestamp(entry.get("dead_worker_failure_classified_at")),
    )


def throttle_failure(fate: WorkerFate | None) -> FailureEvidence | None:
    """The provider-throttle failure a fate carries, else ``None``.

    Rule 6, read-through-fate: a dead worker's throttle classification reaches
    consumers as part of the fate (``Throttled``, or ``Stranded``/``Crashed``
    when a higher-precedence row decided first but still carried the failure),
    so the zero-artifact guard (#1993) reads it off the fate instead of
    re-reading the raw stamp.
    """
    if isinstance(fate, Throttled | Stranded | Crashed):
        failure = fate.failure
        if failure is not None and is_provider_throttle_failure(failure.kind):
            return failure
    return None
