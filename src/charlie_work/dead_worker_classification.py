"""Dead-worker failure classification for the state.json-keyed orphan sweep.

Issue #2002: the dead-session classifier
(``dead_worker_sweep.dead_sessions`` dead-session classifier)
runs BEFORE ``workflow._detect_and_handle_orphaned_workers`` in each repo
pass, but the two lanes answer "is this worker dead" with different
verdicts. The classifier requires ``worker.is_worker_confirmed_dead`` --
the Signal-1 inconclusive-probe deferral (issues #338/#755) plus the
fresh-real-activity veto (#307) -- while the orphan sweep keys on a bare
``worker_pid`` liveness check. A worker whose PID exited but whose probe
is still fresh or inconclusive is skipped by the classifier (not yet
"confirmed dead") and reached by the orphan sweep in the very same pass.
The sweep's death-credit sites then gated on
``entry["dead_worker_failure_kind"]``, a stamp only the classifier writes
-- so a rate-limited death (``Reached free model rate limit`` on a rework
worker, the issue's reproduce case) read back as unclassified and was
credited into ``worker_death_at`` anyway, feeding the ``worker_death_loop``
cap (#1971 escalated at three ``rate_limited`` deaths for exactly this
reason). By the time the classifier stamped the kind on a later pass, the
credit was already persisted, and the operator's subsequent unescalate
erased the stamp -- the attribution was unrecoverable.

Classification is scoped to deaths that could plausibly be provider
kills: the unreviewed-PR advance site (an open PR proves real work, and it
has no clean-exit guard) passes ``classify_log=False`` and only consults an
existing stamp -- see ``classify_and_credit_dead_worker``.

This module is the single point the sweep's credit sites call so the
death's own log is classified at credit time rather than relying on a
stamp another lane may not have written yet. It deliberately does NOT
reimplement classification: the adapter helpers
(``devin_shell.update_session_record_with_failure_classification`` /
``claude_code.update_worker_record_with_failure_classification``) own the
log-tail matching and write the sidecar's ``failure_kind`` themselves, so
the classifier lane sees the same resolved kind when it reaches the
sidecar on a later pass.

The sweep runs inside the caller's ``state_lock``; every effect here is a
cheap local read or an in-place mutation of the already-loaded ``entry`` /
``state`` mappings -- no network I/O, no subprocesses -- matching the
discipline ``dead_worker_sweep.decide_with_pr`` applies to
its other in-lock probes (``find_worker_terminal_status``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import worker_fate
from .dispatch_selection import _credit_worker_death
from .rework_attempt_exemption import (
    exempt_provider_throttle_rework_death,
    is_provider_throttle_rework_death,
)
from .throttle_signatures import (
    is_provider_auth_failure,
    match_quota_tail,
    match_throttle_tail,
)
from .worker import iter_workers
from .write_gate import WriteGate, require_write_gate

if TYPE_CHECKING:
    from .adapter_fate_profile import AdapterFateProfile
    from .config import OrchestratorConfig
    from .worker import WorkerView


# The terminal window the clean-exit throttle check matches (issue #2286):
# the provider's error notice is one line, but the CLI may print a short exit
# footer beneath it, so the check covers the last few non-blank lines rather
# than the final line alone -- while still never reaching mid-transcript
# prose (the #656 false-positive class).
_TERMINAL_THROTTLE_LINES = 5


def _worker_view_for_entry(
    sessions_dir: Path,
    entry: dict[str, Any],
    issue_number: int,
) -> WorkerView | None:
    """Return the sidecar view for this entry's dead worker, or None.

    Matches on ``issue_number`` plus the entry's recorded ``worker_pid``
    when present, so a stale sidecar an earlier dispatch epoch left behind
    (a different pid) cannot have its log misattributed to THIS death.
    ``worker_pid`` absent (legacy entries) accepts a lone matching sidecar.
    Zero or multiple candidates returns None -- the caller falls back to
    the pre-#2002 unclassified behavior rather than guessing which log to
    read.
    """
    worker_pid = entry.get("worker_pid")
    matches = [
        view
        for view in iter_workers(sessions_dir)
        if view.issue_number == issue_number and (worker_pid is None or view.pid == worker_pid)
    ]
    return matches[0] if len(matches) == 1 else None


def _persist_on_locked_entry(
    entry: dict[str, Any],
    state: dict[str, Any],
    issue_number: int,
    failure: worker_fate.FailureEvidence,
    *,
    adapter_kind: str,
    now: datetime,
    source: str,
    write_gate: WriteGate,
) -> None:
    """Persist ``failure`` via ``worker_fate.persist_failure`` and keep ``entry`` live.

    ``persist_failure`` is pure and returns a new state whose ``issues`` entry
    is a fresh mapping. The sweep, however, holds ``entry`` by reference and
    keeps mutating it after this call (and, in the common case, ``entry`` IS
    ``state["issues"][n]``), so the stamp is copied back onto ``entry`` --
    only the keys the primitive changed, never a wholesale overwrite that
    could revert the caller's in-flight edits -- and the caller's own object
    is re-seated in ``state["issues"]`` when it was the stored entry before.
    """
    write_gate = require_write_gate(write_gate)
    key = str(issue_number)
    issues_before = state.get("issues")
    before = issues_before.get(key) if isinstance(issues_before, dict) else None
    new_state = worker_fate.persist_failure(
        state,
        issue_number,
        failure,
        adapter_kind=adapter_kind,
        now=now,
        source=source,
        write_gate=write_gate,
    )
    state.update(new_state)
    stamped = new_state.get("issues", {}).get(key) if isinstance(new_state, dict) else None
    if not isinstance(stamped, dict):
        return
    missing = object()
    prior = before if isinstance(before, dict) else {}
    entry.update({k: v for k, v in stamped.items() if prior.get(k, missing) != v})
    if before is entry:
        state["issues"][key] = entry


def resolve_dead_worker_failure_kind(
    entry: dict[str, Any],
    sessions_dir: Path,
    issue_number: int,
    state: dict[str, Any],
    config: OrchestratorConfig,
    *,
    write_gate: WriteGate,
    now: datetime | None = None,
    classify_log: bool = True,
) -> str | None:
    """Resolve the death's ``failure_kind``, stamping the entry when new.

    First consults the epoch-scoped ``dead_worker_failure_kind`` stamp --
    when the dead-session classifier (or an earlier resolve call this pass)
    already classified this death, no log is re-read. Otherwise finds the
    dead worker's sidecar via ``_worker_view_for_entry`` and runs the same
    adapter-specific ``update_*_record_with_failure_classification`` helper
    the classifier lane uses, with ``fallback_kind=None``: this lane cannot
    fabricate ``stalled``/``unpublished_work`` -- those fallbacks belong to
    the reap lane, which inspects the worktree first; here only a
    classification the log tail itself supports (a provider throttle/auth
    signature, or a kind a previous lane already wrote into the sidecar)
    resolves.

    When the helper resolves a kind it is stamped onto the issue entry
    through ``worker_fate.persist_failure`` -- the same single write primitive
    the classifier lane uses -- and a returned ``throttled_until`` arms the
    fleet-wide provider cooldown in the same call. The caller's locked
    ``entry`` stays live (see ``_persist_on_locked_entry``); an issue with no
    entry in ``state`` is never invented. Arming here is required, not redundant: the
    helper writes ``failure_kind`` to the sidecar, so the classifier lane's
    later call returns the stamp WITHOUT re-running the tail match --
    ``throttled_until`` comes back None there and the cooldown would never
    be set if this lane skipped it.

    ``classify_log=False`` restricts resolution to the existing stamp: the
    sidecar log is never read, no sidecar ``failure_kind`` is written, and no
    cooldown is armed. Callers pass it for a death whose worker demonstrably
    produced real work (see ``classify_and_credit_dead_worker``), where a log
    tail quoting throttle markers is the #656 false-positive class.

    ``state`` is mutated in place (``persist_failure`` returns a new
    mapping; ``update`` folds its keys back into the locked dict). Returns
    the resolved kind, or None when the death stays unclassified -- the
    caller then applies the pre-#2002 behavior (a credit is still a real
    worker death, just not a provider exemption).

    ``write_gate`` (issue #2006) is the emission channel for the
    ``throttle_window_set`` audit event a resolved throttle kind emits when
    it arms the cooldown: the gate's bound ``state_path``/``repo`` dual-write
    puts the row in ``events.db`` (not only the in-memory ring) and its
    ``dry_run`` suppresses the event entirely. Required -- a missing channel
    here was the audit gap the issue was filed against.
    """
    write_gate = require_write_gate(write_gate)
    stamped = worker_fate.persisted_failure(entry).kind
    if stamped is not None:
        return stamped
    if not classify_log:
        return None
    classified = classify_dead_worker_log(entry, sessions_dir, issue_number, config, now=now)
    if classified is None:
        return None
    adapter_kind, failure = classified
    _persist_on_locked_entry(
        entry,
        state,
        issue_number,
        failure,
        adapter_kind=adapter_kind,
        now=now if now is not None else datetime.now(UTC),
        source="dead_worker_classification",
        write_gate=write_gate,
    )
    return failure.kind


def classify_dead_worker_log(
    entry: dict[str, Any],
    sessions_dir: Path,
    issue_number: int,
    config: OrchestratorConfig,
    *,
    now: datetime | None = None,
) -> tuple[str, worker_fate.FailureEvidence] | None:
    """Classify this entry's dead-worker sidecar log; return ``(adapter_kind, failure)``.

    The classification half of :func:`resolve_dead_worker_failure_kind`, with
    no ``state`` write: the adapter helper still stamps the sidecar's
    ``failure_kind`` and records the role-ledger restriction (#2086) itself, but
    the issue-entry stamp and the per-repo cooldown are left to the caller.
    The dead-worker sweep needs that split (#2274): it classifies in its
    lock-free pre phase, whose state copy is discarded, and persists the result
    once under the lock. Returns None when no single sidecar matches this
    entry's epoch, the adapter has no classifier, or the log shows no
    signature. The #656 completion guard lives inside the adapter helper
    (``terminal_record_proves_completion``), so it applies here unchanged.
    """
    view = _worker_view_for_entry(sessions_dir, entry, issue_number)
    if view is None:
        return None

    # The Adapter seam (design doc §7): the profile supplies the sidecar
    # classifier for this view's adapter, so no ``adapter_kind`` branch lives
    # here. A profile with no classifier (command/manual) or an unknown kind
    # stays unclassified, exactly as before.
    profile = worker_fate.profile_for(view.adapter_kind)
    if profile is None or profile.record_failure is None:
        return None
    failure_kind, throttled_until = profile.record_failure(
        sessions_dir,
        issue_number,
        config=config,
        now=now,
    )
    if failure_kind is None:
        return None
    return view.adapter_kind, worker_fate.FailureEvidence.from_classification(
        failure_kind, throttled_until, fresh=True
    )


def _terminal_is_throttle_error(
    terminal: str, profile: AdapterFateProfile | None, config: OrchestratorConfig
) -> bool:
    """True when the log's trailing non-blank lines carry a throttle signature.

    Covers every family ``PROVIDER_THROTTLE_EXEMPT_KINDS`` admits: provider auth
    errors only for the adapter whose profile detects account errors (api), the
    quota signature, then the generic throttle markers.
    """
    if (
        profile is not None
        and profile.account_error_detection
        and is_provider_auth_failure(terminal)
    ):
        return True
    if match_quota_tail(terminal, config.runtime.quota_error_markers):
        return True
    matched, _ = match_throttle_tail(terminal, config.runtime.throttle_error_markers)
    return matched


def classify_clean_exit_throttle(
    entry: dict[str, Any],
    sessions_dir: Path,
    issue_number: int,
    config: OrchestratorConfig,
    *,
    now: datetime | None = None,
) -> tuple[str, worker_fate.FailureEvidence] | None:
    """Classify a clean-exit dead worker as a provider throttle, terminal lines only.

    The clean-exit no-op branch's narrow classifier (issue #2286). A rework
    session that exits 0 having changed nothing is normally a real no-op
    attempt -- unless the provider killed it, and the evidence for that kill
    is that the transcript STOPS on the provider's own throttle error. Unlike
    :func:`classify_dead_worker_log` the signature must therefore anchor the
    last ``_TERMINAL_THROTTLE_LINES`` non-blank lines of the log; a marker
    anywhere earlier is completion-era prose (the #656 class) and never
    classifies here.

    The sidecar writer (``profile.record_failure``) still owns the kind
    resolution, the cooldown, the ledger restriction, and the
    ``terminal_record_proves_completion`` guard: a record proving this pid
    exited 0 WITH a worker outcome ends the check inside the helper, so the
    same #656 authority applies here unchanged.

    Returns ``(adapter_kind, FailureEvidence)`` like
    :func:`classify_dead_worker_log`, or None when the terminal lines carry no
    provider signature (or the resolved kind is not a throttle kind).
    """
    view = _worker_view_for_entry(sessions_dir, entry, issue_number)
    if view is None or not view.log_path:
        return None
    profile = worker_fate.profile_for(view.adapter_kind)
    if profile is None or profile.record_failure is None:
        return None
    log_path = Path(view.log_path)
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    tail = log_text[-2048:] if len(log_text) > 2048 else log_text
    terminal_lines = [line for line in tail.splitlines() if line.strip()][
        -_TERMINAL_THROTTLE_LINES:
    ]
    if not terminal_lines or not _terminal_is_throttle_error(
        "\n".join(terminal_lines), profile, config
    ):
        return None
    # The terminal gate already proved the signature sits at the end, so the
    # adapter helper's (kind, throttled_until) describes THIS death: the
    # emission anchor resolves to the terminal error's own timestamp, not a
    # mid-log quote.
    failure_kind, throttled_until = profile.record_failure(
        sessions_dir,
        issue_number,
        config=config,
        now=now,
    )
    if not is_provider_throttle_rework_death(failure_kind):
        return None
    return view.adapter_kind, worker_fate.FailureEvidence.from_classification(
        failure_kind, throttled_until, fresh=True
    )


def exempt_clean_exit_throttle_death(
    entry: dict[str, Any],
    sessions_dir: Path,
    issue_number: int,
    state: dict[str, Any],
    config: OrchestratorConfig,
    *,
    write_gate: WriteGate,
    dispatched_at: str | None = None,
    pr_number: int | None = None,
    now: datetime | None = None,
) -> str | None:
    """Issue #2286: exempt a clean-exit rework death that ended on a provider throttle.

    The orphan sweep's clean-exit no-op branch never classified the worker's
    log -- a real completion's prose may quote throttle markers mid-transcript
    (the #656 guard). That left one lane where a rate-limited rework session
    that exited 0 still consumed a ``no_op_rework_attempts`` slot and
    escalated a healthy PR at the cap (#2254 / PR #2264).

    Resolution consults the epoch-scoped ``dead_worker_failure_kind`` stamp
    first (the dead-session lane often classified and reaped the sidecar
    before this sweep reaches the issue), then falls back to
    :func:`classify_clean_exit_throttle`'s terminal-line check. A resolved
    throttle kind is stamped via ``worker_fate.persist_failure`` (arming the
    per-repo cooldown) and routed through the single enforcement point,
    ``rework_attempt_exemption.exempt_provider_throttle_rework_death``, which
    refunds the dead session's own ``redispatch_at`` stamp and flags the PR.

    Returns the resolved ``failure_kind`` when the death was exempted, else
    None -- the caller then falls through to the ordinary no-op route.
    """
    write_gate = require_write_gate(write_gate)
    stamped = worker_fate.persisted_failure(entry).kind
    if stamped is not None:
        failure_kind = stamped
    else:
        classified = classify_clean_exit_throttle(
            entry, sessions_dir, issue_number, config, now=now
        )
        if classified is None:
            return None
        adapter_kind, failure = classified
        _persist_on_locked_entry(
            entry,
            state,
            issue_number,
            failure,
            adapter_kind=adapter_kind,
            now=now if now is not None else datetime.now(UTC),
            source="dead_worker_classification",
            write_gate=write_gate,
        )
        failure_kind = failure.kind
    if not exempt_provider_throttle_rework_death(
        state,
        issue_number,
        entry,
        failure_kind,
        dispatched_at=dispatched_at,
        pr_number=pr_number,
        source="orphan_sweep_clean_exit",
        write_gate=write_gate,
    ):
        return None
    return failure_kind


def classify_and_credit_dead_worker(
    entry: dict[str, Any],
    sessions_dir: Path,
    issue_number: int,
    state: dict[str, Any],
    config: OrchestratorConfig,
    *,
    write_gate: WriteGate,
    at: str,
    now: datetime | None = None,
    classify_log: bool = True,
    dispatched_at: str | None = None,
    pr_number: int | None = None,
) -> str | None:
    """Classify the dead worker, then credit its death unless throttle-caused.

    The orphan sweep's three ``worker_death_at`` credit sites
    (``dead_worker_sweep.decide_with_pr``: request_changes
    restore, approved-rework restore, unreviewed-PR advance) call this once
    each instead of gating on a possibly-absent
    ``dead_worker_failure_kind`` stamp. Resolution runs lazily at the
    credit site. The request_changes and approved-rework restore sites are
    only reached after their clean-exit (``terminal_exit_code == 0``) and
    completed-outcome guards, so they log-classify (``classify_log=True``).
    The unreviewed-PR advance site has NO such guard -- it is reached with
    an open PR (real work) whether or not the worker exited cleanly -- so it
    passes ``classify_log=False`` and only consults an existing stamp: a
    session's own completion prose quoting throttle markers (the #656
    false-positive class) must not arm a fleet-wide throttle from a worker
    that produced a PR. That death is still credited, as unclassified.

    A provider-throttle kind (``worker_fate.persisted_failure(...).is_throttle`` -- the
    #1684/#1917 exemption) suppresses the credit: the death is a global
    provider condition, not a worker-quality signal, and must not count
    toward ``worker_death_at`` caps. Every other outcome credits the death
    through ``_credit_worker_death`` with ``kind=`` so the timestamp and
    its attribution are recorded together (``worker_death_failure_kinds``).

    Issue #2282: the same throttle death also goes through
    ``rework_attempt_exemption.exempt_provider_throttle_rework_death``.
    ``dispatched_at`` (the dead epoch's) and ``pr_number`` let that call refund
    the session's own ``redispatch_at`` stamp and flag the PR, so the attempt
    counts toward no cap at all.

    Returns the resolved ``failure_kind`` (None when unclassifiable) so the
    call site can record it in its event payload.
    """
    write_gate = require_write_gate(write_gate)
    failure_kind = resolve_dead_worker_failure_kind(
        entry,
        sessions_dir,
        issue_number,
        state,
        config,
        write_gate=write_gate,
        now=now,
        classify_log=classify_log,
    )
    # Rule 6 read side (worker_fate.persisted_failure): resolution above has
    # stamped ``dead_worker_failure_kind`` on ``entry`` whenever it resolved a
    # kind, so the stamp read agrees with ``failure_kind``.
    if not worker_fate.persisted_failure(entry).is_throttle:
        entry["worker_death_at"] = _credit_worker_death(entry, at=at, kind=failure_kind)
    # Issue #2282: not crediting the death is only half the exemption -- the
    # dead session's dispatch stamp would still count as a no-op. The single
    # enforcement point refunds it and flags the PR.
    exempt_provider_throttle_rework_death(
        state,
        issue_number,
        entry,
        failure_kind,
        dispatched_at=dispatched_at,
        pr_number=pr_number,
        source="orphan_sweep_credit",
        write_gate=write_gate,
    )
    return failure_kind
