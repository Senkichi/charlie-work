"""Dead-worker failure classification for the state.json-keyed orphan sweep.

Issue #2002: the dead-session classifier
(``dead_worker_reap._classify_dead_sessions_and_update_throttle_state``)
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
discipline ``orphaned_worker_sweep.handle_dead_worker_with_pr`` applies to
its other in-lock probes (``find_worker_terminal_status``).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .dispatch_selection import _credit_worker_death
from .state import set_throttled_until
from .throttle_signatures import is_provider_throttle_failure
from .worker import iter_workers

if TYPE_CHECKING:
    from .config import OrchestratorConfig
    from .worker import WorkerView


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


def resolve_dead_worker_failure_kind(
    entry: dict[str, Any],
    sessions_dir: Path,
    issue_number: int,
    state: dict[str, Any],
    config: OrchestratorConfig,
    *,
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

    When the helper resolves a kind it is stamped onto
    ``entry["dead_worker_failure_kind"]`` -- the same field the classifier
    lane writes through ``record_dead_worker_failure_kind``, mutated in
    place on the caller's locked state entry -- and a returned
    ``throttled_until`` arms the fleet-wide provider cooldown via
    ``set_throttled_until``. Arming here is required, not redundant: the
    helper writes ``failure_kind`` to the sidecar, so the classifier lane's
    later call returns the stamp WITHOUT re-running the tail match --
    ``throttled_until`` comes back None there and the cooldown would never
    be set if this lane skipped it.

    ``classify_log=False`` restricts resolution to the existing stamp: the
    sidecar log is never read, no sidecar ``failure_kind`` is written, and no
    cooldown is armed. Callers pass it for a death whose worker demonstrably
    produced real work (see ``classify_and_credit_dead_worker``), where a log
    tail quoting throttle markers is the #656 false-positive class.

    ``state`` is mutated in place (``set_throttled_until`` returns a new
    mapping; ``update`` folds its keys back into the locked dict). Returns
    the resolved kind, or None when the death stays unclassified -- the
    caller then applies the pre-#2002 behavior (a credit is still a real
    worker death, just not a provider exemption).
    """
    stamped = entry.get("dead_worker_failure_kind")
    if stamped is not None:
        return stamped
    if not classify_log:
        return None
    view = _worker_view_for_entry(sessions_dir, entry, issue_number)
    if view is None:
        return None

    from .claude_code import update_worker_record_with_failure_classification
    from .devin_shell import update_session_record_with_failure_classification

    if view.adapter_kind == "devin":
        failure_kind, throttled_until = update_session_record_with_failure_classification(
            sessions_dir,
            issue_number,
            config=config,
            now=now,
        )
    elif view.adapter_kind in ("claude-code", "api"):
        failure_kind, throttled_until = update_worker_record_with_failure_classification(
            sessions_dir,
            issue_number,
            config=config,
            adapter_kind=view.adapter_kind,
            now=now,
        )
    else:
        return None
    if failure_kind is None:
        return None
    entry["dead_worker_failure_kind"] = failure_kind
    if throttled_until:
        state.update(
            set_throttled_until(
                state,
                throttled_until,
                reason=failure_kind,
                adapter_kind=view.adapter_kind,
            )
        )
    return failure_kind


def classify_and_credit_dead_worker(
    entry: dict[str, Any],
    sessions_dir: Path,
    issue_number: int,
    state: dict[str, Any],
    config: OrchestratorConfig,
    *,
    at: str,
    now: datetime | None = None,
    classify_log: bool = True,
) -> str | None:
    """Classify the dead worker, then credit its death unless throttle-caused.

    The orphan sweep's three ``worker_death_at`` credit sites
    (``orphaned_worker_sweep.handle_dead_worker_with_pr``: request_changes
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

    A provider-throttle kind (``is_provider_throttle_failure`` -- the
    #1684/#1917 exemption) suppresses the credit: the death is a global
    provider condition, not a worker-quality signal, and must not count
    toward ``worker_death_at`` caps. Every other outcome credits the death
    through ``_credit_worker_death`` with ``kind=`` so the timestamp and
    its attribution are recorded together (``worker_death_failure_kinds``).

    Returns the resolved ``failure_kind`` (None when unclassifiable) so the
    call site can record it in its event payload.
    """
    failure_kind = resolve_dead_worker_failure_kind(
        entry, sessions_dir, issue_number, state, config, now=now, classify_log=classify_log
    )
    if not is_provider_throttle_failure(failure_kind):
        entry["worker_death_at"] = _credit_worker_death(entry, at=at, kind=failure_kind)
    return failure_kind
