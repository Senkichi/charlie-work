"""Doctor check reporting whether host-wide runner allocation is actually
running (issue #590).

``run_doctor`` calls ``_check_runner_allocation`` as one of the host-wide
fleet-health checks, alongside the provenance refusal streak.

Lives outside ``doctor`` on purpose: ``doctor.py`` is over the 800-line
module cap and pinned by the file-size high-water-mark ratchet
(``tests/test_file_size_ratchet.py``), which never allows an over-cap file to
grow past its recorded mark -- new code lands in a domain module instead.
Extracted for issue #1852, whose pass-runtime-cap addition would otherwise
have grown ``doctor.py`` past its mark.
"""

from __future__ import annotations

import datetime
from typing import Any

from .config import OrchestratorConfig
from .fleet_paths import fleet_dir
from .supervisor_lifecycle import read_supervisor_heartbeat
from ci_fleet.charlie_work_adapter import (
    ALLOCATION_STATE_FILENAME,
    CLI_ALLOCATION_SOURCE,
    UNATTENDED_ALLOCATION_SOURCE,
    load_allocation_stamp,
)


def _allocation_writer_label(source: str | None) -> str:
    """Describe which path wrote an allocation state file, for probe output.

    An unrecognised value is echoed rather than collapsed into "unknown": if a
    future writer forgets to extend ``AllocationSource``, the probe should name
    what it actually found instead of hiding it.
    """
    if source is None:
        return "writer unrecorded — file predates provenance tracking"
    if source == UNATTENDED_ALLOCATION_SOURCE:
        return "unattended fleet pass"
    if source == CLI_ALLOCATION_SOURCE:
        return "a manual `charlie runners allocate`"
    return f"an unrecognised writer {source!r}"


def _positive_seconds(value: Any) -> int | None:
    """Coerce a seconds field (JSON or config) to a positive ``int``, else ``None``."""
    try:
        parsed = int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return parsed if parsed is not None and parsed > 0 else None


def _allocation_pass_runtime_cap(
    config: OrchestratorConfig, heartbeat: dict[str, Any] | None
) -> tuple[int, str]:
    """Resolve the per-pass runtime cap feeding the allocation staleness bound.

    ``runner-allocation.json`` is rewritten only in the pass prologue, so a
    single pass holds the stamp unrewritten for up to
    ``max_pass_runtime_seconds`` — the cap must join the three-interval bound
    or every healthy pass longer than it fires a false "not running
    unattended" (issue #1852).

    The cap's primary source is the supervisor heartbeat's
    ``max_pass_runtime_seconds`` — the running daemon's own record of the
    bound it is operating under — falling back to
    ``config.supervisor.max_pass_runtime_seconds`` when the file is absent,
    unreadable, or predates the field (the same "written before the field
    existed" shape as the recorded-interval fallback below), and to ``0``
    when neither provides a usable value — e.g. a config object built by
    code that predates the knob — which collapses the bound to the pre-#1852
    ``3 × interval``.

    Returns ``(cap_seconds, source_label)``; the label names where the cap
    came from so the warning detail can state the bound it actually used.
    """
    cap = _positive_seconds((heartbeat or {}).get("max_pass_runtime_seconds"))
    if cap is not None:
        return cap, "supervisor-heartbeat.json"
    cap = _positive_seconds(getattr(config.supervisor, "max_pass_runtime_seconds", None))
    if cap is not None:
        return cap, "config supervisor.max_pass_runtime_seconds"
    return 0, "none"


def _check_runner_allocation(
    add: Any,
    config: OrchestratorConfig,
    fleet_dir_override: str | None = None,
    *,
    now: datetime.datetime | None = None,
) -> None:
    """Report whether host-wide runner allocation is actually running (issue #590).

    Every way the allocation prologue can decline to act is silent by nature: a
    false ``enabled`` flag, a config object built by code that predates the
    section, or a registry with no reachable repo root all make it return without
    doing anything — and a converged host looks exactly like one where allocation
    never ran at all. Logs cannot settle it either: the daemon's stderr has proven
    lossy in practice, so the absence of a log line is not evidence.

    ``run_allocation_pass`` rewrites ``runner-allocation.json`` on every non-dry
    pass even when no slot moves, which makes that file's ``updated_at`` the only
    positive evidence that a pass happened.

    Age alone is not enough, though. ``charlie runners allocate`` writes the same
    host-wide file, and CLAUDE.md *requires* post-reboot procedures to call exactly
    that — so an operator's manual run would otherwise make this probe read healthy
    for three intervals, during the very window someone is diagnosing #590. Each
    pass records which path wrote it and only the unattended one is accepted as
    evidence here; a manual write is reported as "cannot confirm" rather than
    "fine", because the file keeps just the latest write.

    Two things the probe used to *guess* are now read from the file (issue #606):

    * **The driving interval.** The staleness bound was
      ``config.supervisor.full_pass_interval_seconds * 3``, resolved through the
      probe's own config load — a different call from the one the daemon used. The
      pass now records the interval it was driven at, and the bound is computed
      from that recorded value (falling back to config only for a file written
      before the interval was recorded).
    * **The skip reason.** A pass that declines to act (no runners under
      ``managed_root``, an unresolvable root) now records *why*. The probe reports
      the recorded reason instead of asserting "not running unattended (#590)" for
      every stale-or-absent file — "the daemon never reached allocation" and "the
      daemon ran it and found no runners" are different problems with different
      fixes.

    The staleness bound is three intervals *plus* the pass runtime cap
    (``max_pass_runtime_seconds``, issue #1852): the stamp is rewritten only in
    the pass prologue, so a single healthy pass holds it unrewritten for the
    whole pass — measured p90 ~20 min, max ~54 min (see
    ``scripts/heartbeat_check.py``) — and a bare three-interval bound (900 s)
    false-flagged a mid-pass stamp at 1,492 s and 2,132 s on 2026-09-23. The
    cap is read from ``supervisor-heartbeat.json`` (the running daemon's own
    record of the bound it operates under), falling back to this probe's config
    when the file is absent or lacks the field, and to the bare three-interval
    bound when neither records a cap.

    ``now`` is the injectable clock used for the age computation below (issue
    #828): defaults to ``datetime.datetime.now(datetime.timezone.utc)`` when
    not supplied, so production behavior is byte-identical. ``run_doctor``
    samples one ``now`` per doctor run and passes it here and to
    ``_probe_api_worker``.
    """
    allocation = getattr(config, "runner_allocation", None)
    if allocation is None or not allocation.enabled:
        return

    state_dir = fleet_dir(override=fleet_dir_override)
    budget = allocation.max_running_runners

    stamp = load_allocation_stamp(state_dir)
    # Measure staleness against the interval the pass was *actually* driven at,
    # not the one this probe re-resolves — a per-repo layer that sets the
    # interval would otherwise make the probe measure against a cadence the
    # daemon is not running at (issue #606). Fall back to config only for a file
    # written before the interval was recorded.
    recorded_interval = stamp.full_pass_interval_seconds if stamp is not None else None
    interval = max(recorded_interval or config.supervisor.full_pass_interval_seconds, 1)

    if stamp is None:
        add(
            "runner allocation",
            False,
            f"enabled (budget {budget}) but has never run: "
            f"{state_dir / ALLOCATION_STATE_FILENAME} absent, "
            f"expected a pass every {interval}s",
            severity="warning",
        )
        return

    if stamp.updated_at is None:
        add(
            "runner allocation",
            False,
            f"enabled but {ALLOCATION_STATE_FILENAME} has no readable updated_at stamp",
            severity="warning",
        )
        return

    # Clock skew or a hand-edited stamp can date the write in the future. A
    # negative age is not freshness evidence, so clamp it instead of reporting
    # "last pass -42s ago" as healthy.
    resolved_now = now if now is not None else datetime.datetime.now(datetime.timezone.utc)
    age = max(0, int((resolved_now - stamp.updated_at).total_seconds()))
    writer = _allocation_writer_label(stamp.source)

    # Three intervals alone are not the bound: the stamp is rewritten only in
    # the pass prologue, so one pass can legitimately hold it unrewritten for
    # up to max_pass_runtime_seconds — a mid-pass stamp is in-flight evidence,
    # not a gap (issue #1852). The cap is read from the supervisor heartbeat
    # (the running daemon's own record), falling back to config and then to
    # the bare three-interval bound. One missed pass is normal jitter (a pass
    # can run long); a gap outliving a whole pass plus three intervals is not.
    cap_seconds, cap_source = _allocation_pass_runtime_cap(
        config, read_supervisor_heartbeat(fleet_dir_override)
    )
    stale_after = cap_seconds + interval * 3
    if cap_seconds:
        bound_detail = (
            f"{stale_after}s staleness bound "
            f"({cap_seconds}s pass-runtime cap from {cap_source} + 3 × {interval}s interval)"
        )
    else:
        bound_detail = (
            f"{stale_after}s staleness bound "
            f"(3 × {interval}s interval, no pass-runtime cap recorded)"
        )

    # A recorded skip reason is the pass saying "I ran, and here is why I did not
    # act." Report that instead of guessing #590: a fresh unattended skip is the
    # daemon reaching allocation and declining — a named, different problem — not
    # the "never reached it" shape #590 describes. A *stale* skip is both: the
    # daemon last found <reason> and has not been back, so the #590 reading joins
    # the recorded reason rather than replacing it.
    #
    # A skip written by a non-unattended source (a manual `charlie runners
    # allocate`, or a file predating provenance) still cannot confirm the daemon
    # is rebalancing — the writer overwrites the same host-wide file, so its
    # skip reason is not evidence the daemon reached allocation. That clause
    # joins the recorded reason the same way staleness does, so a fresh manual
    # skip names *why* it declined *and* flags that it cannot speak for the
    # daemon (issue #590). When both apply (a stale manual skip) the clauses are
    # joined so #590 is cited once rather than duplicated.
    if stamp.skip_reason is not None:
        detail = (
            f"enabled (budget {budget}) but the last pass ({writer}) {age}s ago "
            f"declined to act: {stamp.skip_reason}"
        )
        clauses: list[str] = []
        if stamp.source != UNATTENDED_ALLOCATION_SOURCE:
            clauses.append(
                "this overwrites the same file the unattended pass uses, "
                "so it cannot confirm the daemon is rebalancing"
            )
        if age > stale_after:
            clauses.append(f"over the {bound_detail}, allocation is not running unattended")
        if clauses:
            detail += " — " + "; ".join(clauses) + " (issue #590)"
        add("runner allocation", False, detail, severity="warning")
        return

    if age > stale_after:
        add(
            "runner allocation",
            False,
            f"enabled (budget {budget}) but the last pass ({writer}) was {age}s ago, "
            f"over the {bound_detail} — allocation is configured but "
            f"is not running unattended (issue #590)",
            severity="warning",
        )
        return

    if stamp.source != UNATTENDED_ALLOCATION_SOURCE:
        add(
            "runner allocation",
            False,
            f"enabled (budget {budget}) but the most recent pass {age}s ago was "
            f"{writer}, which overwrites the same file the unattended pass uses — "
            f"this cannot confirm the daemon is rebalancing (issue #590)",
            severity="warning",
        )
        return

    add(
        "runner allocation",
        True,
        f"last unattended pass {age}s ago, budget {budget}",
    )
