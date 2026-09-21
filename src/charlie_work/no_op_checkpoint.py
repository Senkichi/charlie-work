"""No-op rework cap: content-addressed checkpoint and paired death count.

Extracted from ``dispatch_selection.py`` (issue #1784) to keep that module
under the repo's file-size guideline: these four pure helpers are the
no-op-cap-specific half of the windowed-redispatch-accounting family,
distinct from the general windowed readers (``_windowed_redispatch_at``,
``_windowed_worker_death_at``) and the death-crediting writer
(``_credit_worker_death``), which stay in ``dispatch_selection.py`` as the
pre-existing, widely-referenced single points of enforcement they already
were. ``dispatch_selection.py`` imports this module's four symbols; nothing
here imports back, so there is no cycle.

Background (job-cannon #1320 / issue #1784): the no-op rework cap used to
sum ``redispatch_at``/``worker_death_at`` over a rolling time window with no
memory of an intervening review that already confirmed genuine progress,
and no way to tell a content-blind auto-reject from a reviewer actually
reading the diff. ``_no_op_checkpoint``/``_no_op_window_start`` give
``record_review`` a way to narrow the window read-time-only (never
destroying the raw history other caps depend on) once a patch-id genuinely
advances under a trusted verdict provenance. ``_paired_death_count`` closes
a related gap where the death-loop cap could fire before as many real
redispatches occurred as its own threshold.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any


def _no_op_checkpoint(entry: dict[str, Any]) -> datetime | None:
    """Parse ``entry["no_op_checkpoint_at"]``, if present and well-formed.

    Set by ``record_review`` (issue #1784) when a request_changes verdict
    lands against a genuinely advanced, reviewer-confirmed patch-id: it
    marks the moment after which ``redispatch_at``/``worker_death_at``
    describe the issue's CURRENT stall state, not whether the issue ever
    stalled at all. The raw arrays are never truncated to record this --
    they are shared inputs to several independent caps (the redispatch cap,
    the worker-death-loop cap, the launch-time no-op cap), not an
    issue-level no-op counter ``record_review`` alone owns -- so the
    checkpoint only narrows the window ``_windowed_redispatch_at`` and
    ``_windowed_worker_death_at`` apply at read time.
    """
    raw = entry.get("no_op_checkpoint_at")
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _no_op_window_start(entry: dict[str, Any], *, window_minutes: int) -> datetime:
    """Return the effective window start for redispatch/death-at reads.

    The later of the ordinary rolling window and any no-op checkpoint on
    ``entry`` -- never earlier: a checkpoint only narrows the window, it
    cannot resurrect timestamps the ordinary rolling window already
    dropped.
    """
    window_start = datetime.now(UTC) - timedelta(minutes=window_minutes)
    checkpoint = _no_op_checkpoint(entry)
    if checkpoint is not None and checkpoint > window_start:
        return checkpoint
    return window_start


def _normalized_timestamps(raw: Any) -> list[str]:
    """Return ``raw`` filtered to well-formed ISO-8601 timestamp strings.

    Type/parse safety only — no time-window filtering. A shared building
    block for callers (``_credit_worker_death``) that must persist the
    *raw* history: only the windowed readers (``_windowed_redispatch_at``,
    ``_windowed_worker_death_at``) apply a rolling window, and that is a
    read-time concern (issue #1784 finding 4) — truncating at write time
    means a later ``watchdog.redispatch_window_minutes`` increase can never
    see timestamps an earlier, narrower window already discarded on write.
    """
    if not isinstance(raw, list):
        return []
    result: list[str] = []
    for t in raw:
        if not isinstance(t, str):
            continue
        try:
            datetime.fromisoformat(t.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        result.append(t)
    return result


def _paired_death_count(
    *,
    redispatch_at: list[str],
    worker_death_at: list[str],
) -> int:
    """Return the death count a death-loop escalation cap may safely use.

    ``worker_death_at`` can outgrow ``redispatch_at`` when a death is
    credited for a dispatch that was never itself counted as a redispatch
    (issue #1784 finding 3): the orphan sweep's "dead worker, PR still
    unreviewed" branch (``orphaned_worker_advanced_to_pr_open`` in
    ``workflow.py``) credits ``worker_death_at`` for any dead worker in
    that lane, including the ORIGINAL implementer dispatch -- which, unlike
    a rework redispatch, never stamps ``redispatch_at`` (see
    ``_credit_worker_death``'s docstring; that asymmetry is intentional and
    must not be "fixed" by stamping a phantom redispatch there instead).
    Left unpaired, a death-loop escalation (``dead_worker_reap.py``,
    ``state_dispatch_rework.py``) can fire before the issue has actually
    been redispatched that many times. Clamping to the smaller of the two
    counts keeps every death-loop cap consistent with the number of real
    redispatch attempts.
    """
    return min(len(worker_death_at), len(redispatch_at))
