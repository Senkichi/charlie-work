"""Monotone composition of a legacy markdown guard with its scan-based successor.

Three rounds of point fixes to the verdict-fence and example-secret guards kept
surfacing new fail-open edges, because two different fence parsers (origin/main's
hand-rolled regex/line model and ``markdown_fence.scan``) never agree on every
input. Instead of reconciling the parsers, each guard now computes BOTH answers
and combines them with a rule that can only ever move toward the restrictive
side, so the composed guard is never less strict than origin/main's, whatever
the inputs:

* verdict extraction -> the MORE SEVERE decision wins
  (``blocked`` > ``request_changes`` > no verdict > ``approved``; a missing
  verdict is deliberately stricter than an approval, so an approval survives
  only when BOTH parsers found one). Ties resolve to the legacy result, i.e.
  exactly what origin/main returned.
* example-secret masking -> a character is exempt from secret scanning only if
  BOTH masks exempt it (intersection): the composed mask can only shrink.

When the two sides disagree, one ``markdown_guard_disagreement`` event is
emitted, which is what the soak that retires the legacy paths measures. The
guards stay pure: emission goes through an optional ``on_disagreement``
callback whose default logs to the state path bound by the orchestrator.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

logger = logging.getLogger(__name__)

DISAGREEMENT_KIND = "markdown_guard_disagreement"

GUARD_VERDICT = "verdict"
GUARD_RESCUE_VERDICT = "rescue_verdict"
GUARD_OUTBOUND_MASK = "outbound_mask"

NO_VERDICT = "none"

# Higher = more severe = more restrictive. ``NO_VERDICT`` outranks ``approved``
# so an approval is only ever returned when both parsers produced one.
_SEVERITY: dict[str, int] = {"approved": 0, NO_VERDICT: 1, "request_changes": 2, "blocked": 3}
# An unrecognised decision label is treated as maximally severe (fail closed).
_UNKNOWN_SEVERITY = max(_SEVERITY.values()) + 1

_T = TypeVar("_T")

# Half-open ``[start, end)`` character ranges, sorted and non-overlapping.
Ranges = Sequence[tuple[int, int]]


@dataclass(frozen=True)
class Disagreement:
    """One legacy-vs-new divergence, as reported to ``on_disagreement``."""

    guard: str
    legacy: str
    new: str
    chosen: str


DisagreementCallback = Callable[[Disagreement], None]

_bound_state_path: Path | None = None


def bind_state_path(state_path: Path | None) -> None:
    """Bind (or with ``None`` unbind) the ``state.json`` the default emitter writes to.

    Called once by ``OrchestratorApp.__init__``; the guards themselves have no
    state path in scope (they are pure functions used from many layers).
    """
    global _bound_state_path
    _bound_state_path = state_path


def emit_disagreement(disagreement: Disagreement) -> None:
    """Default ``on_disagreement``: one ``markdown_guard_disagreement`` event.

    Best effort by contract -- a telemetry failure must never change a guard's
    verdict. Without a bound state path (a script or test calling the guard
    directly) the divergence is logged through ``logging`` only.
    """
    logger.warning(
        "%s: guard=%s legacy=%s new=%s chosen=%s",
        DISAGREEMENT_KIND,
        disagreement.guard,
        disagreement.legacy,
        disagreement.new,
        disagreement.chosen,
    )
    state_path = _bound_state_path
    if state_path is None:
        return
    try:
        from .instrumentation import log_event

        # write-gate-exempt(issue=1505): pure-guard observability; log_event is best-effort and never raises, no WriteGate is in scope here.
        log_event(  # event-consumer: audit-only -- read by the operator's soak query (events.db kind=markdown_guard_disagreement) that decides when the legacy paths are deleted; no in-repo consumer by design
            state_path,
            DISAGREEMENT_KIND,
            {
                "guard": disagreement.guard,
                "legacy": disagreement.legacy,
                "new": disagreement.new,
                "chosen": disagreement.chosen,
            },
        )
    except Exception:  # noqa: BLE001 -- telemetry must not alter a guard's result
        logger.warning("%s emission failed", DISAGREEMENT_KIND, exc_info=True)


def _report(disagreement: Disagreement, on_disagreement: DisagreementCallback | None) -> None:
    try:
        (on_disagreement or emit_disagreement)(disagreement)
    except Exception:  # noqa: BLE001 -- a broken callback must not alter a guard's result
        logger.warning("%s callback failed", DISAGREEMENT_KIND, exc_info=True)


def decision_severity(decision: str) -> int:
    """Severity rank of a decision label (``NO_VERDICT`` for a missing one)."""
    return _SEVERITY.get(decision, _UNKNOWN_SEVERITY)


def choose_more_severe(
    legacy: _T | None,
    new: _T | None,
    *,
    guard: str,
    decision_of: Callable[[_T], str],
    on_disagreement: DisagreementCallback | None = None,
) -> _T | None:
    """Return the more severe of the legacy and scan-based verdicts.

    ``None`` means "no verdict". Ties (equal decision labels, whatever the
    payloads) return ``legacy``, so agreement is byte-for-byte origin/main's
    behaviour. A differing decision label emits one disagreement event.
    """
    legacy_label = NO_VERDICT if legacy is None else decision_of(legacy)
    new_label = NO_VERDICT if new is None else decision_of(new)
    if legacy_label == new_label:
        return legacy
    if decision_severity(new_label) > decision_severity(legacy_label):
        chosen, chosen_label = new, new_label
    else:
        chosen, chosen_label = legacy, legacy_label
    _report(Disagreement(guard, legacy_label, new_label, chosen_label), on_disagreement)
    return chosen


def intersect_ranges(
    legacy: Ranges,
    new: Ranges,
    *,
    guard: str = GUARD_OUTBOUND_MASK,
    on_disagreement: DisagreementCallback | None = None,
) -> list[tuple[int, int]]:
    """Character ranges masked by BOTH ``legacy`` and ``new`` (both sorted, disjoint).

    A differing pair emits one disagreement event whose fields render each
    mask as compact ``start-end`` runs.
    """
    out: list[tuple[int, int]] = []
    i = j = 0
    while i < len(legacy) and j < len(new):
        start = max(legacy[i][0], new[j][0])
        end = min(legacy[i][1], new[j][1])
        if start < end:
            out.append((start, end))
        if legacy[i][1] < new[j][1]:
            i += 1
        else:
            j += 1
    if list(legacy) != list(new):
        _report(
            Disagreement(guard, _render(legacy), _render(new), _render(out)),
            on_disagreement,
        )
    return out


def _render(ranges: Ranges, limit: int = 20) -> str:
    """Compact, bounded rendering of ``ranges`` for an event payload."""
    shown = ",".join(f"{start}-{end}" for start, end in ranges[:limit])
    extra = len(ranges) - limit
    return shown + (f",+{extra}" if extra > 0 else "") if ranges else "-"
