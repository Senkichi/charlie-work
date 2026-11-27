"""The foreman layer: on-disk state for chartered outcome managers.

A *foreman* (CONTEXT.md) is a chartered client of the hub
(``docs/adr/0010-a-foreman-is-a-chartered-client-of-the-hub.md``): it owns a
single outcome on a single registered repo and lives entirely under
``fleet_dir()/foremen/<name>/``. This package is the *only* module family
allowed to lay those paths out, read them, or write them; the rest of the
codebase reaches it through exactly one function-local seam in
``charlie_work.gang_command`` (the ``charlie gang`` CLI). The import-graph
guard for that rule lives in ``tests/test_foreman_import_direction.py``.

Design: ``docs/design/2026-11-25-foreman-layer-design.md``. Two rules from it
shape everything below:

* **Every path resolves and containment-checks.** A foreman path is derived
  from a validated name and then run through ``safe_path`` — never from
  concatenated strings a caller can smuggle ``..`` or a junction through.
* **Failures are values, never exceptions.** A corrupt charter, a missing
  file, an escaped path: each comes back as a ``Result.error`` naming the
  field or file, so one bad foreman can never stop the others.

``state.json`` under each foreman is a *cache*, not truth — GitHub ownership
labels are truth (ADR-0010). It may be deleted and rebuilt; writes go through
the shared atomic-write helpers like every other JSON state file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

__all__ = [
    "FOREMAN_STATES",
    "Result",
    "display_action_outcome",
    "display_state",
]

T = TypeVar("T")


@dataclass(frozen=True)
class Result(Generic[T]):
    """One operation outcome: a ``value`` on success, an ``error`` message on failure.

    The foreman layer's uniform failure-is-a-value shape. ``error`` names the
    offending field or file so an operator (or a future agent turn) can act on
    it without a traceback.
    """

    value: T | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


#: The foreman lifecycle states (design §10). ``done`` is terminal: the
#: charter's ``done_when.check`` passed and the foreman wrapped up.
FOREMAN_STATES = frozenset({"active", "paused", "degraded", "faulted", "done"})

#: Themed display spelling for ``degraded`` (CONTEXT.md: the names are
#: CLI/display vocabulary only).
_DEGRADED_DISPLAY = "CRICKET"


def display_state(state: str) -> str:
    """Map a foreman state to its CLI display spelling."""
    return _DEGRADED_DISPLAY if state == "degraded" else state


#: Themed display spelling for a refused Apply outcome (design §8.3/§13:
#: display text labels a refused action ``jabroni``; the event kind and the
#: stored reason stay semantic).
_REFUSED_ACTION_DISPLAY = "jabroni"


def display_action_outcome(outcome: str) -> str:
    """Map an Apply action outcome to its CLI display spelling."""
    return _REFUSED_ACTION_DISPLAY if outcome == "refused" else outcome
