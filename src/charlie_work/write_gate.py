"""Additive dry-run write gate for ``OrchestratorApp`` (issue #1264, W6 PR1).

``WriteGate`` is a per-instance wrapper around the six write primitives an
``OrchestratorApp`` method can reach: ``state.save_state``,
``state.append_event``, the ``_record_event``-shaped forwarding wrapper
(``record_event`` here), ``instrumentation.log_event``, ``labels.transition``,
and (issue #1264, W6 PR3, R6a) ``process_utils.kill_orphan_pid`` via
``kill_process``. Each gate method takes the same arguments as the
primitive it wraps (minus the ``state_path``/``repo`` binding, which the
gate auto-supplies from its own fields, mirroring how
``OrchestratorApp._record_event`` auto-binds those today) and adds exactly
one behavior: under ``dry_run=True`` it performs **zero writes, zero process
kills, and zero event emissions** and returns the natural "nothing happened"
value for its shape (the input state dict unchanged for the state-threading
methods, ``None`` for ``log_event`` and ``kill_process`` alike -- the raw
``kill_orphan_pid`` primitive itself always returns ``None``, so this is not
a synthetic dry-run-only value but the same value a real call already
produces -- and the library's own
``TransitionResult(TransitionOutcome.NOTHING_CHANGED, [], [])`` for
``transition``).

This is a strict invariant, not an optimization: no method ever emits an
event and then discards it, and no method ever writes a
``dry_run_suppressed_write``-shaped marker event. "No event at all under
dry-run" is the binding decision (#1264 comment 1, item C1.2) this module
exists to preserve — a caller migrated onto ``WriteGate`` must observe
*exactly* the same events.db/state.json footprint under dry-run as a caller
that never ran at all.

``WriteGate`` is constructed once per ``OrchestratorApp`` instance
(``self.write_gate = WriteGate(...)`` in ``OrchestratorApp.__init__``,
mirroring the existing per-instance ``self.dry_run``/``self._layout``
caching convention) and never shared or reconstructed mid-lifetime. This
module is purely additive: it changes no existing signature and converts no
existing call site. Unmigrated code keeps calling the raw primitives exactly
as before; migrated code (PR2 onward) calls through ``self.write_gate.*`` or
threads an explicit ``write_gate: WriteGate`` parameter through
``require_write_gate()`` for functions with more than one caller. See
``require_write_gate`` below for the missing-gate contract: a caller that
forgets to pass a real ``WriteGate`` gets a loud ``TypeError``, never a
silent write-allowed default.

Deliberately does not import from ``workflow.py`` — that would create an
import cycle, since ``workflow.py`` is this module's only production
importer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import LabelConfig
from .github import GitHubLike
from .instrumentation import log_event
from .labels import TransitionOutcome, TransitionResult, transition
from .process_utils import kill_orphan_pid
from .state import append_event, save_state


@dataclass(frozen=True)
class WriteGate:
    """Per-instance dry-run gate around the six write primitives.

    ``state_path`` and ``repo`` are the same auto-bound values
    ``OrchestratorApp._record_event`` already supplies on every call
    (``self.paths.state_file`` / ``self.repo_root.name``) — carrying them
    here means a migrated call site never repeats that binding.
    """

    dry_run: bool
    state_path: Path
    repo: str

    def save_state(self, data: dict[str, Any]) -> dict[str, Any]:
        """Gate ``state.save_state``. Dry-run: return ``data`` unchanged."""
        if self.dry_run:
            return data
        return save_state(self.state_path, data)

    def append_event(
        self,
        data: dict[str, Any],
        kind: str,
        payload: dict[str, Any],
        max_size: int | None = None,
        *,
        level: str | None = None,
    ) -> dict[str, Any]:
        """Gate ``state.append_event``. Dry-run: return ``data`` unchanged."""
        if self.dry_run:
            return data
        return append_event(
            data,
            kind,
            payload,
            max_size,
            state_path=self.state_path,
            repo=self.repo,
            level=level,
        )

    def record_event(
        self,
        state: dict[str, Any],
        kind: str,
        payload: dict[str, Any],
        *,
        level: str | None = None,
    ) -> dict[str, Any]:
        """Gate the ``OrchestratorApp._record_event`` shape.

        Same forwarding body as ``_record_event`` itself (append_event with
        ``state_path``/``repo`` auto-bound) — this is the gated equivalent
        migrated ``OrchestratorApp`` methods call instead. Dry-run: return
        ``state`` unchanged.
        """
        if self.dry_run:
            return state
        return append_event(
            state,
            kind,
            payload,
            state_path=self.state_path,
            repo=self.repo,
            level=level,
        )

    def log_event(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        correlation_id: str | None = None,
        level: str | None = None,
    ) -> None:
        """Gate ``instrumentation.log_event``. Dry-run: no-op, returns ``None``."""
        if self.dry_run:
            return None
        return log_event(
            self.state_path,
            kind,
            payload,
            repo=self.repo,
            correlation_id=correlation_id,
            level=level,
        )

    def transition(
        self,
        gh: GitHubLike,
        labels: LabelConfig,
        issue_number: int,
        event: str,
    ) -> TransitionResult:
        """Gate ``labels.transition``, in addition to (never instead of) the
        sink-level ``_is_mutating`` gate already enforced in ``github.py``.

        Dry-run: return the library's own no-op value,
        ``TransitionResult(TransitionOutcome.NOTHING_CHANGED, [], [])`` —
        reused rather than inventing a synthetic dry-run-only variant, so a
        caller pattern-matching on ``.outcome`` always sees a real,
        already-meaningful value.
        """
        if self.dry_run:
            return TransitionResult(TransitionOutcome.NOTHING_CHANGED, [], [])
        return transition(gh, labels, issue_number, event)

    def kill_process(self, pid: int) -> None:
        """Gate ``process_utils.kill_orphan_pid``. Dry-run: no kill, returns ``None``.

        Issue #1264 (W6 PR3, R6a): the primitive was hoisted from
        ``workflow.py`` to ``process_utils.py`` so this module can wrap it
        without importing ``workflow.py`` (see the module docstring on why
        that import direction is forbidden). ``kill_orphan_pid`` never
        raises and always returns ``None`` regardless of outcome (best-
        effort kill), so the dry-run short-circuit below returns exactly
        the same value a real call already would -- not a synthetic
        dry-run-only sentinel.
        """
        if self.dry_run:
            return None
        return kill_orphan_pid(pid)


def require_write_gate(write_gate: object) -> "WriteGate":
    """Fail loudly if a migrated call site's gate is missing or wrong-typed.

    Never default-allow: Python does not enforce type hints at runtime, so a
    parameter typed ``write_gate: WriteGate`` with no default can still
    receive ``None`` or the wrong object if a future edit adds a careless
    default. This is the single point every WriteGate-migrated function that
    takes ``write_gate`` as an explicit parameter (mirroring
    ``_reconcile_locked(dry_run=...)``'s explicit-threading convention)
    calls at its own top, instead of re-implementing this check at every one
    of the eventual migrated sites.
    """
    if not isinstance(write_gate, WriteGate):
        raise TypeError(
            f"write_gate is required and must be a WriteGate instance; "
            f"got {write_gate!r}. A missing gate must never be treated as "
            f"an implicit write-allowed default."
        )
    return write_gate
