"""Dead-worker sweep: decision module and apply shell.

``decide`` is the pure decision table; ``run_orphan_sweep`` and
``run_stalled_sweep`` are the shells that apply its requests and commits. The
orchestrator reaches them through the ``workflow`` re-exports
(``_detect_and_handle_orphaned_workers`` / ``_detect_and_handle_stalled_sessions``),
and other modules import the package's helpers directly.
"""

from __future__ import annotations

from .apply import run_orphan_sweep
from .apply_stalled import run_stalled_sweep
from .dead_sessions import classify_dead_sessions
from .decide import PhaseOrderError, decide
from .model import RepoFacts, SweepFacts, SweepPlan
from .ports import SweepPorts, ports_from_workflow

__all__ = [
    "PhaseOrderError",
    "RepoFacts",
    "SweepFacts",
    "SweepPlan",
    "SweepPorts",
    "classify_dead_sessions",
    "decide",
    "ports_from_workflow",
    "run_orphan_sweep",
    "run_stalled_sweep",
]
