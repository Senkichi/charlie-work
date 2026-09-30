"""Dead-worker sweep: decision module and apply shell (dormant until consumers are wired).

``decide`` is the pure decision table; ``run_orphan_sweep`` is the shell that applies
its requests and commits. Nothing in the orchestrator imports this package yet.
"""

from __future__ import annotations

from .apply import run_orphan_sweep
from .decide import PhaseOrderError, decide
from .model import RepoFacts, SweepFacts, SweepPlan
from .ports import SweepPorts, ports_from_workflow

__all__ = [
    "PhaseOrderError",
    "RepoFacts",
    "SweepFacts",
    "SweepPlan",
    "SweepPorts",
    "decide",
    "ports_from_workflow",
    "run_orphan_sweep",
]
