"""Merge path: pure decision stages for ``merge_ready`` (dormant until consumers are wired).

``decide_*`` are the pure decision tables over frozen facts; the gather, apply
and render shells arrive in later steps. Import order matters for the dormancy
gate: ``workflow`` imports ``merge_path.rules``, which makes this package
reachable and, through it, the decision module.
"""

from __future__ import annotations

from .decide import (
    decide_accounting,
    decide_admission,
    decide_branch,
    decide_merge,
    decide_readiness,
)
from .model import (
    Accounting,
    AccountingFacts,
    Admission,
    AdmissionFacts,
    BranchFacts,
    BranchGate,
    BranchStop,
    EffectResults,
    EventSpec,
    FactNotGathered,
    GateInputs,
    Hold,
    HoldFacts,
    MergePathConfig,
    MergePlan,
    PersistedPr,
    PlanKind,
    Readiness,
    ReadinessFacts,
    RevertStatus,
    StageKind,
    SyncOutcome,
    UNAVAILABLE,
    Unavailable,
    VerdictFact,
)

__all__ = [
    "Accounting",
    "AccountingFacts",
    "Admission",
    "AdmissionFacts",
    "BranchFacts",
    "BranchGate",
    "BranchStop",
    "EffectResults",
    "EventSpec",
    "FactNotGathered",
    "GateInputs",
    "Hold",
    "HoldFacts",
    "MergePathConfig",
    "MergePlan",
    "PersistedPr",
    "PlanKind",
    "Readiness",
    "ReadinessFacts",
    "RevertStatus",
    "StageKind",
    "SyncOutcome",
    "UNAVAILABLE",
    "Unavailable",
    "VerdictFact",
    "decide_accounting",
    "decide_admission",
    "decide_branch",
    "decide_merge",
    "decide_readiness",
]
