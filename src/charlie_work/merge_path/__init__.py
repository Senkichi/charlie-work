"""Merge path: the stages behind ``merge_ready`` and its dry-run preview.

``decide_*`` are the pure decision tables over frozen facts; ``gather*`` read the
facts, ``apply*`` run each stage's effects through the ``WriteGate``, and
``render`` builds the result payloads. ``run_merge_ready`` (live) and
``preview_merge_ready`` (dry-run) are the two drivers; they live in
``merge_path.apply`` and ``merge_path.preview`` and are imported from there, not
re-exported here, so importing the decision module stays free of effect code.
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
