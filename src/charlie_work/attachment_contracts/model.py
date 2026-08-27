"""Shared data model for Attachment-Point Contracts.

Design authority: llibrary/docs/plans/2026-08-24-god-object-mitigation-DECISION.md.
The unit of measure is bound-member count per attachment point. No line count is
ever read anywhere in this package (binding operator constraint).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Kind = Literal[
    "typer_app",
    "click_group",
    "blueprint",
    "class",
    "migration_runner",
    "test_module",
]

Severity = Literal["block", "advise", "error"]


@dataclass(frozen=True)
class AttachmentPoint:
    """One derived attachment point: an object that members bind to."""

    kind: Kind
    identity: str
    file: str  # repo-relative posix path
    members: tuple[str, ...]
    is_linear_ledger: bool = False
    # Round-2 review finding #9: `class` is not one archetype -- Protocol
    # bases, Exception subclasses, empty @dataclass shells, and Fake*/Test*
    # doubles share the population with real service classes and drag the
    # Tukey fence the wrong way. Structurally-trivial classes are still
    # scanned and reported (unlike an exclude-set entry) but excluded from
    # the saturation population, the same way ledgers are (see outliers.py).
    is_structurally_trivial: bool = False

    @property
    def member_count(self) -> int:
        return len(self.members)


@dataclass(frozen=True)
class ScanResult:
    """Full-tree scan output. parse_failures are never dropped (G6)."""

    root: str
    points: tuple[AttachmentPoint, ...]
    parse_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class SaturationVerdict:
    """Outlier verdict for one point within its own archetype distribution."""

    point: AttachmentPoint
    saturated: bool
    q3: float
    iqr: float
    boundary: float
    population: int


@dataclass(frozen=True)
class Finding:
    """One actionable result from check_file / check_tree."""

    severity: Severity
    file: str
    identity: str
    message: str
    redirect: str | None = None


@dataclass(frozen=True)
class Bump:
    """A reviewed baseline advance (the escape hatch, decision doc section 4.3)."""

    to: int
    reason: str
    actor: Literal["interactive", "worker"]
    ack: str = ""


@dataclass(frozen=True)
class BaselineEntry:
    """One frozen saturated point in .attachment-budgets.json."""

    kind: Kind
    identity: str
    file: str
    member_count: int
    boundary: float
    bumps: tuple[Bump, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Redirect:
    """A named destination for a member that would grow a saturated point."""

    destination: str  # repo-relative path (existing sibling or proposed new module)
    rationale: str
    is_new_module: bool


@dataclass(frozen=True)
class AdvisoryRecord:
    """One PreToolUse advisory logged to ``.var/attachment-contracts/advisories.jsonl``
    (issue #1460's review packet reads these to compute redirects-not-taken).

    ``redirect``/``timestamp`` are optional so old records written before
    issue #1460 (which lack both fields) still parse -- ``read_advisories``
    tolerates their absence rather than treating them as malformed.
    """

    severity: Severity
    file: str
    identity: str
    message: str
    redirect: str | None = None
    timestamp: str | None = None


@dataclass(frozen=True)
class ScaffoldPlan:
    """G2 pre-wired scaffold: rendered content for the redirect destination.

    The plan is returned, never written by this package; callers decide.
    """

    path: str
    content: str
    registration_note: str
