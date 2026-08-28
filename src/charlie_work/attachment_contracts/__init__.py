"""Attachment-Point Contracts: derived god-object fitness function.

Gates on bound-member count per derived attachment point, ratcheted against the
repo's own archetype distribution. Never reads a line count.
"""

from charlie_work.attachment_contracts.model import (
    AttachmentPoint,
    BaselineEntry,
    Bump,
    Finding,
    Kind,
    Redirect,
    ScaffoldPlan,
    ScanResult,
    SaturationVerdict,
    Severity,
)

__all__ = [
    "AttachmentPoint",
    "BaselineEntry",
    "Bump",
    "Finding",
    "Kind",
    "Redirect",
    "ScaffoldPlan",
    "ScanResult",
    "SaturationVerdict",
    "Severity",
]
