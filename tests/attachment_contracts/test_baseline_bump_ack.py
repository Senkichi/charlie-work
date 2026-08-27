"""Tests for issue #1460: bump_ack_is_external truth table, effective_ceiling."""

from __future__ import annotations

import pytest

from charlie_work.attachment_contracts.baseline import bump_ack_is_external, effective_ceiling
from charlie_work.attachment_contracts.model import BaselineEntry, Bump


def _bump(ack: str, actor: str = "worker") -> Bump:
    return Bump(to=10, reason="growth", actor=actor, ack=ack)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "ack,expected",
    [
        ("https://example.test/issues/123", True),
        ("http://example.test/pull/1", True),
        ("#123", True),
        ("owner/repo#123", True),
        ("source:id", False),
        ("dispatch:abc123", False),
        ("handle:senkichi", False),
        ("", False),
        ("   ", False),
    ],
)
def test_bump_ack_is_external_truth_table(ack: str, expected: bool) -> None:
    assert bump_ack_is_external(_bump(ack)) is expected


def test_effective_ceiling_no_bumps_is_member_count() -> None:
    entry = BaselineEntry(
        kind="class", identity="Foo", file="src/foo.py", member_count=5, boundary=4.0
    )
    assert effective_ceiling(entry) == 5


def test_effective_ceiling_is_max_of_member_count_and_bump_to() -> None:
    entry = BaselineEntry(
        kind="class",
        identity="Foo",
        file="src/foo.py",
        member_count=5,
        boundary=4.0,
        bumps=(
            Bump(to=8, reason="r1", actor="worker", ack="#1"),
            Bump(to=6, reason="r2", actor="worker", ack="#2"),
        ),
    )
    assert effective_ceiling(entry) == 8


def test_effective_ceiling_member_count_can_exceed_bumps() -> None:
    """member_count itself wins when it's already higher than any bump.to
    (a ratchet-down scenario where stale, lower bumps linger transiently)."""
    entry = BaselineEntry(
        kind="class",
        identity="Foo",
        file="src/foo.py",
        member_count=20,
        boundary=4.0,
        bumps=(Bump(to=8, reason="r1", actor="worker", ack="#1"),),
    )
    assert effective_ceiling(entry) == 20
