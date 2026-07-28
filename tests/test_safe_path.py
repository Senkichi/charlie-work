"""Tests for safe_path's containment helpers (GSD sunset salvage item #8).

Covers the two properties that motivated consolidating five ad hoc
``.resolve()`` + ``is_relative_to()`` call sites into one module: symlink/
junction resolution on both sides of the comparison, and correct rejection of
a `..`-escaping candidate that a purely lexical ``is_relative_to`` check would
have missed (the actual bug found in ``worktree.py``'s
``_materialize_directory`` during the consolidation).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from charlie_work.safe_path import contains, require_contained


def test_contains_true_for_direct_child(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    child = base / "child"
    child.mkdir()
    assert contains(base, child) is True


def test_contains_true_for_base_itself(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    assert contains(base, base) is True


def test_contains_false_for_sibling(tmp_path: Path) -> None:
    base = tmp_path / "base"
    sibling = tmp_path / "base-sibling"
    base.mkdir()
    sibling.mkdir()
    assert contains(base, sibling) is False


def test_contains_false_for_dotdot_escape(tmp_path: Path) -> None:
    """A `..` segment must be resolved away, not compared lexically.

    ``Path.is_relative_to`` is purely syntactic: ``base/../escaped`` shares a
    literal prefix with ``base`` even though it names a path outside it. This
    was the real bug in worktree.py's pre-refactor containment check.
    """
    base = tmp_path / "base"
    base.mkdir()
    escaped = tmp_path / "escaped"
    escaped.mkdir()
    candidate = base / ".." / "escaped"
    assert contains(base, candidate) is False


def test_contains_true_for_nonexistent_candidate_inside_base(tmp_path: Path) -> None:
    """Containment must be checkable before the candidate is created."""
    base = tmp_path / "base"
    base.mkdir()
    not_yet_created = base / "not-yet-created" / "nested"
    assert contains(base, not_yet_created) is True


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows-specific")
def test_contains_false_for_junction_escaping_base(tmp_path: Path) -> None:
    """A junction under base that points outside it must not be treated as contained.

    Uses ``_winapi.CreateJunction`` (the same primitive worktree.py's own
    ``_create_junction_or_symlink`` uses) rather than ``os.symlink``, which
    requires an elevated privilege this test process doesn't have on Windows.
    """
    import _winapi

    base = tmp_path / "base"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    junction = base / "escape-hatch"
    _winapi.CreateJunction(str(outside), str(junction))
    assert contains(base, junction) is False


def test_require_contained_returns_resolved_path_when_contained(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    child = base / "child"
    child.mkdir()
    result = require_contained(base, child, context="test")
    assert result == child.resolve()


def test_require_contained_raises_when_escaping(tmp_path: Path) -> None:
    base = tmp_path / "base"
    outside = tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    with pytest.raises(ValueError, match="escapes"):
        require_contained(base, outside, context="test-context")
