"""Shared helpers for the github test modules.

Hoisted verbatim out of ``tests/test_github.py`` (issue #1572, Track 1
shoulder) when that module was split into seam-named siblings -- the
``tests/_*.py`` hoisted-fixture convention is the sanctioned import target
for shared test helpers (see ``tests/test_zero_cross_test_import_guard.py``).
"""

from __future__ import annotations

from pathlib import Path

_FIXTURES = Path(__file__).parent / "fixtures"


def _read_fixture(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")
