"""Shared helpers for the checks test modules.

Hoisted verbatim out of ``tests/test_checks.py`` (issue #1565, Track-1
shoulder) when that module was split into seam-named siblings -- the
``tests/_*.py`` hoisted-fixture convention is the sanctioned import
target for shared test helpers (see
``tests/test_zero_cross_test_import_guard.py``).
"""

from __future__ import annotations


REQUIRED = ("Tests passed", "Lint & Format")


def _link(run_id: int, job_id: int) -> str:
    return f"https://github.com/owner/repo/actions/runs/{run_id}/job/{job_id}"
