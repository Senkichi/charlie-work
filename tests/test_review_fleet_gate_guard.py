"""Structural guard: every reviewer launch site goes through the fleet review cap.

Issue #2084. ``fleet.global_max_concurrent_reviews`` is only enforced where a
launch site carries ``@fleet_review_lock`` (mint/release the fleet lock handle)
and calls ``fleet_review_lock_deferral`` (realize it) and ``read_fleet_review_cap``
(clamp to the fleet budget). A new lane that looks up ``_REVIEW_LAUNCHERS``
without all three is a reviewer launch that is uncounted
against the fleet cap -- the #2039 defect shape on the review side. This scan
fails CI on it; the launch sites are derived from the live source, not listed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import charlie_work

PACKAGE_ROOT = Path(charlie_work.__file__).resolve().parent
LAUNCH_TABLE = "_REVIEW_LAUNCHERS"
# Since wave D6b, launch sites reach the table through the host port
# (``self.host.launch.launch(...)``); the table itself is read only inside
# ``host/launch.py`` (the port's Real), which is not a launch site.
PORT_CALL = "host.launch.launch"
PORT_MODULE = PACKAGE_ROOT / "host" / "launch.py"
REQUIRED_GATES = {"fleet_review_lock", "fleet_review_lock_deferral", "read_fleet_review_cap"}


def _names_in(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            names.add(child.attr)
            if _is_port_call(child):
                names.add(PORT_CALL)
    return names


def _is_port_call(node: ast.Attribute) -> bool:
    """``<x>.host.launch.launch`` -- the reviewer launch call through the port."""
    inner = node.value
    return (
        node.attr == "launch"
        and isinstance(inner, ast.Attribute)
        and inner.attr == "launch"
        and isinstance(inner.value, ast.Attribute)
        and inner.value.attr == "host"
    )


def _launch_sites() -> list[tuple[str, str, set[str]]]:
    """``(module path, function name, names referenced)`` per function using the table."""
    sites: list[tuple[str, str, set[str]]] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if path == PORT_MODULE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names = _names_in(node)
                if LAUNCH_TABLE in names or PORT_CALL in names:
                    sites.append((path.name, node.name, names))
    return sites


def test_every_reviewer_launch_site_is_fleet_gated() -> None:
    sites = _launch_sites()
    # Positive control: the scan must see both known launch sites, or it is
    # matching nothing and a bypass would pass silently.
    assert {name for _, name, _ in sites} >= {
        "dispatch_reviews",
        "_local_dispatch_reviewers",
    }, sites
    ungated = [(mod, fn) for mod, fn, names in sites if not REQUIRED_GATES <= names]
    assert not ungated, (
        f"reviewer launch site(s) {ungated} look up {LAUNCH_TABLE} without "
        f"{sorted(REQUIRED_GATES)} -- they would launch reviewers uncounted "
        "against fleet.global_max_concurrent_reviews"
    )
