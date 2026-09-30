"""Every ``popen_worker`` launch site states its CPU priority, and only the
merge-gate suite runner runs at NORMAL (agent sessions are BELOW_NORMAL).

Derived from the source tree, not a hardcoded site list: a new launch site is
checked automatically, and one that omits ``priority=`` fails here even if its
code path is never exercised by another test.
"""

from __future__ import annotations

import ast
from pathlib import Path

import charlie_work

SRC = Path(charlie_work.__file__).parent
NORMAL_PRIORITY_MODULES = {"local_suite_runner.py"}


def _launch_sites() -> list[tuple[str, int, str | None]]:
    sites = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name != "popen_worker":
                continue
            value = next((kw.value for kw in node.keywords if kw.arg == "priority"), None)
            label = value.attr if isinstance(value, ast.Attribute) else None
            sites.append((path.relative_to(SRC).as_posix(), node.lineno, label))
    return sites


def test_launch_sites_found() -> None:
    """Positive control: the scan sees the known agent and gate launch sites."""
    files = {site[0] for site in _launch_sites()}
    assert {"claude_code.py", "devin_shell.py", "local_suite_runner.py"} <= files


def test_every_launch_site_declares_the_policy_priority() -> None:
    wrong = [
        (f, line, label)
        for f, line, label in _launch_sites()
        if label != ("NORMAL" if f in NORMAL_PRIORITY_MODULES else "BELOW_NORMAL")
    ]
    assert wrong == []
