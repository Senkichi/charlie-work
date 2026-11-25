"""Ratchet: branches on a harness name outside the per-harness profile only go down.

Issue #3092. ``AdapterFateProfile`` (``adapter_fate_profile.py``) is where
harness-specific knowledge lives: a caller asks the profile and never names a
harness. Every ``adapter_kind == "devin"`` / ``adapter_kind in
CLAUDE_RECORD_KINDS`` branch elsewhere is one more file a new harness has to
edit (adding opencode touched 18).

This scan counts, per file under ``src/charlie_work``, the comparisons of an
``adapter_kind`` against a harness name or against ``CLAUDE_RECORD_KINDS``.
The profile module, the Adapter modules and the harness registry are exempt:
they are where a harness is allowed to be named. Everything else is pinned in
``PINNED`` and may only go down.

The harness names and the exempt files are derived from the live registry and
module objects, so a new harness or a renamed module cannot silently fall out
of the scan.
"""

from __future__ import annotations

import ast
from pathlib import Path

import charlie_work
from charlie_work import (
    adapter_fate_profile,
    adapters,
    api_worker,
    claude_code,
    devin_shell,
    harnesses,
    opencode_worker,
)
from _src_ast import parsed, source_files, source_text

PACKAGE_ROOT = Path(charlie_work.__file__).resolve().parent

SUBJECT = "adapter_kind"
RECORD_KINDS = "CLAUDE_RECORD_KINDS"
assert isinstance(getattr(claude_code, RECORD_KINDS), frozenset)  # the name still exists

# Both spellings name a harness: the registry key ("devin-shell") and the
# ``WorkerView.adapter_kind`` value its sidecars carry ("devin").
HARNESS_NAMES = frozenset(harnesses.HARNESS_REGISTRY) | frozenset(
    capabilities.adapter_kind for capabilities in harnesses.HARNESS_REGISTRY.values()
)

# Where a harness may be named: the profile module, the Adapter modules
# themselves, and the harness registry.
ALLOWED = frozenset(
    Path(module.__file__).resolve().relative_to(PACKAGE_ROOT).as_posix()
    for module in (
        adapter_fate_profile,
        harnesses,
        adapters,
        devin_shell,
        claude_code,
        api_worker,
        opencode_worker,
    )
)

# Harness-name branches left outside the allow-list, by file. Counts may only
# go down: move the question onto ``AdapterFateProfile`` and lower the number
# (or delete the entry at zero). Never raise one and never add a file.
PINNED: dict[str, int] = {
    "dead_worker_sweep/decide_dead_sessions.py": 1,
    "devin_review_resume.py": 1,
    "fleet_dispatch.py": 1,
    "orchestration/helpers_worker.py": 1,
    "worker.py": 7,
    "worker_literal_tmp.py": 1,
}


def _is_subject(node: ast.AST) -> bool:
    return (isinstance(node, ast.Name) and node.id == SUBJECT) or (
        isinstance(node, ast.Attribute) and node.attr == SUBJECT
    )


def _names_a_harness(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str) and node.value in HARNESS_NAMES
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return any(_names_a_harness(element) for element in node.elts)
    if isinstance(node, ast.Name):
        return node.id == RECORD_KINDS
    if isinstance(node, ast.Attribute):
        return node.attr == RECORD_KINDS
    return False


def _pattern_names_a_harness(pattern: ast.pattern) -> bool:
    if isinstance(pattern, ast.MatchValue):
        return _names_a_harness(pattern.value)
    if isinstance(pattern, ast.MatchOr):
        return any(_pattern_names_a_harness(option) for option in pattern.patterns)
    return False


def _branch_lines(tree: ast.AST) -> list[int]:
    """Line of every comparison of ``adapter_kind`` against a harness name.

    One hit per ``ast.Compare`` (``==``, ``!=``, ``in``, ``not in``, either
    operand order) and one per ``match adapter_kind`` case that names a harness.
    """
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            if any(_is_subject(o) for o in operands) and any(
                _names_a_harness(o) for o in operands
            ):
                lines.append(node.lineno)
        elif isinstance(node, ast.Match) and _is_subject(node.subject):
            lines.extend(
                case.pattern.lineno
                for case in node.cases
                if _pattern_names_a_harness(case.pattern)
            )
    return sorted(lines)


def _sites(root: Path) -> dict[str, list[int]]:
    """``{relative posix path: branch lines}`` for every non-exempt file under ``root``."""
    sites: dict[str, list[int]] = {}
    for path in source_files(root):
        relative = path.relative_to(root).as_posix()
        # A comparison of ``adapter_kind`` cannot exist in a file whose text
        # lacks the identifier; the filter keeps untouched files out of the parse.
        if relative in ALLOWED or SUBJECT not in source_text(path):
            continue
        lines = _branch_lines(parsed(path))
        if lines:
            sites[relative] = lines
    return sites


def _violations(sites: dict[str, list[int]], pinned: dict[str, int]) -> list[str]:
    """Every way ``sites`` departs from ``pinned``; empty when they agree exactly."""
    problems: list[str] = []
    for relative in sorted(set(sites) | set(pinned)):
        lines = sites.get(relative, [])
        allowed = pinned.get(relative, 0)
        if len(lines) > allowed:
            problems.append(
                f"{relative}: {len(lines)} harness-name branch(es) at line(s) {lines}, "
                f"pinned at {allowed} -- ask adapter_fate_profile.profile_for(...) "
                "instead of naming a harness (issue #3092)"
            )
        elif len(lines) < allowed:
            problems.append(
                f"{relative}: {len(lines)} harness-name branch(es), pinned at {allowed} -- "
                "lower its PINNED entry (delete it at zero) to lock the improvement in"
            )
    return problems


def test_scan_counts_each_shape_of_harness_name_branch() -> None:
    source = """
if w.adapter_kind == "devin": ...
if "claude-code" != adapter_kind: ...
if w.adapter_kind in CLAUDE_RECORD_KINDS: ...
if capabilities.adapter_kind not in claude_code.CLAUDE_RECORD_KINDS: ...
if view.adapter_kind in ("api", "opencode"): ...
if harness == "x" or adapter_kind == "devin-shell": ...
match w.adapter_kind:
    case "manual" | "command": ...
    case _: ...
"""
    assert _branch_lines(ast.parse(source)) == [2, 3, 4, 5, 6, 7, 9]


def test_scan_ignores_comparisons_that_name_no_harness() -> None:
    source = """
if w.adapter_kind == other.adapter_kind: ...
if w.adapter_kind in REVIEWER_ADAPTER_KINDS: ...
if w.adapter_kind is None: ...
if w.adapter_kind == "not-a-harness": ...
if w.harness == "devin": ...
record = Record(adapter_kind="devin")
match w.adapter_kind:
    case _: ...
"""
    assert _branch_lines(ast.parse(source)) == []


def test_scan_sees_the_pinned_branches() -> None:
    """Positive control: an empty scan would pass the ratchet vacuously."""
    sites = _sites(PACKAGE_ROOT)
    assert sites, "the scan found no harness-name branch at all"
    assert ALLOWED.isdisjoint(sites)
    assert all((PACKAGE_ROOT / relative).is_file() for relative in ALLOWED)


def test_harness_name_branches_outside_the_profile_only_go_down() -> None:
    problems = _violations(_sites(PACKAGE_ROOT), PINNED)
    assert not problems, "\n  ".join(["harness-name branch ratchet (issue #3092):", *problems])


def test_ratchet_fails_on_a_new_branch_outside_the_allow_list(tmp_path: Path) -> None:
    (tmp_path / "new_lane.py").write_text(
        'def reap(w):\n    if w.adapter_kind == "devin":\n        return 1\n', encoding="utf-8"
    )
    # The same branch inside an exempt file is not counted.
    exempt = Path(adapter_fate_profile.__file__).name
    assert exempt in ALLOWED
    (tmp_path / exempt).write_text(
        'def path(adapter_kind):\n    return adapter_kind == "devin"\n', encoding="utf-8"
    )

    sites = _sites(tmp_path)

    assert sites == {"new_lane.py": [2]}
    # A file with no pin fails outright ...
    assert any("new_lane.py" in problem for problem in _violations(sites, {}))
    # ... and so does one more branch in a file that is already pinned.
    assert _violations({"worker.py": [1] * (PINNED["worker.py"] + 1)}, PINNED)
    # A count that dropped must be locked in, so it cannot drift back up.
    assert _violations({"worker.py": [1] * (PINNED["worker.py"] - 1)}, PINNED)
    assert not _violations({"new_lane.py": [2]}, {"new_lane.py": 1})
