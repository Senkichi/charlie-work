"""Conformance suite for candidate 3 ("markdown structure"), architecture-
deepening plan `docs/superpowers/plans/2026-09-29-architecture-deepening.md`.

Runs `tests/markdown_conformance_cases.py`'s CommonMark ground-truth table
against **both** CommonMark-correct scan implementations the plan calls for:

* `charlie_work.markdown_fence.scan` -- the shared scan side, wired into
  every one of the seven hand-rolled consumers `md-recon.md` inventories
  (`md-3-wire-notes.md`).
* `scripts/heartbeat_check.py`'s own stdlib-only copy, `_scan_markdown_
  structure` -- a deliberate duplicate, not an import, because that script
  is stdlib-only by design (`scripts/README.md:50-51`) and cannot depend on
  `charlie_work`. It exists purely so this same table can characterize it
  too; `heartbeat_check.py`'s actual `_mentioned_issue_numbers` is wired
  onto it (via `_strip_fenced_code_blocks`), replacing the old,
  CommonMark-deviating `_FENCED_CODE_BLOCK_RE` regex (the flip is pinned by
  `tests/test_markdown_structure_characterization.py`'s two
  `test_flip_heartbeat_check_*` tests, which this file does not touch).

Both implementations expose the same shape (`.fenced_line_spans`,
`.quoted_lines`, `.heading_lines`) so one parametrized test body drives both.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Protocol

import pytest

from _heartbeat_check_fixtures import _load_heartbeat_check
from charlie_work import markdown_fence
from markdown_conformance_cases import CASES, ConformanceCase


class _Scanner(Protocol):
    def __call__(self, text: str) -> object: ...


@pytest.fixture(scope="module")
def hb():
    return _load_heartbeat_check()


@pytest.fixture(params=["markdown_fence.scan", "heartbeat_check._scan_markdown_structure"])
def scan(request: pytest.FixtureRequest, hb: object) -> _Scanner:
    if request.param == "markdown_fence.scan":
        return markdown_fence.scan
    return hb._scan_markdown_structure  # type: ignore[attr-defined]


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_conformance(case: ConformanceCase, scan: _Scanner) -> None:
    structure = scan(case.markdown)

    assert structure.fenced_line_spans == case.fenced_line_spans, (
        f"{case.name}: fenced_line_spans mismatch"
    )
    assert tuple(sorted(structure.quoted_lines)) == case.quoted_lines, (
        f"{case.name}: quoted_lines mismatch"
    )
    assert structure.heading_lines == case.heading_lines, f"{case.name}: heading_lines mismatch"


def test_conformance_table_is_not_vacuous() -> None:
    """A guard against the whole suite silently collecting zero cases."""
    assert len(CASES) >= 14


def test_both_implementations_are_independent_objects(hb: object) -> None:
    """`heartbeat_check`'s copy must be a genuine duplicate, not an import.

    Guards the stdlib-only invariant (`scripts/README.md:50-51`): if a
    future edit made `heartbeat_check` import `markdown_fence.scan` instead
    of keeping its own copy, this would catch it going forward -- the two
    callables would become the exact same function object.
    """
    assert hb._scan_markdown_structure is not markdown_fence.scan


def test_heartbeat_check_never_imports_markdown_fence() -> None:
    """Enforces the stdlib-only property (`scripts/README.md:50-51`) by
    inspecting `heartbeat_check.py`'s imports, which the function-identity
    test above cannot: a thin wrapper around `markdown_fence.scan` would
    pass that test. No script may import `charlie_work.markdown_fence`."""
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    offenders: list[str] = []
    for script in sorted(scripts_dir.glob("*.py")):
        tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                names = [base] + [f"{base}.{alias.name}" for alias in node.names]
            else:
                continue
            if any("markdown_fence" in name for name in names):
                offenders.append(f"{script.name}:{node.lineno}")
    assert offenders == []


def test_fence_inside_blockquote_is_a_documented_gap(scan: _Scanner) -> None:
    """Known, deliberate gap (md-r2 review N6): a fenced block *inside* a
    blockquote is CommonMark structure (`> ```), but neither scan models
    blockquote containers -- each `>` line is just a quoted line and no
    FenceSpan is produced. No consumer is weakened: exclusion guards treat
    every `>` line as quoted, and the masking guard fails closed (nothing is
    exempted). This pins the gap so closing it is a visible, intentional
    change rather than a silent one."""
    structure = scan("> ```\n> x\n> ```\n")

    assert structure.fenced_line_spans == ()
    assert tuple(sorted(structure.quoted_lines)) == (0, 1, 2)
