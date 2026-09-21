"""Enforcement test: the escalated/blocked pair must come from SINK_STATUSES.

Issue #1765 finding 4: ``state_mechanical.py`` (three call sites) and
``instrumentation_ops.py`` (one) hand-spelled ``("escalated", "blocked")``
as a tuple literal instead of importing ``state.SINK_STATUSES`` -- the same
"three independent call sites hardcoding a check" shape ``SINK_STATUSES``'s
own docstring says it exists to prevent (see state.py). A fourth status
added to the sink later would silently miss any hand-spelled literal the
same way #1642's "blocked" missed the pre-#1765 ``== "escalated"`` checks.

Built with :mod:`ast`, not a text/regex search: a regex over source text
cannot tell a live literal from the same characters inside a comment or a
docstring, and would need its own exclusion list to avoid false positives
on this file and on prose like state_mechanical.py's own module docstring
(which describes the selection query in English using that same pair).
The AST sees comments not at all, and represents a docstring as a single
opaque string constant -- never as a nested tuple/list/set node -- so a
scan for actual ``ast.Tuple``/``ast.List``/``ast.Set`` nodes is exempt from
both by construction, with no separate exclusion logic required.

The two-element collection literal (list, tuple, or set -- any is an
equally easy hand-respell) whose *string* elements are exactly
``{"escalated", "blocked"}``, in either order, is forbidden everywhere
except ``state.py`` itself, which defines ``SINK_STATUSES`` as exactly that
frozenset literal on purpose (there being nothing to derive it from -- it is
the derivation's own source).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "charlie_work"

_FORBIDDEN_PAIR = frozenset({"escalated", "blocked"})
_ALLOWED_BASENAMES = frozenset({"state.py"})


def _literal_pair_findings(path: Path) -> list[int]:
    """Return line numbers where a live ``{"escalated", "blocked"}``-shaped
    literal (list/tuple/set of exactly those two strings) appears."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    findings: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            continue
        elts = node.elts
        if len(elts) != 2:
            continue
        if not all(isinstance(elt, ast.Constant) and isinstance(elt.value, str) for elt in elts):
            continue
        values = frozenset(elt.value for elt in elts)  # type: ignore[union-attr]
        if values == _FORBIDDEN_PAIR:
            findings.append(node.lineno)
    return findings


def _source_files() -> list[Path]:
    return sorted(SRC_ROOT.rglob("*.py"))


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: str(p.relative_to(SRC_ROOT)))
def test_no_hardcoded_escalated_blocked_pair(path: Path) -> None:
    if path.name in _ALLOWED_BASENAMES:
        pytest.skip(f"{path.name} defines SINK_STATUSES itself")
    findings = _literal_pair_findings(path)
    assert not findings, (
        f"{path.relative_to(REPO_ROOT)} hand-spells the escalated/blocked pair "
        f"as a literal at line(s) {findings} -- derive from "
        f"charlie_work.state.SINK_STATUSES instead (issue #1765 finding 4)"
    )


def test_guard_catches_a_reintroduced_literal(tmp_path: Path) -> None:
    """The scan itself must fail on the exact shape it is meant to catch.

    Without this, a change to ``_literal_pair_findings`` that quietly stops
    matching (e.g. an over-narrow node-type check) would leave every
    parametrized case above vacuously green.
    """
    sample = tmp_path / "sample.py"
    sample.write_text(
        'STATUS = "blocked"\n'
        "\n"
        "def f(status):\n"
        '    return status not in ("escalated", "blocked")\n',
        encoding="utf-8",
    )
    assert _literal_pair_findings(sample) == [4]


def test_guard_ignores_prose_mentioning_the_pair(tmp_path: Path) -> None:
    """A docstring/comment naming both statuses in English is not a literal.

    Regression guard for the AST-vs-regex design choice explained in this
    module's docstring: the pair only needs to be a real collection literal,
    never a substring match, so ordinary prose referencing both statuses
    (as state_mechanical.py's own docstrings do) must never be flagged.
    """
    sample = tmp_path / "sample.py"
    sample.write_text(
        "def f():\n"
        '    """Selection query: status in ("escalated", "blocked").\n'
        "    # not a real comment, just prose inside the docstring\n"
        '    """\n'
        "    return None\n",
        encoding="utf-8",
    )
    assert _literal_pair_findings(sample) == []
