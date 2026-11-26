"""Who still reaches a name through ``charlie_work.workflow``? One scan for the facade-split tests (#2914).

The ``test_<family>_split.py`` modules each prove that ``workflow.py``'s
re-export block still carries every name a consumer reaches through
``charlie_work.workflow``. Each ran its own copy of the same loop, and the
loop was the most expensive thing in the suite: for every ``.py`` file under
``tests/``, ``scripts/`` and ``src/`` it parsed the file, walked the whole
tree, and then ran two regexes *per candidate name* over the whole text. A
family with ~36 names is ~70 regex passes over ~1500 files (~13 of the ~21 s
of one scan), repeated by twelve tests across six modules -- ~250 s of CI test
time, concentrated by pytest-split's contiguous chunks into whichever shards
hold the late-sorting modules (``test_rework_prompts_split``,
``test_verdict_parsing_split``).

This module keeps the scan's meaning and changes only its cost:

* one regex pass per file finds every ``<receiver>.<ident>`` and
  ``"charlie_work.workflow.<ident>`` token, and the candidate set is applied to
  the identifiers it returns (instead of one compiled pattern per candidate);
* a file whose text lacks every token the scan can match is skipped before it
  is parsed -- a reference in any of the three forms contains the literal
  ``workflow`` (or a receiver such as ``wf``), so the skip cannot hide one.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Iterable
from pathlib import Path

from _src_ast import parsed, source_files, source_text

# The names a consumer binds the facade to: ``import charlie_work.workflow as
# workflow`` / ``wf`` and the two spellings older tests used.
DEFAULT_RECEIVERS: tuple[str, ...] = ("workflow", "workflow_module", "cw_workflow", "wf")

# Every textual form of an ``ImportFrom`` whose module is ``charlie_work.workflow``
# or ``.workflow`` ends the module name and starts the ``import`` keyword with
# nothing between them but whitespace and backslash continuations. A file with
# no such run cannot hold one, so it is not parsed.
_FROM_WORKFLOW_IMPORT = re.compile(r"workflow\s*(?:\\\r?\n\s*)*import\b")


def _reference_pattern(receivers: Iterable[str]) -> re.Pattern[str]:
    """One pattern for forms 2 and 3; the identifier is captured whole.

    ``(?<![.\\w])`` rather than a plain ``\\b``: ``workflow.`` is also the tail
    of the fully qualified ``charlie_work.workflow.``, and the lookbehind makes
    the receiver a standalone token, not the end of a longer dotted chain (the
    fully qualified spelling is the string-literal alternative's job, and it is
    only matched quote-delimited so a backtick-quoted docstring
    cross-reference is not mistaken for a monkeypatch target).

    ``(\\w+)`` takes the maximal identifier, which is exactly what the former
    per-name ``<name>\\b`` required: ``_foo`` does not match inside ``_foobar``.
    """
    alternation = "|".join(re.escape(receiver) for receiver in receivers)
    return re.compile(
        rf"(?<![.\w])(?:{alternation})\.(?P<attribute>\w+)"
        rf"""|['"]charlie_work\.workflow\.(?P<literal>\w+)"""
    )


def _import_tree(path: Path, source: str) -> ast.Module | None:
    """The tree to read ``ImportFrom`` nodes from, or None when it cannot hold one."""
    if not _FROM_WORKFLOW_IMPORT.search(source):
        return None
    try:
        return parsed(path)
    except SyntaxError:
        return None


def consumer_referenced_names(
    candidates: set[str],
    search_roots: list[Path],
    *,
    display: Callable[[Path], str],
    receivers: tuple[str, ...] = DEFAULT_RECEIVERS,
) -> dict[str, str]:
    """name -> the first file (``display(path)``) that reaches it through ``charlie_work.workflow``.

    Walks every ``.py`` file under ``search_roots``, in ``source_files`` order,
    for three reference forms:

    1. ``from charlie_work.workflow import <name>`` / ``from .workflow import
       <name>`` (AST ``ImportFrom``, so a multi-line parenthesized import
       block is caught the same as a single-line one).
    2. ``<receiver>.<name>`` attribute access, word-bounded so e.g.
       ``_extract_review_session_summary`` cannot partially match a longer name.
    3. The string-dotted monkeypatch form
       ``monkeypatch.setattr("charlie_work.workflow.<name>", ...)``.

    The first file to reference a name wins. Forms 2 and 3 are read from the
    text, so a file that does not parse still contributes them; it contributes
    no form-1 reference (a broken tree is a different failure).
    """
    reference = _reference_pattern(receivers)
    needles = ("workflow", *receivers)

    found: dict[str, str] = {}
    for root in search_roots:
        if not root.is_dir():
            continue
        for path in source_files(root):
            source = source_text(path)
            if not any(needle in source for needle in needles):
                continue
            tree = _import_tree(path, source)
            shown = display(path)
            if tree is not None:
                for node in ast.walk(tree):
                    if not isinstance(node, ast.ImportFrom):
                        continue
                    if node.module != "charlie_work.workflow" and not (
                        node.level == 1 and node.module == "workflow"
                    ):
                        continue
                    for alias in node.names:
                        if alias.name in candidates:
                            found.setdefault(alias.name, shown)
            for match in reference.finditer(source):
                name = match.group("attribute") or match.group("literal")
                if name in candidates:
                    found.setdefault(name, shown)
    return found
