"""Seam integrity for the worktree.py -> review_checkouts.py split (#3224 rework).

``review_checkouts.py`` holds the ``pr-<N>`` review-checkout family --
``_review_checkout_path``, ``review_checkout_kwargs``,
``create_review_checkout``, ``remove_review_checkout`` -- moved out of
``worktree.py`` under the file-size ratchet (issue #1442) per the #1283
facade pattern: ``worktree.py`` re-exports the three public names through a
``from .review_checkouts import (...)  # noqa: F401`` block so every existing
``charlie_work.worktree.<name>`` import path and consumer-module monkeypatch
target keeps resolving unchanged.

The failure modes this guards, same three every split suite in the lineage
documents:

* ``review_checkouts.py`` could grow a module-level import of ``worktree``
  -- ``worktree.py`` already imports ``review_checkouts.py`` for the facade,
  so the reverse would be the exact import cycle this pattern exists to
  avoid. The family's worktree-internal dependencies (``remove_worktree``,
  ``list_worktrees``, the fetch helpers, ``WorktreeInfo``) are therefore
  imported lazily inside each function body; only module-level imports close
  the cycle, so this checks top-level import statements only.
* The facade could re-declare a name instead of importing it -- only object
  identity distinguishes a copy from the real implementation.
* The moved code could stop being reachable: ``worktree.py``'s facade import
  is also what keeps ``review_checkouts`` on the live import graph that
  ``tests/test_dormant_fleet_marking.py`` derives.
"""

from __future__ import annotations

import ast
from pathlib import Path

from _src_ast import parsed

_REPO_ROOT = Path(__file__).parents[1]
_MODULE_PATH = _REPO_ROOT / "src" / "charlie_work" / "review_checkouts.py"
_WORKTREE_PATH = _REPO_ROOT / "src" / "charlie_work" / "worktree.py"

_FACADE_NAMES = (
    "create_review_checkout",
    "remove_review_checkout",
    "review_checkout_kwargs",
)


def _module_level_imports_of_worktree(path: Path) -> list[str]:
    """Any module-level import of ``worktree`` in *path*'s own source.

    Walks only ``tree.body`` -- a ``from .worktree import`` nested inside a
    function body is the deliberate cycle break, not a violation.
    """
    offenders: list[str] = []
    for node in parsed(path).body:
        if isinstance(node, ast.ImportFrom):
            if (
                (node.level == 1 and node.module == "worktree")
                or (
                    node.level == 1
                    and node.module is None
                    and any(alias.name == "worktree" for alias in node.names)
                )
                or node.module == "charlie_work.worktree"
            ):
                offenders.append(f"line {node.lineno}: {ast.unparse(node)}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "charlie_work.worktree":
                    offenders.append(f"line {node.lineno}: {ast.unparse(node)}")
    return offenders


def test_review_checkouts_has_no_module_level_worktree_import() -> None:
    """Import-cycle guard: review_checkouts.py must never import worktree at
    module level -- worktree.py's facade already imports review_checkouts,
    so the reverse import would close the cycle. Function-local lazy imports
    are the sanctioned escape hatch and are ignored (they run after both
    modules are fully initialized).
    """
    offenders = _module_level_imports_of_worktree(_MODULE_PATH)
    assert offenders == [], (
        f"review_checkouts.py imports worktree at module level: {offenders} -- "
        "worktree.py's facade already imports FROM review_checkouts.py, so the "
        "reverse import would create the exact cycle this pattern exists to avoid"
    )


def test_module_level_import_detector_flags_a_real_violation() -> None:
    """Positive control: the detector must fire on each import spelling and
    must NOT fire on a function-local lazy import (the sanctioned shape).
    """
    import tempfile

    violations = (
        "from .worktree import remove_worktree\n",
        "from . import worktree\n",
        "import charlie_work.worktree\n",
        "from charlie_work.worktree import remove_worktree\n",
    )
    with tempfile.TemporaryDirectory() as td:
        probe = Path(td) / "probe.py"
        for violation in violations:
            probe.write_text(violation, encoding="utf-8")
            assert _module_level_imports_of_worktree(probe) != [], violation

        # The deliberate cycle break: deferred inside a function body.
        probe.write_text(
            "def f():\n    from .worktree import remove_worktree\n    return remove_worktree\n",
            encoding="utf-8",
        )
        assert _module_level_imports_of_worktree(probe) == []


def test_review_checkouts_module_imports_cleanly() -> None:
    """Behavioral form: importing the new module works and produced the
    family's real symbols.
    """
    import importlib

    module = importlib.import_module("charlie_work.review_checkouts")
    assert module.__name__ == "charlie_work.review_checkouts"
    assert callable(module.create_review_checkout)
    assert callable(module.remove_review_checkout)
    assert callable(module.review_checkout_kwargs)


def test_worktree_facade_reexports_the_family_by_identity() -> None:
    """Every public name review_checkouts.py defines resolves through the
    worktree.py facade to the SAME object, not a re-declared copy.
    """
    import charlie_work.review_checkouts as review_checkouts
    import charlie_work.worktree as worktree

    for name in _FACADE_NAMES:
        assert getattr(worktree, name) is getattr(review_checkouts, name), (
            f"worktree.{name} is not review_checkouts.{name} -- the facade must "
            "import, never redeclare"
        )


def test_worktree_facade_block_covers_exactly_the_public_family() -> None:
    """The facade block's own AST-derived import list contains exactly the
    three public names -- the private ``_review_checkout_path`` stays
    module-internal.
    """
    facade_names: set[str] = set()
    for node in ast.walk(parsed(_WORKTREE_PATH)):
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module == "review_checkouts"
        ):
            facade_names.update(alias.asname or alias.name for alias in node.names)
    assert facade_names == set(_FACADE_NAMES), (
        f"worktree.py's `.review_checkouts` facade block re-exports "
        f"{sorted(facade_names)}, expected exactly {sorted(_FACADE_NAMES)}"
    )
