"""Seam integrity for the adapters.py -> dispatch_results.py split (#2229 rework).

The worker-launch errors-as-values boundary pushed ``adapters.py`` past the
800-line file-size ratchet cap; the dispatch result value object and every
site that constructs one moved verbatim into the ``dispatch_results`` leaf.
``adapters.py`` re-exports the moved names (the same facade pattern as
``workflow.py``'s split leaves), so every ``charlie_work.adapters.<name>``
import path and monkeypatch target keeps resolving unchanged.

Two guards: the facade hands out the leaf's own objects (identity, not a
second definition drifting in place), and ``dispatch_results`` stays a leaf
— it references ``adapters`` under ``TYPE_CHECKING`` only, so no runtime
import edge ``dispatch_results -> adapters`` can creep in and recreate the
cycle the split exists to avoid.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import charlie_work.adapters as adapters
import charlie_work.dispatch_results as dispatch_results
from _src_ast import parsed

MOVED_NAMES = (
    "SessionDispatchResult",
    "_emit_launch_failed",
    "_launch_exc_result",
    "_record_result",
    "_result",
)


def test_adapters_reexports_the_moved_names() -> None:
    for name in MOVED_NAMES:
        moved = getattr(dispatch_results, name)
        assert getattr(adapters, name) is moved, name
        assert inspect.getmodule(moved) is dispatch_results, name


def test_dispatch_results_has_no_runtime_adapters_import() -> None:
    tree = parsed(Path(inspect.getfile(dispatch_results)))
    # Only module-body-level imports count as runtime edges: the adapters
    # names are needed for annotations alone and must stay inside the
    # ``if TYPE_CHECKING:`` block (nested, so absent from tree.body).
    offenders = [
        node.lineno
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and any(
            "adapters" in alias.name or "adapters" in (node.module or "") for alias in node.names
        )
    ]
    assert not offenders, (
        "dispatch_results must late-bind nothing and import adapters only "
        f"under TYPE_CHECKING -- runtime import at line(s): {offenders}"
    )
