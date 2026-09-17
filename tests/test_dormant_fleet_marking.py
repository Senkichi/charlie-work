"""Issue #876: fleet tests that no longer cover the live path must say so.

PR #869 repointed charlie-work's fleet consumers at the extracted ``ci_fleet``
package, and the extraction plan deliberately keeps the superseded modules in
the tree, re-activatable by config, for a month as the rollback path. Their
tests therefore keep running and keep passing -- while the allocator they
exercise is no longer the one making live decisions.

That is a worse failure mode than dead code. Dead code fails loudly; code that
is merely *off* the live path passes forever, and "charlie-work CI is green"
silently stops implying "fleet allocation is healthy" while looking exactly
like it did when it did imply that.

Why this is a test and not a docstring
--------------------------------------
#876's closing condition allows a docstring as the minimum. A docstring alone
would be a claim nobody re-checks -- the same decay this repo has been bitten by
elsewhere -- and, worse, it would be a claim about a *relationship* (this test
covers a module nothing calls) that changes without anyone touching either file.
Repoint one consumer back at ``charlie_work.runner_allocation`` and the docstring
silently becomes a lie.

So the dormant set is **derived from the import graph**, never written down:
a module is live if it is reachable from either package entry point --
``[project.scripts]``'s ``charlie_work.cli:main``, or ``__main__`` for
``python -m charlie_work`` -- and dormant otherwise. The marker set is then
asserted to match exactly, which makes both directions loud:

* Repoint a consumer back and the module becomes live -> its test must lose the
  marker, or this fails.
* Delete the superseded modules at the end of retention and the derived set
  empties -> the leftover markers fail, which is #876's *other* closing
  condition ("the modules and their tests are deleted together, not the modules
  alone") enforced mechanically rather than remembered.

The trap this is most likely to be "corrected" against
------------------------------------------------------
The live supervisor still logs, every pass::

    charlie_work.fleet_dispatch INFO Fleet allocation prologue: started=0 parked=0
    notes=1 (budget=8, managed_root=C:/actions-runners)

and ``fleet_dispatch.py`` really does call ``run_allocation_pass(...)``. Read from
the runtime side that looks exactly like a live ``charlie_work.runner_allocation``
consumer, and someone will eventually conclude this file is wrong. It is not: the
symbol is imported at ``fleet_dispatch.py:32`` from
``ci_fleet.charlie_work_adapter``, so it resolves to the *extracted* package. The
logger name is the module that CALLS the adapter, not the module that does the
work. That misreading is the most likely reason anyone would repoint an import
back at ``charlie_work.runner_allocation`` -- which is precisely the edit this test
exists to catch.

(Independently confirmed from the runtime side by the charlie-work session on
2026-08-03, which went looking for a contradiction and found none: no deferred or
function-local imports of the four modules exist anywhere in ``src/``.)

Known limitation, stated rather than papered over: reachability is computed from
static ``import`` statements, so a module reached only through ``importlib`` or a
plugin registry would look dormant. The failure mode is safe -- it demands a
marker on a test that has one too many, which a human reviews -- and no fleet
module is loaded that way today.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "charlie_work"
TESTS = REPO_ROOT / "tests"

# Ways into the package: `[project.scripts]` in pyproject.toml declares
# charlie = "charlie_work.cli:main", and `python -m charlie_work` enters through
# __main__. Omitting the second classified __main__ itself as dormant, which is
# how this list got checked rather than assumed. merge_preflight_hook is a
# third entry point: `.claude/settings.json` registers it as a PreToolUse
# hook via `python -m charlie_work.merge_preflight_hook` (#894), so nothing
# in src/ imports it by design.
ENTRY_MODULES: tuple[str, ...] = ("cli", "__main__", "merge_preflight_hook")

MARKER = "rollback_path"

# Modules known to be live, used as the positive control below. Naming a few
# knowns is what a control *is* -- the derived set is the thing under test, and
# without this a graph walk that silently returned nothing would classify the
# whole package as dormant.
KNOWN_LIVE = {"cli", "config", "workflow"}


def _package_tree_files() -> list[Path]:
    """Every ``*.py`` in the package tree, derived -- never declared (#1687).

    The top level of ``src/charlie_work`` plus every directory beneath it that
    is a package (contains ``__init__.py``), recursively. The pre-#1687 code
    globbed the top level plus a declared ``orchestration/`` glob, so a new
    subpackage could hold a module's only importer while sitting outside the
    graph entirely -- the same fails-open blind spot this file exists to
    remove. (``prompts/`` is a template directory, not a package, and
    contributes nothing.)
    """
    files = sorted(SRC.glob("*.py"))
    pending = [SRC]
    while pending:
        current = pending.pop()
        for child in sorted(current.iterdir()):
            if child.is_dir() and (child / "__init__.py").is_file():
                files.extend(sorted(child.glob("*.py")))
                pending.append(child)
    return files


def _node_name(path: Path) -> str:
    """Collision-free node identity: path relative to SRC, POSIX, no suffix.

    ``orchestration/dispatch_state.py`` is ``orchestration/dispatch_state``;
    ``github_capabilities/__init__.py`` is ``github_capabilities/__init__``.
    Basenames collide across subpackages (``checks.py``, ``labels.py``,
    ``__main__.py``, ``__init__.py``), so a ``path.stem`` key would merge or
    misattribute nodes silently -- the reason the scan could not widen until
    the key changed (#1687).
    """
    return path.relative_to(SRC).with_suffix("").as_posix()


def _resolve_node(parts: list[str], known: set[str]) -> str | None:
    """Map dotted-path parts (relative to SRC) to a graph node, or ``None``.

    ``["a", "b"]`` names ``a/b`` when ``a/b.py`` exists and ``a/b/__init__``
    when ``a/b`` is a package directory; the longest existing prefix wins, so
    ``from a.b import sym`` still lands on ``a/b`` when ``sym`` is a plain
    symbol. Callers additionally probe ``base + [alias]`` for the
    ``from pkg import submodule`` shape.
    """
    for n in range(len(parts), 0, -1):
        prefix = "/".join(parts[:n])
        if prefix in known:
            return prefix
        if prefix + "/__init__" in known:
            return prefix + "/__init__"
    return None


def _sibling_imports(path: Path, known: set[str]) -> set[str]:
    """Graph nodes imported by one module, resolved to node identities.

    AST rather than regex, and the whole tree rather than just the header: a
    fleet module imported lazily inside a function body is still an edge in
    the graph, and a name mentioned in a comment or a docstring is not.

    Relative imports resolve against the importing file's own package, so
    ``from ._base import x`` inside ``github_capabilities/`` names
    ``github_capabilities/_base`` while ``from ..checks import y`` there names
    top-level ``checks``; absolute ``charlie_work.*`` imports resolve to the
    same node identities. Importing ``a/b`` also marks ``a/__init__`` (and
    every intermediate package init) as reached, because Python executes the
    package's ``__init__`` on any submodule import.

    Known limitation, stated rather than papered over: ``from charlie_work
    import X`` -- the *bare-package* form, no dotted tail -- was invisible to
    the pre-#1687 basename parser (it matched only ``charlie_work.`` prefixes)
    and stays invisible here, deliberately, so this re-key does not change the
    derived dormant set. It is why ``rescue`` currently reads dormant: its
    only importers are ``orchestration/`` delegates using exactly that form.
    """
    pkg_parts = path.parent.relative_to(SRC).parts
    found: set[str] = set()

    def add(node: str | None) -> None:
        if node is None:
            return
        found.add(node)
        # Importing ``a/b`` executes ``a/__init__`` first -- every package
        # ancestor of a resolved node is reached too.
        ancestors = node.split("/")[:-1]
        for i in range(1, len(ancestors) + 1):
            init = "/".join(ancestors[:i]) + "/__init__"
            if init in known:
                found.add(init)

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            level = node.level or 0
            if level > len(pkg_parts) + 1:
                # Climbs past the charlie_work package root -- not a sibling
                # edge this graph can express.
                continue
            if level == 0:
                if not (node.module and node.module.startswith("charlie_work.")):
                    continue
                base = node.module.split(".")[1:]
            else:
                base = list(pkg_parts[: len(pkg_parts) - (level - 1)])
                if node.module:
                    base.extend(node.module.split("."))
            add(_resolve_node(base, known))
            for alias in node.names:
                add(_resolve_node([*base, alias.name], known))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("charlie_work."):
                    add(_resolve_node(alias.name.split(".")[1:], known))
    return found


def _graph() -> dict[str, set[str]]:
    files = _package_tree_files()
    known = {_node_name(p) for p in files}
    edges = {_node_name(p): _sibling_imports(p, known) for p in files}

    # Per-subpackage ownership, DERIVED from the tree (#1687) rather than
    # declared: a subpackage imported by exactly ONE top-level module belongs
    # to that module, so its files' edges fold into the owner's node instead of
    # forming nodes of their own. That is `orchestration/`'s situation today:
    # OrchestratorApp method bodies live in `orchestration/*.py` delegate
    # modules, re-attached onto the class at `charlie_work.workflow` import
    # time by `workflow_delegation._install_delegates`, and `workflow.py` is
    # the subpackage's sole top-level importer (`from . import orchestration`).
    # The delegates ARE `workflow`'s methods in every sense that matters to
    # reachability -- a sibling module a moved body is the sole importer of
    # (e.g. `module_map` after `_build_module_map_value` moved, L05/#1636)
    # would otherwise fall false-dormant (issue #1671). A subpackage imported
    # by zero or two-or-more top-level modules (`attachment_contracts`,
    # `github_capabilities`) keeps its modules as nodes in their own right.
    #
    # Tradeoff (stated, not papered over): folding adds edges and can only
    # *shrink* the detectable dormant set, never grow it, so island-detection
    # precision inside a folded package is reduced -- a module imported only
    # by an orchestration delegate reads as live rather than dormant.
    # Accepted to kill the `module_map` false-dormant; bounded by
    # `test_widened_graph_still_flags_a_genuine_island`, which proves a module
    # reachable from neither the top level nor a folded package is still
    # flagged.
    fold_owner: dict[str, str] = {}
    for child in sorted(SRC.iterdir()):
        if not (child.is_dir() and (child / "__init__.py").is_file()):
            continue
        prefix = child.name + "/"
        importers = {
            name
            for name, targets in edges.items()
            if "/" not in name
            and name != "__init__"
            and any(t == prefix + "__init__" or t.startswith(prefix) for t in targets)
        }
        if len(importers) == 1:
            fold_owner[child.name] = next(iter(importers))

    graph: dict[str, set[str]] = {}
    for name, targets in edges.items():
        owner = name
        if "/" in name:
            owner = fold_owner.get(name.split("/", 1)[0], name)
        # Folded files never become nodes themselves; their edges merge into
        # the owning top-level module's edge set.
        graph.setdefault(owner, set()).update(targets)
    return graph


def _live_modules() -> set[str]:
    """Everything transitively reachable from the package's entry points.

    Entry points are the declared top-level ``ENTRY_MODULES`` plus every
    ``__main__.py`` in the tree -- ``python -m charlie_work.<pkg>`` is an
    entry by construction, derived rather than listed (so
    ``attachment_contracts/__main__``, invoked out-of-band by the
    attachment-contracts CI workflow, is never mistaken for an island).
    """
    graph = _graph()
    seen: set[str] = set()
    stack = [e for e in ENTRY_MODULES if e in graph]
    stack.extend(n for n in graph if n.endswith("/__main__"))
    while stack:
        name = stack.pop()
        if name in seen or name not in graph:
            continue
        seen.add(name)
        stack.extend(graph[name])
    return seen


def _dormant_modules() -> set[str]:
    graph = _graph()
    return {name for name in graph if name not in _live_modules() and name != "__init__"}


def _test_file_candidates(module: str) -> list[str]:
    """Marker-requiring test files for ``module``, relative to ``tests/``.

    Top-level ``foo`` keeps the historical ``test_foo.py`` form. A nested
    module ``pkg/foo`` gets unambiguous forms only: ``pkg/test_foo.py``
    (mirroring the package tree -- the ``tests/attachment_contracts/``
    convention) or ``test_pkg_foo.py`` (flattened). The bare ``test_foo.py``
    is deliberately NOT accepted for a nested module: it cannot be told
    apart from a test for top-level ``foo``, the same basename ambiguity the
    re-key exists to remove.
    """
    parts = module.split("/")
    if len(parts) == 1:
        return [f"test_{parts[0]}.py"]
    parent = "/".join(parts[:-1])
    return [f"{parent}/test_{parts[-1]}.py", f"test_{'_'.join(parts)}.py"]


def _modules_with_marker() -> set[str]:
    """Test modules carrying a module-level ``pytestmark`` for our marker.

    Read from source rather than via pytest's own collection so this test says
    the same thing whether it runs alone or inside the full suite, and so a
    failure names the file rather than a collected item id. Keys are paths
    relative to ``tests/`` (POSIX), not basenames, so a mark inside a mirrored
    subpackage (``tests/<pkg>/test_<mod>.py``) is attributable.
    """
    marked: set[str] = set()
    for path in sorted(TESTS.rglob("test_*.py")):
        if f"pytest.mark.{MARKER}" in path.read_text(encoding="utf-8"):
            marked.add(path.relative_to(TESTS).as_posix())
    return marked


def test_the_reachability_walk_actually_reaches_things() -> None:
    """Positive control. An empty or broken walk would classify every module as
    dormant, and the assertion below would then be measuring nothing -- the
    exact shape of "an absence is not evidence until you show the query could
    have returned something."
    """
    live = _live_modules()
    assert len(live) > 10, f"reachability walk returned {len(live)} modules -- graph is broken"
    missing = KNOWN_LIVE - live
    assert not missing, f"known-live modules were not reached: {sorted(missing)}"


def test_widened_graph_still_flags_a_genuine_island(tmp_path, monkeypatch) -> None:
    """Negative control for the ``_graph()`` widening (#1671, #1636).

    The widening folds ``orchestration/*.py`` delegate imports into the
    ``workflow`` node so a sibling module a moved body is the sole importer of
    (e.g. ``module_map`` after ``_build_module_map_value`` relocated) does not
    fall false-dormant. That fix can only ever *shrink* the detectable dormant
    set -- it adds edges, never removes them -- so the guard's island-detection
    precision is reduced, not strengthened: a module imported only by an
    orchestration delegate would now read as live instead of dormant. This test
    bounds that reduction by proving a module no top-level *or* orchestration
    file imports is still flagged dormant, i.e. a genuine new island is still
    caught.

    A synthetic tree (monkeypatched ``SRC``) is used rather than dropping a file
    into the real ``src/`` tree so the test is hermetic and cannot perturb the
    other two tests in this file (which read the real tree).
    """
    import sys

    src = tmp_path / "charlie_work"
    src.mkdir()
    (src / "cli.py").write_text("from . import workflow\n", encoding="utf-8")
    (src / "__main__.py").write_text("", encoding="utf-8")
    # Sole top-level importer of the orchestration subpackage -> the package
    # folds into this node by derivation, exactly like the real tree.
    (src / "workflow.py").write_text("from . import orchestration\n", encoding="utf-8")
    (src / "module_map.py").write_text("", encoding="utf-8")
    # A genuine island: nothing in src/ imports it, top-level or orchestration.
    (src / "_synthetic_island.py").write_text("", encoding="utf-8")
    orch = src / "orchestration"
    orch.mkdir()
    (orch / "__init__.py").write_text("", encoding="utf-8")
    # A delegate whose sole sibling import is module_map -- the case the fold
    # exists to keep live rather than false-dormant. (`from .. import X` is the
    # subpackage-to-top-level form, resolving to the `module_map` node.)
    (orch / "delegate.py").write_text("from .. import module_map\n", encoding="utf-8")

    monkeypatch.setattr(sys.modules[__name__], "SRC", src)
    dormant = _dormant_modules()

    # The false-dormant fix holds: module_map is reached via the delegate.
    assert "module_map" not in dormant, (
        "module_map fell dormant -- the _graph() widening is not folding "
        "orchestration delegate imports into the workflow node"
    )
    # The sensitivity retained: a genuine island is still flagged.
    assert "_synthetic_island" in dormant, (
        "synthetic island was not flagged dormant -- the widening erased the "
        "guard's ability to detect a module no src/ importer reaches"
    )


def test_multi_importer_subpackage_keeps_modules_as_nodes(tmp_path, monkeypatch) -> None:
    """Ownership derivation (#1687): a subpackage imported by MORE than one
    top-level module does NOT fold -- its files stay graph nodes in their own
    right, so an island inside it is still flagged and a ``__main__.py``
    inside it is a derived ``python -m`` entry point, never dormant.
    """
    import sys

    src = tmp_path / "charlie_work"
    src.mkdir()
    (src / "cli.py").write_text("from . import imp_a, imp_b\n", encoding="utf-8")
    (src / "__main__.py").write_text("", encoding="utf-8")
    (src / "imp_a.py").write_text("from . import pkg\n", encoding="utf-8")
    (src / "imp_b.py").write_text("from . import pkg\n", encoding="utf-8")
    pkg = src / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from . import live_mod\n", encoding="utf-8")
    (pkg / "live_mod.py").write_text("", encoding="utf-8")
    # An island INSIDE an own-nodes subpackage: nothing imports it.
    (pkg / "island_mod.py").write_text("", encoding="utf-8")
    # `python -m charlie_work.pkg` entry point -- live by derivation, not list.
    (pkg / "__main__.py").write_text("", encoding="utf-8")

    monkeypatch.setattr(sys.modules[__name__], "SRC", src)
    dormant = _dormant_modules()

    assert "pkg/island_mod" in dormant, (
        "an island inside a non-folded subpackage must still be flagged -- "
        "the package's modules are nodes in their own right"
    )
    assert "pkg/live_mod" not in dormant
    assert "pkg/__init__" not in dormant
    assert "pkg/__main__" not in dormant, (
        "a subpackage __main__.py is a `python -m` entry point, derived from "
        "the tree -- it must never read as dormant"
    )


def test_every_dormant_fleet_module_has_its_tests_marked_and_no_others_do() -> None:
    """The #876 invariant, in both directions.

    ``expected`` is derived every run; it is never a literal. If the retention
    window closes and the superseded modules are deleted, ``expected`` empties
    and any leftover marker fails here -- which is #876's requirement that the
    modules and their tests go together.
    """
    dormant = _dormant_modules()
    expected = {
        candidate
        for name in dormant
        for candidate in _test_file_candidates(name)
        if (TESTS / candidate).is_file()
    }
    actual = _modules_with_marker()

    assert actual == expected, (
        f"tests marked '{MARKER}' do not match the dormant modules derived from the "
        f"import graph.\n"
        f"  dormant modules:      {sorted(dormant)}\n"
        f"  should be marked:     {sorted(expected)}\n"
        f"  actually marked:      {sorted(actual)}\n"
        f"  missing the marker:   {sorted(expected - actual)}\n"
        f"  marked but now live:  {sorted(actual - expected)}"
    )
