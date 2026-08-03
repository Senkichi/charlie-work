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

# Both ways into the package: `[project.scripts]` in pyproject.toml declares
# charlie = "charlie_work.cli:main", and `python -m charlie_work` enters through
# __main__. Omitting the second classified __main__ itself as dormant, which is
# how this list got checked rather than assumed.
ENTRY_MODULES: tuple[str, ...] = ("cli", "__main__")

MARKER = "rollback_path"

# Modules known to be live, used as the positive control below. Naming a few
# knowns is what a control *is* -- the derived set is the thing under test, and
# without this a graph walk that silently returned nothing would classify the
# whole package as dormant.
KNOWN_LIVE = {"cli", "config", "workflow"}


def _sibling_imports(path: Path) -> set[str]:
    """Names of ``charlie_work`` sibling modules imported by one module.

    AST rather than regex, and the whole tree rather than just the header: a
    fleet module imported lazily inside a function body is still an edge in the
    graph, and a name mentioned in a comment or a docstring is not.
    """
    found: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level and node.module:
                # from .runner_slots import x
                found.add(node.module.split(".")[0])
            elif node.level and node.module is None:
                # from . import runner_slots
                found.update(alias.name for alias in node.names)
            elif node.module and node.module.startswith("charlie_work."):
                found.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("charlie_work."):
                    found.add(alias.name.split(".")[1])
    return found


def _graph() -> dict[str, set[str]]:
    return {p.stem: _sibling_imports(p) for p in sorted(SRC.glob("*.py"))}


def _live_modules() -> set[str]:
    """Everything transitively reachable from the package's entry points."""
    graph = _graph()
    seen: set[str] = set()
    stack = list(ENTRY_MODULES)
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


def _modules_with_marker() -> set[str]:
    """Test modules carrying a module-level ``pytestmark`` for our marker.

    Read from source rather than via pytest's own collection so this test says
    the same thing whether it runs alone or inside the full suite, and so a
    failure names the file rather than a collected item id.
    """
    marked: set[str] = set()
    for path in sorted(TESTS.glob("test_*.py")):
        if f"pytest.mark.{MARKER}" in path.read_text(encoding="utf-8"):
            marked.add(path.name)
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


def test_every_dormant_fleet_module_has_its_tests_marked_and_no_others_do() -> None:
    """The #876 invariant, in both directions.

    ``expected`` is derived every run; it is never a literal. If the retention
    window closes and the superseded modules are deleted, ``expected`` empties
    and any leftover marker fails here -- which is #876's requirement that the
    modules and their tests go together.
    """
    dormant = _dormant_modules()
    expected = {f"test_{name}.py" for name in dormant if (TESTS / f"test_{name}.py").is_file()}
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
