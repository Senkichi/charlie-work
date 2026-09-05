"""Derived delegation installer for ``OrchestratorApp`` (Track 2 Phase B, #1631).

Phase B moves ``OrchestratorApp`` method *bodies* out of
``charlie_work.workflow`` into the domain submodules of
``charlie_work.orchestration`` (design doc
``docs/design/2026-09-04-orchestratorapp-mikado-graph-and-delegation-plan.md``,
Section 3.2). This module is the pure-infrastructure analog of the merged
``github_delegation.py``: it introspects each destination submodule's own
top-level ``def``s, derives a ``name -> module`` route table, and installs each
function onto ``OrchestratorApp`` as a plain class attribute.

Two design points distinguish this from ``github_delegation.py``:

- **Functions are installed unwrapped.** A plain function assigned to a class
  attribute binds as a method through the descriptor protocol, so
  ``app.<name>(...)`` passes ``app`` as ``self`` with no forwarding shim. Not
  wrapping is deliberate and load-bearing: ``patch.object(OrchestratorApp, name)``,
  ``patch("...OrchestratorApp.name")`` and instance
  ``monkeypatch.setattr(app, name, ...)`` all keep intercepting, and the
  AST-equivalence gate (#1607) sees the byte-identical moved ``FunctionDef``
  rather than a wrapper.
- **Owner-shadow is a hard error, not a skip.** ``github_delegation`` *skips* a
  name already on the owner (an unmoved method wins). Phase B instead *raises*:
  a leaf that moves a body must also remove the lexical ``def`` from the class in
  the same PR, so a name existing in *both* places means the author forgot the
  removal, and silently overwriting the real method (``setattr`` runs after the
  class body, so it would win) would be worse than loud failure.

Adapter deferral note: the ``property`` / ``staticmethod`` adapter mechanism
(marker attribute + ``as_property`` / ``as_staticmethod`` decorators + the
``_adapt`` wrapper) is scoped to the L09 leaf (design Section 3.3), which is the
first leaf to move a ``@property`` / ``@staticmethod`` member. It is intentionally
**not** landed here: L00 moves zero members, so the adapter surface would have
zero production callers in this diff. L09 reintroduces it alongside its first
real consumer (``layout``, ``_is_dead_blocker``, ``_write_json``). Until then
every routed function is a plain ``def`` installed unwrapped.

Signature note (rule-9 / #1631 reconciliation): ``_build_routes(modules)`` owns
the cross-module collision check; ``_install_delegates(owner_cls, modules)`` owns
the owner-shadow check, because only it is handed the owner class. #1631's prose
bundles both raises under ``_build_routes``; the functional requirement -- both
conditions raise a clear error, both are gated by tests -- is met by this split.

Module-namespace rule (#1627): this module must not
``from charlie_work.workflow import ...`` anything, and does not import
``charlie_work.workflow`` at all. It takes the owner class as a parameter (like
``github_delegation._install_delegates``), so there is no import to order around
a cycle.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import textwrap
from collections.abc import Iterator
from types import ModuleType
from typing import Any, Callable

# Records, per owner class, the set of names this installer has attached. Lets a
# repeated ``_install_delegates`` call be idempotent (re-attaching a name it
# already owns is a no-op) while still raising on a genuine lexical shadow -- a
# name present on the class that this installer did NOT put there. Stored via
# ``setattr`` after class creation, so it is a plain class attribute (an
# ``Assign``), never a ``FunctionDef``: it does not change the APC member count.
_INSTALLED_MARKER = "__delegate_installed_names__"


def _routable_defs(module: ModuleType) -> Iterator[tuple[str, Callable[..., Any]]]:
    """Yield ``(name, function)`` for every top-level ``def`` ``module`` declares.

    Selection mirrors ``github_delegation._routable_members`` (design Section
    3.2): non-dunder names only, single-underscore private names included (moved
    bodies call each other by name via ``self._name(...)`` and the owner has no
    ``__getattr__`` fallback), and names *imported* into the module excluded --
    a routed function must be one this module actually defines, identified by
    ``obj.__module__ == module.__name__`` rather than by any name list.

    A top-level ``property`` / ``staticmethod`` / ``classmethod`` *object* is a
    hard error, not a silent skip: such an object is not introspectable as a
    routable ``def`` (``inspect.isfunction`` is false for it), so it would be
    silently dropped and surface late as an ``AttributeError`` at
    ``app.<name>`` (a design Section 9 stop condition). A routed member must be a
    plain ``def``; the L09 leaf (design Section 3.3) reintroduces the
    ``as_property`` / ``as_staticmethod`` markers for the property/staticmethod
    cases, but until then a bare decorator object is always a mistake.
    """
    for name, obj in vars(module).items():
        if name.startswith("__") and name.endswith("__"):
            continue
        if isinstance(obj, (property, staticmethod, classmethod)):
            raise TypeError(
                f"{module.__name__}.{name} is a bare "
                f"{type(obj).__name__} object; a routed member must be a plain "
                f"function (def), not a decorator-applied object -- the builtin "
                f"decorator is not introspectable as a routable def and would be "
                f"silently dropped"
            )
        if not inspect.isfunction(obj):
            continue
        if obj.__module__ != module.__name__:
            # Imported from elsewhere (a helper re-exported into this module) --
            # not this module's own member, so not this module's route.
            continue
        yield name, obj


def _build_routes(modules: tuple[ModuleType, ...]) -> dict[str, ModuleType]:
    """Derive a ``name -> owning module`` route table from ``modules``.

    Raises ``ValueError`` on a cross-module name collision: two destination
    modules declaring the same routable name would otherwise silently
    last-writer-wins and drop one route. The table is derived purely by
    introspecting each module's own ``def``s (CLAUDE.md rule 9: no hand-typed
    member list anywhere).
    """
    routes: dict[str, ModuleType] = {}
    for module in modules:
        for name, _fn in _routable_defs(module):
            existing = routes.get(name)
            if existing is not None:
                raise ValueError(
                    f"delegate routing collision: {name!r} is declared by both "
                    f"{existing.__name__!r} and {module.__name__!r}; each routed "
                    f"member must belong to exactly one destination module "
                    f"(last-writer-wins would silently drop one route otherwise)"
                )
            routes[name] = module
    return routes


def _reserved_init_attrs(owner_cls: type) -> frozenset[str]:
    """Return the instance-attribute names ``owner_cls.__init__`` assigns to self.

    Derived by AST-parsing ``inspect.getsource(owner_cls.__init__)`` and
    collecting every ``self.<name> = ...`` target: plain ``Assign``, annotated
    ``AnnAssign``, augmented ``AugAssign``, and names bound through tuple/list
    unpacking (``self.a, self.b = ...``). The self-parameter name is read from
    the ``__init__`` signature rather than assumed, and the whole body is walked
    so assignments nested in ``if`` / ``for`` / ``with`` / ``try`` are seen too.
    No hand-typed name list is used anywhere (CLAUDE.md rule 9): the set is
    entirely a function of the owner's own source.

    A delegate whose name is in this set must be rejected: a plain function
    installed as a class attribute is a non-data descriptor, so an instance
    attribute of the same name set in ``__init__`` wins at lookup and permanently
    masks the delegate (surfacing as a silent wrong-value bug, not an error).

    Limit: only *statically* visible ``self.<name> = ...`` assignments are
    derivable. Dynamic forms (``setattr(self, name, ...)``,
    ``object.__setattr__(self, name, ...)``) are intentionally not caught. If
    ``__init__`` is inherited from ``object`` or its source is unavailable, the
    reserved set is empty (nothing to check).
    """
    init = owner_cls.__init__
    if init is object.__init__:
        return frozenset()
    try:
        source = textwrap.dedent(inspect.getsource(init))
    except (OSError, TypeError):
        # No retrievable source (C-level, exec'd, or stripped) -- treat as empty.
        return frozenset()
    func = next(
        (
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "__init__"
        ),
        None,
    )
    if func is None:
        return frozenset()
    self_name = func.args.args[0].arg if func.args.args else "self"
    reserved: set[str] = set()

    def _collect(target: ast.expr) -> None:
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == self_name
        ):
            reserved.add(target.attr)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                _collect(elt)
        elif isinstance(target, ast.Starred):
            _collect(target.value)

    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                _collect(tgt)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            _collect(node.target)
    return frozenset(reserved)


def _install_delegates(owner_cls: type, modules: tuple[ModuleType, ...]) -> None:
    """Install one class attribute per route onto ``owner_cls``.

    Raises ``ValueError`` if a routed name already exists on ``owner_cls`` as
    something this installer did not put there -- a genuine lexical ``def`` the
    move-PR failed to remove -- or collides with an instance attribute the owner's
    ``__init__`` assigns (``_reserved_init_attrs``): such a delegate would install
    as a class attribute but be permanently masked by the instance attribute.

    Re-installing a name this installer already owns is a genuine no-op: the loop
    ``continue``s before any ``setattr``, so a second run never overwrites
    whatever currently sits on the class (including an active
    ``mock.patch.object``). This is safe because ``pkgutil`` / ``import_module``
    return cached modules, so a repeated run sees the identical function objects
    it already installed; an ``importlib.reload(workflow)`` instead builds a NEW
    class object whose ``_INSTALLED_MARKER`` set is empty, so it installs fresh
    rather than colliding. At L00 ``modules`` is empty, so the route table is
    empty and this is a pure no-op (no ``__init__`` is even parsed).
    """
    routes = _build_routes(modules)
    reserved = _reserved_init_attrs(owner_cls) if routes else frozenset()
    installed: set[str] = set(owner_cls.__dict__.get(_INSTALLED_MARKER, frozenset()))
    for name, module in routes.items():
        if name in installed:
            # Genuine no-op: never re-setattr a name this installer already owns,
            # so a reinstall cannot clobber the current class attribute.
            continue
        if name in reserved:
            raise ValueError(
                f"delegate {name!r} (from {module.__name__!r}) collides with the "
                f"instance attribute 'self.{name}' assigned in "
                f"{owner_cls.__name__}.__init__; a class-level function of that "
                f"name installs but is permanently masked by the instance "
                f"attribute at runtime -- rename the moved member or the attribute"
            )
        if name in owner_cls.__dict__:
            raise ValueError(
                f"delegate {name!r} (from {module.__name__!r}) would shadow an "
                f"existing member already defined on {owner_cls.__name__}; a leaf "
                f"that moves a body must remove the lexical def from the class in "
                f"the same PR before the installer can attach the delegate"
            )
        setattr(owner_cls, name, getattr(module, name))
        installed.add(name)
    setattr(owner_cls, _INSTALLED_MARKER, frozenset(installed))


def discover_delegate_modules(package: ModuleType) -> tuple[ModuleType, ...]:
    """Import and return every direct submodule of ``package``, sorted by name.

    Derives the destination-module list by walking the package's own
    ``__path__`` with ``pkgutil`` (CLAUDE.md rule 9: later leaves add a submodule
    to ``charlie_work.orchestration`` without editing the install line in
    ``workflow.py``). At L00 the package has no submodules, so this returns an
    empty tuple. Sub-packages are skipped -- destinations are flat modules.

    Fail-loud by design: because this runs during ``import charlie_work.workflow``,
    an import or syntax error in any ``charlie_work.orchestration`` submodule
    propagates out of ``importlib.import_module`` here and fails importing
    ``workflow`` at import time, rather than silently dropping the broken module.
    """
    modules: list[ModuleType] = []
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda i: i.name):
        if info.ispkg:
            continue
        modules.append(importlib.import_module(f"{package.__name__}.{info.name}"))
    return tuple(modules)
