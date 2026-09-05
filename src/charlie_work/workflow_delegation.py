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

import importlib
import inspect
import pkgutil
from collections.abc import Iterator
from types import ModuleType
from typing import Any, Callable

# Attribute a destination function may carry to request adapter wrapping at
# install time (the L09 property/staticmethod leaf, design Section 3.3). Set it
# with the ``as_property`` / ``as_staticmethod`` decorators below rather than by
# hand. A function with no such attribute installs as a plain method.
DELEGATE_ADAPTER_ATTR = "__delegate_adapter__"

# Records, per owner class, the set of names this installer has attached. Lets a
# repeated ``_install_delegates`` call be idempotent (re-attaching a name it
# already owns is a no-op) while still raising on a genuine lexical shadow -- a
# name present on the class that this installer did NOT put there. Stored via
# ``setattr`` after class creation, so it is a plain class attribute (an
# ``Assign``), never a ``FunctionDef``: it does not change the APC member count.
_INSTALLED_MARKER = "__delegate_installed_names__"

_ADAPTER_PROPERTY = "property"
_ADAPTER_STATICMETHOD = "staticmethod"


def as_property(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a destination function to be installed as a ``property`` (L09).

    Returns the function unchanged except for the marker attribute, so the AST
    the equivalence gate hashes is still a plain ``FunctionDef`` and the body
    moves verbatim. The wrapping into ``property(fn)`` happens in
    ``_install_delegates``, keeping the destination module free of any
    ``OrchestratorApp`` coupling.
    """
    setattr(fn, DELEGATE_ADAPTER_ATTR, _ADAPTER_PROPERTY)
    return fn


def as_staticmethod(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a destination function to be installed as a ``staticmethod`` (L09)."""
    setattr(fn, DELEGATE_ADAPTER_ATTR, _ADAPTER_STATICMETHOD)
    return fn


def _routable_defs(module: ModuleType) -> Iterator[tuple[str, Callable[..., Any]]]:
    """Yield ``(name, function)`` for every top-level ``def`` ``module`` declares.

    Selection mirrors ``github_delegation._routable_members`` (design Section
    3.2): non-dunder names only, single-underscore private names included (moved
    bodies call each other by name via ``self._name(...)`` and the owner has no
    ``__getattr__`` fallback), and names *imported* into the module excluded --
    a routed function must be one this module actually defines, identified by
    ``obj.__module__ == module.__name__`` rather than by any name list.

    A top-level ``property`` / ``staticmethod`` / ``classmethod`` *object* is a
    hard error, not a silent skip: it is a member a leaf author meant to route
    but reached for the builtin decorator instead of the ``as_property`` /
    ``as_staticmethod`` marker. Silently dropping it would surface late as an
    ``AttributeError`` at ``app.<name>`` (a design Section 9 stop condition), so
    it is caught here at install time with an actionable message.
    """
    for name, obj in vars(module).items():
        if name.startswith("__") and name.endswith("__"):
            continue
        if isinstance(obj, (property, staticmethod, classmethod)):
            raise TypeError(
                f"{module.__name__}.{name} is a bare "
                f"{type(obj).__name__} object; a routed adapter member must be a "
                f"plain function tagged with as_property/as_staticmethod (the "
                f"builtin decorator is not introspectable as a routable def)"
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


def _adapt(fn: Callable[..., Any]) -> Any:
    """Wrap ``fn`` per its adapter marker for class installation.

    No marker -> the plain function (installed unwrapped, binds as a method).
    ``property`` / ``staticmethod`` markers -> the corresponding descriptor, so
    the member drops from the APC ``FunctionDef`` count while ``app.<name>`` /
    ``Owner.<name>(...)`` keep working (design Section 3.3).
    """
    adapter = getattr(fn, DELEGATE_ADAPTER_ATTR, None)
    if adapter == _ADAPTER_PROPERTY:
        return property(fn)
    if adapter == _ADAPTER_STATICMETHOD:
        return staticmethod(fn)
    return fn


def _install_delegates(owner_cls: type, modules: tuple[ModuleType, ...]) -> None:
    """Install one class attribute per route onto ``owner_cls``.

    Raises ``ValueError`` if a routed name already exists on ``owner_cls`` as
    something this installer did not put there -- a genuine lexical ``def`` the
    move-PR failed to remove. Re-installing a name this installer already owns is
    an idempotent no-op (safe under module re-import / a direct second call), not
    a raise: the first-install marker (``_INSTALLED_MARKER``) distinguishes the
    two. At L00 ``modules`` is empty, so the route table is empty and this is a
    pure no-op.
    """
    routes = _build_routes(modules)
    installed: set[str] = set(owner_cls.__dict__.get(_INSTALLED_MARKER, frozenset()))
    for name, module in routes.items():
        if name in owner_cls.__dict__ and name not in installed:
            raise ValueError(
                f"delegate {name!r} (from {module.__name__!r}) would shadow an "
                f"existing member already defined on {owner_cls.__name__}; a leaf "
                f"that moves a body must remove the lexical def from the class in "
                f"the same PR before the installer can attach the delegate"
            )
        setattr(owner_cls, name, _adapt(getattr(module, name)))
        installed.add(name)
    setattr(owner_cls, _INSTALLED_MARKER, frozenset(installed))


def discover_delegate_modules(package: ModuleType) -> tuple[ModuleType, ...]:
    """Import and return every direct submodule of ``package``, sorted by name.

    Derives the destination-module list by walking the package's own
    ``__path__`` with ``pkgutil`` (CLAUDE.md rule 9: later leaves add a submodule
    to ``charlie_work.orchestration`` without editing the install line in
    ``workflow.py``). At L00 the package has no submodules, so this returns an
    empty tuple. Sub-packages are skipped -- destinations are flat modules.
    """
    modules: list[ModuleType] = []
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda i: i.name):
        if info.ispkg:
            continue
        modules.append(importlib.import_module(f"{package.__name__}.{info.name}"))
    return tuple(modules)
