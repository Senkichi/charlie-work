"""Tests for the Track 2 Phase B delegation installer (issue #1631, L00).

L00 moves zero members: it lands ``charlie_work.workflow_delegation`` (the
derived installer) and the empty ``charlie_work.orchestration`` package. These
tests exercise the installer's route derivation, collision/shadow errors, the
method-binding and patch-seam guarantees a moved body will rely on, the
bare-decorator-object guard, and the L00 no-op invariants
(``OrchestratorApp`` still has 133 lexical defs; the package is empty and
workflow-/gh-free).

The ``property`` / ``staticmethod`` adapter mechanism (``as_property`` /
``as_staticmethod`` markers + ``_adapt`` wrapper) is deferred to the L09 leaf
(design Section 3.3), the first leaf to move a ``@property`` / ``@staticmethod``
member, so it has no tests here. L00 installs every routed function as a plain
unwrapped ``def``.

Route/binding behavior is tested against synthetic modules built with
``types.ModuleType`` -- never real domain code -- so the tests describe the
installer's contract independent of which members any later leaf moves.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

import charlie_work.orchestration as _orchestration
import charlie_work.workflow_delegation as wd
from charlie_work.attachment_contracts.archetypes import scan_source

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW_PY = _REPO_ROOT / "src" / "charlie_work" / "workflow.py"


def _make_module(name: str, **defs: object) -> types.ModuleType:
    """Build a synthetic module whose ``defs`` are treated as its own members.

    Every callable passed in has its ``__module__`` rebound to ``name`` so
    ``_routable_defs``' "defined here, not imported" predicate
    (``obj.__module__ == module.__name__``) accepts it -- exactly how a real
    ``def`` in that module would present.
    """
    module = types.ModuleType(name)
    for attr, value in defs.items():
        if callable(value) and hasattr(value, "__module__"):
            try:
                value.__module__ = name
            except (AttributeError, TypeError):
                pass
        setattr(module, attr, value)
    return module


# --------------------------------------------------------------------------
# _routable_defs
# --------------------------------------------------------------------------


def test_routable_defs_positive_control_excludes_dunder_and_imported() -> None:
    """A module with one private def, one public def, one dunder, and one
    imported function yields exactly the two defined-here functions.
    """

    def _private(self) -> str:  # noqa: ARG001
        return "private"

    def public(self) -> str:  # noqa: ARG001
        return "public"

    def __dunderish__(self) -> str:  # noqa: ARG001, N807
        return "dunder"

    def imported(self) -> str:  # noqa: ARG001
        return "imported"

    module = _make_module(
        "synthetic_routable",
        _private=_private,
        public=public,
        __dunderish__=__dunderish__,
    )
    # An imported function keeps its *original* __module__ (a different module),
    # so it must be filtered out.
    imported.__module__ = "some.other.module"
    module.imported = imported

    names = {name for name, _fn in wd._routable_defs(module)}
    assert names == {"_private", "public"}


def test_routable_defs_bare_property_object_raises() -> None:
    """A top-level builtin ``property`` object is a hard error: it is not
    introspectable as a routable ``def`` and would be silently dropped (a
    design Section 9 stop condition). The adapter markers that would make a
    property routable land at L09, not here.
    """

    def layout(self) -> str:  # noqa: ARG001
        return "L"

    module = _make_module("synthetic_bad_property")
    module.layout = property(layout)  # builtin decorator -> object, not a def

    with pytest.raises(TypeError) as exc_info:
        list(wd._routable_defs(module))
    assert "property" in str(exc_info.value)
    assert "plain function" in str(exc_info.value)


def test_routable_defs_bare_staticmethod_object_raises() -> None:
    def helper() -> str:
        return "S"

    module = _make_module("synthetic_bad_static")
    module.helper = staticmethod(helper)

    with pytest.raises(TypeError) as exc_info:
        list(wd._routable_defs(module))
    assert "staticmethod" in str(exc_info.value)
    assert "plain function" in str(exc_info.value)


# --------------------------------------------------------------------------
# _build_routes
# --------------------------------------------------------------------------


def test_build_routes_maps_name_to_module() -> None:
    def foo(self) -> None:  # noqa: ARG001
        ...

    def bar(self) -> None:  # noqa: ARG001
        ...

    mod_a = _make_module("synthetic_a", foo=foo)
    mod_b = _make_module("synthetic_b", bar=bar)

    routes = wd._build_routes((mod_a, mod_b))
    assert routes == {"foo": mod_a, "bar": mod_b}


def test_build_routes_collision_raises() -> None:
    def dup(self) -> None:  # noqa: ARG001
        ...

    def dup2(self) -> None:  # noqa: ARG001
        ...

    mod_a = _make_module("synthetic_c", dup=dup)
    mod_b = _make_module("synthetic_d", dup=dup2)

    with pytest.raises(ValueError) as exc_info:
        wd._build_routes((mod_a, mod_b))
    assert "dup" in str(exc_info.value)


def test_build_routes_positive_control_disjoint() -> None:
    """The positive control proving the collision test isn't just detecting a
    broken registry: disjoint names build the expected route map."""

    def alpha(self) -> None:  # noqa: ARG001
        ...

    def beta(self) -> None:  # noqa: ARG001
        ...

    mod_a = _make_module("synthetic_e", alpha=alpha)
    mod_b = _make_module("synthetic_f", beta=beta)

    routes = wd._build_routes((mod_a, mod_b))
    assert set(routes) == {"alpha", "beta"}


# --------------------------------------------------------------------------
# _install_delegates: shadow, binding, idempotency
# --------------------------------------------------------------------------


def test_install_delegates_shadow_existing_def_raises() -> None:
    """A routed name that already exists as a real def on the owner must raise,
    so a leaf cannot silently shadow (and setattr-overwrite) an unmoved member.
    """

    def existing(self) -> str:  # noqa: ARG001
        return "module-body"

    module = _make_module("synthetic_shadow", existing=existing)

    class _Owner:
        def existing(self) -> str:
            return "class-body"

    with pytest.raises(ValueError) as exc_info:
        wd._install_delegates(_Owner, (module,))
    assert "existing" in str(exc_info.value)
    assert _Owner().existing() == "class-body"  # unchanged


def test_installed_function_binds_as_method_and_receives_instance() -> None:
    def whoami(self) -> object:  # noqa: ARG001
        return self

    module = _make_module("synthetic_bind", whoami=whoami)

    class _Owner:
        pass

    wd._install_delegates(_Owner, (module,))
    inst = _Owner()
    assert inst.whoami() is inst


def test_install_delegates_is_idempotent() -> None:
    """Re-installing a name this installer already owns is a no-op, not a raise
    on its own delegate (module re-import / a direct second call)."""

    def again(self) -> str:  # noqa: ARG001
        return "ok"

    module = _make_module("synthetic_idem", again=again)

    class _Owner:
        pass

    wd._install_delegates(_Owner, (module,))
    wd._install_delegates(_Owner, (module,))  # must not raise
    assert _Owner().again() == "ok"


# --------------------------------------------------------------------------
# patch seams (Tier A / Tier B / subclass) all intercept
# --------------------------------------------------------------------------


def _owner_with_delegate(mod_name: str, fn_name: str, body: object) -> type:
    module = _make_module(mod_name, **{fn_name: body})

    class _Owner:
        pass

    wd._install_delegates(_Owner, (module,))
    return _Owner


def test_tier_a_class_patch_intercepts() -> None:
    def act(self) -> str:  # noqa: ARG001
        return "real"

    owner = _owner_with_delegate("synthetic_tier_a", "act", act)
    inst = owner()
    with mock.patch.object(owner, "act", return_value="patched"):
        assert inst.act() == "patched"
    assert inst.act() == "real"


def test_tier_b_instance_patch_object_intercepts() -> None:
    def act(self) -> str:  # noqa: ARG001
        return "real"

    owner = _owner_with_delegate("synthetic_tier_b1", "act", act)
    inst = owner()
    with mock.patch.object(inst, "act", return_value="patched"):
        assert inst.act() == "patched"
    assert inst.act() == "real"


def test_tier_b_monkeypatch_setattr_intercepts(monkeypatch: pytest.MonkeyPatch) -> None:
    def act(self) -> str:  # noqa: ARG001
        return "real"

    owner = _owner_with_delegate("synthetic_tier_b2", "act", act)
    inst = owner()
    monkeypatch.setattr(inst, "act", lambda: "patched")
    assert inst.act() == "patched"
    monkeypatch.undo()
    assert inst.act() == "real"


def test_subclass_override_intercepts() -> None:
    def act(self) -> str:  # noqa: ARG001
        return "base"

    base = _owner_with_delegate("synthetic_subclass", "act", act)

    class _Sub(base):  # type: ignore[valid-type, misc]
        def act(self) -> str:
            return "sub"

    assert _Sub().act() == "sub"
    assert base().act() == "base"


# --------------------------------------------------------------------------
# Adapter deferral: the property/staticmethod marker surface is NOT landed at L00
# --------------------------------------------------------------------------


def test_adapter_markers_deferred_to_l09() -> None:
    """The ``as_property`` / ``as_staticmethod`` decorators, the
    ``DELEGATE_ADAPTER_ATTR`` marker constant, and the ``_adapt`` wrapper are
    scoped to the L09 leaf (design Section 3.3) -- the first leaf with a real
    ``@property`` / ``@staticmethod`` consumer. L00 moves zero members, so
    landing them here would mean new public, tested functions with zero
    production callers (PR #1641 review finding). They must be absent from the
    module until L09 reintroduces them alongside that consumer.
    """
    for absent in ("as_property", "as_staticmethod", "_adapt", "DELEGATE_ADAPTER_ATTR"):
        assert not hasattr(wd, absent), (
            f"{absent!r} is present on workflow_delegation; the adapter surface "
            f"was deferred to L09 (design Section 3.3) and must not land at L00"
        )


# --------------------------------------------------------------------------
# APC class counter: assignments count 0, defs count N
# --------------------------------------------------------------------------


def test_apc_counter_zero_for_assignment_only_class(tmp_path: Path) -> None:
    """A class whose only attributes are assignments reports 0 members -- the
    structural reason installing delegates (class-level Assign) does not raise
    OrchestratorApp's member_count."""
    src = "class Sample:\n    alpha = 1\n    beta = 2\n    gamma = 3\n"
    f = tmp_path / "assign_only.py"
    f.write_text(src, encoding="utf-8")
    points = scan_source(f.read_text(encoding="utf-8"), "src/charlie_work/assign_only.py")
    cls = next(p for p in points if p.kind == "class" and p.identity == "Sample")
    assert cls.member_count == 0


def test_apc_counter_n_for_def_class(tmp_path: Path) -> None:
    """The same names declared as ``def``s report N members -- the positive
    control for the assignment-counts-zero test above."""
    src = (
        "class Sample:\n"
        "    def alpha(self): ...\n"
        "    def beta(self): ...\n"
        "    def gamma(self): ...\n"
    )
    f = tmp_path / "def_class.py"
    f.write_text(src, encoding="utf-8")
    points = scan_source(f.read_text(encoding="utf-8"), "src/charlie_work/def_class.py")
    cls = next(p for p in points if p.kind == "class" and p.identity == "Sample")
    assert cls.member_count == 3


# --------------------------------------------------------------------------
# L00 no-op invariants
# --------------------------------------------------------------------------


def test_orchestratorapp_still_has_133_lexical_defs() -> None:
    tree = ast.parse(_WORKFLOW_PY.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "OrchestratorApp"
    )
    defs = sum(1 for c in cls.body if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)))
    assert defs == 133


def test_delegate_modules_empty_at_l00_and_install_is_noop() -> None:
    """The real orchestration package has no submodules at L00, so discovery
    yields an empty tuple and installing it changes nothing."""
    assert wd.discover_delegate_modules(_orchestration) == ()

    class _Owner:
        pass

    before = set(_Owner.__dict__)
    wd._install_delegates(_Owner, ())
    # Only the bookkeeping marker is added; no delegate members.
    added = set(_Owner.__dict__) - before
    assert added == {wd._INSTALLED_MARKER}


def test_discover_positive_control_synthetic_package(tmp_path: Path) -> None:
    """The positive control for the empty-at-L00 assertion: a package with one
    real submodule yields one module and one route."""
    pkg_dir = tmp_path / "synthpkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    (pkg_dir / "leaf.py").write_text("def _do_thing(self):\n    return 'done'\n", encoding="utf-8")

    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    created: list[str] = []
    try:
        pkg = importlib.import_module("synthpkg")
        created.append("synthpkg")
        modules = wd.discover_delegate_modules(pkg)
        created.extend(m.__name__ for m in modules)
        assert len(modules) == 1
        routes = wd._build_routes(modules)
        assert set(routes) == {"_do_thing"}
    finally:
        sys.path.remove(str(tmp_path))
        for name in created:
            sys.modules.pop(name, None)
        importlib.invalidate_caches()


def test_workflow_cold_import_succeeds_in_subprocess() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import charlie_work.workflow"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_orchestration_is_workflow_free_and_gh_free_in_subprocess() -> None:
    """Importing the destination package must not pull in workflow or github --
    the #1627 cycle-safety and namespace-isolation property."""
    code = (
        "import sys, charlie_work.orchestration\n"
        "assert 'charlie_work.workflow' not in sys.modules, 'workflow imported'\n"
        "assert 'charlie_work.github' not in sys.modules, 'github imported'\n"
        "print('clean')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout
