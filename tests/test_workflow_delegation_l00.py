"""Tests for the Track 2 Phase B delegation installer (issue #1631, L00).

L00 moves zero members: it lands ``charlie_work.workflow_delegation`` (the
derived installer) and the empty ``charlie_work.orchestration`` package. These
tests exercise the installer's route derivation, collision/shadow errors, the
method-binding and patch-seam guarantees a moved body will rely on, the
bare-decorator-object guard, and the member-surface conservation invariants
that hold across every Phase B leaf: the lexical defs still in ``workflow.py``
plus the installer's delegates always sum to the pre-campaign
``OrchestratorApp`` surface of 133; discovery matches the package's real
pkgutil submodule set; the package stays workflow-/gh-free. Later leaves move
members into ``charlie_work.orchestration`` and the derived assertions below
follow the members wherever they land, so no assertion is keyed to a fixed
member list.

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
import inspect
import json
import pkgutil
import subprocess
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

import charlie_work.orchestration as _orchestration
import charlie_work.workflow_delegation as wd
from charlie_work.attachment_contracts.archetypes import scan_source
from charlie_work.workflow import OrchestratorApp

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW_PY = _REPO_ROOT / "src" / "charlie_work" / "workflow.py"
_BUDGETS_JSON = _REPO_ROOT / ".attachment-budgets.json"

# The OrchestratorApp member surface before Track 2 Phase B began moving bodies
# out of workflow.py. Every leaf relocates lexical ``def``s into
# charlie_work.orchestration submodules and the installer re-attaches them as
# class attributes, so this total -- lexical defs remaining in workflow.py PLUS
# installed delegates -- is conserved across the whole campaign. A single
# documented constant, never a member-name list (CLAUDE.md rule 9).
_PRE_CAMPAIGN_MEMBER_SURFACE = 133


def _committed_orchestratorapp_member_count() -> int:
    """The APC-budgeted ``member_count`` for ``workflow.py::OrchestratorApp``,
    read from the committed ``.attachment-budgets.json``.

    The budget ceiling is keyed by that file path and derived from the class's
    lexical source, so it is the ground truth for "how many defs still live in
    workflow.py" -- the one guard here that is legitimately file-keyed.
    """
    data = json.loads(_BUDGETS_JSON.read_text(encoding="utf-8"))
    entries = data["entries"] if isinstance(data, dict) else data
    matches = [
        e
        for e in entries
        if e.get("file") == "src/charlie_work/workflow.py"
        and e.get("identity") == "OrchestratorApp"
        and e.get("kind") == "class"
    ]
    assert len(matches) == 1, (
        f"expected exactly one OrchestratorApp class budget entry, got {len(matches)}"
    )
    return int(matches[0]["member_count"])


def _installed_delegate_names() -> frozenset[str]:
    """The names the installer attached onto OrchestratorApp, read from the
    class's own bookkeeping marker (never a hard-coded list)."""
    return frozenset(vars(OrchestratorApp).get(wd._INSTALLED_MARKER, frozenset()))


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


def test_reinstall_during_active_patch_preserves_patch() -> None:
    """Re-running the installer while a class-level ``mock.patch.object`` is
    active must NOT clobber the patch: the already-installed name is a genuine
    no-op (``continue`` before ``setattr``), and after the context exits the
    original delegate is restored. Exact reproduction from the PR #1641 review,
    inverted to assert the fix (finding [1])."""

    def act(self) -> str:  # noqa: ARG001
        return "real"

    module = _make_module("synthetic_reinstall_patch", act=act)

    class _Owner:
        pass

    wd._install_delegates(_Owner, (module,))
    with mock.patch.object(_Owner, "act", return_value="patched"):
        assert _Owner().act() == "patched"
        wd._install_delegates(_Owner, (module,))  # reinstall must not clobber
        assert _Owner().act() == "patched"  # patch still in place
    assert _Owner().act() == "real"  # original delegate restored on ctx exit


def test_install_delegates_reserved_instance_attr_raises() -> None:
    """A routed name matching a ``self.<name> = ...`` assignment in the owner's
    __init__ raises: the class attribute would install but be permanently masked
    by the instance attribute (PR #1641 finding [2]). A non-colliding name on the
    same owner is the positive control -- it installs fine."""

    def status(self) -> str:  # noqa: ARG001
        return "delegated"

    def other(self) -> str:  # noqa: ARG001
        return "delegated"

    class _Owner:
        def __init__(self) -> None:
            self.status = "instance-data"

    colliding = _make_module("synthetic_reserved", status=status)
    with pytest.raises(ValueError) as exc_info:
        wd._install_delegates(_Owner, (colliding,))
    assert "status" in str(exc_info.value)

    ok = _make_module("synthetic_reserved_ok", other=other)
    wd._install_delegates(_Owner, (ok,))  # non-colliding name installs
    assert _Owner().other() == "delegated"


def test_install_delegates_object_init_owner_installs() -> None:
    """An owner whose __init__ is inherited from ``object`` has an empty reserved
    set, so a routed delegate installs with no false collision (PR #1641 finding
    [2] boundary case)."""

    def act(self) -> str:  # noqa: ARG001
        return "ok"

    module = _make_module("synthetic_object_init", act=act)

    class _Owner:
        pass

    wd._install_delegates(_Owner, (module,))
    assert _Owner().act() == "ok"


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
# Member-surface conservation invariants (hold across every Phase B leaf)
# --------------------------------------------------------------------------


def test_orchestratorapp_member_surface_conserved_lexical_plus_installed() -> None:
    """The OrchestratorApp member surface is conserved as bodies move out.

    Three derived invariants, none keyed to a member-name list:
      1. the lexical FunctionDef/AsyncFunctionDef count still in workflow.py
         equals the committed .attachment-budgets.json member_count for
         workflow.py::OrchestratorApp (the APC ceiling tracks the source);
      2. lexical defs + installed delegates == the pre-campaign surface (133) --
         every def that left the class body is re-attached by the installer, so
         nothing is lost or double-counted;
      3. no installed delegate name is also a lexical def -- a name in both
         places is exactly the shadow the installer raises on, cross-checked
         here as two disjoint sets.
    """
    tree = ast.parse(_WORKFLOW_PY.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "OrchestratorApp"
    )
    lexical_names = {
        c.name for c in cls.body if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    lexical_defs = len(lexical_names)

    assert lexical_defs == _committed_orchestratorapp_member_count()

    installed = _installed_delegate_names()
    assert lexical_defs + len(installed) == _PRE_CAMPAIGN_MEMBER_SURFACE

    shadowed = installed & lexical_names
    assert not shadowed, (
        f"installed delegate name(s) also present as lexical defs on "
        f"OrchestratorApp: {sorted(shadowed)}"
    )


def test_delegate_modules_match_pkgutil_and_all_defs_installed() -> None:
    """Discovery matches the package's real pkgutil submodule set and every
    non-dunder top-level def of every discovered module is installed on
    OrchestratorApp as that exact function object (identity).

    The pkgutil submodule set is enumerated here directly (not via the
    installer's own selection helper) so the assertion is an independent oracle,
    not a self-consistent restatement of the code under test. It is a positive
    control: after L01 the package is non-empty, so an accidental regression to
    an empty package (or a def the installer failed to attach) fails loudly.
    """
    discovered = wd.discover_delegate_modules(_orchestration)
    expected_names = tuple(
        sorted(
            info.name for info in pkgutil.iter_modules(_orchestration.__path__) if not info.ispkg
        )
    )
    assert tuple(m.__name__.rsplit(".", 1)[-1] for m in discovered) == expected_names
    assert discovered, (
        "orchestration package exposes no submodules -- positive control failed; "
        "discovery can no longer prove it attaches anything"
    )

    for module in discovered:
        top_defs = [
            name
            for name, obj in vars(module).items()
            if not (name.startswith("__") and name.endswith("__"))
            and inspect.isfunction(obj)
            and obj.__module__ == module.__name__
        ]
        assert top_defs, f"{module.__name__} declares no top-level defs to install"
        for name in top_defs:
            assert name in vars(OrchestratorApp), (
                f"{name} from {module.__name__} was not installed on OrchestratorApp"
            )
            assert vars(OrchestratorApp)[name] is getattr(module, name), (
                f"installed {name} is not the same function object as {module.__name__}.{name}"
            )


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
