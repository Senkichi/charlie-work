"""`charlie_work.cli` must import without ci_fleet's shadow/rollback cluster.

Issue #929. ``ci_fleet``'s ``diff_journal`` / ``shadow_gate`` / ``shadow_pass``
serve exactly one command -- ``charlie runners shadow-status`` -- and ci_fleet is
retiring them as the legacy-planner rollback path closes. Imported at module
scope they took down the entire CLI, including ``charlie runners allocate`` and
``fleet supervise``, whose entry points import :mod:`charlie_work.cli`.

That is an observed failure, not a hypothetical: on 2026-08-05 ci_fleet's working
tree carried a deletion of ``shadow_pass.py`` and ``main`` collected 19 import
errors.

**Why this needs a subprocess and a meta-path blocker rather than monkeypatching
``sys.modules``.** The regression is about *import time* -- whether the module can
be imported at all -- so it has to be observed in a process where
:mod:`charlie_work.cli` has not already been imported. By the time a test function
runs, pytest has long since imported it, and deleting the entry from ``sys.modules``
would only re-execute it with the real ci_fleet still installed and importable.
The blocker also reproduces the true failure shape: the module is *absent*, not
merely un-imported.
"""

from __future__ import annotations

import ast
import functools
import pathlib
import subprocess
import sys
import textwrap

import pytest

# The three modules ci_fleet is retiring. Blocked as a set rather than only the
# one that was actually deleted, because the point of the fix is that *none* of
# them may be reachable from module scope -- testing only shadow_pass would keep
# passing if someone re-hoisted the other two.
_RETIRING = ("ci_fleet.diff_journal", "ci_fleet.shadow_gate", "ci_fleet.shadow_pass")

_BLOCKER = '''
import sys

BLOCKED = {"ci_fleet.diff_journal", "ci_fleet.shadow_gate", "ci_fleet.shadow_pass"}


class _Blocker:
    """Make the retiring modules look deleted, not merely unimported."""

    def find_spec(self, name, path=None, target=None):
        if name in BLOCKED:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None


sys.meta_path.insert(0, _Blocker())
'''


def _run(body: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        [sys.executable, "-c", _BLOCKER + textwrap.dedent(body)],
        capture_output=True,
        text=True,
    )


def _run_unblocked(body: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
    )


def _module_scope_imports(path: pathlib.Path) -> set[str]:
    """Every module name imported at module scope, ignoring function/class bodies.

    A module-scope ``try:``/``if:`` still executes at import, so this descends into
    those; it stops only at ``def``/``class``, which is exactly the boundary the fix
    moved the imports across.
    """
    names: set[str] = set()

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(ast.parse(path.read_text(encoding="utf-8")))
    return names


def test_cli_has_no_module_scope_shadow_import() -> None:
    """The invariant, asserted statically so no ci_fleet tree state can hide it.

    This is the guard that actually stops a regression. The runtime tests below are
    stronger evidence -- they prove real behaviour rather than a source property --
    but they can only run when ci_fleet's own tree permits the cluster to be absent
    (see :func:`_shadow_cluster_is_severable`). This one holds unconditionally,
    including on a ci_fleet tree where the cluster is load-bearing, which is
    precisely when the runtime tests go quiet.

    Locating ``cli.py`` through ``charlie_work.__path__`` rather than importing
    ``charlie_work.cli`` is deliberate: ``charlie_work/__init__.py`` declares no
    imports, so this test stays answerable even in the broken state it guards
    against.
    """
    import charlie_work

    cli_py = pathlib.Path(charlie_work.__path__[0]) / "cli.py"
    assert cli_py.is_file(), f"expected {cli_py} to exist"

    offenders = sorted(
        name
        for name in _module_scope_imports(cli_py)
        if any(name == mod or name.startswith(f"{mod}.") for mod in _RETIRING)
    )

    assert not offenders, (
        "cli.py imports ci_fleet's retiring shadow cluster at module scope: "
        f"{offenders}. Issue #929 -- these must live inside "
        "run_runners_shadow_status so that retiring one of them breaks only "
        "`charlie runners shadow-status`, not `charlie runners allocate` and "
        "`fleet supervise`, whose entry points import this module."
    )


@functools.lru_cache(maxsize=1)
def _shadow_cluster_is_severable() -> str:
    """Can this environment even observe the confinement? Returns '' if yes.

    The runtime tests block the shadow cluster and then assert that the rest of the
    CLI survives. That question is only meaningful if the cluster is severable *in
    ci_fleet's own tree*. On ci_fleet's committed ``main`` it is not:
    ``charlie_work_adapter`` imports ``runner_allocation_pass``, which imports
    ``shadow_pass``. Blocking the cluster there breaks the adapter -- which every
    ``charlie_work`` module imports -- so every runtime test below fails for a
    reason that has nothing to do with our import site.

    That is not a hypothetical either: it is why this file went red on PR #930's
    first CI run while passing locally, where ci_fleet's working tree had already
    severed it.

    Skipping rather than failing is the honest outcome, and it is not a silent hole:
    :func:`test_cli_has_no_module_scope_shadow_import` above covers the regression
    unconditionally, and this returns to life on its own once ci_fleet lands the
    severing change on their main.
    """
    probe = _run(
        """
        import ci_fleet.charlie_work_adapter
        print("ADAPTER_OK")
        """
    )
    if probe.returncode == 0 and "ADAPTER_OK" in probe.stdout:
        return ""
    return (
        "ci_fleet's installed tree couples the shadow cluster into "
        "charlie_work_adapter (adapter -> runner_allocation_pass -> shadow_pass), "
        "so the cluster cannot be made absent without breaking the adapter, and "
        "confinement is not observable here. Static coverage in "
        "test_cli_has_no_module_scope_shadow_import still applies. "
        f"Adapter probe stderr:\n{probe.stderr.strip()[-400:]}"
    )


def _require_severable_cluster() -> None:
    if reason := _shadow_cluster_is_severable():
        pytest.skip(reason)


def test_the_blocker_actually_blocks() -> None:
    """Positive control, written as a differential rather than a single result.

    A one-sided assertion here is worthless: at the time this was written,
    ``shadow_pass`` was *genuinely* deleted in the live ci_fleet working tree, so
    "importing it fails" would have held with the blocker removed entirely. The
    control therefore has to show the same import *succeeding* without the blocker
    and *failing* with it. ``diff_journal`` is used because it is the module the
    retirement had not yet touched.

    If the unblocked leg ever fails, the message is the finding: ci_fleet has
    retired the cluster, and per #929 the right response is to delete
    ``shadow-status`` and this file, not to repair either.
    """
    control = _run_unblocked(
        """
        import ci_fleet.diff_journal
        print("PRESENT")
        """
    )
    assert control.returncode == 0, (
        "fixture premise gone: ci_fleet.diff_journal is not importable even without "
        f"the blocker, so this file can no longer prove anything.\n{control.stderr}"
    )
    assert "PRESENT" in control.stdout

    blocked = _run(
        """
        import ci_fleet.diff_journal
        """
    )

    assert blocked.returncode != 0, "blocker did not block; the other tests are vacuous"
    assert "diff_journal" in blocked.stderr


def test_ci_fleet_itself_still_imports_under_the_blocker() -> None:
    """The blocker must remove only the shadow cluster.

    If blocking these three also broke ``import ci_fleet`` or the adapter, the
    main test below would pass or fail for a reason unrelated to #929.
    """
    _require_severable_cluster()
    result = _run(
        """
        import ci_fleet
        import ci_fleet.charlie_work_adapter
        print("ADAPTER_OK")
        """
    )

    assert result.returncode == 0, result.stderr
    assert "ADAPTER_OK" in result.stdout


def test_cli_imports_without_the_shadow_cluster() -> None:
    """The regression itself: a retired shadow module must not break the CLI."""
    _require_severable_cluster()
    result = _run(
        """
        import charlie_work.cli
        print("IMPORT_OK")
        """
    )

    assert result.returncode == 0, result.stderr
    assert "IMPORT_OK" in result.stdout


def test_the_parser_still_builds_without_the_shadow_cluster() -> None:
    """Importing is necessary but not sufficient.

    ``charlie runners allocate`` is only reachable if the argument parser can be
    constructed, which touches every subcommand's registration -- including
    ``shadow-status``. A fix that moved the imports but left a module-scope
    reference in the parser wiring would pass the import test and still break the
    fleet's critical path.
    """
    _require_severable_cluster()
    result = _run(
        """
        import charlie_work.cli as cli

        parser = cli.build_parser()
        args = parser.parse_args(["runners", "allocate", "--dry-run"])
        assert args is not None
        print("PARSER_OK")
        """
    )

    assert result.returncode == 0, result.stderr
    assert "PARSER_OK" in result.stdout


def test_shadow_status_is_the_only_casualty() -> None:
    """Confinement, stated as an assertion rather than left implied.

    The command that reports on the retired cluster is *supposed* to fail, and to
    fail loudly enough that the traceback names the missing module. Binding the
    names to ``None`` behind a ``try/except ImportError`` would satisfy the tests
    above while turning this into an ``AttributeError`` with the cause erased.
    """
    _require_severable_cluster()
    result = _run(
        """
        import argparse
        import charlie_work.cli as cli

        try:
            cli.run_runners_shadow_status(argparse.Namespace())
        except ModuleNotFoundError as exc:
            print("CONFINED:", exc.name)
        else:
            raise AssertionError("expected ModuleNotFoundError from the lazy import")
        """
    )

    assert result.returncode == 0, result.stderr
    assert "CONFINED: ci_fleet." in result.stdout
