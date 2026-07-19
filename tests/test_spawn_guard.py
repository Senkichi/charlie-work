"""AST-based guard that every subprocess/os spawn site routes through
``no_console_window_kwargs()`` (leaf spawns) or ``hidden_console_kwargs()``
(worker spawns) (issues #399, #459).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import charlie_work

HELPER_NAMES = {"no_console_window_kwargs", "hidden_console_kwargs"}
ALLOWLIST_RE = re.compile(r"#\s*spawn-guard:\s*allow")

SPAWN_ATTRS: dict[str, set[str]] = {
    "subprocess": {"run", "Popen", "check_output", "check_call", "call"},
}


def _is_os_spawn(name: str) -> bool:
    return name in {"system", "popen", "startfile"} or name.startswith(("spawn", "posix_spawn"))


def _is_target_attr(module: str, attr: str) -> bool:
    if module == "subprocess":
        return attr in SPAWN_ATTRS["subprocess"]
    if module == "os":
        return _is_os_spawn(attr)
    return False


def _dotted_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        if prefix is not None:
            return f"{prefix}.{node.attr}"
    return None


def _collect_aliases(
    tree: ast.AST,
) -> tuple[dict[str, str], dict[str, tuple[str, str]], set[str], set[str]]:
    """Collect import aliases for subprocess/os modules/functions and the helpers.

    Returns:
        module_aliases: name -> module (e.g. "subprocess" -> "subprocess", "sp" -> "subprocess")
        func_aliases: name -> (module, attr) for from-imports
        helper_names: names bound to ``no_console_window_kwargs`` or ``hidden_console_kwargs``
        helper_modules: names bound to a module whose last component is ``subprocess_runner``
    """
    module_aliases: dict[str, str] = {}
    func_aliases: dict[str, tuple[str, str]] = {}
    helper_names: set[str] = set()
    helper_modules: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname if alias.asname else alias.name
                if alias.name in ("subprocess", "os"):
                    module_aliases[bound] = alias.name
                if alias.name.split(".")[-1] == "subprocess_runner":
                    module_aliases[bound] = alias.name
                    helper_modules.add(bound)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                bound = alias.asname if alias.asname else alias.name
                if module in ("subprocess", "os") and _is_target_attr(module, alias.name):
                    func_aliases[bound] = (module, alias.name)
                if module.endswith("subprocess_runner") and alias.name in HELPER_NAMES:
                    helper_names.add(bound)
                if module == "charlie_work" and alias.name == "subprocess_runner":
                    module_aliases[bound] = "charlie_work.subprocess_runner"
                    helper_modules.add(bound)
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in HELPER_NAMES
        ):
            helper_names.add(node.name)

    return module_aliases, func_aliases, helper_names, helper_modules


def _resolve_spawn_call(
    node: ast.expr,
    module_aliases: dict[str, str],
    func_aliases: dict[str, tuple[str, str]],
) -> tuple[str, str] | None:
    """Resolve a call expression to (module, function_name) if it is a target spawn site."""
    if isinstance(node, ast.Name):
        resolved = func_aliases.get(node.id)
        if resolved and _is_target_attr(resolved[0], resolved[1]):
            return resolved
        return None
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        module = module_aliases.get(node.value.id)
        if module and _is_target_attr(module, node.attr):
            return (module, node.attr)
    return None


def _is_helper_reference(
    node: ast.expr,
    helper_names: set[str],
    helper_modules: set[str],
    module_aliases: dict[str, str],
) -> bool:
    """Return True if ``node`` is a reference to an allowed spawn helper."""
    if isinstance(node, ast.Name):
        return node.id in helper_names
    if isinstance(node, ast.Attribute) and node.attr in HELPER_NAMES:
        full = _dotted_name(node.value)
        if full is not None:
            # Already a dotted module name; check if it points to the helper module.
            if full in helper_modules:
                return True
            if (
                full in module_aliases
                and module_aliases[full].split(".")[-1] == "subprocess_runner"
            ):
                return True
        if isinstance(node.value, ast.Name) and node.value.id in helper_modules:
            return True
    return False


def _contains_helper(
    node: ast.AST,
    helper_names: set[str],
    helper_modules: set[str],
    module_aliases: dict[str, str],
) -> bool:
    """Return True if the AST subtree contains any reference to the helper."""
    return any(
        _is_helper_reference(child, helper_names, helper_modules, module_aliases)
        for child in ast.walk(node)
        if isinstance(child, (ast.Name, ast.Attribute))
    )


class _ParentMap(ast.NodeVisitor):
    def __init__(self) -> None:
        self.parents: dict[ast.AST, ast.AST] = {}

    def visit(self, node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            self.parents[child] = node
            self.visit(child)


def _enclosing_scope(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.AST:
    while node is not None and not isinstance(
        node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    ):
        node = parents.get(node)
    return node


def _statement_uses_name_with_helper(
    stmt: ast.stmt,
    name: str,
    helper_names: set[str],
    helper_modules: set[str],
    module_aliases: dict[str, str],
) -> bool:
    """Check whether ``stmt`` assigns to/updates ``name`` using the helper."""
    if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
        target = stmt.target if isinstance(stmt, ast.AnnAssign) else stmt.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            return _contains_helper(stmt.value, helper_names, helper_modules, module_aliases)
    if (
        isinstance(stmt, ast.AugAssign)
        and isinstance(stmt.target, ast.Name)
        and stmt.target.id == name
    ):
        return _contains_helper(stmt.value, helper_names, helper_modules, module_aliases)
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        call = stmt.value
        if (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == name
            and call.func.attr == "update"
        ):
            return any(
                _contains_helper(arg, helper_names, helper_modules, module_aliases)
                for arg in call.args
            )
    return False


def _dict_var_routes_through_helper(
    name: str,
    call: ast.Call,
    scope: ast.AST,
    parents: dict[ast.AST, ast.AST],
    helper_names: set[str],
    helper_modules: set[str],
    module_aliases: dict[str, str],
) -> bool:
    """Return True if ``name`` (used as ``**name``) is assigned/updated from the helper in the same scope before ``call``."""
    for stmt in ast.walk(scope):
        if not isinstance(stmt, ast.stmt):
            continue
        if stmt is call:
            continue
        if _enclosing_scope(stmt, parents) is not scope:
            continue
        stmt_line = getattr(stmt, "lineno", 0)
        call_line = getattr(call, "lineno", 0)
        if stmt_line >= call_line:
            continue
        if _statement_uses_name_with_helper(
            stmt, name, helper_names, helper_modules, module_aliases
        ):
            return True
    return False


def _call_has_spawn_kwargs(
    call: ast.Call,
    scope: ast.AST,
    parents: dict[ast.AST, ast.AST],
    helper_names: set[str],
    helper_modules: set[str],
    module_aliases: dict[str, str],
) -> bool:
    """Return True if the spawn call is routed through an allowed helper."""
    # Direct helper reference anywhere in the call (e.g. **no_console_window_kwargs(...)).
    if _contains_helper(call, helper_names, helper_modules, module_aliases):
        return True

    # **kwargs where kwargs was assigned/updated from the helper in the same scope.
    for kw in call.keywords:
        if kw.arg is None and isinstance(kw.value, ast.Name):
            if _dict_var_routes_through_helper(
                kw.value.id,
                call,
                scope,
                parents,
                helper_names,
                helper_modules,
                module_aliases,
            ):
                return True
    return False


def _source_package_dir() -> Path:
    installed = Path(charlie_work.__file__).resolve().parent
    if (installed / "subprocess_runner.py").exists():
        return installed
    repo_src = Path(__file__).resolve().parents[1] / "src" / "charlie_work"
    if repo_src.is_dir():
        return repo_src
    raise RuntimeError("Cannot locate charlie_work source directory")


def find_spawn_guard_violations(root: Path) -> list[str]:
    """Statically scan ``root/**/*.py`` and return actionable violation messages."""
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            violations.append(f"{path}:{exc.lineno}: syntax error while scanning")
            continue

        lines = source.splitlines()
        module_aliases, func_aliases, helper_names, helper_modules = _collect_aliases(tree)
        parent_map = _ParentMap()
        parent_map.visit(tree)
        parents = parent_map.parents

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            spawn = _resolve_spawn_call(node.func, module_aliases, func_aliases)
            if spawn is None:
                continue

            module, attr = spawn
            line_no = node.lineno
            if line_no and line_no <= len(lines):
                line_text = lines[line_no - 1]
                if ALLOWLIST_RE.search(line_text):
                    continue

            scope = _enclosing_scope(node, parents)
            if _call_has_spawn_kwargs(
                node, scope, parents, helper_names, helper_modules, module_aliases
            ):
                continue

            violations.append(
                f"{path}:{line_no}: {module}.{attr}() call must route "
                f"creationflags/startupinfo through one of {sorted(HELPER_NAMES)}, or add "
                f"'# spawn-guard: allow' to the call line"
            )

    return violations


def test_all_source_spawn_sites_rout_through_helper() -> None:
    """Every existing subprocess/os spawn site in src/charlie_work routes through the helper."""
    violations = find_spawn_guard_violations(_source_package_dir())
    assert not violations, "\n".join(violations)


def test_guard_flags_bare_subprocess_run(tmp_path: Path) -> None:
    """A bare ``subprocess.run(...)`` without the helper is flagged."""
    source_dir = tmp_path / "pkg"
    source_dir.mkdir()
    (source_dir / "bad.py").write_text(
        "import subprocess\nsubprocess.run(['echo'])\n", encoding="utf-8"
    )
    (source_dir / "good.py").write_text(
        "import subprocess\n"
        "from charlie_work.subprocess_runner import no_console_window_kwargs\n"
        "subprocess.run(['echo'], **no_console_window_kwargs())\n",
        encoding="utf-8",
    )
    violations = find_spawn_guard_violations(source_dir)
    assert len(violations) == 1
    assert "bad.py" in violations[0]
    assert "subprocess.run" in violations[0]


def test_guard_flags_bare_os_spawn(tmp_path: Path) -> None:
    """A bare ``os.spawn*`` call without the helper is flagged."""
    source_dir = tmp_path / "pkg"
    source_dir.mkdir()
    (source_dir / "bad.py").write_text(
        "import os\nos.spawnl(os.P_NOWAIT, 'foo')\n", encoding="utf-8"
    )
    violations = find_spawn_guard_violations(source_dir)
    assert len(violations) == 1
    assert "bad.py" in violations[0]
    assert "os.spawnl" in violations[0]


def test_guard_allowlist_comment(tmp_path: Path) -> None:
    """A ``# spawn-guard: allow`` comment on the call line suppresses the violation."""
    source_dir = tmp_path / "pkg"
    source_dir.mkdir()
    (source_dir / "allowed.py").write_text(
        "import subprocess\nsubprocess.run(['echo'])  # spawn-guard: allow\n",
        encoding="utf-8",
    )
    violations = find_spawn_guard_violations(source_dir)
    assert violations == []
