"""The fleet pause flag gates every one-shot pass entrypoint (issue #2660).

#1776 taught only the supervisor and ``fleet-pass.ps1`` to honour
``fleet-pause.json``; ``fleet bash-rats`` (and per-repo ``bash-rats``/``work``)
went straight to ``self_deploy()`` and a full pass. The fix is one gate in
``cli.main`` keyed on a parser-declared ``runs_pass`` marker
(``fleet_pause.mark_pass_running``).

Two kinds of test live here:

* Behavioural: for every command carrying the marker, a pending flag means exit
  ``EXIT_FLEET_PAUSED`` with no ``self_deploy``/``fleet_loop``/``build_app``/
  ``run_supervised`` reached; ``--ignore-pause`` lets it through, logged.
* Structural: the set of pass-running commands is *derived from the code* (an
  AST walk of ``cli.main``/``cli.run_command`` to the pass primitives), then
  compared with the parser markers -- so a new entrypoint that calls a pass
  primitive without ``mark_pass_running`` fails here instead of silently
  skipping the gate.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from charlie_work import cli, layout
from charlie_work import supervise as supervise_module
from charlie_work.command_result import CommandResult
from charlie_work.fleet_pause import (
    PASS_RUNNING_DEST,
    fleet_pause_pending,
    refuse_pass_while_paused,
    write_fleet_pause,
)
from charlie_work.supervise import SelfDeployResult
from charlie_work.supervise_loop import EXIT_FLEET_PAUSED

PACKAGE_DIR = Path(cli.__file__).resolve().parent

# ---------------------------------------------------------------------------
# Derivation: which CLI commands run a pass?
# ---------------------------------------------------------------------------

#: Calls that start a pass (or the self-deploy that precedes one). The derived
#: set is "commands whose handler reaches one of these".
PASS_PRIMITIVE_NAMES = frozenset(
    {"fleet_loop", "self_deploy", "run_supervised", "run_fleet_supervise"}
)
#: ``app.<method>()`` calls that run a per-repo pass / dispatch.
PASS_PRIMITIVE_APP_METHODS = frozenset({"loop", "dispatch"})
#: Dispatcher functions, analysed per-branch rather than expanded as callees.
DISPATCHERS = frozenset({"main", "run_command"})
#: Commands that enforce the flag themselves at their own boundary instead of
#: through the shared gate (so they can stamp the heartbeat on the way out).
#: Verified below, not trusted: the named function must itself consult the flag.
SELF_GUARDED = {("fleet", "supervise"): ("fleet_dispatch", "run_fleet_supervise")}


def _parse(module_stem: str) -> ast.Module:
    return ast.parse((PACKAGE_DIR / f"{module_stem}.py").read_text(encoding="utf-8"))


def _functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _relative_imports(tree: ast.Module) -> dict[str, tuple[str, str]]:
    """local name -> (module stem, original name) for ``from .x import y as z``."""
    out: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            for alias in node.names:
                out[alias.asname or alias.name] = (node.module.split(".")[0], alias.name)
    return out


def _calls(node: ast.AST) -> Iterator[ast.Call]:
    return (n for n in ast.walk(node) if isinstance(n, ast.Call))


def _is_primitive(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id in PASS_PRIMITIVE_NAMES
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "app"
        and func.attr in PASS_PRIMITIVE_APP_METHODS
    )


def _reaches_primitive(
    module_stem: str, fn_name: str, seen: set[tuple[str, str]] | None = None
) -> bool:
    """Whether ``module_stem.fn_name`` calls a pass primitive, following
    same-module calls and relative imports (names resolved, not guessed)."""
    seen = set() if seen is None else seen
    if (module_stem, fn_name) in seen:
        return False
    seen.add((module_stem, fn_name))
    tree = _parse(module_stem)
    fn = _functions(tree).get(fn_name)
    if fn is None:
        return False
    imports = _relative_imports(tree)
    local = _functions(tree)
    for call in _calls(fn):
        if _is_primitive(call):
            return True
        if isinstance(call.func, ast.Name):
            name = call.func.id
            if name in DISPATCHERS and module_stem == "cli":
                continue
            if name in local and _reaches_primitive(module_stem, name, seen):
                return True
            if name in imports:
                target_module, original = imports[name]
                if (PACKAGE_DIR / f"{target_module}.py").exists() and _reaches_primitive(
                    target_module, original, seen
                ):
                    return True
    return False


def _command_key(test: ast.expr) -> tuple[str, str] | None:
    """``args.command == "x"`` / ``args.fleet_command == "x"`` -> (attr, "x")."""
    if (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and isinstance(test.left, ast.Attribute)
        and isinstance(test.left.value, ast.Name)
        and test.left.value.id == "args"
        and test.left.attr in ("command", "fleet_command")
        and isinstance(test.comparators[0], ast.Constant)
        and isinstance(test.comparators[0].value, str)
    ):
        return test.left.attr, test.comparators[0].value
    return None


def _walk_branches(
    stmts: list[ast.stmt], path: tuple[str, ...], out: dict[tuple[str, ...], list[ast.AST]]
) -> None:
    """Attribute each statement under a command-test ``if`` to its command path."""
    for stmt in stmts:
        if isinstance(stmt, ast.If):
            key = _command_key(stmt.test)
            if key is not None:
                # `args.fleet_command` branches sit under `args.command == "fleet"`.
                sub = path + (key[1],)
                _walk_branches(stmt.body, sub, out)
                _walk_branches(stmt.orelse, path, out)
                continue
        if isinstance(stmt, ast.Try):
            _walk_branches(stmt.body, path, out)
            continue
        out.setdefault(path, []).append(stmt)


def _derive_pass_running_commands() -> set[tuple[str, ...]]:
    """Command paths (``("fleet", "bash-rats")``, ``("work",)``) that reach a pass."""
    cli_tree = _parse("cli")
    imports = _relative_imports(cli_tree)
    cli_functions = _functions(cli_tree)
    found: set[tuple[str, ...]] = set()
    for dispatcher in sorted(DISPATCHERS):
        branches: dict[tuple[str, ...], list[ast.AST]] = {}
        _walk_branches(cli_functions[dispatcher].body, (), branches)
        for path, nodes in branches.items():
            if not path:
                continue
            for node in nodes:
                for call in _calls(node):
                    if _is_primitive(call):
                        found.add(path)
                    elif isinstance(call.func, ast.Name):
                        name = call.func.id
                        if name in DISPATCHERS:
                            continue
                        if name in cli_functions and _reaches_primitive("cli", name):
                            found.add(path)
                        elif name in imports:
                            module, original = imports[name]
                            if (PACKAGE_DIR / f"{module}.py").exists() and _reaches_primitive(
                                module, original
                            ):
                                found.add(path)
    return found


def _leaf_commands(
    parser: argparse.ArgumentParser, prefix: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], argparse.ArgumentParser]]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                yield from _leaf_commands(sub, (*prefix, name))
            return
    yield prefix, parser


def _marked_commands() -> set[tuple[str, ...]]:
    return {
        path
        for path, sub in _leaf_commands(cli.build_parser())
        if sub.get_default(PASS_RUNNING_DEST)
    }


def _consults_pause_flag(module_stem: str, fn_name: str) -> bool:
    fn = _functions(_parse(module_stem))[fn_name]
    return any(
        isinstance(c.func, ast.Name) and c.func.id == "fleet_pause_pending" for c in _calls(fn)
    )


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


def test_derivation_finds_the_entrypoints_named_in_the_issue() -> None:
    """Positive control: an empty/short derived set would make the guard below vacuous."""
    derived = _derive_pass_running_commands()
    assert {("fleet", "bash-rats"), ("bash-rats",), ("fleet", "work"), ("work",)} <= derived
    assert ("fleet", "supervise") in derived  # the walk sees the already-guarded one too


def test_derivation_does_not_flag_commands_that_run_no_pass() -> None:
    """Negative control: the walk discriminates; it is not "everything"."""
    derived = _derive_pass_running_commands()
    for path in (("fleet", "status"), ("fleet", "pause"), ("fleet", "resume"), ("roll-call",)):
        assert path not in derived


def test_every_pass_running_command_is_gated() -> None:
    """A command reaching a pass primitive needs the shared gate or its own flag check."""
    derived = _derive_pass_running_commands()
    marked = _marked_commands()
    self_guarded = {
        path for path, (module, fn) in SELF_GUARDED.items() if _consults_pause_flag(module, fn)
    }
    unguarded = derived - marked - self_guarded
    assert not unguarded, (
        f"pass-running command(s) {sorted(unguarded)} skip the fleet pause gate: call "
        "fleet_pause.mark_pass_running(<parser>) on the command's parser (#2660)"
    )


def test_marker_is_only_on_commands_that_run_a_pass() -> None:
    """A stale marker would add `--ignore-pause` to a command the gate has no business on."""
    assert _marked_commands() <= _derive_pass_running_commands()


def test_self_guarded_commands_do_not_also_carry_the_shared_gate() -> None:
    # `fleet supervise` must still start while paused so it can stamp the heartbeat
    # exit (test_fleet_pause.py::test_supervisor_started_while_paused_runs_no_pass).
    assert not set(SELF_GUARDED) & _marked_commands()


# ---------------------------------------------------------------------------
# Behavioural tests
# ---------------------------------------------------------------------------


def _pass_running_argvs() -> list[list[str]]:
    argvs = [list(path) for path in sorted(_marked_commands())]
    argvs.append(["bash-rats", "--once"])  # the other per-repo mode of the same command
    return argvs


def _argv_id(argv: list[str]) -> str:
    return "-".join(argv)


@pytest.fixture
def paused_fleet(tmp_path: Path) -> Path:
    fleet = tmp_path / "paused-fleet"
    fleet.mkdir()
    write_fleet_pause(str(fleet), reason="maintenance window")
    assert fleet_pause_pending(str(fleet))
    return fleet


@pytest.fixture
def pass_sentinels(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace every way a pass can start with a recorder that fails loudly.

    Real handlers stay in place on purpose: a gate placed *after* a handler had
    already begun (the original bug -- ``self_deploy`` ran first) must trip.
    """
    reached: list[str] = []

    def _sentinel(name: str) -> Any:
        def _called(*_args: Any, **_kwargs: Any) -> Any:
            reached.append(name)
            raise AssertionError(f"{name} reached while the fleet is paused")

        return _called

    for name in ("self_deploy", "fleet_loop", "build_app", "run_fleet_supervise"):
        monkeypatch.setattr(cli, name, _sentinel(name))
    monkeypatch.setattr(supervise_module, "run_supervised", _sentinel("run_supervised"))
    # Keep a (regressed) real handler from reading the operator's real config.
    monkeypatch.setattr(cli, "load_layered_config", lambda *a, **k: None)
    return reached


@pytest.mark.parametrize("argv", _pass_running_argvs(), ids=_argv_id)
def test_paused_fleet_refuses_pass_running_command(
    argv: list[str],
    paused_fleet: Path,
    pass_sentinels: list[str],
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with caplog.at_level(logging.WARNING, logger="charlie_work.fleet_pause"):
        rc = cli.main(["--fleet-dir", str(paused_fleet), *argv])

    assert rc == EXIT_FLEET_PAUSED == 5
    assert pass_sentinels == []  # no self_deploy, no pass, no app built, no supervisor loop
    # The log line names the flag (and the command that was turned away).
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert layout.FLEET_PAUSE_FILENAME in logged
    assert f"charlie {' '.join(argv[:-1] if argv[-1] == '--once' else argv)}" in logged
    assert layout.FLEET_PAUSE_FILENAME in capsys.readouterr().out
    # The gate reads the flag; it never consumes it.
    assert fleet_pause_pending(str(paused_fleet))


@pytest.fixture
def handlers(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub every handler/builder; record which one a command actually reached."""
    reached: list[str] = []
    ok = CommandResult(True, "stub", {"repos": {}, "digest": {}})

    def _stub(name: str) -> Any:
        def _called(*_args: Any, **_kwargs: Any) -> CommandResult:
            reached.append(name)
            return ok

        return _called

    for name in ("run_fleet_work", "run_fleet_bash_rats", "run_command"):
        monkeypatch.setattr(cli, name, _stub(name))
    monkeypatch.setattr(cli, "build_app", lambda _args: MagicMock(name="app"))
    return reached


@pytest.mark.parametrize("argv", _pass_running_argvs(), ids=_argv_id)
def test_ignore_pause_lets_the_command_through_and_logs_the_override(
    argv: list[str],
    paused_fleet: Path,
    handlers: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="charlie_work.fleet_pause"):
        rc = cli.main(["--fleet-dir", str(paused_fleet), *argv, "--ignore-pause"])

    assert rc == 0
    assert handlers, "--ignore-pause must reach the command's handler"
    overrides = [r.getMessage() for r in caplog.records if "--ignore-pause" in r.getMessage()]
    assert len(overrides) == 1
    assert layout.FLEET_PAUSE_FILENAME in overrides[0]
    assert "maintenance window" in overrides[0]  # who/why the fleet was paused
    assert fleet_pause_pending(str(paused_fleet))  # an override does not resume the fleet


@pytest.mark.parametrize("argv", _pass_running_argvs(), ids=_argv_id)
def test_unpaused_fleet_runs_pass_running_command_unlogged(
    argv: list[str], tmp_path: Path, handlers: list[str], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="charlie_work.fleet_pause"):
        rc = cli.main(["--fleet-dir", str(tmp_path / "running-fleet"), *argv])

    assert rc == 0
    assert handlers
    assert not [r for r in caplog.records if r.name == "charlie_work.fleet_pause"]


def test_dry_run_is_still_refused_while_paused(
    paused_fleet: Path, pass_sentinels: list[str]
) -> None:
    # Uniform on purpose: the pause is a safety control, and `--ignore-pause`
    # is the one explicit way past it.
    rc = cli.main(["--dry-run", "--fleet-dir", str(paused_fleet), "fleet", "bash-rats"])
    assert rc == EXIT_FLEET_PAUSED
    assert pass_sentinels == []


def test_json_output_reports_the_refusal(
    paused_fleet: Path, pass_sentinels: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(["--json", "--fleet-dir", str(paused_fleet), "fleet", "bash-rats"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == EXIT_FLEET_PAUSED
    assert payload["data"]["fleet_paused"] is True
    assert payload["data"]["reason"] == "maintenance window"
    assert payload["data"]["command"] == "fleet bash-rats"
    assert payload["data"]["flag"].endswith(layout.FLEET_PAUSE_FILENAME)


def test_torn_pause_flag_still_refuses(tmp_path: Path, pass_sentinels: list[str]) -> None:
    """Fail closed, like every other reader of the flag (read_fleet_pause)."""
    fleet = tmp_path / "torn"
    fleet.mkdir()
    layout.fleet_pause_path(str(fleet)).write_text("{torn", encoding="utf-8")
    assert cli.main(["--fleet-dir", str(fleet), "fleet", "bash-rats"]) == EXIT_FLEET_PAUSED
    assert pass_sentinels == []


def test_commands_that_run_no_pass_are_not_gated_while_paused(
    paused_fleet: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`fleet resume` in particular must keep working: it is the way out of the pause."""
    status = MagicMock(return_value=CommandResult(True, "status", {"repos": {}}))
    monkeypatch.setattr(cli, "run_fleet_status", status)
    assert cli.main(["--fleet-dir", str(paused_fleet), "fleet", "status"]) == 0
    status.assert_called_once()

    assert cli.main(["--fleet-dir", str(paused_fleet), "fleet", "resume"]) == 0
    assert not fleet_pause_pending(str(paused_fleet))


def test_gate_ignores_namespaces_without_the_marker(paused_fleet: Path) -> None:
    """Hand-built Namespaces (some tests, run_command callers) are never gated."""
    args = argparse.Namespace(command="work", fleet_dir=str(paused_fleet))
    assert refuse_pass_while_paused(args) is None


def test_real_fleet_bash_rats_handler_never_self_deploys_when_paused(
    paused_fleet: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reported failure end to end: real handler, real gate, self_deploy watched."""
    deploy = MagicMock(
        return_value=SelfDeployResult(
            ok=True, pulled=True, changed=False, synced=False, message="up to date"
        )
    )
    loop = MagicMock(return_value=CommandResult(True, "ok", {"repos": {}}))
    monkeypatch.setattr(cli, "self_deploy", deploy)
    monkeypatch.setattr(cli, "fleet_loop", loop)
    monkeypatch.setattr(cli, "load_layered_config", lambda *a, **k: None)

    assert cli.main(["--fleet-dir", str(paused_fleet), "fleet", "bash-rats"]) == 5
    deploy.assert_not_called()
    loop.assert_not_called()

    # Same call with the override reaches both, in order (self-deploy first).
    assert (
        cli.main(["--fleet-dir", str(paused_fleet), "fleet", "bash-rats", "--ignore-pause"]) == 0
    )
    deploy.assert_called_once()
    loop.assert_called_once()
