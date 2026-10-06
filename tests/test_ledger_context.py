"""Worker and gate launches tag their test runs for the ci-fleet test ledger."""

from __future__ import annotations

import ast
from pathlib import Path

from charlie_work.ledger_context import CONTEXT_VAR, TICKET_VAR, ledger_env

SRC = Path(__file__).resolve().parent.parent / "src" / "charlie_work"


def test_ledger_env_names_the_context_and_the_bare_ticket() -> None:
    assert ledger_env("worker", 123) == {CONTEXT_VAR: "worker", TICKET_VAR: "123"}
    assert ledger_env("gate", 7) == {CONTEXT_VAR: "gate", TICKET_VAR: "7"}


def _calls_ledger_env(path: Path, context: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ledger_env"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == context
        ):
            return True
    return False


def test_every_launcher_exports_the_ledger_context() -> None:
    # One assertion per launch path: claude-code and api share launch_claude_worker.
    assert _calls_ledger_env(SRC / "claude_code.py", "worker")
    assert _calls_ledger_env(SRC / "devin_shell.py", "worker")
    assert _calls_ledger_env(SRC / "orchestration" / "local_merge_gate.py", "gate")
