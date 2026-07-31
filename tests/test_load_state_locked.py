from __future__ import annotations

import ast
import threading
from pathlib import Path

from _stubs import StubGitHubLike
from charlie_work import state as state_module
from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    append_event,
    load_state,
    load_state_locked,
    save_state,
    state_lock,
)
from charlie_work.workflow import OrchestratorApp


class FakeGitHub(StubGitHubLike):
    """Minimal stub sufficient for OrchestratorApp.status()."""

    def __init__(self) -> None:
        self.issues = [
            {
                "number": 123,
                "title": "Fix search",
                "url": "https://example.test/issues/123",
                "body": "Search is broken",
                "labels": [{"name": "automated-ready"}],
                "state": "OPEN",
            }
        ]
        self.prs = [
            {
                "number": 456,
                "title": "Fix #123: search",
                "url": "https://example.test/pull/456",
                "headRefName": "agent/issue-123-fix-search",
                "headRefOid": "sha-abc123",
                "mergeStateStatus": "CLEAN",
                "body": "Closes #123\n\nTests: regression coverage added.",
                "labels": [],
                "isCrossRepository": False,
                "state": "OPEN",
            }
        ]

    def issue_list(self, labels=None, state=None):
        if isinstance(labels, str):
            return [
                issue
                for issue in self.issues
                if labels in {label["name"] for label in issue.get("labels", [])}
            ]
        elif labels:
            return [
                issue
                for issue in self.issues
                if any(
                    label in {label_obj["name"] for label_obj in issue.get("labels", [])}
                    for label in labels
                )
            ]
        return self.issues

    def pr_list(self):
        return [pr for pr in self.prs if pr.get("state", "OPEN").upper() == "OPEN"]

    def check_graphql_rate_limit(self, threshold: int) -> tuple[bool, int, int | None]:
        return (True, 10000, 0)

    def run(self, args, *, json_output=False, allow_failure=False):
        return [] if json_output else ""


def test_load_state_locked_uses_state_lock(tmp_path: Path, monkeypatch) -> None:
    """load_state_locked must acquire the advisory lock before reading."""
    state_path = tmp_path / "state.json"
    lock_calls: list[Path] = []
    orig_state_lock = state_module.state_lock

    def tracking_state_lock(path: Path):
        lock_calls.append(path)
        return orig_state_lock(path)

    monkeypatch.setattr(state_module, "state_lock", tracking_state_lock)

    result = load_state_locked(state_path)

    assert result["version"] == 1
    assert lock_calls == [state_path]


def test_load_state_locked_matches_load_state(tmp_path: Path) -> None:
    """A locked read returns the same state an explicit lock+load would."""
    state_path = tmp_path / "state.json"
    save_state(state_path, {"issues": {"1": {"number": 1}}})

    with state_lock(state_path):
        expected = load_state(state_path)

    actual = load_state_locked(state_path)

    assert actual == expected


def test_status_concurrent_writer_no_quarantine(tmp_path: Path) -> None:
    """Concurrent status() calls and a state writer must never quarantine a healthy state file.

    Regression for issue #310: status() read state.json without state_lock, so a
    writer's tmp+replace could race a read and (after retry exhaustion) cause the
    reader to quarantine a perfectly healthy state.json.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())
    state_file = paths.state_file

    iterations = 200
    errors: list[Exception] = []

    def writer() -> None:
        for i in range(iterations):
            try:
                with state_lock(state_file):
                    state = load_state(state_file)
                    state = append_event(state, "stress", {"i": i})
                    save_state(state_file, state)
            except Exception as exc:  # pragma: no cover - failures fail the test
                errors.append(exc)
                break

    def reader() -> None:
        for _ in range(iterations):
            try:
                app.status()
            except Exception as exc:  # pragma: no cover - failures fail the test
                errors.append(exc)
                break

    writer_thread = threading.Thread(target=writer)
    reader_thread = threading.Thread(target=reader)
    writer_thread.start()
    reader_thread.start()
    writer_thread.join(timeout=30)
    reader_thread.join(timeout=30)

    assert not errors, f"exceptions during concurrent stress: {errors}"

    quarantine_files = list(state_file.parent.glob(f"{state_file.name}.corrupt-*"))
    assert not quarantine_files, f"healthy state file was quarantined: {quarantine_files}"

    final_state = load_state(state_file)
    assert final_state["version"] == 1
    assert len(final_state.get("events", [])) == iterations


def test_no_unlocked_load_state_in_production_code() -> None:
    """Every load_state call in src/charlie_work must be inside a state_lock context.

    This is the lint-style enforcement from issue #310: the locked read helper
    (load_state_locked) should be used for read-only callers, and all raw
    load_state calls must occur inside a state_lock block.
    """
    src_root = Path(__file__).resolve().parents[1] / "src" / "charlie_work"

    class LockVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self._lock_depth = 0
            self.errors: list[str] = []

        def visit_With(self, node: ast.With) -> None:
            is_state_lock = any(
                isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Name)
                and item.context_expr.func.id == "state_lock"
                for item in node.items
            )
            self._lock_depth += is_state_lock
            self.generic_visit(node)
            self._lock_depth -= is_state_lock

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name) and node.func.id == "load_state":
                if self._lock_depth <= 0:
                    self.errors.append(f"line {node.lineno}: {ast.unparse(node)}")
            self.generic_visit(node)

    errors: list[str] = []
    for source_file in src_root.rglob("*.py"):
        if source_file.name == "test_load_state_locked.py":
            continue
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        visitor = LockVisitor()
        visitor.visit(tree)
        errors.extend(visitor.errors)

    assert not errors, f"load_state calls outside state_lock: {errors}"
