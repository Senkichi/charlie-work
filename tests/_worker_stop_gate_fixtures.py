"""Shared helpers and fixtures for the worker-stop-gate test modules.

Hoisted verbatim out of ``tests/test_worker_stop_gate.py`` (issue #1573,
Track 1) when that module was split into seam-named siblings -- the
``tests/_*.py`` hoisted-fixture convention is the sanctioned import target
for shared test helpers (see ``tests/test_zero_cross_test_import_guard.py``).

The gate module under test is loaded via ``_script_loader.load_script_module``
(scripts/ is not on sys.path) -- the same recipe as
tests/test_merge_autonomy_ratio.py, and required by tests/test_script_loader.py's
``test_no_hand_rolled_spec_from_file_location_in_tests`` guard.
"""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from types import ModuleType

import pytest
from _script_loader import load_script_module


_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "worker_stop_gate.py"


def _load_module() -> ModuleType:
    return load_script_module(_SCRIPT_PATH, "worker_stop_gate_under_test")


def _run_git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _git_output(args: list[str], cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _set_origin_main(root: Path, sha: str) -> None:
    """Point a bare local ``refs/remotes/origin/main`` at ``sha``, decoupled
    from whatever the local ``init.defaultBranch`` happens to be -- the
    ``repo`` fixture has no real remote configured, so this is the only way
    to give ``_committed_diff_files`` something to diverge from."""
    _run_git(["update-ref", "refs/remotes/origin/main", sha], cwd=root)


@pytest.fixture()
def gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    """A fresh module instance per test, with its state dir pinned under
    ``tmp_path`` so no test touches the real system temp dir (which would
    both pollute a real worker session's counters and leak across test
    runs). Function-scoped rather than module-scoped like
    test_merge_autonomy_ratio.py's ``mar`` fixture: several tests here
    monkeypatch module-level functions (``_run``, ``_repo_root``), and a
    module shared across tests would make patch ordering/teardown surprises
    possible.
    """
    module = _load_module()
    state_dir = tmp_path / "gate-state"
    monkeypatch.setattr(module, "_state_dir", lambda: _ensure_dir(state_dir))
    return module


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A real, throwaway git repo with one committed file."""
    root = tmp_path / "repo"
    root.mkdir()
    _run_git(["init"], cwd=root)
    _run_git(["config", "user.email", "test@example.com"], cwd=root)
    _run_git(["config", "user.name", "Test"], cwd=root)
    (root / "README.md").write_text("placeholder\n", encoding="utf-8")
    _run_git(["add", "README.md"], cwd=root)
    _run_git(["commit", "-m", "init"], cwd=root)
    return root


def _stdin(payload: dict) -> io.StringIO:
    return io.StringIO(json.dumps(payload))
