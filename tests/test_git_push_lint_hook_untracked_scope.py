"""#1704 regression tests: ``git_push_lint_hook``'s ruff scope must exclude
untracked (``??``) ``.py`` debris -- the same narrowing
``worker_stop_gate._evaluate`` applies for #1306.

Split out of ``tests/test_git_push_lint_hook.py`` because that module is a
saturated ``test_module`` attachment point (member ceiling 42) and also
carries a file-size-ratchet mark -- new members bind here instead.

Every test drives ``main()`` against a REAL ``git init``-ed ``tmp_path``
repo and the REAL ``worker_stop_gate`` module (loaded through
``hook._load_stop_gate()``); only ``_run_ruff`` is stubbed, and it answers
the way real ruff would on the violating file, so a deny can only come
from debris leaking into the lint scope.
"""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from _script_loader import load_script_module

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = REPO_ROOT / "scripts" / "git_push_lint_hook.py"


@pytest.fixture()
def hook() -> ModuleType:
    """A fresh module instance per test."""
    return load_script_module(_SCRIPT_PATH, "git_push_lint_hook_under_test")


def _stdin(payload: dict[str, Any]) -> io.StringIO:
    return io.StringIO(json.dumps(payload))


def _bash_payload(command: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"tool_name": "Bash", "tool_input": {"command": command}}
    payload.update(extra)
    return payload


def _run_git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_repo(root: Path) -> None:
    """A minimal git repo with one commit and no ``origin`` remote.

    No ``origin/main`` ref means ``worker_stop_gate._committed_diff_files``
    returns ``()`` (its documented fail-open narrowing), so the changed set
    is exactly the working-tree state this test plants.
    """
    root.mkdir()
    _run_git(["init"], cwd=root)
    _run_git(["config", "user.email", "test@example.com"], cwd=root)
    _run_git(["config", "user.name", "Test"], cwd=root)
    (root / "README.md").write_text("placeholder\n", encoding="utf-8")
    _run_git(["add", "README.md"], cwd=root)
    _run_git(["commit", "-m", "init"], cwd=root)


def test_main_push_untracked_py_debris_does_not_deny(
    hook: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """#1704: a pre-existing untracked (``??``) ``.py`` file with a real
    ruff violation must not deny the push. ``git status`` cannot tell it
    apart from debris that predates the session, so the push hook's ruff
    scope excludes untracked files exactly like the Stop gate (#1306) --
    before the fix, the push hook's inline predicate lacked the
    ``not cf.untracked`` filter and denied on exactly this input."""
    root = tmp_path / "repo"
    _init_repo(root)
    # Pre-existing untracked debris with a real format violation -- never
    # staged, never committed, not a session-modified tracked file, and
    # not part of any committed-since-base diff.
    (root / "debris.py").write_text("x  =  1\n", encoding="utf-8")

    real_gate = hook._load_stop_gate()

    def _ruff_as_run(repo_root: Path, py_files: tuple[str, ...]) -> Any:
        # Real ruff would fail on debris.py's format violation, so being
        # asked to lint it produces a deny -- only reachable if the scope
        # predicate wrongly includes untracked files.
        if "debris.py" in py_files:
            return real_gate.GateResult(
                block=True,
                reason="ruff format --check failed:\nwould reformat debris.py",
            )
        return real_gate.GateResult(block=False)

    monkeypatch.setattr(real_gate, "_run_ruff", _ruff_as_run)
    monkeypatch.setattr(hook, "_load_stop_gate", lambda: real_gate)
    monkeypatch.setattr(hook.sys, "stdin", _stdin(_bash_payload("git push", cwd=str(root))))

    rc = hook.main()

    assert rc == 0
    assert capsys.readouterr().out.strip() == "", (
        "untracked debris.py must be excluded from the push hook's ruff "
        "scope (#1704) -- a deny here is the #1306 false-positive class "
        "reintroduced at push time"
    )


def test_main_push_tracked_py_violation_still_denies(
    hook: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """Companion to the untracked-debris test: a session-authored ``.py``
    file that is NOT untracked (staged -- ``A `` in ``git status``, not
    ``??``) with a real ruff violation must still deny the push. The
    #1704 fix narrows the ruff scope; it does not disarm it."""
    root = tmp_path / "repo"
    _init_repo(root)
    # Session-authored file, already staged -> tracked (``A ``), so it IS
    # in ruff scope even though it is new this session.
    (root / "dirty.py").write_text("x  =  1\n", encoding="utf-8")
    _run_git(["add", "dirty.py"], cwd=root)

    real_gate = hook._load_stop_gate()
    captured: list[tuple[str, ...]] = []

    def _ruff_as_run(repo_root: Path, py_files: tuple[str, ...]) -> Any:
        captured.append(py_files)
        if "dirty.py" in py_files:
            return real_gate.GateResult(
                block=True,
                reason="ruff format --check failed:\nwould reformat dirty.py",
            )
        return real_gate.GateResult(block=False)

    monkeypatch.setattr(real_gate, "_run_ruff", _ruff_as_run)
    monkeypatch.setattr(hook, "_load_stop_gate", lambda: real_gate)
    monkeypatch.setattr(hook.sys, "stdin", _stdin(_bash_payload("git push", cwd=str(root))))

    rc = hook.main()

    assert rc == 0
    assert captured == [("dirty.py",)], (
        f"staged session file must be the ruff scope, got {captured}"
    )
    out = json.loads(capsys.readouterr().out)
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "ruff format" in out["hookSpecificOutput"]["permissionDecisionReason"]


def test_check_push_lint_delegates_scope_to_stop_gate_helper(
    hook: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """#1704: the push hook must not reimplement the ruff-scope predicate
    inline -- it delegates to ``worker_stop_gate._ruff_lint_paths`` so the
    two gates share one predicate and cannot silently diverge again (the
    exact gap that let this bug in). Spies on the REAL helper."""
    root = tmp_path / "repo"
    _init_repo(root)
    # One untracked file so the changed set is non-empty and the scope
    # helper is actually consulted (an empty set returns early).
    (root / "debris.py").write_text("x = 1\n", encoding="utf-8")

    real_gate = hook._load_stop_gate()
    consulted: list[tuple[Any, ...]] = []
    original = real_gate._ruff_lint_paths

    def _spy(changed: tuple[Any, ...]) -> tuple[str, ...]:
        consulted.append(changed)
        return original(changed)

    monkeypatch.setattr(real_gate, "_ruff_lint_paths", _spy)
    monkeypatch.setattr(hook, "_load_stop_gate", lambda: real_gate)
    monkeypatch.setattr(hook.sys, "stdin", _stdin(_bash_payload("git push", cwd=str(root))))

    rc = hook.main()

    assert rc == 0
    assert capsys.readouterr().out.strip() == ""
    assert len(consulted) == 1, (
        "_check_push_lint must scope ruff via stop_gate._ruff_lint_paths, not an inline predicate"
    )
